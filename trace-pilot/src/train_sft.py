"""LoRA SFT for Qwen3-VL on STEMO-Ambig SFT data.

Launch via:
  bash trace-pilot/scripts/launch_sft.sh

Or directly:
  accelerate launch --config_file trace-pilot/configs/accelerate_ds.yaml \\
    trace-pilot/src/train_sft.py --config trace-pilot/configs/sft_lora.yaml

Data shape (each JSONL line):
  {
    "video_path": "/abs/path/x.mp4",
    "messages": [
      {"role": "system", "content": "..."},
      {"role": "user", "content": [{"type":"video","video":"..."},{"type":"text","text":"..."}]},
      {"role": "assistant", "content": "<think>...</think>\\n\\nFinal..."}
    ],
    "meta": {...}
  }

Training masks loss to assistant tokens only (system+user are context, not learning targets).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import torch
import yaml
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)


# ---------- data ----------

def load_jsonl(p):
    return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]


def _decode_video(path, num_frames, target_size=336):
    """Pre-decode video to (T, target_size, target_size, 3) uint8 array using decord.

    Resizing to a fixed (target_size, target_size) ensures the processor produces
    a consistent patch count regardless of source video resolution. Without this,
    variable-resolution rehearsal videos cause a video-feature/token mismatch at
    model forward.

    Bypasses transformers' torchcodec-based backend (broken in this env).
    """
    import decord  # noqa: WPS433
    import numpy as np
    from PIL import Image
    vr = decord.VideoReader(str(path))
    total = len(vr)
    n = min(num_frames, total)
    if n == 0:
        raise RuntimeError(f"empty video: {path}")
    idxs = [int(i * (total - 1) / max(n - 1, 1)) for i in range(n)]
    frames = vr.get_batch(idxs).asnumpy()
    # Resize each frame to target_size × target_size for consistent patching.
    resized = np.empty((frames.shape[0], target_size, target_size, 3), dtype=np.uint8)
    for i in range(frames.shape[0]):
        resized[i] = np.array(
            Image.fromarray(frames[i]).resize((target_size, target_size), Image.BILINEAR)
        )
    return resized


class SFTDataset(Dataset):
    def __init__(self, jsonl_path, processor, max_seq_len, video_fps, video_max_frames):
        self.records = load_jsonl(jsonl_path)
        self.processor = processor
        self.max_seq_len = max_seq_len
        self.video_fps = video_fps
        self.video_max_frames = video_max_frames

    def __len__(self):
        return len(self.records)

    def _wrap_messages(self, messages):
        """Wrap plain-string content in [{"type":"text","text":...}] so
        apply_chat_template can iterate uniformly. Keep video parts as-is;
        the actual video pixels are passed via the processor's `videos` arg.
        """
        out = []
        for m in messages:
            content = m["content"]
            if isinstance(content, str):
                new_content = [{"type": "text", "text": content}]
            else:
                new_content = content
            out.append({"role": m["role"], "content": new_content})
        return out

    def __getitem__(self, idx):
        rec = self.records[idx]
        # Robust decode: some videos in the wild are malformed. Skip to next on failure.
        for attempt in range(8):
            try:
                frames = _decode_video(rec["video_path"], self.video_max_frames)
                break
            except Exception as e:  # noqa: BLE001
                print(f"[dataset] decode failed for {rec['video_path']!r}: {e!r}; trying next idx", flush=True)
                idx = (idx + 1) % len(self.records)
                rec = self.records[idx]
        else:
            raise RuntimeError("8 consecutive video decode failures; aborting")
        messages = self._wrap_messages(rec["messages"])

        # Get text prompts (with <|video|> placeholders inserted by chat template).
        prompt_full_text = self.processor.apply_chat_template(
            messages, add_generation_prompt=False, tokenize=False,
        )
        prompt_only_msgs = [m for m in messages if m["role"] != "assistant"]
        prompt_only_text = self.processor.apply_chat_template(
            prompt_only_msgs, add_generation_prompt=True, tokenize=False,
        )

        # Process full prompt with pre-decoded video frames.
        full = self.processor(
            text=[prompt_full_text], videos=[frames], return_tensors="pt",
        )
        # Process prompt-only with the same frames to find the label-mask boundary.
        prompt_only = self.processor(
            text=[prompt_only_text], videos=[frames], return_tensors="pt",
        )
        prompt_len = prompt_only["input_ids"].shape[1]

        input_ids = full["input_ids"][0]
        if input_ids.shape[0] > self.max_seq_len:
            input_ids = input_ids[: self.max_seq_len]
        labels = input_ids.clone()
        labels[: min(prompt_len, labels.shape[0])] = -100

        item = {}
        per_token_keys = {"input_ids", "attention_mask", "mm_token_type_ids"}
        for k, v in full.items():
            if k in per_token_keys:
                t = v[0]
                if t.shape[0] > self.max_seq_len:
                    t = t[: self.max_seq_len]
                item[k] = t
            else:
                # Media tensors (pixel_values_videos, video_grid_thw) — already
                # batch-free in the processor's output (shape e.g. (num_patches, D)
                # or (num_videos, 3)). Keep as-is; collator cats along dim 0.
                item[k] = v
        item["labels"] = labels
        return item


@dataclass
class PadCollator:
    pad_token_id: int

    def __call__(self, batch):
        max_len = max(b["input_ids"].shape[0] for b in batch)
        out = {}
        # Per-token sequence keys (1D, padded to max_len).
        per_token_keys = {"input_ids", "labels", "attention_mask", "mm_token_type_ids"}
        for k in per_token_keys:
            if k not in batch[0]:
                continue
            if k == "input_ids":
                pad_val = self.pad_token_id
            elif k == "labels":
                pad_val = -100
            else:
                pad_val = 0
            stacked = []
            for b in batch:
                t = b[k]
                pad = torch.full((max_len - t.shape[0],), pad_val, dtype=t.dtype)
                stacked.append(torch.cat([t, pad]))
            out[k] = torch.stack(stacked)
        # Concatenate media tensors along dim 0 (Qwen3-VL packs videos this way).
        for k in batch[0]:
            if k in per_token_keys:
                continue
            tensors = [b[k] for b in batch if k in b]
            if not tensors:
                continue
            out[k] = torch.cat(tensors, dim=0)
        return out


# ---------- in-training light validation ----------

import re  # noqa: E402

_ENUM_LINE_RE = re.compile(r"→")
_BARE_YESNO_RE = re.compile(r"^\s*(yes|no)\b", re.IGNORECASE)


def _compute_local_metrics(responses):
    """Compute enumeration_rate, single_commit_rate, truncation_rate from
    list of raw generated strings (model output after <think>...</think>)."""
    n = len(responses)
    if n == 0:
        return {}
    n_enum = 0
    n_commit = 0
    n_trunc = 0
    n_think_only = 0
    for r in responses:
        # Strip <think>...</think> if present
        cleaned = re.sub(r"<think>.*?</think>\s*", "", r, flags=re.DOTALL).strip()
        if not cleaned:
            # nothing past </think> → either truncated mid-think or model emitted only thinking
            if "<think>" in r and "</think>" not in r:
                n_trunc += 1
            else:
                n_think_only += 1
            continue
        # Enumeration heuristic: multiple "→" arrows or "Interpretation N:" markers
        if len(_ENUM_LINE_RE.findall(cleaned)) >= 2 or len(re.findall(r"interpretation\s*\d", cleaned, re.IGNORECASE)) >= 2:
            n_enum += 1
        # Single-commit: bare yes/no as first content word
        if _BARE_YESNO_RE.match(cleaned):
            n_commit += 1
    return {
        "val/enumeration_rate": n_enum / n,
        "val/single_commit_rate": n_commit / n,
        "val/truncation_rate": n_trunc / n,
        "val/think_only_rate": n_think_only / n,
        "val/n_samples": n,
    }


class STEMOLightValCallback(TrainerCallback):
    """Generate on a small STEMO-Ambig subset at each save and log heuristic metrics.

    Uses local pattern-matching only (no Gemini judge) so it doesn't block training.
    Full Gemini-judged validation runs post-training via validate_checkpoint.sh.
    """

    def __init__(self, val_jsonl, processor, max_seq_len, video_max_frames,
                 n_samples=30, max_new_tokens=512):
        self.processor = processor
        self.max_seq_len = max_seq_len
        self.video_max_frames = video_max_frames
        self.max_new_tokens = max_new_tokens
        rows = [json.loads(l) for l in Path(val_jsonl).read_text().splitlines() if l.strip()]
        self.records = rows[:n_samples]
        print(f"[STEMOLightValCallback] loaded {len(self.records)} val examples")

    def _generate_one(self, model, rec):
        try:
            frames = _decode_video(rec["video_path"], self.video_max_frames)
        except Exception as e:  # noqa: BLE001
            return f"[decode_failed: {e!r}]"
        # Build chat messages: system + user (video + question)
        sys_prompt = (
            "You are an expert at answering questions about video content.\n"
            "Watch the video carefully and answer the question.\n"
            "Think step by step before giving your final answer.\n"
            "If the question has multiple valid interpretations because of an ambiguous "
            "referent, enumerate each interpretation explicitly and provide an answer for each."
        )
        messages = [
            {"role": "system", "content": [{"type": "text", "text": sys_prompt}]},
            {"role": "user", "content": [
                {"type": "video"}, {"type": "text", "text": rec["prompt"]}
            ]},
        ]
        prompt_text = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False,
        )
        inputs = self.processor(
            text=[prompt_text], videos=[frames], return_tensors="pt",
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.inference_mode():
            out = model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=0.0,
            )
        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        return self.processor.tokenizer.decode(new_tokens, skip_special_tokens=True)

    def on_save(self, args, state, control, **kwargs):
        # Best-effort. If generation fails for any reason (e.g. Zero-3 gather
        # issues), log and continue — training must not crash because of
        # validation.
        if not state.is_world_process_zero:
            return
        model = kwargs.get("model")
        if model is None:
            return
        try:
            print(f"\n[STEMOLightValCallback @ step {state.global_step}] generating on "
                  f"{len(self.records)} STEMO-Ambig items...")
            was_training = model.training
            model.eval()
            responses = []
            for i, rec in enumerate(self.records):
                try:
                    resp = self._generate_one(model, rec)
                except Exception as e:  # noqa: BLE001
                    print(f"  val gen failed item {i}: {e!r}")
                    resp = ""
                responses.append(resp)
            if was_training:
                model.train()
            metrics = _compute_local_metrics(responses)
            print(f"[STEMOLightValCallback @ step {state.global_step}] {metrics}")
            # Log to Trainer (will route to TB via TensorBoardCallback)
            trainer = kwargs.get("trainer")
            if trainer is not None and hasattr(trainer, "log"):
                trainer.log(metrics)
            else:
                try:
                    state.log_history.append({"step": state.global_step, **metrics})
                except Exception:
                    pass
        except Exception as e:  # noqa: BLE001
            print(f"[STEMOLightValCallback @ step {state.global_step}] FAILED: {e!r}")
            try:
                if model.training is False:
                    model.train()
            except Exception:
                pass


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    lora_cfg = cfg["lora"]
    tr_cfg = cfg["training"]

    processor = AutoProcessor.from_pretrained(
        model_cfg["model_id"],
        trust_remote_code=model_cfg.get("trust_remote_code", False),
    )

    model = AutoModelForImageTextToText.from_pretrained(
        model_cfg["model_id"],
        torch_dtype=getattr(torch, model_cfg.get("torch_dtype", "bfloat16")),
        trust_remote_code=model_cfg.get("trust_remote_code", False),
    )
    model.config.use_cache = False
    # Required for gradient checkpointing + PEFT: input embeddings must produce
    # a tensor that requires grad so the checkpoint can backprop through them.
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    if lora_cfg.get("r", 0) > 0:
        lora_config = LoraConfig(
            r=lora_cfg["r"],
            lora_alpha=lora_cfg["alpha"],
            lora_dropout=lora_cfg.get("dropout", 0.0),
            target_modules=lora_cfg["target_modules"],
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    else:
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"FFT mode (lora.r=0): all {n_trainable/1e6:.1f}M params trainable")

    train_ds = SFTDataset(
        data_cfg["train_file"], processor,
        max_seq_len=data_cfg["max_seq_len"],
        video_fps=data_cfg.get("video_fps", 1.0),
        video_max_frames=data_cfg.get("video_max_frames", 32),
    )
    eval_ds = SFTDataset(
        data_cfg["dev_file"], processor,
        max_seq_len=data_cfg["max_seq_len"],
        video_fps=data_cfg.get("video_fps", 1.0),
        video_max_frames=data_cfg.get("video_max_frames", 32),
    )
    print(f"train n={len(train_ds)}  dev n={len(eval_ds)}")

    pad_id = processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id
    collator = PadCollator(pad_token_id=pad_id)

    training_args = TrainingArguments(
        output_dir=tr_cfg["output_dir"],
        num_train_epochs=tr_cfg["num_train_epochs"],
        per_device_train_batch_size=tr_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=tr_cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=tr_cfg["gradient_accumulation_steps"],
        learning_rate=tr_cfg["learning_rate"],
        lr_scheduler_type=tr_cfg["lr_scheduler_type"],
        warmup_ratio=tr_cfg.get("warmup_ratio", 0.0),
        weight_decay=tr_cfg.get("weight_decay", 0.0),
        logging_steps=tr_cfg.get("logging_steps", 10),
        eval_strategy="steps",
        eval_steps=tr_cfg.get("eval_steps", 100),
        save_strategy="steps",
        save_steps=tr_cfg.get("save_steps", 200),
        save_total_limit=tr_cfg.get("save_total_limit", 3),
        bf16=tr_cfg.get("bf16", True),
        gradient_checkpointing=tr_cfg.get("gradient_checkpointing", True),
        gradient_checkpointing_kwargs={"use_reentrant": True},
        ddp_find_unused_parameters=tr_cfg.get("ddp_find_unused_parameters", False),
        remove_unused_columns=False,
        report_to=tr_cfg.get("report_to", "none"),
        logging_dir=tr_cfg.get("logging_dir", None),
        seed=tr_cfg.get("seed", 0),
        deepspeed=tr_cfg.get("deepspeed_config", None) or "trace-pilot/configs/ds_zero3.json",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        processing_class=processor,
    )

    # In-training light validation callback.
    val_jsonl = data_cfg.get("light_val_jsonl")
    if val_jsonl and Path(val_jsonl).exists():
        n_val = int(data_cfg.get("light_val_n_samples", 30))
        cb = STEMOLightValCallback(
            val_jsonl=val_jsonl, processor=processor,
            max_seq_len=data_cfg["max_seq_len"],
            video_max_frames=data_cfg.get("video_max_frames", 16),
            n_samples=n_val,
            max_new_tokens=int(data_cfg.get("light_val_max_new_tokens", 512)),
        )
        trainer.add_callback(cb)
        print(f"added STEMOLightValCallback (n={n_val}) — fires every save_steps")
    else:
        print("no light_val_jsonl configured; in-training validation disabled")

    # Auto-resume from any existing checkpoint in output_dir.
    out_dir = Path(tr_cfg["output_dir"])
    has_ckpt = out_dir.exists() and any(p.name.startswith("checkpoint-") for p in out_dir.iterdir())
    if has_ckpt:
        print(f"Found existing checkpoint(s) in {out_dir} — resuming")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_model(tr_cfg["output_dir"])
    processor.save_pretrained(tr_cfg["output_dir"])


if __name__ == "__main__":
    main()
