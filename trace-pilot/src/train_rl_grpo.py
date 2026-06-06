"""v5 GRPO training: TRL GRPOTrainer with Gemini-judge reward.

Loads v4 LoRA adapter as init policy, samples rollouts, calls Gemini judge for
reward, applies KL penalty against the v4 reference.
"""
from __future__ import annotations

import argparse
import json
import sys
import yaml
from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from peft import PeftModel, LoraConfig
from trl import GRPOConfig, GRPOTrainer
from datasets import Dataset

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # so rl_reward (same dir) imports

from rl_reward import combined_reward  # noqa: E402  (file in same dir; trace-pilot has dash so no package import)


def _decode_video(path, num_frames=16):
    import decord
    vr = decord.VideoReader(str(path))
    total = len(vr)
    n = min(num_frames, total)
    idxs = [int(i * (total - 1) / max(n - 1, 1)) for i in range(n)]
    return vr.get_batch(idxs).asnumpy()


def build_prompt(processor, item, video_max_frames=16):
    frames = _decode_video(item["video_path"], video_max_frames)
    sys_msg = ("You are an expert at answering questions about video content.\n"
               "If the question has multiple valid interpretations, enumerate each one with an answer.")
    msgs = [
        {"role": "system", "content": [{"type": "text", "text": sys_msg}]},
        {"role": "user", "content": [{"type": "video"}, {"type": "text", "text": item["prompt"]}]},
    ]
    text = processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    return text, frames


def reward_fn(prompts=None, completions=None, completion_ids=None, **reward_kwargs):
    """TRL 1.5.1+ reward signature.

    TRL passes:
      prompts: list[str | chat_messages]   (length = batch_size * num_generations)
      completions: list[str | chat_messages]
      completion_ids: list[list[int]]
      **reward_kwargs: every other dataset column, repeated num_generations times.
        We expect: id, kind, interpretations (list of dict per row).

    Returns: list[float] aligned with completions.
    """
    if completions is None:
        return []
    n = len(completions)

    def _norm(c):
        if isinstance(c, str):
            return c
        if isinstance(c, list) and c and isinstance(c[0], dict):
            # Chat-format completion: concat assistant content
            parts = []
            for m in c:
                if m.get("role") == "assistant":
                    txt = m.get("content")
                    if isinstance(txt, str):
                        parts.append(txt)
                    elif isinstance(txt, list):
                        for piece in txt:
                            if isinstance(piece, dict) and piece.get("type") == "text":
                                parts.append(piece.get("text", ""))
            return "\n".join(parts)
        return str(c)

    samples = [_norm(c) for c in completions]

    # Reconstruct per-completion item dicts from reward_kwargs
    items = []
    ids       = reward_kwargs.get("id", [None] * n)
    kinds     = reward_kwargs.get("kind", ["ambig"] * n)
    interps   = reward_kwargs.get("interpretations", [[]] * n)
    qtexts    = reward_kwargs.get("prompt", prompts if prompts is not None else [""] * n)
    for i in range(n):
        items.append({
            "id": ids[i] if i < len(ids) else None,
            "kind": kinds[i] if i < len(kinds) else "ambig",
            "interpretations": interps[i] if i < len(interps) else [],
            "prompt": qtexts[i] if i < len(qtexts) else "",
        })
    return combined_reward(samples, items)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())

    model_id = cfg["model"]["model_id"]
    adapter_init = cfg["model"]["adapter_init"]

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    # device_map=None: each distributed rank places full model on its assigned GPU.
    # device_map='auto' would shard the model across all visible GPUs and is incompatible
    # with accelerate distributed training (raises ValueError in accelerator.prepare).
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, trust_remote_code=True,
    )
    # Load v4 LoRA adapter as starting policy
    model = PeftModel.from_pretrained(model, adapter_init, is_trainable=True)

    # Build dataset
    rows = [json.loads(l) for l in Path(cfg["data"]["train_file"]).read_text().splitlines() if l.strip()]
    ds = Dataset.from_list(rows)

    # GRPO config
    grpo_cfg = GRPOConfig(
        output_dir=cfg["training"]["output_dir"],
        num_train_epochs=cfg["training"]["num_train_epochs"],
        per_device_train_batch_size=cfg["training"]["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["training"]["gradient_accumulation_steps"],
        learning_rate=cfg["training"]["learning_rate"],
        lr_scheduler_type=cfg["training"]["lr_scheduler_type"],
        warmup_ratio=cfg["training"]["warmup_ratio"],
        logging_steps=cfg["training"]["logging_steps"],
        save_steps=cfg["training"]["save_steps"],
        save_total_limit=cfg["training"]["save_total_limit"],
        bf16=cfg["training"]["bf16"],
        gradient_checkpointing=cfg["training"]["gradient_checkpointing"],
        num_generations=cfg["rl"]["n_rollouts"],
        temperature=cfg["rl"]["temperature"],
        max_completion_length=cfg["rl"]["max_new_tokens"],
        beta=cfg["rl"]["kl_beta"],
        seed=cfg["training"]["seed"],
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=processor,
        reward_funcs=[reward_fn],
        args=grpo_cfg,
        train_dataset=ds,
    )
    trainer.train()
    trainer.save_model(cfg["training"]["output_dir"])


if __name__ == "__main__":
    main()
