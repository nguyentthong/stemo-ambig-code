"""SFT trainer for InternVL3.5 family (8B / 38B).

Differences from train_sft.py (which uses AutoModelForImageTextToText for Qwen):
  - InternVL uses AutoModel + AutoTokenizer (not Processor)
  - Image preprocessing handled manually via torchvision transforms
  - Chat template is custom: <s> system\n...\n<|im_start|>user\n<image>\n...<|im_end|>...
  - Pixel values shape: (T*P, 3, 448, 448) where T=frames, P=patches/frame

Both 8B and 38B use the same training interface; size is set via the model_id.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from PIL import Image
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import (
    AutoModel,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMAGE_SIZE = 448


def build_transform():
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def decode_video_frames(path, num_frames=16):
    import decord
    vr = decord.VideoReader(str(path))
    total = len(vr)
    n = min(num_frames, total)
    idxs = [int(i * (total - 1) / max(n - 1, 1)) for i in range(n)]
    frames = vr.get_batch(idxs).asnumpy()
    tf = build_transform()
    return torch.stack([tf(Image.fromarray(f)) for f in frames])


def build_chat_text(messages, tokenizer, num_frames):
    """InternVL chat-template construction. Inserts num_frames <image> tokens
    into the user message before the question text."""
    parts = []
    for m in messages:
        role = m["role"]
        content = m["content"]
        if isinstance(content, list):
            text = ""
            for c in content:
                if c.get("type") == "video":
                    text += "<image>\n" * num_frames
                elif c.get("type") == "text":
                    text += c["text"]
            content = text
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
    parts.append("<|im_start|>assistant\n")
    return "".join(parts)


class InternVLDataset(Dataset):
    def __init__(self, path, tokenizer, max_seq_len=4096, num_frames=16):
        self.rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
        self.tok = tokenizer
        self.max_seq_len = max_seq_len
        self.num_frames = num_frames

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        msgs = r["messages"]
        # Assistant target is the LAST message of role=assistant
        assistant = next((m["content"] for m in msgs if m["role"] == "assistant"), "")
        prompt_msgs = [m for m in msgs if m["role"] != "assistant"]
        prompt_text = build_chat_text(prompt_msgs, self.tok, self.num_frames)
        full_text = prompt_text + assistant + self.tok.eos_token

        ids = self.tok(full_text, return_tensors="pt", truncation=True,
                       max_length=self.max_seq_len, add_special_tokens=False)
        input_ids = ids["input_ids"][0]
        attention_mask = ids["attention_mask"][0]

        # Mask prompt tokens in the loss
        prompt_ids = self.tok(prompt_text, return_tensors="pt",
                              add_special_tokens=False)["input_ids"][0]
        n_prompt = prompt_ids.shape[0]
        labels = input_ids.clone()
        labels[:n_prompt] = -100

        # Pixel values from video
        pixels = decode_video_frames(r["video_path"], self.num_frames)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pixel_values": pixels.to(torch.bfloat16),
        }


def collate(batch):
    # Pad input_ids/labels to longest in batch
    max_len = max(x["input_ids"].shape[0] for x in batch)
    pad_id = 0
    out = {"input_ids": [], "attention_mask": [], "labels": [], "pixel_values": []}
    for x in batch:
        pad_n = max_len - x["input_ids"].shape[0]
        out["input_ids"].append(torch.cat([x["input_ids"], torch.full((pad_n,), pad_id, dtype=torch.long)]))
        out["attention_mask"].append(torch.cat([x["attention_mask"], torch.zeros(pad_n, dtype=torch.long)]))
        out["labels"].append(torch.cat([x["labels"], torch.full((pad_n,), -100, dtype=torch.long)]))
        out["pixel_values"].append(x["pixel_values"])
    return {
        "input_ids": torch.stack(out["input_ids"]),
        "attention_mask": torch.stack(out["attention_mask"]),
        "labels": torch.stack(out["labels"]),
        "pixel_values": torch.cat(out["pixel_values"]),  # (sum_T, 3, H, W)
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())

    mid = cfg["model"]["model_id"]
    tok = AutoTokenizer.from_pretrained(mid, trust_remote_code=True, use_fast=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModel.from_pretrained(
        mid,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model.config.use_cache = False
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    if cfg["lora"]["r"] > 0:
        # LoRA mode
        lora_config = LoraConfig(
            r=cfg["lora"]["r"], lora_alpha=cfg["lora"]["alpha"],
            lora_dropout=cfg["lora"].get("dropout", 0.0),
            target_modules=cfg["lora"]["target_modules"],
            bias="none", task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    else:
        # Full fine-tuning mode (LoRA r=0)
        print(f"FFT mode: all {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.1f}M params trainable")

    train_ds = InternVLDataset(cfg["data"]["train_file"], tok,
                               max_seq_len=cfg["data"]["max_seq_len"],
                               num_frames=cfg["data"].get("video_max_frames", 16))
    eval_ds = InternVLDataset(cfg["data"]["dev_file"], tok,
                              max_seq_len=cfg["data"]["max_seq_len"],
                              num_frames=cfg["data"].get("video_max_frames", 16))
    print(f"train n={len(train_ds)}  dev n={len(eval_ds)}")

    training_args = TrainingArguments(
        output_dir=cfg["training"]["output_dir"],
        num_train_epochs=cfg["training"]["num_train_epochs"],
        per_device_train_batch_size=cfg["training"]["per_device_train_batch_size"],
        per_device_eval_batch_size=cfg["training"]["per_device_eval_batch_size"],
        gradient_accumulation_steps=cfg["training"]["gradient_accumulation_steps"],
        learning_rate=cfg["training"]["learning_rate"],
        lr_scheduler_type=cfg["training"]["lr_scheduler_type"],
        warmup_ratio=cfg["training"]["warmup_ratio"],
        weight_decay=cfg["training"]["weight_decay"],
        logging_steps=cfg["training"]["logging_steps"],
        eval_strategy="steps", eval_steps=cfg["training"]["eval_steps"],
        save_steps=cfg["training"]["save_steps"],
        save_total_limit=cfg["training"]["save_total_limit"],
        bf16=cfg["training"]["bf16"],
        gradient_checkpointing=cfg["training"]["gradient_checkpointing"],
        ddp_find_unused_parameters=cfg["training"].get("ddp_find_unused_parameters", False),
        remove_unused_columns=cfg["training"].get("remove_unused_columns", False),
        report_to=cfg["training"].get("report_to", "tensorboard"),
        logging_dir=cfg["training"].get("logging_dir"),
        seed=cfg["training"]["seed"],
    )
    trainer = Trainer(
        model=model, args=training_args,
        train_dataset=train_ds, eval_dataset=eval_ds,
        data_collator=collate,
    )
    trainer.train()
    trainer.save_model(cfg["training"]["output_dir"])


if __name__ == "__main__":
    main()
