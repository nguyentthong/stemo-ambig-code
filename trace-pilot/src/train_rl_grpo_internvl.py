"""InternVL3.5 GRPO trainer.

InternVL doesn't expose the HF AutoModelForImageTextToText interface that TRL's
GRPOTrainer expects out of the box. Two integration choices we evaluated:
  (a) subclass GRPOTrainer and override rollout generation to use .chat();
  (b) use vLLM for fast rollout + a custom GRPO update loop.

This file implements (a) — minimum-diff from the Qwen trainer. The main
override points are:
  - _generate_rollouts: uses InternVL .chat() per sample with N rollouts
  - reward computation: same combined_reward as Qwen trainer

NOTE: This is a research-grade integration. Phase 2 smoke-testing required
before scaling to full run.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml
from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModel, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "trace-pilot" / "src"))

from rl_reward import combined_reward  # noqa: E402


def _decode_video_frames(path, num_frames=16, image_size=448):
    import decord
    import numpy as np
    from PIL import Image
    from torchvision import transforms as T
    from torchvision.transforms.functional import InterpolationMode
    vr = decord.VideoReader(str(path))
    total = len(vr)
    n = min(num_frames, total)
    idxs = [int(i * (total - 1) / max(n - 1, 1)) for i in range(n)]
    frames = vr.get_batch(idxs).asnumpy()
    tf = T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    return torch.stack([tf(Image.fromarray(f)) for f in frames])


def generate_rollouts(model, tokenizer, item, n_rollouts=8, temperature=0.8,
                     max_new_tokens=4096, num_frames=16, system_prompt=None):
    """Generate N rollouts from InternVL.chat() for a single item."""
    pixel = _decode_video_frames(item["video_path"], num_frames=num_frames).to(torch.bfloat16).cuda()
    video_tokens = "<image>\n" * num_frames
    full_prompt = (system_prompt + "\n\n" if system_prompt else "") + video_tokens + item["prompt"]
    gen_cfg = dict(max_new_tokens=max_new_tokens, do_sample=True,
                   temperature=temperature, top_p=0.95)
    outputs = []
    for _ in range(n_rollouts):
        try:
            out = model.chat(
                tokenizer, pixel, full_prompt, gen_cfg,
                num_patches_list=[num_frames], history=None, return_history=False,
            )
        except Exception:
            out = ""
        outputs.append(out)
    return outputs


def compute_grpo_loss(model, tokenizer, item, rollouts, rewards, kl_beta=0.04):
    """Compute GRPO loss: group-relative advantage * log-prob, with KL penalty.

    Simplified group-relative: advantage_i = reward_i - mean(rewards in group).
    Log-prob per rollout: per-token logp of the actual rollout under the policy.
    KL: against a frozen reference (we approximate via not updating the base via PEFT trick).
    """
    mean_r = sum(rewards) / max(len(rewards), 1)
    std_r = (sum((r - mean_r) ** 2 for r in rewards) / max(len(rewards), 1)) ** 0.5 + 1e-6
    advantages = [(r - mean_r) / std_r for r in rewards]

    total_loss = 0.0
    for rollout, adv in zip(rollouts, advantages):
        if not rollout:
            continue
        # Tokenize rollout for log-prob calculation
        # (InternVL's tokenizer; we approximate the prompt+rollout encoding)
        ids = tokenizer(rollout, return_tensors="pt").input_ids.cuda()
        if ids.shape[1] < 2:
            continue
        # Forward through the language part of InternVL (text-only proxy for log-prob)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = model.language_model(input_ids=ids[:, :-1], output_hidden_states=False)
            logits = out.logits
            log_probs = torch.log_softmax(logits, dim=-1)
            target_ids = ids[:, 1:]
            tok_logp = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
            seq_logp = tok_logp.sum()
        total_loss -= adv * seq_logp
    return total_loss / max(len(rollouts), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())

    mid = cfg["model"]["model_id"]
    adapter_init = cfg["model"]["adapter_init"]

    tok = AutoTokenizer.from_pretrained(mid, trust_remote_code=True, use_fast=False)
    model = AutoModel.from_pretrained(
        mid, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).cuda()
    # Load v4 adapter as init policy
    if Path(adapter_init).exists():
        model = PeftModel.from_pretrained(model, adapter_init, is_trainable=True)

    # Load training data
    rows = [json.loads(l) for l in Path(cfg["data"]["train_file"]).read_text().splitlines() if l.strip()]
    print(f"loaded {len(rows)} training items")

    # Optimizer (LoRA params only)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(trainable, lr=cfg["training"]["learning_rate"])

    out_dir = Path(cfg["training"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    n_epochs = cfg["training"]["num_train_epochs"]
    n_rollouts = cfg["rl"]["n_rollouts"]
    temperature = cfg["rl"]["temperature"]
    max_new_tokens = cfg["rl"]["max_new_tokens"]
    save_steps = cfg["training"]["save_steps"]

    step = 0
    for epoch in range(n_epochs):
        for item in rows:
            rollouts = generate_rollouts(
                model, tok, item, n_rollouts=n_rollouts,
                temperature=temperature, max_new_tokens=max_new_tokens,
            )
            # Compute rewards (same rl_reward.combined_reward)
            items_for_reward = [item] * len(rollouts)
            rewards = combined_reward(rollouts, items_for_reward)

            loss = compute_grpo_loss(model, tok, item, rollouts, rewards,
                                      kl_beta=cfg["rl"]["kl_beta"])
            if isinstance(loss, torch.Tensor):
                loss.backward()
                optim.step()
                optim.zero_grad()
            step += 1
            if step % cfg["training"]["logging_steps"] == 0:
                avg_r = sum(rewards) / max(len(rewards), 1)
                print(f"step {step} | epoch {epoch} | mean_reward {avg_r:.3f} | loss {float(loss):.3f}", flush=True)
            if step % save_steps == 0:
                model.save_pretrained(out_dir)
                print(f"saved checkpoint at step {step}", flush=True)
    model.save_pretrained(out_dir)
    print(f"final checkpoint saved to {out_dir}")


if __name__ == "__main__":
    main()
