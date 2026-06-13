"""InternVL native-HF video QA runner.

Uses InternVLForConditionalGeneration + InternVLProcessor.apply_chat_template,
which handles InternVL's dynamic 448px tiling + video-token insertion natively.
This is the correct path — the Qwen runner's frame-stacking is incompatible with
InternVL's vision tower (produces empty output / vision-reshape crashes).

CLI mirrors run_qwen_video.py so chain_v4.sh can call it interchangeably:
  --model-id --adapter --input --output --system-prompt --max-new-tokens
  --temperature --num-samples --no-thinking --limit --resume
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import decord
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor


def decode_frames_pil(path, num_frames=8):
    """Decode num_frames uniformly with decord (no torchcodec dependency)."""
    vr = decord.VideoReader(str(path))
    total = len(vr)
    n = min(num_frames, total)
    if n == 0:
        raise RuntimeError(f"empty video: {path}")
    idxs = [int(i * (total - 1) / max(n - 1, 1)) for i in range(n)]
    frames = vr.get_batch(idxs).asnumpy()  # (n, H, W, 3)
    return [Image.fromarray(f).convert("RGB") for f in frames]


def load_model(model_id: str, adapter: str | None, dtype="bfloat16"):
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, torch_dtype=getattr(torch, dtype), device_map="auto",
    )
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
        model = model.merge_and_unload()
        print(f"loaded LoRA adapter from {adapter}", flush=True)
    model.eval()
    return model, processor


@torch.inference_mode()
def run_one(model, processor, video_path, prompt, system_prompt,
            max_new_tokens=512, num_frames=8, temperature=0.0, num_samples=1):
    # Decode frames ourselves (decord) and pass them as a pre-loaded video tensor,
    # avoiding the processor's torchcodec path. InternVL handles a frame list as
    # video natively (one set of 448px tiles per frame).
    frames = decode_frames_pil(video_path, num_frames)
    # Pass frames as N inline IMAGES, not as a "video": the -HF video path has a
    # broken vision reshape in transformers 5.5.4 (tile-count vs config mismatch),
    # but multi-image works correctly. InternVL handles a sequence of images as the
    # standard way to represent video frames.
    content = [{"type": "image", "image": f} for f in frames]
    content.append({"type": "text", "text": prompt})
    conv = []
    if system_prompt:
        conv.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})
    conv.append({"role": "user", "content": content})

    inputs = processor.apply_chat_template(
        conv, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt",
    )
    inputs = {k: (v.to(model.device) if hasattr(v, "to") else v) for k, v in inputs.items()}

    do_sample = temperature > 0.0 or num_samples > 1
    plen = inputs["input_ids"].shape[1]
    # Generate samples in a LOOP rather than num_return_sequences>1: with InternVL
    # multimodal inputs, num_return_sequences does NOT replicate pixel_values across
    # the returned sequences, so all but the first come back empty/garbage. One
    # single-sequence generate per sample keeps pixel_values aligned.
    texts = []
    for _ in range(max(1, num_samples)):
        gen_kwargs = dict(max_new_tokens=max_new_tokens, do_sample=do_sample)
        if do_sample:
            gen_kwargs["temperature"] = temperature if temperature > 0 else 0.7
            gen_kwargs["top_p"] = 0.95
        out = model.generate(**inputs, **gen_kwargs)
        texts.append(processor.decode(out[0][plen:], skip_special_tokens=True).strip())
    return texts[0] if num_samples == 1 else texts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--system-prompt", default=None)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--no-resume", dest="resume", action="store_false")
    ap.add_argument("--no-thinking", action="store_true")  # accepted for CLI parity, unused
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--num-samples", type=int, default=1)
    args = ap.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    done_ids = set()
    if args.resume and args.output.exists():
        for line in args.output.read_text().splitlines():
            if line.strip():
                done_ids.add(json.loads(line)["id"])

    rows = [json.loads(l) for l in args.input.read_text().splitlines() if l.strip()]
    pending = [r for r in rows if r["id"] not in done_ids]
    if args.limit:
        pending = pending[: args.limit]
    print(f"total={len(rows)} done={len(done_ids)} pending={len(pending)}", flush=True)
    if not pending:
        return

    print(f"loading {args.model_id} adapter={args.adapter}", flush=True)
    model, processor = load_model(args.model_id, args.adapter)
    print("model loaded.", flush=True)

    with args.output.open("a") as fout:
        t0 = time.time()
        for i, r in enumerate(pending, 1):
            try:
                text = run_one(model, processor, r["video_path"], r["prompt"],
                               args.system_prompt, args.max_new_tokens,
                               temperature=args.temperature, num_samples=args.num_samples)
                err = None
            except Exception as e:  # noqa: BLE001
                text, err = ("" if args.num_samples == 1 else []), repr(e)[:300]
            key = "raw_response" if args.num_samples == 1 else "raw_responses"
            fout.write(json.dumps({**r, key: text, "error": err}) + "\n")
            fout.flush()
            if i % 10 == 0 or i == len(pending):
                rate = i / max(time.time() - t0, 0.01)
                eta = (len(pending) - i) / max(rate, 0.01) / 60
                print(f"[{i}/{len(pending)}] rate={rate:.2f}/s eta={eta:.1f}min", flush=True)


if __name__ == "__main__":
    main()
