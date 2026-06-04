"""InternVL3.5-38B video inference runner, mirroring run_qwen_video.py interface.

Uses InternVL's native `chat` method on a 16-frame video sample.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import numpy as np
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
from PIL import Image


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size: int):
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def _decode_video_frames(path, num_frames=16, image_size=448):
    import decord
    vr = decord.VideoReader(str(path))
    total = len(vr)
    n = min(num_frames, total)
    idxs = [int(i * (total - 1) / max(n - 1, 1)) for i in range(n)]
    frames = vr.get_batch(idxs).asnumpy()  # (T, H, W, 3)
    tf = build_transform(image_size)
    tensors = [tf(Image.fromarray(f)) for f in frames]
    return torch.stack(tensors)  # (T, 3, H, W)


def load_model(model_id: str, adapter: str | None = None):
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=False)
    model = AutoModel.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).eval().cuda()
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
        model = model.merge_and_unload().eval().cuda()
        print(f"loaded LoRA adapter from {adapter}")
    return model, tok


@torch.inference_mode()
def run_one(model, tok, video_path, prompt, system_prompt,
            max_new_tokens=2048, num_frames=16, temperature=0.0, num_samples=1):
    pixel = _decode_video_frames(video_path, num_frames=num_frames).to(torch.bfloat16).cuda()
    # InternVL's video chat formatting
    video_tokens = "<image>\n" * num_frames
    full = (system_prompt + "\n\n" if system_prompt else "") + video_tokens + prompt
    generation_cfg = dict(
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0.0 or num_samples > 1,
    )
    if temperature > 0:
        generation_cfg["temperature"] = temperature
        generation_cfg["top_p"] = 0.95
    if num_samples > 1:
        outputs = []
        for _ in range(num_samples):
            out = model.chat(
                tok, pixel, full, generation_cfg,
                num_patches_list=[num_frames], history=None, return_history=False,
            )
            outputs.append(out)
        return outputs
    out = model.chat(
        tok, pixel, full, generation_cfg,
        num_patches_list=[num_frames], history=None, return_history=False,
    )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="OpenGVLab/InternVL3_5-38B")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--system-prompt", default=None)
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--resume", action="store_true", default=True)
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
    print(f"total={len(rows)} done={len(done_ids)} pending={len(pending)}")
    if not pending:
        return

    print(f"loading model {args.model_id} adapter={args.adapter}")
    model, tok = load_model(args.model_id, args.adapter)
    print("model loaded.")

    with args.output.open("a") as fout:
        t0 = time.time()
        for i, r in enumerate(pending, 1):
            try:
                text = run_one(model, tok, r["video_path"], r["prompt"],
                               args.system_prompt, args.max_new_tokens,
                               temperature=args.temperature, num_samples=args.num_samples)
                err = None
            except Exception as e:  # noqa: BLE001
                text, err = ("" if args.num_samples == 1 else []), repr(e)[:300]
            key = "raw_response" if args.num_samples == 1 else "raw_responses"
            rec = {**r, key: text, "error": err}
            fout.write(json.dumps(rec) + "\n")
            fout.flush()
            if i % 10 == 0 or i == len(pending):
                rate = i / max(time.time() - t0, 0.01)
                eta = (len(pending) - i) / max(rate, 0.01) / 60
                print(f"[{i}/{len(pending)}] rate={rate:.2f}/s eta={eta:.1f}min")


if __name__ == "__main__":
    main()
