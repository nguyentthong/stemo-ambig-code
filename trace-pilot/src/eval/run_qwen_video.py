"""Generic Qwen3-VL inference runner for regression evals.

Reads a JSONL of {id, video_path, prompt, ...} and writes predictions JSONL
with {id, prediction, raw_response, ...}.

Supports optional LoRA adapter:
  --adapter <path-to-trained-lora>

Single-GPU; for large benchmarks run multiple shards.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor


def _decode_video(path, num_frames=16):
    import decord  # noqa: WPS433
    vr = decord.VideoReader(str(path))
    total = len(vr)
    n = min(num_frames, total)
    if n == 0:
        raise RuntimeError(f"empty video: {path}")
    idxs = [int(i * (total - 1) / max(n - 1, 1)) for i in range(n)]
    return vr.get_batch(idxs).asnumpy()


def load_model(model_id: str, adapter: str | None, dtype="bfloat16"):
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        torch_dtype=getattr(torch, dtype),
        trust_remote_code=True,
        device_map="auto",
    )
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
        model = model.merge_and_unload()
        print(f"loaded LoRA adapter from {adapter}")
    model.eval()
    return model, processor


def build_messages(prompt: str, system_prompt: str | None):
    """Build chat messages with list-of-dict content (Qwen3-VL processor requires).
    Video placeholder is included; actual frames are passed separately."""
    msgs = []
    if system_prompt:
        msgs.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})
    msgs.append({"role": "user", "content": [
        {"type": "video"},
        {"type": "text", "text": prompt},
    ]})
    return msgs


@torch.inference_mode()
def run_one(model, processor, video_path, prompt, system_prompt, max_new_tokens=512,
            video_max_frames=16, enable_thinking=None, temperature=0.0, num_samples=1):
    """Returns a single decoded string when num_samples==1, else a list of strings."""
    frames = _decode_video(video_path, video_max_frames)
    messages = build_messages(prompt, system_prompt)
    tmpl_kwargs = {"add_generation_prompt": True, "tokenize": False}
    if enable_thinking is not None:
        tmpl_kwargs["enable_thinking"] = enable_thinking
    try:
        prompt_text = processor.apply_chat_template(messages, **tmpl_kwargs)
    except TypeError:
        tmpl_kwargs.pop("enable_thinking", None)
        prompt_text = processor.apply_chat_template(messages, **tmpl_kwargs)
    inputs = processor(text=[prompt_text], videos=[frames], return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    do_sample = temperature > 0.0 or num_samples > 1
    gen_kwargs = dict(max_new_tokens=max_new_tokens, do_sample=do_sample)
    if do_sample:
        gen_kwargs["temperature"] = temperature if temperature > 0 else 0.7
        gen_kwargs["top_p"] = 0.95
        gen_kwargs["num_return_sequences"] = num_samples
    out = model.generate(**inputs, **gen_kwargs)
    prompt_len = inputs["input_ids"].shape[1]
    texts = [processor.tokenizer.decode(out[i][prompt_len:], skip_special_tokens=True).strip()
             for i in range(out.shape[0])]
    return texts[0] if num_samples == 1 else texts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="Qwen/Qwen3-VL-32B-Thinking")
    ap.add_argument("--adapter", default=None, help="Optional LoRA adapter path")
    ap.add_argument("--input", type=Path, required=True,
                    help="JSONL with {id, video_path, prompt} per line")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--system-prompt", default=None)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--no-thinking", action="store_true",
                    help="Pass enable_thinking=False to the chat template.")
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="Sampling temperature (0 = greedy).")
    ap.add_argument("--num-samples", type=int, default=1,
                    help="Number of samples per item (for STaR-style data generation).")
    args = ap.parse_args()
    enable_thinking = False if args.no_thinking else None

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
    model, processor = load_model(args.model_id, args.adapter)
    print("model loaded.")

    with args.output.open("a") as fout:
        t0 = time.time()
        for i, r in enumerate(pending, 1):
            try:
                text = run_one(model, processor, r["video_path"], r["prompt"],
                               args.system_prompt, args.max_new_tokens,
                               enable_thinking=enable_thinking,
                               temperature=args.temperature,
                               num_samples=args.num_samples)
                err = None
            except Exception as e:  # noqa: BLE001
                text, err = ("" if args.num_samples == 1 else []), repr(e)[:300]
            key = "raw_response" if args.num_samples == 1 else "raw_responses"
            rec = {**r, key: text, "error": err}
            fout.write(json.dumps(rec) + "\n")
            fout.flush()
            if i % 10 == 0 or i == len(pending):
                rate = i / max(time.time() - t0, 0.01)
                eta_min = (len(pending) - i) / max(rate, 0.01) / 60
                print(f"[{i}/{len(pending)}] rate={rate:.2f}/s eta={eta_min:.1f}min")


if __name__ == "__main__":
    main()
