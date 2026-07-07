"""Binary ambiguity-detection pass against a local vLLM OpenAI-compatible server.

Poses the human study's exact yes/no question to an open-weight model over the
same 140 items (100 ambiguous + 40 controls) used for the closed models in
experiments/model_binary_judgment.py, so the Table 4 / Figure 3 detection rate
is a measured binary judgment rather than a protocol-derived proxy.

Question wording, item set, hit/false-alarm definitions, and the output JSON
schema are identical to experiments/model_binary_judgment.py. Frames match the
model's protocol setting (8 for InternVL, 16 otherwise) via the same decord
path as run_iaa_vllm.py, so detection and clarification are read off the same
stimuli the model saw under the protocol.

Usage (run_all.sh BINARY=1 drives this):
  python server_eval/run_binary_vllm.py --base-url http://localhost:8000/v1 \
      --served-name TAG --frames 16 --output analysis/binary_judgment_TAG.json
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

VIDEOS = REPO_ROOT / "stemo/videos_h264"
ASSIGN = REPO_ROOT / "experiments/human_baseline/assignments.json"

# Identical wording to experiments/model_binary_judgment.py QUESTION_TMPL.
QUESTION_TMPL = (
    "You are shown frames sampled uniformly from a video, in temporal order. "
    "Someone asked this question about the video:\n\n\"{q}\"\n\n"
    "Could this question be about more than one moment, event, or person in "
    "this video, or is it about exactly one? Answer with exactly one word: "
    "\"multiple\" if it could be about more than one, or \"one\" if it is "
    "about exactly one.")

_frame_cache: dict = {}
_cache_lock = threading.Lock()


def encode_frames(video_path: str, num_frames: int) -> list[str]:
    """Uniform decord sampling, JPEG base64 — identical to run_iaa_vllm.py."""
    key = (video_path, num_frames)
    with _cache_lock:
        if key in _frame_cache:
            return _frame_cache[key]
    import decord
    from PIL import Image
    vr = decord.VideoReader(video_path)
    total = len(vr)
    n = min(num_frames, total)
    idxs = [int(i * (total - 1) / max(n - 1, 1)) for i in range(n)]
    arrs = vr.get_batch(idxs).asnumpy()
    encoded = []
    for a in arrs:
        buf = io.BytesIO()
        Image.fromarray(a).save(buf, format="JPEG", quality=90)
        encoded.append(base64.b64encode(buf.getvalue()).decode())
    with _cache_lock:
        _frame_cache[key] = encoded
    return encoded


def load_items() -> list[dict]:
    a = json.load(open(ASSIGN))
    return [it for it in a["items"].values() if not it.get("practice")]


def parse(ans: str):
    t = (ans or "").strip().lower()
    if re.search(r"\bmultiple\b", t):
        return "ambiguous"
    if re.search(r"\bone\b", t):
        return "unambiguous"
    return None


def chat(client, model: str, messages: list, max_tokens: int = 16) -> str:
    last = None
    for attempt in range(4):
        try:
            resp = client.chat.completions.create(
                model=model, messages=messages,
                temperature=0.0, max_tokens=max_tokens)
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"chat failed after retries: {last!r}")


def build_messages(question: str, frames_b64: list[str]) -> list:
    content = [{"type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{f}"}}
               for f in frames_b64]
    content.append({"type": "text", "text": QUESTION_TMPL.format(q=question)})
    return [{"role": "user", "content": content}]


def run_item(client, served_name: str, item: dict, frames_n: int) -> dict:
    vp = VIDEOS / f"{item['video_id']}.mp4"
    if not vp.exists():
        return {"id": item["id"], "control": item["control"],
                "error": f"FileNotFoundError({vp})"}
    try:
        frames = encode_frames(str(vp), frames_n)
        ans = chat(client, served_name, build_messages(item["question"], frames))
        return {"id": item["id"], "control": item["control"],
                "raw": ans[:60], "judgment": parse(ans)}
    except Exception as e:  # noqa: BLE001
        return {"id": item["id"], "control": item["control"],
                "error": repr(e)[:150]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--served-name", required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--concurrency", type=int, default=16)
    args = ap.parse_args()

    from openai import OpenAI
    client = OpenAI(base_url=args.base_url, api_key="local", timeout=600)

    items = load_items()
    n_amb = sum(1 for i in items if not i["control"])
    print(f"[binary] {args.served_name}: {len(items)} items "
          f"({n_amb} amb / {len(items) - n_amb} ctl), {args.frames}f", flush=True)

    rows = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futs = [pool.submit(run_item, client, args.served_name, it, args.frames)
                for it in items]
        for n, fut in enumerate(as_completed(futs), 1):
            rows.append(fut.result())
            if n % 20 == 0 or n == len(items):
                print(f"[binary] {n}/{len(items)}", flush=True)

    ok = [r for r in rows if r.get("judgment")]
    amb = [r for r in ok if not r["control"]]
    ctl = [r for r in ok if r["control"]]
    summary = {
        "model": args.served_name, "n_ok": len(ok), "n_error": len(rows) - len(ok),
        "hit": round(100 * sum(1 for r in amb if r["judgment"] == "ambiguous")
                     / max(1, len(amb)), 1),
        "false_alarm": round(100 * sum(1 for r in ctl if r["judgment"] == "ambiguous")
                             / max(1, len(ctl)), 1),
        "n_ambiguous": len(amb), "n_controls": len(ctl),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
