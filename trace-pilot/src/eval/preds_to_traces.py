"""Convert run_qwen_video.py predictions into the trace schema expected by
judge_stemo_traces.py.

run_qwen_video output: {id, video_id, video_path, prompt, raw_response, k_group, ...}
trace schema:          {id, video_id, question, category, subcategory, k_group,
                        thinking_trace, final_answer, thinking_char_count, ...}
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)


def split_response(raw: str) -> tuple[str, str]:
    if not raw:
        return "", ""
    m = _THINK_RE.search(raw)
    if not m:
        # No <think> block — treat entire response as final answer, thinking empty.
        return "", raw.strip()
    thinking = m.group(1).strip()
    final = (raw[m.end():]).strip()
    return thinking, final


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    rows = [json.loads(l) for l in args.preds.read_text().splitlines() if l.strip()]
    out = []
    for r in rows:
        thinking, final = split_response(r.get("raw_response") or "")
        out.append({
            "id": r["id"],
            "video_id": r.get("video_id"),
            "question": r.get("prompt") or r.get("question"),
            "category": r.get("category"),
            "subcategory": r.get("subcategory"),
            "k_group": r.get("k_group"),
            "thinking_trace": thinking,
            "final_answer": final,
            "thinking_char_count": len(thinking),
            "elapsed_sec": r.get("elapsed_sec"),
            "error": r.get("error"),
        })
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(json.dumps(r) for r in out) + "\n")
    print(f"wrote {len(out)} traces -> {args.out}")


if __name__ == "__main__":
    main()
