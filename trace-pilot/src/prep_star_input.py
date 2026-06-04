"""Prepare STaR-style sampling input: same questions, NO gold scaffold.

Unlike self_distill_input.jsonl (which gave the model gold answers as context),
STaR samples ask the model to genuinely reason about the video and enumerate
interpretations on its own. We then filter samples by judging against gold.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SRC = REPO_ROOT / "data_v0" / "stemo_ambig_sft" / "ambig" / "all_questions.json"
DEFAULT_VIDEO_DIR = REPO_ROOT / "data_v0" / "stemo_ambig_sft" / "videos"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC)
    ap.add_argument("--video-dir", type=Path, default=DEFAULT_VIDEO_DIR)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    payload = json.loads(args.src.read_text())
    out = []
    for q in payload["questions"]:
        vp = args.video_dir / f"{q['video_id']}.mp4"
        if not vp.exists():
            continue
        # Prompt: just the question. No scaffold. The system prompt (passed at
        # inference time) tells the model to enumerate if ambiguous.
        out.append({
            "id": q["id"],
            "video_id": q["video_id"],
            "video_path": str(vp),
            "prompt": q["question"],
            "k": len(q["interpretations"]),
            "k_group": q["k_group"],
            "interpretations": q["interpretations"],  # kept for judging only
        })
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(json.dumps(r) for r in out) + "\n")
    print(f"wrote {len(out)} STaR-input items -> {args.out}")


if __name__ == "__main__":
    main()
