"""Convert STEMO-Ambig test candidates to eval JSONL for run_qwen_video.py."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SRC = REPO_ROOT / "data_v0" / "stemo_ambig_candidates" / "all_questions.json"
DEFAULT_VIDEO_DIR = REPO_ROOT / "stemo" / "videos_h264"

# Match the system prompt used during SFT (without it, model behavior is undefined).
SYSTEM_PROMPT = (
    "You are an expert at answering questions about video content.\n"
    "Watch the video carefully and answer the question.\n"
    "Think step by step before giving your final answer.\n"
    "If the question has multiple valid interpretations because of an ambiguous "
    "referent, enumerate each interpretation explicitly and provide an answer for each."
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC)
    ap.add_argument("--video-dir", type=Path, default=DEFAULT_VIDEO_DIR)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    data = json.loads(args.src.read_text())
    questions = data["questions"]
    if args.limit:
        questions = questions[: args.limit]

    out = []
    missing = 0
    for q in questions:
        vp = args.video_dir / f"{q['video_id']}.mp4"
        if not vp.exists():
            missing += 1
            continue
        out.append({
            "id": q["id"],
            "video_id": q["video_id"],
            "video_path": str(vp),
            "prompt": q["question"],
            "k_group": q["k_group"],
            "category": q["category"],
            "subcategory": q["subcategory"],
        })
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(json.dumps(r) for r in out) + "\n")
    print(f"wrote {len(out)} items ({missing} missing videos) -> {args.out}")
    # Print system prompt for the user to feed into run_qwen_video.py
    print(f"\nSystem prompt to pass via --system-prompt:")
    print(repr(SYSTEM_PROMPT))


if __name__ == "__main__":
    main()
