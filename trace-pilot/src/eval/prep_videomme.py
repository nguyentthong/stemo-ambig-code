"""Convert VideoMME parquet to standard eval JSONL."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd

DEFAULT_PARQUET = "/mnt/ceph3/ec/thong/Video-MME/videomme/test-00000-of-00001.parquet"
DEFAULT_VIDEO_DIR = "/mnt/ceph3/ec/thong/Video-MME/videos/data"

PROMPT_TMPL = """{question}
{options}
Answer the question with A, B, C, or D."""


def find_video(video_dir, video_id):
    for ext in ("mp4", "MP4", "mkv", "webm"):
        p = Path(video_dir) / f"{video_id}.{ext}"
        if p.exists():
            return str(p)
    # try recursively
    for ext in ("mp4", "MP4"):
        hits = list(Path(video_dir).rglob(f"{video_id}.{ext}"))
        if hits:
            return str(hits[0])
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default=DEFAULT_PARQUET)
    ap.add_argument("--video-dir", default=DEFAULT_VIDEO_DIR)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    df = pd.read_parquet(args.parquet)
    print(f"loaded {len(df)} rows from {args.parquet}")
    print("cols:", list(df.columns))

    out = []
    missing = 0
    for _, row in df.iterrows():
        video_id = row.get("videoID") or row.get("video_id")
        vp = find_video(args.video_dir, video_id)
        if not vp:
            missing += 1
            continue
        options = row["options"]
        if hasattr(options, "tolist"):
            options = options.tolist()
        options_str = "\n".join(options) if isinstance(options, list) else str(options)
        out.append({
            "id": f"videomme_{row.get('question_id', video_id) }",
            "video_path": vp,
            "prompt": PROMPT_TMPL.format(question=row["question"], options=options_str),
            "gold": row["answer"],
            "duration": row.get("duration"),
            "domain": row.get("domain"),
            "subcategory": row.get("sub_category"),
        })
    print(f"prepared {len(out)} items, {missing} missing videos")
    if args.limit:
        out = out[: args.limit]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(json.dumps(r) for r in out) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
