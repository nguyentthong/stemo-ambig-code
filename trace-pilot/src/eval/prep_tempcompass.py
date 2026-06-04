"""Convert TempCompass parquet → standard eval JSONL.

Expects videos already extracted under <cache>/videos_extracted/videos/<id>.mp4.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default="/mnt/ceph3/ec/thong/tempcompass_cache")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--task", default="multi-choice",
                    choices=["multi-choice", "yes_no", "caption_matching"])
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    cache = Path(args.cache_dir)
    parquet = cache / args.task / "test-00000-of-00001.parquet"
    if not parquet.exists():
        raise SystemExit(f"missing {parquet}")
    df = pd.read_parquet(parquet)
    print(f"loaded {len(df)} rows from {parquet}")

    video_dir = cache / "videos_extracted" / "videos"
    if not video_dir.exists():
        raise SystemExit(f"no videos at {video_dir}. Extract tempcompass_videos.zip first.")

    out = []
    n_missing = 0
    for _, row in df.iterrows():
        vid = str(row["video_id"])
        vp = video_dir / f"{vid}.mp4"
        if not vp.exists():
            n_missing += 1
            continue
        # Question already has A/B/C inline. Answer like "A. dunking a basketball" — extract letter.
        gold_letter_match = re.match(r"\s*([A-H])\b", row["answer"])
        if not gold_letter_match:
            continue
        gold = gold_letter_match.group(1)
        out.append({
            "id": f"tempcompass_mc_{len(out):05d}",
            "video_path": str(vp),
            "prompt": f"{row['question']}\nAnswer with the option letter directly.",
            "gold": gold,
            "dim": row.get("dim"),
        })
    if args.limit:
        out = out[: args.limit]
    print(f"prepared {len(out)} TempCompass items ({n_missing} missing videos)")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(json.dumps(r) for r in out) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
