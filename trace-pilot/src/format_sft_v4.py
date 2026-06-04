"""Combine STaR-filtered ambig traces (CoT PRESERVED) + rehearsal into v4 SFT JSONL.

Key difference from format_sft_v2.py (v3 recipe): we DO NOT strip <think>...</think>.
The training target for ambig items is the full self-generated trace:
  <think>{Qwen's own real video reasoning}</think>\\n\\n{correct enumeration}

This teaches the model to think briefly AND enumerate. Robust to inference mode.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

SYSTEM_PROMPT = (
    "You are an expert at answering questions about video content.\n"
    "Watch the video carefully and answer the question.\n"
    "Think step by step before giving your final answer.\n"
    "If the question has multiple valid interpretations because of an ambiguous "
    "referent, enumerate each interpretation explicitly and provide an answer for each."
)


def build_ambig_chat(row):
    # Preserve CoT — full response (think + final) is the target.
    if row.get("think"):
        target = f"<think>{row['think']}</think>\n\n{row['final']}"
    else:
        target = row["final"] or row.get("full_response", "")
    return {
        "video_path": row["video_path"],
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "video", "video": row["video_path"]},
                {"type": "text", "text": row["prompt"]},
            ]},
            {"role": "assistant", "content": target},
        ],
        "meta": {"id": row["id"], "kind": "ambig_star", "video_id": row["video_id"],
                 "k_group": row["k_group"], "n_correct": row.get("n_correct"),
                 "n_addressed": row.get("n_addressed")},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--star-kept", type=Path, required=True,
                    help="STaR-filtered ambig items (output of star_filter.py)")
    ap.add_argument("--rehearsal", type=Path, required=True,
                    help="JSONL of rehearsal records, already in chat format")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--dev-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--target-ambig-frac", type=float, default=0.20,
                    help="Up-sample ambig records to hit this fraction of the train mix.")
    args = ap.parse_args()

    ambig_rows = [json.loads(l) for l in args.star_kept.read_text().splitlines() if l.strip()]
    ambig_records_unique = [build_ambig_chat(r) for r in ambig_rows]
    rehearsal_records = [json.loads(l) for l in args.rehearsal.read_text().splitlines() if l.strip()]

    # Up-sample ambig to hit target fraction: ambig_n = frac*(ambig_n + rehearsal_n)
    # → ambig_n = rehearsal_n * frac / (1-frac).  Round to nearest multiple of unique count.
    if 0.0 < args.target_ambig_frac < 1.0 and ambig_records_unique:
        target_ambig_n = max(
            len(ambig_records_unique),
            int(round(len(rehearsal_records) * args.target_ambig_frac / (1.0 - args.target_ambig_frac))),
        )
        n_repeat = max(1, round(target_ambig_n / len(ambig_records_unique)))
        ambig_records = ambig_records_unique * n_repeat
        print(f"ambig unique: {len(ambig_records_unique)}  upsample x{n_repeat} -> {len(ambig_records)}  "
              f"rehearsal: {len(rehearsal_records)}  target_ambig_frac={args.target_ambig_frac}")
    else:
        ambig_records = ambig_records_unique
        print(f"ambig (STaR): {len(ambig_records)}  rehearsal: {len(rehearsal_records)}")

    all_records = ambig_records + rehearsal_records
    rng = random.Random(args.seed)
    vids = sorted({r["meta"]["video_id"] for r in all_records})
    rng.shuffle(vids)
    n_dev = max(1, int(len(vids) * args.dev_frac))
    dev_vids = set(vids[:n_dev])
    train = [r for r in all_records if r["meta"]["video_id"] not in dev_vids]
    dev = [r for r in all_records if r["meta"]["video_id"] in dev_vids]
    rng.shuffle(train)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "sft_train.jsonl").write_text("\n".join(json.dumps(r) for r in train) + "\n")
    (args.out_dir / "sft_dev.jsonl").write_text("\n".join(json.dumps(r) for r in dev) + "\n")
    meta = {
        "n_train": len(train), "n_dev": len(dev),
        "n_ambig_star": len(ambig_records), "n_rehearsal": len(rehearsal_records),
        "ambig_fraction": len(ambig_records) / max(len(all_records), 1),
        "system_prompt": SYSTEM_PROMPT,
        "cot_preserved": True,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
    }
    (args.out_dir / "sft_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
