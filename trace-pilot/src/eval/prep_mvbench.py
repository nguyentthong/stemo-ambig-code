"""Convert MVBench to standard eval JSONL.

Expects videos already extracted under <cache>/videos_extracted/<source>/...
Builds a basename→path index across all sources, then looks up each annotation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

PROMPT_TMPL = """{question}
{options}
Answer with the option letter from the given choices directly."""

MVBENCH_TASKS = [
    "action_sequence", "action_prediction", "action_antonym", "fine_grained_action",
    "unexpected_action", "object_existence", "object_interaction", "object_shuffle",
    "moving_direction", "action_count", "moving_count", "moving_attribute",
    "state_change", "fine_grained_pose", "character_order", "scene_transition",
    "egocentric_navigation", "counterfactual_inference", "episodic_reasoning",
    "action_localization",
]


def build_video_index(extracted_root: Path) -> dict[str, str]:
    """Map basename → full path for every video under extracted_root."""
    index = {}
    for p in extracted_root.rglob("*"):
        if p.suffix.lower() in (".mp4", ".webm", ".avi", ".mov"):
            # Prefer the first hit; we don't expect collisions.
            index.setdefault(p.name, str(p))
    return index


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default="/mnt/ceph3/ec/thong/mvbench_cache")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit-per-task", type=int, default=0)
    ap.add_argument("--tasks", nargs="*", default=None)
    args = ap.parse_args()

    cache = Path(args.cache_dir)
    extracted = cache / "videos_extracted"
    if not extracted.exists():
        raise SystemExit(f"no extracted videos at {extracted}. Extract first.")
    print(f"building video index under {extracted} ...")
    vid_index = build_video_index(extracted)
    print(f"  indexed {len(vid_index)} videos")

    tasks = args.tasks or MVBENCH_TASKS
    out_records = []
    n_missing = 0
    by_task_kept = {}
    for task in tasks:
        ann_path = cache / "json" / f"{task}.json"
        if not ann_path.exists():
            continue
        records = json.loads(ann_path.read_text())
        if args.limit_per_task:
            records = records[: args.limit_per_task]
        letters = "ABCDEFGH"
        kept = 0
        for i, r in enumerate(records):
            vname = r.get("video", "")
            # Some records have a path-like value; take the basename.
            vbase = Path(vname).name if vname else ""
            vp = vid_index.get(vbase)
            if not vp:
                n_missing += 1
                continue
            options = r.get("candidates") or []
            if r["answer"] not in options:
                continue
            gold_idx = options.index(r["answer"])
            options_str = "\n".join(f"({letters[j]}) {o}" for j, o in enumerate(options))
            out_records.append({
                "id": f"mvbench_{task}_{i:04d}",
                "video_path": vp,
                "prompt": PROMPT_TMPL.format(question=r["question"], options=options_str),
                "gold": letters[gold_idx],
                "task": task,
            })
            kept += 1
        by_task_kept[task] = kept

    print(f"\nprepared {len(out_records)} MVBench items ({n_missing} missing videos)")
    for t, k in sorted(by_task_kept.items()):
        print(f"  {t:<28} {k}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(json.dumps(r) for r in out_records) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
