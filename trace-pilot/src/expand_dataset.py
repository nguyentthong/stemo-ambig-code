"""Expand TSH/STH from 7 -> 30 each, preserving the existing 7+7 as a strict subset.

Reads data/pilot_examples.jsonl (the original 4+3+7+7 = 21), samples 23 more
TSH and 23 more STH from the full annotation pool (excluding the existing
video_ids), and writes data/pilot_examples_v2.jsonl with 4+3+30+30 = 67 rows.
The original 21 come first, in the same order; new rows are appended.

Uses an independent per-slice RNG (seeded distinctly from the original
seed=42) so the new samples don't collide with the originals.
"""

import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VIDHALLUC_DIR = PROJECT_ROOT.parent / "vidhalluc"
V1_PATH = PROJECT_ROOT / "data" / "pilot_examples.jsonl"
V2_PATH = PROJECT_ROOT / "data" / "pilot_examples_v2.jsonl"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from load_dataset import (  # noqa: E402
    FLATTENERS, ANNO_FILES, find_video_path,
)

EXPANSION_SEED = 43
TARGET_PER_SLICE = 30
EXPAND_SLICES = ["TSH", "STH"]


def main():
    if not V1_PATH.exists():
        sys.exit(f"missing {V1_PATH}; run load_dataset.py first")

    v1_rows = [json.loads(l) for l in V1_PATH.read_text().splitlines() if l.strip()]
    print(f"loaded {len(v1_rows)} rows from {V1_PATH.name}")

    existing_ids = {s: set() for s in EXPAND_SLICES}
    for r in v1_rows:
        if r["slice"] in existing_ids:
            existing_ids[r["slice"]].add(r["video_id"])

    new_rows = []
    for slice_name in EXPAND_SLICES:
        obj = json.load(ANNO_FILES[slice_name].open())
        flat = FLATTENERS[slice_name](obj)
        seen = set()
        flat_unique = []
        for rec in flat:
            if rec["video_id"] not in seen and rec["video_id"] not in existing_ids[slice_name]:
                seen.add(rec["video_id"])
                flat_unique.append(rec)
        need = TARGET_PER_SLICE - len(existing_ids[slice_name])
        if need <= 0:
            print(f"[{slice_name}] already have {len(existing_ids[slice_name])} >= target {TARGET_PER_SLICE}, skipping")
            continue
        rng = random.Random(EXPANSION_SEED + hash(slice_name) % 1000)
        chosen = rng.sample(flat_unique, min(need, len(flat_unique)))
        print(f"[{slice_name}] existing={len(existing_ids[slice_name])} new={len(chosen)} -> total={len(existing_ids[slice_name]) + len(chosen)}")
        for rec in chosen:
            vp = find_video_path(rec["video_id"], slice_name)
            new_rows.append({"slice": slice_name, "video_path": vp, **rec})

    with V2_PATH.open("w") as f:
        for r in v1_rows + new_rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(v1_rows) + len(new_rows)} rows to {V2_PATH}")

    missing = sum(1 for r in new_rows if not r["video_path"])
    if missing:
        print(f"WARNING: {missing}/{len(new_rows)} new rows have empty video_path")


if __name__ == "__main__":
    main()
