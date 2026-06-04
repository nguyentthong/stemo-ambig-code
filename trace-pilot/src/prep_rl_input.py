"""Build the RL training pool: ambig (STEMO-Ambig source) + unambig (VideoMME sample).

Each row tagged with kind ∈ {ambig, unambig}. Output:
  data_v0/stemo_ambig_rl/rl_train.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ambig-src", type=Path,
                    default=REPO_ROOT / "data_v0/stemo_ambig_sft_qwen35_v4/star_input.jsonl")
    ap.add_argument("--unambig-src", type=Path,
                    default=REPO_ROOT / "data_v0/eval/videomme.jsonl")
    ap.add_argument("--n-unambig", type=int, default=500)
    ap.add_argument("--out", type=Path,
                    default=REPO_ROOT / "data_v0/stemo_ambig_rl/rl_train.jsonl")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    # Load ambig source (already in our format)
    ambig = [json.loads(l) for l in args.ambig_src.read_text().splitlines() if l.strip()]
    for r in ambig:
        r["kind"] = "ambig"

    # Load unambig source (VideoMME). Sample n_unambig items.
    if args.unambig_src.exists():
        unambig_all = [json.loads(l) for l in args.unambig_src.read_text().splitlines() if l.strip()]
        rng.shuffle(unambig_all)
        unambig = unambig_all[: args.n_unambig]
    else:
        print(f"WARN: unambig source {args.unambig_src} not found; using empty unambig set")
        unambig = []
    for r in unambig:
        r["kind"] = "unambig"
        # Normalize fields: ensure prompt, video_path
        r.setdefault("prompt", r.get("question") or r.get("query") or "")

    all_rows = ambig + unambig
    rng.shuffle(all_rows)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(json.dumps(r) for r in all_rows) + "\n")
    n_amb = sum(1 for r in all_rows if r["kind"] == "ambig")
    n_unamb = sum(1 for r in all_rows if r["kind"] == "unambig")
    print(f"wrote {len(all_rows)} rows ({n_amb} ambig, {n_unamb} unambig) -> {args.out}")


if __name__ == "__main__":
    main()
