"""v5 offline RL: select top-K rollouts per item (by reward) for SFT.

Input:
  --judged    judged_rollouts.jsonl   (from v5_judge_rollouts.py — has `reward`)
  --input     star_input.jsonl        (gold metadata, for fields downstream SFT needs)
  --topk N    how many rollouts per item to keep (default 2)
  --min-reward FLOAT  drop rollouts below this reward (default 0.5)
  --out       output JSONL formatted as v4 SFT input (one row per kept rollout,
              with `chosen_response`, `kept_idx`, plus gold fields).

Output format matches what format_sft_v4.py expects in `star_kept_aug.jsonl`:
  - id, video_id, video_path, prompt, k, k_group, interpretations
  - kept_idx, chosen_response (the rollout we picked)
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


def split_think(raw):
    """Split a response into (think, final). Mirrors star_filter's logic."""
    if not raw:
        return "", ""
    m = re.search(r"<think>(.*?)</think>", raw, flags=re.DOTALL)
    if m:
        return m.group(1).strip(), raw[m.end():].strip()
    if "</think>" in raw:
        idx = raw.index("</think>")
        return raw[:idx].strip(), raw[idx + len("</think>"):].strip()
    return "", raw.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judged", type=Path, required=True)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--topk", type=int, default=2)
    ap.add_argument("--min-reward", type=float, default=0.5,
                    help="drop rollouts below this reward (must demonstrate at least half-K correct on K=2 items)")
    args = ap.parse_args()

    inputs_by_id = {json.loads(l)["id"]: json.loads(l)
                    for l in args.input.read_text().splitlines() if l.strip()}
    judged = [json.loads(l) for l in args.judged.read_text().splitlines() if l.strip()]
    print(f"loaded {len(inputs_by_id)} inputs, {len(judged)} judged rollouts", flush=True)

    # group by item
    by_id = defaultdict(list)
    for j in judged:
        by_id[j["id"]].append(j)

    kept_rows = []
    n_items_with_any = 0
    n_kept = 0
    reward_dist = []
    for qid, rolls in by_id.items():
        item = inputs_by_id.get(qid)
        if item is None:
            continue
        # filter + sort by reward
        good = [r for r in rolls if r["reward"] >= args.min_reward and r["n_correct"] > 0]
        if not good:
            continue
        good.sort(key=lambda r: (-r["reward"], len(r["response"])))
        picks = good[:args.topk]
        n_items_with_any += 1
        for r in picks:
            think, final = split_think(r["response"])
            kept_rows.append({
                "id": item["id"],
                "video_id": item.get("video_id", ""),
                "video_path": item.get("video_path", ""),
                "prompt": item["prompt"],
                "k": item.get("k", len(item.get("interpretations", []))),
                "k_group": item.get("k_group", ""),
                "interpretations": item["interpretations"],
                "kept_idx": r["rollout_idx"],
                "chosen_response": r["response"],
                "full_response": r["response"],
                "think": think,
                "final": final,
                "reward": r["reward"],
                "n_correct": r["n_correct"],
                "n_addressed": r["n_addressed"],
            })
            n_kept += 1
            reward_dist.append(r["reward"])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for r in kept_rows:
            f.write(json.dumps(r) + "\n")
    mean = sum(reward_dist) / len(reward_dist) if reward_dist else 0
    print(f"kept {n_kept} rollouts across {n_items_with_any} items (mean reward {mean:.3f}) -> {args.out}")


if __name__ == "__main__":
    main()
