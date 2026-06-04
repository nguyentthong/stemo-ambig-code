"""Sample rehearsal data from LLaVA-Video-178K.

For capability preservation: pull videos+QA from non-test, non-train-leakage subsets
of LLaVA-Video-178K. Each item retains its ORIGINAL short response (no re-distillation).

Default sources: 4 ActivityNet-QA subsets (0_30s / 30_60s / 1_2m / 2_3m).
Excludes NeXT-QA (used for ambig SFT) and PerceptionTest (potential leakage with MVBench).

Output: rehearsal.jsonl in our SFT chat format (no <think>, short response).
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

LLV178_ROOT = Path("/mnt/ceph3/ec/xiaonan/data/LLaVA-Video-178K")

DEFAULT_SUBSETS = [
    "0_30_s_activitynetqa",
    "30_60_s_activitynetqa",
    "1_2_m_activitynetqa",
    "2_3_m_activitynetqa",
]

SYSTEM_PROMPT = (
    "You are an expert at answering questions about video content.\n"
    "Watch the video carefully and answer the question.\n"
    "Think step by step before giving your final answer.\n"
    "If the question has multiple valid interpretations because of an ambiguous "
    "referent, enumerate each interpretation explicitly and provide an answer for each."
)


def resolve_video_path(subset_dir: Path, rel_path: str) -> Path | None:
    """LLaVA-Video-178K uses paths like 'ActivityNet-QA/activitynet/train/X.mp4'.
    Resolve against the subset dir, plus try a few common variants.
    """
    candidates = [
        subset_dir / rel_path,
        subset_dir / Path(rel_path).name,
        # Sometimes the rel_path has v1-3/train_val/ that doesn't exist on disk
        subset_dir / "ActivityNet-QA" / "activitynet" / "train" / Path(rel_path).name,
        subset_dir / "ActivityNet-QA" / "activitynet" / Path(rel_path).name,
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def load_subset(subset_name: str) -> list[dict]:
    """Read one LLaVA-Video-178K subset's OE+MC JSONs, return items with absolute video_path."""
    sub_dir = LLV178_ROOT / subset_name
    out = []
    for suffix in ("_oe_qa_processed.json", "_mc_qa_processed.json"):
        p = sub_dir / f"{subset_name}{suffix}"
        if not p.exists():
            continue
        records = json.loads(p.read_text())
        for r in records:
            convs = r.get("conversations") or []
            if len(convs) < 2:
                continue
            human_turn = next((c for c in convs if c.get("from") == "human"), None)
            gpt_turn = next((c for c in convs if c.get("from") == "gpt"), None)
            if not human_turn or not gpt_turn:
                continue
            # Strip the <image>\n prefix and the "Answer with..." suffix the LLaVA-Video data uses.
            q = (human_turn.get("value") or "").replace("<image>\n", "").strip()
            a = (gpt_turn.get("value") or "").strip()
            if not q or not a:
                continue
            vp = resolve_video_path(sub_dir, r.get("video", ""))
            if vp is None:
                continue
            out.append({
                "id": f"rehearsal_{r.get('id', f'{subset_name}_{len(out)}')}",
                "video_id": Path(r.get("video", "")).stem,
                "video_path": str(vp),
                "question": q,
                "answer": a,
                "data_source": subset_name,
            })
    return out


def to_chat_record(item: dict) -> dict:
    """Convert a rehearsal item into our SFT chat schema.

    Assistant target is the original short answer — NO <think> block.
    This explicitly trains the model to give direct answers when the question
    is not ambiguous, preserving base-model behavior on standard QA.
    """
    return {
        "video_path": item["video_path"],
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "video", "video": item["video_path"]},
                {"type": "text", "text": item["question"]},
            ]},
            {"role": "assistant", "content": item["answer"]},
        ],
        "meta": {"id": item["id"], "kind": "rehearsal",
                 "video_id": item["video_id"], "source": item["data_source"]},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subsets", nargs="*", default=DEFAULT_SUBSETS)
    ap.add_argument("--n", type=int, default=8500)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    pool = []
    for sub in args.subsets:
        items = load_subset(sub)
        print(f"  {sub}: loaded {len(items)} items with resolvable videos")
        pool.extend(items)
    print(f"total pool: {len(pool)}")

    rng = random.Random(args.seed)
    rng.shuffle(pool)
    sample = pool[: args.n]
    print(f"sampled {len(sample)} for rehearsal")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for it in sample:
            f.write(json.dumps(to_chat_record(it)) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
