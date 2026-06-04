"""Combine self-distilled ambig predictions + rehearsal data into v2 SFT JSONL.

Two components:
  - Ambig: each item's training target is the Qwen3-VL-generated enumeration
    (produced with gold scaffold, but scaffold is stripped from the training prompt
     — model only sees the original question at training time).
  - Rehearsal: each item's training target is the ORIGINAL short answer from
    LLaVA-Video-178K (no <think>, no re-distillation).

Splits by video_id so train/dev share no videos.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from pathlib import Path

SYSTEM_PROMPT = (
    "You are an expert at answering questions about video content.\n"
    "Watch the video carefully and answer the question.\n"
    "Think step by step before giving your final answer.\n"
    "If the question has multiple valid interpretations because of an ambiguous "
    "referent, enumerate each interpretation explicitly and provide an answer for each."
)


def parse_self_distill_response(raw: str) -> str:
    """Extract the clean enumeration response from a possibly noisy Qwen output.

    Qwen3-VL-Thinking may emit a <think>...</think> block followed by the answer.
    We want the assistant target to NOT contain a long unsupervised think block —
    take the post-</think> portion if present, else use raw.
    """
    if not raw:
        return ""
    raw = raw.strip()
    m = re.search(r"</think>\s*", raw)
    if m:
        return raw[m.end():].strip()
    return raw


def build_ambig_chat_record(input_row, pred_row):
    """User message is the ORIGINAL question (no scaffold).
    Assistant target is the Qwen-generated enumeration."""
    target = parse_self_distill_response(pred_row.get("raw_response", ""))
    return {
        "video_path": input_row["video_path"],
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "video", "video": input_row["video_path"]},
                {"type": "text", "text": input_row["original_question"]},
            ]},
            {"role": "assistant", "content": target},
        ],
        "meta": {"id": input_row["id"], "kind": "ambig_self_distilled",
                 "video_id": input_row["video_id"], "k_group": input_row["k_group"]},
    }


def validate_ambig_target(text: str, n_interpretations: int) -> bool:
    """Heuristic check that the self-distilled response is actually enumerated.
    Rejects empties, single-commits, and responses that don't list ≥2 interps."""
    if not text or len(text) < 20:
        return False
    # Look for at least 2 enumeration markers (→ or "Interpretation N:")
    n_arrows = text.count("→")
    n_markers = len(re.findall(r"interpretation\s*\d", text, re.IGNORECASE))
    if n_arrows >= 2 or n_markers >= 2:
        return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-distill-input", type=Path, required=True,
                    help="JSONL of self-distillation prompts (with original_question, k_group, etc)")
    ap.add_argument("--self-distill-preds", type=Path, required=True,
                    help="JSONL of Qwen predictions on the self-distillation prompts")
    ap.add_argument("--rehearsal", type=Path, required=True,
                    help="JSONL of rehearsal records (already in chat format)")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--dev-frac", type=float, default=0.05,
                    help="fraction of video_ids held out as dev")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    inputs_by_id = {}
    for line in Path(args.self_distill_input).read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            inputs_by_id[r["id"]] = r
    print(f"loaded {len(inputs_by_id)} ambig inputs")

    ambig_records = []
    n_dropped = 0
    for line in Path(args.self_distill_preds).read_text().splitlines():
        if not line.strip():
            continue
        pred = json.loads(line)
        item_id = pred.get("id")
        inp = inputs_by_id.get(item_id)
        if inp is None:
            n_dropped += 1
            continue
        target = parse_self_distill_response(pred.get("raw_response", ""))
        if not validate_ambig_target(target, inp["k"]):
            n_dropped += 1
            continue
        ambig_records.append(build_ambig_chat_record(inp, pred))
    print(f"kept {len(ambig_records)} ambig items (dropped {n_dropped})")

    rehearsal_records = [json.loads(l) for l in Path(args.rehearsal).read_text().splitlines() if l.strip()]
    print(f"rehearsal: {len(rehearsal_records)} items")

    all_records = ambig_records + rehearsal_records
    print(f"v2 total: {len(all_records)} "
          f"(ambig {len(ambig_records)}={len(ambig_records)/len(all_records)*100:.0f}%, "
          f"rehearsal {len(rehearsal_records)}={len(rehearsal_records)/len(all_records)*100:.0f}%)")

    # Split by video_id
    rng = random.Random(args.seed)
    vids = sorted({r["meta"]["video_id"] for r in all_records})
    rng.shuffle(vids)
    n_dev_vids = max(1, int(len(vids) * args.dev_frac))
    dev_vids = set(vids[:n_dev_vids])
    train = [r for r in all_records if r["meta"]["video_id"] not in dev_vids]
    dev = [r for r in all_records if r["meta"]["video_id"] in dev_vids]
    rng.shuffle(train)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "sft_train.jsonl").write_text("\n".join(json.dumps(r) for r in train) + "\n")
    (args.out_dir / "sft_dev.jsonl").write_text("\n".join(json.dumps(r) for r in dev) + "\n")
    meta = {
        "n_train": len(train), "n_dev": len(dev),
        "n_ambig": len(ambig_records), "n_rehearsal": len(rehearsal_records),
        "n_videos_train": len({r["meta"]["video_id"] for r in train}),
        "n_videos_dev": len(dev_vids),
        "ambig_fraction": len(ambig_records) / max(len(all_records), 1),
        "system_prompt": SYSTEM_PROMPT,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
    }
    (args.out_dir / "sft_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
