"""Compute per-(example, prompt_variant) diagnostics from the temp=0.7 sweep.

Reads outputs/traces_temp07.jsonl (126 records: 21 examples x 2 variants x 3
seeds) and writes outputs/diagnostics.csv with one row per (example, variant)
= 42 rows.
"""

import csv
import json
import re
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IN_PATH = PROJECT_ROOT / "outputs" / "traces_temp07.jsonl"
OUT_PATH = PROJECT_ROOT / "outputs" / "diagnostics.csv"

REVISION_MARKERS = [
    "wait,",
    "wait.",
    "wait ",
    "actually,",
    "let me reconsider",
    "let me re-examine",
    "let me re-evaluate",
    "let me double check",
    "let me re-watch",
    "let's reconsider",
    "let's re-examine",
    "hmm,",
    "or maybe",
    "or is it",
]

# Heuristic: max_tokens=4096 ~ 12k chars conservatively.
TRUNCATED_CHAR_THRESHOLD = 12000

_ws_re = re.compile(r"\s+")


def normalize(s):
    return _ws_re.sub(" ", s.strip().lower())


def marker_count(text):
    t = text.lower()
    return sum(t.count(m) for m in REVISION_MARKERS)


def is_truncated(answer, thinking_chars):
    return (not answer or not answer.strip()) and thinking_chars > TRUNCATED_CHAR_THRESHOLD


def main():
    recs = [json.loads(l) for l in IN_PATH.read_text().splitlines() if l.strip()]
    grouped = defaultdict(dict)  # (vid, variant) -> {seed: rec}
    for r in recs:
        grouped[(r["video_id"], r["prompt_variant"])][r["seed"]] = r

    rows = []
    for (vid, variant), by_seed in grouped.items():
        # Take any record for shared metadata
        any_rec = next(iter(by_seed.values()))
        gt = normalize(any_rec["gt_answer"])

        seed_answers = {s: normalize(by_seed[s]["final_answer"]) for s in (0, 1, 2)}
        seed_chars = {s: by_seed[s]["thinking_char_count"] for s in (0, 1, 2)}
        seed_traces = {s: by_seed[s]["thinking_trace"] for s in (0, 1, 2)}
        seed_raw_ans = {s: by_seed[s]["final_answer"] for s in (0, 1, 2)}

        unique_answers = {seed_answers[0], seed_answers[1], seed_answers[2]}
        seed_variance = 1 if len(unique_answers) > 1 else 0
        n_correct = sum(1 for a in seed_answers.values() if a == gt)
        avg_chars = sum(seed_chars.values()) / 3
        max_chars = max(seed_chars.values())
        avg_markers = sum(marker_count(seed_traces[s]) for s in (0, 1, 2)) / 3
        any_trunc = 1 if any(is_truncated(seed_raw_ans[s], seed_chars[s]) for s in (0, 1, 2)) else 0

        rows.append({
            "example_id": vid,
            "slice": any_rec["slice"],
            "prompt_variant": variant,
            "gt_answer": any_rec["gt_answer"],
            "seed0_answer": seed_answers[0],
            "seed1_answer": seed_answers[1],
            "seed2_answer": seed_answers[2],
            "seed_variance": seed_variance,
            "n_correct": n_correct,
            "avg_trace_chars": round(avg_chars, 1),
            "max_trace_chars": max_chars,
            "avg_marker_count": round(avg_markers, 2),
            "any_truncated": any_trunc,
        })

    rows.sort(key=lambda r: (r["slice"], r["example_id"], r["prompt_variant"]))
    fieldnames = list(rows[0].keys())
    with OUT_PATH.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"wrote {len(rows)} rows to {OUT_PATH}")


if __name__ == "__main__":
    main()
