"""Score predictions JSONL against gold.

Handles MCQ (extract A/B/C/D) and yes/no normalization. Reports overall accuracy
and per-task / per-category breakdowns.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


def extract_mcq_letter(text: str) -> str | None:
    """Robust MCQ-letter extractor for reasoning-model outputs.

    Tries in order:
      1. Explicit answer phrase  ("the answer is X", "final answer: X", "(X)")
      2. Standalone letter at the start of cleaned text
      3. Last standalone A-H letter in the response (often the conclusion)
    """
    if not text:
        return None
    # Strip <think>...</think> blocks if present.
    cleaned = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()
    if not cleaned:
        cleaned = text.strip()
    if not cleaned:
        return None

    # 1. Explicit answer phrases — strongest signal.
    explicit_patterns = [
        r"(?:final\s+answer|the\s+answer|correct\s+answer|answer\s+is|answer)\s*[:\-=is]*\s*\*{0,2}\(?([A-H])\)?\b",
        r"\boption\s+\(?([A-H])\)?\b",
        r"\bchoose\s+\(?([A-H])\)?\b",
    ]
    for pat in explicit_patterns:
        m = re.search(pat, cleaned, re.IGNORECASE)
        if m:
            return m.group(1).upper()

    # 2. Standalone letter at start.
    m = re.match(r"\s*\(?([A-H])\)?[\.\)\:\s]", cleaned + " ")
    if m:
        return m.group(1).upper()

    # 3. Letter in parens anywhere — usually a deliberate choice marker.
    m = re.search(r"\(([A-H])\)", cleaned)
    if m:
        return m.group(1).upper()

    # 4. LAST standalone A-H letter as fallback (often the conclusion).
    # Avoid catching "A" inside words by requiring word boundaries on both sides.
    matches = re.findall(r"(?<![A-Za-z])([A-H])(?![A-Za-z])", cleaned)
    if matches:
        return matches[-1].upper()

    return None


def extract_yesno(text: str) -> str | None:
    """Robust yes/no extractor for reasoning-model outputs.

    Reasoning models (base Qwen3-VL-Thinking) bury the verdict at the END of a
    long ramble, so first-N-chars heuristics fail. Strategy:
      1. strip <think>...</think>
      2. explicit answer phrase ("answer is yes", "the answer: no")
      3. bare yes/no at the very start
      4. LAST standalone yes/no token in the whole response (the conclusion)
    """
    if not text:
        return None
    cleaned = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()
    if not cleaned:
        cleaned = text.strip()
    low = cleaned.lower()

    # 1. Explicit answer phrase
    m = re.search(r"(?:final answer|the answer|answer is|answer:|conclusion:?)\s*[:\-is]*\s*\**\s*\b(yes|no)\b",
                  low, re.IGNORECASE)
    if m:
        return m.group(1).capitalize()

    # 2. Bare yes/no at the start
    m = re.match(r"\s*\**\s*(yes|no)\b", low)
    if m:
        return m.group(1).capitalize()

    # 3. LAST standalone yes/no in the response (the conclusion)
    matches = re.findall(r"\b(yes|no)\b", low)
    if matches:
        return matches[-1].capitalize()

    return None


def score(rec, kind):
    raw = (rec.get("raw_response") or "").strip()
    # Strip any <think>...</think> wrapper before scoring (Qwen3-VL Thinking emits it)
    cleaned = re.sub(r"<think>.*?</think>\s*", "", raw, flags=re.DOTALL).strip()
    gold = rec.get("gold", "").strip()
    if kind == "yesno":
        pred = extract_yesno(cleaned)
        ok = pred is not None and pred.lower() == gold.lower()
    else:
        pred = extract_mcq_letter(cleaned)
        ok = pred is not None and pred.upper() == gold.upper()
    return pred, ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", type=Path, required=True)
    ap.add_argument("--kind", choices=["mcq", "yesno"], default="mcq")
    ap.add_argument("--group-by", default=None,
                    help="meta field to break down by (e.g. duration, domain, task, subtask)")
    ap.add_argument("--out", type=Path, default=None,
                    help="optional output JSON for the metrics summary")
    args = ap.parse_args()

    rows = [json.loads(l) for l in args.predictions.read_text().splitlines() if l.strip()]
    n = len(rows)
    n_correct = 0
    n_parsed = 0
    by_group = defaultdict(lambda: [0, 0])  # group -> [correct, total]
    for r in rows:
        pred, ok = score(r, args.kind)
        if pred is not None:
            n_parsed += 1
        if ok:
            n_correct += 1
        if args.group_by:
            g = r.get(args.group_by, "_unknown")
            by_group[g][1] += 1
            if ok:
                by_group[g][0] += 1
    summary = {
        "n": n,
        "n_parsed": n_parsed,
        "n_correct": n_correct,
        "accuracy": n_correct / max(n, 1),
        "parse_rate": n_parsed / max(n, 1),
        "by_group": {g: {"correct": c, "total": t, "acc": c / max(t, 1)}
                     for g, (c, t) in by_group.items()},
    }
    print(json.dumps({"n": n, "accuracy": summary["accuracy"],
                       "parse_rate": summary["parse_rate"]}, indent=2))
    if summary["by_group"]:
        print("\nby_group:")
        for g, m in sorted(summary["by_group"].items()):
            print(f"  {g:<24} {m['correct']:>4}/{m['total']:<4} = {m['acc']:.3f}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
