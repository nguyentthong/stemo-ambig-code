"""Re-grade the 126 temp=0.7 traces using Gemini as a semantic-equivalence judge.

Reads outputs/traces_temp07.jsonl, groups by (example_id, prompt_variant) into
42 groups, and per group makes 6 judge calls (3 correctness + 3 pairwise
agreement). Writes outputs/diagnostics_v2.csv with the original diagnostic
columns plus judge-based seed_correct/agree/n_correct/seed_variance.

After the first 3 groups (18 judge calls), pauses for a human spot-check.
"""

import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import google.generativeai as genai

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IN_PATH = PROJECT_ROOT / "outputs" / "traces_temp07.jsonl"
OUT_PATH = PROJECT_ROOT / "outputs" / "diagnostics_v2.csv"

if not os.environ.get("GEMINI_API_KEY"):
    sys.exit("GEMINI_API_KEY not set; aborting (judge required).")

genai.configure(api_key=os.environ["GEMINI_API_KEY"])
JUDGE = genai.GenerativeModel("gemini-3-flash-preview")

REVISION_MARKERS = [
    "wait,", "wait.", "wait ", "actually,", "let me reconsider",
    "let me re-examine", "let me re-evaluate", "let me double check",
    "let me re-watch", "let's reconsider", "let's re-examine",
    "hmm,", "or maybe", "or is it",
]
TRUNCATED_CHAR_THRESHOLD = 12000

SLICE_RUBRICS = {
    "ACH_MCQ": "The answer is a single letter (A/B/C/D) or Yes/No. Match exactly on the core letter or Yes/No, ignoring any surrounding text.",
    "ACH_BQA": "The answer is a single letter (A/B/C/D) or Yes/No. Match exactly on the core letter or Yes/No, ignoring any surrounding text.",
    "TSH": "The answer expresses a temporal ordering of two actions. Two answers are EQUIVALENT if they assert the same ordering (e.g. 'Action A before B', 'A then B', 'AB' all mean the same thing). They are DIFFERENT if they assert opposite orderings, or if one detects only one action while the other detects both.",
    "STH": "The answer states whether a scene change occurs and, if so, names two locations. Two answers are EQUIVALENT if: (a) they agree on yes/no scene change AND (b) if yes, the two location descriptions refer to recognizably the same places, allowing for paraphrase ('grassy area' ~ 'field', 'house' ~ 'living room' ~ 'indoors', 'pond' ~ 'lake' ~ 'water'). They are DIFFERENT if they disagree on yes/no, or if locations are clearly different places (e.g. 'kitchen' vs 'bedroom').",
}

PROMPT_TMPL = """You are grading whether two answers to a video question express the same conclusion. Phrasing differences should be ignored; only the underlying meaning matters.

Question: {question}

Answer A: {answer_a}

Answer B: {answer_b}

Slice-specific guidance:
{slice_rubric}

Reply with exactly one word: "EQUIVALENT" or "DIFFERENT"."""


def judge_equivalence(answer_a, answer_b, slice_type, question):
    """Return True/False/None (None = judge failed)."""
    prompt = PROMPT_TMPL.format(
        question=question,
        answer_a=answer_a,
        answer_b=answer_b,
        slice_rubric=SLICE_RUBRICS[slice_type],
    )
    try:
        resp = JUDGE.generate_content(prompt, generation_config={"temperature": 0.0})
        text = (resp.text or "").upper()
    except Exception as e:
        print(f"  judge FAILED: {type(e).__name__}: {str(e)[:120]}")
        return None
    if "EQUIVALENT" in text:
        return True
    if "DIFFERENT" in text:
        return False
    print(f"  judge returned unparseable: {text[:80]!r}")
    return None


def marker_count(text):
    t = text.lower()
    return sum(t.count(m) for m in REVISION_MARKERS)


def is_truncated(answer, chars):
    return (not answer or not answer.strip()) and chars > TRUNCATED_CHAR_THRESHOLD


def main():
    recs = [json.loads(l) for l in IN_PATH.read_text().splitlines() if l.strip()]
    groups = defaultdict(dict)
    for r in recs:
        groups[(r["video_id"], r["prompt_variant"])][r["seed"]] = r

    # Stable ordering: slice, video_id, variant
    group_keys = sorted(
        groups.keys(),
        key=lambda k: (groups[k][0]["slice"], k[0], k[1]),
    )

    rows = []
    total_calls = 0
    total_failures = 0

    for i, key in enumerate(group_keys, 1):
        vid, variant = key
        by_seed = groups[key]
        any_rec = by_seed[0]
        slice_type = any_rec["slice"]
        gt = any_rec["gt_answer"]
        question = any_rec["question"]

        print(f"[{i:2d}/42] {slice_type:<8s} {vid:<25s} {variant}")

        correct = {}
        failures = 0
        for s in (0, 1, 2):
            v = judge_equivalence(by_seed[s]["final_answer"], gt, slice_type, question)
            if v is None:
                failures += 1
            correct[s] = v
            total_calls += 1

        agree = {}
        for a, b in [(0, 1), (0, 2), (1, 2)]:
            v = judge_equivalence(
                by_seed[a]["final_answer"], by_seed[b]["final_answer"],
                slice_type, question,
            )
            if v is None:
                failures += 1
            agree[(a, b)] = v
            total_calls += 1

        total_failures += failures
        n_correct = sum(1 for v in correct.values() if v is True)
        all_agree = all(v is True for v in agree.values())
        seed_variance = 0 if all_agree else 1

        chars = [by_seed[s]["thinking_char_count"] for s in (0, 1, 2)]
        avg_chars = sum(chars) / 3
        max_chars = max(chars)
        avg_markers = sum(marker_count(by_seed[s]["thinking_trace"]) for s in (0, 1, 2)) / 3
        any_trunc = 1 if any(is_truncated(by_seed[s]["final_answer"], chars[s]) for s in (0, 1, 2)) else 0

        rows.append({
            "example_id": vid,
            "slice": slice_type,
            "prompt_variant": variant,
            "gt_answer": gt,
            "seed0_answer": by_seed[0]["final_answer"].strip(),
            "seed1_answer": by_seed[1]["final_answer"].strip(),
            "seed2_answer": by_seed[2]["final_answer"].strip(),
            "seed0_correct": correct[0],
            "seed1_correct": correct[1],
            "seed2_correct": correct[2],
            "n_correct": n_correct,
            "seed01_agree": agree[(0, 1)],
            "seed02_agree": agree[(0, 2)],
            "seed12_agree": agree[(1, 2)],
            "seed_variance": seed_variance,
            "avg_trace_chars": round(avg_chars, 1),
            "max_trace_chars": max_chars,
            "avg_marker_count": round(avg_markers, 2),
            "any_truncated": any_trunc,
            "judge_failures": failures,
        })

        if i == 3:
            print("\n=== First 3 groups judged ===")
            for r in rows:
                short = {
                    "example_id": r["example_id"],
                    "slice": r["slice"],
                    "variant": r["prompt_variant"],
                    "gt": r["gt_answer"][:40],
                    "s0_ans": r["seed0_answer"][:40],
                    "s1_ans": r["seed1_answer"][:40],
                    "s2_ans": r["seed2_answer"][:40],
                    "correct": [r["seed0_correct"], r["seed1_correct"], r["seed2_correct"]],
                    "agree(01,02,12)": [r["seed01_agree"], r["seed02_agree"], r["seed12_agree"]],
                    "n_correct": r["n_correct"],
                    "seed_variance": r["seed_variance"],
                }
                print(json.dumps(short, default=str))
            print(
                "\nFirst 3 groups judged. Spot-check the seed_correct and *_agree "
                "columns above before continuing with the remaining 39 groups."
            )
            print("Press Enter to continue, or Ctrl-C to abort.")
            try:
                input()
            except EOFError:
                print("(non-interactive stdin; auto-continuing)")

    with OUT_PATH.open("w", newline="") as f:
        fieldnames = list(rows[0].keys())
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    # Brief summary
    old_rows = list(csv.DictReader(open(PROJECT_ROOT / "outputs" / "diagnostics.csv")))
    old_var1 = sum(1 for r in old_rows if r["seed_variance"] == "1")
    old_full = sum(1 for r in old_rows if r["n_correct"] == "3")
    new_var1 = sum(1 for r in rows if r["seed_variance"] == 1)
    new_full = sum(1 for r in rows if r["n_correct"] == 3)

    print(f"\n=== summary ===")
    print(f"seed_variance=1 rows:  old={old_var1}/42  new={new_var1}/42")
    print(f"n_correct=3 rows:      old={old_full}/42  new={new_full}/42")
    print(f"total judge calls: {total_calls}, total failures: {total_failures}")
    print(f"wrote {len(rows)} rows to {OUT_PATH}")


if __name__ == "__main__":
    main()
