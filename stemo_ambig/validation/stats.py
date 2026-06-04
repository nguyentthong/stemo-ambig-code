"""Validation statistics summary.

Run:
    python -m stemo_ambig.validation.stats
    python -m stemo_ambig.validation.stats --candidates-dir ... --validations-dir ...
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def _load_json(p: Path) -> dict | None:
    try:
        return json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _pct(num: int, denom: int) -> str:
    if denom == 0:
        return "  -"
    return f"{100.0 * num / denom:5.1f}%"


def _per_interp_correctness(validation: dict) -> tuple[int, int, int]:
    """Return (yes, no, unsure) over per-interpretation answer_correct ratings."""
    pi = validation.get("per_interpretation") or {}
    yes = sum(1 for v in pi.values() if v.get("answer_correct") == "yes")
    no = sum(1 for v in pi.values() if v.get("answer_correct") == "no")
    unsure = sum(1 for v in pi.values() if v.get("answer_correct") == "unsure")
    return yes, no, unsure


def _is_fully_clean(validation: dict) -> bool:
    if validation.get("is_genuinely_ambiguous") != "yes":
        return False
    if validation.get("interpretation_set_complete") != "yes":
        return False
    yes, no, unsure = _per_interp_correctness(validation)
    return no == 0 and unsure == 0 and yes > 0


def _stratum(ambiguity_type: str, gold_k: int | None) -> str:
    if gold_k is None:
        return f"K=? / {ambiguity_type}"
    bucket = f"K={gold_k}" if gold_k <= 4 else "K>=5"
    return f"{bucket} / {ambiguity_type}"


def _candidate_k(cand: dict) -> int:
    return len(cand.get("interpretations", []))


def report(candidates_dir: Path, validations_dir: Path) -> None:
    cand_by_id: dict[str, dict] = {}
    for p in sorted(candidates_dir.rglob("stemo_ambig_*.json")):
        if p.name.startswith("_"):
            continue
        if any(part.startswith("_") for part in p.relative_to(candidates_dir).parts):
            continue  # skip _archive/, _scratch/, etc.
        c = _load_json(p)
        if c:
            cand_by_id[c["candidate_id"]] = c

    val_by_id: dict[str, dict] = {}
    for p in sorted(validations_dir.glob("*.json")):
        v = _load_json(p)
        if v:
            val_by_id[v["candidate_id"]] = v

    total = len(cand_by_id)
    done = sum(1 for cid in cand_by_id if cid in val_by_id)

    print(f"== STEMO-Ambig validation stats ==")
    print(f"candidates:  {total}")
    print(f"validated:   {done}  ({_pct(done, total)})")
    print(f"remaining:   {total - done}")
    if done == 0:
        return

    # Top-level counters across validated.
    amb_yes = amb_no = amb_unsure = 0
    set_yes = set_no = set_unsure = 0
    fully_clean = 0
    pi_yes = pi_no = pi_unsure = 0

    by_cat: dict[str, Counter] = defaultdict(Counter)
    by_stratum: dict[str, Counter] = defaultdict(Counter)
    gold_k_dist: Counter = Counter()
    cand_k_dist: Counter = Counter()
    by_k: dict[int, Counter] = defaultdict(Counter)

    for cid, v in val_by_id.items():
        cand = cand_by_id.get(cid)
        if cand is None:
            continue
        atype = cand.get("ambiguity_type", "?")

        amb = v.get("is_genuinely_ambiguous", "unsure")
        amb_yes += amb == "yes"
        amb_no += amb == "no"
        amb_unsure += amb == "unsure"

        sc = v.get("interpretation_set_complete", "unsure")
        set_yes += sc == "yes"
        set_no += sc == "no"
        set_unsure += sc == "unsure"

        y, n, u = _per_interp_correctness(v)
        pi_yes += y
        pi_no += n
        pi_unsure += u

        clean = _is_fully_clean(v)
        fully_clean += clean

        gold_k = v.get("gold_k")
        if gold_k is not None:
            gold_k_dist[gold_k] += 1

        by_cat[atype]["validated"] += 1
        by_cat[atype]["ambiguous_yes"] += amb == "yes"
        by_cat[atype]["set_complete_yes"] += sc == "yes"
        by_cat[atype]["fully_clean"] += clean

        cand_k = _candidate_k(cand)
        cand_k_dist[cand_k] += 1
        by_k[cand_k]["validated"] += 1
        by_k[cand_k]["ambig_yes"] += amb == "yes"
        by_k[cand_k]["clean"] += clean

        stratum = _stratum(atype, gold_k)
        by_stratum[stratum]["n"] += 1
        by_stratum[stratum]["clean"] += clean
        by_stratum[stratum]["ambig_yes"] += amb == "yes"

    print()
    print(f"-- Across {done} validated --")
    print(f"genuinely ambiguous: yes {amb_yes}  no {amb_no}  unsure {amb_unsure}  ({_pct(amb_yes, done)} yes)")
    print(f"interp set complete: yes {set_yes}  no {set_no}  unsure {set_unsure}  ({_pct(set_yes, done)} yes)")
    print(f"fully clean:         {fully_clean}  ({_pct(fully_clean, done)})")
    pi_total = pi_yes + pi_no + pi_unsure
    print(f"per-interp answers:  correct {pi_yes}  wrong {pi_no}  unsure {pi_unsure}  ({_pct(pi_yes, pi_total)} correct over {pi_total} judged)")

    print()
    print("-- By K (candidate-emitted, primary axis) --")
    print(f"  {'K':<4} {'n':>4} {'amb%':>6} {'clean%':>7}")
    for k in sorted(by_k):
        n = by_k[k]["validated"]
        print(
            f"  K={k:<2} {n:>4} "
            f"{_pct(by_k[k]['ambig_yes'], n):>6} "
            f"{_pct(by_k[k]['clean'], n):>7}"
        )

    print()
    print("-- By ambiguity_type (descriptive, source-prompt tag) --")
    print(f"  {'type':<30} {'n':>4} {'amb%':>6} {'set%':>6} {'clean%':>7}")
    for atype, c in sorted(by_cat.items()):
        n = c["validated"]
        print(
            f"  {atype:<30} {n:>4} "
            f"{_pct(c['ambiguous_yes'], n):>6} "
            f"{_pct(c['set_complete_yes'], n):>6} "
            f"{_pct(c['fully_clean'], n):>7}"
        )

    print()
    print("-- Gold K distribution --")
    if gold_k_dist:
        for k in sorted(gold_k_dist):
            bar = "#" * gold_k_dist[k]
            print(f"  K={k}: {gold_k_dist[k]:>3}  {bar}")
    else:
        print("  (no gold_k recorded yet -- earlier validations did not capture this field)")

    print()
    print("-- By stratum (ambiguity_type x K-bucket) --")
    print(f"  {'stratum':<45} {'n':>4} {'amb%':>6} {'clean%':>7}")
    for stratum, c in sorted(by_stratum.items()):
        n = c["n"]
        print(
            f"  {stratum:<45} {n:>4} "
            f"{_pct(c['ambig_yes'], n):>6} "
            f"{_pct(c['clean'], n):>7}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidates-dir", type=Path,
        default=Path("data/stemo_ambig_candidates"),
    )
    parser.add_argument(
        "--validations-dir", type=Path,
        default=Path("data/stemo_ambig_validations"),
    )
    args = parser.parse_args()
    report(args.candidates_dir.resolve(), args.validations_dir.resolve())


if __name__ == "__main__":
    main()
