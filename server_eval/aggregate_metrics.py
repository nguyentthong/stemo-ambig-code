"""Aggregate IAA predictions into the metrics the paper's figures need.

Reads eval_runs/<tag>/iaa_predictions.jsonl for every tag given, joins with
all_questions.json for subset labels (entity vs event) and K bins, then
writes analysis/open_weight_iaa_metrics.json and prints, ready to paste:
  - Table 1 row per model (score, strict-K, recognition, clarification, F/T)
  - fig_perk SCORE lines (ReQueST score by K bin: 2 / 3 / 4-6 / 7+)
  - fig_subsets rows (entity strict-K, event strict-K)

Subset mapping (sums to Entity 555 / Event 490 / Mixed 11, matching S2):
  entity: shared_attribute_different_entities, entities_in_same_event,
          multiple_entities, repeated_entities
  event:  repeated_action, same_entity_multiple_moments,
          repeated_temporal_referent
  mixed:  none
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
QUESTIONS = REPO_ROOT / "data_v0/stemo_ambig_candidates/all_questions.json"

ENTITY = {"shared_attribute_different_entities", "entities_in_same_event",
          "multiple_entities", "repeated_entities"}
EVENT = {"repeated_action", "same_entity_multiple_moments",
         "repeated_temporal_referent"}


def kbin(K: int) -> str:
    return "2" if K == 2 else "3" if K == 3 else "4-6" if K <= 6 else "7+"


def subset_of(subcat: str) -> str:
    if subcat in ENTITY:
        return "Entity"
    if subcat in EVENT:
        return "Event"
    return "Mixed"


def safe_mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def summarize(recs: list[dict]) -> dict:
    valid = [r for r in recs if r.get("score") and not r.get("error")]
    cats = [r["classification"]["category"] for r in valid]
    clar_cats = {"clarified_scope", "clarified_vague"}
    recog_cats = {"enumerated", "clarified_scope"}
    clar = [r for r in valid
            if r["classification"]["category"] in clar_cats]
    return {
        "n": len(valid),
        "n_errored": sum(1 for r in recs if r.get("error")),
        "iaa": safe_mean([r["score"]["iaa_score"] for r in valid]),
        "strict_K": safe_mean(
            [1.0 if r["score"]["strict_K_correct"] else 0.0 for r in valid]),
        "recognition": safe_mean(
            [1.0 if c in recog_cats else 0.0 for c in cats]),
        "clarification_rate": safe_mean(
            [1.0 if c in clar_cats else 0.0 for c in cats]),
        "follow_through": safe_mean(
            [1.0 if r["score"]["follow_through_correct"] else 0.0
             for r in clar]) if clar else 0.0,
    }


def main():
    tags = sys.argv[1:]
    if not tags:
        tags = sorted(p.parent.name for p in
                      (REPO_ROOT / "eval_runs").glob("*/iaa_predictions.jsonl"))
    qmeta = {q["id"]: q for q in json.load(open(QUESTIONS))["questions"]}

    out = {}
    for tag in tags:
        pred = REPO_ROOT / "eval_runs" / tag / "iaa_predictions.jsonl"
        if not pred.exists():
            print(f"[aggregate] SKIP {tag}: no predictions file")
            continue
        recs = [json.loads(l) for l in pred.read_text().splitlines()
                if l.strip()]
        for r in recs:
            meta = qmeta.get(r["id"], {})
            r["_subset"] = subset_of(meta.get("subcategory", "none"))
            r["_kbin"] = kbin(r["K"])
        m = summarize(recs)
        m["per_K"] = {b: summarize([r for r in recs if r["_kbin"] == b])
                      for b in ["2", "3", "4-6", "7+"]}
        m["per_subset"] = {s: summarize([r for r in recs if r["_subset"] == s])
                           for s in ["Entity", "Event", "Mixed"]}
        m["judge_version"] = "gemini-3-flash-preview@" + (
            recs[0].get("protocol_version", "?") if recs else "?")
        out[tag] = m

    dest = REPO_ROOT / "analysis" / "open_weight_iaa_metrics.json"
    dest.parent.mkdir(exist_ok=True)
    dest.write_text(json.dumps(out, indent=2))
    print(f"[aggregate] wrote {dest}\n")

    pct = lambda x: round(100 * x, 1)  # noqa: E731
    print("=== Table 1 rows (Score / strict-K / Recog / Clar / F/T) ===")
    for tag, m in out.items():
        print(f"{tag}: {pct(m['iaa'])} & {pct(m['strict_K'])} & "
              f"{pct(m['recognition'])} & {pct(m['clarification_rate'])} & "
              f"{pct(m['follow_through'])}   (n={m['n']}, err={m['n_errored']})")
    print("\n=== fig_perk SCORE lines (ReQueST score by K bin) ===")
    for tag, m in out.items():
        vals = [pct(m["per_K"][b]["iaa"]) for b in ["2", "3", "4-6", "7+"]]
        print(f'    "{tag}": {vals},')
    print("\n=== fig_subsets rows (entity strict-K, event strict-K) ===")
    for tag, m in out.items():
        e = pct(m["per_subset"]["Entity"]["strict_K"])
        v = pct(m["per_subset"]["Event"]["strict_K"])
        print(f'    ("{tag}", {e}, {v}),')


if __name__ == "__main__":
    main()
