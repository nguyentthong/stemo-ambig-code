"""STEMO-Ambig v1.0 reference scorer.

Public entry point: given a predictions JSONL (single- or multi-turn), produce
the canonical metrics.json.

Usage:
    python -m stemo_ambig.score predictions.jsonl --gold all_questions.json \
        --metrics-out metrics.json

The scorer wraps `iaa.sub_judge.classify_turn1` and `iaa.sub_judge.extract_yesno`
to apply the v1.0 protocol uniformly across submitter runners.

Input record schema (one JSON per line):
    {
      "id": str,                # gold item id
      "turn_1_response": str,
      "turn_2_response": str?,  # optional; required if model clarified at turn 1
      "turn_3_response": str?,  # optional; required if model clarified at turn 2
    }
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "trace-pilot" / "src"))

from iaa.sub_judge import classify_turn1, extract_yesno, PROTOCOL_VERSION  # noqa: E402


def select_referent_index(item_id: str, K: int) -> int:
    h = int(hashlib.md5(item_id.encode()).hexdigest(), 16)
    return h % K


def score_one(gold_item: dict, pred: dict) -> dict:
    K = len(gold_item["interpretations"])
    qid = gold_item["id"]
    selected_idx = select_referent_index(qid, K)
    selected_ref = gold_item["interpretations"][selected_idx]
    gold_ans = (selected_ref.get("predicted_answer") or "").strip().lower()

    r1 = pred.get("turn_1_response", "") or ""
    r2 = pred.get("turn_2_response")
    r3 = pred.get("turn_3_response")

    cls = classify_turn1(gold_item["question"], gold_item["interpretations"], r1)
    cat = cls["category"]

    result = {
        "id": qid,
        "K": K,
        "selected_idx": selected_idx,
        "category": cat,
        "strict_K_correct": False,
        "aar_loose_correct": cat in {"enumerated", "clarified_scope"},
        "iaa_score": 0.0,
        "follow_through_correct": False,
        "turn2_decision": None,
        "turn3_decision": None,
        "classification": cls,
    }

    if cat == "enumerated":
        matches = cls.get("enumerated_matches", []) or []
        m_map = {(m.get("referent_description") or "").strip().lower():
                 (m.get("decision") or "").strip().lower() for m in matches}
        gold_map = {(ip.get("referent_description") or "").strip().lower():
                    (ip.get("predicted_answer") or "").strip().lower()
                    for ip in gold_item["interpretations"]}
        all_ok = len(matches) >= K
        if all_ok:
            for gold_desc, gold_a in gold_map.items():
                found = False
                for mk, mv in m_map.items():
                    if gold_desc in mk or mk in gold_desc or (
                        len(gold_desc) > 5 and len(mk) > 5 and
                        (gold_desc[:8] == mk[:8] or gold_desc[-8:] == mk[-8:])
                    ):
                        if mv == gold_a:
                            found = True
                            break
                if not found:
                    all_ok = False
                    break
        result["strict_K_correct"] = all_ok
        result["iaa_score"] = 1.0 if all_ok else 0.0
        return result

    if cat in {"single_commit", "refused"}:
        return result

    # clarification — need turn-2 to score
    if r2:
        ext = extract_yesno(gold_item["question"], selected_ref["referent_description"],
                            selected_ref.get("disambiguated_question", ""), r2)
        d = ext["decision"]
        result["turn2_decision"] = d
        if d in {"yes", "no"}:
            correct = (d == gold_ans)
            result["follow_through_correct"] = correct
            base = 1.0 if correct else 0.0
            if cat == "clarified_vague":
                base *= 0.5
            result["iaa_score"] = base
            return result
        if r3:
            ext3 = extract_yesno(gold_item["question"], selected_ref["referent_description"],
                                  selected_ref.get("disambiguated_question", ""), r3)
            d3 = ext3["decision"]
            result["turn3_decision"] = d3
            if d3 in {"yes", "no"}:
                correct = (d3 == gold_ans)
                result["follow_through_correct"] = correct
                base = 1.0 if correct else 0.0
                if cat == "clarified_vague":
                    base *= 0.5
                result["iaa_score"] = base
    return result


def aggregate(per_item: list, gold: dict) -> dict:
    n = len(per_item)
    if n == 0:
        return {"n": 0, "iaa": 0.0}

    def safe(xs):
        return sum(xs) / len(xs) if xs else 0.0

    iaa = safe([r["iaa_score"] for r in per_item])
    strict = safe([1.0 if r["strict_K_correct"] else 0.0 for r in per_item])
    aar = safe([1.0 if r["aar_loose_correct"] else 0.0 for r in per_item])
    clar = safe([1.0 if r["category"] in {"clarified_scope", "clarified_vague"} else 0.0 for r in per_item])
    rnr = safe([1.0 if r["category"] == "clarified_vague" else 0.0 for r in per_item])
    clar_items = [r for r in per_item if r["category"] in {"clarified_scope", "clarified_vague"}]
    follow = safe([1.0 if r["follow_through_correct"] else 0.0 for r in clar_items]) if clar_items else None

    # Per-K bucket
    by_K = {}
    for r in per_item:
        K = r["K"]
        b = "2" if K == 2 else "3" if K == 3 else "4-6" if 4 <= K <= 6 else "7+"
        by_K.setdefault(b, []).append(r)

    per_k = {}
    for k, items in sorted(by_K.items()):
        per_k[k] = {
            "n": len(items),
            "iaa": safe([r["iaa_score"] for r in items]),
            "strict_K": safe([1.0 if r["strict_K_correct"] else 0.0 for r in items]),
            "aar_loose": safe([1.0 if r["aar_loose_correct"] else 0.0 for r in items]),
            "clarification_rate": safe([1.0 if r["category"] in {"clarified_scope", "clarified_vague"} else 0.0 for r in items]),
        }

    # Per-subset (Entity/Event/TempBias)
    by_sub = {}
    for r in per_item:
        sub = gold.get(r["id"], {}).get("category", "Unknown")
        by_sub.setdefault(sub, []).append(r)

    per_sub = {}
    for s, items in sorted(by_sub.items()):
        per_sub[s] = {
            "n": len(items),
            "iaa": safe([r["iaa_score"] for r in items]),
            "strict_K": safe([1.0 if r["strict_K_correct"] else 0.0 for r in items]),
            "aar_loose": safe([1.0 if r["aar_loose_correct"] else 0.0 for r in items]),
        }

    return {
        "iaa": iaa,
        "strict_K": strict,
        "aar_loose": aar,
        "clarification_rate": clar,
        "recognition_no_recall": rnr,
        "follow_through_rate": follow,
        "per_K": per_k,
        "per_subset": per_sub,
        "judge_version": f"gemini-3-flash-preview@{PROTOCOL_VERSION}",
        "n": n,
    }


def main():
    ap = argparse.ArgumentParser(description="STEMO-Ambig v1.0 scorer")
    ap.add_argument("predictions", help="predictions JSONL")
    ap.add_argument("--gold", required=True, help="gold all_questions.json")
    ap.add_argument("--metrics-out", required=True, help="output metrics.json path")
    ap.add_argument("--per-item-out", default=None, help="optional per-item scoring JSONL")
    args = ap.parse_args()

    gold = {q["id"]: q for q in json.load(open(args.gold))["questions"]}
    preds = {}
    for line in Path(args.predictions).read_text().splitlines():
        if line.strip():
            p = json.loads(line)
            preds[p["id"]] = p

    per_item = []
    n_missing = 0
    for qid in gold:
        if qid not in preds:
            n_missing += 1
            continue
        per_item.append(score_one(gold[qid], preds[qid]))

    metrics = aggregate(per_item, gold)
    metrics["n_missing"] = n_missing
    metrics["protocol_version"] = PROTOCOL_VERSION

    Path(args.metrics_out).write_text(json.dumps(metrics, indent=2))
    print(f"Wrote {args.metrics_out}: IAA={metrics['iaa']:.3f} strict-K={metrics['strict_K']:.3f}")

    if args.per_item_out:
        with open(args.per_item_out, "w") as f:
            for r in per_item:
                f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
