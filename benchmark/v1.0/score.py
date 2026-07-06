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


def match_gold_readings(matches: list, interpretations: list) -> list:
    """Fuzzy-match enumerated referent-answer pairs against gold readings.

    Returns one dict per gold reading:
        {"addressed": bool, "correct": bool}
    addressed — some enumerated pair's description overlaps this reading.
    correct   — some overlapping pair also gives the reading's gold answer.

    Coverage counts only readings a response EXPLICITLY addresses. A grouped
    quantified statement ("every attempt after the first succeeds") extracts
    as few pairs and therefore undercounts; the scorer makes no attempt to
    expand quantifiers. This conservative choice is disclosed in the paper.
    """
    m_map = {(m.get("referent_description") or "").strip().lower():
             (m.get("decision") or "").strip().lower() for m in matches}
    out = []
    for ip in interpretations:
        gold_desc = (ip.get("referent_description") or "").strip().lower()
        gold_a = (ip.get("predicted_answer") or "").strip().lower()
        addressed = False
        correct = False
        for mk, mv in m_map.items():
            if gold_desc in mk or mk in gold_desc or (
                len(gold_desc) > 5 and len(mk) > 5 and
                (gold_desc[:8] == mk[:8] or gold_desc[-8:] == mk[-8:])
            ):
                addressed = True
                if mv == gold_a:
                    correct = True
                    break
        out.append({"addressed": addressed, "correct": correct})
    return out


def score_one(gold_item: dict, pred: dict, reuse_classification: bool = False) -> dict:
    K = len(gold_item["interpretations"])
    qid = gold_item["id"]
    selected_idx = select_referent_index(qid, K)
    selected_ref = gold_item["interpretations"][selected_idx]
    gold_ans = (selected_ref.get("predicted_answer") or "").strip().lower()

    r1 = pred.get("turn_1_response", "") or ""
    r2 = pred.get("turn_2_response")
    r3 = pred.get("turn_3_response")

    cls = pred.get("classification") if reuse_classification else None
    if not cls or cls.get("judge_error"):
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
        "readings_addressed": 0,
        "readings_addressed_correct": 0,
        "reading_coverage": 0.0,
        "classification": cls,
    }

    if cat == "enumerated":
        assignments = cls.get("reading_assignments", []) or []
        if assignments:
            # per-reading determinate assignment (explicit pairs and grouped
            # quantified statements alike), aligned to gold order by the judge
            per_reading = []
            for i, ip in enumerate(gold_item["interpretations"]):
                d = assignments[i]["decision"] if i < len(assignments) else "unanswered"
                gold_a = (ip.get("predicted_answer") or "").strip().lower()
                addressed = d in {"yes", "no", "conflict"}
                per_reading.append({"addressed": addressed,
                                    "correct": addressed and d == gold_a})
            all_ok = all(pr["correct"] for pr in per_reading)
        else:
            # fallback for legacy judge output: fuzzy-match explicit pairs
            matches = cls.get("enumerated_matches", []) or []
            per_reading = match_gold_readings(matches, gold_item["interpretations"])
            all_ok = len(matches) >= K and all(pr["correct"] for pr in per_reading)
        result["strict_K_correct"] = all_ok
        result["readings_addressed"] = sum(1 for pr in per_reading if pr["addressed"])
        result["readings_addressed_correct"] = sum(1 for pr in per_reading if pr["correct"])
        result["reading_coverage"] = result["readings_addressed"] / K
        # proportional credit: the fraction of gold readings the response
        # determinately answers correctly, i.e. the fraction of possible
        # intended users who receive a correct answer for their reading.
        # strict_K_correct (full credit) remains reported as a diagnostic.
        result["iaa_score"] = result["readings_addressed_correct"] / K
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
            # the follow-up explicitly addresses the interlocutor-named reading
            result["readings_addressed"] = 1
            result["readings_addressed_correct"] = 1 if correct else 0
            result["reading_coverage"] = 1.0 / K
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
                result["readings_addressed"] = 1
                result["readings_addressed_correct"] = 1 if correct else 0
                result["reading_coverage"] = 1.0 / K
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

    # Diagnostics decomposing IAA: reading coverage is the macro mean over items
    # of the fraction of gold readings the response explicitly addresses;
    # conditional correctness pools yes/no accuracy over the addressed readings.
    coverage = safe([r.get("reading_coverage", 0.0) for r in per_item])
    n_addr = sum(r.get("readings_addressed", 0) for r in per_item)
    n_addr_ok = sum(r.get("readings_addressed_correct", 0) for r in per_item)
    cond_correct = (n_addr_ok / n_addr) if n_addr else None

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
            "reading_coverage": safe([r.get("reading_coverage", 0.0) for r in items]),
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
            "reading_coverage": safe([r.get("reading_coverage", 0.0) for r in items]),
        }

    return {
        "iaa": iaa,
        "strict_K": strict,
        "aar_loose": aar,
        "clarification_rate": clar,
        "recognition_no_recall": rnr,
        "follow_through_rate": follow,
        "reading_coverage": coverage,
        "conditional_correctness": cond_correct,
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
    ap.add_argument("--reuse-classification", action="store_true",
                    help="reuse the classification stored in each prediction row "
                         "(same judge/protocol version) instead of re-judging live")
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
        per_item.append(score_one(gold[qid], preds[qid],
                                  reuse_classification=args.reuse_classification))

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
