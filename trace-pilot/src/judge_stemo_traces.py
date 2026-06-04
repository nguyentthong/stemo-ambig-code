"""Gemini-judge eval for Qwen3.5-27B traces on STEMO-Ambig.

For each trace in `outputs_stemo/stemo_ambig_traces.jsonl`:
  - Look up the K gold interpretations from `all_questions.json`.
  - If `final_answer` is empty, mark `truncated=True` and skip the judge call
    (the model burned its budget inside <think> and never finalized).
  - Otherwise, ask Gemini to map the model's final answer into structured
    per-interpretation labels: was each interpretation addressed, and if so
    what yes/no answer did the model give for it?

Writes:
  - outputs_stemo/stemo_ambig_judgments.jsonl  (one record per trace)
  - outputs_stemo/stemo_ambig_metrics.json     (aggregate metrics)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from stemo_ambig import GEMINI_MODEL  # noqa: E402
from stemo_ambig.llm import get_client, parse_json  # noqa: E402
from google.genai import types  # noqa: E402


DEFAULT_TRACES = REPO_ROOT / "trace-pilot" / "outputs_stemo" / "stemo_ambig_traces.jsonl"
DEFAULT_GOLD = REPO_ROOT / "data_v0" / "stemo_ambig_candidates" / "all_questions.json"
DEFAULT_JUDGMENTS = REPO_ROOT / "trace-pilot" / "outputs_stemo" / "stemo_ambig_judgments.jsonl"
DEFAULT_METRICS = REPO_ROOT / "trace-pilot" / "outputs_stemo" / "stemo_ambig_metrics.json"


JUDGE_PROMPT = """You are evaluating a VLM's answer to an ambiguous video question.

The question has {n_interp} distinct valid interpretations. For each interpretation, there is a gold yes/no answer.

Your job is to look at the VLM's final response and decide, for each interpretation:
  1. Did the VLM address this interpretation (explicitly mention/distinguish this reading)?
  2. If addressed, what yes/no answer did the VLM give for it?

Also decide whether the VLM enumerated multiple interpretations at all, or just committed to a single yes/no without acknowledging ambiguity.

--- SURFACE QUESTION ---
{question}

--- GOLD INTERPRETATIONS ---
{interpretations_block}

--- VLM FINAL ANSWER ---
{vlm_answer}

--- INSTRUCTIONS ---
Return strict JSON with this exact shape:
{{
  "enumerated": <true|false>,                  // did the VLM explicitly distinguish 2+ interpretations?
  "single_commit": <true|false>,               // did the VLM give one bare yes/no with no acknowledgment of ambiguity?
  "per_interp": [
    {{
      "interp_id": "<id from gold>",
      "addressed": <true|false>,
      "model_answer": "<yes|no|abstain|unclear>"   // 'abstain' if VLM said it couldn't tell for this reading; 'unclear' if the VLM addressed it but you can't parse a yes/no
    }},
    ...
  ],
  "notes": "<one short sentence, optional>"
}}

Definitions:
- "addressed" is True if the VLM's response refers to the specific referent (e.g., "the man in the blue shirt") or otherwise distinguishes this reading from others. Mentioning an interpretation only to dismiss it still counts as addressed; pick the resulting yes/no the VLM landed on for that interpretation (use 'abstain' if it explicitly refused).
- A bare "Yes" or "No" with no referent-specific elaboration is NOT addressing any specific interpretation individually unless the VLM also stated which reading it was answering.
- "enumerated" requires the VLM to surface 2+ readings in its output. A single yes/no is not enumerated.
- "single_commit" is True when the VLM gives one global yes/no answer to the surface question without acknowledging that multiple readings exist.
- "enumerated" and "single_commit" should generally be opposites, but both can be False (e.g., the VLM gave a non-yes/no descriptive answer).
"""


def build_interpretations_block(gold_interpretations):
    lines = []
    for i, interp in enumerate(gold_interpretations, 1):
        lines.append(
            f"[{i}] interp_id={interp['interpretation_id']}\n"
            f"    referent: {interp['referent_description']}\n"
            f"    disambiguated question: {interp['disambiguated_question']}\n"
            f"    gold answer: {interp['predicted_answer']}"
        )
    return "\n\n".join(lines)


def judge_one(client, trace, gold_q):
    """Call Gemini to label one trace. Returns a dict ready to be JSONL'd."""
    interp_ids = [i["interpretation_id"] for i in gold_q["interpretations"]]
    gold_map = {i["interpretation_id"]: i["predicted_answer"].strip().lower()
                for i in gold_q["interpretations"]}

    base = {
        "id": trace["id"],
        "video_id": trace["video_id"],
        "question": trace["question"],
        "k_group": trace["k_group"],
        "subcategory": trace["subcategory"],
        "n_interpretations_total": len(interp_ids),
        "answer_changes_across_interpretations": gold_q.get(
            "answer_changes_across_interpretations"
        ),
        "thinking_char_count": trace.get("thinking_char_count"),
        "final_answer": trace.get("final_answer", ""),
        "gold": {i["interpretation_id"]: i["predicted_answer"] for i in gold_q["interpretations"]},
    }

    final_answer = (trace.get("final_answer") or "").strip()
    if not final_answer:
        base.update({
            "truncated": True,
            "enumerated": False,
            "single_commit": False,
            "per_interp": [
                {"interp_id": iid, "gold": gold_map[iid], "addressed": False,
                 "model_answer": "abstain", "match": False}
                for iid in interp_ids
            ],
            "n_addressed": 0,
            "n_matched": 0,
            "judge_raw": None,
            "judge_error": None,
        })
        return base

    prompt = JUDGE_PROMPT.format(
        n_interp=len(interp_ids),
        question=trace["question"],
        interpretations_block=build_interpretations_block(gold_q["interpretations"]),
        vlm_answer=final_answer,
    )
    cfg = types.GenerateContentConfig(
        temperature=0.0,
        response_mime_type="application/json",
    )
    try:
        resp = client.models.generate_content(
            model=GEMINI_MODEL, contents=[prompt], config=cfg,
        )
        raw = resp.text or ""
        parsed = parse_json(raw)
    except Exception as e:  # noqa: BLE001
        base.update({
            "truncated": False, "enumerated": None, "single_commit": None,
            "per_interp": [
                {"interp_id": iid, "gold": gold_map[iid], "addressed": None,
                 "model_answer": None, "match": None}
                for iid in interp_ids
            ],
            "n_addressed": 0, "n_matched": 0,
            "judge_raw": None,
            "judge_error": f"{type(e).__name__}: {e}",
        })
        return base

    # Normalize the parsed judgment. Gemini occasionally returns a bare list
    # (just per_interp) at the top level instead of the requested object.
    if isinstance(parsed, list):
        parsed = {"per_interp": parsed, "enumerated": None, "single_commit": None}
    elif not isinstance(parsed, dict):
        parsed = {}
    enumerated = bool(parsed.get("enumerated"))
    single_commit = bool(parsed.get("single_commit"))
    judge_per_interp = {p["interp_id"]: p for p in parsed.get("per_interp", [])
                        if isinstance(p, dict) and "interp_id" in p}
    per_interp = []
    for iid in interp_ids:
        j = judge_per_interp.get(iid, {})
        addressed = bool(j.get("addressed"))
        model_ans = (j.get("model_answer") or "").strip().lower() or "unclear"
        gold_ans = gold_map[iid]
        match = addressed and model_ans == gold_ans
        per_interp.append({
            "interp_id": iid,
            "gold": gold_ans,
            "addressed": addressed,
            "model_answer": model_ans,
            "match": match,
        })

    n_addressed = sum(1 for p in per_interp if p["addressed"])
    n_matched = sum(1 for p in per_interp if p["match"])
    base.update({
        "truncated": False,
        "enumerated": enumerated,
        "single_commit": single_commit,
        "per_interp": per_interp,
        "n_addressed": n_addressed,
        "n_matched": n_matched,
        "judge_notes": parsed.get("notes"),
        "judge_raw": raw,
        "judge_error": None,
    })
    return base


def load_gold(path):
    data = json.loads(Path(path).read_text())
    return {q["id"]: q for q in data["questions"]}


def load_traces(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def load_done(path):
    if not path.exists():
        return set()
    done = set()
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            done.add(json.loads(line)["id"])
        except Exception:
            pass
    return done


def run(args):
    gold = load_gold(args.gold)
    traces = load_traces(args.traces)

    missing = [t["id"] for t in traces if t["id"] not in gold]
    if missing:
        print(f"WARN: {len(missing)} traces have no matching gold; first 3: {missing[:3]}")
        traces = [t for t in traces if t["id"] in gold]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done_ids = set() if args.no_resume else load_done(out_path)
    pending = [t for t in traces if t["id"] not in done_ids]
    if args.limit:
        pending = pending[: args.limit]

    print(f"traces: total={len(traces)} done={len(done_ids)} pending={len(pending)} "
          f"workers={args.workers} model={GEMINI_MODEL}")
    if not pending:
        return out_path

    client = get_client()
    t0 = time.time()
    with out_path.open("a") as f_out:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            fut_map = {ex.submit(judge_one, client, t, gold[t["id"]]): t["id"] for t in pending}
            for i, fut in enumerate(as_completed(fut_map), 1):
                rec = fut.result()
                f_out.write(json.dumps(rec) + "\n")
                f_out.flush()
                status = "trunc" if rec.get("truncated") else (
                    "err" if rec.get("judge_error") else
                    f"{rec['n_matched']}/{rec['n_addressed']}/{rec['n_interpretations_total']}"
                )
                print(f"[{i}/{len(pending)}] {rec['id']} -> {status}")
    print(f"done in {time.time()-t0:.1f}s -> {out_path}")
    return out_path


# ---------- metrics ----------

def compute_metrics(judgments_path, metrics_path):
    rows = [json.loads(l) for l in Path(judgments_path).read_text().splitlines() if l.strip()]
    if not rows:
        print("no judgments to aggregate")
        return

    def bucket_stats(items):
        n = len(items)
        if n == 0:
            return {"n": 0}
        truncated = [r for r in items if r.get("truncated")]
        errored = [r for r in items if r.get("judge_error")]
        scorable = [r for r in items if not r.get("truncated") and not r.get("judge_error")]
        ambig = [r for r in scorable if r.get("answer_changes_across_interpretations")]

        n_addr = sum(r["n_addressed"] for r in scorable)
        n_total = sum(r["n_interpretations_total"] for r in scorable)
        n_match = sum(r["n_matched"] for r in scorable)

        def safe_div(a, b):
            return None if b == 0 else a / b

        return {
            "n": n,
            "n_truncated": len(truncated),
            "n_judge_errors": len(errored),
            "n_scorable": len(scorable),
            "truncation_rate": safe_div(len(truncated), n),
            "enumeration_rate": safe_div(
                sum(1 for r in scorable if r.get("enumerated")), len(scorable)
            ),
            "single_commit_rate": safe_div(
                sum(1 for r in scorable if r.get("single_commit")), len(scorable)
            ),
            # Single-commit hallucination = single yes/no on an item whose gold answers diverge.
            "single_commit_on_ambig_rate": safe_div(
                sum(1 for r in ambig if r.get("single_commit")), len(ambig)
            ),
            "interp_coverage": safe_div(n_addr, n_total),     # fraction of gold interps the model addressed
            "per_interp_accuracy_addressed": safe_div(n_match, n_addr),  # match-rate among addressed
            "per_interp_accuracy_overall": safe_div(n_match, n_total),   # match-rate over all gold interps
            "strict_ambig_aware_accuracy": safe_div(
                sum(1 for r in scorable if r["n_matched"] == r["n_interpretations_total"]),
                len(scorable),
            ),
        }

    overall = bucket_stats(rows)
    by_k = {}
    for k in sorted({r["k_group"] for r in rows}, key=lambda s: int(s[1:])):
        by_k[k] = bucket_stats([r for r in rows if r["k_group"] == k])
    by_sub = {}
    for sc in sorted({r["subcategory"] for r in rows}):
        by_sub[sc] = bucket_stats([r for r in rows if r["subcategory"] == sc])

    out = {"overall": overall, "by_k_group": by_k, "by_subcategory": by_sub}
    Path(metrics_path).write_text(json.dumps(out, indent=2))
    print(f"\n=== overall ({overall['n']} items) ===")
    for k, v in overall.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.3f}")
        else:
            print(f"  {k}: {v}")
    print("\nby_k_group:")
    for k, v in by_k.items():
        print(f"  {k:>4} n={v['n']:>3}  enum={v['enumeration_rate']!s:>5}  "
              f"strict={v['strict_ambig_aware_accuracy']!s:>5}  "
              f"interp_acc(addr)={v['per_interp_accuracy_addressed']!s:>5}  "
              f"trunc={v['truncation_rate']!s:>5}")
    print(f"\nwrote {metrics_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", type=Path, default=DEFAULT_TRACES)
    ap.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    ap.add_argument("--out", type=Path, default=DEFAULT_JUDGMENTS)
    ap.add_argument("--metrics-out", type=Path, default=DEFAULT_METRICS)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--no-resume", action="store_true", help="re-judge everything")
    ap.add_argument("--metrics-only", action="store_true",
                    help="skip judging, just recompute metrics from existing judgments")
    args = ap.parse_args()

    if not args.metrics_only:
        run(args)
    compute_metrics(args.out, args.metrics_out)


if __name__ == "__main__":
    main()
