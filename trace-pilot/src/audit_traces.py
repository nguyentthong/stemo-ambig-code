"""Audit cached distillation traces for alignment with the gold scaffold.

For each (cache_key, trace) pair, look up the source item, and ask Gemini whether
the trace's per-interpretation conclusions match the gold answers.

Outputs:
  - <out-jsonl>      : per-trace verdict
  - <bad-keys-file>  : newline-separated cache_keys with verdict != aligned
                        (or empty/truncated/missing-source)
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from stemo_ambig import GEMINI_MODEL  # noqa: E402
from stemo_ambig.llm import get_client, parse_json  # noqa: E402
from google.genai import types  # noqa: E402


AMBIG_VERIFY_PROMPT = """You are auditing a reasoning trace meant to answer a video QA item with K valid interpretations.

Surface question: {question}

Interpretations and gold answers:
{interp_block}

Reasoning trace to audit:
\"\"\"{trace}\"\"\"

Decide whether the reasoning trace REACHES each interpretation's gold yes/no answer.
A trace REACHES an interpretation's gold if it either (a) explicitly assigns the gold
answer to that interpretation, or (b) reasons in a way that clearly implies the gold
answer for that reading (not just descriptive text).

Return STRICT JSON:
{{
  "per_interp": [
    {{"interpretation_id": "<id>", "reached_gold": <true|false>, "note": "<one short sentence>"}}
  ],
  "all_aligned": <true|false>,
  "confabulated_evidence": <true|false>,
  "verdict": "<aligned|partial|misaligned|unclear>"
}}
"""

UNAMBIG_VERIFY_PROMPT = """You are auditing a reasoning trace meant to answer a single-interpretation video QA item.

Surface question: {question}
Gold answer: {gold}

Reasoning trace to audit:
\"\"\"{trace}\"\"\"

Decide whether the reasoning trace REACHES the gold yes/no answer (explicit or clearly implied).

Return STRICT JSON:
{{
  "reaches_gold": <true|false>,
  "confabulated_evidence": <true|false>,
  "verdict": "<aligned|misaligned|unclear>",
  "note": "<one short sentence>"
}}
"""


def render_interp_block(interps):
    lines = []
    for i, ip in enumerate(interps, 1):
        lines.append(
            f"[{i}] interp_id={ip['interpretation_id']}\n"
            f"    referent: {ip['referent_description']}\n"
            f"    disambiguated question: {ip['disambiguated_question']}\n"
            f"    gold answer: {ip['predicted_answer']}"
        )
    return "\n\n".join(lines)


def parse_cache_key(key):
    """cache_key format: {kind}_{item_id}_{hash}"""
    # split off the trailing 16-char hash
    if "_" not in key:
        return None, None
    parts = key.rsplit("_", 1)
    head, _hash = parts
    if head.startswith("ambig_"):
        return "ambig", head[len("ambig_"):]
    if head.startswith("unambig_"):
        return "unambig", head[len("unambig_"):]
    return None, None


def load_sources(ambig_path, unambig_path):
    ambig_payload = json.loads(Path(ambig_path).read_text())
    ambig_by_id = {q["id"]: q for q in ambig_payload["questions"]}
    unambig_by_id = {}
    if unambig_path and Path(unambig_path).exists():
        for line in Path(unambig_path).read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            unambig_by_id[r["id"]] = r
    return ambig_by_id, unambig_by_id


def audit_one(client, cache_row, ambig_by_id, unambig_by_id):
    key = cache_row["cache_key"]
    trace = (cache_row.get("trace") or "").strip()
    kind, item_id = parse_cache_key(key)
    if kind is None:
        return {"cache_key": key, "verdict": "bad_key", "kind": None}
    if not trace:
        return {"cache_key": key, "verdict": "empty", "kind": kind, "item_id": item_id}
    if kind == "ambig":
        item = ambig_by_id.get(item_id)
        if item is None:
            return {"cache_key": key, "verdict": "no_source", "kind": kind, "item_id": item_id}
        prompt = AMBIG_VERIFY_PROMPT.format(
            question=item["question"],
            interp_block=render_interp_block(item["interpretations"]),
            trace=trace,
        )
    else:
        item = unambig_by_id.get(item_id)
        if item is None:
            return {"cache_key": key, "verdict": "no_source", "kind": kind, "item_id": item_id}
        prompt = UNAMBIG_VERIFY_PROMPT.format(
            question=item["question"], gold=item["gold_answer"], trace=trace,
        )
    cfg = types.GenerateContentConfig(
        temperature=0.0, response_mime_type="application/json", max_output_tokens=2048,
    )
    try:
        resp = client.models.generate_content(model=GEMINI_MODEL, contents=[prompt], config=cfg)
        parsed = parse_json(resp.text or "")
    except Exception as e:  # noqa: BLE001
        return {"cache_key": key, "verdict": "judge_error", "kind": kind, "error": repr(e)[:200]}
    verdict = (parsed.get("verdict") or "").lower()
    out = {"cache_key": key, "kind": kind, "item_id": item_id, "verdict": verdict,
           "judge_raw": parsed}
    if kind == "ambig":
        out["k"] = f"k{len(item.get('interpretations', []))}"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path,
                    default=REPO_ROOT / "data_v0" / "stemo_ambig_sft" / "distill_cache.jsonl")
    ap.add_argument("--ambig-source", type=Path,
                    default=REPO_ROOT / "data_v0" / "stemo_ambig_sft" / "ambig" / "all_questions.json")
    ap.add_argument("--unambig-source", type=Path,
                    default=REPO_ROOT / "data_v0" / "stemo_ambig_sft" / "unambig.jsonl")
    ap.add_argument("--sample", type=int, default=50,
                    help="N traces to audit (0 = audit all)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out-jsonl", type=Path,
                    default=REPO_ROOT / "data_v0" / "stemo_ambig_sft" / "alignment_audit.jsonl")
    ap.add_argument("--bad-keys-out", type=Path,
                    default=REPO_ROOT / "data_v0" / "stemo_ambig_sft" / "bad_cache_keys.txt")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cache_rows = [json.loads(l) for l in Path(args.cache).read_text().splitlines() if l.strip()]
    print(f"cache rows: {len(cache_rows)}")
    if args.sample and args.sample < len(cache_rows):
        rng = random.Random(args.seed)
        # Stratified-ish sample: split ambig / unambig roughly proportionally
        ambig = [r for r in cache_rows if r["cache_key"].startswith("ambig_")]
        unambig = [r for r in cache_rows if r["cache_key"].startswith("unambig_")]
        frac = args.sample / len(cache_rows)
        n_a = max(1, int(len(ambig) * frac))
        n_u = max(1, args.sample - n_a)
        rng.shuffle(ambig)
        rng.shuffle(unambig)
        sample = ambig[:n_a] + unambig[:n_u]
    else:
        sample = cache_rows
    print(f"auditing {len(sample)} traces with {args.workers} workers")

    ambig_by_id, unambig_by_id = load_sources(args.ambig_source, args.unambig_source)
    client = get_client()

    results = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(audit_one, client, r, ambig_by_id, unambig_by_id) for r in sample]
        for n, fut in enumerate(as_completed(futs), 1):
            results.append(fut.result())
            if n % 10 == 0:
                print(f"  {n}/{len(sample)}")
    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.out_jsonl.write_text("\n".join(json.dumps(r) for r in results) + "\n")
    print(f"wrote {args.out_jsonl} in {time.time()-t0:.1f}s")

    # Aggregate
    from collections import Counter
    verdicts = Counter(r.get("verdict") for r in results)
    print("\n=== verdicts ===")
    for v, n in verdicts.most_common():
        print(f"  {v}: {n}")
    print(f"\nalignment rate (verdict==aligned): {verdicts.get('aligned', 0) / len(results):.1%}")

    # Per-kind
    by_kind = {}
    for r in results:
        by_kind.setdefault(r.get("kind"), []).append(r.get("verdict"))
    print("\nper-kind alignment:")
    for kind, vs in by_kind.items():
        n_aligned = sum(1 for v in vs if v == "aligned")
        print(f"  {kind}: {n_aligned}/{len(vs)} = {n_aligned/max(len(vs),1):.1%}")

    # Per-K (ambig only)
    by_k = {}
    for r in results:
        if r.get("kind") != "ambig":
            continue
        by_k.setdefault(r.get("k"), []).append(r.get("verdict"))
    if by_k:
        print("\nper-K alignment (ambig):")
        for k in sorted(by_k.keys(), key=lambda s: int((s or "k0")[1:])):
            vs = by_k[k]
            n_aligned = sum(1 for v in vs if v == "aligned")
            print(f"  {k}: {n_aligned}/{len(vs)} = {n_aligned/max(len(vs),1):.1%}")

    # Bad keys for the cleanup pass
    bad = [r["cache_key"] for r in results if r.get("verdict") not in ("aligned",)]
    args.bad_keys_out.write_text("\n".join(bad) + "\n")
    print(f"\nbad keys written: {len(bad)} -> {args.bad_keys_out}")


if __name__ == "__main__":
    main()
