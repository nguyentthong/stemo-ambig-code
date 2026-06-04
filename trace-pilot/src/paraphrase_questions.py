"""Augment star_kept.jsonl with Gemini-paraphrased question variants.

For each kept ambig item, generate N paraphrases of the question text. CRITICAL:
the paraphrase must preserve the ambiguous referent expression (e.g. "the boy",
"the man") so the K interpretations remain valid. Only the surrounding wording
varies.

Outputs:
  star_kept_augmented.jsonl with original + N paraphrases per item, schema
  identical to star_kept.jsonl. Each row carries the SAME think/final response
  (training target) because the underlying K=… answer set is unchanged.

Rationale: 235 unique ambig items × ~4 paraphrases ≈ 940 augmented training
items, paired with the same self-distilled CoT — broadens question-surface
diversity without re-introducing alien-CoT failure mode from v1.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from stemo_ambig import GEMINI_MODEL  # noqa: E402
from stemo_ambig.llm import get_client, parse_json  # noqa: E402
from google.genai import types  # noqa: E402


PARA_PROMPT = """You are paraphrasing a referentially ambiguous video question for data augmentation.

ORIGINAL QUESTION: "{question}"

This question is ambiguous because the phrase "{referent_phrase}" could refer to
{k} different entities/events in the video. We want to KEEP the question
ambiguous in the same way — just vary the surrounding wording.

Generate {n} paraphrases that:
  1. Preserve the exact ambiguous noun phrase "{referent_phrase}" verbatim.
  2. Keep the same yes/no question form and the same underlying meaning.
  3. Vary surface form: word order, auxiliary verbs, synonyms for non-referent
     words only.
  4. Stay natural and grammatical (something a real viewer would ask).
  5. Do NOT add visual details that disambiguate (no colors, positions, times).

Return STRICT JSON:
{{"paraphrases": ["<paraphrase 1>", "<paraphrase 2>", ...]}}
"""


def extract_referent_phrase(question, interpretations):
    """Heuristic: ambiguous referent = first <article + head noun> in the question.
    "Is the person on the stage..." -> "the person". Used for paraphrase validation
    (paraphrase must contain the head noun "person")."""
    m = re.search(r"\b(the|a|an)\s+([A-Za-z]+)", question.lower())
    if m:
        return f"{m.group(1)} {m.group(2)}"
    return ""


def head_noun_of(referent_phrase):
    parts = referent_phrase.split()
    return parts[-1] if parts else ""


def paraphrase_one(client, row, n_para, debug=False):
    question = row["prompt"]
    referent = extract_referent_phrase(question, row.get("interpretations", []))
    if not referent:
        if debug: print(f"  NO_REFERENT: {question}", flush=True)
        return None
    prompt = PARA_PROMPT.format(
        question=question,
        referent_phrase=referent,
        k=row["k"],
        n=n_para,
    )
    cfg = types.GenerateContentConfig(
        temperature=0.3, response_mime_type="application/json", max_output_tokens=1024,
    )
    last_err = None
    for attempt in range(2):
        try:
            resp = client.models.generate_content(
                model=GEMINI_MODEL, contents=[prompt], config=cfg
            )
            parsed = parse_json(resp.text or "")
            paras = parsed.get("paraphrases", []) if isinstance(parsed, dict) else []
            # Validation: paraphrase must contain the head noun (e.g. "person")
            # so the ambiguous reference is preserved. Article check is optional
            # (we accept "person" in any position as long as the noun survives).
            head = head_noun_of(referent)
            clean = []
            rejected_no_head = 0
            for p in paras:
                if not isinstance(p, str) or len(p) < 5:
                    continue
                pl = p.lower()
                if head and not re.search(rf"\b{re.escape(head)}s?\b", pl):
                    rejected_no_head += 1
                    continue
                if p.strip().lower() == question.strip().lower():
                    continue
                clean.append(p.strip())
            if debug and not clean:
                print(f"  [empty] q={question!r}  head={head!r}  raw_paras={paras}  rejected_no_head={rejected_no_head}", flush=True)
            return clean[:n_para] or None
        except Exception as e:  # noqa: BLE001
            last_err = repr(e)[:120]
            time.sleep(1.0)
    if debug:
        print(f"  [error] q={question!r}  err={last_err}", flush=True)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True,
                    help="star_kept.jsonl")
    ap.add_argument("--star-input", type=Path, required=True,
                    help="star_input.jsonl (to recover interpretations field)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-paraphrases", type=int, default=4)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    # Load star_kept and reattach interpretations from star_input
    inputs_by_id = {json.loads(l)["id"]: json.loads(l)
                    for l in args.star_input.read_text().splitlines() if l.strip()}
    kept = [json.loads(l) for l in args.input.read_text().splitlines() if l.strip()]
    for r in kept:
        ip = inputs_by_id.get(r["id"], {})
        r.setdefault("interpretations", ip.get("interpretations", []))
    print(f"loaded {len(kept)} ambig items; generating {args.n_paraphrases} paraphrases each")

    client = get_client()
    t0 = time.time()
    results = {}  # id -> [paraphrases]
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(paraphrase_one, client, r, args.n_paraphrases, args.debug): r for r in kept}
        for n, fut in enumerate(as_completed(futs), 1):
            r = futs[fut]
            paras = fut.result()
            if paras:
                results[r["id"]] = paras
            if n % 25 == 0 or n == len(futs):
                print(f"  [{n}/{len(futs)}] with paraphrases: {len(results)}", flush=True)

    # Emit augmented file: original + paraphrases as separate rows
    out_rows = []
    n_para_total = 0
    for r in kept:
        # Original
        out_rows.append({**r, "is_paraphrase": False, "orig_prompt": r["prompt"]})
        for p in results.get(r["id"], []):
            new_row = {**r, "prompt": p, "is_paraphrase": True, "orig_prompt": r["prompt"]}
            out_rows.append(new_row)
            n_para_total += 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(json.dumps(r) for r in out_rows) + "\n")
    print(f"\nwrote {len(out_rows)} rows ({len(kept)} original + {n_para_total} paraphrases) to {args.out}")
    print(f"elapsed: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
