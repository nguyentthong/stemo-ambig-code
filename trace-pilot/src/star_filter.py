"""Filter STaR samples: keep traces whose enumeration matches the gold scaffold.

Input:
  - star_input.jsonl: items with `interpretations` (gold) + `id` + `k`
  - star_predictions.jsonl: same items, each with `raw_responses` (list of N samples)

For each item, for each sample:
  1. Strip <think>...</think> if present; keep both think (for training target) + final
  2. Use Gemini judge to score the final enumeration vs gold (n_addressed correct out of K)
  3. Keep the BEST sample (highest n_correct, tie-break by addressing more interps,
     tie-break by shorter total length)
  4. If best sample doesn't address >=2 interps correctly, drop the item

Output: star_kept.jsonl — items with the chosen sample's full response (CoT preserved).
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


JUDGE_PROMPT = """You are scoring a candidate response against gold answers for a referentially ambiguous video question.

Question: {question}

Gold interpretations and answers:
{interp_block}

Candidate response (the model's attempt):
\"\"\"{response}\"\"\"

For each gold interpretation, decide:
  - addressed: did the candidate response identify this referent and state an answer for it?
  - correct: if addressed, does the candidate's yes/no match gold?

Return STRICT JSON:
{{
  "per_interp": [
    {{"interp_id": "<id from gold>", "addressed": <true|false>, "model_answer": "<yes|no|unclear|abstain>", "correct": <true|false>}},
    ...
  ],
  "n_addressed": <int>,
  "n_correct": <int>,
  "notes": "<one short sentence>"
}}
"""


def render_interp_block(interps):
    lines = []
    for ip in interps:
        lines.append(
            f"- interp_id={ip['interpretation_id']}: \"{ip['referent_description']}\" → {ip['predicted_answer']}"
        )
    return "\n".join(lines)


def split_think(raw):
    """Return (think, final) — final is post-</think> if present, else whole."""
    if not raw:
        return "", ""
    m = re.search(r"<think>(.*?)</think>", raw, flags=re.DOTALL)
    if m:
        return m.group(1).strip(), raw[m.end():].strip()
    # If only </think> without open (some chat templates), treat the prefix as think
    if "</think>" in raw:
        idx = raw.index("</think>")
        return raw[:idx].strip(), raw[idx + len("</think>"):].strip()
    return "", raw.strip()


def judge_sample(client, item, sample_idx, response):
    final = split_think(response)[1] or response
    prompt = JUDGE_PROMPT.format(
        question=item["prompt"],
        interp_block=render_interp_block(item["interpretations"]),
        response=final,
    )
    cfg = types.GenerateContentConfig(
        temperature=0.0, response_mime_type="application/json", max_output_tokens=2048,
    )
    try:
        resp = client.models.generate_content(model=GEMINI_MODEL, contents=[prompt], config=cfg)
        parsed = parse_json(resp.text or "")
        if isinstance(parsed, list):
            parsed = {"per_interp": parsed}
        n_corr = parsed.get("n_correct")
        n_addr = parsed.get("n_addressed")
        if n_corr is None:
            n_corr = sum(1 for p in parsed.get("per_interp", []) if p.get("correct"))
        if n_addr is None:
            n_addr = sum(1 for p in parsed.get("per_interp", []) if p.get("addressed"))
        return sample_idx, int(n_corr or 0), int(n_addr or 0)
    except Exception as e:  # noqa: BLE001
        return sample_idx, -1, -1


def process_item(client, item, samples, min_correct=2, strict_full_k=False):
    """Judge each sample, return best (sample_idx, n_correct, n_addressed) or None.

    strict_full_k: require best sample to have n_correct == K (i.e. all gold
    interpretations correctly enumerated).
    """
    judgments = []
    for i, s in enumerate(samples):
        if not s or len(s) < 10:
            continue
        idx, n_corr, n_addr = judge_sample(client, item, i, s)
        if n_corr < 0:
            continue
        judgments.append((idx, n_corr, n_addr, len(s)))
    if not judgments:
        return None
    judgments.sort(key=lambda x: (-x[1], -x[2], x[3]))
    best_idx, best_corr, best_addr, best_len = judgments[0]
    K = len(item.get("interpretations", []))
    threshold = K if strict_full_k else min_correct
    if best_corr < threshold:
        return None
    return best_idx, best_corr, best_addr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True,
                    help="star input jsonl (with interpretations)")
    ap.add_argument("--predictions", type=Path, required=True,
                    help="run_qwen_video.py output with raw_responses (list)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--min-correct", type=int, default=2,
                    help="minimum correctly-answered interps to keep (ignored if --strict-full-k)")
    ap.add_argument("--strict-full-k", action="store_true",
                    help="require best sample to enumerate ALL K interpretations correctly")
    args = ap.parse_args()

    inputs_by_id = {json.loads(l)["id"]: json.loads(l)
                    for l in args.input.read_text().splitlines() if l.strip()}
    preds = [json.loads(l) for l in args.predictions.read_text().splitlines() if l.strip()]
    print(f"loaded {len(inputs_by_id)} inputs, {len(preds)} prediction rows")

    client = get_client()
    t0 = time.time()
    kept = []
    dropped = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {}
        for p in preds:
            item = inputs_by_id.get(p["id"])
            if item is None:
                continue
            samples = p.get("raw_responses") or ([p["raw_response"]] if p.get("raw_response") else [])
            if not samples:
                continue
            futs[ex.submit(process_item, client, item, samples, args.min_correct, args.strict_full_k)] = (p, item, samples)
        for n, fut in enumerate(as_completed(futs), 1):
            p, item, samples = futs[fut]
            result = fut.result()
            if result is None:
                dropped += 1
                continue
            best_idx, n_corr, n_addr = result
            best = samples[best_idx]
            think, final = split_think(best)
            kept.append({
                "id": item["id"],
                "video_id": item["video_id"],
                "video_path": item["video_path"],
                "prompt": item["prompt"],
                "k": item["k"],
                "k_group": item["k_group"],
                "best_sample_idx": best_idx,
                "n_correct": n_corr,
                "n_addressed": n_addr,
                "think": think,
                "final": final,
                "full_response": best,
            })
            if n % 100 == 0:
                print(f"  [{n}/{len(futs)}] kept={len(kept)} dropped={dropped}")
    print(f"\nkept {len(kept)} / {len(preds)} items (dropped {dropped}) in {time.time()-t0:.1f}s")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(json.dumps(r) for r in kept) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
