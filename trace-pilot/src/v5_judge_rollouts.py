"""v5 offline RL: judge ALL rollouts per item with Gemini → per-rollout reward.

Differs from star_filter.py (which keeps only the best sample): this script
returns one judged record per (item, rollout) so a downstream selector can do
reward-weighted SFT.

Reward formula (matches trace-pilot/src/rl_reward.py for consistency):
  reward = n_correct / K                       # core ambig reward
         + (-0.5 if unambig_item and enumerated else 0)   # unambig penalty
         + (-0.1 * max(0, len_tokens-2048) / 2048)        # length penalty

Length is rough char→token (chars/4).

Input:
  --input   star_input.jsonl   (gold: id, interpretations, prompt, kind)
  --predictions  raw rollouts JSONL with `raw_responses: list[str]`

Output:
  judged_rollouts.jsonl — one row per (item, rollout):
    {id, rollout_idx, response, n_correct, n_addressed, enumerated, reward, K}
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
  "enumerated": <true|false>,
  "notes": "<one short sentence>"
}}
"""


def render_interp_block(interps):
    return "\n".join(
        f"- interp_id={ip['interpretation_id']}: \"{ip['referent_description']}\" → {ip['predicted_answer']}"
        for ip in interps
    )


def split_think(raw):
    if not raw:
        return "", ""
    m = re.search(r"<think>(.*?)</think>", raw, flags=re.DOTALL)
    if m:
        return m.group(1).strip(), raw[m.end():].strip()
    if "</think>" in raw:
        idx = raw.index("</think>")
        return raw[:idx].strip(), raw[idx + len("</think>"):].strip()
    return "", raw.strip()


def judge_one(client, item, rollout_idx, response):
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
        enum = parsed.get("enumerated")
        if n_corr is None:
            n_corr = sum(1 for p in parsed.get("per_interp", []) if p.get("correct"))
        if n_addr is None:
            n_addr = sum(1 for p in parsed.get("per_interp", []) if p.get("addressed"))
        if enum is None:
            enum = bool(parsed.get("per_interp"))
        return {
            "rollout_idx": rollout_idx,
            "response": response,
            "n_correct": int(n_corr or 0),
            "n_addressed": int(n_addr or 0),
            "enumerated": bool(enum),
            "judge_error": None,
        }
    except Exception as e:
        return {
            "rollout_idx": rollout_idx,
            "response": response,
            "n_correct": -1,
            "n_addressed": -1,
            "enumerated": False,
            "judge_error": repr(e)[:200],
        }


def compute_reward(item, jr):
    """Reward = n_correct/K + unambig-penalty + length-penalty."""
    K = max(1, len(item.get("interpretations", [])))
    if jr["judge_error"] or jr["n_correct"] < 0:
        return -1.0
    r = jr["n_correct"] / K
    if item.get("kind") == "unambig" and jr["enumerated"]:
        r -= 0.5
    n_tok = max(1, len(jr["response"]) // 4)
    excess = max(0, n_tok - 2048)
    r -= 0.1 * excess / 2048
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--predictions", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    inputs_by_id = {json.loads(l)["id"]: json.loads(l)
                    for l in args.input.read_text().splitlines() if l.strip()}
    preds = [json.loads(l) for l in args.predictions.read_text().splitlines() if l.strip()]
    print(f"loaded {len(inputs_by_id)} gold items, {len(preds)} prediction rows", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    client = get_client()

    tasks = []
    for pr in preds:
        qid = pr["id"]
        item = inputs_by_id.get(qid)
        if item is None:
            continue
        responses = pr.get("raw_responses") or [pr.get("raw_response", "")]
        for idx, resp in enumerate(responses):
            if resp and len(resp.strip()) > 10:
                tasks.append((item, idx, resp))
    print(f"{len(tasks)} (item, rollout) pairs to judge", flush=True)

    t0 = time.time()
    n_done = 0
    with args.out.open("w") as fout, ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(judge_one, client, item, idx, resp): (item, idx)
                for (item, idx, resp) in tasks}
        for fut in as_completed(futs):
            item, idx = futs[fut]
            jr = fut.result()
            K = len(item.get("interpretations", []))
            jr["id"] = item["id"]
            jr["K"] = K
            jr["kind"] = item.get("kind", "ambig")
            jr["reward"] = compute_reward(item, jr)
            fout.write(json.dumps(jr) + "\n")
            fout.flush()
            n_done += 1
            if n_done % 50 == 0:
                rate = n_done / (time.time() - t0)
                eta = (len(tasks) - n_done) / max(rate, 0.01) / 60
                print(f"  [{n_done}/{len(tasks)}] rate={rate:.1f}/s eta={eta:.1f}min", flush=True)
    print(f"done. {n_done} rows -> {args.out}")


if __name__ == "__main__":
    main()
