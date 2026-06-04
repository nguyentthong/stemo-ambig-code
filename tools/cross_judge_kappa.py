"""Cross-judge κ: re-judge 200 items with GPT-4o, compute agreement vs Gemini judge.

Uses GPT-4o as a second judge on the same (question, model response, gold interpretations)
trio that the Gemini judge already scored. Compares per-item agreement on:
  - enumerated  (binary)
  - single_commit (binary)
  - n_matched   (integer)
"""
from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from math import sqrt
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# Load OpenAI
os.environ["OPENAI_API_KEY"] = (REPO / ".env").read_text().split("OPENAI_API_KEY=")[1].split("\n")[0].strip()
from openai import OpenAI  # noqa
oai = OpenAI()

GEMINI_JUDGE = "eval_runs/qwen35_v3/stemo_ambig_judgments.jsonl"
GEMINI_TRACES = "eval_runs/qwen35_v3/stemo_ambig_traces.jsonl"
GOLD_SRC = "data_v0/stemo_ambig_candidates/all_questions.json"
N_SAMPLE = 200
OUT = REPO / "analysis/cross_judge_gpt4o.jsonl"

JUDGE_PROMPT = """You are scoring a candidate response for a referentially ambiguous video question.

Question: {question}

Gold interpretations and answers:
{interp_block}

Candidate response:
\"\"\"{response}\"\"\"

Decide:
- enumerated: did the candidate respond with a multi-interpretation enumeration (listing several referents)?
- single_commit: did the candidate give a single yes/no answer ignoring the ambiguity?
- n_matched: how many gold interpretations the candidate correctly answered (count integer)

Return STRICT JSON:
{{"enumerated": <true|false>, "single_commit": <true|false>, "n_matched": <int>, "n_addressed": <int>}}
"""


def render_interps(interps):
    return "\n".join(
        f"- {ip['interpretation_id']}: \"{ip['referent_description']}\" -> {ip['predicted_answer']}"
        for ip in interps
    )


def judge_one(item):
    """item = dict with question, gold interpretations, response."""
    prompt = JUDGE_PROMPT.format(
        question=item["question"],
        interp_block=render_interps(item["interps"]),
        response=item["response"],
    )
    try:
        resp = oai.chat.completions.create(
            model="gpt-4o-2024-08-06",
            messages=[
                {"role": "system", "content": "You are scoring video QA candidates. Output strict JSON only."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=512,
            response_format={"type": "json_object"},
        )
        out = json.loads(resp.choices[0].message.content)
        return {
            "id": item["id"],
            "enumerated": bool(out.get("enumerated", False)),
            "single_commit": bool(out.get("single_commit", False)),
            "n_matched": int(out.get("n_matched", 0)),
            "n_addressed": int(out.get("n_addressed", 0)),
        }
    except Exception as e:
        return {"id": item["id"], "error": repr(e)[:200]}


def main():
    import random
    rng = random.Random(0)

    # Load Gemini judgments + traces + gold to build the judgment task
    gemini = {json.loads(l)["id"]: json.loads(l)
              for l in (REPO / GEMINI_JUDGE).read_text().splitlines() if l.strip()}
    traces = {json.loads(l)["id"]: json.loads(l)
              for l in (REPO / GEMINI_TRACES).read_text().splitlines() if l.strip()}
    gold = {q["id"]: q for q in json.load(open(REPO / GOLD_SRC))["questions"]}

    ids = sorted(set(gemini) & set(traces) & set(gold))
    rng.shuffle(ids)
    ids = ids[:N_SAMPLE]
    print(f"Re-judging {len(ids)} items with GPT-4o...", flush=True)

    items = [{
        "id": i,
        "question": gold[i]["question"],
        "interps": gold[i]["interpretations"],
        "response": traces[i].get("final_answer", "") or "",
    } for i in ids]

    results = []
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=12) as ex, OUT.open("w") as fout:
        futs = {ex.submit(judge_one, it): it for it in items}
        for n, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            results.append(r)
            fout.write(json.dumps(r) + "\n"); fout.flush()
            if n % 25 == 0:
                print(f"  [{n}/{len(items)}]", flush=True)

    # Compute κ vs Gemini judgments
    def kappa(field):
        common = [r for r in results if r.get("id") in gemini and "error" not in r]
        n = len(common)
        if n == 0: return None
        a = [bool(gemini[r["id"]].get(field)) for r in common]
        b = [bool(r[field]) for r in common]
        agree = sum(1 for x, y in zip(a, b) if x == y) / n
        ap = sum(a)/n; bp = sum(b)/n
        exp = ap*bp + (1-ap)*(1-bp)
        return {"n": n, "agree": agree, "kappa": (agree-exp)/(1-exp) if exp<1 else 1.0}

    def n_matched_corr():
        common = [r for r in results if r.get("id") in gemini and "error" not in r]
        if not common: return None
        xs = [gemini[r["id"]].get("n_matched",0) for r in common]
        ys = [r["n_matched"] for r in common]
        mx, my = sum(xs)/len(xs), sum(ys)/len(ys)
        cov = sum((x-mx)*(y-my) for x,y in zip(xs,ys))/len(xs)
        sdx = sqrt(sum((x-mx)**2 for x in xs)/len(xs))
        sdy = sqrt(sum((y-my)**2 for y in ys)/len(ys))
        r = cov/(sdx*sdy) if sdx*sdy>0 else 0
        exact = sum(1 for x,y in zip(xs,ys) if x==y) / len(xs)
        return {"n": len(common), "pearson_r": r, "exact_match": exact}

    summary = {
        "n_judged_by_gpt4o": len([r for r in results if "error" not in r]),
        "n_failed": len([r for r in results if "error" in r]),
        "enumerated": kappa("enumerated"),
        "single_commit": kappa("single_commit"),
        "n_matched": n_matched_corr(),
    }
    print("\n=== Cross-judge agreement (Gemini-3-Flash vs GPT-4o) ===")
    print(json.dumps(summary, indent=2))
    (REPO / "analysis/cross_judge_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
