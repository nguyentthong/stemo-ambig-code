"""IAA sub-judges.

Two callables exported:
  classify_turn1(question, gold_interpretations, response_1) -> dict
      Classifies a Turn-1 model response into one of:
        enumerated, clarified_scope, clarified_vague, single_commit, refused
      Also extracts per-referent enumeration scoring if applicable.

  extract_yesno(question, referent_description, response) -> "yes" | "no" | "unknown"
      Extracts the model's yes/no commitment from a Turn-2/3 response
      for the disambiguated question.

Both use gemini-3-flash-preview at temperature 0 for reproducibility.
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

JUDGE_MODEL = "gemini-3-flash-preview"
PROTOCOL_VERSION = "iaa-v1.0"

CLASSIFY_PROMPT = """You are classifying a model's response to a video-QA question for a referential-ambiguity benchmark.

Question: {question}

Gold interpretations of the question (each names a distinct referent the question could pick out):
{interp_block}

Model response:
\"\"\"{response}\"\"\"

Classify the response into EXACTLY ONE of these categories:

- "enumerated": The response explicitly lists ≥2 candidate referents AND provides an answer (Yes/No or equivalent) for each. The referent descriptions should plausibly correspond to the gold interpretations.

- "clarified_scope": The response asks the asker to clarify, AND it explicitly names the ambiguous noun/phrase from the question. Examples: "Which boy do you mean?", "Which color?", "Which point in the video?". The named phrase must correspond to the part of the question that admits multiple interpretations (e.g. for "Does the boy fall?" with gold referents about boys, asking "which boy?" qualifies; asking "what kind of fall?" does not).

- "clarified_vague": The response acknowledges ambiguity or asks for clarification WITHOUT naming the specific ambiguous noun/phrase. Examples: "Which one?", "Can you be more specific?", "I am not sure which you mean.". Generic hedges qualify here, not under clarified_scope.

- "single_commit": The response gives a single Yes/No (or commits to one referent) without acknowledging the ambiguity.

- "refused": The model declines to answer, produces off-topic content, errors out, or otherwise provides no scorable answer.

Additionally extract:
- "enumerated_count": int. If category is "enumerated", how many distinct referent-answer pairs the response provides; else 0.
- "enumerated_matches": list of {{"referent_description": str, "decision": "yes"|"no"|"unknown"}}. If category is "enumerated", one entry per enumerated pair; else [].
- "ambiguous_phrase": str. If category is clarified_scope, the specific noun/phrase the model named (e.g. "boy", "color"); else "".

Return STRICT JSON only. No prose, no backticks.

{{"category": "...", "enumerated_count": 0, "enumerated_matches": [], "ambiguous_phrase": ""}}
"""

EXTRACT_PROMPT = """You are extracting a model's yes/no decision for a referentially disambiguated video question.

Original (ambiguous) question: {question}
Specific referent the asker meant: {referent_description}
Disambiguated question (effective meaning): {disambig_question}

Model response:
\"\"\"{response}\"\"\"

What is the model's yes/no decision for the disambiguated question?

Return STRICT JSON only:
{{"decision": "yes"|"no"|"unknown"}}

Rules:
- "yes" / "no" only when the model commits clearly.
- "unknown" if the model hedges, asks another clarifying question, says "cannot determine", contradicts itself, or never commits.
- A long explanation that ends in a clear yes/no commitment is fine — extract the final commitment.
"""


def _render_interps(interps):
    lines = []
    for i, ip in enumerate(interps):
        d = ip.get("referent_description", "")
        a = ip.get("predicted_answer", "")
        lines.append(f"  {i+1}. \"{d}\" → {a}")
    return "\n".join(lines)


def _get_gemini_client():
    """Lazy load. Falls back to direct REST if google-genai unavailable."""
    from google import genai
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        env = REPO_ROOT / ".env"
        if env.exists():
            for line in env.read_text().splitlines():
                if line.startswith("GEMINI_API_KEY="):
                    api_key = line.split("=", 1)[1].strip()
                    break
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY missing")
    return genai.Client(api_key=api_key)


def _call_gemini_json(prompt: str, max_tokens: int = 1024) -> dict:
    """Issue a single Gemini call expecting JSON output, return parsed dict."""
    from google.genai import types
    client = _get_gemini_client()
    cfg = types.GenerateContentConfig(
        temperature=0.0,
        max_output_tokens=max_tokens,
        response_mime_type="application/json",
    )
    resp = client.models.generate_content(
        model=JUDGE_MODEL,
        contents=[types.Content(role="user", parts=[types.Part(text=prompt)])],
        config=cfg,
    )
    text = resp.text or ""
    # Strip optional fences (defensive: response_mime_type should prevent this)
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.MULTILINE)
    return json.loads(text)


def classify_turn1(question: str, gold_interpretations: list, response_1: str) -> dict:
    """Classify a Turn-1 response. Returns a dict with category + extracted fields."""
    prompt = CLASSIFY_PROMPT.format(
        question=question,
        interp_block=_render_interps(gold_interpretations),
        response=response_1 or "",
    )
    try:
        out = _call_gemini_json(prompt, max_tokens=2048)
        cat = out.get("category", "refused")
        if cat not in {"enumerated", "clarified_scope", "clarified_vague", "single_commit", "refused"}:
            cat = "refused"
        return {
            "category": cat,
            "enumerated_count": int(out.get("enumerated_count", 0) or 0),
            "enumerated_matches": out.get("enumerated_matches", []) or [],
            "ambiguous_phrase": out.get("ambiguous_phrase", "") or "",
            "judge_error": None,
        }
    except Exception as e:
        return {
            "category": "refused",
            "enumerated_count": 0,
            "enumerated_matches": [],
            "ambiguous_phrase": "",
            "judge_error": repr(e)[:300],
        }


def extract_yesno(question: str, referent_description: str,
                  disambig_question: str, response: str) -> dict:
    """Extract yes/no commitment from a disambiguated-turn response."""
    prompt = EXTRACT_PROMPT.format(
        question=question,
        referent_description=referent_description,
        disambig_question=disambig_question,
        response=response or "",
    )
    try:
        out = _call_gemini_json(prompt, max_tokens=128)
        d = (out.get("decision", "unknown") or "unknown").strip().lower()
        if d not in {"yes", "no", "unknown"}:
            d = "unknown"
        return {"decision": d, "judge_error": None}
    except Exception as e:
        return {"decision": "unknown", "judge_error": repr(e)[:300]}


if __name__ == "__main__":
    # quick smoke test
    fake_interps = [
        {"referent_description": "the man in the blue shirt", "predicted_answer": "no"},
        {"referent_description": "the man in the red jacket", "predicted_answer": "yes"},
    ]
    q = "Does the man reach 1 point first?"
    print("=== enumerated ===")
    r = "the man in the blue shirt → No\nthe man in the red jacket → Yes"
    print(json.dumps(classify_turn1(q, fake_interps, r), indent=2))
    print("=== clarified_scope ===")
    r = "Which man do you mean — the one in blue or the one in red?"
    print(json.dumps(classify_turn1(q, fake_interps, r), indent=2))
    print("=== clarified_vague ===")
    r = "Could you be more specific about which one you mean?"
    print(json.dumps(classify_turn1(q, fake_interps, r), indent=2))
    print("=== single_commit ===")
    r = "Yes, the man reaches 1 point first."
    print(json.dumps(classify_turn1(q, fake_interps, r), indent=2))
    print("=== extract_yesno ===")
    print(json.dumps(extract_yesno(q, "the man in the blue shirt",
                                   "Does the man in the blue shirt reach 1 point first?",
                                   "No, he does not score first."), indent=2))
