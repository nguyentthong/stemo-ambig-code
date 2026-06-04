"""Reward functions for v5 GRPO RL on STEMO-Ambig.

Three reward components:
  - reward_ambig_strict_k:  n_correct/K from Gemini judge for ambig items
  - reward_unambig_no_enum: -0.5 if model enumerates an UNAMBIG control item
  - reward_length_penalty:  light length penalty above 2048 tokens

Persistent judge cache keyed on (item_id, hash(response)).
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from stemo_ambig import GEMINI_MODEL  # noqa: E402
from stemo_ambig.llm import get_client, parse_json  # noqa: E402
from google.genai import types  # noqa: E402

CACHE_PATH = REPO_ROOT / "data_v0" / "stemo_ambig_rl" / "gemini_judge_cache.jsonl"
CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)

_cache_lock = threading.Lock()
_cache: dict[str, dict] = {}
if CACHE_PATH.exists():
    for line in CACHE_PATH.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            _cache[r["key"]] = r["value"]


def _cache_key(item_id: str, response: str) -> str:
    h = hashlib.sha256(response.encode("utf-8")).hexdigest()[:24]
    return f"{item_id}::{h}"


def _cache_set(key: str, value: dict) -> None:
    with _cache_lock:
        if key in _cache:
            return
        _cache[key] = value
        with CACHE_PATH.open("a") as f:
            f.write(json.dumps({"key": key, "value": value}) + "\n")


JUDGE_PROMPT = """You are scoring a candidate response against gold answers for a video question.

Question: {question}

Gold interpretations and answers:
{interp_block}

Candidate response:
\"\"\"{response}\"\"\"

For each gold interpretation, decide:
  - addressed: did the candidate response identify this referent and state an answer?
  - correct: if addressed, does the candidate's yes/no match gold?

Return STRICT JSON:
{{"per_interp": [{{"interp_id": "...", "addressed": true|false, "correct": true|false}}, ...],
  "n_addressed": <int>, "n_correct": <int>, "enumerated": <true|false>}}
"""


def _render_interps(interps):
    return "\n".join(
        f"- interp_id={ip['interpretation_id']}: \"{ip['referent_description']}\" -> {ip['predicted_answer']}"
        for ip in interps
    )


def _judge_one(client, item, response):
    key = _cache_key(item["id"], response)
    if key in _cache:
        return _cache[key]
    prompt = JUDGE_PROMPT.format(
        question=item["prompt"],
        interp_block=_render_interps(item["interpretations"]),
        response=response,
    )
    cfg = types.GenerateContentConfig(
        temperature=0.0, response_mime_type="application/json", max_output_tokens=2048,
    )
    try:
        resp = client.models.generate_content(model=GEMINI_MODEL, contents=[prompt], config=cfg)
        parsed = parse_json(resp.text or "")
        if not isinstance(parsed, dict):
            parsed = {"n_correct": 0, "n_addressed": 0, "enumerated": False}
    except Exception:  # noqa: BLE001
        parsed = {"n_correct": 0, "n_addressed": 0, "enumerated": False, "_error": True}
    value = {
        "n_correct": int(parsed.get("n_correct") or 0),
        "n_addressed": int(parsed.get("n_addressed") or 0),
        "enumerated": bool(parsed.get("enumerated", False)),
    }
    _cache_set(key, value)
    return value


def reward_ambig_strict_k(samples, items, n_workers: int = 12):
    """For ambig items: n_correct / K. Returns list of float rewards aligned with samples."""
    client = get_client()
    rewards = [0.0] * len(samples)
    pairs = [(i, samples[i], items[i]) for i in range(len(samples)) if items[i].get("kind") == "ambig"]
    if not pairs:
        return rewards
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futs = {ex.submit(_judge_one, client, item, response): (i, item)
                for i, response, item in pairs}
        for fut in futs:
            idx, item = futs[fut]
            r = fut.result()
            K = len(item.get("interpretations", []))
            if K > 0:
                rewards[idx] = r["n_correct"] / K
    return rewards


_AMBIG_DETECTOR = None
def _detect_enumeration(text: str) -> bool:
    """Heuristic: response contains 'K valid interpretations' or '- "...' -> Yes/No' pattern."""
    import re
    if re.search(r"\b(\d+)\s+valid\s+interpretations", text, re.IGNORECASE):
        return True
    if re.search(r"^\s*-\s+\"[^\"]+\"\s*->\s*(yes|no)", text, re.IGNORECASE | re.MULTILINE):
        return True
    if re.search(r"\binterpretation\s*[0-9]*\s*[:\-]", text, re.IGNORECASE):
        return True
    return False


def reward_unambig_no_enum(samples, items):
    """For UNAMBIG control items: penalty if model enumerates."""
    rewards = [0.0] * len(samples)
    for i, (s, it) in enumerate(zip(samples, items)):
        if it.get("kind") != "unambig":
            continue
        if _detect_enumeration(s):
            rewards[i] = -0.5
    return rewards


def reward_length_penalty(samples):
    """Light length penalty: -0.1 * max(0, len-2048)/2048. Discourages padding-for-coverage."""
    rewards = []
    for s in samples:
        n_tok = max(1, len(s) // 4)  # rough char→token
        excess = max(0, n_tok - 2048)
        rewards.append(-0.1 * excess / 2048)
    return rewards


def combined_reward(samples, items):
    """Sum of three reward components."""
    r_strict = reward_ambig_strict_k(samples, items)
    r_unambig = reward_unambig_no_enum(samples, items)
    r_len = reward_length_penalty(samples)
    return [a + b + c for a, b, c in zip(r_strict, r_unambig, r_len)]


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    args = ap.parse_args()
    if args.test:
        # Smoke test on a single item from star_input
        items = [json.loads(l) for l in
                 open(REPO_ROOT / "data_v0/stemo_ambig_sft_qwen35_v4/star_input.jsonl")
                 .read().splitlines()[:2]]
        for it in items:
            it["kind"] = "ambig"
        samples = [
            "This question has 2 valid interpretations.\n- \"the first man\" -> Yes\n- \"the second man\" -> No",
            "Yes.",
        ]
        rewards = combined_reward(samples, items)
        for s, it, r in zip(samples, items, rewards):
            print(f"reward={r:.3f}  sample={s[:60]!r}")
