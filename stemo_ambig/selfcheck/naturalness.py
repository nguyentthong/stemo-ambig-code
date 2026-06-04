"""Naturalness 4-axis check.

Terseness is checked deterministically first (len(question) <
min(len(rewrites))). If that fails, no LLM call is made. Otherwise the LLM
judges all four axes.

Per spec deviation: the judge model is Gemini (same family as the generator),
documented in METHODOLOGY.md.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..llm import generate_text, parse_json
from ..video.cache import Cache, sha12
from .schema import Candidate


PROMPT_PATH = Path(__file__).resolve().parent.parent.parent / "prompts" / "naturalness.txt"


PARTIAL_RESOLVE_TOKENS = (
    "the first", "the earlier", "the later", "the last",
    "at the beginning", "near the beginning",
    "at the end", "near the end",
    "right after", "right before",
    "later in the video", "earlier in the video",
)


def _has_partial_resolution(question: str) -> bool:
    q = question.lower()
    return any(tok in q for tok in PARTIAL_RESOLVE_TOKENS)


def check(
    candidate: Candidate,
    client,
    cache: Cache,
) -> tuple[bool, str]:
    rewrites = [i.disambiguated_question for i in candidate.interpretations]
    if not rewrites:
        return False, "no disambiguated rewrites"

    # Deterministic gate (d): terseness.
    if len(candidate.question) >= min(len(r) for r in rewrites):
        return False, "question is not shorter than its shortest disambiguated rewrite"

    # Deterministic gate (d): partial-resolution qualifiers.
    if _has_partial_resolution(candidate.question):
        return False, "question contains partial-resolution qualifier"

    rewrites_block = "\n".join(f"  - {r}" for r in rewrites)
    prompt = PROMPT_PATH.read_text().format(
        question=candidate.question,
        rewrites_block=rewrites_block,
    )
    input_key = sha12(prompt)
    cached = cache.get_selfcheck("naturalness", input_key)
    if cached is None:
        raw = generate_text(client, prompt, json_mode=True)
        cache.put_selfcheck("naturalness", input_key, raw)
    else:
        raw = cached

    try:
        obj = parse_json(raw)
    except json.JSONDecodeError as e:
        return False, f"naturalness judge returned non-JSON: {e}"

    if not obj.get("overall_pass"):
        fails = []
        for axis in (
            "naturalness", "non_adversarial",
            "genuinely_temporal", "no_partial_resolution",
            "no_dominant_unintended_reading",
        ):
            ax = obj.get(axis) or {}
            if not ax.get("pass"):
                fails.append(f"{axis}: {ax.get('reason','')}")
        return False, "; ".join(fails) or "naturalness rubric not passed"
    return True, ""
