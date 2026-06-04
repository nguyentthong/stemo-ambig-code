"""Anchoring: the candidate question text must actually reference the entity or
event named by each ambiguity_source_id.

Uses an LLM paraphrase check, cached. The check happens AFTER grounding so the
IDs are guaranteed to resolve.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..llm import generate_text, parse_json
from ..loader import Substrate
from ..video.cache import Cache, sha12
from .schema import Candidate


PROMPT_PATH = Path(__file__).resolve().parent.parent.parent / "prompts" / "anchoring.txt"


def _anchor_items(candidate: Candidate, sub: Substrate) -> list[str]:
    sq_map = sub.sqid_to_text()
    tq_map = sub.tqid_to_text()
    items = []
    for sid in candidate.substrate_anchor.ambiguity_source_ids:
        if sid in sq_map:
            items.append(f"[{sid}] {sq_map[sid]}")
        elif sid in tq_map:
            items.append(f"[{sid}] {tq_map[sid]}")
    return items


def check(
    candidate: Candidate,
    sub: Substrate,
    client,
    cache: Cache,
) -> tuple[bool, str]:
    items = _anchor_items(candidate, sub)
    if not items:
        return False, "no anchor items resolvable for ambiguity_source_ids"

    anchor_block = "\n".join(f"  {i}. {t}" for i, t in enumerate(items))
    prompt = PROMPT_PATH.read_text().format(
        anchor_items_block=anchor_block,
        question=candidate.question,
    )
    input_key = sha12(prompt)
    cached = cache.get_selfcheck("anchoring", input_key)
    if cached is None:
        raw = generate_text(client, prompt, json_mode=True)
        cache.put_selfcheck("anchoring", input_key, raw)
    else:
        raw = cached

    try:
        obj = parse_json(raw)
    except json.JSONDecodeError as e:
        return False, f"anchoring judge returned non-JSON: {e}"

    if not obj.get("all_anchored"):
        reasons = [
            f"item {x.get('anchor_item_index')}: {x.get('reason','')}"
            for x in obj.get("per_item", [])
            if not x.get("anchored")
        ]
        return False, "; ".join(reasons) or "not anchored"
    return True, ""
