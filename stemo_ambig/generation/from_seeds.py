"""Seed-driven candidate generation.

Takes a human-written seeds file plus the video; produces full STEMO-Ambig
candidates. The substrate (STEMO Q/A) is NOT used here -- the human's seeds
are the entire question source. Each seed is expanded into one or more
candidates by Gemini (slash-notation drives the expansion).

Default behavior is **per-seed**: each seed becomes one Gemini call, which
forces full K compliance because Gemini has no excuse to sample across seeds.
Use `chunk_size > 1` to batch multiple seeds per call (cheaper, less reliable).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

from .. import GEMINI_MODEL
from ..llm import generate_with_video, parse_json
from ..video.cache import Cache, sha12
from ..video.upload import get_or_upload
from .base import PROMPT_DIR


PROMPT_PATH = PROMPT_DIR / "from_seeds.txt"


@dataclass
class VideoCtx:
    video_id: str
    video_path: Path
    duration_seconds: float | None


def _render_seeds_block(seeds: dict) -> str:
    lines = []
    for a in seeds["annotations"]:
        lines.append(f"  [seed_{a['id']:02d}] {a['text']}")
        if a.get("note"):
            lines.append(f"      note: {a['note']}")
    return "\n".join(lines)


def render_prompt(vc: VideoCtx, seeds: dict) -> str:
    template = PROMPT_PATH.read_text()
    return template.format(
        video_id=vc.video_id,
        duration_seconds=(
            f"{vc.duration_seconds:.1f}" if vc.duration_seconds else "unknown"
        ),
        seeds_block=_render_seeds_block(seeds),
    )


def prompt_version(seeds: dict) -> str:
    """Sha of (template + seed contents). Edits to either invalidate cache."""
    seeds_payload = json.dumps(seeds.get("annotations", []), sort_keys=True)
    return sha12(PROMPT_PATH.read_text() + "||" + seeds_payload)


_VALID_TYPES = {
    "repeated_temporal_referent",
    "ambiguous_temporal_anchor",
    "fuzzy_event_boundary",
}


def _post_process(cands: list[dict], vc: VideoCtx, pver: str) -> list[dict]:
    """Stamp candidate_id, fix common Gemini mistakes."""
    for i, c in enumerate(cands):
        c["generator"] = GEMINI_MODEL
        c["generator_prompt_version"] = pver
        c["candidate_id"] = (
            f"stemo_ambig_{vc.video_id}_seed_{pver}_{i:03d}"
        )
        # ambiguity_type: if Gemini misfiled the subtype here, swap.
        if c.get("ambiguity_type") not in _VALID_TYPES:
            misplaced = c.get("ambiguity_type", "")
            c["ambiguity_type"] = "repeated_temporal_referent"
            if not c.get("ambiguity_subtype") and misplaced:
                c["ambiguity_subtype"] = misplaced
        # substrate_anchor is optional now; just drop it if Gemini included
        # an empty stub, leave alone if populated.
        c.setdefault("ambiguity_subtype", "shared_attribute_different_entities")
    return cands


def generate(
    vc: VideoCtx,
    seeds: dict,
    client,
    cache: Cache,
    *,
    force: bool = False,
    chunk_size: int = 1,
):
    """Expand seeds into candidates.

    Default `chunk_size=1` -> one Gemini call per seed (full K compliance).
    Set chunk_size higher to batch seeds (cheaper, less reliable for high K).
    """
    annots = seeds.get("annotations", [])
    if not annots:
        return [], ""

    # When chunk_size >= len(annots), single-shot is fine. Otherwise iterate
    # (chunk_size == 1 by default = one seed per Gemini call).
    if chunk_size >= len(annots):
        # Still show a single-tick bar so the user sees activity.
        with tqdm(total=1, desc=f"{vc.video_id} (1 call, {len(annots)} seeds)",
                  unit="call", leave=True) as bar:
            out = _generate_single(vc, seeds, client, cache, force=force)
            bar.update(1)
        return out

    all_cands = []
    chunk_pvers = []
    n_chunks = -(-len(annots) // chunk_size)  # ceil div
    if chunk_size == 1:
        desc = f"{vc.video_id} ({len(annots)} seeds)"
        unit = "seed"
    else:
        desc = f"{vc.video_id} ({n_chunks} chunks of {chunk_size}, {len(annots)} seeds)"
        unit = "chunk"
    bar = tqdm(
        range(0, len(annots), chunk_size),
        total=n_chunks, desc=desc, unit=unit, leave=True,
    )
    for ci in bar:
        chunk = annots[ci : ci + chunk_size]
        chunk_seeds = dict(seeds)
        chunk_seeds["annotations"] = chunk
        sub_cands, sub_pver = _generate_single(
            vc, chunk_seeds, client, cache, force=force
        )
        all_cands.extend(sub_cands)
        chunk_pvers.append(sub_pver)
        bar.set_postfix(emitted=len(all_cands))

    combined = sha12("|".join(chunk_pvers))
    for i, c in enumerate(all_cands):
        c["candidate_id"] = (
            f"stemo_ambig_{vc.video_id}_seed_{combined}_{i:03d}"
        )
        c["generator_prompt_version"] = combined
    return all_cands, combined


def _generate_single(vc: VideoCtx, seeds: dict, client, cache: Cache, *, force: bool):
    prompt = render_prompt(vc, seeds)
    pver = prompt_version(seeds)
    category_key = "from_seeds"

    cached = None if force else cache.get_generation(vc.video_id, category_key, pver)
    if cached is None:
        file = get_or_upload(client, vc.video_path, vc.video_id, cache)
        raw = generate_with_video(client, file, prompt, json_mode=True)
        cache.put_generation(vc.video_id, category_key, pver, raw)
    else:
        raw = cached

    obj = parse_json(raw)
    if isinstance(obj, list):
        cands = obj
    elif isinstance(obj, dict):
        cands = obj.get("candidates", [])
    else:
        cands = []

    cands = _post_process(cands, vc, pver)
    return cands, pver
