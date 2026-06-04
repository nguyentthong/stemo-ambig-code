"""Shared generation plumbing. Each category is a one-prompt-per-call instance."""

from __future__ import annotations

from pathlib import Path

from .. import GEMINI_MODEL
from ..llm import generate_with_video, parse_json
from ..loader import Substrate
from ..video.cache import Cache, sha12
from ..video.upload import get_or_upload


PROMPT_DIR = Path(__file__).resolve().parent.parent.parent / "prompts"


def render_substrate(sub: Substrate) -> tuple[str, str]:
    target_lines = []
    for tq in sub.target_questions:
        target_lines.append(
            f"  [{tq.tqid}] (answer: {tq.answer}) {tq.text}\n"
            f"    backed by: {', '.join(tq.sub_question_ids)}"
        )
    target_block = "\n".join(target_lines)

    sub_lines = []
    for sq in sub.sub_questions:
        sub_lines.append(
            f"  [{sq.sqid}] (target tq_{sq.target_idx:02d}, answer: {sq.answer}) {sq.text}"
        )
    sub_block = "\n".join(sub_lines)
    return target_block, sub_block


def render_prompt(template_path: Path, sub: Substrate) -> str:
    template = template_path.read_text()
    target_block, sub_block = render_substrate(sub)
    return template.format(
        video_id=sub.video_id,
        duration_seconds=(
            f"{sub.duration_seconds:.1f}" if sub.duration_seconds else "unknown"
        ),
        target_block=target_block,
        sub_block=sub_block,
    )


def prompt_version(template_path: Path) -> str:
    return sha12(template_path.read_text())


def run_category(
    *,
    sub: Substrate,
    client,
    cache: Cache,
    category: str,
    template_path: Path,
    force: bool = False,
) -> tuple[list[dict], str]:
    """Return (raw_candidate_dicts, prompt_version)."""
    prompt = render_prompt(template_path, sub)
    pver = prompt_version(template_path)

    cached = None if force else cache.get_generation(sub.video_id, category, pver)
    if cached is None:
        file = get_or_upload(client, sub.video_path, sub.video_id, cache)
        raw = generate_with_video(client, file, prompt, json_mode=True)
        cache.put_generation(sub.video_id, category, pver, raw)
    else:
        raw = cached

    obj = parse_json(raw)
    if isinstance(obj, list):
        cands = obj
    elif isinstance(obj, dict):
        cands = obj.get("candidates", [])
    else:
        cands = []

    for i, c in enumerate(cands):
        c["generator"] = GEMINI_MODEL
        c["generator_prompt_version"] = pver
        # Force-override any Gemini-supplied candidate_id: must include the
        # prompt sha so versions never collide on disk.
        c["candidate_id"] = (
            f"stemo_ambig_{sub.video_id}_{category}_{pver}_{i:03d}"
        )
    return cands, pver
