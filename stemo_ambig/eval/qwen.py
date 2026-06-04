"""Qwen multimodal evaluation client.

Targets a vLLM-style OpenAI-compatible endpoint (default http://localhost:8000/v1)
serving a Qwen multimodal model. For each candidate question, sends the
ambiguous question alongside the GCS-hosted video URL and records the response.

This implements the (A) "multi-answer recall" eval mode: ask the ambiguous
question, see whether the model recognizes and enumerates interpretations.
"""

from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

import click
from openai import OpenAI


VIDEO_URL_TEMPLATE = (
    "https://storage.googleapis.com/video_data_bucket-19052026/"
    "stemo_videos/{video_id}.mp4"
)

SYSTEM_PROMPT_DEFAULT = (
    "You are a careful video question-answering assistant. Watch the video "
    "and answer concisely. If the question has multiple valid interpretations, "
    "ENUMERATE EACH interpretation explicitly and provide an answer for each."
)


def _video_url(video_id: str) -> str:
    return VIDEO_URL_TEMPLATE.format(video_id=video_id)


def _list_candidates(candidates_dir: Path) -> list[dict]:
    items: list[dict] = []
    seen: set[str] = set()
    for p in sorted(candidates_dir.rglob("stemo_ambig_*.json")):
        if p.name.startswith("_"):
            continue
        if any(part.startswith("_") for part in p.relative_to(candidates_dir).parts):
            continue
        try:
            c = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        cid = c.get("candidate_id")
        if not cid or cid in seen:
            continue
        seen.add(cid)
        items.append(c)
    return items


def evaluate_one(
    client: OpenAI,
    model: str,
    candidate: dict,
    *,
    out_path: Path | None = None,
    system_prompt: str = SYSTEM_PROMPT_DEFAULT,
    max_tokens: int = 16384,
    temperature: float = 0.0,
    flush_every_chunks: int = 20,
    disable_thinking: bool = False,
) -> dict:
    """Stream chat completion. If out_path is given, flush partial state every
    `flush_every_chunks` chunks so progress survives interruption/truncation."""
    video_url = _video_url(candidate["video_id"])
    user_content = [
        {"type": "video_url", "video_url": {"url": video_url}},
        {"type": "text", "text": candidate["question"]},
    ]

    started = dt.datetime.now(dt.timezone.utc)
    state: dict = {
        "candidate_id": candidate["candidate_id"],
        "video_id": candidate["video_id"],
        "question": candidate["question"],
        "model": model,
        "endpoint": str(client.base_url),
        "system_prompt": system_prompt,
        "video_url": video_url,
        "started_at": started.isoformat(),
        "finished_at": None,
        "duration_s": None,
        "response_text": "",
        "reasoning_content": "",
        "finish_reason": None,
        "status": "streaming",
        "n_chunks_received": 0,
        "max_tokens": max_tokens,
        "usage": None,
    }

    def _flush():
        if out_path is None:
            return
        snap = dict(state)
        now = dt.datetime.now(dt.timezone.utc)
        snap["finished_at"] = now.isoformat()
        snap["duration_s"] = (now - started).total_seconds()
        out_path.write_text(json.dumps(snap, indent=2))

    text_parts: list[str] = []
    reasoning_parts: list[str] = []

    try:
        extra = {}
        if disable_thinking:
            extra["chat_template_kwargs"] = {"enable_thinking": False}
        stream = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
            stream_options={"include_usage": True},
            extra_body=extra or None,
        )
        for chunk in stream:
            state["n_chunks_received"] += 1
            if chunk.choices:
                ch = chunk.choices[0]
                # Dump delta to dict so we catch non-standard fields like
                # vLLM/Qwen3's `reasoning` (which isn't `reasoning_content`).
                delta_dict = ch.delta.model_dump() if ch.delta else {}
                c = delta_dict.get("content")
                if c:
                    text_parts.append(c)
                # Try both spellings: `reasoning` (vLLM/Qwen3) and
                # `reasoning_content` (OpenAI o-series style).
                r = delta_dict.get("reasoning") or delta_dict.get("reasoning_content")
                if r:
                    reasoning_parts.append(r)
                if ch.finish_reason:
                    state["finish_reason"] = ch.finish_reason
            if getattr(chunk, "usage", None):
                state["usage"] = chunk.usage.model_dump()
            state["response_text"] = "".join(text_parts)
            state["reasoning_content"] = "".join(reasoning_parts)
            if out_path is not None and state["n_chunks_received"] % flush_every_chunks == 0:
                _flush()

        state["status"] = "ok"
    except Exception as e:
        state["status"] = "error"
        state["error"] = f"{type(e).__name__}: {str(e)[:300]}"

    finished = dt.datetime.now(dt.timezone.utc)
    state["finished_at"] = finished.isoformat()
    state["duration_s"] = (finished - started).total_seconds()
    state["response_text"] = "".join(text_parts).strip()
    state["reasoning_content"] = "".join(reasoning_parts).strip()
    _flush()
    return state


@click.command()
@click.option(
    "--candidates-dir", type=click.Path(exists=True, path_type=Path),
    default=Path("data/stemo_ambig_candidates"),
)
@click.option(
    "--out-dir", type=click.Path(path_type=Path),
    default=Path("data/stemo_ambig_eval/qwen"),
)
@click.option("--base-url", default="http://127.0.0.1:8000/v1")
@click.option("--api-key", default="EMPTY", help="vLLM doesn't check the key by default")
@click.option("--model", default=None, help="If omitted, queries /v1/models and picks the first.")
@click.option("--force", is_flag=True, help="Re-query even if a response file already exists.")
@click.option("--limit", type=int, default=None, help="Cap number of candidates evaluated.")
@click.option("--max-tokens", type=int, default=16384,
              help="Generation budget. Qwen3 reasoning mode burns 8-12k easily.")
@click.option("--disable-thinking", is_flag=True,
              help="Pass chat_template_kwargs={enable_thinking: False} to skip <think> mode.")
def main(
    candidates_dir: Path,
    out_dir: Path,
    base_url: str,
    api_key: str,
    model: str | None,
    force: bool,
    limit: int | None,
    max_tokens: int,
    disable_thinking: bool,
) -> None:
    """Evaluate Qwen at base_url on all active candidates."""
    out_dir.mkdir(parents=True, exist_ok=True)
    client = OpenAI(base_url=base_url, api_key=api_key)

    if model is None:
        models_page = client.models.list()
        model = models_page.data[0].id
        sys.stderr.write(f"auto-selected model: {model}\n")

    candidates = _list_candidates(candidates_dir.resolve())
    if limit is not None:
        candidates = candidates[:limit]
    sys.stderr.write(f"candidates: {len(candidates)}\n")

    n_done = 0
    n_skipped = 0
    n_failed = 0
    for c in candidates:
        out_path = out_dir / f"{c['candidate_id']}.json"
        if out_path.exists() and not force:
            # Honor existing only if it has a complete response (status=ok).
            try:
                prev = json.loads(out_path.read_text())
                if prev.get("status") == "ok" and prev.get("response_text"):
                    n_skipped += 1
                    continue
            except (OSError, json.JSONDecodeError):
                pass

        result = evaluate_one(
            client, model, c, out_path=out_path,
            max_tokens=max_tokens, disable_thinking=disable_thinking,
        )
        if result.get("status") == "ok":
            n_done += 1
        else:
            n_failed += 1
        sys.stderr.write(json.dumps({
            "event": "eval_ok" if result.get("status") == "ok" else "eval_partial",
            "candidate_id": c["candidate_id"],
            "duration_s": round(result["duration_s"], 2),
            "response_chars": len(result["response_text"]),
            "reasoning_chars": len(result.get("reasoning_content") or ""),
            "finish_reason": result.get("finish_reason"),
            "status": result.get("status"),
        }) + "\n")

    sys.stderr.write(json.dumps({
        "event": "summary",
        "done": n_done, "skipped": n_skipped, "failed": n_failed,
        "total": len(candidates),
    }) + "\n")


if __name__ == "__main__":
    main()
