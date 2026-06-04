"""Thin Gemini SDK wrapper. Pinned model: gemini-3-flash-preview."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from google import genai
from google.genai import types

from . import GEMINI_MODEL


def _load_env_file() -> None:
    """Best-effort .env loader (no python-dotenv dep)."""
    p = Path(__file__).resolve().parent.parent / ".env"
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())


def get_client() -> genai.Client:
    _load_env_file()
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError(
            "GEMINI_API_KEY not set. Put it in .env or export it."
        )
    return genai.Client(api_key=key)


def generate_with_video(
    client: genai.Client,
    file,
    prompt: str,
    *,
    json_mode: bool = True,
    temperature: float = 0.7,
    max_output_tokens: int = 65536,
) -> str:
    """Run Gemini on (video, prompt). Returns the response text.

    `max_output_tokens` default is set high so seed-driven expansions with
    many slash-templates (e.g. 86 seeds on 0077) don't get truncated.
    """
    config = types.GenerateContentConfig(
        temperature=temperature,
        response_mime_type="application/json" if json_mode else "text/plain",
        max_output_tokens=max_output_tokens,
    )
    resp = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[file, prompt],
        config=config,
    )
    return resp.text or ""


def generate_text(
    client: genai.Client,
    prompt: str,
    *,
    json_mode: bool = True,
    temperature: float = 0.2,
) -> str:
    config = types.GenerateContentConfig(
        temperature=temperature,
        response_mime_type="application/json" if json_mode else "text/plain",
    )
    resp = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[prompt],
        config=config,
    )
    return resp.text or ""


_MMSS_TIME_RE = re.compile(r"(?<=[,\[\s])(\d+):(\d{1,2})(?=[,\]\s])")


def _normalize_timestamps(s: str) -> str:
    """Rewrite `M:SS` (a video-timestamp format Gemini sometimes emits) into
    total seconds, so the result is valid JSON. We only match when bordered by
    JSON array tokens to avoid touching legitimate `"key": value` colons."""
    def repl(m):
        mins, secs = int(m.group(1)), int(m.group(2))
        return str(mins * 60 + secs)
    return _MMSS_TIME_RE.sub(repl, s)


def parse_json(text: str):
    """Lenient JSON parse for Gemini outputs:
      1. strip ``` fences and any json prefix;
      2. try strict json.loads;
      3. fall back to raw_decode (handles trailing junk);
      4. fall back to normalizing M:SS timestamps inside arrays and retrying.
    """
    s = text.strip()
    if s.startswith("```"):
        s = s.strip("`")
        if s.startswith("json"):
            s = s[4:]
        s = s.strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError as e:
        # extra data: keep first valid value, drop trailing garbage
        if "Extra data" in str(e):
            obj, _end = json.JSONDecoder().raw_decode(s)
            return obj
        # try timestamp normalization
        fixed = _normalize_timestamps(s)
        if fixed != s:
            try:
                return json.loads(fixed)
            except json.JSONDecodeError:
                # last resort: raw_decode the fixed string
                obj, _end = json.JSONDecoder().raw_decode(fixed)
                return obj
        raise
