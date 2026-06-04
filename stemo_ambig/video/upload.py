"""Gemini Files API upload with cache-backed reuse.

The Files API stores uploads with a TTL (~48h). We cache the URI; on cache hit
we verify it's still ACTIVE before reuse, and re-upload on miss or expiry.
"""

from __future__ import annotations

import time
from pathlib import Path

from google import genai

from .cache import Cache


def get_or_upload(client: genai.Client, video_path: Path, video_id: str, cache: Cache):
    """Return a Gemini file object usable in generate_content."""
    cached = cache.get_upload(video_id)
    if cached is not None:
        _, file_name = cached
        try:
            f = client.files.get(name=file_name)
            if getattr(f, "state", None) and str(f.state).endswith("ACTIVE"):
                return f
        except Exception:
            cache.delete_upload(video_id)

    f = client.files.upload(file=str(video_path))
    # Wait until ACTIVE (videos are PROCESSING for a short window).
    deadline = time.time() + 600
    while time.time() < deadline:
        state = str(getattr(f, "state", "")).upper()
        if state.endswith("ACTIVE"):
            break
        if state.endswith("FAILED"):
            raise RuntimeError(f"Gemini upload failed for {video_id}: {f}")
        time.sleep(2)
        f = client.files.get(name=f.name)
    else:
        raise RuntimeError(f"Gemini upload did not become ACTIVE for {video_id}")

    expires_at = None
    if getattr(f, "expiration_time", None):
        try:
            expires_at = f.expiration_time.timestamp()
        except Exception:
            expires_at = None

    cache.put_upload(video_id, f.uri, f.name, expires_at)
    return f
