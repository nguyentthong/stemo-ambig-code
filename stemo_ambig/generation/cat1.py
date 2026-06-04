"""Category 1: repeated temporal referent."""

from __future__ import annotations

from .base import PROMPT_DIR, run_category


PROMPT_PATH = PROMPT_DIR / "cat1.txt"


def generate(sub, client, cache, *, force: bool = False):
    return run_category(
        sub=sub, client=client, cache=cache,
        category="cat1", template_path=PROMPT_PATH, force=force,
    )
