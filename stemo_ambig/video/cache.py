"""SQLite-backed cache for video uploads, generation outputs, and LLM self-checks.

Re-runs must not re-bill: every Gemini call is keyed and persisted here.
"""

from __future__ import annotations

import hashlib
import sqlite3
import time
from pathlib import Path


def sha12(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:12]


_SCHEMA = """
CREATE TABLE IF NOT EXISTS video_uploads (
    video_id    TEXT PRIMARY KEY,
    file_uri    TEXT NOT NULL,
    file_name   TEXT NOT NULL,
    uploaded_at REAL NOT NULL,
    expires_at  REAL
);
CREATE TABLE IF NOT EXISTS generations (
    cache_key     TEXT PRIMARY KEY,
    video_id      TEXT NOT NULL,
    category      TEXT NOT NULL,
    prompt_sha12  TEXT NOT NULL,
    raw_response  TEXT NOT NULL,
    created_at    REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS selfcheck_llm (
    cache_key   TEXT PRIMARY KEY,
    check_name  TEXT NOT NULL,
    input_sha12 TEXT NOT NULL,
    response    TEXT NOT NULL,
    created_at  REAL NOT NULL
);
"""


class Cache:
    def __init__(self, db_path: Path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        self.conn = sqlite3.connect(str(db_path))
        self.conn.executescript(_SCHEMA)
        self.conn.commit()

    # -- video uploads --
    def get_upload(self, video_id: str) -> tuple[str, str] | None:
        row = self.conn.execute(
            "SELECT file_uri, file_name FROM video_uploads WHERE video_id = ?",
            (video_id,),
        ).fetchone()
        return (row[0], row[1]) if row else None

    def put_upload(
        self, video_id: str, file_uri: str, file_name: str, expires_at: float | None
    ) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO video_uploads VALUES (?, ?, ?, ?, ?)",
            (video_id, file_uri, file_name, time.time(), expires_at),
        )
        self.conn.commit()

    def delete_upload(self, video_id: str) -> None:
        self.conn.execute(
            "DELETE FROM video_uploads WHERE video_id = ?", (video_id,)
        )
        self.conn.commit()

    # -- generations --
    @staticmethod
    def gen_key(video_id: str, category: str, prompt_sha: str) -> str:
        return f"{video_id}|{category}|{prompt_sha}"

    def get_generation(
        self, video_id: str, category: str, prompt_sha: str
    ) -> str | None:
        row = self.conn.execute(
            "SELECT raw_response FROM generations WHERE cache_key = ?",
            (self.gen_key(video_id, category, prompt_sha),),
        ).fetchone()
        return row[0] if row else None

    def put_generation(
        self, video_id: str, category: str, prompt_sha: str, raw_response: str
    ) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO generations VALUES (?, ?, ?, ?, ?, ?)",
            (
                self.gen_key(video_id, category, prompt_sha),
                video_id,
                category,
                prompt_sha,
                raw_response,
                time.time(),
            ),
        )
        self.conn.commit()

    # -- selfcheck llm --
    @staticmethod
    def selfcheck_key(check_name: str, input_sha: str) -> str:
        return f"{check_name}|{input_sha}"

    def get_selfcheck(self, check_name: str, input_sha: str) -> str | None:
        row = self.conn.execute(
            "SELECT response FROM selfcheck_llm WHERE cache_key = ?",
            (self.selfcheck_key(check_name, input_sha),),
        ).fetchone()
        return row[0] if row else None

    def put_selfcheck(
        self, check_name: str, input_sha: str, response: str
    ) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO selfcheck_llm VALUES (?, ?, ?, ?, ?)",
            (
                self.selfcheck_key(check_name, input_sha),
                check_name,
                input_sha,
                response,
                time.time(),
            ),
        )
        self.conn.commit()
