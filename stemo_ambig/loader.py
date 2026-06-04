"""Adapter from the real STEMO QA schema to a typed substrate object.

The on-disk schema (in `stemo/questions/sample_*.json`) is positional:

    video_name:    "0016_NtTb-Cw6JVs.mp4"
    questions:     [...]          # target questions, length N
    answers:       [...]          # parallel
    sub-questions: [[...], ...]   # length N, list of lists
    sub-answers:   [[...], ...]   # parallel

There are no IDs and sub-questions are NOT globally deduplicated -- the same
text can recur across targets (e.g. "Is there a man on the right side?").
We treat that recurrence as a substrate signal of multiplicity, which Cat-1
anchoring depends on, so we DO preserve duplicates and key sub-questions by
(target_idx, sub_idx) rather than by text identity.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from pydantic import BaseModel


class SubQuestion(BaseModel):
    sqid: str
    text: str
    answer: str
    target_idx: int


class TargetQuestion(BaseModel):
    tqid: str
    text: str
    answer: str
    sub_question_ids: list[str]


class Substrate(BaseModel):
    video_id: str
    video_path: Path
    duration_seconds: float | None
    target_questions: list[TargetQuestion]
    sub_questions: list[SubQuestion]

    def sqid_to_text(self) -> dict[str, str]:
        return {sq.sqid: sq.text for sq in self.sub_questions}

    def tqid_to_text(self) -> dict[str, str]:
        return {tq.tqid: tq.text for tq in self.target_questions}


def _probe_duration_seconds(video_path: Path) -> float | None:
    try:
        out = subprocess.check_output(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(video_path),
            ],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return float(out) if out else None
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
        return None


def load_substrate(qa_json_path: Path, video_dir: Path) -> Substrate:
    raw = json.loads(qa_json_path.read_text())
    video_name = raw["video_name"]
    video_id = Path(video_name).stem

    video_path = video_dir / video_name
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found for {video_id}: {video_path}")

    questions = raw["questions"]
    answers = raw["answers"]
    nested_subs = raw["sub-questions"]
    nested_sub_answers = raw["sub-answers"]

    # Tolerate STEMO files where parallel arrays are slightly off (e.g. an
    # orphan question 19 with no matching answer). Truncate to the well-formed
    # prefix and warn.
    n = min(len(questions), len(answers), len(nested_subs), len(nested_sub_answers))
    if not (len(questions) == len(answers) == len(nested_subs) == len(nested_sub_answers)):
        import sys
        sys.stderr.write(
            f"{qa_json_path}: parallel arrays mismatched "
            f"(q={len(questions)} a={len(answers)} sq={len(nested_subs)} sa={len(nested_sub_answers)}), "
            f"truncating to {n} targets.\n"
        )
        questions = questions[:n]
        answers = answers[:n]
        nested_subs = nested_subs[:n]
        nested_sub_answers = nested_sub_answers[:n]

    targets: list[TargetQuestion] = []
    sub_pool: list[SubQuestion] = []

    for ti, (q, a, subs, sub_as) in enumerate(
        zip(questions, answers, nested_subs, nested_sub_answers)
    ):
        if len(subs) != len(sub_as):
            # Per-target mismatch: truncate to the shorter of the two arrays.
            m = min(len(subs), len(sub_as))
            import sys
            sys.stderr.write(
                f"{qa_json_path}: target {ti} sub-arrays mismatched "
                f"(subs={len(subs)} sub_as={len(sub_as)}), truncating to {m}.\n"
            )
            subs = subs[:m]
            sub_as = sub_as[:m]
        sub_ids: list[str] = []
        for si, (st, sa) in enumerate(zip(subs, sub_as)):
            sqid = f"sq_{ti:02d}_{si:02d}"
            sub_pool.append(
                SubQuestion(sqid=sqid, text=st, answer=sa, target_idx=ti)
            )
            sub_ids.append(sqid)
        targets.append(
            TargetQuestion(
                tqid=f"tq_{ti:02d}",
                text=q,
                answer=a,
                sub_question_ids=sub_ids,
            )
        )

    return Substrate(
        video_id=video_id,
        video_path=video_path,
        duration_seconds=_probe_duration_seconds(video_path),
        target_questions=targets,
        sub_questions=sub_pool,
    )


def iter_substrate_dir(qa_dir: Path, video_dir: Path) -> list[Substrate]:
    out: list[Substrate] = []
    for p in sorted(qa_dir.glob("*.json")):
        try:
            out.append(load_substrate(p, video_dir))
        except FileNotFoundError:
            continue
    return out
