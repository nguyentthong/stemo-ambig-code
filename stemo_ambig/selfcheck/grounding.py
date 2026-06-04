"""Grounding: substrate IDs referenced by a candidate must actually exist."""

from __future__ import annotations

from ..loader import Substrate
from .schema import Candidate


def check(candidate: Candidate, sub: Substrate) -> tuple[bool, str]:
    sqids = set(sub.sqid_to_text())
    tqids = set(sub.tqid_to_text())

    sa = candidate.substrate_anchor
    bad_sq = [x for x in sa.referenced_sub_question_ids if x not in sqids]
    bad_tq = [x for x in sa.referenced_target_question_ids if x not in tqids]
    if bad_sq or bad_tq:
        return False, f"unresolved ids: sq={bad_sq} tq={bad_tq}"

    for interp in candidate.interpretations:
        bad = [x for x in interp.supporting_sub_question_ids if x not in sqids]
        if bad:
            return False, f"interpretation {interp.interpretation_id}: unresolved sqids {bad}"

    return True, ""
