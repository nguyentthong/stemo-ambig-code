"""Distinctness: interpretations must differ on answer, referent, or evidence."""

from __future__ import annotations

from .schema import Candidate, Interpretation


def _spans_iou(a: list[tuple[float, float]], b: list[tuple[float, float]]) -> float:
    """Union-IoU over evidence spans, treating each list as a union of intervals."""
    if not a or not b:
        return 0.0

    def _length(spans: list[tuple[float, float]]) -> float:
        merged = _merge(spans)
        return sum(max(0.0, e - s) for s, e in merged)

    def _merge(spans: list[tuple[float, float]]) -> list[tuple[float, float]]:
        if not spans:
            return []
        ss = sorted((float(s), float(e)) for s, e in spans if e >= s)
        out: list[tuple[float, float]] = []
        for s, e in ss:
            if out and s <= out[-1][1]:
                out[-1] = (out[-1][0], max(out[-1][1], e))
            else:
                out.append((s, e))
        return out

    def _intersect(
        xs: list[tuple[float, float]], ys: list[tuple[float, float]]
    ) -> list[tuple[float, float]]:
        out = []
        i = j = 0
        xs2 = _merge(xs)
        ys2 = _merge(ys)
        while i < len(xs2) and j < len(ys2):
            s = max(xs2[i][0], ys2[j][0])
            e = min(xs2[i][1], ys2[j][1])
            if e > s:
                out.append((s, e))
            if xs2[i][1] < ys2[j][1]:
                i += 1
            else:
                j += 1
        return out

    la, lb = _length(a), _length(b)
    inter = sum(max(0.0, e - s) for s, e in _intersect(a, b))
    union = la + lb - inter
    return inter / union if union > 0 else 0.0


def _pairwise_distinct(x: Interpretation, y: Interpretation) -> bool:
    if x.predicted_answer.strip().lower() != y.predicted_answer.strip().lower():
        return True
    if x.referent_description.strip().lower() != y.referent_description.strip().lower():
        return True
    iou = _spans_iou(x.vlm_proposed_evidence_spans, y.vlm_proposed_evidence_spans)
    return iou < 0.5


def check(candidate: Candidate) -> tuple[bool, str]:
    interps = candidate.interpretations
    if len(interps) < 2:
        return False, "fewer than 2 interpretations"

    distinct: list[Interpretation] = []
    for cand in interps:
        if all(_pairwise_distinct(cand, prev) for prev in distinct):
            distinct.append(cand)
    if len(distinct) < 2:
        return False, "interpretations not distinct on answer/referent/IoU<0.5"
    return True, ""
