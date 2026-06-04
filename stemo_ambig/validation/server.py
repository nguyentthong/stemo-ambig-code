"""Flask-based human-validation UI for STEMO-Ambig candidates.

Run:
    python -m stemo_ambig.validation.server \
        --candidates-dir data/stemo_ambig_candidates \
        --validations-dir data/stemo_ambig_validations \
        --video-dir stemo/videos_h264

Opens on http://127.0.0.1:5000/. The index redirects to the next unvalidated
candidate; from each review page you submit and the next unvalidated candidate
is auto-loaded.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path

from flask import Flask, abort, redirect, render_template, request, send_from_directory, url_for


def _list_candidates(candidates_dir: Path) -> list[dict]:
    """Find candidates anywhere under candidates_dir (versioned subdirs allowed).

    Any path component starting with '_' is treated as archive/hidden and
    skipped (e.g. ``_archive/``).
    """
    items: list[dict] = []
    seen_ids: set[str] = set()
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
        if not cid or cid in seen_ids:
            continue
        seen_ids.add(cid)
        items.append(c)
    items.sort(key=lambda c: c["candidate_id"])
    return items


def _validation_path(validations_dir: Path, candidate_id: str) -> Path:
    return validations_dir / f"{candidate_id}.json"


def _progress(candidates: list[dict], validations_dir: Path) -> dict:
    total = len(candidates)
    done_ids = {p.stem for p in validations_dir.glob("*.json")}
    done = sum(1 for c in candidates if c["candidate_id"] in done_ids)
    return {
        "total": total,
        "done": done,
        "remaining": total - done,
        "pct": (100.0 * done / total) if total else 0.0,
    }


def _ordered_candidate_ids(candidates: list[dict]) -> list[str]:
    return [c["candidate_id"] for c in candidates]


def create_app(
    candidates_dir: Path,
    validations_dir: Path,
    video_dir: Path,
    validator_name: str,
) -> Flask:
    validations_dir.mkdir(parents=True, exist_ok=True)
    app = Flask(__name__)
    app.config["JSON_SORT_KEYS"] = False

    def _candidates() -> list[dict]:
        return _list_candidates(candidates_dir)

    def _next_unvalidated(skip_id: str | None = None) -> str | None:
        cands = _candidates()
        done_ids = {p.stem for p in validations_dir.glob("*.json")}
        for c in cands:
            cid = c["candidate_id"]
            if cid == skip_id:
                continue
            if cid not in done_ids:
                return cid
        return None

    @app.route("/")
    def index():
        nxt = _next_unvalidated()
        if nxt is None:
            return render_template(
                "done.html",
                progress=_progress(_candidates(), validations_dir),
            )
        return redirect(url_for("review", candidate_id=nxt))

    @app.route("/candidate/<candidate_id>")
    def review(candidate_id: str):
        cands = _candidates()
        cand = next((c for c in cands if c["candidate_id"] == candidate_id), None)
        if cand is None:
            abort(404)

        ids = _ordered_candidate_ids(cands)
        idx = ids.index(candidate_id)
        prev_id = ids[idx - 1] if idx > 0 else None
        next_id = ids[idx + 1] if idx + 1 < len(ids) else None

        existing = _validation_path(validations_dir, candidate_id)
        existing_data = json.loads(existing.read_text()) if existing.exists() else None

        progress = _progress(cands, validations_dir)
        return render_template(
            "review.html",
            cand=cand,
            cand_pretty=json.dumps(cand, indent=2),
            idx=idx,
            n=len(ids),
            prev_id=prev_id,
            next_id=next_id,
            existing=existing_data,
            progress=progress,
            validator_name=validator_name,
        )

    @app.route("/validate/<candidate_id>", methods=["POST"])
    def save(candidate_id: str):
        cand = next(
            (c for c in _candidates() if c["candidate_id"] == candidate_id),
            None,
        )
        if cand is None:
            abort(404)

        form = request.form
        per_interp = {}
        for interp in cand["interpretations"]:
            iid = interp["interpretation_id"]
            per_interp[iid] = {
                "answer_correct": form.get(f"answer_correct__{iid}", "unsure"),
                "note": form.get(f"note__{iid}", "").strip(),
            }

        gold_k_raw = form.get("gold_k", "").strip()
        try:
            gold_k = int(gold_k_raw) if gold_k_raw else None
        except ValueError:
            gold_k = None

        validation = {
            "candidate_id": candidate_id,
            "validated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "validator": form.get("validator", "").strip() or "unknown",
            "is_genuinely_ambiguous": form.get("is_genuinely_ambiguous", "unsure"),
            "interpretation_set_complete": form.get("interpretation_set_complete", "unsure"),
            "gold_k": gold_k,
            "missing_interpretation_note": form.get("missing_interpretation_note", "").strip(),
            "overall_note": form.get("overall_note", "").strip(),
            "per_interpretation": per_interp,
        }
        _validation_path(validations_dir, candidate_id).write_text(
            json.dumps(validation, indent=2)
        )

        action = form.get("action", "save_and_next")
        if action == "save_and_next":
            nxt = _next_unvalidated(skip_id=candidate_id)
            if nxt is None:
                return redirect(url_for("index"))
            return redirect(url_for("review", candidate_id=nxt))
        return redirect(url_for("review", candidate_id=candidate_id))

    @app.route("/video/<video_id>.mp4")
    def serve_video(video_id: str):
        # video_id is the basename stem (no .mp4); video_dir holds <stem>.mp4
        return send_from_directory(video_dir, f"{video_id}.mp4", conditional=True)

    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidates-dir", type=Path,
        default=Path("data/stemo_ambig_candidates"),
    )
    parser.add_argument(
        "--validations-dir", type=Path,
        default=Path("data/stemo_ambig_validations"),
    )
    parser.add_argument(
        "--video-dir", type=Path,
        default=Path("stemo/videos_h264"),
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument(
        "--validator-name",
        default=os.environ.get("USER", "unknown"),
        help="Default name pre-filled in the validator field",
    )
    args = parser.parse_args()

    candidates_dir = args.candidates_dir.resolve()
    validations_dir = args.validations_dir.resolve()
    video_dir = args.video_dir.resolve()

    app = create_app(candidates_dir, validations_dir, video_dir, args.validator_name)
    print(f"\n  candidates: {candidates_dir}")
    print(f"  validations: {validations_dir}")
    print(f"  videos:      {video_dir}")
    print(f"\n  open http://{args.host}:{args.port}/\n")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
