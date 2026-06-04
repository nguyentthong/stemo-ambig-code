"""Human seed-annotation UI for STEMO-Ambig.

Annotators write FREE-FORM question templates per video. Each annotation is one
text box; multiple templates per video are allowed. Slash-notation
("1/2/3.../10") inside a template signals downstream Gemini that the template
expands to multiple concrete questions. The annotations are SEEDS — not
emitted candidates. A downstream Gemini run consumes them as exemplars when
producing the full benchmark.

Run:
    python -m stemo_ambig.annotate.server
    # default: http://127.0.0.1:5001/

Seeds land in data/stemo_ambig_human_seeds/<video_id>.json.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path

from flask import Flask, abort, redirect, render_template, request, url_for

from ..loader import load_substrate


VIDEO_URL_TEMPLATE = (
    "https://storage.googleapis.com/video_data_bucket-19052026/"
    "stemo_videos/{video_name}"
)


def _video_url(video_name: str) -> str:
    return VIDEO_URL_TEMPLATE.format(video_name=video_name)


def _list_videos(qa_dir: Path) -> list[dict]:
    out: list[dict] = []
    for p in sorted(qa_dir.glob("*.json")):
        try:
            q = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        vname = q.get("video_name")
        if not vname:
            continue
        vid = Path(vname).stem
        out.append(
            {
                "video_id": vid,
                "video_name": vname,
                "video_url": _video_url(vname),
                "qa_file": p.name,
                "n_targets": len(q.get("questions", [])),
                "n_subs": sum(len(s) for s in q.get("sub-questions", [])),
            }
        )
    return out


def _load_seeds(seeds_dir: Path, video_id: str) -> dict | None:
    p = seeds_dir / f"{video_id}.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _save_seeds(seeds_dir: Path, video_id: str, data: dict) -> None:
    seeds_dir.mkdir(parents=True, exist_ok=True)
    (seeds_dir / f"{video_id}.json").write_text(json.dumps(data, indent=2))


def create_app(
    qa_dir: Path,
    video_dir: Path,
    seeds_dir: Path,
    annotator_name: str,
) -> Flask:
    seeds_dir.mkdir(parents=True, exist_ok=True)
    app = Flask(__name__)

    @app.route("/")
    def index():
        videos = _list_videos(qa_dir)
        for v in videos:
            existing = _load_seeds(seeds_dir, v["video_id"])
            v["n_seeds"] = len(existing["annotations"]) if existing else 0
        total_seeds = sum(v["n_seeds"] for v in videos)
        videos_with_seeds = sum(1 for v in videos if v["n_seeds"] > 0)
        return render_template(
            "index.html",
            videos=videos,
            total_videos=len(videos),
            total_seeds=total_seeds,
            videos_with_seeds=videos_with_seeds,
        )

    @app.route("/seed/<video_id>")
    def seed_page(video_id: str):
        videos = _list_videos(qa_dir)
        video = next((v for v in videos if v["video_id"] == video_id), None)
        if video is None:
            abort(404)

        # Substrate Q/A for reference. Show the page even if loading the
        # substrate fails (missing local video, malformed parallel arrays,
        # bad JSON, etc.) -- annotators can still write seeds from the video.
        qa_file = qa_dir / video["qa_file"]
        substrate = None
        substrate_error = None
        try:
            substrate = load_substrate(qa_file, video_dir)
        except (FileNotFoundError, ValueError, json.JSONDecodeError) as e:
            substrate_error = f"{type(e).__name__}: {e}"

        existing = _load_seeds(seeds_dir, video_id) or {
            "video_id": video_id,
            "video_url": video["video_url"],
            "annotator": annotator_name,
            "annotations": [],
        }

        # Adjacent navigation
        ids = [v["video_id"] for v in videos]
        idx = ids.index(video_id)
        prev_id = ids[idx - 1] if idx > 0 else None
        next_id = ids[idx + 1] if idx + 1 < len(ids) else None

        return render_template(
            "seed.html",
            video=video,
            substrate=substrate,
            substrate_error=substrate_error,
            seeds=existing,
            annotator_name=annotator_name,
            prev_id=prev_id,
            next_id=next_id,
            idx=idx,
            n_videos=len(ids),
        )

    @app.route("/seed/<video_id>", methods=["POST"])
    def save_page(video_id: str):
        videos = _list_videos(qa_dir)
        video = next((v for v in videos if v["video_id"] == video_id), None)
        if video is None:
            abort(404)

        form = request.form
        annotations: list[dict] = []
        idx = 0
        while True:
            text = form.get(f"text_{idx}")
            if text is None:
                break
            text = text.strip()
            note = form.get(f"note_{idx}", "").strip()
            if text:
                annotations.append({"id": len(annotations), "text": text, "note": note})
            idx += 1

        data = {
            "video_id": video_id,
            "video_url": video["video_url"],
            "last_modified": dt.datetime.now(dt.timezone.utc).isoformat(),
            "annotator": (form.get("annotator") or "").strip() or annotator_name,
            "annotations": annotations,
        }
        _save_seeds(seeds_dir, video_id, data)

        action = form.get("action", "save_stay")
        if action == "save_next":
            ids = [v["video_id"] for v in videos]
            i = ids.index(video_id)
            if i + 1 < len(ids):
                return redirect(url_for("seed_page", video_id=ids[i + 1]))
            return redirect(url_for("index"))
        return redirect(url_for("seed_page", video_id=video_id))

    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--qa-dir", type=Path, default=Path("stemo/questions")
    )
    parser.add_argument(
        "--video-dir", type=Path, default=Path("stemo/videos_h264")
    )
    parser.add_argument(
        "--seeds-dir", type=Path,
        default=Path("data/stemo_ambig_human_seeds"),
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5001)
    parser.add_argument(
        "--annotator-name",
        default=os.environ.get("USER", "unknown"),
    )
    args = parser.parse_args()

    app = create_app(
        args.qa_dir.resolve(),
        args.video_dir.resolve(),
        args.seeds_dir.resolve(),
        args.annotator_name,
    )
    print(f"\n  qa-dir:    {args.qa_dir.resolve()}")
    print(f"  seeds-dir: {args.seeds_dir.resolve()}")
    print(f"\n  open http://{args.host}:{args.port}/\n")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
