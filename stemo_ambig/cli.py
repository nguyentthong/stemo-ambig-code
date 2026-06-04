"""STEMO-Ambig candidate generation CLI."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click
from pydantic import ValidationError

from .generation import cat1 as gen_cat1
from .generation import cat2 as gen_cat2
from .generation import cat3 as gen_cat3
from .generation import from_seeds as gen_from_seeds
from .llm import get_client
from .loader import Substrate, iter_substrate_dir, load_substrate
from .selfcheck import anchoring, distinctness, grounding, naturalness
from .selfcheck.schema import Candidate
from .video.cache import Cache


CATEGORY_MODULES = {"1": gen_cat1, "2": gen_cat2, "3": gen_cat3}


def _log(rec: dict) -> None:
    sys.stderr.write(json.dumps(rec) + "\n")
    sys.stderr.flush()


def _run_one(
    sub: Substrate,
    category: str,
    client,
    cache: Cache,
    out_dir: Path,
    max_per: int,
) -> dict:
    mod = CATEGORY_MODULES[category]
    raw_cands, pver = mod.generate(sub, client, cache)
    raw_cands = raw_cands[:max_per]

    counts = {
        "raw": len(raw_cands),
        "schema_pass": 0,
        "grounding_pass": 0,
        "anchoring_pass": 0,
        "distinctness_pass": 0,
        "naturalness_pass": 0,
        "emitted": 0,
    }
    rejections: list[dict] = []
    emitted: list[Candidate] = []

    # New layout: candidates are organized by K (number of interpretations) only.
    # Emitted candidates land in data/stemo_ambig_candidates/k<N>/<candidate_id>.json.
    # Raw + rejections (per-prompt-version, not per-K) live in _runs/<source>/<pver>/.
    run_dir = out_dir / "_runs" / f"cat{category}" / pver
    run_dir.mkdir(parents=True, exist_ok=True)
    raw_path = run_dir / f"_raw_{sub.video_id}.json"
    raw_path.write_text(json.dumps({"prompt_version": pver, "candidates": raw_cands}, indent=2))

    for i, c in enumerate(raw_cands):
        # candidate_id is already set by base.py to include prompt sha
        try:
            cand = Candidate(**c)
        except ValidationError as e:
            rejections.append({
                "i": i, "stage": "schema", "reason": str(e)[:400],
                "question": c.get("question"), "candidate": c,
            })
            continue
        counts["schema_pass"] += 1

        ok, why = grounding.check(cand, sub)
        if not ok:
            rejections.append({
                "i": i, "stage": "grounding", "reason": why,
                "question": cand.question, "candidate": cand.model_dump(mode="json"),
            })
            continue
        counts["grounding_pass"] += 1

        ok, why = anchoring.check(cand, sub, client, cache)
        if not ok:
            rejections.append({
                "i": i, "stage": "anchoring", "reason": why,
                "question": cand.question, "candidate": cand.model_dump(mode="json"),
            })
            continue
        counts["anchoring_pass"] += 1

        ok, why = distinctness.check(cand)
        if not ok:
            rejections.append({
                "i": i, "stage": "distinctness", "reason": why,
                "question": cand.question, "candidate": cand.model_dump(mode="json"),
            })
            continue
        counts["distinctness_pass"] += 1

        ok, why = naturalness.check(cand, client, cache)
        if not ok:
            rejections.append({
                "i": i, "stage": "naturalness", "reason": why,
                "question": cand.question, "candidate": cand.model_dump(mode="json"),
            })
            continue
        counts["naturalness_pass"] += 1

        emitted.append(cand)
        counts["emitted"] += 1

    for cand in emitted:
        k = len(cand.interpretations)
        k_dir = out_dir / f"k{k}"
        k_dir.mkdir(parents=True, exist_ok=True)
        (k_dir / f"{cand.candidate_id}.json").write_text(
            cand.model_dump_json(indent=2)
        )
    if rejections:
        (run_dir / f"_rejections_{sub.video_id}.json").write_text(
            json.dumps(rejections, indent=2)
        )

    return {
        "video_id": sub.video_id,
        "category": category,
        "prompt_version": pver,
        "counts": counts,
        "rejection_count": len(rejections),
    }


@click.group()
def cli() -> None:
    pass


@cli.command()
@click.option("--qa-dir", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--video-dir", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--out-dir", type=click.Path(path_type=Path), required=True)
@click.option("--category", default="1", help="comma-separated subset of {1,2,3}")
@click.option("--max-per-video-per-cat", type=int, default=10)
@click.option("--video-id", default=None, help="run a single video by stem id")
@click.option("--seed", type=int, default=42)
@click.option("--dry-run", is_flag=True)
def generate(
    qa_dir: Path,
    video_dir: Path,
    out_dir: Path,
    category: str,
    max_per_video_per_cat: int,
    video_id: str | None,
    seed: int,
    dry_run: bool,
) -> None:
    cats = [c.strip() for c in category.split(",") if c.strip()]
    for c in cats:
        if c not in CATEGORY_MODULES:
            raise click.ClickException(f"category {c!r} not implemented yet")

    if video_id is not None:
        matches = list(qa_dir.glob(f"*{video_id}*.json"))
        if not matches:
            raise click.ClickException(f"no QA file matches video id {video_id!r}")
        if len(matches) > 1:
            raise click.ClickException(f"multiple QA files match {video_id!r}: {matches}")
        substrates = [load_substrate(matches[0], video_dir)]
    else:
        substrates = iter_substrate_dir(qa_dir, video_dir)

    cache = Cache(Path("data/.stemo_ambig_cache.sqlite"))

    if dry_run:
        for sub in substrates:
            _log(
                {
                    "event": "dry_run_video",
                    "video_id": sub.video_id,
                    "duration_seconds": sub.duration_seconds,
                    "n_targets": len(sub.target_questions),
                    "n_sub_questions": len(sub.sub_questions),
                    "categories": cats,
                }
            )
        return

    client = get_client()
    for sub in substrates:
        for c in cats:
            summary = _run_one(
                sub, c, client, cache, out_dir, max_per_video_per_cat
            )
            _log({"event": "video_done", **summary})


def _print_candidate(c: dict, *, prefix: str = "") -> None:
    sa = c.get("substrate_anchor", {})
    print(f"{prefix}Q: {c.get('question')}")
    print(f"{prefix}  subtype: {c.get('ambiguity_subtype')}")
    print(f"{prefix}  anchored to (ambiguity_source_ids): {sa.get('ambiguity_source_ids')}")
    print(f"{prefix}  rationale: {sa.get('anchor_rationale')}")
    for interp in c.get("interpretations", []):
        spans = interp.get("vlm_proposed_evidence_spans") or []
        print(
            f"{prefix}  [{interp.get('interpretation_id')}] "
            f"answer={interp.get('predicted_answer')} "
            f"conf={interp.get('vlm_confidence')} "
            f"spans={spans}"
        )
        print(f"{prefix}    ref:    {interp.get('referent_description')}")
        print(f"{prefix}    rewrite: {interp.get('disambiguated_question')}")


@cli.command()
@click.argument("path", type=click.Path(exists=True, path_type=Path))
def inspect(path: Path) -> None:
    """Pretty-print a candidate, _raw_*, or _rejections_* JSON file."""
    obj = json.loads(path.read_text())
    if isinstance(obj, dict) and "candidates" in obj:
        print(f"# {path.name}  (prompt_version={obj.get('prompt_version')})")
        for i, c in enumerate(obj["candidates"]):
            print(f"\n--- candidate {i} ---")
            _print_candidate(c, prefix="  ")
    elif isinstance(obj, list):
        print(f"# {path.name}  (n={len(obj)} rejections)")
        for r in obj:
            print(f"\n--- i={r.get('i')} stage={r.get('stage')} ---")
            print(f"  reason: {r.get('reason')}")
            if r.get("candidate"):
                _print_candidate(r["candidate"], prefix="  ")
    else:
        print(f"# {path.name}  (single candidate)")
        _print_candidate(obj, prefix="  ")


@cli.command("generate-from-seeds")
@click.option("--video-dir", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--out-dir", type=click.Path(path_type=Path), required=True)
@click.option("--seeds-dir", type=click.Path(exists=True, path_type=Path),
              default=Path("data/stemo_ambig_human_seeds"))
@click.option("--video-id", required=True, help="video stem to expand seeds for")
@click.option("--force", is_flag=True,
              help="Override skip-if-emitted AND bypass the generation cache (re-bills Gemini).")
@click.option("--chunk-size", type=int, default=1,
              help="Seeds per Gemini call. Default 1 = one seed per call (full K compliance, more calls).")
def generate_from_seeds(
    video_dir: Path, out_dir: Path,
    seeds_dir: Path, video_id: str, force: bool,
    chunk_size: int,
) -> None:
    """Expand human-written seeds for one video into full candidates.

    Substrate-free: only depends on the seed file and the video. Each seed is
    sent to Gemini in its own call (chunk_size=1 default) so K compliance is
    enforced per seed.
    """
    seeds_path = seeds_dir / f"{video_id}.json"
    if not seeds_path.exists():
        raise click.ClickException(f"no seeds file: {seeds_path}")
    seeds = json.loads(seeds_path.read_text())
    if not seeds.get("annotations"):
        raise click.ClickException(f"seeds file has no annotations: {seeds_path}")

    # Skip-if-emitted: if this video already has any seed-driven candidates
    # on disk, don't re-run. Override with --force.
    existing = list(out_dir.glob(f"k*/stemo_ambig_{video_id}_seed_*.json"))
    if existing and not force:
        _log({
            "event": "skip_already_emitted",
            "video_id": video_id,
            "existing_count": len(existing),
        })
        return

    # Resolve the video file directly -- no QA / substrate dependency.
    video_path = video_dir / f"{video_id}.mp4"
    if not video_path.exists():
        raise click.ClickException(f"video not found: {video_path}")

    # Cheap ffprobe for duration; tolerate failure.
    from .loader import _probe_duration_seconds  # noqa: PLC0415
    duration = _probe_duration_seconds(video_path)

    vc = gen_from_seeds.VideoCtx(
        video_id=video_id, video_path=video_path, duration_seconds=duration,
    )

    cache = Cache(Path("data/.stemo_ambig_cache.sqlite"))
    client = get_client()
    raw_cands, pver = gen_from_seeds.generate(
        vc, seeds, client, cache, force=force, chunk_size=chunk_size,
    )

    counts = {"raw": len(raw_cands), "schema_pass": 0, "emitted": 0}
    rejections: list[dict] = []
    emitted: list[Candidate] = []

    run_dir = out_dir / "_runs" / "from_seeds" / pver
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / f"_raw_{video_id}.json").write_text(
        json.dumps({"prompt_version": pver, "candidates": raw_cands}, indent=2)
    )

    for i, c in enumerate(raw_cands):
        try:
            cand = Candidate(**c)
        except ValidationError as e:
            rejections.append({
                "i": i, "stage": "schema", "reason": str(e)[:400],
                "question": c.get("question"), "candidate": c,
            })
            continue
        counts["schema_pass"] += 1
        # Substrate-free seed-driven: trust the human seed, no further checks.
        emitted.append(cand)
        counts["emitted"] += 1

    # Before writing, archive any prior seed-driven set for this video.
    import datetime as _dt  # noqa: PLC0415
    archive_root = out_dir / "_archive" / f"prev_{video_id}_{_dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    archived_prior = 0
    for stale in out_dir.glob(f"k*/stemo_ambig_{video_id}_seed_*.json"):
        archive_root.mkdir(parents=True, exist_ok=True)
        dest = archive_root / stale.parent.name / stale.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        stale.rename(dest)
        archived_prior += 1

    for cand in emitted:
        k = len(cand.interpretations)
        k_dir = out_dir / f"k{k}"
        k_dir.mkdir(parents=True, exist_ok=True)
        (k_dir / f"{cand.candidate_id}.json").write_text(
            cand.model_dump_json(indent=2)
        )
    if rejections:
        (run_dir / f"_rejections_{video_id}.json").write_text(
            json.dumps(rejections, indent=2)
        )

    _log({
        "event": "from_seeds_done",
        "video_id": video_id,
        "prompt_version": pver,
        "counts": counts,
        "rejection_count": len(rejections),
        "archived_prior": archived_prior,
    })


if __name__ == "__main__":
    cli()
