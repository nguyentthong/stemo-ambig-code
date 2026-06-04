"""Generate ambiguous-question candidates on a new video pool via Gemini.

Unlike the existing stemo_ambig pipeline, this works on any (video, question, answer)
source — no STEMO substrate needed. Suitable for NeXT-QA train, Charades, VATEX, etc.

Input:  --video-dir <dir of .mp4>   and/or  --pool-jsonl <file>
        pool-jsonl format (one per line):
            {"video_id": "...", "video_path": "...", "seed_question": "...", "seed_answer": "..."}
        if --pool-jsonl is omitted, we ask Gemini to invent seed questions for each video.

Output: <out-dir>/all_questions.json   (same schema as data_v0/stemo_ambig_candidates/all_questions.json)

For each video we generate K-interpretation candidates by prompting Gemini-3-flash-preview
with the video + a "find a referent that admits multiple readings" instruction. The
output schema mirrors STEMO-Ambig so downstream tooling (judge_stemo_traces.py,
make_sft_data.py) just works.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from stemo_ambig import GEMINI_MODEL  # noqa: E402
from stemo_ambig.llm import get_client, parse_json  # noqa: E402
from stemo_ambig.video.cache import Cache  # noqa: E402
from stemo_ambig.video.upload import get_or_upload  # noqa: E402
from google.genai import types  # noqa: E402


GEN_PROMPT = """You are constructing items for a benchmark on REFERENTIAL AMBIGUITY in video QA.

Watch the video. Produce up to {max_per_video} ambiguous questions about the video where:
- A surface phrase (typically "the man", "the woman", "the person", "the object", "the X")
  has MULTIPLE valid grounded referents in the video.
- Each referent yields a different gold yes/no answer to the question (i.e., the answer
  changes depending on which referent the reader picks).
- Specifically focus on temporal-referent ambiguity: the same surface phrase could refer
  to entities at different moments.

IMPORTANT: VARY K (number of interpretations per question) across the items you produce.
Stratified targets for each batch of {max_per_video} questions:
  - About 1/3 of items should have K=2 (two competing readings)
  - About 1/3 of items should have K=3, 4, or 5
  - About 1/3 of items should have K=6 or more (push higher whenever the video supports it)
For high-K items, look for videos that show many same-type entities (multiple players in a
game, many objects of the same kind, repeated similar actions over time). When the video
admits K=8 or K=12 readings, DO emit them — enumerate every distinct grounded referent,
do not stop at 2. Generating high-K items is the most valuable contribution; prioritize
them when the footage allows.

For each question, list ALL K valid interpretations the video supports. Each interpretation:
  - a short referent description (e.g., "the man in the blue shirt at 0:15")
  - the disambiguated question
  - the gold yes/no answer for that interpretation

Output STRICT JSON, no markdown fences:
{{
  "items": [
    {{
      "question": "<surface yes/no question with the ambiguous phrase>",
      "category": "repeated_temporal_referent",
      "subcategory": "<one of: shared_attribute_different_entities | repeated_action | same_entity_multiple_moments>",
      "interpretations": [
        {{
          "interpretation_id": "<short slug>",
          "referent_description": "<short phrase>",
          "disambiguated_question": "<same question rewritten with the referent made explicit>",
          "predicted_answer": "<yes|no>",
          "vlm_confidence": "high"
        }},
        ...
      ],
      "answer_changes_across_interpretations": true,
      "generation_notes": "<one sentence>"
    }},
    ...
  ]
}}

Constraints:
- K must be >= 2 (at least two distinct readings).
- predicted_answer must be 'yes' or 'no' lowercase.
- Reject any item where the surface question naturally implies one specific referent
  to a typical reader; the readings must be genuinely competing.
- Reject items resolved by spatial regions, body parts, or scene partitions (this is a
  TEMPORAL ambiguity benchmark).
"""


def _video_id_from_path(p: Path) -> str:
    return p.stem


def build_video_list(args):
    """Return list of (video_id, video_path) tuples to process."""
    items = []
    if args.pool_jsonl:
        for line in Path(args.pool_jsonl).read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            items.append((r["video_id"], Path(r["video_path"])))
    if args.video_dir:
        for p in sorted(Path(args.video_dir).glob("*.mp4")):
            vid = _video_id_from_path(p)
            if vid in {i[0] for i in items}:
                continue
            items.append((vid, p))
    if args.exclude_ids_file:
        excl = set(Path(args.exclude_ids_file).read_text().split())
        items = [(v, p) for v, p in items if v not in excl]
    if args.limit:
        items = items[: args.limit]
    return items


def gen_one(client, file, video_id, max_per_video):
    """Call Gemini on a pre-uploaded video file, return raw candidate list."""
    prompt = GEN_PROMPT.format(max_per_video=max_per_video)
    cfg = types.GenerateContentConfig(
        temperature=0.5,
        response_mime_type="application/json",
        max_output_tokens=32768,
    )
    try:
        resp = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[file, prompt],
            config=cfg,
        )
        raw = resp.text or ""
        parsed = parse_json(raw)
        items = parsed.get("items", []) if isinstance(parsed, dict) else []
        # Stamp identity fields the downstream pipeline expects.
        stamped = []
        for i, item in enumerate(items):
            interps = item.get("interpretations") or []
            if len(interps) < 2:
                continue
            qid = f"sft_{video_id}_{i:03d}"
            k = len(interps)
            stamped.append({
                "id": qid,
                "video_id": video_id,
                "question": item.get("question", "").strip(),
                "k_group": f"k{k}",
                "category": item.get("category", "repeated_temporal_referent"),
                "subcategory": item.get("subcategory", "shared_attribute_different_entities"),
                "interpretations": interps,
                "answer_changes_across_interpretations": item.get(
                    "answer_changes_across_interpretations", True
                ),
                "evidence_changes_across_interpretations": True,
                "generator": GEMINI_MODEL,
                "generator_prompt_version": "sft_v1",
                "generation_notes": item.get("generation_notes", ""),
            })
        return stamped, None
    except Exception as e:  # noqa: BLE001
        return [], f"generate_failed: {e!r}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-dir", type=Path, default=None)
    ap.add_argument("--pool-jsonl", type=Path, default=None)
    ap.add_argument("--exclude-ids-file", type=Path, default=None,
                    help="newline-separated list of video_ids to exclude (e.g., test-set videos)")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--max-per-video", type=int, default=15)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    if not (args.video_dir or args.pool_jsonl):
        sys.exit("Need --video-dir or --pool-jsonl")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    videos = build_video_list(args)
    print(f"Processing {len(videos)} videos.")

    # Resume: skip videos that already have a per-video file.
    per_video_dir = args.out_dir / "per_video"
    per_video_dir.mkdir(parents=True, exist_ok=True)
    pending = []
    for vid, vp in videos:
        if (per_video_dir / f"{vid}.json").exists():
            continue
        pending.append((vid, vp))
    print(f"Pending after resume: {len(pending)}")

    client = get_client()
    cache_db = REPO_ROOT / "data_v0" / "sft_gemini_uploads" / "cache.sqlite"
    cache_db.parent.mkdir(parents=True, exist_ok=True)
    cache = Cache(cache_db)

    # Upload all videos sequentially (Cache is sqlite — one thread only).
    print(f"uploading/reusing {len(pending)} videos via Files API...")
    files = {}
    for vid, vp in pending:
        try:
            files[vid] = get_or_upload(client, vp, vid, cache)
        except Exception as e:  # noqa: BLE001
            (per_video_dir / f"{vid}.json").write_text(
                json.dumps({"video_id": vid, "items": [], "error": f"upload_failed: {e!r}"}, indent=2)
            )
            print(f"  FAIL {vid}: {e!r}")
    print(f"uploaded {len(files)}/{len(pending)} videos.")

    t0 = time.time()
    n_done = 0
    n_items = 0
    gen_pending = [(v, p) for v, p in pending if v in files]
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(gen_one, client, files[vid], vid, args.max_per_video): vid
                for vid, _ in gen_pending}
        for fut in as_completed(futs):
            vid = futs[fut]
            items, err = fut.result()
            (per_video_dir / f"{vid}.json").write_text(
                json.dumps({"video_id": vid, "items": items, "error": err}, indent=2)
            )
            n_done += 1
            n_items += len(items)
            print(f"[{n_done}/{len(gen_pending)}] {vid}: {len(items)} items"
                  + (f" err={err}" if err else ""))
    print(f"generated {n_items} items across {n_done} videos in {time.time()-t0:.1f}s")

    # Roll up to all_questions.json
    all_items = []
    videos_seen = set()
    for f in sorted(per_video_dir.glob("*.json")):
        d = json.loads(f.read_text())
        for it in d.get("items", []):
            all_items.append(it)
        if d.get("items"):
            videos_seen.add(d["video_id"])
    from collections import Counter
    out = {
        "n_questions": len(all_items),
        "n_videos": len(videos_seen),
        "k_group_distribution": dict(Counter(it["k_group"] for it in all_items)),
        "category_distribution": dict(Counter(
            f"{it['category']}/{it['subcategory']}" for it in all_items
        )),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
        "questions": all_items,
    }
    (args.out_dir / "all_questions.json").write_text(json.dumps(out, indent=2))
    print(f"wrote {args.out_dir / 'all_questions.json'}")


if __name__ == "__main__":
    main()
