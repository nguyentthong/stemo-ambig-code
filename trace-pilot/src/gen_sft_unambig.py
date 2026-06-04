"""Generate unambig yes/no questions on a video pool via Gemini.

Mirror of gen_sft_candidates.py but produces single-answer yes/no questions
(no referential ambiguity). Used as the matched-volume "negative" SFT material
so the fine-tuned model learns when NOT to enumerate.

Output JSONL (one per line):
  {"id": "unambig_<vid>_NNN", "video_id": "...", "video_path": "...",
   "question": "...", "gold_answer": "yes|no"}
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


UNAMBIG_PROMPT = """You are constructing training items for video QA.

Watch the video. Produce up to {max_per_video} UNAMBIGUOUS yes/no questions about the video.

Constraints:
- Each question MUST have exactly one natural reading. Avoid surface phrases that could
  refer to multiple people/objects/moments (no "the man" if there are several men; use
  "the man in the blue shirt at 0:15" instead).
- Each question's answer must be unambiguous: either Yes or No.
- Spread across content types: actions, attributes, events, timing.
- Vary which answer is correct; aim for a roughly even Yes/No split.

Output STRICT JSON, no markdown fences:
{{
  "items": [
    {{
      "question": "<single-reading yes/no question>",
      "gold_answer": "<yes|no>"
    }},
    ...
  ]
}}
"""


def build_video_list(args):
    items = []
    if args.pool_jsonl:
        for line in Path(args.pool_jsonl).read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            items.append((r["video_id"], Path(r["video_path"])))
    if args.video_dir:
        for p in sorted(Path(args.video_dir).rglob("*.mp4")):
            vid = p.stem
            if vid in {i[0] for i in items}:
                continue
            items.append((vid, p))
    if args.exclude_ids_file:
        excl = set(Path(args.exclude_ids_file).read_text().split())
        items = [(v, p) for v, p in items if v not in excl]
    if args.limit:
        items = items[: args.limit]
    return items


def gen_one(client, file, video_id, video_path, max_per_video):
    prompt = UNAMBIG_PROMPT.format(max_per_video=max_per_video)
    cfg = types.GenerateContentConfig(
        temperature=0.7,
        response_mime_type="application/json",
        max_output_tokens=8192,
    )
    try:
        resp = client.models.generate_content(
            model=GEMINI_MODEL, contents=[file, prompt], config=cfg,
        )
        parsed = parse_json(resp.text or "")
        items = parsed.get("items", []) if isinstance(parsed, dict) else []
        out = []
        for i, it in enumerate(items):
            q = (it.get("question") or "").strip()
            a = (it.get("gold_answer") or "").strip().lower()
            if not q or a not in ("yes", "no"):
                continue
            out.append({
                "id": f"unambig_{video_id}_{i:03d}",
                "video_id": video_id,
                "video_path": str(video_path),
                "question": q,
                "gold_answer": a,
            })
        return out, None
    except Exception as e:  # noqa: BLE001
        return [], f"generate_failed: {e!r}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-dir", type=Path, default=None)
    ap.add_argument("--pool-jsonl", type=Path, default=None)
    ap.add_argument("--exclude-ids-file", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-per-video", type=int, default=12)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    if not (args.video_dir or args.pool_jsonl):
        sys.exit("Need --video-dir or --pool-jsonl")
    args.out.parent.mkdir(parents=True, exist_ok=True)

    videos = build_video_list(args)
    print(f"Processing {len(videos)} videos.")

    # Resume from existing rows
    done_vids = set()
    if args.out.exists():
        for line in args.out.read_text().splitlines():
            if not line.strip():
                continue
            done_vids.add(json.loads(line)["video_id"])
    pending = [(v, p) for v, p in videos if v not in done_vids]
    print(f"Pending after resume: {len(pending)}")

    client = get_client()
    cache_db = REPO_ROOT / "data_v0" / "sft_gemini_uploads" / "cache.sqlite"
    cache_db.parent.mkdir(parents=True, exist_ok=True)
    cache = Cache(cache_db)

    # Upload videos sequentially first (Cache is single-thread sqlite).
    print(f"uploading/reusing {len(pending)} videos...")
    files = {}
    paths = {v: p for v, p in pending}
    for vid, vp in pending:
        try:
            files[vid] = get_or_upload(client, vp, vid, cache)
        except Exception as e:  # noqa: BLE001
            print(f"  FAIL {vid}: {e!r}")
    print(f"uploaded {len(files)}/{len(pending)} videos.")
    gen_pending = [(v, p) for v, p in pending if v in files]

    t0 = time.time()
    n_items = 0
    with args.out.open("a") as fout:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(gen_one, client, files[vid], vid, paths[vid], args.max_per_video): vid
                    for vid, _ in gen_pending}
            for n, fut in enumerate(as_completed(futs), 1):
                vid = futs[fut]
                items, err = fut.result()
                for it in items:
                    fout.write(json.dumps(it) + "\n")
                fout.flush()
                n_items += len(items)
                print(f"[{n}/{len(pending)}] {vid}: {len(items)} items"
                      + (f" err={err}" if err else ""))
    print(f"generated {n_items} unambig items in {time.time()-t0:.1f}s -> {args.out}")


if __name__ == "__main__":
    main()
