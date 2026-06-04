"""Format SFT data from candidate JSON into Qwen-VL chat JSONL.

Inputs:
  --ambig-candidates  : path to all_questions.json (same schema as data_v0/stemo_ambig_candidates)
  --unambig-jsonl     : optional JSONL with single-answer items
                         {"video_id": "...", "question": "...", "gold_answer": "yes|no", "video_path": "..."}
  --video-dir         : directory containing the .mp4 files referenced by video_id

For each ambig candidate:
  1. Distill a Gemini reasoning trace given the gold scaffold (cached per id).
  2. Build the assistant target: "<think>{distilled}</think>\n\n{enumerated answer}"
  3. Emit a Qwen-VL chat message record.

For each unambig item:
  1. Distill a Gemini reasoning trace for the single-answer case (do NOT enumerate).
  2. Build the assistant target: "<think>{distilled}</think>\n\n{Yes|No}"

Output:
  <out-dir>/sft_train.jsonl
  <out-dir>/sft_dev.jsonl
  <out-dir>/sft_meta.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from stemo_ambig import GEMINI_MODEL  # noqa: E402
from stemo_ambig.llm import get_client  # noqa: E402
from google.genai import types  # noqa: E402


SYSTEM_PROMPT = (
    "You are an expert at answering questions about video content.\n"
    "Watch the video carefully and answer the question.\n"
    "Think step by step before giving your final answer.\n"
    "If the question has multiple valid interpretations because of an ambiguous "
    "referent, enumerate each interpretation explicitly and provide an answer for each."
)

AMBIG_DISTILL_PROMPT = """You are writing a reasoning trace for a video QA model.

The model will see a video and be asked a question that has K valid interpretations
(referential ambiguity). For each interpretation we know the gold yes/no answer.

Surface question: {question}

Interpretations and gold answers:
{interp_block}

Write a natural reasoning trace (200-500 words) that:
- Notices the referent in the surface question could refer to several things in the video
- Walks through each possible reading and what evidence in the video makes it valid
- For each reading, arrives at the gold yes/no answer above
- Does NOT mention "gold answer" or that this is a benchmark item
- Sounds like a careful viewer thinking out loud, not a template

Output the reasoning trace ONLY (no preamble, no JSON, plain prose).
"""

UNAMBIG_DISTILL_PROMPT = """You are writing a reasoning trace for a video QA model.

Surface question: {question}
Gold answer: {gold}

Write a natural reasoning trace (100-300 words) that:
- Identifies the referent in the question (no ambiguity here)
- Walks through the relevant video evidence
- Arrives at the gold yes/no answer
- Sounds like a careful viewer thinking out loud

Output the reasoning trace ONLY (no preamble, no JSON, plain prose).
"""


def render_interp_block(interps):
    lines = []
    for i, ip in enumerate(interps, 1):
        lines.append(
            f"[{i}] {ip['referent_description']}\n"
            f"    disambiguated question: {ip['disambiguated_question']}\n"
            f"    gold answer: {ip['predicted_answer']}"
        )
    return "\n\n".join(lines)


def format_ambig_answer(interps):
    """Build the user-facing enumerated answer the model should learn to produce."""
    lines = [f"This question has {len(interps)} valid interpretations."]
    for ip in interps:
        ans = ip["predicted_answer"].strip().lower()
        lines.append(f'- "{ip["referent_description"]}" → {ans.capitalize()}')
    return "\n".join(lines)


def _generate_clean_text(client, prompt, max_tokens):
    """Return Gemini text only if finish_reason==STOP and text non-empty; else None.

    None signals 'retry next pass with bigger budget' to the caller, who must
    not cache None as a successful result.
    """
    cfg = types.GenerateContentConfig(
        temperature=0.7, response_mime_type="text/plain", max_output_tokens=max_tokens,
    )
    resp = client.models.generate_content(
        model=GEMINI_MODEL, contents=[prompt], config=cfg
    )
    text = (resp.text or "").strip()
    if not text:
        return None
    # Check finish_reason on the first candidate.
    try:
        fr = str(resp.candidates[0].finish_reason).upper()
    except Exception:  # noqa: BLE001
        fr = ""
    if "MAX_TOKENS" in fr or "LENGTH" in fr:
        return None
    if fr and "STOP" not in fr and "FINISH" not in fr:
        # SAFETY, RECITATION, OTHER → drop & retry later
        return None
    return text


def distill_one_ambig(client, item, max_tokens=4096):
    interps = item["interpretations"]
    prompt = AMBIG_DISTILL_PROMPT.format(
        question=item["question"],
        interp_block=render_interp_block(interps),
    )
    return _generate_clean_text(client, prompt, max_tokens)


def distill_one_unambig(client, item, max_tokens=2048):
    prompt = UNAMBIG_DISTILL_PROMPT.format(
        question=item["question"], gold=item["gold_answer"]
    )
    return _generate_clean_text(client, prompt, max_tokens)


def build_chat_record(video_path, question, system_prompt, thinking, final_answer, meta):
    assistant_text = f"<think>{thinking}</think>\n\n{final_answer}"
    return {
        "video_path": str(video_path),
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {"type": "video", "video": str(video_path)},
                {"type": "text", "text": question},
            ]},
            {"role": "assistant", "content": assistant_text},
        ],
        "meta": meta,
    }


def load_cache(p):
    if not p.exists():
        return {}
    return {json.loads(l)["cache_key"]: json.loads(l) for l in p.read_text().splitlines() if l.strip()}


def cache_key(kind, item_id, question):
    h = hashlib.sha256(f"{kind}|{item_id}|{question}".encode()).hexdigest()[:16]
    return f"{kind}_{item_id}_{h}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ambig-candidates", type=Path, required=True)
    ap.add_argument("--unambig-jsonl", type=Path, default=None)
    ap.add_argument("--video-dir", type=Path, required=True,
                    help="dir containing <video_id>.mp4")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--dev-frac", type=float, default=0.1)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    distill_cache_path = args.out_dir / "distill_cache.jsonl"
    cache = load_cache(distill_cache_path)
    cache_lock_path = args.out_dir / ".lock"

    # Load ambig
    ambig_payload = json.loads(Path(args.ambig_candidates).read_text())
    ambig_items = ambig_payload.get("questions", [])
    if args.limit:
        ambig_items = ambig_items[: args.limit]

    # Filter to items whose video exists locally
    ambig_items = [
        it for it in ambig_items
        if (args.video_dir / f"{it['video_id']}.mp4").exists()
    ]
    print(f"ambig items with local video: {len(ambig_items)}")

    # Load unambig
    unambig_items = []
    if args.unambig_jsonl:
        for line in Path(args.unambig_jsonl).read_text().splitlines():
            if not line.strip():
                continue
            unambig_items.append(json.loads(line))
        # Keep at most as many unambig as ambig (1:1 mix).
        unambig_items = unambig_items[: len(ambig_items)]
        print(f"unambig items: {len(unambig_items)}")

    client = get_client()

    cache_writer_lock = __import__("threading").Lock()
    cache_file_handle = distill_cache_path.open("a")

    def get_or_distill(kind, item_id, question, distill_fn, item):
        key = cache_key(kind, item_id, question)
        if key in cache:
            cached = cache[key]["trace"]
            if cached and cached.strip():
                return cached
            # Cached row exists but is empty/truncated from an earlier buggy run.
            # Fall through to regenerate.
        trace = distill_fn(client, item)
        if trace is None or not trace.strip():
            # Don't cache bad results — let them retry on the next pass.
            return None
        with cache_writer_lock:
            cache[key] = {"cache_key": key, "trace": trace}
            cache_file_handle.write(json.dumps({"cache_key": key, "trace": trace}) + "\n")
            cache_file_handle.flush()
        return trace

    # Distill ambig
    print("distilling ambig traces...")
    ambig_records = [None] * len(ambig_items)
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {
            ex.submit(get_or_distill, "ambig", it["id"], it["question"], distill_one_ambig, it): i
            for i, it in enumerate(ambig_items)
        }
        for n, fut in enumerate(as_completed(futs), 1):
            i = futs[fut]
            try:
                trace = fut.result()
            except Exception as e:  # noqa: BLE001
                print(f"  ambig {ambig_items[i]['id']} distill failed: {e!r}")
                continue
            if trace is None or not trace.strip():
                continue  # skip items with empty/truncated traces
            it = ambig_items[i]
            ambig_records[i] = build_chat_record(
                video_path=args.video_dir / f"{it['video_id']}.mp4",
                question=it["question"],
                system_prompt=SYSTEM_PROMPT,
                thinking=trace,
                final_answer=format_ambig_answer(it["interpretations"]),
                meta={"id": it["id"], "kind": "ambig", "k_group": it["k_group"],
                      "video_id": it["video_id"]},
            )
            if n % 50 == 0:
                print(f"  ambig {n}/{len(ambig_items)}")

    # Distill unambig
    unambig_records = [None] * len(unambig_items)
    if unambig_items:
        print("distilling unambig traces...")
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {
                ex.submit(get_or_distill, "unambig",
                          it.get("id") or f"u_{it['video_id']}_{i}", it["question"],
                          distill_one_unambig, it): i
                for i, it in enumerate(unambig_items)
            }
            for n, fut in enumerate(as_completed(futs), 1):
                i = futs[fut]
                try:
                    trace = fut.result()
                except Exception as e:  # noqa: BLE001
                    print(f"  unambig {unambig_items[i].get('id')} failed: {e!r}")
                    continue
                if trace is None or not trace.strip():
                    continue  # skip items with empty/truncated traces
                it = unambig_items[i]
                final_ans = it["gold_answer"].strip().capitalize()
                vp = it.get("video_path") or str(args.video_dir / f"{it['video_id']}.mp4")
                unambig_records[i] = build_chat_record(
                    video_path=vp,
                    question=it["question"],
                    system_prompt=SYSTEM_PROMPT,
                    thinking=trace,
                    final_answer=final_ans,
                    meta={"id": it.get("id"), "kind": "unambig", "video_id": it["video_id"]},
                )
                if n % 50 == 0:
                    print(f"  unambig {n}/{len(unambig_items)}")

    records = [r for r in ambig_records + unambig_records if r is not None]
    print(f"total formatted records: {len(records)}")

    # Split by video_id so train/dev don't share videos.
    rng = random.Random(args.seed)
    vids = sorted({r["meta"]["video_id"] for r in records})
    rng.shuffle(vids)
    n_dev = max(1, int(len(vids) * args.dev_frac))
    dev_vids = set(vids[:n_dev])
    train = [r for r in records if r["meta"]["video_id"] not in dev_vids]
    dev = [r for r in records if r["meta"]["video_id"] in dev_vids]

    rng.shuffle(train)

    (args.out_dir / "sft_train.jsonl").write_text(
        "\n".join(json.dumps(r) for r in train) + "\n"
    )
    (args.out_dir / "sft_dev.jsonl").write_text(
        "\n".join(json.dumps(r) for r in dev) + "\n"
    )

    meta = {
        "n_train": len(train),
        "n_dev": len(dev),
        "n_ambig": sum(1 for r in records if r["meta"]["kind"] == "ambig"),
        "n_unambig": sum(1 for r in records if r["meta"]["kind"] == "unambig"),
        "n_videos_train": len({r["meta"]["video_id"] for r in train}),
        "n_videos_dev": len(dev_vids),
        "system_prompt": SYSTEM_PROMPT,
        "ambig_candidates_source": str(args.ambig_candidates),
        "unambig_source": str(args.unambig_jsonl) if args.unambig_jsonl else None,
        "video_dir": str(args.video_dir),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
    }
    (args.out_dir / "sft_meta.json").write_text(json.dumps(meta, indent=2))
    cache_file_handle.close()
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
