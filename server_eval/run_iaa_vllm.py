"""IAA multi-turn runner against a local vLLM OpenAI-compatible server.

Protocol-identical to trace-pilot/src/iaa/run_iaa_open.py (same system
prompt, referent hash, turn logic, judge, and score_item — all imported
from it), but generation goes through vLLM for throughput: one tp=8
server per model, this driver fans out concurrent requests.

Frames: uniform sampling via decord, sent as base64 JPEG image parts
(the paper's setting is N uniform frames; 16 default, 8 for InternVL).

Usage (run_all.sh drives this):
  python server_eval/run_iaa_vllm.py --base-url http://localhost:8000/v1 \
      --served-name MODEL --frames 16 --output eval_runs/TAG/iaa_predictions.jsonl
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "trace-pilot" / "src"))

from iaa.run_iaa_open import (  # noqa: E402
    SYSTEM_PROMPT, MAX_TOKENS, build_disambig_utterance,
    select_referent_index, score_item,
)
from iaa.sub_judge import classify_turn1, PROTOCOL_VERSION  # noqa: E402

QUESTIONS = REPO_ROOT / "data_v0/stemo_ambig_candidates/all_questions.json"
VIDEOS = REPO_ROOT / "stemo/videos_h264"

_frame_cache: dict = {}
_cache_lock = threading.Lock()
_write_lock = threading.Lock()


def encode_frames(video_path: str, num_frames: int) -> list[str]:
    key = (video_path, num_frames)
    with _cache_lock:
        if key in _frame_cache:
            return _frame_cache[key]
    import decord
    from PIL import Image
    vr = decord.VideoReader(video_path)
    total = len(vr)
    n = min(num_frames, total)
    idxs = [int(i * (total - 1) / max(n - 1, 1)) for i in range(n)]
    arrs = vr.get_batch(idxs).asnumpy()
    encoded = []
    for a in arrs:
        img = Image.fromarray(a)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=90)
        encoded.append(base64.b64encode(buf.getvalue()).decode())
    with _cache_lock:
        _frame_cache[key] = encoded
    return encoded


def build_items() -> list[dict]:
    qs = json.load(open(QUESTIONS))["questions"]
    rows, missing = [], 0
    for q in qs:
        vp = VIDEOS / f"{q['video_id']}.mp4"
        if not vp.exists():
            missing += 1
            continue
        rows.append({
            "id": q["id"],
            "video_id": q["video_id"],
            "video_path": str(vp),
            "question": q["question"],
            "K": len(q["interpretations"]),
            "interpretations": [
                {"referent_description": ip["referent_description"],
                 "predicted_answer": ip["predicted_answer"],
                 "disambiguated_question": ip.get("disambiguated_question", "")}
                for ip in q["interpretations"]
            ],
        })
    print(f"[iaa-vllm] {len(rows)} items ({missing} missing videos)", flush=True)
    return rows


def chat(client, model: str, messages: list, max_tokens: int) -> str:
    last_err = None
    for attempt in range(4):
        try:
            resp = client.chat.completions.create(
                model=model, messages=messages,
                temperature=0.0, max_tokens=max_tokens,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"chat failed after retries: {last_err!r}")


def turn1_messages(question: str, frames_b64: list[str]) -> list:
    content = [{"type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{f}"}}
               for f in frames_b64]
    content.append({"type": "text", "text": question})
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]


def append_followup(messages: list, prior_response: str, followup: str) -> list:
    return messages + [
        {"role": "assistant", "content": prior_response},
        {"role": "user", "content": followup},
    ]


def run_item(client, served_name: str, r: dict, frames_n: int,
             max_tokens: int) -> dict:
    qid = r["id"]
    K = len(r["interpretations"])
    selected_idx = select_referent_index(qid, K)
    rec = {
        "id": qid,
        "video_id": r.get("video_id", ""),
        "video_path": r["video_path"],
        "question": r["question"],
        "K": K,
        "selected_idx": selected_idx,
        "selected_referent": r["interpretations"][selected_idx]["referent_description"],
        "turn_1_response": None,
        "turn_2_response": None,
        "turn_3_response": None,
        "classification": None,
        "score": None,
        "error": None,
        "elapsed_sec": 0.0,
        "protocol_version": PROTOCOL_VERSION,
        "engine": "vllm",
    }
    t0 = time.time()
    try:
        frames = encode_frames(r["video_path"], frames_n)
        msgs = turn1_messages(r["question"], frames)
        r1 = chat(client, served_name, msgs, max_tokens)
        rec["turn_1_response"] = r1
        cls = classify_turn1(r["question"], r["interpretations"], r1)
        rec["classification"] = cls
        if cls["category"] in {"clarified_scope", "clarified_vague"}:
            disambig = build_disambig_utterance(r["interpretations"], selected_idx)
            msgs2 = append_followup(msgs, r1, disambig)
            r2 = chat(client, served_name, msgs2, max_tokens)
            rec["turn_2_response"] = r2
            cls2 = classify_turn1(r["question"], r["interpretations"], r2)
            if cls2["category"] in {"clarified_scope", "clarified_vague"}:
                msgs3 = append_followup(msgs2, r2, disambig)
                r3 = chat(client, served_name, msgs3, max_tokens)
                rec["turn_3_response"] = r3
        rec["score"] = score_item(r["interpretations"], cls,
                                  rec["turn_2_response"], rec["turn_3_response"],
                                  selected_idx, r["question"])
    except Exception as e:  # noqa: BLE001
        rec["error"] = repr(e)[:500]
    rec["elapsed_sec"] = round(time.time() - t0, 2)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--served-name", required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=MAX_TOKENS)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    from openai import OpenAI
    client = OpenAI(base_url=args.base_url, api_key="local", timeout=600)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    # resume: keep clean rows, re-attempt errored ones
    done_ids = set()
    if args.output.exists():
        clean = []
        for line in args.output.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if not rec.get("error"):
                done_ids.add(rec["id"])
                clean.append(line)
        args.output.write_text("\n".join(clean) + ("\n" if clean else ""))

    rows = build_items()
    pending = [r for r in rows if r["id"] not in done_ids]
    if args.limit:
        pending = pending[: args.limit]
    print(f"[iaa-vllm] total={len(rows)} done={len(done_ids)} "
          f"pending={len(pending)} concurrency={args.concurrency}", flush=True)
    if not pending:
        return

    t0 = time.time()
    n_done = 0
    n_err = 0
    with args.output.open("a") as fout, \
         ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futs = {pool.submit(run_item, client, args.served_name, r,
                            args.frames, args.max_new_tokens): r["id"]
                for r in pending}
        for fut in as_completed(futs):
            rec = fut.result()
            with _write_lock:
                fout.write(json.dumps(rec) + "\n")
                fout.flush()
            n_done += 1
            if rec.get("error"):
                n_err += 1
            if n_done % 20 == 0 or n_done == len(pending):
                rate = n_done / max(time.time() - t0, 0.01)
                eta = (len(pending) - n_done) / max(rate, 1e-6) / 60
                print(f"[iaa-vllm] [{n_done}/{len(pending)}] errors={n_err} "
                      f"rate={rate:.2f}/s eta={eta:.0f}min", flush=True)
    print(f"[iaa-vllm] finished: {n_done} items, {n_err} errors", flush=True)


if __name__ == "__main__":
    main()
