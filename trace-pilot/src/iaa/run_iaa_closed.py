"""Run IAA (Interactive Ambig-Aware Accuracy) protocol on closed APIs.

For each STEMO-Ambig item:
  1. Turn-1: model sees video + question (with IAA system prompt).
  2. Sub-judge classifies Turn-1 response.
  3. If clarification: Turn-2 disambiguator → model Turn-2 response.
  4. Score per PROTOCOL_IAA.md.

Providers: gpt4o, gemini.

Output: predictions_iaa.jsonl with per-item full multi-turn transcript +
classification + scoring fields.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "trace-pilot" / "src"))

from iaa.sub_judge import classify_turn1, extract_yesno, PROTOCOL_VERSION  # noqa: E402

STEMO_VIDEOS = REPO_ROOT / "stemo" / "videos_h264"
GOLD_FILE = REPO_ROOT / "data_v0" / "stemo_ambig_candidates" / "all_questions.json"
DEFAULT_TRACES_REF = REPO_ROOT / "trace-pilot" / "outputs_stemo" / "stemo_ambig_traces.jsonl"

# IAA system prompt — gives explicit permission to clarify
SYSTEM_PROMPT = """You are an expert at answering questions about video content.

You may respond in any of three ways:
(1) If the question is unambiguous, give a single yes/no answer with a brief explanation.
(2) If the question has multiple valid interpretations because of an ambiguous referent, you may enumerate each interpretation and provide an answer for each, using the format:
    "<referent description 1>" -> Yes/No
    "<referent description 2>" -> Yes/No
(3) Alternatively, you may ask a clarifying question that identifies the ambiguous noun phrase (e.g., "Which boy do you mean?"). If you do, the asker will respond with a specific referent, and you must then answer for that referent.

Think step by step before responding."""

FRAMES = 16
MAX_TOKENS = 2048
TURN_CAP = 3


def load_env():
    env_path = REPO_ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


def load_gold():
    data = json.load(open(GOLD_FILE))
    return {q["id"]: q for q in data["questions"]}


def load_done(path):
    if not Path(path).exists():
        return set()
    return {json.loads(line)["id"] for line in Path(path).read_text().splitlines() if line.strip()}


def select_referent_index(item_id: str, K: int) -> int:
    h = int(hashlib.md5(item_id.encode()).hexdigest(), 16)
    return h % K


def sample_frames(video_path, n=FRAMES):
    import decord
    from PIL import Image
    vr = decord.VideoReader(str(video_path))
    total = len(vr)
    idxs = [int(i * (total - 1) / max(n - 1, 1)) for i in range(n)]
    frames = vr.get_batch(idxs).asnumpy()
    out = []
    for fr in frames:
        buf = io.BytesIO()
        Image.fromarray(fr).save(buf, format="JPEG", quality=85)
        out.append(base64.b64encode(buf.getvalue()).decode())
    return out


_FRAME_CACHE = {}
_FRAME_CACHE_LOCK = Lock()


def get_frames_cached(video_id):
    with _FRAME_CACHE_LOCK:
        if video_id in _FRAME_CACHE:
            return _FRAME_CACHE[video_id]
    path = STEMO_VIDEOS / f"{video_id}.mp4"
    if not path.exists():
        raise FileNotFoundError(str(path))
    frames = sample_frames(path)
    with _FRAME_CACHE_LOCK:
        _FRAME_CACHE[video_id] = frames
    return frames


# ===================== GPT-4o multi-turn =====================

def gpt4o_messages_init(frames_b64, question):
    content = [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b}", "detail": "low"}}
        for b in frames_b64
    ]
    content.append({"type": "text", "text": question})
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]


def gpt4o_call(client, model, messages):
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.0,
        max_tokens=MAX_TOKENS,
        seed=0,
    )
    return resp.choices[0].message.content or ""


def run_gpt4o_turn(client, model, video_id, question, prior_messages=None, followup_text=None):
    """One turn. If prior_messages is None this is turn 1 (initialize); else continue."""
    if prior_messages is None:
        frames = get_frames_cached(video_id)
        msgs = gpt4o_messages_init(frames, question)
    else:
        msgs = list(prior_messages)
        if followup_text is not None:
            msgs.append({"role": "user", "content": followup_text})
    response = gpt4o_call(client, model, msgs)
    msgs.append({"role": "assistant", "content": response})
    return response, msgs


# ===================== Gemini multi-turn =====================

def _gemini_helpers():
    from stemo_ambig.llm import get_client as get_gemini_client
    from stemo_ambig.video.cache import Cache
    from stemo_ambig.video.upload import get_or_upload
    return get_gemini_client, Cache, get_or_upload


def gemini_contents_init(video_file, question):
    from google.genai import types
    return [
        types.Content(role="user", parts=[
            types.Part(file_data=types.FileData(file_uri=video_file.uri, mime_type="video/mp4")),
            types.Part(text=question),
        ]),
    ]


def gemini_call(client, model, contents):
    from google.genai import types
    cfg = types.GenerateContentConfig(
        temperature=0.0,
        max_output_tokens=MAX_TOKENS,
        response_mime_type="text/plain",
        system_instruction=SYSTEM_PROMPT,
    )
    resp = client.models.generate_content(model=model, contents=contents, config=cfg)
    answer_chunks = []
    try:
        for cand in (resp.candidates or []):
            for part in (cand.content.parts or []):
                if getattr(part, "thought", False):
                    continue
                if getattr(part, "text", None):
                    answer_chunks.append(part.text)
    except Exception:
        pass
    return "".join(answer_chunks) or (resp.text or "")


def run_gemini_turn(client, model, video_file, question, prior_contents=None, followup_text=None):
    from google.genai import types
    if prior_contents is None:
        contents = gemini_contents_init(video_file, question)
    else:
        contents = list(prior_contents)
        if followup_text is not None:
            contents.append(types.Content(role="user", parts=[types.Part(text=followup_text)]))
    response = gemini_call(client, model, contents)
    contents.append(types.Content(role="model", parts=[types.Part(text=response)]))
    return response, contents


# ===================== Per-item IAA pipeline =====================

def score_item(gold_item: dict, turn1_response: str,
               classification: dict, turn2_response: str | None,
               turn3_response: str | None,
               selected_idx: int) -> dict:
    """Score per PROTOCOL_IAA.md §Scoring summary."""
    K = len(gold_item["interpretations"])
    cat = classification["category"]
    selected_ref = gold_item["interpretations"][selected_idx]
    gold_ans = (selected_ref.get("predicted_answer") or "").strip().lower()

    result = {
        "category": cat,
        "selected_idx": selected_idx,
        "K": K,
        "strict_K_correct": False,
        "aar_loose_correct": cat in {"enumerated", "clarified_scope"},
        "iaa_score": 0.0,
        "follow_through_correct": False,
        "turn2_decision": None,
        "turn3_decision": None,
    }

    if cat == "enumerated":
        # strict-K: did the response correctly answer for all K referents?
        # Use the judge's extracted enumerated_matches; match referents loosely
        matches = classification.get("enumerated_matches", []) or []
        # Map: lowercase referent_description -> decision
        m_map = {(m.get("referent_description") or "").strip().lower(): (m.get("decision") or "").strip().lower()
                 for m in matches}
        gold_map = {(ip.get("referent_description") or "").strip().lower(): (ip.get("predicted_answer") or "").strip().lower()
                    for ip in gold_item["interpretations"]}
        # Exact-match: all K gold referents present in matches with correct decision
        all_ok = True
        if len(matches) < K:
            all_ok = False
        else:
            for gold_desc, gold_a in gold_map.items():
                # find best-match key in m_map (loose substring)
                found = False
                for mk, mv in m_map.items():
                    if gold_desc in mk or mk in gold_desc or (
                        len(gold_desc) > 5 and len(mk) > 5 and (
                            gold_desc[:8] == mk[:8] or gold_desc[-8:] == mk[-8:]
                        )
                    ):
                        if mv == gold_a:
                            found = True
                            break
                if not found:
                    all_ok = False
                    break
        result["strict_K_correct"] = all_ok
        result["iaa_score"] = 1.0 if all_ok else 0.0
        return result

    if cat == "single_commit" or cat == "refused":
        return result

    # Clarification path. Need turn-2 yes/no extraction.
    if cat in {"clarified_scope", "clarified_vague"} and turn2_response is not None:
        ext = extract_yesno(
            gold_item["question"],
            selected_ref["referent_description"],
            selected_ref.get("disambiguated_question", ""),
            turn2_response,
        )
        decision = ext["decision"]
        result["turn2_decision"] = decision
        if decision in {"yes", "no"}:
            correct = (decision == gold_ans)
            result["follow_through_correct"] = correct
            base = 1.0 if correct else 0.0
            if cat == "clarified_vague":
                base *= 0.5
            result["iaa_score"] = base
            return result
        # decision unknown → maybe turn-3 if model clarified again
        if turn3_response is not None:
            ext3 = extract_yesno(
                gold_item["question"],
                selected_ref["referent_description"],
                selected_ref.get("disambiguated_question", ""),
                turn3_response,
            )
            d3 = ext3["decision"]
            result["turn3_decision"] = d3
            if d3 in {"yes", "no"}:
                correct = (d3 == gold_ans)
                result["follow_through_correct"] = correct
                base = 1.0 if correct else 0.0
                if cat == "clarified_vague":
                    base *= 0.5
                result["iaa_score"] = base
                return result

    return result


def build_disambig_utterance(gold_item: dict, selected_idx: int) -> str:
    ref = gold_item["interpretations"][selected_idx]
    desc = ref.get("referent_description", "this referent")
    return f"I am asking specifically about {desc}."


def needs_turn2(classification_cat: str) -> bool:
    return classification_cat in {"clarified_scope", "clarified_vague"}


# ===================== Provider drivers =====================

def do_one_gpt4o(client, model, gold_item):
    """End-to-end IAA for one item via gpt-4o."""
    qid = gold_item["id"]
    K = len(gold_item["interpretations"])
    selected_idx = select_referent_index(qid, K)

    t0 = time.time()
    rec = {
        "id": qid,
        "video_id": gold_item["video_id"],
        "question": gold_item["question"],
        "K": K,
        "selected_idx": selected_idx,
        "selected_referent": gold_item["interpretations"][selected_idx]["referent_description"],
        "turn_1_response": None,
        "turn_2_response": None,
        "turn_3_response": None,
        "classification": None,
        "score": None,
        "error": None,
        "elapsed_sec": 0.0,
        "protocol_version": PROTOCOL_VERSION,
    }
    try:
        r1, msgs = run_gpt4o_turn(client, model, gold_item["video_id"], gold_item["question"])
        rec["turn_1_response"] = r1
        cls = classify_turn1(gold_item["question"], gold_item["interpretations"], r1)
        rec["classification"] = cls
        if needs_turn2(cls["category"]):
            disambig = build_disambig_utterance(gold_item, selected_idx)
            r2, msgs = run_gpt4o_turn(client, model, gold_item["video_id"], gold_item["question"],
                                       prior_messages=msgs, followup_text=disambig)
            rec["turn_2_response"] = r2
            # Check if r2 is yet another clarification — if so we'd allow one more turn (cap=3)
            cls2 = classify_turn1(gold_item["question"], gold_item["interpretations"], r2)
            if cls2["category"] in {"clarified_scope", "clarified_vague"}:
                r3, msgs = run_gpt4o_turn(client, model, gold_item["video_id"], gold_item["question"],
                                           prior_messages=msgs, followup_text=disambig)
                rec["turn_3_response"] = r3
        rec["score"] = score_item(gold_item, r1, cls, rec["turn_2_response"], rec["turn_3_response"], selected_idx)
    except Exception as e:
        rec["error"] = repr(e)[:500]
    rec["elapsed_sec"] = round(time.time() - t0, 2)
    return rec


def do_one_gemini(client, model, video_file, gold_item):
    qid = gold_item["id"]
    K = len(gold_item["interpretations"])
    selected_idx = select_referent_index(qid, K)
    t0 = time.time()
    rec = {
        "id": qid,
        "video_id": gold_item["video_id"],
        "question": gold_item["question"],
        "K": K,
        "selected_idx": selected_idx,
        "selected_referent": gold_item["interpretations"][selected_idx]["referent_description"],
        "turn_1_response": None,
        "turn_2_response": None,
        "turn_3_response": None,
        "classification": None,
        "score": None,
        "error": None,
        "elapsed_sec": 0.0,
        "protocol_version": PROTOCOL_VERSION,
    }
    try:
        r1, contents = run_gemini_turn(client, model, video_file, gold_item["question"])
        rec["turn_1_response"] = r1
        cls = classify_turn1(gold_item["question"], gold_item["interpretations"], r1)
        rec["classification"] = cls
        if needs_turn2(cls["category"]):
            disambig = build_disambig_utterance(gold_item, selected_idx)
            r2, contents = run_gemini_turn(client, model, video_file, gold_item["question"],
                                            prior_contents=contents, followup_text=disambig)
            rec["turn_2_response"] = r2
            cls2 = classify_turn1(gold_item["question"], gold_item["interpretations"], r2)
            if cls2["category"] in {"clarified_scope", "clarified_vague"}:
                r3, contents = run_gemini_turn(client, model, video_file, gold_item["question"],
                                                prior_contents=contents, followup_text=disambig)
                rec["turn_3_response"] = r3
        rec["score"] = score_item(gold_item, r1, cls, rec["turn_2_response"], rec["turn_3_response"], selected_idx)
    except Exception as e:
        rec["error"] = repr(e)[:500]
    rec["elapsed_sec"] = round(time.time() - t0, 2)
    return rec


def compute_metrics(records):
    """Compute aggregate IAA metrics from per-item records."""
    valid = [r for r in records if r.get("score") and not r.get("error")]
    if not valid:
        return {"n": 0}
    n = len(valid)
    iaa = sum(r["score"]["iaa_score"] for r in valid) / n
    strict = sum(1 for r in valid if r["score"]["strict_K_correct"]) / n
    aar_loose = sum(1 for r in valid if r["score"]["aar_loose_correct"]) / n
    clar_rate = sum(1 for r in valid if r["classification"]["category"] in {"clarified_scope", "clarified_vague"}) / n
    rec_no_recall = sum(1 for r in valid if r["classification"]["category"] == "clarified_vague") / n
    clar_items = [r for r in valid if r["classification"]["category"] in {"clarified_scope", "clarified_vague"}]
    follow = (sum(1 for r in clar_items if r["score"]["follow_through_correct"]) / len(clar_items)) if clar_items else None
    # Per-K
    by_K = {}
    for r in valid:
        K = r["K"]
        bucket = "2" if K == 2 else "3" if K == 3 else "4-6" if 4 <= K <= 6 else "7+"
        by_K.setdefault(bucket, []).append(r)
    per_k = {}
    for k, items in sorted(by_K.items()):
        m = len(items)
        per_k[k] = {
            "n": m,
            "iaa": sum(r["score"]["iaa_score"] for r in items) / m,
            "strict_K": sum(1 for r in items if r["score"]["strict_K_correct"]) / m,
            "aar_loose": sum(1 for r in items if r["score"]["aar_loose_correct"]) / m,
            "clarification_rate": sum(1 for r in items if r["classification"]["category"] in {"clarified_scope", "clarified_vague"}) / m,
        }
    return {
        "n": n,
        "iaa": iaa,
        "strict_K": strict,
        "aar_loose": aar_loose,
        "clarification_rate": clar_rate,
        "recognition_no_recall": rec_no_recall,
        "follow_through_rate": follow,
        "per_K": per_k,
        "judge_version": "gemini-3-flash-preview@iaa-v1.0",
        "n_errored": sum(1 for r in records if r.get("error")),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", choices=["gpt4o", "gemini"], required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--out", required=True, help="output JSONL path")
    ap.add_argument("--metrics-out", default=None, help="metrics JSON path (default: <out>.metrics.json)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()

    if args.model is None:
        args.model = "gpt-4o-2024-08-06" if args.provider == "gpt4o" else "gemini-3-flash-preview"

    load_env()
    gold = load_gold()
    ids = sorted(gold.keys())
    if args.limit:
        ids = ids[: args.limit]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set() if args.no_resume else load_done(out_path)
    pending = [g for g in (gold[i] for i in ids) if g["id"] not in done]
    print(f"[IAA] provider={args.provider} model={args.model} total={len(ids)} done={len(done)} pending={len(pending)} workers={args.workers}")

    if not pending:
        print("[IAA] nothing to do.")
        return

    out_lock = Lock()
    results = []
    if not args.no_resume and out_path.exists():
        for line in out_path.read_text().splitlines():
            if line.strip():
                try:
                    results.append(json.loads(line))
                except Exception:
                    pass

    mode = "a" if not args.no_resume else "w"

    if args.provider == "gpt4o":
        if not os.environ.get("OPENAI_API_KEY"):
            sys.exit("OPENAI_API_KEY not set")
        from openai import OpenAI
        client = OpenAI()
        # pre-warm frames
        unique_videos = sorted({g["video_id"] for g in pending})
        print(f"[IAA] prewarming frames for {len(unique_videos)} videos...")
        with ThreadPoolExecutor(max_workers=4) as ex:
            for _ in ex.map(get_frames_cached, unique_videos):
                pass

        def do(g):
            return do_one_gpt4o(client, args.model, g)

    elif args.provider == "gemini":
        get_gemini_client, Cache, get_or_upload = _gemini_helpers()
        gclient = get_gemini_client()
        cache_db = REPO_ROOT / "data_v0" / "stemo_ambig_gemini_uploads" / "cache.sqlite"
        cache_db.parent.mkdir(parents=True, exist_ok=True)
        cache = Cache(cache_db)
        unique_videos = sorted({g["video_id"] for g in pending})
        print(f"[IAA] checking/uploading {len(unique_videos)} videos...", flush=True)
        files = {}
        # Two-phase: parallel verify of cached entries, then sequential re-upload for misses.
        # sqlite is thread-local, so we read cache snapshot once.
        cache_snapshot = {}
        for vid in unique_videos:
            cached = cache.get_upload(vid)
            if cached is not None:
                cache_snapshot[vid] = cached  # (file_uri, file_name)

        # Phase 1: parallel verify cached entries (network-only, no sqlite)
        def _verify(vid):
            if vid not in cache_snapshot:
                return vid, None
            _, file_name = cache_snapshot[vid]
            try:
                f = gclient.files.get(name=file_name)
                if getattr(f, "state", None) and str(f.state).endswith("ACTIVE"):
                    return vid, f
            except Exception:
                pass
            return vid, None

        from concurrent.futures import ThreadPoolExecutor as _TPE
        verified = 0
        to_upload = []
        with _TPE(max_workers=8) as ex:
            for i, (vid, f) in enumerate(ex.map(_verify, unique_videos), 1):
                if f is not None:
                    files[vid] = f
                    verified += 1
                else:
                    to_upload.append(vid)
                if i % 20 == 0:
                    print(f"  verify {i}/{len(unique_videos)}: {verified} valid", flush=True)
        print(f"[IAA] verified {verified} cached; {len(to_upload)} need upload", flush=True)

        # Phase 2: sequential re-upload (sqlite-bound writes happen here, single-threaded)
        for j, vid in enumerate(to_upload, 1):
            p = STEMO_VIDEOS / f"{vid}.mp4"
            if not p.exists():
                print(f"  upload {j}/{len(to_upload)}: MISSING {vid}", flush=True)
                continue
            try:
                f = get_or_upload(gclient, p, vid, cache)
                files[vid] = f
                print(f"  upload {j}/{len(to_upload)}: {vid} OK", flush=True)
            except Exception as e:
                print(f"  upload {j}/{len(to_upload)}: {vid} FAIL {repr(e)[:100]}", flush=True)
        print(f"[IAA] total {len(files)} videos ready.", flush=True)

        def do(g):
            f = files.get(g["video_id"])
            if f is None:
                return {"id": g["id"], "error": f"no uploaded video for {g['video_id']}"}
            return do_one_gemini(gclient, args.model, f, g)

    with open(out_path, mode) as fout, ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(do, g): g for g in pending}
        for n, fut in enumerate(as_completed(futs), 1):
            try:
                rec = fut.result()
            except Exception as e:
                rec = {"id": futs[fut]["id"], "error": repr(e)[:500]}
            results.append(rec)
            with out_lock:
                fout.write(json.dumps(rec) + "\n")
                fout.flush()
            if n % 10 == 0 or n == len(pending):
                err = sum(1 for r in results if r.get("error"))
                print(f"  [{n}/{len(pending)}] errors={err}", flush=True)

    metrics = compute_metrics(results)
    mp = Path(args.metrics_out) if args.metrics_out else out_path.with_suffix(".metrics.json")
    mp.write_text(json.dumps(metrics, indent=2))
    print(f"[IAA] wrote metrics: {mp}")
    print(json.dumps({k: v for k, v in metrics.items() if k != "per_K"}, indent=2))


if __name__ == "__main__":
    main()
