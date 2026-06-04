"""Run GPT-4o or Gemini-3-flash-preview on the same STEMO-Ambig questions Qwen3.5-27B saw.

Same system prompt as Qwen (no enumeration ask), same question text. For an
apples-to-apples comparison against `outputs_stemo/stemo_ambig_traces.jsonl`.

Providers:
  --provider gpt4o    : sample 16 frames per video, multi-image chat completion
  --provider gemini   : native video via Gemini Files API (uploads cached)

Output JSONL columns mirror the Qwen trace schema where possible:
  id, video_id, question, category, subcategory, k_group,
  thinking_trace, final_answer, thinking_char_count, elapsed_sec, error.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

DEFAULT_TRACES_REF = REPO_ROOT / "trace-pilot" / "outputs_stemo" / "stemo_ambig_traces.jsonl"
STEMO_VIDEOS = REPO_ROOT / "stemo" / "videos_h264"
DEFAULT_OUT_GPT4O = REPO_ROOT / "trace-pilot" / "outputs_stemo" / "stemo_ambig_traces_gpt4o.jsonl"
DEFAULT_OUT_GEMINI = REPO_ROOT / "trace-pilot" / "outputs_stemo" / "stemo_ambig_traces_gemini.jsonl"

# Match Qwen run exactly.
SYSTEM_PROMPT = (
    "You are an expert at answering questions about video content.\n"
    "Watch the video carefully and answer the question.\n"
    "Think step by step before giving your final answer."
)
BQA_SUFFIX = " Only answer with a single word 'Yes' or 'No'."

FRAMES = 16
MAX_TOKENS = 4096  # match the practical Qwen ceiling that produced the truncation pattern


def load_env():
    env_path = REPO_ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


def load_reference_traces(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    return rows


def load_done(path):
    if not Path(path).exists():
        return set()
    return {
        json.loads(line)["id"]
        for line in Path(path).read_text().splitlines() if line.strip()
    }


# ---------------- GPT-4o ----------------

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


def call_gpt4o(client, model, question, frames_b64):
    content = [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b}", "detail": "low"}}
        for b in frames_b64
    ]
    content.append({"type": "text", "text": question + BQA_SUFFIX})
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ],
        temperature=0.0,
        max_tokens=MAX_TOKENS,
        seed=0,
    )
    msg = resp.choices[0].message
    return "", (msg.content or "")  # GPT-4o exposes no reasoning trace via chat completions


# ---------------- Gemini ----------------

def get_gemini_helpers():
    from stemo_ambig.llm import get_client as get_gemini_client
    from stemo_ambig.video.cache import Cache
    from stemo_ambig.video.upload import get_or_upload
    return get_gemini_client, Cache, get_or_upload


def call_gemini(client, model, question, video_file):
    from google.genai import types
    cfg = types.GenerateContentConfig(
        temperature=0.0,
        max_output_tokens=MAX_TOKENS,
        response_mime_type="text/plain",
    )
    contents = [
        types.Content(role="user", parts=[
            types.Part(file_data=types.FileData(file_uri=video_file.uri, mime_type="video/mp4")),
            types.Part(text=question + BQA_SUFFIX),
        ]),
    ]
    resp = client.models.generate_content(
        model=model,
        contents=contents,
        config=types.GenerateContentConfig(
            temperature=0.0,
            max_output_tokens=MAX_TOKENS,
            response_mime_type="text/plain",
            system_instruction=SYSTEM_PROMPT,
        ),
    )
    # Extract any thinking/reasoning content if present (Gemini-3 may emit thought parts).
    thinking_chunks = []
    answer_chunks = []
    try:
        for cand in (resp.candidates or []):
            for part in (cand.content.parts or []):
                if getattr(part, "thought", False):
                    if getattr(part, "text", None):
                        thinking_chunks.append(part.text)
                elif getattr(part, "text", None):
                    answer_chunks.append(part.text)
    except Exception:
        pass
    answer = "".join(answer_chunks) or (resp.text or "")
    return "".join(thinking_chunks), answer


# ---------------- runner ----------------

def run_provider(args):
    load_env()
    ref = load_reference_traces(args.ref)
    # Resolve gold/question text from the reference traces themselves (already complete).
    questions = [
        {
            "id": r["id"], "video_id": r["video_id"], "question": r["question"],
            "category": r["category"], "subcategory": r["subcategory"], "k_group": r["k_group"],
        }
        for r in ref if not r.get("error")
    ]
    if args.limit:
        questions = questions[: args.limit]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set() if args.no_resume else load_done(out_path)
    pending = [q for q in questions if q["id"] not in done]
    print(f"provider={args.provider} total={len(questions)} done={len(done)} pending={len(pending)} workers={args.workers}")
    if not pending:
        return

    if args.provider == "gpt4o":
        if not os.environ.get("OPENAI_API_KEY"):
            sys.exit("OPENAI_API_KEY not set")
        from openai import OpenAI
        client = OpenAI()
        # Pre-warm frame cache for all referenced videos.
        unique_videos = sorted({q["video_id"] for q in pending})
        print(f"prewarming frames for {len(unique_videos)} videos...")
        with ThreadPoolExecutor(max_workers=4) as ex:
            for _ in ex.map(get_frames_cached, unique_videos):
                pass
        print("prewarm done.")

        def do_one(q):
            t0 = time.time()
            err = None
            thinking, answer = "", ""
            try:
                frames = get_frames_cached(q["video_id"])
                thinking, answer = call_gpt4o(client, args.model, q["question"], frames)
            except Exception as e:  # noqa: BLE001
                err = repr(e)[:500]
            return q, thinking, answer, err, time.time() - t0

    elif args.provider == "gemini":
        get_gemini_client, Cache, get_or_upload = get_gemini_helpers()
        gclient = get_gemini_client()
        cache_db = REPO_ROOT / "data_v0" / "stemo_ambig_gemini_uploads" / "cache.sqlite"
        cache_db.parent.mkdir(parents=True, exist_ok=True)
        cache = Cache(cache_db)

        # Upload all videos up-front; sequential because Files API hates parallel large uploads.
        unique_videos = sorted({q["video_id"] for q in pending})
        print(f"uploading/reusing {len(unique_videos)} videos via Files API...")
        files = {}
        for vid in unique_videos:
            path = STEMO_VIDEOS / f"{vid}.mp4"
            if not path.exists():
                print(f"  MISSING video {vid}")
                continue
            try:
                f = get_or_upload(gclient, path, vid, cache)
                files[vid] = f
                print(f"  ok {vid}")
            except Exception as e:  # noqa: BLE001
                print(f"  FAIL {vid}: {e!r}")
        print(f"uploaded {len(files)} videos.")

        def do_one(q):
            t0 = time.time()
            err = None
            thinking, answer = "", ""
            try:
                f = files.get(q["video_id"])
                if f is None:
                    raise RuntimeError(f"no uploaded video for {q['video_id']}")
                thinking, answer = call_gemini(gclient, args.model, q["question"], f)
            except Exception as e:  # noqa: BLE001
                err = repr(e)[:500]
            return q, thinking, answer, err, time.time() - t0
    else:
        sys.exit(f"unknown provider {args.provider}")

    t_start = time.time()
    n_done = 0
    with out_path.open("a") as fout:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(do_one, q) for q in pending]
            for fut in as_completed(futs):
                q, thinking, answer, err, elapsed = fut.result()
                rec = {
                    "id": q["id"], "video_id": q["video_id"], "question": q["question"],
                    "category": q["category"], "subcategory": q["subcategory"], "k_group": q["k_group"],
                    "thinking_trace": thinking, "final_answer": answer,
                    "thinking_char_count": len(thinking),
                    "elapsed_sec": round(elapsed, 2),
                    "provider": args.provider, "model": args.model,
                    "error": err,
                }
                fout.write(json.dumps(rec) + "\n")
                fout.flush()
                n_done += 1
                ans_preview = (answer or "").strip()[:40].replace("\n", " ")
                print(f"[{n_done}/{len(pending)}] {q['id']} ans={ans_preview!r} "
                      f"think={len(thinking)}ch elapsed={elapsed:.1f}s err={err}")
    print(f"done in {(time.time()-t_start):.1f}s -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", choices=["gpt4o", "gemini"], required=True)
    ap.add_argument("--model", default=None,
                    help="default: gpt-4o for gpt4o; gemini-3-flash-preview for gemini")
    ap.add_argument("--ref", type=Path, default=DEFAULT_TRACES_REF,
                    help="reference trace file (drives the question set)")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--system-prompt", default=None,
                    help="Override the system prompt (verbatim string).")
    ap.add_argument("--system-prompt-file", type=Path, default=None,
                    help="Read system prompt from a file (text).")
    args = ap.parse_args()
    if args.model is None:
        args.model = "gpt-4o" if args.provider == "gpt4o" else "gemini-3-flash-preview"
    if args.out is None:
        args.out = DEFAULT_OUT_GPT4O if args.provider == "gpt4o" else DEFAULT_OUT_GEMINI
    # Optional system-prompt override — patch the global at runtime
    if args.system_prompt or args.system_prompt_file:
        import sys as _sys
        prompt = args.system_prompt
        if args.system_prompt_file:
            prompt = args.system_prompt_file.read_text().strip()
        _sys.modules[__name__].SYSTEM_PROMPT = prompt
        global SYSTEM_PROMPT
        SYSTEM_PROMPT = prompt
    run_provider(args)


if __name__ == "__main__":
    main()
