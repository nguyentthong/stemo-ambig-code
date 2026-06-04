"""Run Qwen3.5-27B over the 1056-question STEMO-Ambig candidate set.

Sends each ambiguous question with its STEMO video, captures the model's
reasoning_content and final answer, writes one JSONL line per call to
outputs_stemo/stemo_ambig_traces.jsonl. Resumable: skips question ids
already present in the output.
"""

import argparse
import base64
import json
import time
from pathlib import Path

from openai import OpenAI

BASE_URL = "http://localhost:8000/v1"
MODEL = "Qwen/Qwen3.5-27B"
TEMPERATURE = 0.0
SEED = 0
MAX_TOKENS = 16384
FRAMES_FALLBACK = 16
BQA_SUFFIX = " Only answer with a single word 'Yes' or 'No'."

SYSTEM_PROMPT = (
    "You are an expert at answering questions about video content.\n"
    "Watch the video carefully and answer the question.\n"
    "Think step by step before giving your final answer."
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    PROJECT_ROOT.parent / "data" / "stemo_ambig_candidates" / "all_questions.json"
)
DEFAULT_OUT = PROJECT_ROOT / "outputs_stemo" / "stemo_ambig_traces.jsonl"
STEMO_VIDEOS = PROJECT_ROOT.parent / "stemo" / "videos_h264"


def encode_video_b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def sample_frames_b64(path, n=FRAMES_FALLBACK):
    import io

    import decord
    from PIL import Image

    vr = decord.VideoReader(path)
    total = len(vr)
    idxs = [int(i * (total - 1) / max(n - 1, 1)) for i in range(n)]
    frames = vr.get_batch(idxs).asnumpy()
    out = []
    for fr in frames:
        buf = io.BytesIO()
        Image.fromarray(fr).save(buf, format="JPEG", quality=85)
        out.append(base64.b64encode(buf.getvalue()).decode())
    return out


def build_messages_video_b64(question, video_path):
    b64 = encode_video_b64(video_path)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "video_url", "video_url": {"url": f"data:video/mp4;base64,{b64}"}},
                {"type": "text", "text": question + BQA_SUFFIX},
            ],
        },
    ]


def build_messages_frame_list(question, video_path):
    frames = sample_frames_b64(video_path)
    content = [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b}"}}
        for b in frames
    ]
    content.append({"type": "text", "text": question + BQA_SUFFIX})
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]


def call_model(client, messages):
    resp = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        seed=SEED,
        extra_body={"chat_template_kwargs": {"enable_thinking": True}},
    )
    msg = resp.choices[0].message
    reasoning = getattr(msg, "reasoning", None) or getattr(msg, "reasoning_content", None)
    if not reasoning:
        raw = getattr(msg, "model_extra", None) or {}
        reasoning = raw.get("reasoning") or raw.get("reasoning_content") or ""
    return reasoning, msg.content or ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    args = ap.parse_args()

    payload = json.loads(args.input.read_text())
    questions = payload["questions"]
    if args.limit:
        questions = questions[: args.limit]
    print(f"Loaded {len(questions)} questions from {args.input.name}.")

    done = set()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        for line in args.out.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if not r.get("error"):
                done.add(r["id"])
    print(f"Already complete: {len(done)} / {len(questions)}.")

    client = OpenAI(base_url=BASE_URL, api_key="EMPTY")
    fout = args.out.open("a")

    total = len(questions)
    t_start = time.time()
    n_run = 0
    for idx, q in enumerate(questions, 1):
        if q["id"] in done:
            continue
        video_path = STEMO_VIDEOS / f"{q['video_id']}.mp4"
        if not video_path.exists():
            rec = {
                "id": q["id"],
                "video_id": q["video_id"],
                "question": q["question"],
                "category": q["category"],
                "subcategory": q["subcategory"],
                "k_group": q["k_group"],
                "thinking_trace": "",
                "final_answer": "",
                "thinking_char_count": 0,
                "elapsed_sec": 0.0,
                "error": f"video_not_found: {video_path}",
            }
            fout.write(json.dumps(rec) + "\n")
            fout.flush()
            print(f"[{idx}/{total}] MISSING VIDEO {q['video_id']}")
            continue

        t0 = time.time()
        thinking, answer, err, method_used = "", "", None, ""
        try:
            try:
                msgs = build_messages_video_b64(q["question"], str(video_path))
                thinking, answer = call_model(client, msgs)
                method_used = "base64_video"
            except Exception as e_video:
                if "Qwen3VLProcessor" in repr(e_video) or "BadRequestError" in repr(e_video):
                    msgs = build_messages_frame_list(q["question"], str(video_path))
                    thinking, answer = call_model(client, msgs)
                    method_used = "frame_list"
                else:
                    raise
        except Exception as e:
            err = repr(e)
            print(f"[{idx}/{total}] ERROR on {q['id']}: {err}")
        elapsed = time.time() - t0

        rec = {
            "id": q["id"],
            "video_id": q["video_id"],
            "question": q["question"],
            "category": q["category"],
            "subcategory": q["subcategory"],
            "k_group": q["k_group"],
            "thinking_trace": thinking,
            "final_answer": answer,
            "thinking_char_count": len(thinking),
            "elapsed_sec": round(elapsed, 2),
            "video_input_method": method_used,
            "error": err,
        }
        fout.write(json.dumps(rec) + "\n")
        fout.flush()
        n_run += 1
        avg = (time.time() - t_start) / max(n_run, 1)
        eta_min = avg * (total - len(done) - n_run) / 60
        print(
            f"[{idx}/{total}] {q['id']} k={q['k_group']} method={method_used} "
            f"think={len(thinking)}ch ans={answer[:30]!r} elapsed={elapsed:.1f}s "
            f"avg={avg:.1f}s eta={eta_min:.0f}min"
        )

    fout.close()
    print(f"Done. Wrote {args.out}")


if __name__ == "__main__":
    main()
