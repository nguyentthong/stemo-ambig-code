"""Run Qwen3.5-27B over the 21 v1 examples at temperature=0.7 with 3 seeds.

Writes one JSON line per call to outputs/traces_temp07.jsonl. Skips any
(video_id, prompt_variant, seed) triple already present so re-runs only
fill in new work. Pauses for human inspection after the first example
(6 calls = 2 prompts x 3 seeds) on a fresh run.
"""

import argparse
import base64
import json
import time
from pathlib import Path

from openai import OpenAI

BASE_URL = "http://localhost:8000/v1"
MODEL = "Qwen/Qwen3.5-27B"
TEMPERATURE = 0.7
SEEDS = [0, 1, 2]
MAX_TOKENS = 4096
FRAMES_FALLBACK = 16

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXAMPLES = PROJECT_ROOT / "data" / "pilot_examples.jsonl"
PROMPTS_PATH     = PROJECT_ROOT / "prompts" / "system.txt"
DEFAULT_OUT      = PROJECT_ROOT / "outputs" / "traces_temp07.jsonl"


def load_prompts():
    text = PROMPTS_PATH.read_text()
    a = text.split("###PROMPT_A_PLAIN###")[1].split("###PROMPT_B_CITED###")[0].strip()
    b = text.split("###PROMPT_B_CITED###")[1].strip()
    return {"A_plain": a, "B_cited": b}


def encode_video_b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def sample_frames_b64(path, n=FRAMES_FALLBACK):
    import decord
    vr = decord.VideoReader(path)
    total = len(vr)
    idxs = [int(i * (total - 1) / max(n - 1, 1)) for i in range(n)]
    frames = vr.get_batch(idxs).asnumpy()
    import io
    from PIL import Image
    out = []
    for fr in frames:
        buf = io.BytesIO()
        Image.fromarray(fr).save(buf, format="JPEG", quality=85)
        out.append(base64.b64encode(buf.getvalue()).decode())
    return out


def build_messages_video_b64(system_prompt, question, video_path):
    b64 = encode_video_b64(video_path)
    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "video_url", "video_url": {"url": f"data:video/mp4;base64,{b64}"}},
                {"type": "text", "text": question},
            ],
        },
    ]


def build_messages_frame_list(system_prompt, question, video_path):
    frames = sample_frames_b64(video_path)
    content = [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b}"}}
        for b in frames
    ]
    content.append({"type": "text", "text": question})
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content},
    ]


def call_model(client, messages, seed):
    resp = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        seed=seed,
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
    ap.add_argument("--examples", type=Path, default=DEFAULT_EXAMPLES)
    ap.add_argument("--out",      type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    examples_path = args.examples
    out_path = args.out

    client = OpenAI(base_url=BASE_URL, api_key="EMPTY")
    prompts = load_prompts()

    examples = [json.loads(l) for l in examples_path.read_text().splitlines() if l.strip()]
    print(f"Loaded {len(examples)} examples from {examples_path.name}.")

    # Skip (video_id, prompt_variant, seed) triples already present in out_path
    # so re-runs only fill in new work.
    done = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if not r.get("error"):
                done.add((r["video_id"], r["prompt_variant"], r["seed"]))
    total_calls = len(examples) * len(prompts) * len(SEEDS)
    print(f"Already complete: {len(done)} calls. To run: {total_calls - len(done)} of {total_calls}.")

    chosen_method = "base64_video" if done else None
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fout = out_path.open("a")

    call_idx = 0
    new_examples_seen = 0
    for ex_idx, ex in enumerate(examples):
        had_new_work = False
        for variant, sys_prompt in prompts.items():
            for seed in SEEDS:
                if (ex["video_id"], variant, seed) in done:
                    continue
                had_new_work = True
                call_idx += 1
                t0 = time.time()
                method_used = None
                thinking, answer, err = "", "", None

                try:
                    if chosen_method in (None, "base64_video"):
                        try:
                            msgs = build_messages_video_b64(sys_prompt, ex["question"], ex["video_path"])
                            thinking, answer = call_model(client, msgs, seed)
                            method_used = "base64_video"
                            if chosen_method is None:
                                chosen_method = "base64_video"
                                print("Using video-input method: base64_video")
                        except Exception as e:
                            if chosen_method is None:
                                print(f"base64_video failed ({e}); trying frame_list ...")
                                msgs = build_messages_frame_list(sys_prompt, ex["question"], ex["video_path"])
                                thinking, answer = call_model(client, msgs, seed)
                                method_used = "frame_list"
                                chosen_method = "frame_list"
                                print("Using video-input method: frame_list")
                            else:
                                raise
                    else:
                        msgs = build_messages_frame_list(sys_prompt, ex["question"], ex["video_path"])
                        thinking, answer = call_model(client, msgs, seed)
                        method_used = "frame_list"
                except Exception as e:
                    err = repr(e)
                    print(f"[{call_idx}] ERROR on {ex['video_id']} ({variant}, seed={seed}): {err}")

                elapsed = time.time() - t0
                rec = {
                    "video_id": ex["video_id"],
                    "slice": ex["slice"],
                    "question": ex["question"],
                    "gt_answer": ex["gt_answer"],
                    "prompt_variant": variant,
                    "seed": seed,
                    "temperature": TEMPERATURE,
                    "thinking_trace": thinking,
                    "final_answer": answer,
                    "thinking_char_count": len(thinking),
                    "elapsed_sec": round(elapsed, 2),
                    "video_input_method": method_used or "",
                    "error": err,
                }
                fout.write(json.dumps(rec) + "\n")
                fout.flush()

                print(
                    f"[{call_idx:3d}/{total_calls - len(done)}] slice={ex['slice']} "
                    f"vid={ex['video_id']} variant={variant} seed={seed} "
                    f"think_chars={len(thinking)} elapsed={elapsed:.1f}s"
                )

        if had_new_work:
            new_examples_seen += 1
        if new_examples_seen == 1 and had_new_work and not done:
            print(
                f"\nFirst example complete. 6 traces written to {out_path.name}. "
                "Inspect to verify seed variation is present (the 3 seeds should "
                "produce visibly different thinking traces, even if the final "
                "answers agree). Press Enter to continue, or Ctrl-C to abort."
            )
            try:
                input()
            except EOFError:
                print("(non-interactive stdin; auto-continuing)")

    fout.close()
    print(f"Done. Wrote {out_path}")


if __name__ == "__main__":
    main()
