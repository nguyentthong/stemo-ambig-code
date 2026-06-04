"""Re-run the 4 long-trace examples at temperature=0.7 with 3 different seeds.

Purpose (from the research notes): test whether the looping behavior visible
in the long traces is a seed-variance artifact or a stable phenomenon. If
the loops produce different final answers across seeds, the looping is
destabilizing; if they always converge, the looping is wasteful but stable.

For each (video_id, prompt_variant) in TARGETS x both prompts, calls the
model 3 times with temperature=0.7 and seed in {0, 1, 2}. Writes one JSON
line per call to outputs/reruns.jsonl.

Skips (video_id, variant, seed) triples already present in reruns.jsonl
so the script is restartable.
"""

import json
import time
from pathlib import Path

from openai import OpenAI

# Reuse helpers from run_inference.py
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_inference import (  # noqa: E402
    BASE_URL, MODEL, MAX_TOKENS, build_messages_video_b64, call_model, load_prompts,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_PATH = PROJECT_ROOT / "data" / "pilot_examples.jsonl"
OUT_PATH = PROJECT_ROOT / "outputs" / "reruns.jsonl"

TEMPERATURE = 0.7
SEEDS = [0, 1, 2]
TARGETS = [
    "2VYZeOa6804_clip_4",
    "JNFUZz1bqmg_clip_10_7",
    "fSY-uG_uayI_clip_1_6",
    "oaOdliOVL6g_clip_2_4",
]


def call_with_seed(client, messages, seed):
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
    client = OpenAI(base_url=BASE_URL, api_key="EMPTY")
    prompts = load_prompts()

    examples = [json.loads(l) for l in EXAMPLES_PATH.read_text().splitlines() if l.strip()]
    by_id = {ex["video_id"]: ex for ex in examples}
    missing = [t for t in TARGETS if t not in by_id]
    if missing:
        sys.exit(f"target video_ids not found in {EXAMPLES_PATH.name}: {missing}")

    done = set()
    if OUT_PATH.exists():
        for line in OUT_PATH.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if not r.get("error"):
                done.add((r["video_id"], r["prompt_variant"], r["seed"]))

    plan = [
        (vid, variant, seed)
        for vid in TARGETS
        for variant in prompts
        for seed in SEEDS
        if (vid, variant, seed) not in done
    ]
    print(f"Already complete: {len(done)} reruns. To run: {len(plan)} calls.")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fout = OUT_PATH.open("a")

    for i, (vid, variant, seed) in enumerate(plan, 1):
        ex = by_id[vid]
        t0 = time.time()
        thinking, answer, err = "", "", None
        try:
            msgs = build_messages_video_b64(prompts[variant], ex["question"], ex["video_path"])
            thinking, answer = call_with_seed(client, msgs, seed)
        except Exception as e:
            err = repr(e)
            print(f"[{i}/{len(plan)}] ERROR {vid} {variant} seed={seed}: {err}")
        elapsed = time.time() - t0

        rec = {
            "video_id": vid,
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
            "error": err,
        }
        fout.write(json.dumps(rec) + "\n")
        fout.flush()
        print(
            f"[{i}/{len(plan)}] vid={vid} variant={variant} seed={seed} "
            f"think_chars={len(thinking)} elapsed={elapsed:.1f}s"
        )

    fout.close()
    print(f"Done. Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
