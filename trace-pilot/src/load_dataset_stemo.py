"""Sample 21 STEMO pilot examples (binary Yes/No), seed=42.

STEMO source: ../stemo/questions/*.json — each file has a list of binary
questions plus per-question sub-questions and sub-answers. We flatten every
(file, question_idx) into a record, then random-sample 21.

Output schema matches load_dataset.py / pilot_examples.jsonl so run_inference.py
can consume it via --examples.
"""

import json
import random
from pathlib import Path

SEED = 42
N_SAMPLE = 21

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STEMO_DIR    = PROJECT_ROOT.parent / "stemo"
QUESTIONS    = STEMO_DIR / "questions"
VIDEOS       = STEMO_DIR / "videos"
OUT_PATH     = PROJECT_ROOT / "data" / "pilot_examples_stemo.jsonl"

BQA_SUFFIX = " Only answer with a single word 'Yes' or 'No'."


def flatten():
    out = []
    for qf in sorted(QUESTIONS.glob("*.json")):
        d = json.loads(qf.read_text())
        vname = d["video_name"]
        vpath = VIDEOS / vname
        if not vpath.exists():
            continue
        base = qf.stem  # sample_0016_NtTb-Cw6JVs
        for i, (q, a) in enumerate(zip(d["questions"], d["answers"])):
            out.append({
                "slice": "STEMO",
                "video_path": str(vpath),
                "video_id": f"{base}__q{i}",
                "question": q + BQA_SUFFIX,
                "gt_answer": a,
            })
    return out


def main():
    random.seed(SEED)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    flat = flatten()
    print(f"[STEMO] {len(flat)} flat examples, sampling {N_SAMPLE}")
    sampled = random.sample(flat, N_SAMPLE)

    with OUT_PATH.open("w") as f:
        for r in sampled:
            f.write(json.dumps(r) + "\n")
    print(f"\nWrote {len(sampled)} rows to {OUT_PATH}")


if __name__ == "__main__":
    main()
