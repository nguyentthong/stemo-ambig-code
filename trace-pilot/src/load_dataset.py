"""Sample 21 pilot examples from VidHalluc: 4 ACH_MCQ + 3 ACH_BQA + 7 TSH + 7 STH.

Reads the four annotation JSONs that live alongside the cloned vidhalluc/
repo (downloaded from HuggingFace huggingface.co/datasets/chaoyuli/VidHalluc),
flattens each into a list of {video_id, question, gt_answer}, samples with
seed=42, then searches for the matching .mp4 under vidhalluc/.

Prompts follow the official VidHalluc eval scripts (eval/inference/*.py) so
the model sees the same task framing the benchmark intends.
"""

import json
import random
import sys
from pathlib import Path

SEED = 42
SLICES_SPEC = [
    ("ACH_MCQ", 4),
    ("ACH_BQA", 3),
    ("TSH", 7),
    ("STH", 7),
]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VIDHALLUC_DIR = PROJECT_ROOT.parent / "vidhalluc"
OUT_PATH = PROJECT_ROOT / "data" / "pilot_examples.jsonl"

ANNO_FILES = {
    "ACH_MCQ": VIDHALLUC_DIR / "ach_mcq.json",
    "ACH_BQA": VIDHALLUC_DIR / "ach_binaryqa.json",
    "TSH": VIDHALLUC_DIR / "tsh.json",
    "STH": VIDHALLUC_DIR / "sth.json",
}

# Prompts copied from vidhalluc/eval/inference/*.py so the model sees what the
# benchmark expects.
STH_QUESTION = (
    "Watch the given video and determine if a scene change occurs. "
    "If no change occurs, respond: 'Scene change: No, Locations: None'. "
    "If there is a scene change, respond in the format: "
    "'Scene change: Yes, Locations: from [location1] to [location2].'"
)
TSH_SUFFIX = (
    "Sort these two actions in the order they occur in the video, and "
    "return which action happen before which one. If you only detect "
    "one action, return that action."
)
MCQ_SUFFIX = (
    " Please select the correct answer (one or more options), only return "
    "your answer(s). (e.g., ABCD)\nChoices:\n"
)
BQA_SUFFIX = " Only answer with a single word 'Yes' or 'No'."


def die(msg):
    print(msg)
    sys.exit(1)


for slice_name, path in ANNO_FILES.items():
    if not path.exists():
        die(f"Missing annotation file for {slice_name}: {path}")


def flatten_ach_mcq(obj):
    """{outer_idx: {clip_name: {Question, Choices, Correct Answer}}} -> flat list."""
    out = []
    for clips in obj.values():
        for clip_name, rec in clips.items():
            q = rec["Question"] + MCQ_SUFFIX
            for k, v in rec["Choices"].items():
                q += f"{k}. {v}\n"
            out.append({"video_id": clip_name, "question": q, "gt_answer": rec["Correct Answer"]})
    return out


def flatten_ach_bqa(obj):
    """{outer_idx: [{q, a: {clip_name: 'Yes'|'No'}}]} -> flat list."""
    out = []
    for qlist in obj.values():
        for entry in qlist:
            q_text = entry["q"] + BQA_SUFFIX
            for clip_name, ans in entry["a"].items():
                out.append({"video_id": clip_name, "question": q_text, "gt_answer": ans})
    return out


def flatten_tsh(obj):
    """{idx: {video, Question, Correct Answer}} -> flat list."""
    return [
        {
            "video_id": rec["video"],
            "question": rec["Question"] + TSH_SUFFIX,
            "gt_answer": rec["Correct Answer"],
        }
        for rec in obj.values()
    ]


def flatten_sth(obj):
    """{video_id: {Scene change, Locations}} -> flat list. Hardcoded question."""
    out = []
    for vid, rec in obj.items():
        sc = rec.get("Scene change", "")
        loc = rec.get("Locations", "")
        if sc.lower().startswith("y"):
            gt = f"Scene change: Yes, Locations: {loc}".rstrip(".")
        else:
            gt = "Scene change: No, Locations: None"
        out.append({"video_id": vid, "question": STH_QUESTION, "gt_answer": gt})
    return out


FLATTENERS = {
    "ACH_MCQ": flatten_ach_mcq,
    "ACH_BQA": flatten_ach_bqa,
    "TSH": flatten_tsh,
    "STH": flatten_sth,
}


SLICE_DIR_HINTS = {
    "ACH_MCQ": ("ACH",),
    "ACH_BQA": ("ACH",),
    "TSH": ("TSH",),
    "STH": ("STH",),
}


def find_video_path(video_id, slice_name):
    # Prefer the slice-specific folder so we don't cross-match when the same
    # video_id appears in multiple slice zips.
    hints = SLICE_DIR_HINTS.get(slice_name, ())
    for ext in (".mp4", ".mkv", ".avi", ".webm"):
        for hint in hints:
            for p in VIDHALLUC_DIR.rglob(f"{video_id}{ext}"):
                if f"/{hint}/" in str(p):
                    return str(p)
    for ext in (".mp4", ".mkv", ".avi", ".webm"):
        hits = list(VIDHALLUC_DIR.rglob(f"{video_id}{ext}"))
        if hits:
            return str(hits[0])
    return ""


def main():
    random.seed(SEED)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    missing_videos = 0
    for slice_name, n in SLICES_SPEC:
        obj = json.load(ANNO_FILES[slice_name].open())
        flat = FLATTENERS[slice_name](obj)
        print(f"[{slice_name}] {len(flat)} flat examples, sampling {n}")
        sampled = random.sample(flat, n)
        for rec in sampled:
            vp = find_video_path(rec["video_id"], slice_name)
            if not vp:
                missing_videos += 1
            rows.append({"slice": slice_name, "video_path": vp, **rec})

    with OUT_PATH.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    print(f"\nWrote {len(rows)} rows to {OUT_PATH}")
    if missing_videos:
        print(
            f"WARNING: {missing_videos}/{len(rows)} examples have empty video_path. "
            "Either video zips are still extracting or some clips are not yet present "
            f"under {VIDHALLUC_DIR}. Re-run after extraction completes to fill them in."
        )


if __name__ == "__main__":
    main()
