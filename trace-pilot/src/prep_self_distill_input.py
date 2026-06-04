"""Prepare input JSONL for self-distillation of ambig items.

For each ambig candidate in all_questions.json, build a prompt that includes
BOTH the original question AND a gold scaffold (the K interpretations + gold
answers). The model produces an enumeration response in ITS OWN VOICE while
guaranteed to align with gold (because gold is in the prompt).

This avoids importing alien CoT patterns from a different teacher (Gemini).

At training time, the scaffold is stripped — only the original question
appears in the user message. The model learns to produce enumeration unprompted.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SRC = REPO_ROOT / "data_v0" / "stemo_ambig_sft" / "ambig" / "all_questions.json"
DEFAULT_VIDEO_DIR = REPO_ROOT / "data_v0" / "stemo_ambig_sft" / "videos"


SCAFFOLD_TMPL = """Question: {question}

You have determined that this question is REFERENTIALLY AMBIGUOUS — the phrasing admits {k} valid interpretations, depending on which entity in the video the reader takes as the referent. For each interpretation, write a brief sentence stating the referent and giving your yes/no answer.

The K=#{k} interpretations and their gold answers (use these for your response — do NOT change the answers):
{interp_block}

Write your response in this exact format:
This question has {k} valid interpretations.
- "<referent description>" → Yes
- "<referent description>" → No
- ...

Do NOT include a separate reasoning block — go straight to the enumerated answer above. Keep it concise."""


def render_interp_block(interps):
    lines = []
    for i, ip in enumerate(interps, 1):
        lines.append(
            f"{i}. {ip['referent_description']} → {ip['predicted_answer'].strip().capitalize()}"
        )
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC)
    ap.add_argument("--video-dir", type=Path, default=DEFAULT_VIDEO_DIR)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    payload = json.loads(args.src.read_text())
    questions = payload["questions"]
    if args.limit:
        questions = questions[: args.limit]

    out = []
    missing = 0
    for q in questions:
        vp = args.video_dir / f"{q['video_id']}.mp4"
        if not vp.exists():
            missing += 1
            continue
        interps = q["interpretations"]
        if len(interps) < 2:
            continue
        out.append({
            "id": q["id"],
            "video_id": q["video_id"],
            "video_path": str(vp),
            "original_question": q["question"],
            "k": len(interps),
            "prompt": SCAFFOLD_TMPL.format(
                question=q["question"],
                k=len(interps),
                interp_block=render_interp_block(interps),
            ),
            "interpretations": interps,
            "k_group": q["k_group"],
        })
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(json.dumps(r) for r in out) + "\n")
    print(f"wrote {len(out)} self-distillation prompts ({missing} missing videos) -> {args.out}")


if __name__ == "__main__":
    main()
