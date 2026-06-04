"""Convert VidHalluc (chaoyuli/VidHalluc) to standard eval JSONL.

VidHalluc has 3 subtasks (ACH, STH, TSH). Each annotation dict groups questions
by group id; each question has multiple (video_clip → yes/no) pairs.
We expand into one item per (question, clip) pair.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

PROMPT_TMPL = """{question}
Answer with Yes or No."""


def build_video_index(extracted_root: Path) -> dict[str, str]:
    """Map clip stem (e.g. 'OBb4013eIc8_clip_1') → full path."""
    idx = {}
    for p in extracted_root.rglob("*.mp4"):
        idx.setdefault(p.stem, str(p))
    return idx


def expand_ach_binary(data, vid_idx, subtask="ACH"):
    """{group_id: [{q, a:{clip:Yes/No}}]} → one item per (q, clip)."""
    out = []
    for group_id, items in data.items():
        for i, item in enumerate(items):
            q = item.get("q") or item.get("question")
            answers = item.get("a") or item.get("answers") or {}
            if not q or not answers:
                continue
            for clip_id, ans in answers.items():
                vp = vid_idx.get(clip_id)
                if not vp:
                    continue
                gold = (ans or "").strip().lower()
                if gold not in ("yes", "no"):
                    continue
                out.append({
                    "id": f"vidhalluc_{subtask}_g{group_id}_i{i}_{clip_id}",
                    "video_path": vp,
                    "prompt": PROMPT_TMPL.format(question=q),
                    "gold": "Yes" if gold == "yes" else "No",
                    "subtask": subtask,
                    "group": group_id,
                })
    return out


def expand_sth(data, vid_idx, subtask="STH"):
    """{video_id: {"Scene change": Yes/No, "Locations": text}} → one yes/no item per video."""
    out = []
    for video_id, entry in data.items():
        if not isinstance(entry, dict):
            continue
        scene_change = entry.get("Scene change")
        if scene_change is None:
            continue
        gold = (scene_change or "").strip().lower()
        if gold not in ("yes", "no"):
            continue
        vp = vid_idx.get(video_id)
        if not vp:
            continue
        out.append({
            "id": f"vidhalluc_{subtask}_{video_id}",
            "video_path": vp,
            "prompt": PROMPT_TMPL.format(question="Is there a scene change in this video?"),
            "gold": "Yes" if gold == "yes" else "No",
            "subtask": subtask,
        })
    return out


def expand_tsh(data, vid_idx, subtask="TSH"):
    """{group_id: {"video": clip_id, "Question": "...", "Correct Answer": "..."}}.

    Correct Answer values aren't necessarily yes/no — could be a clip name or a label.
    We only emit yes/no items here; non-binary items get a separate path if needed.
    """
    out = []
    for group_id, item in data.items():
        if not isinstance(item, dict):
            continue
        clip_id = item.get("video")
        q = item.get("Question")
        gold = (item.get("Correct Answer") or "").strip()
        if not (clip_id and q):
            continue
        vp = vid_idx.get(clip_id)
        if not vp:
            continue
        gold_lower = gold.lower()
        if gold_lower in ("yes", "no"):
            prompt = PROMPT_TMPL.format(question=q)
            out.append({
                "id": f"vidhalluc_{subtask}_{group_id}_{clip_id}",
                "video_path": vp,
                "prompt": prompt,
                "gold": "Yes" if gold_lower == "yes" else "No",
                "subtask": subtask,
                "group": group_id,
            })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default="/mnt/ceph3/ec/thong/vidhalluc_cache")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--subtasks", nargs="*",
                    default=["ach_binaryqa", "sth", "tsh"],
                    help="json files to include")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    cache = Path(args.cache_dir)
    extracted = cache / "videos_extracted"
    print(f"building video index under {extracted} ...")
    vid_idx = build_video_index(extracted)
    print(f"  indexed {len(vid_idx)} clips")

    out = []
    expanders = {
        "ach_binaryqa": expand_ach_binary,
        "sth": expand_sth,
        "tsh": expand_tsh,
    }
    for sub in args.subtasks:
        ann = cache / f"{sub}.json"
        if not ann.exists():
            print(f"  skip {sub}: no annotation")
            continue
        data = json.loads(ann.read_text())
        expander = expanders.get(sub)
        if expander is None:
            print(f"  skip {sub}: no expander defined")
            continue
        subtask_label = sub.upper().replace("_BINARYQA", "")
        items = expander(data, vid_idx, subtask=subtask_label)
        out.extend(items)
        print(f"  {sub}: {len(items)} items")
    if args.limit:
        out = out[: args.limit]
    print(f"prepared {len(out)} VidHalluc items")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(json.dumps(r) for r in out) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
