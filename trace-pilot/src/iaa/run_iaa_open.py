"""IAA multi-turn runner for open-weight Qwen3-VL / InternVL models.

Mirrors the closed-API IAA runner but uses local model inference.
Designed to run per-shard (single GPU); a wrapper script merges shards.

Input shard JSONL line schema:
  {id, video_path, K, interpretations: [{referent_description, predicted_answer, ...}], question}

Output: same as run_iaa_closed.py — full multi-turn transcript + classification + score.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "trace-pilot" / "src"))

from iaa.sub_judge import classify_turn1, extract_yesno, PROTOCOL_VERSION  # noqa: E402

# IAA system prompt — must match closed-API runner verbatim
SYSTEM_PROMPT = """You are an expert at answering questions about video content.

You may respond in any of three ways:
(1) If the question is unambiguous, give a single yes/no answer with a brief explanation.
(2) If the question has multiple valid interpretations because of an ambiguous referent, you may enumerate each interpretation and provide an answer for each, using the format:
    "<referent description 1>" -> Yes/No
    "<referent description 2>" -> Yes/No
    You may group interpretations that share the same answer, as long as the grouping identifies exactly which interpretations it covers (for example, "every attempt after the first").
(3) Alternatively, you may ask a clarifying question that identifies the ambiguous noun phrase (e.g., "Which boy do you mean?"). If you do, the asker will respond with a specific referent, and you must then answer for that referent.

Think step by step before responding."""

MAX_TOKENS = 2048
FRAMES = 16
TURN_CAP = 3


def select_referent_index(item_id: str, K: int) -> int:
    h = int(hashlib.md5(item_id.encode()).hexdigest(), 16)
    return h % K


def build_disambig_utterance(interpretations: list, selected_idx: int) -> str:
    desc = interpretations[selected_idx].get("referent_description", "this referent")
    return f"I am asking specifically about {desc}."


def _decode_video(path, num_frames=FRAMES):
    import decord  # noqa: WPS433
    vr = decord.VideoReader(str(path))
    total = len(vr)
    n = min(num_frames, total)
    idxs = [int(i * (total - 1) / max(n - 1, 1)) for i in range(n)]
    return vr.get_batch(idxs).asnumpy()


def load_model(model_id: str, adapter: str | None, dtype="bfloat16"):
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        torch_dtype=getattr(torch, dtype),
        trust_remote_code=True,
        device_map="auto",
    )
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
        model = model.merge_and_unload()
        print(f"loaded LoRA adapter from {adapter}", flush=True)
    model.eval()
    return model, processor


def build_turn1_messages(question: str):
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [
            {"type": "video"},
            {"type": "text", "text": question},
        ]},
    ]


def append_followup(messages: list, prior_response: str, followup_text: str):
    """Append an assistant turn for the previous model response, then a new user turn."""
    return messages + [
        {"role": "assistant", "content": [{"type": "text", "text": prior_response}]},
        {"role": "user", "content": [{"type": "text", "text": followup_text}]},
    ]


@torch.inference_mode()
def _generate(model, processor, messages, frames, max_new_tokens, enable_thinking=None):
    tmpl_kwargs = {"add_generation_prompt": True, "tokenize": False}
    if enable_thinking is not None:
        tmpl_kwargs["enable_thinking"] = enable_thinking
    try:
        prompt_text = processor.apply_chat_template(messages, **tmpl_kwargs)
    except TypeError:
        tmpl_kwargs.pop("enable_thinking", None)
        prompt_text = processor.apply_chat_template(messages, **tmpl_kwargs)
    inputs = processor(text=[prompt_text], videos=[frames], return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    prompt_len = inputs["input_ids"].shape[1]
    text = processor.tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True).strip()
    return text


def score_item(gold_interps: list, classification: dict,
               turn2_response: str | None, turn3_response: str | None,
               selected_idx: int, question: str) -> dict:
    """Mirror run_iaa_closed.py score_item; copied here to avoid import cycle."""
    K = len(gold_interps)
    cat = classification["category"]
    selected_ref = gold_interps[selected_idx]
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
        matches = classification.get("enumerated_matches", []) or []
        m_map = {(m.get("referent_description") or "").strip().lower():
                 (m.get("decision") or "").strip().lower() for m in matches}
        gold_map = {(ip.get("referent_description") or "").strip().lower():
                    (ip.get("predicted_answer") or "").strip().lower() for ip in gold_interps}
        all_ok = len(matches) >= K
        if all_ok:
            for gold_desc, gold_a in gold_map.items():
                found = False
                for mk, mv in m_map.items():
                    if gold_desc in mk or mk in gold_desc or (
                        len(gold_desc) > 5 and len(mk) > 5 and
                        (gold_desc[:8] == mk[:8] or gold_desc[-8:] == mk[-8:])
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

    if cat in {"single_commit", "refused"}:
        return result

    if cat in {"clarified_scope", "clarified_vague"} and turn2_response:
        ext = extract_yesno(question, selected_ref["referent_description"],
                            selected_ref.get("disambiguated_question", ""), turn2_response)
        d = ext["decision"]
        result["turn2_decision"] = d
        if d in {"yes", "no"}:
            correct = (d == gold_ans)
            result["follow_through_correct"] = correct
            base = 1.0 if correct else 0.0
            if cat == "clarified_vague":
                base *= 0.5
            result["iaa_score"] = base
            return result
        if turn3_response:
            ext3 = extract_yesno(question, selected_ref["referent_description"],
                                  selected_ref.get("disambiguated_question", ""), turn3_response)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--adapter", default=None, help="Optional LoRA adapter path")
    ap.add_argument("--input", type=Path, required=True,
                    help="Shard JSONL with {id, video_path, question, K, interpretations[]}")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--max-new-tokens", type=int, default=MAX_TOKENS)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--no-thinking", action="store_true")
    args = ap.parse_args()
    enable_thinking = False if args.no_thinking else None

    args.output.parent.mkdir(parents=True, exist_ok=True)
    done_ids = set()
    if args.resume and args.output.exists():
        for line in args.output.read_text().splitlines():
            if line.strip():
                done_ids.add(json.loads(line)["id"])

    rows = [json.loads(l) for l in args.input.read_text().splitlines() if l.strip()]
    pending = [r for r in rows if r["id"] not in done_ids]
    if args.limit:
        pending = pending[: args.limit]
    print(f"[IAA-open] total={len(rows)} done={len(done_ids)} pending={len(pending)}", flush=True)
    if not pending:
        return

    print(f"[IAA-open] loading {args.model_id} adapter={args.adapter}", flush=True)
    model, processor = load_model(args.model_id, args.adapter)
    print("[IAA-open] model loaded.", flush=True)

    with args.output.open("a") as fout:
        t0 = time.time()
        for i, r in enumerate(pending, 1):
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
            }
            t_item = time.time()
            try:
                frames = _decode_video(r["video_path"])
                msgs = build_turn1_messages(r["question"])
                r1 = _generate(model, processor, msgs, frames, args.max_new_tokens, enable_thinking)
                rec["turn_1_response"] = r1
                cls = classify_turn1(r["question"], r["interpretations"], r1)
                rec["classification"] = cls
                if cls["category"] in {"clarified_scope", "clarified_vague"}:
                    disambig = build_disambig_utterance(r["interpretations"], selected_idx)
                    msgs2 = append_followup(msgs, r1, disambig)
                    r2 = _generate(model, processor, msgs2, frames, args.max_new_tokens, enable_thinking)
                    rec["turn_2_response"] = r2
                    cls2 = classify_turn1(r["question"], r["interpretations"], r2)
                    if cls2["category"] in {"clarified_scope", "clarified_vague"}:
                        msgs3 = append_followup(msgs2, r2, disambig)
                        r3 = _generate(model, processor, msgs3, frames, args.max_new_tokens, enable_thinking)
                        rec["turn_3_response"] = r3
                rec["score"] = score_item(r["interpretations"], cls,
                                          rec["turn_2_response"], rec["turn_3_response"],
                                          selected_idx, r["question"])
            except Exception as e:
                rec["error"] = repr(e)[:500]
            rec["elapsed_sec"] = round(time.time() - t_item, 2)
            fout.write(json.dumps(rec) + "\n")
            fout.flush()
            if i % 5 == 0 or i == len(pending):
                rate = i / max(time.time() - t0, 0.01)
                eta_min = (len(pending) - i) / max(rate, 0.01) / 60
                print(f"[IAA-open] [{i}/{len(pending)}] rate={rate:.2f}/s eta={eta_min:.1f}min", flush=True)


if __name__ == "__main__":
    main()
