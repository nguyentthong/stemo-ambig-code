#!/usr/bin/env bash
# IAA open-weight launcher.
# Usage: iaa_open_launcher.sh <model_id> <adapter_or_NONE> <out_tag> <gpu_list>
# Example: iaa_open_launcher.sh "Qwen/Qwen3-VL-32B-Thinking" NONE qwen3vl32b_iaa_base "0,1,2,3,4,5,6,7"
set -uo pipefail
REPO=/home/thong/weride_project/weride/overthinking_hallu
cd $REPO

MODEL_ID="${1:?model_id required}"
ADAPTER="${2:?adapter or NONE required}"
OUT_TAG="${3:?out_tag required}"
GPU_LIST="${4:-0,1,2,3,4,5,6,7}"

IFS=',' read -ra GPUS <<< "$GPU_LIST"
NSHARDS=${#GPUS[@]}
OUT_DIR="$REPO/eval_runs/$OUT_TAG"
SHARD_DIR="$OUT_DIR/iaa_shards"
mkdir -p "$SHARD_DIR"

echo "[iaa-open] model=$MODEL_ID adapter=$ADAPTER out=$OUT_TAG nshards=$NSHARDS"

# Build per-shard input JSONLs from gold
python3 - <<PY
import json, os
from pathlib import Path
REPO = Path("$REPO")
NSHARDS = $NSHARDS
SHARD_DIR = Path("$SHARD_DIR")
SHARD_DIR.mkdir(parents=True, exist_ok=True)
data = json.load(open(REPO/"data_v0/stemo_ambig_candidates/all_questions.json"))
qs = data["questions"]
videos_dir = REPO/"stemo"/"videos_h264"
rows = []
missing = 0
for q in qs:
    vp = videos_dir / f"{q['video_id']}.mp4"
    if not vp.exists():
        missing += 1
        continue
    rows.append({
        "id": q["id"],
        "video_id": q["video_id"],
        "video_path": str(vp),
        "question": q["question"],
        "K": len(q["interpretations"]),
        "interpretations": [
            {"referent_description": ip["referent_description"],
             "predicted_answer": ip["predicted_answer"],
             "disambiguated_question": ip.get("disambiguated_question", "")}
            for ip in q["interpretations"]
        ],
    })
print(f"[iaa-open shards] {len(rows)} items ({missing} missing videos)")
for i in range(NSHARDS):
    shard = rows[i::NSHARDS]
    p = SHARD_DIR / f"shard_{i}.jsonl"
    with p.open("w") as f:
        for r in shard:
            f.write(json.dumps(r) + "\n")
    print(f"  shard_{i}: {len(shard)} items -> {p}")
PY

# Launch one process per GPU
PIDS=()
for i in "${!GPUS[@]}"; do
  GPU="${GPUS[$i]}"
  ADAPTER_ARG=""
  [ "$ADAPTER" != "NONE" ] && ADAPTER_ARG="--adapter $ADAPTER"
  CUDA_VISIBLE_DEVICES="$GPU" python -u $REPO/trace-pilot/src/iaa/run_iaa_open.py \
    --model-id "$MODEL_ID" $ADAPTER_ARG \
    --input "$SHARD_DIR/shard_${i}.jsonl" \
    --output "$SHARD_DIR/preds_${i}.jsonl" \
    > "$REPO/tmp/iaa_${OUT_TAG}_shard${i}.log" 2>&1 &
  PIDS+=($!)
  echo "[iaa-open] shard $i on GPU $GPU pid=${PIDS[-1]}"
done

# Wait for all
for p in "${PIDS[@]}"; do wait $p || true; done
echo "[iaa-open] all shards done $(date)"

# Merge + compute metrics
python3 - <<PY
import json
from pathlib import Path
REPO = Path("$REPO")
OUT_DIR = REPO/"eval_runs/$OUT_TAG"
SHARD_DIR = OUT_DIR/"iaa_shards"
merged = OUT_DIR/"iaa_predictions.jsonl"
records = []
with merged.open("w") as fout:
    for p in sorted(SHARD_DIR.glob("preds_*.jsonl")):
        for l in p.read_text().splitlines():
            if l.strip():
                rec = json.loads(l)
                records.append(rec)
                fout.write(l + "\n")
print(f"[iaa-open] merged {len(records)} records -> {merged}")

# Compute metrics inline
valid = [r for r in records if r.get("score") and not r.get("error")]
n = len(valid)
def safe_mean(xs): return sum(xs)/len(xs) if xs else 0.0
metrics = {
    "n": n,
    "iaa": safe_mean([r["score"]["iaa_score"] for r in valid]),
    "strict_K": safe_mean([1.0 if r["score"]["strict_K_correct"] else 0.0 for r in valid]),
    "aar_loose": safe_mean([1.0 if r["score"]["aar_loose_correct"] else 0.0 for r in valid]),
    "clarification_rate": safe_mean([1.0 if r["classification"]["category"] in {"clarified_scope","clarified_vague"} else 0.0 for r in valid]),
    "recognition_no_recall": safe_mean([1.0 if r["classification"]["category"] == "clarified_vague" else 0.0 for r in valid]),
    "judge_version": "gemini-3-flash-preview@iaa-v1.0",
    "n_errored": sum(1 for r in records if r.get("error")),
}
clar = [r for r in valid if r["classification"]["category"] in {"clarified_scope","clarified_vague"}]
metrics["follow_through_rate"] = safe_mean([1.0 if r["score"]["follow_through_correct"] else 0.0 for r in clar]) if clar else None

# Per-K
by_K = {}
for r in valid:
    K = r["K"]
    b = "2" if K == 2 else "3" if K == 3 else "4-6" if 4 <= K <= 6 else "7+"
    by_K.setdefault(b, []).append(r)
metrics["per_K"] = {}
for k, items in sorted(by_K.items()):
    m = len(items)
    metrics["per_K"][k] = {
        "n": m,
        "iaa": safe_mean([r["score"]["iaa_score"] for r in items]),
        "strict_K": safe_mean([1.0 if r["score"]["strict_K_correct"] else 0.0 for r in items]),
        "aar_loose": safe_mean([1.0 if r["score"]["aar_loose_correct"] else 0.0 for r in items]),
        "clarification_rate": safe_mean([1.0 if r["classification"]["category"] in {"clarified_scope","clarified_vague"} else 0.0 for r in items]),
    }

out_mp = OUT_DIR/"iaa_metrics.json"
out_mp.write_text(json.dumps(metrics, indent=2))
print(f"[iaa-open] wrote {out_mp}")
print(json.dumps({k:v for k,v in metrics.items() if k != "per_K"}, indent=2))
PY
