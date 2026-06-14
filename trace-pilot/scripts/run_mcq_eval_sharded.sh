#!/usr/bin/env bash
# Generic sharded MCQ-style eval (VideoMME / MVBench / TempCompass / etc.)
#
# Usage:
#   GPUS=0,1,2,3 bash run_mcq_eval_sharded.sh <bench_name> <tag> <adapter-path-or-empty>
#     e.g.: GPUS=0,1,2,3 bash ... videomme base ""
#           GPUS=4,5,6,7 bash ... videomme sft  /path/to/adapter
#
# Inputs: data_v0/eval/<bench_name>.jsonl (must exist; see prep_*.py)
# Output: eval_runs/<tag>/<bench_name>_{predictions,metrics}.json

set -euo pipefail

REPO=/home/thong/weride_project/weride/overthinking_hallu
BENCH="$1"; TAG="$2"; ADAPTER="${3:-}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
KIND="${KIND:-mcq}"
GROUP_BY="${GROUP_BY:-duration}"

ADAPTER_FLAG=""
[ -n "$ADAPTER" ] && ADAPTER_FLAG="--adapter $ADAPTER"
NOTHINK_FLAG=""
[ "${NO_THINKING:-0}" = "1" ] && NOTHINK_FLAG="--no-thinking"

INPUT="$REPO/data_v0/eval/${BENCH}.jsonl"
OUT="$REPO/eval_runs/$TAG"
SHARDS_DIR="$OUT/shards_${BENCH}"
mkdir -p "$SHARDS_DIR"

IFS=',' read -ra GPU_ARR <<< "$GPUS"
NSHARDS=${#GPU_ARR[@]}

# 1) Split input
echo "[${BENCH}/${TAG}] splitting $INPUT into $NSHARDS shards"
python - "$INPUT" "$SHARDS_DIR" "$NSHARDS" <<'PY'
import sys
src, dst_dir, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
lines = [l for l in open(src).read().splitlines() if l.strip()]
print(f"  total {len(lines)} items")
for i in range(n):
    sub = lines[i::n]
    with open(f"{dst_dir}/shard_{i}.jsonl", "w") as f:
        f.write("\n".join(sub) + "\n")
    print(f"  shard {i}: {len(sub)} items")
PY

# 2) Launch one inference per GPU
echo "[${BENCH}/${TAG}] launching ${NSHARDS} processes across GPUs ${GPUS}"
PIDS=()
for ((i=0; i<NSHARDS; i++)); do
  GPU=${GPU_ARR[$i]}
  MCQ_RUNNER="$REPO/trace-pilot/src/eval/run_qwen_video.py"
  case "${RUNNER_FAMILY:-qwen}" in
    internvl_hf) MCQ_RUNNER="$REPO/trace-pilot/src/eval/run_internvl_hf_video.py" ;;
    internvl)    MCQ_RUNNER="$REPO/trace-pilot/src/eval/run_internvl_video.py" ;;
  esac
  CUDA_VISIBLE_DEVICES=$GPU python "$MCQ_RUNNER" \
    --model-id "${MODEL_ID:-Qwen/Qwen3-VL-32B-Thinking}" \
    $ADAPTER_FLAG $NOTHINK_FLAG \
    --input  "$SHARDS_DIR/shard_$i.jsonl" \
    --output "$SHARDS_DIR/preds_$i.jsonl" \
    --max-new-tokens 512 \
    > "$SHARDS_DIR/log_$i.txt" 2>&1 &
  PIDS+=($!)
  echo "  shard $i -> pid ${PIDS[$i]} on GPU $GPU"
done

# 3) Wait
FAIL=0
for pid in "${PIDS[@]}"; do
  wait "$pid" || FAIL=$((FAIL+1))
done
[ "$FAIL" -gt 0 ] && echo "[${BENCH}/${TAG}] WARNING: $FAIL/$NSHARDS shards failed; see $SHARDS_DIR/log_*.txt"

# 4) Merge + score
PREDS="$OUT/${BENCH}_predictions.jsonl"
METRICS="$OUT/${BENCH}_metrics.json"
cat "$SHARDS_DIR/"preds_*.jsonl > "$PREDS"
echo "[${BENCH}/${TAG}] merged $(wc -l < "$PREDS") predictions -> $PREDS"
python "$REPO/trace-pilot/src/eval/score.py" \
  --predictions "$PREDS" --kind "$KIND" --group-by "$GROUP_BY" --out "$METRICS"
