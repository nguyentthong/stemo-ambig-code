#!/usr/bin/env bash
# Shard STEMO-Ambig validation across N GPUs.
#
# Usage:
#   bash trace-pilot/scripts/run_stemo_ambig_eval_sharded.sh <tag> <adapter-path-or-empty>
#
# Splits the eval JSONL into NGPU chunks, runs N parallel inference jobs (one per GPU),
# then merges + runs preds_to_traces + Gemini judge.

set -euo pipefail

REPO=/home/thong/weride_project/weride/overthinking_hallu
TAG="$1"; ADAPTER="${2:-}"
NGPU="${NGPU:-8}"

ADAPTER_FLAG=""
[ -n "$ADAPTER" ] && ADAPTER_FLAG="--adapter $ADAPTER"
NOTHINK_FLAG=""
[ "${NO_THINKING:-0}" = "1" ] && NOTHINK_FLAG="--no-thinking"

OUT="$REPO/eval_runs/$TAG"
mkdir -p "$OUT/shards"

INPUT="$OUT/stemo_ambig_test.jsonl"
[ -f "$INPUT" ] || python "$REPO/trace-pilot/src/eval/prep_stemo_ambig_test.py" --out "$INPUT"

# 1) Split input into NGPU shards
echo "Splitting $INPUT across $NGPU shards..."
python - "$INPUT" "$OUT/shards" "$NGPU" <<'PY'
import sys, json
src, dst_dir, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
lines = [l for l in open(src).read().splitlines() if l.strip()]
print(f"  total {len(lines)} items")
for i in range(n):
    sub = lines[i::n]   # stride-n striping = even per-shard mix
    with open(f"{dst_dir}/shard_{i}.jsonl", "w") as f:
        f.write("\n".join(sub) + "\n")
    print(f"  shard {i}: {len(sub)} items")
PY

SYSTEM_PROMPT='You are an expert at answering questions about video content.
Watch the video carefully and answer the question.
Think step by step before giving your final answer.
If the question has multiple valid interpretations because of an ambiguous referent, enumerate each interpretation explicitly and provide an answer for each.'

# 2) Launch one inference per GPU in parallel
echo "Launching $NGPU parallel inference processes..."
PIDS=()
case "${RUNNER_FAMILY:-qwen}" in
  internvl) RUNNER_SCRIPT="$REPO/trace-pilot/src/eval/run_internvl_video.py" ;;
  *)        RUNNER_SCRIPT="$REPO/trace-pilot/src/eval/run_qwen_video.py" ;;
esac

for ((i=0; i<NGPU; i++)); do
  CUDA_VISIBLE_DEVICES=$i python "$RUNNER_SCRIPT" \
    --model-id "${MODEL_ID:-Qwen/Qwen3-VL-32B-Thinking}" \
    $ADAPTER_FLAG $NOTHINK_FLAG \
    --input  "$OUT/shards/shard_$i.jsonl" \
    --output "$OUT/shards/preds_$i.jsonl" \
    --system-prompt "$SYSTEM_PROMPT" \
    --max-new-tokens 2048 \
    > "$OUT/shards/log_$i.txt" 2>&1 &
  PIDS+=($!)
  echo "  shard $i -> pid ${PIDS[$i]} on GPU $i"
done

# 3) Wait for all
echo "Waiting on ${NGPU} processes..."
FAIL=0
for pid in "${PIDS[@]}"; do
  wait "$pid" || FAIL=$((FAIL+1))
done
if [ "$FAIL" -gt 0 ]; then
  echo "WARNING: $FAIL/$NGPU shards exited non-zero — see logs in $OUT/shards/"
fi

# 4) Merge predictions
PREDS="$OUT/stemo_ambig_predictions.jsonl"
cat "$OUT/shards/"preds_*.jsonl > "$PREDS"
echo "Merged predictions: $(wc -l < "$PREDS") lines -> $PREDS"

# 5) Convert to traces format
TRACES="$OUT/stemo_ambig_traces.jsonl"
python "$REPO/trace-pilot/src/eval/preds_to_traces.py" --preds "$PREDS" --out "$TRACES"

# 6) Gemini judge
JUDGMENTS="$OUT/stemo_ambig_judgments.jsonl"
METRICS="$OUT/stemo_ambig_metrics.json"
python "$REPO/trace-pilot/src/judge_stemo_traces.py" \
  --traces "$TRACES" --out "$JUDGMENTS" --metrics-out "$METRICS" \
  --workers 12

echo
echo "============ $TAG STEMO-Ambig results ============"
python -c "
import json
m = json.load(open('$METRICS'))
o = m['overall']
print(f\"  n={o['n']}\")
for k in ['enumeration_rate','single_commit_rate','single_commit_on_ambig_rate','interp_coverage','per_interp_accuracy_addressed','per_interp_accuracy_overall','strict_ambig_aware_accuracy','truncation_rate']:
    v = o.get(k)
    print(f\"  {k:<30} {v if v is None else f'{v:.3f}'}\")"
