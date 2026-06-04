#!/usr/bin/env bash
# Validate a LoRA checkpoint on (1) STEMO-Ambig test + (2) regression benchmarks.
#
# Usage:
#   bash trace-pilot/scripts/validate_checkpoint.sh <tag> <adapter-path>
#       e.g.: validate_checkpoint.sh sft_step100 checkpoints/qwen3vl32b_stemo_ambig_lora_v1/checkpoint-100
#
#   bash trace-pilot/scripts/validate_checkpoint.sh base ""    # base model, no adapter
#
# Optional env:
#   STEMO_LIMIT  = N    (limit STEMO-Ambig test to N items; default = all 1056)
#   SKIP_STEMO   = 1    (skip STEMO ambig accuracy eval)
#   SKIP_REGR    = 1    (skip regression evals)

set -euo pipefail

REPO=/home/thong/weride_project/weride/overthinking_hallu
TAG="$1"; ADAPTER="${2:-}"
ADAPTER_FLAG=""
[ -n "$ADAPTER" ] && ADAPTER_FLAG="--adapter $ADAPTER"

OUT="$REPO/eval_runs/$TAG"
mkdir -p "$OUT"

SYSTEM_PROMPT='You are an expert at answering questions about video content.
Watch the video carefully and answer the question.
Think step by step before giving your final answer.
If the question has multiple valid interpretations because of an ambiguous referent, enumerate each interpretation explicitly and provide an answer for each.'

# --- (1) STEMO-Ambig accuracy eval ---
if [ "${SKIP_STEMO:-0}" != "1" ]; then
  echo "=== [1/2] STEMO-Ambig test inference + judge ==="
  STEMO_INPUT="$OUT/stemo_ambig_test.jsonl"
  STEMO_PREDS="$OUT/stemo_ambig_predictions.jsonl"
  STEMO_TRACES="$OUT/stemo_ambig_traces.jsonl"
  STEMO_JUDGMENTS="$OUT/stemo_ambig_judgments.jsonl"
  STEMO_METRICS="$OUT/stemo_ambig_metrics.json"

  [ -f "$STEMO_INPUT" ] || python "$REPO/trace-pilot/src/eval/prep_stemo_ambig_test.py" --out "$STEMO_INPUT" ${STEMO_LIMIT:+--limit $STEMO_LIMIT}

  python "$REPO/trace-pilot/src/eval/run_qwen_video.py" \
    $ADAPTER_FLAG \
    --input "$STEMO_INPUT" \
    --output "$STEMO_PREDS" \
    --system-prompt "$SYSTEM_PROMPT" \
    --max-new-tokens 4096

  python "$REPO/trace-pilot/src/eval/preds_to_traces.py" \
    --preds "$STEMO_PREDS" --out "$STEMO_TRACES"

  python "$REPO/trace-pilot/src/judge_stemo_traces.py" \
    --traces "$STEMO_TRACES" --out "$STEMO_JUDGMENTS" --metrics-out "$STEMO_METRICS" \
    --workers 12
fi

# --- (2) Regression evals ---
if [ "${SKIP_REGR:-0}" != "1" ]; then
  echo "=== [2/2] Regression evals (VideoMME, MVBench, TempCompass, VidHalluc) ==="
  bash "$REPO/trace-pilot/scripts/run_regression_evals.sh" "$TAG" "$ADAPTER"
fi

# --- Summary ---
echo
echo "============ $TAG SUMMARY ============"
if [ -f "$OUT/stemo_ambig_metrics.json" ]; then
  python -c "
import json
m = json.load(open('$OUT/stemo_ambig_metrics.json'))
o = m['overall']
print(f\"STEMO-Ambig (n={o['n']}):\")
print(f\"  enumeration_rate          : {o['enumeration_rate']:.3f}\")
print(f\"  single_commit_on_ambig    : {o['single_commit_on_ambig_rate']:.3f}\")
print(f\"  per_interp_acc_overall    : {o['per_interp_accuracy_overall']:.3f}\")
print(f\"  strict_ambig_aware_acc    : {o['strict_ambig_aware_accuracy']:.3f}\")
print(f\"  truncation_rate           : {o['truncation_rate']:.3f}\")
"
fi
for f in "$OUT"/{videomme,mvbench,tempcompass,vidhalluc}_metrics.json; do
  [ -f "$f" ] || continue
  python -c "
import json, os
m = json.load(open('$f'))
print(f\"{os.path.basename('$f').replace('_metrics.json',''):<14} n={m['n']:>4}  acc={m['accuracy']:.3f}\")"
done
