#!/usr/bin/env bash
# Run the 4 regression evals on a Qwen3-VL model (base or LoRA-trained).
#
# Usage:
#   bash trace-pilot/scripts/run_regression_evals.sh base                            # base model
#   bash trace-pilot/scripts/run_regression_evals.sh sft  <adapter-path>             # LoRA adapter
#   bash trace-pilot/scripts/run_regression_evals.sh ...  --limit 50                 # smoke test
#
# Outputs land in eval_runs/<tag>/{benchmark}_predictions.jsonl and {benchmark}_metrics.json.

set -euo pipefail

REPO=/home/thong/weride_project/weride/overthinking_hallu
TAG="$1"; shift
ADAPTER_FLAG=""
if [ "$TAG" != "base" ]; then
  ADAPTER="$1"; shift
  ADAPTER_FLAG="--adapter $ADAPTER"
fi
EXTRA_ARGS="$@"

OUT_DIR="$REPO/eval_runs/$TAG"
DATA_DIR="$REPO/data_v0/eval"
mkdir -p "$OUT_DIR" "$DATA_DIR"

# --- prep all 4 benchmarks once (idempotent) ---
[ -f "$DATA_DIR/videomme.jsonl" ]   || python "$REPO/trace-pilot/src/eval/prep_videomme.py"   --out "$DATA_DIR/videomme.jsonl"
[ -f "$DATA_DIR/mvbench.jsonl" ]    || python "$REPO/trace-pilot/src/eval/prep_mvbench.py"    --out "$DATA_DIR/mvbench.jsonl"
[ -f "$DATA_DIR/tempcompass.jsonl" ]|| python "$REPO/trace-pilot/src/eval/prep_tempcompass.py" --out "$DATA_DIR/tempcompass.jsonl"
[ -f "$DATA_DIR/vidhalluc.jsonl" ]  || python "$REPO/trace-pilot/src/eval/prep_vidhalluc.py"   --out "$DATA_DIR/vidhalluc.jsonl"

run_one() {
  local NAME=$1 KIND=$2 GROUP=$3
  local INPUT="$DATA_DIR/$NAME.jsonl"
  local PREDS="$OUT_DIR/${NAME}_predictions.jsonl"
  local METRICS="$OUT_DIR/${NAME}_metrics.json"
  echo "=== $NAME ==="
  python "$REPO/trace-pilot/src/eval/run_qwen_video.py" \
    $ADAPTER_FLAG --input "$INPUT" --output "$PREDS" $EXTRA_ARGS
  python "$REPO/trace-pilot/src/eval/score.py" \
    --predictions "$PREDS" --kind "$KIND" --group-by "$GROUP" --out "$METRICS"
}

run_one videomme    mcq   duration
run_one mvbench     mcq   task
run_one tempcompass mcq   dim
run_one vidhalluc   yesno subtask

echo
echo "Done. Summary metrics:"
for f in "$OUT_DIR"/*_metrics.json; do
  echo "--- $(basename "$f") ---"
  python -c "import json; m=json.load(open('$f')); print(f\"  n={m['n']} acc={m['accuracy']:.3f}\")"
done
