#!/usr/bin/env bash
# Wait for MVBench (already launched) to finish, then chain TempCompass + VidHalluc.
#
# Each stage runs base on GPUs 0-3 and SFT on GPUs 4-7 in parallel.

set -uo pipefail

REPO=/home/thong/weride_project/weride/overthinking_hallu
ADAPTER="$REPO/checkpoints/qwen3vl32b_stemo_ambig_lora_v1"

wait_metrics () {
  local BENCH=$1
  echo "[chain] waiting for $BENCH metrics on both base and sft_final ..."
  until [ -s "$REPO/eval_runs/base/${BENCH}_metrics.json" ] \
        && [ -s "$REPO/eval_runs/sft_final/${BENCH}_metrics.json" ]; do
    sleep 30
  done
  echo "[chain] $BENCH done."
}

# Stage 1: MVBench already launched externally; just wait.
wait_metrics mvbench

# Stage 2: TempCompass
echo "[chain] launching TempCompass..."
GPUS=0,1,2,3 GROUP_BY=dim bash "$REPO/trace-pilot/scripts/run_mcq_eval_sharded.sh" \
  tempcompass base "" > "$REPO/eval_runs/base/tempcompass_launch.log" 2>&1 &
P1=$!
GPUS=4,5,6,7 GROUP_BY=dim bash "$REPO/trace-pilot/scripts/run_mcq_eval_sharded.sh" \
  tempcompass sft_final "$ADAPTER" > "$REPO/eval_runs/sft_final/tempcompass_launch.log" 2>&1 &
P2=$!
echo "[chain] tempcompass pids: $P1 (base) $P2 (sft)"
wait $P1 $P2 || true
wait_metrics tempcompass

# Stage 3: VidHalluc (2k subsample, yes/no)
echo "[chain] launching VidHalluc-2k..."
# vidhalluc.jsonl input path differs (use the 2k subsample)
mkdir -p "$REPO/data_v0/eval"
cp "$REPO/data_v0/eval/vidhalluc_2k.jsonl" "$REPO/data_v0/eval/vidhalluc.jsonl.full_backup" 2>/dev/null || true
GPUS=0,1,2,3 KIND=yesno GROUP_BY=subtask bash "$REPO/trace-pilot/scripts/run_mcq_eval_sharded.sh" \
  vidhalluc_2k base "" > "$REPO/eval_runs/base/vidhalluc_launch.log" 2>&1 &
P1=$!
GPUS=4,5,6,7 KIND=yesno GROUP_BY=subtask bash "$REPO/trace-pilot/scripts/run_mcq_eval_sharded.sh" \
  vidhalluc_2k sft_final "$ADAPTER" > "$REPO/eval_runs/sft_final/vidhalluc_launch.log" 2>&1 &
P2=$!
echo "[chain] vidhalluc pids: $P1 (base) $P2 (sft)"
wait $P1 $P2 || true
wait_metrics vidhalluc_2k

# Final summary
echo
echo "============================ FINAL REGRESSION MATRIX ============================"
python - <<'PY'
import json, os
ROOT = "/home/thong/weride_project/weride/overthinking_hallu/eval_runs"
benches = ["videomme", "mvbench", "tempcompass", "vidhalluc_2k"]
print(f"{'benchmark':<14} {'base_acc':>10} {'sft_acc':>10} {'delta':>10}")
for b in benches:
    try:
        mb = json.load(open(f"{ROOT}/base/{b}_metrics.json"))
        ms = json.load(open(f"{ROOT}/sft_final/{b}_metrics.json"))
        delta = ms['accuracy'] - mb['accuracy']
        print(f"{b:<14} {mb['accuracy']:>10.3f} {ms['accuracy']:>10.3f} {delta:>+10.3f}")
    except Exception as e:
        print(f"{b:<14} (missing: {e})")
PY
