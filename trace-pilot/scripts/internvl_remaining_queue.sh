#!/usr/bin/env bash
# Single sequential controller for all remaining InternVL work — eliminates the
# multi-queue race. Each step waits for GPUs to drain, runs to completion, then
# the next. Order: 38B v4 eval rerun -> 8B FFT -> v3 (8B) -> v3 (38B).
# v5 for both is handled by the still-running ladder queue; this naturally waits
# behind it (wait_gpus also matches v5_offline_chain.sh / chain_v4.sh).
set -uo pipefail
REPO=/home/thong/weride_project/weride/overthinking_hallu
cd "$REPO"
LOG="$REPO/tmp/internvl_remaining_queue.log"
log(){ echo "[$(date -u +%FT%TZ)] $*" >> "$LOG"; }
wait_gpus(){
  while pgrep -f "run_internvl_hf_video|train_sft|accelerate.commands|run_qwen_video|chain_v4.sh|v5_offline_chain.sh" >/dev/null 2>&1; do
    sleep 180
  done
}

log "remaining queue START (38B-eval -> 8B-FFT -> v3-8B -> v3-38B)"

# 1) 38B v4 eval rerun (adapter exists; fixed _no_split runner)
if [ ! -f "$REPO/eval_runs/internvl38b_hf_v4/stemo_ambig_metrics.json" ]; then
  wait_gpus
  log "=== 38B v4 eval rerun start ==="
  MID="OpenGVLab/InternVL3_5-38B-HF"
  AD="$REPO/checkpoints/internvl38b_hf_stemo_ambig_lora_v4"
  MODEL_ID="$MID" NGPU=8 RUNNER_FAMILY=internvl_hf bash "$REPO/trace-pilot/scripts/run_stemo_ambig_eval_sharded.sh" internvl38b_hf_v4_base "" >> "$LOG" 2>&1 || true
  MODEL_ID="$MID" NGPU=8 RUNNER_FAMILY=internvl_hf bash "$REPO/trace-pilot/scripts/run_stemo_ambig_eval_sharded.sh" internvl38b_hf_v4 "$AD" >> "$LOG" 2>&1 || true
  for B in videomme mvbench; do
    GB=duration; [ "$B" = mvbench ] && GB=task
    MODEL_ID="$MID" RUNNER_FAMILY=internvl_hf GROUP_BY=$GB GPUS=0,1,2,3 bash "$REPO/trace-pilot/scripts/run_mcq_eval_sharded.sh" "$B" internvl38b_hf_v4_base "" >> "$LOG" 2>&1 &
    MODEL_ID="$MID" RUNNER_FAMILY=internvl_hf GROUP_BY=$GB GPUS=4,5,6,7 bash "$REPO/trace-pilot/scripts/run_mcq_eval_sharded.sh" "$B" internvl38b_hf_v4 "$AD" >> "$LOG" 2>&1 &
    wait
  done
  log "=== 38B v4 eval rerun done ==="
fi

# 2) 8B FFT
if [ ! -f "$REPO/eval_runs/internvl8b_hf_fft_v4/stemo_ambig_metrics.json" ]; then
  wait_gpus
  log "=== 8B FFT start ==="
  bash "$REPO/trace-pilot/scripts/internvl_8b_fft.sh" >> "$REPO/tmp/internvl_8b_fft.log" 2>&1 || true
  log "=== 8B FFT done ==="
fi

# 3) v3 for both sizes
for TAG in internvl8b_hf internvl38b_hf; do
  if [ ! -f "$REPO/eval_runs/${TAG}_v3/stemo_ambig_metrics.json" ]; then
    wait_gpus
    log "=== $TAG v3 start ==="
    bash "$REPO/trace-pilot/scripts/internvl_v3.sh" "$TAG" >> "$REPO/tmp/v3_${TAG}.log" 2>&1 || true
    log "=== $TAG v3 done ==="
  fi
done

log "remaining queue END"
python "$REPO/tools/dashboard.py" >/dev/null 2>&1 || true
cp "$REPO/STATUS.md" "$REPO/dashboard_repo/STATUS.md" 2>/dev/null || true
cd "$REPO/dashboard_repo" && git add STATUS.md && \
  git -c user.name="dashboard" -c user.email="bot@local" commit -m "InternVL remaining (38B eval + FFT + v3) complete" -q 2>/dev/null && git push -q 2>&1 || true
