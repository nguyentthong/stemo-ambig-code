#!/usr/bin/env bash
# InternVL full ladder: v4 (tested) then v5-offline for 8B and 38B.
# v3 appended last (chain_v3 parameterization pending). Each stage idempotent
# (skips if its metrics already exist). Sequential — one model on 8 GPUs at a time.
set -uo pipefail
REPO=/home/thong/weride_project/weride/overthinking_hallu
cd $REPO
LOG=$REPO/tmp/internvl_ladder_queue.log
log(){ echo "[$(date -u +%FT%TZ)] $*" >> "$LOG"; echo "[$(date -u +%FT%TZ)] $*"; }

# Wait for any residual GPU jobs (base evals) to finish first
while pgrep -f "run_qwen_video|run_internvl_hf_video|train_sft|accelerate.commands" >/dev/null 2>&1; do
  log "waiting for residual GPU jobs to drain..."; sleep 120
done

log "InternVL ladder queue START"

# ---- v4 (base + v4 + VideoMME + MVBench), 8B then 38B ----
for TAG in internvl8b_hf internvl38b_hf; do
  if [ -f "$REPO/eval_runs/${TAG}_v4/stemo_ambig_metrics.json" ]; then
    log "$TAG v4 already done, skipping"; continue
  fi
  log "=== $TAG v4 chain start ==="
  bash $REPO/trace-pilot/scripts/chain_v4.sh $TAG > $REPO/tmp/v4_${TAG}.log 2>&1
  log "=== $TAG v4 chain done; exit=$? ==="
done

# ---- v5 offline (requires v4 adapter), 8B then 38B ----
for TAG in internvl8b_hf internvl38b_hf; do
  if [ -f "$REPO/checkpoints/${TAG}_stemo_ambig_lora_v5_offline/adapter_model.safetensors" ]; then
    log "$TAG v5 already done, skipping"; continue
  fi
  if [ ! -f "$REPO/checkpoints/${TAG}_stemo_ambig_lora_v4/adapter_model.safetensors" ]; then
    log "$TAG v5 SKIPPED — no v4 adapter"; continue
  fi
  log "=== $TAG v5-offline chain start ==="
  bash $REPO/trace-pilot/scripts/v5_offline_chain.sh $TAG > $REPO/tmp/v5_${TAG}.log 2>&1
  log "=== $TAG v5-offline chain done; exit=$? ==="
  # v5 strict-K eval (IAA done separately if desired)
  log "=== $TAG v5 strict-K eval ==="
  MODEL_ID=$(case $TAG in internvl8b_hf) echo OpenGVLab/InternVL3_5-8B-HF;; *) echo OpenGVLab/InternVL3_5-38B-HF;; esac)
  MODEL_ID="$MODEL_ID" NGPU=8 RUNNER_FAMILY=internvl_hf \
    bash $REPO/trace-pilot/scripts/run_stemo_ambig_eval_sharded.sh ${TAG}_v5_offline "$REPO/checkpoints/${TAG}_stemo_ambig_lora_v5_offline"
done

log "InternVL ladder queue END (v4 + v5; v3 pending chain_v3 parameterization)"
python $REPO/tools/dashboard.py >/dev/null 2>&1
cp $REPO/STATUS.md $REPO/dashboard_repo/STATUS.md 2>/dev/null
cd $REPO/dashboard_repo && git add STATUS.md && git -c user.name="dashboard" -c user.email="bot@local" commit -m "InternVL v4+v5 complete" -q 2>/dev/null && git push -q 2>&1 || true
