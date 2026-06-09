#!/usr/bin/env bash
# v5 27B full queue: after the v5 IAA eval on Qwen3.5-27B finishes, run the v5
# offline chain on the other two open-weight 27/32B models (qwen36 + qwen3vl32b)
# sequentially, then eval each. Keeps GPUs busy and produces breadth for the
# paper's RL section (3 models with v5 instead of 1).
set -uo pipefail
REPO=/home/thong/weride_project/weride/overthinking_hallu
LOG=$REPO/tmp/v5_27b_full_queue.log

log() { echo "[$(date -u +%FT%TZ)] $*" >> "$LOG"; echo "[$(date -u +%FT%TZ)] $*"; }

# Wait for the Qwen3.5-27B v5 IAA eval to finish (metrics.json indicates done)
wait_for() {
  local label="$1"; shift
  local target="$1"
  log "waiting for $label at $target"
  while [ ! -f "$target" ]; do sleep 60; done
  log "$label ready"
}

log "queue START"

# 1) Wait for v5 IAA eval on qwen35 to finish
wait_for "qwen35 v5_offline IAA metrics" \
  "$REPO/eval_runs/qwen35_iaa_v5_offline/iaa_metrics.json"

# 2) Run v5 offline chain on qwen36 (second open-weight 27B)
log "starting v5 offline chain on qwen36 (Qwen3.6-27B)"
bash $REPO/trace-pilot/scripts/v5_offline_chain.sh qwen36 \
  > $REPO/tmp/v5_27b_offline_qwen36.log 2>&1
log "qwen36 chain done; exit=$?"

# 3) Eval qwen36 v5_offline
log "evaluating qwen36 v5_offline (8-shard IAA)"
bash $REPO/trace-pilot/src/iaa/iaa_open_launcher.sh \
  "Qwen/Qwen3.6-27B" \
  "$REPO/checkpoints/qwen36_stemo_ambig_lora_v5_offline" \
  "qwen36_iaa_v5_offline" \
  "0,1,2,3,4,5,6,7" \
  > $REPO/tmp/iaa_qwen36_v5_offline.log 2>&1
log "qwen36 v5 IAA eval done; exit=$?"

# 4) Run v5 offline chain on qwen3vl32b
log "starting v5 offline chain on qwen3vl32b (Qwen3-VL-32B)"
bash $REPO/trace-pilot/scripts/v5_offline_chain.sh qwen3vl32b \
  > $REPO/tmp/v5_27b_offline_qwen3vl32b.log 2>&1
log "qwen3vl32b chain done; exit=$?"

# 5) Eval qwen3vl32b v5_offline
log "evaluating qwen3vl32b v5_offline (8-shard IAA)"
bash $REPO/trace-pilot/src/iaa/iaa_open_launcher.sh \
  "Qwen/Qwen3-VL-32B-Thinking" \
  "$REPO/checkpoints/qwen3vl32b_stemo_ambig_lora_v5_offline" \
  "qwen3vl32b_iaa_v5_offline" \
  "0,1,2,3,4,5,6,7" \
  > $REPO/tmp/iaa_qwen3vl32b_v5_offline.log 2>&1
log "qwen3vl32b v5 IAA eval done; exit=$?"

log "queue END — all 3 open-weight 27/32B models have v5 + IAA"
