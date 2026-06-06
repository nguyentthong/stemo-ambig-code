#!/usr/bin/env bash
# Wait for 9B v5 RL to finish, then run 27B offline RL chain.
# Order of operations:
#   1. Poll for /checkpoints/qwen35_9b_stemo_ambig_lora_v5/adapter_model.safetensors
#   2. When seen, sleep 60s to let any post-training eval finish
#   3. Run v5_offline_chain.sh qwen35 (the 27B model)
#   4. Log everything to tmp/v5_27b_offline.log
set -uo pipefail
REPO=/home/thong/weride_project/weride/overthinking_hallu
ADAPTER_9B=$REPO/checkpoints/qwen35_9b_stemo_ambig_lora_v5/adapter_model.safetensors
LOG=$REPO/tmp/v5_27b_offline.log

log() { echo "[$(date -u +%FT%TZ)] $*" >> "$LOG"; }

log "queue START — waiting for 9B v5 adapter at $ADAPTER_9B"

# Poll every 5 minutes
while [ ! -f "$ADAPTER_9B" ]; do
  sleep 300
done

log "9B v5 adapter detected. Waiting 60s for post-training settling..."
sleep 60

log "Starting 27B offline chain: trace-pilot/scripts/v5_offline_chain.sh qwen35"
bash $REPO/trace-pilot/scripts/v5_offline_chain.sh qwen35 >> "$LOG" 2>&1
EC=$?

if [ $EC -eq 0 ] && [ -f "$REPO/checkpoints/qwen35_stemo_ambig_lora_v5_offline/adapter_model.safetensors" ]; then
  log "queue SUCCESS — 27B v5_offline adapter saved"
else
  log "queue FAILED — exit_code=$EC, check $REPO/tmp/v5_27b_offline.log for details"
fi

log "queue END"
