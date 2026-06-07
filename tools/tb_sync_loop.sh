#!/usr/bin/env bash
# Periodically convert the latest v5 trainer_state.json into TensorBoard events.
# Runs every 5 minutes. Detects any checkpoints/*_lora_v5*/checkpoint-* directories.
set -uo pipefail
REPO=/home/thong/weride_project/weride/overthinking_hallu

while true; do
  for adapter in $REPO/checkpoints/*_lora_v5* $REPO/checkpoints/*_lora_v5_offline*; do
    [ -d "$adapter" ] || continue
    # Find latest checkpoint
    latest=$(ls -d $adapter/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
    [ -z "$latest" ] && continue
    state="$latest/trainer_state.json"
    [ -f "$state" ] || continue
    python $REPO/tools/state_to_tensorboard.py \
      --state "$state" --logdir "$adapter/runs" > /dev/null 2>&1
  done
  sleep 300
done
