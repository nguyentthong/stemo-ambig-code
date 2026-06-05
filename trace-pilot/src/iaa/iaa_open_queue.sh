#!/usr/bin/env bash
# IAA open-weight queue: run multi-turn IAA protocol on existing trained models.
# Each model uses all 8 GPUs (sharded) so they run sequentially.
set -uo pipefail
REPO=/home/thong/weride_project/weride/overthinking_hallu
cd $REPO

LAUNCHER="$REPO/trace-pilot/src/iaa/iaa_open_launcher.sh"

run_one() {
  local model_id="$1"; local adapter="$2"; local out_tag="$3"
  echo "[iaa-queue] === $out_tag start $(date) ==="
  bash "$LAUNCHER" "$model_id" "$adapter" "$out_tag" "0,1,2,3,4,5,6,7" \
    > "$REPO/tmp/iaa_${out_tag}.log" 2>&1
  echo "[iaa-queue] === $out_tag done $(date) ==="
}

echo "[iaa-queue] START $(date)"

# Qwen3-VL-32B base + v4 (most important for headline)
run_one "Qwen/Qwen3-VL-32B-Thinking" \
        "NONE" \
        "qwen3vl32b_iaa_base"

run_one "Qwen/Qwen3-VL-32B-Thinking" \
        "$REPO/checkpoints/qwen3vl32b_stemo_ambig_lora_v4" \
        "qwen3vl32b_iaa_v4"

# Qwen3.6-27B base + v4
run_one "Qwen/Qwen3.6-27B" \
        "NONE" \
        "qwen36_iaa_base"

run_one "Qwen/Qwen3.6-27B" \
        "$REPO/checkpoints/qwen36_stemo_ambig_lora_v4" \
        "qwen36_iaa_v4"

# Qwen3.5-27B base + v4
run_one "Qwen/Qwen3.5-27B" \
        "NONE" \
        "qwen35_iaa_base"

run_one "Qwen/Qwen3.5-27B" \
        "$REPO/checkpoints/qwen35_stemo_ambig_lora_v4" \
        "qwen35_iaa_v4"

echo "[iaa-queue] ALL IAA OPEN-WEIGHT DONE $(date)"
