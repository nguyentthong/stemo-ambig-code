#!/usr/bin/env bash
# Launch LoRA SFT on 8 GPUs via accelerate + deepspeed zero-3.
#
# Usage:
#   bash trace-pilot/scripts/launch_sft.sh [--config trace-pilot/configs/sft_lora.yaml]

set -euo pipefail

CONFIG="${1:-trace-pilot/configs/sft_lora.yaml}"

# Use 8 GPUs by default; override with NUM_GPUS env var.
export NUM_GPUS=${NUM_GPUS:-8}
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-0}
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1

TRAINER_SCRIPT="${TRAINER:-trace-pilot/src/train_sft.py}"
accelerate launch \
  --num_processes "$NUM_GPUS" \
  --num_machines 1 \
  --mixed_precision bf16 \
  --use_deepspeed \
  --deepspeed_config_file trace-pilot/configs/ds_zero3.json \
  --deepspeed_multinode_launcher standard \
  "$TRAINER_SCRIPT" --config "$CONFIG"

# Post-training: full validation on the final checkpoint.
if [ "${SKIP_POST_VALIDATION:-0}" != "1" ]; then
  OUT_DIR=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['training']['output_dir'])")
  if [ -d "$OUT_DIR" ]; then
    echo
    echo "=============================="
    echo "Training complete. Running full validation on $OUT_DIR ..."
    echo "=============================="
    bash trace-pilot/scripts/validate_checkpoint.sh sft_final "$OUT_DIR" || true
  fi
fi
