#!/usr/bin/env bash
# Auto-chain v2 pipeline: wait for self-distillation → format SFT JSONL → train → eval.

set -uo pipefail

REPO=/home/thong/weride_project/weride/overthinking_hallu
V2=$REPO/data_v0/stemo_ambig_sft_v2

# 1) Wait for self-distillation
echo "[chain-v2] waiting for self-distillation to finish ..."
until [ -f "$V2/self_distill_predictions.jsonl" ] && [ "$(wc -l < $V2/self_distill_predictions.jsonl)" -ge 2100 ]; do
  sleep 30
done
echo "[chain-v2] self-distillation done: $(wc -l < $V2/self_distill_predictions.jsonl) predictions"

# 2) Format v2 SFT JSONL
echo "[chain-v2] formatting v2 SFT JSONL ..."
python $REPO/trace-pilot/src/format_sft_v2.py \
  --self-distill-input  $V2/self_distill_input.jsonl \
  --self-distill-preds  $V2/self_distill_predictions.jsonl \
  --rehearsal           $V2/rehearsal.jsonl \
  --out-dir             $V2

# 3) Train v2 with light-touch hyperparams
echo "[chain-v2] launching v2 training ..."
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 NUM_GPUS=6 PYTORCH_ALLOC_CONF=expandable_segments:True \
  bash $REPO/trace-pilot/scripts/launch_sft.sh $REPO/trace-pilot/configs/sft_lora_v2.yaml

# 4) Evaluate v2 (post-train chain in launch_sft.sh runs validate_checkpoint.sh automatically)
echo "[chain-v2] training finished. validate_checkpoint should have fired from launch_sft.sh."
