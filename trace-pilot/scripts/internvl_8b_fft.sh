#!/usr/bin/env bash
# InternVL3.5-8B FFT (full-parameter fine-tuning) of v4. Reuses the v4 SFT data;
# train_sft.py with lora.r=0 = FFT, now with the InternVL multi-image collator.
# 8 frames + internvl_hf eval family.
set -uo pipefail
REPO=/home/thong/weride_project/weride/overthinking_hallu
cd $REPO
TAG=internvl8b_hf
MID="OpenGVLab/InternVL3_5-8B-HF"
V4DIR=$REPO/data_v0/stemo_ambig_sft_${TAG}_v4
CKPT=$REPO/checkpoints/${TAG}_stemo_ambig_fft_v4
CONFIG=$REPO/trace-pilot/configs/sft_fft_${TAG}_v4.yaml
LOG=$REPO/tmp/internvl_8b_fft.log
log(){ echo "[$(date -u +%FT%TZ)] $*" >> "$LOG"; echo "[$(date -u +%FT%TZ)] $*"; }

[ -f "$V4DIR/sft_train.jsonl" ] || { log "MISSING v4 sft data; abort"; exit 1; }

cat > $CONFIG <<YAML
model:
  model_id: $MID
  trust_remote_code: true
  torch_dtype: bfloat16
data:
  train_file: $V4DIR/sft_train.jsonl
  dev_file:   $V4DIR/sft_dev.jsonl
  max_seq_len: 4096
  video_fps: 1.0
  video_max_frames: 8
lora:
  r: 0
  alpha: 0
  dropout: 0.0
  target_modules: []
training:
  output_dir: $CKPT
  num_train_epochs: 2
  per_device_train_batch_size: 1
  per_device_eval_batch_size: 1
  gradient_accumulation_steps: 16
  learning_rate: 5.0e-6
  lr_scheduler_type: cosine
  warmup_ratio: 0.05
  weight_decay: 0.01
  logging_steps: 5
  eval_steps: 50
  save_steps: 200
  save_total_limit: 2
  bf16: true
  gradient_checkpointing: true
  ddp_find_unused_parameters: false
  remove_unused_columns: false
  report_to: tensorboard
  logging_dir: $CKPT/runs
  seed: 0
YAML

log "FFT — waiting for GPUs free"
while pgrep -f "run_internvl_hf_video|train_sft|accelerate.commands|run_qwen_video" >/dev/null 2>&1; do sleep 300; done
log "GPUs free — FFT training start"
TRAINER="$REPO/trace-pilot/src/train_sft.py" SKIP_POST_VALIDATION=1 \
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 NUM_GPUS=6 PYTORCH_ALLOC_CONF=expandable_segments:True \
  bash $REPO/trace-pilot/scripts/launch_sft.sh $CONFIG
until [ -f "$CKPT/model.safetensors" ] || [ -f "$CKPT/model-00001-of-00004.safetensors" ] || [ -f "$CKPT/pytorch_model.bin" ]; do
  # FFT saves full weights (not adapter); break also if training proc gone
  pgrep -f "train_sft" >/dev/null 2>&1 || break
  sleep 60
done
log "FFT training done — eval"
MODEL_ID="$MID" NGPU=8 RUNNER_FAMILY=internvl_hf \
  bash $REPO/trace-pilot/scripts/run_stemo_ambig_eval_sharded.sh ${TAG}_fft_v4 "$CKPT"
log "FFT eval done"
python $REPO/tools/dashboard.py >/dev/null 2>&1
cp $REPO/STATUS.md $REPO/dashboard_repo/STATUS.md 2>/dev/null
cd $REPO/dashboard_repo && git add STATUS.md && git -c user.name="dashboard" -c user.email="bot@local" commit -m "InternVL 8B FFT complete" -q 2>/dev/null && git push -q 2>&1 || true
