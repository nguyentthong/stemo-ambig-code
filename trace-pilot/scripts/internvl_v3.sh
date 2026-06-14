#!/usr/bin/env bash
# InternVL v3 (strip-CoT SFT) for one -HF tag. Mirrors small_model_base_v3.sh's
# run_v3 but with: video_max_frames=8 and RUNNER_FAMILY=internvl_hf for eval
# (the shared helper omits the family → Qwen runner → empty InternVL output).
set -uo pipefail
REPO=/home/thong/weride_project/weride/overthinking_hallu
TAG="$1"
case "$TAG" in
  internvl8b_hf)  MID="OpenGVLab/InternVL3_5-8B-HF" ;;
  internvl38b_hf) MID="OpenGVLab/InternVL3_5-38B-HF" ;;
  *) echo "unknown tag $TAG"; exit 1 ;;
esac
V2DIR=$REPO/data_v0/stemo_ambig_sft_v2
CKPT=$REPO/checkpoints/${TAG}_stemo_ambig_lora_v3
CONFIG=$REPO/trace-pilot/configs/sft_lora_${TAG}_v3.yaml
cat > $CONFIG <<YAML
model:
  model_id: $MID
  trust_remote_code: true
  torch_dtype: bfloat16
data:
  train_file: $V2DIR/sft_train.jsonl
  dev_file:   $V2DIR/sft_dev.jsonl
  max_seq_len: 4096
  video_fps: 1.0
  video_max_frames: 8
lora:
  r: 128
  alpha: 128
  dropout: 0.0
  target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
training:
  output_dir: $CKPT
  num_train_epochs: 2
  per_device_train_batch_size: 1
  per_device_eval_batch_size: 1
  gradient_accumulation_steps: 8
  learning_rate: 5.0e-5
  lr_scheduler_type: cosine
  warmup_ratio: 0.03
  weight_decay: 0.0
  logging_steps: 5
  eval_steps: 50
  save_steps: 100
  save_total_limit: 3
  bf16: true
  gradient_checkpointing: true
  ddp_find_unused_parameters: false
  remove_unused_columns: false
  report_to: tensorboard
  logging_dir: $CKPT/runs
  seed: 0
YAML
echo "[$TAG v3] training start $(date)"
TRAINER="$REPO/trace-pilot/src/train_sft.py" SKIP_POST_VALIDATION=1 \
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 NUM_GPUS=6 PYTORCH_ALLOC_CONF=expandable_segments:True \
  bash $REPO/trace-pilot/scripts/launch_sft.sh $CONFIG
until [ -f "$CKPT/adapter_model.safetensors" ]; do sleep 30; done
echo "[$TAG v3] eval start $(date)"
MODEL_ID="$MID" NGPU=8 RUNNER_FAMILY=internvl_hf \
  bash $REPO/trace-pilot/scripts/run_stemo_ambig_eval_sharded.sh ${TAG}_v3 "$CKPT"
echo "[$TAG v3] done $(date)"
