#!/usr/bin/env bash
# Launch GRPO RL on STEMO-Ambig.
# Usage: launch_rl.sh <model-tag>  (qwen35 | qwen36 | qwen3vl32b)
set -uo pipefail
REPO=/home/thong/weride_project/weride/overthinking_hallu
TAG="$1"
case "$TAG" in
  qwen35)     MID="Qwen/Qwen3.5-27B";       TRAINER_FAMILY=qwen ;;
  qwen36)     MID="Qwen/Qwen3.6-27B";       TRAINER_FAMILY=qwen ;;
  qwen3vl32b) MID="Qwen/Qwen3-VL-32B-Thinking"; TRAINER_FAMILY=qwen ;;
  internvl38b)MID="OpenGVLab/InternVL3_5-38B"; TRAINER_FAMILY=internvl ;;
  *) echo "unknown tag $TAG"; exit 1 ;;
esac

# Family dispatcher: InternVL needs the InternVL-specific RL trainer (TBD).
case "$TRAINER_FAMILY" in
  internvl) RL_TRAINER="$REPO/trace-pilot/src/train_rl_grpo_internvl.py" ;;
  *)        RL_TRAINER="$REPO/trace-pilot/src/train_rl_grpo.py" ;;
esac
export RL_TRAINER

RLDIR=$REPO/data_v0/stemo_ambig_rl
V4_ADAPTER=$REPO/checkpoints/${TAG}_stemo_ambig_lora_v4
V5_ADAPTER=$REPO/checkpoints/${TAG}_stemo_ambig_lora_v5
CONFIG=$REPO/trace-pilot/configs/rl_grpo_${TAG}.yaml

# Prep RL input if not present
[ -f $RLDIR/rl_train.jsonl ] || python $REPO/trace-pilot/src/prep_rl_input.py

cat > $CONFIG <<EOF
model:
  model_id: $MID
  adapter_init: $V4_ADAPTER
  torch_dtype: bfloat16
data:
  train_file: $RLDIR/rl_train.jsonl
  video_max_frames: 16
  video_fps: 1.0
rl:
  algorithm: grpo
  n_rollouts: 8
  temperature: 0.8
  max_new_tokens: 4096
  kl_beta: 0.04
  reward_fn: trace-pilot.src.rl_reward.combined_reward
lora:
  r: 128
  alpha: 128
  dropout: 0.0
training:
  output_dir: $V5_ADAPTER
  num_train_epochs: 3
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 4
  learning_rate: 1.0e-6
  lr_scheduler_type: cosine
  warmup_ratio: 0.05
  logging_steps: 5
  save_steps: 200
  save_total_limit: 3
  bf16: true
  gradient_checkpointing: true
  seed: 0
EOF

# Launch
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 NUM_GPUS=6 PYTORCH_ALLOC_CONF=expandable_segments:True \
  python -m accelerate.commands.launch --num_processes 6 --num_machines 1 --mixed_precision bf16 \
    --use_deepspeed --deepspeed_config_file $REPO/trace-pilot/configs/ds_zero3.json \
    $RL_TRAINER --config $CONFIG

until [ -f "$V5_ADAPTER/adapter_model.safetensors" ]; do sleep 30; done
echo "[v5 $TAG] adapter saved"

# Eval the v5 adapter
MODEL_ID="$MID" NGPU=8 bash $REPO/trace-pilot/scripts/run_stemo_ambig_eval_sharded.sh ${TAG}_v5 "$V5_ADAPTER"
for B in videomme mvbench; do
  MODEL_ID="$MID" GROUP_BY=$([ "$B" = mvbench ] && echo task || echo duration) \
    GPUS=0,1,2,3,4,5,6,7 bash $REPO/trace-pilot/scripts/run_mcq_eval_sharded.sh $B ${TAG}_v5 "$V5_ADAPTER"
done

# Final report including hacking diagnostic
python - <<PY
import json
R = '$REPO/eval_runs'
print()
print('============ v5 $TAG eval ============')
for tag, label in [('${TAG}_v4', 'v4'), ('${TAG}_v5', 'v5')]:
    try:
        m = json.load(open(f'{R}/{tag}/stemo_ambig_metrics.json'))['overall']
        print(f"  {label}: enum={m['enumeration_rate']:.3f} commit={m['single_commit_rate']:.3f} strict={m['strict_ambig_aware_accuracy']:.3f}")
    except FileNotFoundError:
        print(f"  {label}: (missing)")
for b in ('videomme', 'mvbench'):
    try:
        bv4 = json.load(open(f'{R}/${TAG}_v4/{b}_metrics.json'))['accuracy']
        bv5 = json.load(open(f'{R}/${TAG}_v5/{b}_metrics.json'))['accuracy']
        print(f"  {b}: v4={bv4:.3f} v5={bv5:.3f} (Δ={bv5-bv4:+.3f})")
    except FileNotFoundError:
        pass
PY
