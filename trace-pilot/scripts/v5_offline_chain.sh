#!/usr/bin/env bash
# v5 offline RL: STaR-style sample → reward-judge → top-K select → SFT.
# Decouples generation (uses run_qwen_video.py, all 8 GPUs) from training
# (uses train_sft.py, fits in memory) so we avoid the TRL co-location wall.
#
# Usage: v5_offline_chain.sh <tag>     e.g.  v5_offline_chain.sh qwen35
#
# Requires:
#   - data_v0/stemo_ambig_sft_${tag}_v4/star_input.jsonl   (gold input)
#   - checkpoints/${tag}_stemo_ambig_lora_v4/              (v4 LoRA adapter)
set -uo pipefail
REPO=/home/thong/weride_project/weride/overthinking_hallu
TAG="$1"
case "$TAG" in
  qwen35)     MID="Qwen/Qwen3.5-27B" ;;
  qwen36)     MID="Qwen/Qwen3.6-27B" ;;
  qwen3vl32b) MID="Qwen/Qwen3-VL-32B-Thinking" ;;
  internvl8b_hf)  MID="OpenGVLab/InternVL3_5-8B-HF" ;;
  internvl38b_hf) MID="OpenGVLab/InternVL3_5-38B-HF" ;;
  *) echo "unknown tag $TAG (must be qwen35 / qwen36 / qwen3vl32b / internvl*_hf)"; exit 1 ;;
esac
case "$TAG" in internvl*_hf) VMAXFRAMES=8 ;; *) VMAXFRAMES=16 ;; esac

V4DIR=$REPO/data_v0/stemo_ambig_sft_${TAG}_v4
V4_ADAPTER=$REPO/checkpoints/${TAG}_stemo_ambig_lora_v4
V5_DIR=$REPO/data_v0/stemo_ambig_v5_offline_${TAG}
V5_ADAPTER=$REPO/checkpoints/${TAG}_stemo_ambig_lora_v5_offline
V2DIR=$REPO/data_v0/stemo_ambig_sft_v2

if [ ! -f $V4_ADAPTER/adapter_model.safetensors ]; then
  echo "ERROR: v4 adapter missing at $V4_ADAPTER"; exit 1
fi
if [ ! -f $V4DIR/star_input.jsonl ]; then
  echo "ERROR: star input missing at $V4DIR/star_input.jsonl"; exit 1
fi

mkdir -p $V5_DIR
echo "[v5-off $TAG] START $(date -u +%FT%TZ)"

# Reuse the same scaffold system prompt the v4 chain used (so rollouts match v4's prior)
SYS='You are an expert at answering questions about video content. Watch the video carefully and answer the question.

If the question contains a referentially ambiguous phrase (e.g. "the man" when there are multiple men in the video), enumerate each valid interpretation and give an answer per interpretation. If the question is unambiguous, give a single direct answer.

Format for ambiguous questions (use exactly this shape):
This question has K valid interpretations.
- "<referent description>" → Yes
- "<referent description>" → No
- ...

Two illustrative examples (different videos; shown only to clarify the output format):

Example 1.
Question: "Does the boy fall down?"
(Suppose the video shows two boys: one in red who slips and falls at 0:05, one in blue who runs ahead without falling.)
Response: This question has 2 valid interpretations.
- "the boy in red who slips at 0:05" → Yes
- "the boy in blue who runs ahead" → No

Example 2.
Question: "Is the color added third?"
(Suppose the video shows a person painting bands in this order: black, red, blue, green.)
Response: This question has 4 valid interpretations.
- "black" → No
- "red" → No
- "blue" → Yes
- "green" → No

Now answer the question about the video provided. Think step by step before giving your final answer.'

# ===== 1) Sample from v4-adapted policy (8 GPUs, generation-only, fits) =====
SD=$V5_DIR/rollout_shards
mkdir -p $SD
if [ ! -f $V5_DIR/rollouts.jsonl ]; then
  echo "[v5-off $TAG] sampling N=8 rollouts at temp=0.8 from v4 policy..."
  # Shard input into 8 pieces
  python - $V4DIR/star_input.jsonl $SD 8 <<'PY'
import sys
src,dst,n=sys.argv[1],sys.argv[2],int(sys.argv[3])
lines=[l for l in open(src).read().splitlines() if l.strip()]
import os
for i in range(n):
    with open(f"{dst}/shard_{i}.jsonl","w") as f:
        f.write("\n".join(lines[i::n])+"\n")
print(f"split {len(lines)} into {n} shards")
PY
  PIDS=()
  for i in 0 1 2 3 4 5 6 7; do
    # max-new-tokens 2048 (was 4096) + num-samples 4 (was 8): 4x faster sampling.
    # Same effective token budget; top-K=2 selector still gets pick-from-4 diversity.
    SAMPLER=$REPO/trace-pilot/src/eval/run_qwen_video.py
    case "$TAG" in internvl*_hf) SAMPLER=$REPO/trace-pilot/src/eval/run_internvl_hf_video.py ;; esac
    CUDA_VISIBLE_DEVICES=$i python $SAMPLER \
      --model-id "$MID" \
      --adapter "$V4_ADAPTER" \
      --input "$SD/shard_$i.jsonl" --output "$SD/preds_$i.jsonl" \
      --system-prompt "$SYS" \
      --max-new-tokens 2048 \
      --temperature 0.8 --num-samples 4 \
      > "$SD/log_$i.txt" 2>&1 &
    PIDS+=($!)
  done
  for p in "${PIDS[@]}"; do wait $p || true; done
  cat $SD/preds_*.jsonl > $V5_DIR/rollouts.jsonl
  echo "[v5-off $TAG] sampled $(wc -l < $V5_DIR/rollouts.jsonl) prediction rows"
else
  echo "[v5-off $TAG] reusing existing rollouts ($(wc -l < $V5_DIR/rollouts.jsonl) rows)"
fi

# ===== 2) Judge every rollout (no GPU; ~$5 in Gemini calls for 8×1056=8.5k judgments) =====
if [ ! -f $V5_DIR/judged_rollouts.jsonl ]; then
  echo "[v5-off $TAG] judging all rollouts with Gemini (reward = n_correct/K + penalties)..."
  python $REPO/trace-pilot/src/v5_judge_rollouts.py \
    --input $V4DIR/star_input.jsonl \
    --predictions $V5_DIR/rollouts.jsonl \
    --out $V5_DIR/judged_rollouts.jsonl \
    --workers 16
else
  echo "[v5-off $TAG] reusing existing judgments"
fi

# ===== 3) Select top-2 rollouts per item by reward, format as SFT input =====
echo "[v5-off $TAG] selecting top-2 rollouts per item (min_reward=0.5)..."
python $REPO/trace-pilot/src/v5_select_topk.py \
  --judged $V5_DIR/judged_rollouts.jsonl \
  --input $V4DIR/star_input.jsonl \
  --out $V5_DIR/star_kept_v5.jsonl \
  --topk 2 --min-reward 0.5

# ===== 4) Format as SFT data (reuses format_sft_v4.py) =====
echo "[v5-off $TAG] formatting SFT data..."
python $REPO/trace-pilot/src/format_sft_v4.py \
  --star-kept $V5_DIR/star_kept_v5.jsonl \
  --rehearsal $V2DIR/rehearsal.jsonl \
  --out-dir   $V5_DIR \
  --target-ambig-frac 0.20

# ===== 5) Train v5 LoRA from v4 init =====
CONFIG=$REPO/trace-pilot/configs/v5_offline_${TAG}.yaml
cat > $CONFIG <<EOF
model:
  model_id: $MID
  adapter_init: $V4_ADAPTER     # continue from v4 LoRA
  trust_remote_code: true
  torch_dtype: bfloat16
data:
  train_file: $V5_DIR/sft_train.jsonl
  dev_file:   $V5_DIR/sft_dev.jsonl
  max_seq_len: 4096
  video_fps: 1.0
  video_max_frames: $VMAXFRAMES
lora:
  r: 128
  alpha: 128
  dropout: 0.0
  target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
training:
  output_dir: $V5_ADAPTER
  num_train_epochs: 1            # v5 is a gentle nudge from v4
  per_device_train_batch_size: 1
  per_device_eval_batch_size: 1
  gradient_accumulation_steps: 8
  learning_rate: 1.0e-5          # half of v4's lr (continuing from a trained adapter)
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
  logging_dir: $V5_ADAPTER/runs
  seed: 0
EOF

echo "[v5-off $TAG] training v5 LoRA from v4 adapter..."
SKIP_POST_VALIDATION=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 NUM_GPUS=6 PYTORCH_ALLOC_CONF=expandable_segments:True \
  bash $REPO/trace-pilot/scripts/launch_sft.sh $CONFIG

echo "[v5-off $TAG] DONE $(date -u +%FT%TZ)"
