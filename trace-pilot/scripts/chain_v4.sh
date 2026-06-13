#!/usr/bin/env bash
# v4 STaR pipeline: sample → judge-filter → format → train → eval (thinking ON).
# Usage: chain_v4.sh <model-tag>  e.g.  chain_v4.sh qwen35 / qwen36 / qwen3vl32b
# Model-tag determines model_id, data dir, checkpoint dir, eval tags.

set -uo pipefail
REPO=/home/thong/weride_project/weride/overthinking_hallu
TAG="$1"
case "$TAG" in
  qwen35)     MID="Qwen/Qwen3.5-27B"      ;;
  qwen36)     MID="Qwen/Qwen3.6-27B"      ;;
  qwen3vl32b) MID="Qwen/Qwen3-VL-32B-Thinking" ;;
  qwen36_9b)  MID="Qwen/Qwen3.5-9B"       ;;  # Qwen3.6 has no 9B variant; using closest Qwen3.5-9B for smaller-scale comparison
  internvl38b)MID="OpenGVLab/InternVL3_5-38B" ;;
  internvl8b) MID="OpenGVLab/InternVL3_5-8B"  ;;
  # Native-HF variants: official InternVLForConditionalGeneration class, no
  # trust_remote_code, so they load through the standard AutoModelForImageTextToText
  # runner/trainer (the custom modeling code is incompatible with transformers 5.x).
  internvl8b_hf)  MID="OpenGVLab/InternVL3_5-8B-HF"  ;;
  internvl38b_hf) MID="OpenGVLab/InternVL3_5-38B-HF" ;;
  *) echo "unknown tag $TAG"; exit 1 ;;
esac

# Family dispatcher: legacy InternVL custom-code tags use the custom runner;
# the -HF tags + all Qwen tags use the standard AutoModel runner/trainer.
case "$TAG" in
  internvl8b_hf|internvl38b_hf)
              RUNNER="$REPO/trace-pilot/src/eval/run_qwen_video.py"
              TRAINER="$REPO/trace-pilot/src/train_sft.py" ;;
  internvl*)  RUNNER="$REPO/trace-pilot/src/eval/run_internvl_video.py"
              TRAINER="$REPO/trace-pilot/src/train_sft_internvl.py" ;;
  *)          RUNNER="$REPO/trace-pilot/src/eval/run_qwen_video.py"
              TRAINER="$REPO/trace-pilot/src/train_sft.py" ;;
esac
export RUNNER TRAINER

V2DIR=$REPO/data_v0/stemo_ambig_sft_v2          # reuse rehearsal
V4DIR=$REPO/data_v0/stemo_ambig_sft_${TAG}_v4
ADAPTER=$REPO/checkpoints/${TAG}_stemo_ambig_lora_v4
CONFIG=$REPO/trace-pilot/configs/sft_lora_${TAG}_v4.yaml
mkdir -p $V4DIR

# 1) Prep STaR input (no scaffold)
[ -f $V4DIR/star_input.jsonl ] || python $REPO/trace-pilot/src/prep_star_input.py --out $V4DIR/star_input.jsonl

# 2) Sample N=8 with thinking-ON, real video reasoning (no scaffold)
SD=$V4DIR/star_shards
mkdir -p $SD
echo "[v4 $TAG] STaR sampling (N=8 per item, thinking ON)..."
python - $V4DIR/star_input.jsonl $SD 8 <<'PY'
import sys
src,dst,n=sys.argv[1],sys.argv[2],int(sys.argv[3])
lines=[l for l in open(src).read().splitlines() if l.strip()]
for i in range(n): open(f"{dst}/shard_{i}.jsonl","w").write("\n".join(lines[i::n])+"\n")
print(f"split {len(lines)} into {n} shards")
PY
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
PIDS=()
for i in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$i python $RUNNER \
    --model-id "$MID" \
    --input "$SD/shard_$i.jsonl" --output "$SD/preds_$i.jsonl" \
    --system-prompt "$SYS" \
    --max-new-tokens 16384 \
    --temperature 0.7 --num-samples 4 \
    > "$SD/log_$i.txt" 2>&1 &
  PIDS+=($!)
done
for p in "${PIDS[@]}"; do wait $p || true; done
cat $SD/preds_*.jsonl > $V4DIR/star_predictions.jsonl
echo "[v4 $TAG] sampled $(wc -l < $V4DIR/star_predictions.jsonl) items"

# 3) Judge-filter
echo "[v4 $TAG] Gemini judge filter..."
python $REPO/trace-pilot/src/star_filter.py \
  --input $V4DIR/star_input.jsonl \
  --predictions $V4DIR/star_predictions.jsonl \
  --out $V4DIR/star_kept.jsonl \
  --workers 12 --strict-full-k

# 3.5) Paraphrase kept questions (~4 per item) to broaden surface form
echo "[v4 $TAG] paraphrasing kept ambig questions..."
python $REPO/trace-pilot/src/paraphrase_questions.py \
  --input $V4DIR/star_kept.jsonl \
  --star-input $V4DIR/star_input.jsonl \
  --out $V4DIR/star_kept_aug.jsonl \
  --n-paraphrases 4 --workers 12

# 4) Format v4 SFT data (CoT preserved + rehearsal) — use augmented kept
echo "[v4 $TAG] formatting SFT..."
python $REPO/trace-pilot/src/format_sft_v4.py \
  --star-kept $V4DIR/star_kept_aug.jsonl \
  --rehearsal $V2DIR/rehearsal.jsonl \
  --out-dir   $V4DIR \
  --target-ambig-frac 0.20

# 5) Generate config from template (just swap model_id + paths)
cat > $CONFIG <<EOF
model:
  model_id: $MID
  trust_remote_code: true
  torch_dtype: bfloat16
data:
  train_file: $V4DIR/sft_train.jsonl
  dev_file:   $V4DIR/sft_dev.jsonl
  max_seq_len: 4096            # CoT preserved → longer targets, bump from 2048
  video_fps: 1.0
  video_max_frames: 16
lora:
  r: 128
  alpha: 128
  dropout: 0.0
  target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
training:
  output_dir: $ADAPTER
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
  logging_dir: $ADAPTER/runs
  seed: 0
EOF

# 6) Train
echo "[v4 $TAG] training..."
SKIP_POST_VALIDATION=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 NUM_GPUS=6 PYTORCH_ALLOC_CONF=expandable_segments:True \
  bash $REPO/trace-pilot/scripts/launch_sft.sh $CONFIG
until [ -f "$ADAPTER/adapter_model.safetensors" ]; do sleep 30; done

# 7) Eval — thinking ON (the point of v4)
echo "[v4 $TAG] eval base (thinking on)..."
MODEL_ID="$MID" NGPU=8 bash $REPO/trace-pilot/scripts/run_stemo_ambig_eval_sharded.sh ${TAG}_v4_base ""
echo "[v4 $TAG] eval v4 (thinking on)..."
MODEL_ID="$MID" NGPU=8 bash $REPO/trace-pilot/scripts/run_stemo_ambig_eval_sharded.sh ${TAG}_v4 "$ADAPTER"

for B in videomme mvbench; do
  GB=duration; [ "$B" = mvbench ] && GB=task
  MODEL_ID="$MID" GROUP_BY=$GB GPUS=0,1,2,3 bash $REPO/trace-pilot/scripts/run_mcq_eval_sharded.sh $B ${TAG}_v4_base "" &
  MODEL_ID="$MID" GROUP_BY=$GB GPUS=4,5,6,7 bash $REPO/trace-pilot/scripts/run_mcq_eval_sharded.sh $B ${TAG}_v4 "$ADAPTER" &
  wait
done

echo
echo "============ v4 $TAG: base vs v4 (thinking ON) ============"
python - <<PY
import json
R='$REPO/eval_runs'
for t,l in [('${TAG}_v4_base','base'),('${TAG}_v4','v4')]:
    m=json.load(open(f'{R}/{t}/stemo_ambig_metrics.json'))['overall']
    print(f'  {l}: enum={m["enumeration_rate"]:.3f} commit={m["single_commit_rate"]:.3f} strict={m["strict_ambig_aware_accuracy"]:.3f}')
for b in ('videomme','mvbench'):
    try:
        mb=json.load(open(f'{R}/${TAG}_v4_base/{b}_metrics.json'))['accuracy']
        mv=json.load(open(f'{R}/${TAG}_v4/{b}_metrics.json'))['accuracy']
        print(f'  {b}: {mb:.3f} -> {mv:.3f} ({mv-mb:+.3f})')
    except: pass
PY
