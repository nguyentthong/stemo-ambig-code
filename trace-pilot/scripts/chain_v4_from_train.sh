#!/usr/bin/env bash
# Resume chain_v4.sh from step 6 (training). Assumes sampling + judge filter +
# format already completed. Used to avoid redoing the ~100min Gemini filter.
# Usage: chain_v4_from_train.sh <model-tag>

set -uo pipefail
REPO=/home/thong/weride_project/weride/overthinking_hallu
TAG="$1"
case "$TAG" in
  qwen35)     MID="Qwen/Qwen3.5-27B"      ;;
  qwen36)     MID="Qwen/Qwen3.6-27B"      ;;
  qwen3vl32b) MID="Qwen/Qwen3-VL-32B-Thinking" ;;
  *) echo "unknown tag $TAG"; exit 1 ;;
esac

V4DIR=$REPO/data_v0/stemo_ambig_sft_${TAG}_v4
ADAPTER=$REPO/checkpoints/${TAG}_stemo_ambig_lora_v4
CONFIG=$REPO/trace-pilot/configs/sft_lora_${TAG}_v4.yaml

# Sanity: required inputs must exist
for f in $V4DIR/sft_train.jsonl $V4DIR/sft_dev.jsonl $CONFIG; do
  if [ ! -f "$f" ]; then
    echo "MISSING: $f — cannot resume from train step"; exit 2
  fi
done

# Train
echo "[v4 $TAG] training..."
SKIP_POST_VALIDATION=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 NUM_GPUS=6 PYTORCH_ALLOC_CONF=expandable_segments:True \
  bash $REPO/trace-pilot/scripts/launch_sft.sh $CONFIG
until [ -f "$ADAPTER/adapter_model.safetensors" ]; do sleep 30; done

# Eval — thinking ON
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
