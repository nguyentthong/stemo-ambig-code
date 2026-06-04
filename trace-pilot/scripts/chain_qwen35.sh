#!/usr/bin/env bash
# Full generalization pipeline for Qwen3.5-27B: self-distill -> format -> train -> eval.
# Mirrors the proven v3 recipe; only model_id + paths differ.

set -uo pipefail
REPO=/home/thong/weride_project/weride/overthinking_hallu
MID="Qwen/Qwen3.5-27B"
V2DIR=$REPO/data_v0/stemo_ambig_sft_v2          # reuse self_distill_input + rehearsal
OUT=$REPO/data_v0/stemo_ambig_sft_qwen35
ADAPTER=$REPO/checkpoints/qwen35_27b_stemo_ambig_lora_v3
TAG=qwen35_v3
mkdir -p $OUT

# 1) Self-distill ambig items with Qwen3.5-27B (its own voice, gold scaffold)
echo "[qwen35] self-distillation..."
SD=$OUT/self_distill_shards
mkdir -p $SD
python - "$V2DIR/self_distill_input.jsonl" "$SD" 8 <<'PY'
import sys
src,dst,n=sys.argv[1],sys.argv[2],int(sys.argv[3])
lines=[l for l in open(src).read().splitlines() if l.strip()]
for i in range(n):
    open(f"{dst}/shard_{i}.jsonl","w").write("\n".join(lines[i::n])+"\n")
print(f"split {len(lines)} into {n} shards")
PY
PIDS=()
for i in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$i python $REPO/trace-pilot/src/eval/run_qwen_video.py \
    --model-id "$MID" --input "$SD/shard_$i.jsonl" --output "$SD/preds_$i.jsonl" \
    --max-new-tokens 1024 > "$SD/log_$i.txt" 2>&1 &
  PIDS+=($!)
done
for p in "${PIDS[@]}"; do wait $p || true; done
cat "$SD/"preds_*.jsonl > $OUT/self_distill_predictions.jsonl
echo "[qwen35] self-distill done: $(wc -l < $OUT/self_distill_predictions.jsonl) preds"

# 2) Format: Qwen3.5 self-distilled ambig + shared rehearsal
echo "[qwen35] formatting..."
python $REPO/trace-pilot/src/format_sft_v2.py \
  --self-distill-input  $V2DIR/self_distill_input.jsonl \
  --self-distill-preds  $OUT/self_distill_predictions.jsonl \
  --rehearsal           $V2DIR/rehearsal.jsonl \
  --out-dir             $OUT

# 3) Train
echo "[qwen35] training..."
SKIP_POST_VALIDATION=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 NUM_GPUS=6 PYTORCH_ALLOC_CONF=expandable_segments:True \
  bash $REPO/trace-pilot/scripts/launch_sft.sh $REPO/trace-pilot/configs/sft_lora_qwen35.yaml
until [ -f "$ADAPTER/adapter_model.safetensors" ]; do sleep 30; done
echo "[qwen35] training done"

# 4) Eval — base (no adapter) + v3 (adapter), STEMO + regression
echo "[qwen35] eval: base STEMO..."
MODEL_ID="$MID" NGPU=8 bash $REPO/trace-pilot/scripts/run_stemo_ambig_eval_sharded.sh qwen35_base ""
echo "[qwen35] eval: sft STEMO..."
MODEL_ID="$MID" NGPU=8 bash $REPO/trace-pilot/scripts/run_stemo_ambig_eval_sharded.sh $TAG "$ADAPTER"

echo "[qwen35] eval: regression (base + sft, VideoMME/MVBench)..."
for B in videomme mvbench; do
  GB=duration; [ "$B" = mvbench ] && GB=task
  MODEL_ID="$MID" GROUP_BY=$GB GPUS=0,1,2,3 bash $REPO/trace-pilot/scripts/run_mcq_eval_sharded.sh $B qwen35_base "" &
  MODEL_ID="$MID" GROUP_BY=$GB GPUS=4,5,6,7 bash $REPO/trace-pilot/scripts/run_mcq_eval_sharded.sh $B $TAG "$ADAPTER" &
  wait
done

echo
echo "============ Qwen3.5-27B: base vs v3 ============"
python - <<'PY'
import json
R='/home/thong/weride_project/weride/overthinking_hallu/eval_runs'
print("STEMO-Ambig:")
for t,l in [('qwen35_base','base'),('qwen35_v3','v3')]:
    try:
        m=json.load(open(f'{R}/{t}/stemo_ambig_metrics.json'))['overall']
        print(f'  {l}: enum={m["enumeration_rate"]:.3f} commit={m["single_commit_rate"]:.3f} strict={m["strict_ambig_aware_accuracy"]:.3f}')
    except: print(f'  {l}: missing')
print("Regression:")
for b in ('videomme','mvbench'):
    try:
        mb=json.load(open(f'{R}/qwen35_base/{b}_metrics.json'))['accuracy']
        mv=json.load(open(f'{R}/qwen35_v3/{b}_metrics.json'))['accuracy']
        print(f'  {b}: base={mb:.3f} v3={mv:.3f} delta={mv-mb:+.3f}')
    except: print(f'  {b}: missing')
PY
