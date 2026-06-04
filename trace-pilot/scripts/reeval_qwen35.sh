#!/usr/bin/env bash
# Re-evaluate Qwen3.5-27B base + v3 with thinking DISABLED (adapter already trained).
set -uo pipefail
REPO=/home/thong/weride_project/weride/overthinking_hallu
MID="Qwen/Qwen3.5-27B"
ADAPTER=$REPO/checkpoints/qwen35_27b_stemo_ambig_lora_v3

echo "[qwen35 re-eval] base STEMO (no-thinking)..."
MODEL_ID="$MID" NO_THINKING=1 NGPU=8 bash $REPO/trace-pilot/scripts/run_stemo_ambig_eval_sharded.sh qwen35_base ""
echo "[qwen35 re-eval] v3 STEMO (no-thinking)..."
MODEL_ID="$MID" NO_THINKING=1 NGPU=8 bash $REPO/trace-pilot/scripts/run_stemo_ambig_eval_sharded.sh qwen35_v3 "$ADAPTER"

echo "[qwen35 re-eval] regression (base + v3, VideoMME/MVBench)..."
for B in videomme mvbench; do
  GB=duration; [ "$B" = mvbench ] && GB=task
  MODEL_ID="$MID" NO_THINKING=1 GROUP_BY=$GB GPUS=0,1,2,3 bash $REPO/trace-pilot/scripts/run_mcq_eval_sharded.sh $B qwen35_base "" &
  MODEL_ID="$MID" NO_THINKING=1 GROUP_BY=$GB GPUS=4,5,6,7 bash $REPO/trace-pilot/scripts/run_mcq_eval_sharded.sh $B qwen35_v3 "$ADAPTER" &
  wait
done

echo
echo "============ Qwen3.5-27B (no-thinking): base vs v3 ============"
python - <<'PY'
import json
R='/home/thong/weride_project/weride/overthinking_hallu/eval_runs'
print("STEMO-Ambig:")
for t,l in [('qwen35_base','base'),('qwen35_v3','v3')]:
    m=json.load(open(f'{R}/{t}/stemo_ambig_metrics.json'))['overall']
    print(f'  {l}: enum={m["enumeration_rate"]:.3f} commit={m["single_commit_rate"]:.3f} strict={m["strict_ambig_aware_accuracy"]:.3f} per_interp={m["per_interp_accuracy_overall"]:.3f}')
print("Regression (base -> v3):")
for b in ('videomme','mvbench'):
    mb=json.load(open(f'{R}/qwen35_base/{b}_metrics.json'))['accuracy']
    mv=json.load(open(f'{R}/qwen35_v3/{b}_metrics.json'))['accuracy']
    print(f'  {b}: {mb:.3f} -> {mv:.3f} ({mv-mb:+.3f})')
PY
