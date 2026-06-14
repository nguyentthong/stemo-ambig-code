#!/usr/bin/env bash
set -uo pipefail
REPO=/home/thong/weride_project/weride/overthinking_hallu
cd $REPO
LOG=$REPO/tmp/internvl_8b_mcq_rerun.log
log(){ echo "[$(date -u +%FT%TZ)] $*" >> "$LOG"; echo "[$(date -u +%FT%TZ)] $*"; }
MID="OpenGVLab/InternVL3_5-8B-HF"
ADAPTER=$REPO/checkpoints/internvl8b_hf_stemo_ambig_lora_v4
log "8B MCQ rerun — waiting for GPUs free"
while pgrep -f "run_internvl_hf_video|train_sft|accelerate.commands|run_qwen_video" >/dev/null 2>&1; do sleep 300; done
log "GPUs free — rerunning videomme+mvbench (base + v4) with internvl_hf runner"
for B in videomme mvbench; do
  GB=duration; [ "$B" = mvbench ] && GB=task
  MODEL_ID="$MID" RUNNER_FAMILY=internvl_hf GROUP_BY=$GB GPUS=0,1,2,3 bash $REPO/trace-pilot/scripts/run_mcq_eval_sharded.sh $B internvl8b_hf_v4_base "" &
  MODEL_ID="$MID" RUNNER_FAMILY=internvl_hf GROUP_BY=$GB GPUS=4,5,6,7 bash $REPO/trace-pilot/scripts/run_mcq_eval_sharded.sh $B internvl8b_hf_v4 "$ADAPTER" &
  wait
done
log "8B MCQ rerun done"
python $REPO/tools/dashboard.py >/dev/null 2>&1
cp $REPO/STATUS.md $REPO/dashboard_repo/STATUS.md 2>/dev/null
cd $REPO/dashboard_repo && git add STATUS.md && git -c user.name="dashboard" -c user.email="bot@local" commit -m "InternVL 8B VideoMME/MVBench rerun (fixed runner)" -q 2>/dev/null && git push -q 2>&1 || true
