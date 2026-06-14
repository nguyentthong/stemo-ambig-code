#!/usr/bin/env bash
# Re-run 38B v4 evals (STEMO base+v4, VideoMME, MVBench) with the fixed runner
# (_no_split_modules sanitization). Adapter already trained. Waits for GPUs free.
set -uo pipefail
REPO=/home/thong/weride_project/weride/overthinking_hallu
cd $REPO
LOG=$REPO/tmp/internvl_38b_eval_rerun.log
log(){ echo "[$(date -u +%FT%TZ)] $*" >> "$LOG"; echo "[$(date -u +%FT%TZ)] $*"; }
MID="OpenGVLab/InternVL3_5-38B-HF"
ADAPTER=$REPO/checkpoints/internvl38b_hf_stemo_ambig_lora_v4
log "38B v4 eval rerun — waiting for GPUs free"
while pgrep -f "run_internvl_hf_video|train_sft|accelerate.commands|run_qwen_video|chain_v4.sh internvl38b" >/dev/null 2>&1; do sleep 300; done
log "GPUs free — STEMO base+v4 eval (internvl_hf, fixed runner)"
MODEL_ID="$MID" NGPU=8 RUNNER_FAMILY=internvl_hf bash $REPO/trace-pilot/scripts/run_stemo_ambig_eval_sharded.sh internvl38b_hf_v4_base ""
MODEL_ID="$MID" NGPU=8 RUNNER_FAMILY=internvl_hf bash $REPO/trace-pilot/scripts/run_stemo_ambig_eval_sharded.sh internvl38b_hf_v4 "$ADAPTER"
log "STEMO done — MCQ videomme+mvbench"
for B in videomme mvbench; do
  GB=duration; [ "$B" = mvbench ] && GB=task
  MODEL_ID="$MID" RUNNER_FAMILY=internvl_hf GROUP_BY=$GB GPUS=0,1,2,3 bash $REPO/trace-pilot/scripts/run_mcq_eval_sharded.sh $B internvl38b_hf_v4_base "" &
  MODEL_ID="$MID" RUNNER_FAMILY=internvl_hf GROUP_BY=$GB GPUS=4,5,6,7 bash $REPO/trace-pilot/scripts/run_mcq_eval_sharded.sh $B internvl38b_hf_v4 "$ADAPTER" &
  wait
done
log "38B v4 eval rerun done"
python $REPO/tools/dashboard.py >/dev/null 2>&1
cp $REPO/STATUS.md $REPO/dashboard_repo/STATUS.md 2>/dev/null
cd $REPO/dashboard_repo && git add STATUS.md && git -c user.name="dashboard" -c user.email="bot@local" commit -m "38B v4 eval rerun (fixed _no_split runner)" -q 2>/dev/null && git push -q 2>&1 || true
