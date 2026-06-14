#!/usr/bin/env bash
set -uo pipefail
REPO=/home/thong/weride_project/weride/overthinking_hallu
cd $REPO
LOG=$REPO/tmp/internvl_v3_queue.log
log(){ echo "[$(date -u +%FT%TZ)] $*" >> "$LOG"; echo "[$(date -u +%FT%TZ)] $*"; }
log "v3 queue START — waiting for ladder (v4+v5) GPU jobs to drain"
# Wait until the ladder queue is done (no GPU training/eval procs)
while pgrep -f "internvl_ladder_queue|run_internvl_hf_video|train_sft|accelerate.commands|run_qwen_video" >/dev/null 2>&1; do
  sleep 300
done
log "GPUs free. Running InternVL v3 for both sizes."
for TAG in internvl8b_hf internvl38b_hf; do
  if [ -f "$REPO/eval_runs/${TAG}_v3/stemo_ambig_metrics.json" ]; then
    log "$TAG v3 already done, skipping"; continue
  fi
  log "=== $TAG v3 start ==="
  bash $REPO/trace-pilot/scripts/internvl_v3.sh $TAG > $REPO/tmp/v3_${TAG}.log 2>&1
  log "=== $TAG v3 done; exit=$? ==="
done
log "v3 queue END"
python $REPO/tools/dashboard.py >/dev/null 2>&1
cp $REPO/STATUS.md $REPO/dashboard_repo/STATUS.md 2>/dev/null
cd $REPO/dashboard_repo && git add STATUS.md && git -c user.name="dashboard" -c user.email="bot@local" commit -m "InternVL v3 complete — full ladder done" -q 2>/dev/null && git push -q 2>&1 || true
