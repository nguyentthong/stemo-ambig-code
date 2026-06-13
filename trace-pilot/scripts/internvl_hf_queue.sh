#!/usr/bin/env bash
# InternVL (native-HF) queue. Adds a genuine second open-weight architecture
# family to the paper. Uses the -HF variants (InternVLForConditionalGeneration),
# which load through the standard Qwen runner/trainer — the custom-code variants
# are incompatible with transformers 5.x.
#
# Waits for the completion queue's final cell (qwen36 maximal/explicit metrics)
# so it never competes for GPUs, then runs base+v4 chains for 8B then 38B.
# chain_v4.sh internally evaluates BOTH base and v4 (+ VideoMME + MVBench).
set -uo pipefail
REPO=/home/thong/weride_project/weride/overthinking_hallu
cd $REPO
LOG=$REPO/tmp/internvl_hf_queue.log
log() { echo "[$(date -u +%FT%TZ)] $*" >> "$LOG"; echo "[$(date -u +%FT%TZ)] $*"; }

log "InternVL-HF queue START — waiting for completion queue's last cell"
# Final cell of the completion queue is qwen36 maximal (explicit) prompt-sensitivity
while [ ! -f "$REPO/eval_runs/qwen36_prompt_explicit/stemo_ambig_metrics.json" ]; do
  sleep 120
done
log "completion queue done (qwen36_prompt_explicit present). Starting InternVL-HF."

# Guard: make sure no GPU jobs are still running before we grab all 8 GPUs
while pgrep -f "run_qwen_video|train_sft|accelerate.commands" >/dev/null 2>&1; do
  log "waiting for residual GPU jobs to drain..."
  sleep 120
done

for TAG in internvl8b_hf internvl38b_hf; do
  if [ -f "$REPO/eval_runs/${TAG}_v4/stemo_ambig_metrics.json" ]; then
    log "$TAG already done, skipping"
    continue
  fi
  log "=== $TAG v4 chain (base + v4 + VideoMME + MVBench) start ==="
  bash $REPO/trace-pilot/scripts/chain_v4.sh $TAG > $REPO/tmp/v4_${TAG}.log 2>&1
  log "=== $TAG chain done; exit=$? ==="
done

log "InternVL-HF queue END"
# Refresh dashboard
python $REPO/tools/dashboard.py > /dev/null 2>&1
cp $REPO/STATUS.md $REPO/dashboard_repo/STATUS.md 2>/dev/null
cd $REPO/dashboard_repo && git add STATUS.md && \
  git -c user.name="dashboard" -c user.email="bot@local" commit -m "InternVL-HF base+v4 complete" -q 2>/dev/null && git push -q 2>&1 || true
