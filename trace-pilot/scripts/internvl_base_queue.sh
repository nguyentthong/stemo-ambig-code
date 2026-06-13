#!/usr/bin/env bash
# InternVL base eval (cross-family base behavior) via the verified -HF multi-image
# runner. Greedy single-sample, the path proven in smoke. base only — v4-SFT dropped
# (needs a training-collator port; SFT generalization already shown on 4 Qwen models).
set -uo pipefail
REPO=/home/thong/weride_project/weride/overthinking_hallu
cd $REPO
LOG=$REPO/tmp/internvl_base_queue.log
log(){ echo "[$(date -u +%FT%TZ)] $*" >> "$LOG"; echo "[$(date -u +%FT%TZ)] $*"; }
log "InternVL base-eval queue START"
for pair in "internvl8b_hf:OpenGVLab/InternVL3_5-8B-HF" "internvl38b_hf:OpenGVLab/InternVL3_5-38B-HF"; do
  TAG="${pair%%:*}"; MID="${pair##*:}"
  if [ -f "$REPO/eval_runs/${TAG}_base/stemo_ambig_metrics.json" ]; then
    log "$TAG base already done, skipping"; continue
  fi
  log "=== $TAG base eval start ($MID) ==="
  MODEL_ID="$MID" NGPU=8 RUNNER_FAMILY=internvl_hf \
    bash $REPO/trace-pilot/scripts/run_stemo_ambig_eval_sharded.sh ${TAG}_base ""
  log "=== $TAG base eval done; exit=$? ==="
done
log "InternVL base-eval queue END"
python $REPO/tools/dashboard.py > /dev/null 2>&1
cp $REPO/STATUS.md $REPO/dashboard_repo/STATUS.md 2>/dev/null
cd $REPO/dashboard_repo && git add STATUS.md && git -c user.name="dashboard" -c user.email="bot@local" commit -m "InternVL base evals complete" -q 2>/dev/null && git push -q 2>&1 || true
