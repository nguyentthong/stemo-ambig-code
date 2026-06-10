#!/usr/bin/env bash
# Completion queue: runs every remaining experiment cell so the dashboard hits 100%.
# Remaining (10 cells):
#   1. qwen36_9b base eval                  (~2h)
#   2. qwen36_9b v3 SFT + eval              (~6h)
#   3. qwen36_9b v4 chain                   (~10h)
#   4. qwen36_9b FFT v4 + eval              (~10h, reuses v4 sft_train.jsonl)
#   5-10. prompt-sensitivity 6 configs      (~3h each, qwen35/qwen36 × neutral/fewshot/explicit)
set -uo pipefail
REPO=/home/thong/weride_project/weride/overthinking_hallu
cd $REPO
LOG=$REPO/tmp/completion_queue.log
log() { echo "[$(date -u +%FT%TZ)] $*" >> "$LOG"; echo "[$(date -u +%FT%TZ)] $*"; }

log "completion queue START"

# 1+2) qwen36_9b base eval + v3 (script already patched: Qwen3.5-9B, internvl dropped)
if [ ! -f $REPO/eval_runs/qwen36_9b_v3/stemo_ambig_metrics.json ]; then
  log "step 1-2: qwen36_9b base + v3"
  bash $REPO/tmp/small_model_base_v3.sh > $REPO/tmp/small_model_base_v3.log 2>&1
  log "step 1-2 done; exit=$?"
else
  log "step 1-2 already done, skipping"
fi

# 3) qwen36_9b v4 chain
if [ ! -f $REPO/eval_runs/qwen36_9b_v4/stemo_ambig_metrics.json ]; then
  log "step 3: qwen36_9b v4 chain"
  bash $REPO/trace-pilot/scripts/chain_v4.sh qwen36_9b > $REPO/tmp/v4_qwen36_9b.log 2>&1
  log "step 3 done; exit=$?"
else
  log "step 3 already done, skipping"
fi

# 4) qwen36_9b FFT (needs v4 sft_train.jsonl from step 3)
if [ ! -f $REPO/eval_runs/qwen36_9b_fft_v4/stemo_ambig_metrics.json ]; then
  log "step 4: qwen36_9b FFT"
  bash $REPO/tmp/fft_variant.sh > $REPO/tmp/fft_variant.log 2>&1
  log "step 4 done; exit=$?"
else
  log "step 4 already done, skipping"
fi

# 5-10) open-weight prompt-sensitivity (6 configs)
NEED_PS=0
for t in qwen35_prompt_neutral qwen35_prompt_fewshot qwen35_prompt_explicit \
         qwen36_prompt_neutral qwen36_prompt_fewshot qwen36_prompt_explicit; do
  [ -f $REPO/eval_runs/$t/stemo_ambig_metrics.json ] || NEED_PS=1
done
if [ $NEED_PS -eq 1 ]; then
  log "step 5-10: open-weight prompt-sensitivity (6 configs)"
  bash $REPO/tmp/prompt_sensitivity_ablation.sh > $REPO/tmp/prompt_sensitivity_ablation.log 2>&1
  log "step 5-10 done; exit=$?"
else
  log "step 5-10 already done, skipping"
fi

log "completion queue END — refreshing dashboard"
python $REPO/tools/dashboard.py > $REPO/STATUS.md 2>&1
cp $REPO/STATUS.md $REPO/dashboard_repo/STATUS.md
cd $REPO/dashboard_repo && git add STATUS.md && \
  git -c user.name="dashboard" -c user.email="bot@local" commit -m "completion queue finished" -q && git push -q
log "dashboard pushed"
