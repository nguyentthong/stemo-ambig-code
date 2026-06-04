#!/usr/bin/env bash
# Complete the base (Qwen3-VL-32B-Thinking, no adapter) eval matrix:
# STEMO-Ambig + TempCompass + VidHalluc (VideoMME + MVBench base already done).

set -uo pipefail
REPO=/home/thong/weride_project/weride/overthinking_hallu
TAG=base

# 1) STEMO-Ambig — base, no adapter (8-way sharded + judge)
echo "[base] STEMO-Ambig..."
NGPU=8 bash $REPO/trace-pilot/scripts/run_stemo_ambig_eval_sharded.sh $TAG ""

# 2) TempCompass + VidHalluc base in parallel
echo "[base] TempCompass + VidHalluc..."
GROUP_BY=dim GPUS=0,1,2,3 bash $REPO/trace-pilot/scripts/run_mcq_eval_sharded.sh tempcompass $TAG "" &
P1=$!
KIND=yesno GROUP_BY=subtask GPUS=4,5,6,7 bash $REPO/trace-pilot/scripts/run_mcq_eval_sharded.sh vidhalluc_2k $TAG "" &
P2=$!
wait $P1 $P2 || true

# Final complete matrix
echo
echo "================ COMPLETE MATRIX: base | v3 ================"
python - <<'PY'
import json
ROOT='/home/thong/weride_project/weride/overthinking_hallu/eval_runs'
print("REGRESSION:")
print(f"{'bench':<14}{'base':>9}{'v3':>9}{'delta':>9}")
for b in ('videomme','mvbench','tempcompass','vidhalluc_2k'):
    a={}
    for t in ('base','sft_v3_final'):
        try: a[t]=json.load(open(f'{ROOT}/{t}/{b}_metrics.json'))['accuracy']
        except: a[t]=None
    fmt=lambda v: f'{v:.3f}' if v is not None else '  -  '
    d='' if a['base'] is None or a['sft_v3_final'] is None else f'{a["sft_v3_final"]-a["base"]:+.3f}'
    print(f'{b:<14}{fmt(a["base"]):>9}{fmt(a["sft_v3_final"]):>9}{d:>9}')

print("\nSTEMO-Ambig (target):")
print(f"{'model':<8}{'enum':>8}{'strict':>8}{'overall':>9}")
for t,l in [('base','base'),('sft_v3_final','v3')]:
    try:
        m=json.load(open(f'{ROOT}/{t}/stemo_ambig_metrics.json'))['overall']
        print(f'{l:<8}{m["enumeration_rate"]:>8.3f}{m["strict_ambig_aware_accuracy"]:>8.3f}{m["per_interp_accuracy_overall"]:>9.3f}')
    except: print(f'{l:<8} missing')
PY
