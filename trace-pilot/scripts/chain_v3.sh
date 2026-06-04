#!/usr/bin/env bash
# Train v3 then auto-fire full evaluation.

set -uo pipefail
REPO=/home/thong/weride_project/weride/overthinking_hallu

# 1) Train v3
echo "[v3] launching training (LR 5e-5, 2 epochs)..."
SKIP_POST_VALIDATION=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 NUM_GPUS=6 PYTORCH_ALLOC_CONF=expandable_segments:True \
  bash $REPO/trace-pilot/scripts/launch_sft.sh $REPO/trace-pilot/configs/sft_lora_v3.yaml

# 2) Wait until checkpoint files exist
ADAPTER=$REPO/checkpoints/qwen3vl32b_stemo_ambig_lora_v3
until [ -f "$ADAPTER/adapter_model.safetensors" ]; do
  sleep 30
done
echo "[v3] training done, launching full v3 evaluation (tag=sft_v3_final)"

# 3) Full v3 evaluation — same script as v2, just different tag
TAG=sft_v3_final
mkdir -p $REPO/eval_runs/$TAG

# STEMO-Ambig sharded
NGPU=8 bash $REPO/trace-pilot/scripts/run_stemo_ambig_eval_sharded.sh $TAG $ADAPTER

# VideoMME + MVBench parallel
GPUS=0,1,2,3 bash $REPO/trace-pilot/scripts/run_mcq_eval_sharded.sh videomme $TAG $ADAPTER &
P1=$!
GROUP_BY=task GPUS=4,5,6,7 bash $REPO/trace-pilot/scripts/run_mcq_eval_sharded.sh mvbench $TAG $ADAPTER &
P2=$!
wait $P1 $P2 || true

# TempCompass + VidHalluc parallel
GROUP_BY=dim GPUS=0,1,2,3 bash $REPO/trace-pilot/scripts/run_mcq_eval_sharded.sh tempcompass $TAG $ADAPTER &
P1=$!
KIND=yesno GROUP_BY=subtask GPUS=4,5,6,7 bash $REPO/trace-pilot/scripts/run_mcq_eval_sharded.sh vidhalluc_2k $TAG $ADAPTER &
P2=$!
wait $P1 $P2 || true

# Final 3-way comparison
echo
echo "================ FINAL: base | v1 | v2 | v3 ================"
python - <<'PY'
import json
ROOT='/home/thong/weride_project/weride/overthinking_hallu/eval_runs'
print(f"{'bench':<14}{'base':>10}{'v1':>10}{'v2':>10}{'v3':>10}{'v3-base':>12}")
for b in ('videomme','mvbench','tempcompass','vidhalluc_2k'):
    a={}
    for t in ('base','sft_v1_final','sft_v2_final','sft_v3_final'):
        try: a[t]=json.load(open(f'{ROOT}/{t}/{b}_metrics.json'))['accuracy']
        except: a[t]=None
    fmt=lambda v: f'{v:.3f}' if v is not None else '  -  '
    d='' if a.get('sft_v3_final') is None or a.get('base') is None else f'{a["sft_v3_final"]-a["base"]:+.3f}'
    print(f'{b:<14}{fmt(a["base"]):>10}{fmt(a.get("sft_v1_final")):>10}{fmt(a.get("sft_v2_final")):>10}{fmt(a.get("sft_v3_final")):>10}{d:>12}')

print("\nSTEMO-Ambig:")
for t in ('sft_v1_final','sft_v2_final','sft_v3_final'):
    try:
        m=json.load(open(f'{ROOT}/{t}/stemo_ambig_metrics.json'))['overall']
        print(f'  {t:<18} enum={m["enumeration_rate"]:.3f} commit={m["single_commit_rate"]:.3f} strict={m["strict_ambig_aware_accuracy"]:.3f} per_interp={m["per_interp_accuracy_overall"]:.3f}')
    except Exception as e:
        print(f'  {t:<18} missing ({e})')
PY
