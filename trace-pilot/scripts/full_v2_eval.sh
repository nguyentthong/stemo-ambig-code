#!/usr/bin/env bash
# Full v2 evaluation: STEMO-Ambig + 4 regression benchmarks.
# Tag = sft_v2_final. All inference fresh (no resume from v1).

set -uo pipefail

REPO=/home/thong/weride_project/weride/overthinking_hallu
TAG=sft_v2_final
ADAPTER=$REPO/checkpoints/qwen3vl32b_stemo_ambig_lora_v2

OUT=$REPO/eval_runs/$TAG
mkdir -p "$OUT"

# Prep STEMO-Ambig input if not already there
[ -f "$OUT/stemo_ambig_test.jsonl" ] || python $REPO/trace-pilot/src/eval/prep_stemo_ambig_test.py --out "$OUT/stemo_ambig_test.jsonl"

# 1) STEMO-Ambig sharded inference (8-way) + judge
echo "[v2 eval] STEMO-Ambig sharded inference..."
NGPU=8 bash $REPO/trace-pilot/scripts/run_stemo_ambig_eval_sharded.sh $TAG $ADAPTER

# 2) Regression benchmarks: launch base+sft in parallel where possible
echo "[v2 eval] launching VideoMME..."
GPUS=0,1,2,3 bash $REPO/trace-pilot/scripts/run_mcq_eval_sharded.sh videomme $TAG $ADAPTER &
P1=$!
echo "[v2 eval] launching MVBench (GPUs 4-7)..."
GROUP_BY=task GPUS=4,5,6,7 bash $REPO/trace-pilot/scripts/run_mcq_eval_sharded.sh mvbench $TAG $ADAPTER &
P2=$!
wait $P1 $P2 || true

echo "[v2 eval] launching TempCompass..."
GROUP_BY=dim GPUS=0,1,2,3 bash $REPO/trace-pilot/scripts/run_mcq_eval_sharded.sh tempcompass $TAG $ADAPTER &
P1=$!
echo "[v2 eval] launching VidHalluc (yes/no)..."
KIND=yesno GROUP_BY=subtask GPUS=4,5,6,7 bash $REPO/trace-pilot/scripts/run_mcq_eval_sharded.sh vidhalluc_2k $TAG $ADAPTER &
P2=$!
wait $P1 $P2 || true

# Final summary
echo
echo "============================ V2 vs V1 vs BASE ============================"
python - <<'PY'
import json, os
ROOT = "/home/thong/weride_project/weride/overthinking_hallu/eval_runs"
benches = ["videomme", "mvbench", "tempcompass", "vidhalluc_2k"]
print(f"{'benchmark':<14}{'base':>10}{'v1':>10}{'v2':>10}{'v2-base':>12}")
for b in benches:
    accs = {}
    for tag in ("base", "sft_v1_final", "sft_v2_final"):
        try:
            m = json.load(open(f"{ROOT}/{tag}/{b}_metrics.json"))
            accs[tag] = m['accuracy']
        except: accs[tag] = None
    fmt = lambda v: f"{v:.3f}" if v is not None else "  -  "
    delta = "" if accs.get('sft_v2_final') is None or accs.get('base') is None else f"{accs['sft_v2_final']-accs['base']:+.3f}"
    print(f"{b:<14}{fmt(accs['base']):>10}{fmt(accs['sft_v1_final']):>10}{fmt(accs['sft_v2_final']):>10}{delta:>12}")

# STEMO
print("\n=== STEMO-Ambig ===")
for tag in ("sft_v1_final","sft_v2_final"):
    try:
        m = json.load(open(f"{ROOT}/{tag}/stemo_ambig_metrics.json"))["overall"]
        print(f"  {tag}: enum={m['enumeration_rate']:.3f} commit={m['single_commit_rate']:.3f} strict={m['strict_ambig_aware_accuracy']:.3f} per_interp={m['per_interp_accuracy_overall']:.3f}")
    except: print(f"  {tag}: missing")
PY
