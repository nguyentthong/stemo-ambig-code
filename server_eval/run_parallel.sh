#!/usr/bin/env bash
# Parallel-lane wrapper around run_all.sh: fits the 5-model queue onto
# 8 GPUs in 3 phases instead of running them back to back.
#
#   GEMINI_API_KEY=... nohup bash server_eval/run_parallel.sh > run_parallel.log 2>&1 &
#
#   phase 1: qwen35_27b  (GPUs 0-3, port 8199) || qwen36_27b (GPUs 4-7, port 8299)
#   phase 2: qwen3vl_32b (GPUs 0-3, port 8199) || internvl8b (GPU 4,   port 8299)
#   phase 3: internvl38b (all 8 GPUs)
#   then aggregate.
#
# Each lane is a run_all.sh invocation with RUN_ONLY/GPUS/PORT/SKIP_AGG,
# so all of its behavior (resume, judge, HF fallback on lane GPUs) carries
# over. Rerunning this script is safe for the same reasons run_all.sh is.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
[ -n "${GEMINI_API_KEY:-}" ] || { echo "FATAL: export GEMINI_API_KEY (judge)"; exit 1; }
mkdir -p tmp
PY="$REPO/.venv_eval/bin/python"

# weights go to RAM (see run_all.sh); export here so the prefetch matches
export HF_HOME="${HF_HOME:-/dev/shm/hf_eval}"
mkdir -p "$HF_HOME"

# prefetch phase-2/3 weights while phase 1 runs (network only, no GPU)
(
  for m in Qwen/Qwen3-VL-32B-Thinking OpenGVLab/InternVL3_5-38B-HF; do
    $PY -c "from huggingface_hub import snapshot_download; snapshot_download('$m')" \
      && echo "[prefetch] $m done" || echo "[prefetch] $m FAILED (server will retry)"
  done
) > tmp/prefetch.log 2>&1 &
PREFETCH_PID=$!

lane() {  # $1=tag(s) $2=gpus $3=port
  RUN_ONLY="$1" GPUS="$2" PORT="$3" SKIP_AGG=1 bash server_eval/run_all.sh
}

echo "[parallel] phase 1+2 start $(date)"
( lane qwen35_27b 0,1,2,3 8199; lane qwen3vl_32b 0,1,2,3 8199 ) > tmp/laneA.log 2>&1 &
LANE_A=$!
( lane qwen36_27b 4,5,6,7 8299; lane internvl8b 4 8299 ) > tmp/laneB.log 2>&1 &
LANE_B=$!
echo "[parallel] lane A (qwen35_27b -> qwen3vl_32b) pid=$LANE_A"
echo "[parallel] lane B (qwen36_27b -> internvl8b) pid=$LANE_B"
wait "$LANE_A"; echo "[parallel] lane A done $(date)"
wait "$LANE_B"; echo "[parallel] lane B done $(date)"

echo "[parallel] phase 3: internvl38b on all 8 GPUs $(date)"
lane internvl38b 0,1,2,3,4,5,6,7 8199 2>&1 | tee tmp/laneC.log

kill "$PREFETCH_PID" 2>/dev/null || true
echo "[parallel] aggregating $(date)"
$PY server_eval/aggregate_metrics.py internvl8b qwen35_27b qwen36_27b qwen3vl_32b internvl38b
echo "[parallel] ALL DONE $(date)"
