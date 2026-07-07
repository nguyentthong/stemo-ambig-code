#!/usr/bin/env bash
# One-command open-weight evaluation for the ReQueST paper figures.
#
#   GEMINI_API_KEY=... nohup bash server_eval/run_all.sh > run_all.log 2>&1 &
#
# Smoke test first (~5 min, 6 items on the 8B model):
#   GEMINI_API_KEY=... bash server_eval/run_all.sh smoke
#
# Runs the 5 open-weight BASE models through the IAA protocol via vLLM
# (tp=8, sequential models), judges inline with Gemini, resumes from
# partial output on rerun, falls back to the HF sharded runner if a
# model's vLLM server fails to come up, and finishes by printing the
# figure-ready data blocks (aggregate_metrics.py).
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
mkdir -p eval_runs tmp

# no CUDA toolkit (nvcc) on the box: FlashInfer cannot JIT its sampling
# kernel, so force vLLM's native torch sampler (we decode greedily anyway)
export VLLM_USE_FLASHINFER_SAMPLER=0

# model weights live in RAM: the 193G root disk cannot hold the ~265G of
# checkpoints, but /dev/shm is a 1TB tmpfs on this box (lost on reboot,
# re-downloaded on demand — fine for eval runs)
export HF_HOME="${HF_HOME:-/dev/shm/hf_eval}"
mkdir -p "$HF_HOME"

SMOKE=""
[ "${1:-}" = "smoke" ] && SMOKE=1

# ---------- preflight ----------
[ -n "${GEMINI_API_KEY:-}" ] || { echo "FATAL: export GEMINI_API_KEY (judge)"; exit 1; }
[ -f data_v0/stemo_ambig_candidates/all_questions.json ] || { echo "FATAL: all_questions.json missing"; exit 1; }
N_VIDEOS=$(ls stemo/videos_h264/*.mp4 2>/dev/null | wc -l)
[ "$N_VIDEOS" -gt 0 ] || { echo "FATAL: no videos in stemo/videos_h264 (rsync them first)"; exit 1; }
echo "[run_all] $N_VIDEOS videos present"

# ---------- environment ----------
if [ ! -d .venv_eval ]; then
  echo "[run_all] creating venv + installing deps (one-time, ~10 min)"
  python3 -m venv .venv_eval
  ./.venv_eval/bin/pip install -q --upgrade pip
  ./.venv_eval/bin/pip install -q "vllm>=0.8" openai google-genai pillow \
      "transformers>=4.51" accelerate || { echo "FATAL: pip install failed"; exit 1; }
  # decord has no wheels on some pythons; eva-decord is a drop-in fork
  ./.venv_eval/bin/pip install -q decord || ./.venv_eval/bin/pip install -q eva-decord \
      || { echo "FATAL: decord install failed"; exit 1; }
fi
PY="$REPO/.venv_eval/bin/python"

# judge sanity check: fail fast on a bad key, not 90 min into the run
$PY - <<'PYEOF' || { echo "FATAL: Gemini judge call failed (check GEMINI_API_KEY)"; exit 1; }
import sys
sys.path.insert(0, "trace-pilot/src")
from iaa.sub_judge import classify_turn1
r = classify_turn1("Does the boy fall?",
                   [{"referent_description": "the boy in red"},
                    {"referent_description": "the boy in blue"}],
                   "Yes, he falls.")
assert r.get("category"), r
print("[run_all] judge OK:", r["category"])
PYEOF

# expected item count (dataset questions whose video is present) drives the
# "model complete" / fallback thresholds instead of a hardcoded 1000
N_EXPECTED=$($PY - <<'PYEOF' | tail -1
import sys
sys.path.insert(0, "server_eval")
from run_iaa_vllm import build_items
print(len(build_items()))
PYEOF
)
MIN_ROWS=$(( N_EXPECTED * 95 / 100 ))
echo "[run_all] expecting $N_EXPECTED items; completeness threshold $MIN_ROWS rows"

# ---------- model queue: tag | hf_id | frames | tensor_parallel ----------
# InternVL runs at 8 frames (paper setting), others at 16.
# TP sized per model (8B fits one GPU; NCCL only where needed).
QUEUE=(
  "internvl8b|OpenGVLab/InternVL3_5-8B-HF|8|1"
  "qwen35_27b|Qwen/Qwen3.5-27B|16|4"
  "qwen36_27b|Qwen/Qwen3.6-27B|16|4"
  "qwen3vl_32b|Qwen/Qwen3-VL-32B-Thinking|16|4"
  "internvl38b|OpenGVLab/InternVL3_5-38B-HF|8|8"
)

# Lane overrides (used by run_parallel.sh):
#   RUN_ONLY=<tag>[,<tag>]  restrict the queue
#   GPUS=<id>[,<id>]        pin server + fallback to these GPUs
#   PORT=<port>             per-lane server port
#   SKIP_AGG=1              skip final aggregation (wrapper aggregates)
PORT="${PORT:-8199}"
if [ -n "${GPUS:-}" ]; then
  export CUDA_VISIBLE_DEVICES="$GPUS"
fi
LIMIT_ARG=""
if [ -n "$SMOKE" ]; then
  # SMOKE_MODEL=<tag> picks a different queue entry (default: 8B InternVL)
  SMOKE_TAG="${SMOKE_MODEL:-internvl8b}"
  PICKED=""
  for e in "${QUEUE[@]}"; do [[ "$e" == "$SMOKE_TAG|"* ]] && PICKED="$e"; done
  [ -n "$PICKED" ] || { echo "FATAL: unknown SMOKE_MODEL=$SMOKE_TAG"; exit 1; }
  QUEUE=("$PICKED"); LIMIT_ARG="--limit 6"
fi
if [ -n "${RUN_ONLY:-}" ]; then
  FILTERED=()
  for e in "${QUEUE[@]}"; do
    IFS='|' read -r t _ <<< "$e"
    [[ ",$RUN_ONLY," == *",$t,"* ]] && FILTERED+=("$e")
  done
  [ ${#FILTERED[@]} -gt 0 ] || { echo "FATAL: RUN_ONLY=$RUN_ONLY matched nothing"; exit 1; }
  QUEUE=("${FILTERED[@]}")
fi

wait_for_server() {  # $1=pid  $2=log
  local waited=0
  while true; do
    curl -sf "http://localhost:$PORT/v1/models" >/dev/null 2>&1 && return 0
    if ! kill -0 "$1" 2>/dev/null; then
      echo "[run_all] server process died. FIRST traceback in $2:"
      awk '/Traceback|ERROR|CUDA out of memory/{found=1} found' "$2" | head -50
      return 1
    fi
    sleep 15; waited=$((waited+15))
    # generous: first run downloads weights from HF hub
    [ $waited -ge 5400 ] && { echo "[run_all] server timeout after 90 min"; return 1; }
  done
}

run_hf_fallback() {  # $1=tag $2=hf_id $3=frames
  echo "[run_all] FALLBACK: HF sharded runner for $1"
  local shard_dir="eval_runs/$1/iaa_shards"
  mkdir -p "$shard_dir"
  # one shard per lane GPU (all 8 when no GPUS override)
  IFS=',' read -ra GPU_IDS <<< "${GPUS:-0,1,2,3,4,5,6,7}"
  local n=${#GPU_IDS[@]}
  $PY - "$1" "$n" <<'PYEOF'
import json, sys
from pathlib import Path
sys.path.insert(0, "server_eval")
from run_iaa_vllm import build_items
rows = build_items()
n = int(sys.argv[2])
sd = Path("eval_runs") / sys.argv[1] / "iaa_shards"
for i in range(n):
    with (sd / f"shard_{i}.jsonl").open("w") as f:
        for r in rows[i::n]:
            f.write(json.dumps(r) + "\n")
PYEOF
  local pids=()
  for idx in "${!GPU_IDS[@]}"; do
    CUDA_VISIBLE_DEVICES=${GPU_IDS[$idx]} $PY -u trace-pilot/src/iaa/run_iaa_open.py \
      --model-id "$2" \
      --input "$shard_dir/shard_${idx}.jsonl" \
      --output "$shard_dir/preds_${idx}.jsonl" \
      > "tmp/hf_${1}_shard${idx}.log" 2>&1 &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p" || true; done
  cat "$shard_dir"/preds_*.jsonl > "eval_runs/$1/iaa_predictions.jsonl"
}

# ---------- main queue ----------
DONE_TAGS=()
for entry in "${QUEUE[@]}"; do
  IFS='|' read -r TAG HF_ID FRAMES TP <<< "$entry"
  # lane override (single-model lanes): shrink/grow TP to the lane's GPUs
  [ -n "${TP_OVERRIDE:-}" ] && TP="$TP_OVERRIDE"
  OUT="eval_runs/$TAG/iaa_predictions.jsonl"
  mkdir -p "eval_runs/$TAG"

  # skip if complete (>= 95% of expected rows); FORCE=1 reruns anyway
  # (the driver's resume keeps clean rows and retries only errored ones)
  if [ -z "$SMOKE" ] && [ -z "${FORCE:-}" ] && [ -f "$OUT" ] && [ "$(wc -l < "$OUT")" -ge "$MIN_ROWS" ]; then
    echo "[run_all] $TAG already complete, skipping"
    DONE_TAGS+=("$TAG"); continue
  fi

  echo "[run_all] ===== $TAG ($HF_ID, ${FRAMES}f, tp=$TP) start $(date) ====="
  SERVER_LOG="tmp/vllm_${TAG}.log"
  # InternVL -HF conversions omit tie_word_embeddings=false in config.json;
  # vLLM then ties lm_head to embed_tokens and generates gibberish (the
  # checkpoint has a distinct lm_head). Must be top-level: a text_config-
  # nested override is clobbered by get_text_config() propagation.
  EXTRA_ARGS=()
  [[ "$TAG" == internvl* ]] && EXTRA_ARGS=(--hf-overrides '{"tie_word_embeddings": false}')
  $PY -m vllm.entrypoints.openai.api_server \
      --model "$HF_ID" --served-model-name "$TAG" \
      --tensor-parallel-size "$TP" --port "$PORT" \
      --limit-mm-per-prompt '{"image": 16}' \
      --max-model-len 32768 --gpu-memory-utilization 0.90 \
      --enforce-eager --no-enable-prefix-caching \
      "${EXTRA_ARGS[@]}" \
      --trust-remote-code > "$SERVER_LOG" 2>&1 &
  SERVER_PID=$!

  if wait_for_server "$SERVER_PID" "$SERVER_LOG"; then
    $PY -u server_eval/run_iaa_vllm.py \
        --base-url "http://localhost:$PORT/v1" --served-name "$TAG" \
        --frames "$FRAMES" --output "$OUT" --concurrency 16 $LIMIT_ARG \
        2>&1 | tee "tmp/driver_${TAG}.log"
    DRIVER_OK=$?
  else
    DRIVER_OK=1
  fi

  kill "$SERVER_PID" 2>/dev/null; sleep 20
  # scope to this lane's server: a bare pkill would kill parallel lanes too
  pkill -f "vllm.entrypoints.*served-model-name $TAG" 2>/dev/null; sleep 10

  if [ -z "$SMOKE" ]; then
    ROWS=$(wc -l < "$OUT" 2>/dev/null || echo 0)
    if [ "$DRIVER_OK" -ne 0 ] || [ "$ROWS" -lt "$MIN_ROWS" ]; then
      run_hf_fallback "$TAG" "$HF_ID" "$FRAMES"
    fi
  fi
  DONE_TAGS+=("$TAG")
  echo "[run_all] ===== $TAG done $(date) ====="
done

# ---------- aggregate ----------
if [ -z "${SKIP_AGG:-}" ]; then
  echo "[run_all] aggregating $(date)"
  $PY server_eval/aggregate_metrics.py "${DONE_TAGS[@]}"
fi
echo "[run_all] ALL DONE $(date)"
