#!/usr/bin/env bash
# Sharded self-distillation: base Qwen3-VL produces enumeration responses
# given gold scaffold in the prompt. 8 shards parallel.

set -euo pipefail

REPO=/home/thong/weride_project/weride/overthinking_hallu
INPUT="$REPO/data_v0/stemo_ambig_sft_v2/self_distill_input.jsonl"
OUT="$REPO/data_v0/stemo_ambig_sft_v2/self_distill_shards"
mkdir -p "$OUT"

NGPU=8

echo "[self-distill] splitting $INPUT into $NGPU shards"
python - "$INPUT" "$OUT" "$NGPU" <<'PY'
import sys
src, dst_dir, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
lines = [l for l in open(src).read().splitlines() if l.strip()]
print(f"  total {len(lines)}")
for i in range(n):
    sub = lines[i::n]
    open(f"{dst_dir}/shard_{i}.jsonl","w").write("\n".join(sub)+"\n")
    print(f"  shard {i}: {len(sub)}")
PY

echo "[self-distill] launching ${NGPU} inferences"
PIDS=()
for ((i=0; i<NGPU; i++)); do
  CUDA_VISIBLE_DEVICES=$i python "$REPO/trace-pilot/src/eval/run_qwen_video.py" \
    --model-id "${MODEL_ID:-Qwen/Qwen3-VL-32B-Thinking}" \
    --input  "$OUT/shard_$i.jsonl" \
    --output "$OUT/preds_$i.jsonl" \
    --max-new-tokens 1024 \
    > "$OUT/log_$i.txt" 2>&1 &
  PIDS+=($!)
  echo "  shard $i -> pid ${PIDS[$i]} on GPU $i"
done
FAIL=0
for pid in "${PIDS[@]}"; do
  wait "$pid" || FAIL=$((FAIL+1))
done
[ "$FAIL" -gt 0 ] && echo "[self-distill] $FAIL/$NGPU shards failed"

cat "$OUT/"preds_*.jsonl > "$REPO/data_v0/stemo_ambig_sft_v2/self_distill_predictions.jsonl"
echo "[self-distill] merged $(wc -l < $REPO/data_v0/stemo_ambig_sft_v2/self_distill_predictions.jsonl) predictions"
