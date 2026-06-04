#!/usr/bin/env bash
# End-to-end SFT data pipeline.
# Default video pool: NeXT-QA (827 videos locally at /mnt/ceph3/...).
#
# Usage:
#   bash trace-pilot/scripts/run_sft_pipeline.sh               # default scale (~600 videos for 3K ambig)
#   N_VIDEOS=100 bash trace-pilot/scripts/run_sft_pipeline.sh  # smaller pilot

set -euo pipefail

REPO=/home/thong/weride_project/weride/overthinking_hallu
NEXTQA_SRC=/mnt/ceph3/ec/xiaonan/data/LLaVA-Video-178K/NextQA/NExTVideo

N_VIDEOS=${N_VIDEOS:-600}
WORKERS=${WORKERS:-6}
OUT=$REPO/data_v0/stemo_ambig_sft
VIDEO_STAGE=$OUT/videos

mkdir -p "$VIDEO_STAGE"

echo "[1/4] staging $N_VIDEOS videos -> $VIDEO_STAGE"
i=0
find "$NEXTQA_SRC" -name "*.mp4" | shuf --random-source=<(yes 0 2>/dev/null) | head -n "$N_VIDEOS" | while read f; do
  ln -sf "$f" "$VIDEO_STAGE/$(basename "$(dirname "$f")")_$(basename "$f")"
done

echo "[2/4] generating ambig candidates"
python "$REPO/trace-pilot/src/gen_sft_candidates.py" \
  --video-dir "$VIDEO_STAGE" \
  --out-dir "$OUT/ambig" \
  --max-per-video 6 \
  --workers "$WORKERS"

echo "[3/4] generating unambig items"
python "$REPO/trace-pilot/src/gen_sft_unambig.py" \
  --video-dir "$VIDEO_STAGE" \
  --out "$OUT/unambig.jsonl" \
  --max-per-video 6 \
  --workers "$WORKERS"

echo "[4/4] distilling + formatting"
python "$REPO/trace-pilot/src/make_sft_data.py" \
  --ambig-candidates "$OUT/ambig/all_questions.json" \
  --unambig-jsonl "$OUT/unambig.jsonl" \
  --video-dir "$VIDEO_STAGE" \
  --out-dir "$OUT" \
  --workers "$WORKERS" \
  --dev-frac 0.1

echo
echo "Done."
cat "$OUT/sft_meta.json"
echo
echo "Train ready at:  $OUT/sft_train.jsonl"
echo "Dev ready at:    $OUT/sft_dev.jsonl"
echo "Launch training: bash $REPO/trace-pilot/scripts/launch_sft.sh"
