#!/usr/bin/env bash
# Closed-API prompt-sensitivity ablation: 3 APIs × 3 prompts.
# Existing closed-API baselines (gpt4o_base, gemini3flash_base, gemini35flash_base)
# already use the bare-style prompt. We add few-shot and maximal here (6 new runs).
# All runs are GPU-free; can execute in parallel with the master queue.
set -uo pipefail
REPO=/home/thong/weride_project/weride/overthinking_hallu
cd $REPO

# Prompt files
mkdir -p tmp/prompts
cat > tmp/prompts/fewshot.txt <<'EOF'
You are an expert at answering questions about video content.
Watch the video carefully and answer the question.

If the question contains a referentially ambiguous phrase (e.g. "the man" when there are multiple men in the video), enumerate each valid interpretation and give an answer per interpretation. If the question is unambiguous, give a single direct answer.

Format for ambiguous questions (use exactly this shape):
This question has K valid interpretations.
- "<referent description>" -> Yes
- "<referent description>" -> No
- ...

Two illustrative examples (different videos; shown only to clarify the output format):

Example 1.
Question: "Does the boy fall down?"
(Suppose the video shows two boys: one in red who slips and falls at 0:05, one in blue who runs ahead without falling.)
Response: This question has 2 valid interpretations.
- "the boy in red who slips at 0:05" -> Yes
- "the boy in blue who runs ahead" -> No

Example 2.
Question: "Is the color added third?"
(Suppose the video shows a person painting bands in this order: black, red, blue, green.)
Response: This question has 4 valid interpretations.
- "black" -> No
- "red" -> No
- "blue" -> Yes
- "green" -> No

Now answer the question about the video provided. Think step by step before giving your final answer.
EOF

cat > tmp/prompts/maximal.txt <<'EOF'
You are answering a question about a video that may have several valid interpretations. Identify the number of valid interpretations, denoted K, by listing each distinct referent the question could pick out. Then provide a yes/no answer for each interpretation.

Output format (use exactly this shape):
This question has K valid interpretations.
- "<referent description 1>" -> Yes/No
- "<referent description 2>" -> Yes/No
- ...

If the question has only one valid interpretation, output a single yes/no answer. Think carefully before responding.
EOF

# Define (api_provider, model_id, eval_tag) tuples
declare -a APIS=(
  "gpt4o:gpt-4o-2024-08-06:gpt4o"
  "gemini:gemini-3-flash-preview:gemini3flash"
  "gemini:gemini-3.5-flash:gemini35flash"
)
declare -a PROMPTS=(
  "fewshot:tmp/prompts/fewshot.txt"
  "maximal:tmp/prompts/maximal.txt"
)

# Launch all 6 runs in parallel
PIDS=()
for api_spec in "${APIS[@]}"; do
  provider="${api_spec%%:*}"
  rest="${api_spec#*:}"
  model="${rest%%:*}"
  tag="${rest##*:}"
  for prompt_spec in "${PROMPTS[@]}"; do
    pname="${prompt_spec%%:*}"
    pfile="${prompt_spec##*:}"
    out_tag="${tag}_${pname}"
    out_dir="$REPO/eval_runs/${out_tag}"
    mkdir -p "$out_dir"
    /home/thong/anaconda3/bin/python $REPO/trace-pilot/src/run_inference_baselines.py \
      --provider "$provider" --model "$model" \
      --ref $REPO/trace-pilot/outputs_stemo/stemo_ambig_traces.jsonl \
      --out "$out_dir/stemo_ambig_predictions.jsonl" \
      --system-prompt-file "$pfile" \
      --workers 4 \
      > "$REPO/tmp/closed_${out_tag}.log" 2>&1 &
    PIDS+=($!)
    echo "launched $out_tag (pid $!)"
  done
done

# Wait for all inferences
for p in "${PIDS[@]}"; do wait $p || true; done
echo "[closed-api] all inferences done $(date)"

# Convert + judge each
for api_spec in "${APIS[@]}"; do
  rest="${api_spec#*:}"
  tag="${rest##*:}"
  for prompt_spec in "${PROMPTS[@]}"; do
    pname="${prompt_spec%%:*}"
    out_tag="${tag}_${pname}"
    out_dir="$REPO/eval_runs/${out_tag}"
    # The baselines runner already writes trace-compatible records, so we feed
    # the predictions file directly to the judge.
    python $REPO/trace-pilot/src/judge_stemo_traces.py \
      --traces "$out_dir/stemo_ambig_predictions.jsonl" \
      --out "$out_dir/stemo_ambig_judgments.jsonl" \
      --metrics-out "$out_dir/stemo_ambig_metrics.json" \
      --workers 12 > "$REPO/tmp/judge_${out_tag}.log" 2>&1 &
  done
done
wait
echo "[closed-api] all judging done $(date)"

# Final report
python3 - <<'PY'
import json
from pathlib import Path
R = Path("/home/thong/weride_project/weride/overthinking_hallu/eval_runs")
print()
print("============ Closed-API prompt-sensitivity (STEMO-Ambig) ============")
print(f"{'Model':<22} {'Prompt':<10} {'enum':>6} {'commit':>7} {'strict':>7}")
for tag in ["gpt4o_base", "gemini3flash_base", "gemini35flash_base",
            "gpt4o_fewshot", "gemini3flash_fewshot", "gemini35flash_fewshot",
            "gpt4o_maximal", "gemini3flash_maximal", "gemini35flash_maximal"]:
    f = R / tag / "stemo_ambig_metrics.json"
    if f.exists():
        m = json.loads(f.read_text())["overall"]
        api, prompt = tag.rsplit("_", 1)
        print(f"{api:<22} {prompt:<10} {m['enumeration_rate']:>6.3f} "
              f"{m['single_commit_rate']:>7.3f} {m['strict_ambig_aware_accuracy']:>7.3f}")
PY
