# Open-weight evaluation package (8×A100 server)

Runs the 5 open-weight base models through the full IAA protocol
(1,056 questions, 3-route, multi-turn, Gemini judge) via vLLM, and ends
by printing the data blocks for fig_perk, fig_subsets, and Table 1.

## On the server (3 commands + 1 run)

```bash
git clone https://github.com/nguyentthong/stemo-ambig-code && cd stemo-ambig-code

# videos are not in git (1.3 GB) — from your laptop:
#   rsync -avP stemo/videos_h264/ SERVER:stemo-ambig-code/stemo/videos_h264/

# 5-minute smoke test (6 items, 8B model) BEFORE leaving:
GEMINI_API_KEY=<key> bash server_eval/run_all.sh smoke

# the real run (leave it; ~30-60 min per model, 5 models):
GEMINI_API_KEY=<key> nohup bash server_eval/run_all.sh > run_all.log 2>&1 &
tail -f run_all.log

# OR roughly halve wall time by pairing models across the 8 GPUs
# (27B||27B, then 32B||8B, then 38B on all 8; logs in tmp/lane{A,B,C}.log):
GEMINI_API_KEY=<key> nohup bash server_eval/run_parallel.sh > run_parallel.log 2>&1 &
```

### Binary ambiguity-detection pass (Table 4 / Figure 3 x-axis)

Separately measures each open model's *detection* rate: it poses the human
study's exact yes/no question ("could this be about more than one moment,
event, or person?") over the same 140 items (100 ambiguous + 40 controls) the
closed models answered in `experiments/model_binary_judgment.py`. No judge and
no API key needed (answers are parsed locally as "multiple"/"one"); the whole
sweep is ~5 min per model.

```bash
# smoke (8B only): BINARY=1 RUN_ONLY=internvl8b bash server_eval/run_all.sh
BINARY=1 nohup bash server_eval/run_all.sh > run_binary.log 2>&1 &
tail -f run_binary.log
```

Writes `analysis/binary_judgment_<tag>.json` per model (same schema as the
closed-model files) and prints a hit / false-alarm table at the end. Copy back:

```bash
rsync -avP 'SERVER:stemo-ambig-code/analysis/binary_judgment_*.json' analysis/
```

Rerunning `run_all.sh` is safe: finished models are skipped, partial
models resume (errored items are retried), and a model whose vLLM server
will not start falls back to the proven HF sharded runner
(`trace-pilot/src/iaa/run_iaa_open.py`, one shard per GPU).

## Outputs

- `eval_runs/<tag>/iaa_predictions.jsonl` — full multi-turn transcripts + scores
- `analysis/open_weight_iaa_metrics.json` — aggregate, per-K, per-subset
- stdout of the final step — paste-ready lines for `make_fig_perk.py`
  (SCORE dict), `make_fig_subsets.py` (entity/event rows), and Table 1

Copy back to the laptop when done:

```bash
rsync -avP SERVER:stemo-ambig-code/analysis/open_weight_iaa_metrics.json analysis/
rsync -avP SERVER:stemo-ambig-code/eval_runs/ eval_runs/ --include='*/' \
      --include='iaa_predictions.jsonl' --exclude='*'
```

## Notes

- Judge = `gemini-3-flash-preview` at temperature 0, same as the paper's
  validated configuration (`trace-pilot/src/iaa/sub_judge.py`). Roughly
  6-8k judge calls total across the run.
- Frames: 16 uniform per video, except InternVL at 8 (paper setting,
  S4.1). The HF fallback path runs at 16 frames for all models —
  if a model finished via fallback, note it before using its numbers.
- Engine note for the paper: these runs regenerate the open-weight rows
  under vLLM. Use them as complete replacement rows (Table 1 + figures),
  not as patches onto the old HF-generated rows.
- Models (base only, no adapters): InternVL3_5-8B-HF, Qwen3.5-27B,
  Qwen3.6-27B, Qwen3-VL-32B-Thinking, InternVL3_5-38B-HF. Weights
  auto-download from HF hub on first serve (~250 GB total; the 90-min
  server-start timeout accounts for this).
