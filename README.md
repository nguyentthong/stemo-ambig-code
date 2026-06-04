# STEMO-Ambig: Code + Configs

Full code for the STEMO-Ambig paper: a video QA benchmark for referentially ambiguous questions, plus baselines, SFT, RL, and ablations across Qwen, GPT-4o, Gemini-3-Flash, Gemini-3.5-Flash, and InternVL families.

## What this repo contains

- `trace-pilot/` — training (SFT, RL) + inference + judging code
- `stemo_ambig/` — benchmark construction utilities + Gemini client
- `tools/` — dashboard, bug-tracker, sweep scripts
- `data_v0/stemo_ambig_candidates/all_questions.json` — the 1,056-item benchmark
- `analysis/*.json` — computed metrics, per-K curves, judge robustness, etc.
- `figures/*.png` + `*.pdf` — paper figures
- `paper_draft.md` — current paper draft (~50k chars, 8 sections)

## What lives in the GCS bucket

Large artifacts are not in git; they live in `gs://video_data_bucket-19052026/`:

- `stemo_videos.zip` — raw video files (~9 GB)
- `stemo_normalized_videos.tar.gz` — preprocessed clips
- `stemo_questions.tar.gz` — questions snapshot
- `stemo_ambig_adapters.tar.gz` — LoRA adapters (qwen35/qwen36/qwen3-vl/internvl × v3+v4 — ~8 GB)
- `stemo_ambig_eval_runs.tar.gz` — predictions/judgments for every model+config
- `stemo_ambig_sft_data.tar.gz` — sampled STaR predictions, kept, SFT train/dev jsonl files

## Setup on a new server

```bash
# 1. Clone the code
git clone https://github.com/nguyentthong/stemo-ambig-code.git
cd stemo-ambig-code

# 2. Install deps (anaconda recommended)
pip install -r requirements.txt  # if present
# or recreate the env manually

# 3. Pull large data from GCS
mkdir -p checkpoints data_v0
gsutil cp gs://video_data_bucket-19052026/stemo_videos.zip data_v0/
gsutil cp gs://video_data_bucket-19052026/stemo_ambig_adapters.tar.gz .
gsutil cp gs://video_data_bucket-19052026/stemo_ambig_eval_runs.tar.gz .
gsutil cp gs://video_data_bucket-19052026/stemo_ambig_sft_data.tar.gz .

tar -xzf stemo_ambig_adapters.tar.gz   # → checkpoints/
tar -xzf stemo_ambig_eval_runs.tar.gz  # → eval_runs/
tar -xzf stemo_ambig_sft_data.tar.gz   # → data_v0/

# 4. Set API keys (not in repo)
cat > .env <<EOF2
GEMINI_API_KEY=<your-key>
OPENAI_API_KEY=<your-key>
EOF2
```

## Reproducing experiments

- Cross-Qwen v4 pipeline: `bash trace-pilot/scripts/chain_v4.sh <qwen35|qwen36|qwen3vl32b|qwen36_9b|internvl8b|internvl38b>`
- v5 RL: `bash trace-pilot/scripts/launch_rl.sh <tag>` (requires v4 adapter in `checkpoints/`)
- Ablations: see `tmp/maximal_prompting_ablation.sh`, `tmp/paraphrase_ablation.sh`, `tmp/prompt_sensitivity_ablation.sh`, `tmp/fft_variant.sh`

## Paper status

- Target venue: ARR August 2026 (deadline 3 Aug → EMNLP)
- Fallback: CVPR Nov 2026

See `paper_draft.md` for the current draft.

## Live experiment dashboard

A separate repo tracks live experiment progress: <https://github.com/nguyentthong/stemo-ambig-dashboard> (auto-updated every 30 min).
