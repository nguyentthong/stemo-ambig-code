# trace-pilot

Qwen3.5-27B thinking-trace extraction over a 21-example VidHalluc slice
(42 calls total, 2 prompt variants each).

## Prereqs

- vLLM running at `localhost:8000` serving model `Qwen3.5-27B`, started with
  `--reasoning-parser qwen3`.
- VidHalluc cloned to `../vidhalluc/` with videos downloaded per its README.

## Run

```
pip install -r requirements.txt
python src/smoke_test.py          # confirm server + reasoning_content
# (clone vidhalluc into ../vidhalluc and download videos)
python src/load_dataset.py        # first run prints schema; fill in FIELD_MAPS, re-run
python src/run_inference.py       # ~1-2 hours; pauses after first 2 examples
```

## Outputs

- `data/pilot_examples.jsonl` — 21 sampled examples.
- `outputs/traces.jsonl` — one JSON record per inference call (appended live).
- `annotation/annotation_sheet.csv` — header-only template to fill by hand.
