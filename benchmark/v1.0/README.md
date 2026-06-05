# STEMO-Ambig v1.0 — Benchmark Release

A video question-answering benchmark for **referential ambiguity** and **interactive disambiguation**.

## What this benchmark measures

Given a video and a question containing an under-specified noun phrase (e.g., "Does the man reach 1 point first?" when the video shows multiple men), a model is scored on whether it can:

1. **Recognize** the ambiguity,
2. **Resolve** it correctly — either by enumerating each valid interpretation with the right per-referent answer, OR by asking a clarifying question and then answering correctly when given a specific referent.

The headline metric is **IAA (Interactive Ambig-Aware Accuracy)**. See `PROTOCOL.md`.

## Contents

- `PROTOCOL.md` — exact specification (turn cap, disambiguator behavior, scoring rules).
- `score.py` — reference scorer. `python -m stemo_ambig.score predictions.jsonl --gold all_questions.json --metrics-out metrics.json`
- `judge_prompts/` — pinned judge instruction strings.
- `splits/` — `dev.json` (256 items, public gold) and `test.json` (800 items, gold withheld).
- `examples/` — reference predictions JSONL for a baseline closed-API model.

## Submitting

A submission consists of one JSONL file with this record schema:

```json
{
  "id": "stemo_ambig_NNNN_...",
  "turn_1_response": "model's response to (video + question)",
  "turn_2_response": "model's response to the disambiguator (if turn 1 was a clarification)",
  "turn_3_response": "model's response to the disambiguator again (if turn 2 was also a clarification)"
}
```

Submitters can choose any inference framework, model family, prompting strategy, frame sampling, or reasoning budget — these will be reported alongside the score.

We will run the canonical scoring (multi-judge ensemble + IAA aggregation) and publish results on the leaderboard.

## Headline metrics

For each submission we report:

| Metric              | Definition                                                              | Headline? |
|---------------------|-------------------------------------------------------------------------|-----------|
| **IAA**             | Interactive Ambig-Aware Accuracy (turn-1 enum OR turn-2 resolved)        | **yes**   |
| strict-K            | Turn-1 enumerates all K referents with correct answers                  | no (diag) |
| AAR-loose           | Turn-1 enumerates OR clarifies with scope-anchored question             | no (diag) |
| clarification rate  | Turn-1 asks a clarifying question                                       | no (diag) |
| recognition/no-recall | Turn-1 is a vague clarification (acknowledges ambiguity, no referent) | no (diag) |
| follow-through rate | Conditional on clarification, did the model resolve correctly?          | no (diag) |
| Per-K breakdown     | Same metrics stratified by K=2, K=3, K=4–6, K=7+                        | reported  |
| Per-subset          | Entity / Event / TempBias subsets                                       | reported  |

## Reproducibility pins (v1.0)

- Judge model: `gemini-3-flash-preview` (cross-judge: `gpt-4o-2024-08-06`)
- Judge temperature: 0.0
- Disambiguator: deterministic (referent index = hash(item_id) % K)
- Frame sampling for the reference baselines: 16 frames, uniform (decord)
- Inference temperature for reference baselines: 0.0
- Max tokens per turn: 2048
- Turn cap: 3

## Status

This is v1.0. Future revisions will preserve this version's results for reproducibility.
