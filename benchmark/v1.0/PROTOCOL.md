# STEMO-Ambig — ReQueST Score Protocol v1.0

## Goal

Measure whether a model can both **recognize** referential ambiguity in a video question and **resolve** it correctly when given a disambiguation.

This is the headline metric of the benchmark, the ReQueST score (the scorer's output key remains `iaa` for backward compatibility). It generalizes strict enumeration (`strict-K`) and recognition-only (`AAR-loose`) by requiring the model to commit to a correct answer when disambiguated — not merely list options or ask a clarifying question.

## Protocol

A single evaluation episode for one item:

```
Turn 1
  user: <video> + <question>           (system prompt: see below)
  model → response_1

Sub-judge classifies response_1:
  enumerated         → proportional credit: (readings answered correctly)/K (see §4); STOP
  single_commit      → 0.0; STOP
  refused            → 0.0; STOP
  clarified_scope    → proceed to Turn 2
  clarified_vague    → proceed to Turn 2 (with vague-clarification penalty rule)

Turn 2 (only if Turn 1 was a clarification)
  user: "<disambiguator utterance>"
  model → response_2

Score response_2:
  1.0 iff response_2's yes/no decision == gold[selected_referent].predicted_answer
  0.0 otherwise

Hard turn cap: 3. If Turn 2 is also a clarification, send the disambiguator
one more time as Turn 3. If Turn 3 is also a clarification, score 0.0.
```

## Disambiguator

The disambiguator agent is **non-LLM, fully deterministic**. It is part of the benchmark, not part of the model under test.

### Referent selection

For each item, select referent index:

```python
selected = rng.choice(K)  # rng seeded by hash(item_id)
```

This pins reproducibility while avoiding positional bias toward `interpretations[0]`.

### Disambiguator utterance

```python
if clarification_class == "clarified_scope":
    utterance = f"I am asking about {gold.interpretations[selected].referent_description}."
elif clarification_class == "clarified_vague":
    # Vague clarifications get the same utterance but flagged for the penalty rule.
    utterance = f"I am asking specifically about {gold.interpretations[selected].referent_description}."
```

**Verbatim rule**: the `referent_description` field is quoted exactly. No paraphrasing, no completion, no enrichment.

### Vague-clarification penalty

If Turn 1 was `clarified_vague` (model asked but did not anchor the ambiguous noun phrase), the Turn-2 disambiguator still answers — but the final score for the item is multiplied by 0.5. This captures the conversational cost of an under-specified clarification ("which one?") relative to a scoped clarification ("which boy?").

> Rationale: a vague clarification places the disambiguation burden entirely on the asker. We treat it as half-credit because the model demonstrated recognition without referent recall, then succeeded only because the asker explicitly enumerated.

## Sub-judge for response classification

The sub-judge runs after Turn 1 and assigns one of five labels:

| Label              | Definition                                                                                                  |
|--------------------|-------------------------------------------------------------------------------------------------------------|
| `enumerated`       | Lists ≥K (or all gold-K) referent–answer pairs explicitly.                                                  |
| `clarified_scope`  | Asks a clarifying question that names the ambiguous noun/phrase from the question (e.g. "which boy?").       |
| `clarified_vague`  | Acknowledges ambiguity OR asks for clarification without naming the ambiguous phrase (e.g. "which one?").    |
| `single_commit`    | Provides a single yes/no decision (or single-referent answer) without ambiguity acknowledgment.              |
| `refused`          | Declines to answer / off-topic / produces non-answer text.                                                   |

Sub-judge model: `gemini-3-flash-preview` (default). Reproducibility via temperature=0, response_mime_type=`application/json`.

### Final-answer sub-judge

For scoring Turn 2/3 answers (yes/no decisions), the same Gemini judge extracts the binary decision from the model's response and compares against `gold.interpretations[selected].predicted_answer`. The extraction prompt instructs the judge to default to `"unknown"` if no clear yes/no commitment can be derived; `"unknown"` scores 0.0.

## Scoring summary

For one item with K gold interpretations:

| Turn-1 class       | Turn-2 needed | Item score                                                                  |
|--------------------|---------------|-----------------------------------------------------------------------------|
| `enumerated`       | no            | proportional: (gold readings answered correctly)/K. 1.0 = strict-K, reported as a diagnostic |
| `clarified_scope`  | yes           | 1.0 iff Turn-2 yes/no matches gold[selected], else 0.0                      |
| `clarified_vague`  | yes           | 0.5 × (1.0 iff Turn-2 yes/no matches gold[selected], else 0.0) = 0.5 or 0.0 |
| `single_commit`    | no            | 0.0                                                                         |
| `refused`          | no            | 0.0                                                                         |

Per-model aggregate metrics:

```
strict_K              = mean over items of (Turn-1 enumerated AND all K correct)
aar_loose             = mean over items of (Turn-1 ∈ {enumerated, clarified_scope})
iaa                   = mean over items of item_score (defined above; enumeration credit is proportional)
clarification_rate    = mean over items of (Turn-1 ∈ {clarified_scope, clarified_vague})
recognition_no_recall = mean over items of (Turn-1 ∈ {clarified_vague})
follow_through_rate   = (item resolved correctly given Turn-2 disambiguator) | (Turn-1 was a clarification)
reading_coverage      = mean over items of (readings explicitly addressed / K)
conditional_correctness = (correct answers on explicitly addressed readings) / (all explicitly addressed readings), pooled over items
```

A response "addresses" a gold reading when it determinately assigns that reading an
answer, either with an explicit referent-answer pair or with a grouped statement that
covers it ("every attempt after the first succeeds" assigns yes to each covered attempt
reading). The judge decides the assignment per reading, blind to the gold answers, and
cites the quote that licenses each assignment. Hedged clauses, bare plurals without a
universal quantifier, and counts without identification assign nothing; contradictory
commitments score as wrong (see judge_prompts/classify_turn1.txt for the full rules).
A clarification's follow-up addresses the reading the disambiguator named. A bare
commitment addresses no reading: it gives the user no signal for locating which reading
was answered, so it earns 0.0 even when its answer happens to fit the intended reading
(that harm is measured separately as SMR). Enumeration credit is proportional:
(readings determinately answered correctly)/K, the fraction of possible intended users
who receive a correct answer for their reading. Per-reading assignment accuracy is
validated against human annotators in the scheduled validation session, alongside the
existing judge robustness checks (which must be re-run under this judge prompt).

Per-K breakdowns: report each of the above stratified by K = 2, 3, 4-6, 7+ (matches paper bins).

## System prompt for inference

Provided verbatim to the model under test:

```
You are an expert at answering questions about video content.

You may respond in any of three ways:

(1) If the question is unambiguous, give a single yes/no answer followed by a brief
    explanation.

(2) If the question has multiple valid interpretations because of an ambiguous
    referent, you may enumerate each interpretation explicitly and provide an
    answer for each. Use the format:
        "<referent description 1>" → Yes/No
        "<referent description 2>" → Yes/No
    You may group interpretations that share the same answer, as long as the grouping identifies exactly which interpretations it covers (for example, "every attempt after the first").

(3) Alternatively, you may ask a clarifying question that identifies the
    ambiguous noun phrase (e.g., "which boy do you mean?"). If you do, the
    asker will respond with a specific referent, and you must then answer
    the question for that referent.

Think step by step before responding.
```

This prompt grants explicit permission to clarify — the benchmark does **not** require enumeration, removing the format-mimicry attack surface.

## Reproducibility pins

- Sub-judge model: `gemini-3-flash-preview` (pin version string in scorer)
- Final-answer extractor: `gemini-3-flash-preview` (same)
- Random selection seed: `hash(item_id)` — deterministic per-item
- Disambiguator text: verbatim from `gold.interpretations[k].referent_description`
- Temperature: 0.0 everywhere
- Turn cap: 3
- Frame count for video models: 16 frames, uniform sample (decord)
- Max tokens per turn: 2048

## Inputs / outputs

### Input (predictions JSONL)

The benchmark scorer accepts a JSONL file where each line is:

```json
{
  "id": "stemo_ambig_NNNN_...",
  "turn_1_response": "model output for turn 1",
  "turn_2_response": "model output for turn 2 (null if no turn 2 fired)",
  "turn_3_response": "model output for turn 3 (null if no turn 3 fired)"
}
```

If a submitter has only single-turn outputs, they may submit just `turn_1_response`; the scorer will classify as enumeration / single-commit / refused / clarification, and items that clarified will be scored 0.0 (treated as "unresolved").

### Output (metrics JSON)

```json
{
  "iaa": 0.143,
  "strict_K": 0.111,
  "aar_loose": 0.179,
  "clarification_rate": 0.087,
  "recognition_no_recall": 0.014,
  "follow_through_rate": 0.621,
  "per_k": {
    "2": {"iaa": 0.34, "strict_K": 0.21, "aar_loose": 0.31},
    "3": {...},
    "4-6": {...},
    "7+": {...}
  },
  "per_subset": {
    "Entity": {...},
    "Event": {...},
    "TempBias": {...}
  },
  "judge_version": "gemini-3-flash-preview@2026-06-05",
  "n_items": 1056,
  "n_classification_failed": 0
}
```

## Versioning

This protocol is **v1.0**. Future changes will be released as v1.1, v2.0, etc. Numbers reported under v1.0 remain reproducible.

## What this protocol does NOT do

- It does not test whether a model can ask a *useful* clarifying question on items that are actually unambiguous (no false-positive ambiguity detection). That is a separate diagnostic that may be added in v2.0.
- It does not evaluate multi-step reasoning beyond two-turn clarification.
- It does not evaluate non-binary answers (extension planned for v1.1).
