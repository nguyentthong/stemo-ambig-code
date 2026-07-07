# ReQueST — paper structure plan v8 FINAL (ACL/ARR August)

ADJUDICATION RECORD (verification round, two judges):
- Gemini (2/5) won the central argument: without a second dataset, a section-level
  second investigation imitates RAcQUEt's rhythm without its justification. RAcQUEt
  itself places the selection analysis at §4.4 and earned its second sections via a
  NEW dataset (BIAS) with a new harm dimension. §5 is therefore COLLAPSED into §4
  (the author's own earlier instinct, confirmed).
- ChatGPT (3.8/5) contributed three adopted precision fixes: the SMR definition must
  require ANSWER DIVERGENCE between the committed and intended readings (same-answer
  cases are "misgrounding", reported separately); the committed-answer -> gold-reading
  mapping is a hidden classifier that needs its own validation; one compact
  validation table is the cheapest defense against the top rejection risk.
- Both judges: the introduction must preview the selection finding and RL. APPLIED to
  main.tex (two amendments; compiles; intro ends at line 142 of the 150 budget).
- Rejected: Gemini's demand to cut the frozen intro to 1.25 pages (infeasible);
  ChatGPT's §5 patches (superseded by the collapse).

## Page budget (8.0 content pages; Limitations/Ethics uncounted)

| § | Title | Pages | Status |
|---|-------|-------|--------|
| 1 | Introduction (+ Figure 1) | 2.00 | DONE (previews added for selection + RL) |
| 2 | Related Work | 0.75 | draft from paper_draft.md §2 |
| 3 | ReQueST: A Benchmark for Referential Ambiguity in Video | 1.30 | DRAFTED in main.tex (Fig 2 stats + Tab 1 comparison + Tab 2 validation, human cells pending) |
| 4 | Investigating Hallucinated Commitment with ReQueST | 2.40 | DRAFTED in main.tex (Tab 3 main results; 4.4/4.6 + SMR/human/per-K figure = gray pending stubs; all numbers pre-D2' scorer, re-score TODO banner in source) |
| 5 | On the Difficulty of Mitigating Hallucinated Commitment | 1.25 | v3/v4 done, v5 partial |
| 6 | Conclusion (one paragraph) | 0.30 | new |
| — | Limitations / Ethics (uncounted) | ~1.0 | new |

## Section plan

### 2 Related Work (0.75)
2.1 Ambiguity in language and QA (AmbigQA, Stengel-Eskin; clarification literature).
2.2 Referential ambiguity in vision and video (RAcQUEt, ClearVQA, MUCAR; video-QA
    single-answer assumption; video hallucination; long-video QA).
END with the 3-axis contrast (temporal vs co-present; per-reading gold vs
response-type classification; follow-through + trainability vs first-response only).

### 3 ReQueST: A Benchmark for Referential Ambiguity in Video (1.30)
3.1 Dataset construction — 80 videos, 1,056 questions, exhaustive readings,
    entity/event subdivisions; Fig 2 stats; Table 1 benchmark comparison.
3.2 Interactive protocol and metrics — response taxonomy; fixed-intention scripted
    interlocutor; full/partial credit; SMR DEFINED CORRECTLY here: confident
    commitment to a non-intended reading WHOSE ANSWER DIFFERS from the intended
    reading's answer (the same-answer case is reported separately as misgrounding);
    anti-gaming + reproducibility in two sentences (details App C).
3.3 Judge and validation suite — answer-correctness judge (cross-judge done; human
    kappa PENDING); READING-SELECTION MAPPING (committed answer -> which gold
    reading) named as its own validated component with human agreement + error
    audit (PENDING, same session); IAA on reading lists (PENDING); ONE COMPACT
    VALIDATION TABLE consolidating all of these with confidence intervals
    (30-example adjudicated error audit in App B).

### 4 Investigating Hallucinated Commitment with ReQueST (2.40)  [the investigation]
4.1 Setup — models, conditions (bare / option-to-ask / interactive). Half a column.
4.2 Main results — humans vs models (baseline PENDING); commitment dominance;
    SMR with the corrected definition + misgrounding rate; per-K degradation.
    Table 2 + per_k figure.
4.3 Entity vs event — temporal recall difficulty (prediction-confirmation framing).
4.4 What do models commit to? — selection analysis at RAcQUEt's exact structural
    placement: temporal position (recency/primacy) and entity saliency of the
    committed reading vs a random-selection baseline, significance tested; connect
    to SMR: biased selection means specific asker intents are systematically
    betrayed. First figure here = selected-reading distribution vs random baseline
    (per ChatGPT: never another commitment-rate plot).
4.5 Diagnostics — gold-reading scaffold (finding vs answering); clarification
    follow-through failure (failure_funnel).
4.6 Case study: long-form video — opens by picking up the introduction's warning
    sentence explicitly (this subsection is its owing evidence). Duration and
    evidence-distance stratification computed on CORE data, INCLUDING SMR stratified
    by video duration and by K, so every link of the warning's causal chain (longer
    footage -> more recurrence -> more readings -> higher SMR) is measured within
    core; then 3-5 long videos as qualitative illustration only, no dataset-scale
    claim (details App D).

### 5 On the Difficulty of Mitigating Hallucinated Commitment (1.25)
5.1 Prompting and clarification instructions — ladder, bounded gains.
5.2 Fine-tuning — format mimicry (signature: enumeration rate up, conditional
    correctness flat) + the saturation demonstration (~99.9% "acknowledges" under
    response-type classification vs ~20% strict). Given +0.20 pages vs v7 so the
    methodology survives review (Gemini's "starved mitigation" point).
5.3 RL — GRPO as preliminary evidence, now PREVIEWED in the intro; reward hacking
    and training details in App E.

### 6 Conclusion (0.30, ONE paragraph, titled "Conclusion")
Sweep: ReQueST; hallucinated commitment dominant; bottleneck is finding readings;
selection is systematic and its cost quantified (SMR); remedies teach format, not
substance; hedged outlook for long-form deployment; equipping models to determine
what a question asks before answering it.

### Limitations (uncounted)
Annotation exhaustiveness; judge + selection-mapping reliability; long-video
generalization limited to the case study; scripted single-turn dialogue; yes/no
format; English-only. Optional intro softening if a reviewer presses: "largely
ambiguity in time" -> the entity subset is co-occurring (flagged by ChatGPT;
currently kept since "largely" already hedges).

### Ethics (uncounted)
YouTube-sourced footage with recognizable people; licenses; release policy;
annotator consent and pay.

## Appendices
A. Annotation protocol, guidelines, IAA details, dataset statistics tables.
B. Judge + reading-selection mapping: prompts, validation, grouped-answer rules,
   30-example adjudicated error audit.
C. Interlocutor spec: templates, seeding, worked dialogues, anti-gaming rationale.
D. Case-study details: the 3-5 long videos, ingestion/token budgets per model.
E. Experimental details: prompts, SFT/RL hyperparameters, training curves,
   reward-hacking rollouts.
F. Extended results: per-model tables, extended selection breakdowns, per-K tables.
G. Qualitative examples: untruncated Fig-1 responses; wavering thinking trace;
   case-study failures.
H. Extended related work.

## Experiments still owed (critical path)
1. Human study FINAL v5 (2026-07-05, author's reduction): binary ambiguity
   judgments only. 100 ambiguous + 40 control questions, all judged by all
   4 volunteers (142 tasks each incl. practice, ~1.5-2h). Yields premise
   validation (hit vs false-alarm), per-item human salience covariate,
   entity-vs-event perceived ambiguity, Fleiss kappa. No human benchmark
   row, no free text, no judge in the loop. §4.6 stub updated. App live,
   experiments/human_baseline/, DESIGN.md v5. James's 3-block pilot data
   retained for a possible appendix note only.
2. Judge human-validation kappa — §3.3. Same session. (2026-07-05 adjudication chain:
   R1+R3 -> D1 -> FINAL D2' at Thong's direction, GPT-5+Gemini unanimous ADOPT.
   Enumeration credit is proportional, (readings determinately answered
   correctly)/K, where the judge assigns each gold reading per-reading via
   entailment: universals cover their readings, exceptions assign the binary
   complement, hedges/bare plurals/counts assign nothing, contradictions = wrong.
   Judge is blind to gold answers during assignment and must quote the licensing
   clause. Live smoke tests 2026-07-05 pass all five probe cases. Validation
   session MUST include constructed per-reading items: grouped exceptions,
   hedged sweeps, count-only, ambiguous pointers, partial overlap groups.)
3. Reading-selection mapping validation (human agreement + error audit) — §3.3.
   Same session (NEW, from ChatGPT's hidden-classifier catch).
4. IAA on reading lists — §3.3. Same session.
5. SMR with corrected definition + misgrounding rate — §4.2. Analysis-only.
6. Selection analysis (temporal position + saliency vs random baseline) — §4.4.
   Analysis-only.
7. Duration + evidence-distance stratification within core, INCLUDING SMR by
   duration and by K — §4.6. Analysis-only.
8. Case-study runs on 3-5 long videos (1 proprietary + 1 open model) — §4.6. Small.
9. v5 RL completion — §5.3. Include if stable; appendix fallback.
10. Re-score existing eval runs with the updated v1.0 scorer (2026-07-05): it now
    emits reading_coverage and conditional_correctness (the two diagnostics §3.2
    promises), overall, per-K, and per-subset. Old metrics.json files lack them.
16. CRITICAL (2026-07-06): extract_yesno ran the thinking judge with a
    128-token output cap, starving it into decision="unknown" on ~98% of
    turn-2/3 extractions (fixed: 1024 tokens, sub_judge.py). Every
    follow-through value computed before the fix is likely UNDERSTATED,
    including the Table 3 F/T column from the box runs. Re-score all runs
    with the fixed judge before submission (folds into item 10). Local
    verification: GPT-5 clarified rows now extract cleanly.
15. NEW: add GPT-5 to the roster (author decision 2026-07-06: GPT-4o's zero
    rows read as dated frontier evidence). run_iaa_closed.py patched for
    gpt-5 (max_completion_tokens fallback + 4096 reasoning headroom; vision
    format verified live). Run ON THE TRAINING BOX (videos required):
      python trace-pilot/src/iaa/run_iaa_closed.py --provider gpt4o \
        --model gpt-5 --out outputs_stemo/iaa_gpt5.jsonl --workers 4
    Then re-score, add to Table 3 + figures + Setup roster + binary-judgment
    run. Figure 5 note: feature models with nonzero subset signal (Gemini
    family, InternVL-38B); mention all-zero models in the caption instead.
13. NEW: InternVL3.5-8B/38B rows in Table 3 (evals ran on the training box,
    per git log: internvl ladder + 38B eval rerun). Import the run outputs,
    re-score with the v1.0 scorer, fill the pending rows. Note InternVL uses
    8 frames (its default), disclosed in §4.1.
14. NEW: frame-count sensitivity ablation (16 vs 32 frames, one model, one
    K-stratified subset) for the appendix, defending the sampling choice
    beyond the scaffold argument already in §4.1.
12. NEW: model binary ambiguity-judgment run — pose the human study's exact
    question ("could this question be about more than one moment, event, or
    person in this video?") to every evaluated model over the same 140 items
    (100 ambiguous + 40 controls). Report per-model hit/false-alarm beside
    the human row in §4.6: the like-for-like recognition contrast (flagged
    by Thong: §4.6 alone does not show models are weak at recognizing).
11. Re-run judge robustness (re-run + cross-judge) under the D2' entailment judge
    prompt. Table 2 "Per-reading assignment" row is \pending; old pair-matching
    numbers (.93/.85 re-run, .73/.62 cross) are kept in §3.3 prose as provenance
    only. Also note: run_iaa_open/closed.py contain a legacy inline fuzzy scorer,
    superseded by benchmark/v1.0/score.py (kept working via a compat field).
    Anti-gaming stat for §3.2/§4 if a reviewer raises the "all yes" sweep: 92.0%
    of questions have divergent gold answers, so uniform sweeps fail by design.

Timeline: W-6 patch judge + recruit + case-study video selection; W-5 annotation
session (1-4) + analyses (5-7); W-4 case-study runs (8) + §3-4 drafting; W-3 §5
drafting + v5 decision; W-2 full draft; W-1 polish/anonymize.

## Intro-claims -> section coverage audit

| Introduction claim | Owing section |
|---|---|
| Hallucinated commitment dominates every family, even with ask option | 4.2 |
| Humans would ask or answer for each reading | 4.2 |
| Gold readings -> near-perfect; finding vs answering | 4.5 |
| Event subset harder (prediction-confirmation) | 4.3 |
| Commitments systematic, favoring temporal positions + salient entities (NEW preview) | 4.4 |
| Precise askers answer incorrectly after the reply | 4.5 |
| Resists prompting / interactive clarification / fine-tuning / RL (RL now previewed) | 5 |
| Format mimicry | 5.2 |
| SOFTENED warning: as footage grows longer... may deliver | 4.2 (SMR) + 4.3 (recurrence) + 4.6 (SMR-by-duration/K + case study) |
| Exhaustive human-validated readings, per-reading gold answers | 3.1 + App A |
| Validated automatic judge | 3.3 + App B |
| Fixed-intention interlocutor, credit, reproducibility | 3.2 + App C |
| >half of naturally occurring questions ambiguous (cited) | 2.1 |

## Structural decisions (v8)
1. COLLAPSE adopted: one investigation section (§4), with the selection analysis at
   4.4 — RAcQUEt's exact placement — and SMR inside the main results. Structural
   honesty over section count: a second investigation requires a second dataset.
2. SMR corrected: requires answer divergence; misgrounding reported separately.
3. Reading-selection mapping treated as a validated component, not a hidden step.
4. §5 mitigation gets 1.25 pages so fine-tuning + RL methodology survive review.
5. Intro frozen at 2.00; preview sentences added instead of shrinking it.
6. All prior decisions (descope, one-paragraph Conclusion, softened warning) stand.


## Terminology update (2026-07-05)
The headline metric is renamed from IAA to "the ReQueST score" (GPT-5+Gemini unanimous, option B). Reasons: IAA collides with inter-annotator agreement at ACL (this very outline used both senses), and coining an acronym for a mean was awkward. Scorer JSON key `iaa` is unchanged for compatibility. The scoring paragraph now states pooling explicitly: each question contributes one credit from whichever route the model takes.
