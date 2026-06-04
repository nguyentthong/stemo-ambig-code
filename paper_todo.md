# STEMO-Ambig paper — full to-do list (target: ARR July 2026)

## Paper framing (decided)
- **Benchmark + phenomenon paper**, not a method paper.
- Story: referential ambiguity on video QA is a real, cross-family failure mode that does not yield to targeted SFT or RL under realistic budgets.
- See [[feedback-stemo-ambig-paper-framing]]. Modest method gains are a *feature*, not a bug.

## Time budget (rough, weeks to ARR cutoff)
| Week | Focus |
|---|---|
| W-7 (now) | Finish v4 generalization runs (qwen36, qwen3vl32b); start v5 RL infra |
| W-6 | v5 RL full runs across 3 models; cross-family black-box runs (GPT-4o, Gemini) |
| W-5 | Ablation experiments (per-K, false-positive enum, judge κ, prompt-only) |
| W-4 | InternVL3.5-38B cross-family run; benchmark validation finalized |
| W-3 | Paper draft (intro, method, results) |
| W-2 | Paper draft (analysis, related work, limitations); figures + tables |
| W-1 | Polish, anonymize, supplementary, submission package |

---

## 1. Benchmark (mostly done)
- [x] 1056 STEMO-Ambig questions, 80 videos, K=2…56
- [x] Spec doc: ambiguity_study/, candidate-generation pipeline
- [ ] **Human validation of Gemini judge**: Cohen's κ on 100 randomly-sampled (model_response, judge_verdict) pairs. Need second annotator. [+2 d]
- [ ] **Inter-annotator agreement** on a 100-item subset of the benchmark itself (do the K interpretations match what humans find?). [+3 d]
- [ ] **Benchmark statistics figure/table**: K distribution, category × subcategory, video duration buckets, source dataset breakdown. [+0.5 d]
- [ ] **Sample-question table** for the paper: 5–8 worked examples showing different K, categories, and one failure mode each. [+0.5 d]

## 2. Phenomenon — baseline failures
- [x] Qwen3-VL-32B-Thinking base eval
- [x] Qwen3.5-27B base eval
- [x] Qwen3.6-27B base eval
- [ ] **InternVL3.5-38B base eval** (cross-family). [+1 d, blocked on chain_v4 generalization or new eval-only script]
- [ ] **GPT-4o base eval** on STEMO-Ambig (black-box, via API). [+0.5 d]
- [ ] **Gemini-2.5-Pro base eval** (black-box, via API). [+0.5 d]
- [ ] **Claude-3.7-Sonnet base eval** (optional, third black-box family). [+0.5 d]
- [ ] **Baseline-failures table**: one row per model, columns enum / commit / strict-K / per-K-breakdown. [+0.5 d]

## 3. Phenomenon — characterization
- [ ] **Per-K degradation curves**: strict-K as function of K, across all base models, single plot. **This is the headline figure.** [+0.5 d]
- [ ] **Failure-mode taxonomy**: classify base-model failures into (a) single-commit, (b) partial-enumerate (covers some K), (c) refuses-to-answer, (d) hallucinated interpretations. Counts + examples. [+1 d]
- [ ] **Question-feature analysis**: which features (ambig phrase length, K, category, video duration) predict single-commit rate? Logistic regression or simple correlations. [+1 d]
- [ ] **Length / "overthinking" analysis**: does reasoning length correlate with enum behavior? Cite earlier overthinking_hallu observations. [+0.5 d]

## 4. Method attempts (the "ambig is hard" demonstrations)

### 4a. SFT — v3 strip-CoT (done, needs to be repositioned as negative result)
- [x] v3 across qwen3vl32b, qwen35, qwen36 — high enum (99 %+), but strict-K only ~20 %
- [ ] **Format-mimicry diagnosis**: report (i) conditional strict | enumerated stays flat ~20–25 %, (ii) interp_coverage << enum_rate, (iii) per_interp_addressed >> per_interp_overall. Build a 3-panel figure or compact table. [+0.5 d]
- [ ] **False-positive enumeration on unambig questions**: run v3 on a 200-item unambig probe set (sampled VideoMME / hand-curated). Measure `enum_unambig_rate`. Smoking gun if >> 5 %. [+1 d]
- [ ] **Per-K v3 breakdown**: does v3 only "succeed" at K=2, collapse at K≥4? Re-derive metrics from saved judgments. [+0.5 d]

### 4b. SFT — v4 CoT-preserved STaR (mostly done)
- [x] v4 qwen35 — enum 36 → 38 %, strict 7 → 9 % (modest)
- [ ] v4 qwen36 — running (chain in sampling phase, ~18 h to done)
- [ ] v4 qwen3vl32b — queued after qwen36
- [ ] v4 InternVL3.5-38B — need to adapt chain to non-Qwen tokenizer/chat-template. [+2 d engineering]
- [ ] **v4 cross-model summary table**: ensure all 4 models have all metrics. [+0.5 d]

### 4c. RL — v5 (planned, see v5_rl_plan.md)
- [ ] Phase 1 infra (~1.5 d)
- [ ] Phase 2 smoke (~1 d)
- [ ] Phase 3 qwen35 full RL (~2 d)
- [ ] Phase 4 eval (~0.5 d)
- [ ] Phase 5 qwen36, qwen3vl32b (~2 d)
- [ ] Phase 6 writeup (~1 d)

### 4d. Prompt-only baselines (cheap, important)
- [ ] **Bare base** (no system prompt): the 99 % single-commit number. Need a clean run, no few-shot help. [+0.5 d]
- [ ] **Few-shot prompt only** (no SFT): isolates what prompting alone buys. Compare to v4-base. [+0.5 d]
- [ ] **Chain-of-thought-only prompt** ("think step by step"): vs few-shot enumeration prompt. [+0.5 d]
- [ ] **Self-consistency / N-shot voting** baseline: if model samples N times and votes, does enum stabilize? [+1 d]

## 5. Cross-family generalization

- [ ] **Qwen-family**: v3 + v4 + v5 across 32B, 27B-3.5, 27B-3.6 (in flight)
- [ ] **InternVL3.5-38B**: v4 and v5. Needs chain port. [+3 d]
- [ ] **Black-box**: GPT-4o, Gemini, Claude evals (no SFT possible; just base + prompt variants). [+1 d total]
- [ ] **Cross-family summary table** + **per-K curve overlay** across all models. [+0.5 d]

## 6. Analysis ablations
- [ ] **Judge-human agreement (Cohen's κ)** — see §1. Target κ ≥ 0.7 to defend automated metrics. [+2 d]
- [ ] **Sensitivity to system prompt phrasing**: try 3 alternate phrasings of the enumeration instruction; report base & v4 metrics each. Reviewers will ask. [+1 d]
- [ ] **Sensitivity to K-detection**: does the model know K? Probe by asking "how many valid interpretations does this question have?" without forcing format. [+1 d]
- [ ] **Truncation analysis**: when reasoning exceeds budget, does enum get cut off? Report `truncation_rate` × strict-K. [+0.5 d]
- [ ] **Effect of paraphrase augmentation**: did our 4× paraphrasing actually help v4? Ablate by training one config with only originals (235 items × 9 upsample) vs aug (1047 × 2 upsample). One model is enough. [+1 d]

## 7. Paper writing

### 7a. Outline & structure
- [ ] Lock paper outline (intro / related / benchmark / phenomenon / method / analysis / discussion). [+0.5 d]
- [ ] Write **abstract + intro** drafts. Lead with phenomenon, not method. [+1 d]
- [ ] **Related work**: video QA benchmarks, ambiguity in QA / NLP, hallucination metrics. [+1 d]

### 7b. Sections
- [ ] §2 STEMO-Ambig benchmark construction + statistics + sample questions. [+1 d]
- [ ] §3 Failure mode characterization (per-K curve, taxonomy). [+1 d]
- [ ] §4 Method attempts (SFT v3, SFT v4, RL v5) — frame as "what doesn't work" or "what only partially works". [+1.5 d]
- [ ] §5 Analysis (κ, sensitivity, format-mimicry, false-positive enum). [+1 d]
- [ ] §6 Discussion / limitations / future work. [+0.5 d]

### 7c. Figures & tables
- [ ] Headline figure: **per-K strict-K accuracy curve** across model families. [+0.5 d]
- [ ] Baseline-failures table. [+0.5 d]
- [ ] Method-results table (per-model: base vs v3 vs v4 vs v5). [+0.5 d]
- [ ] Format-mimicry figure (3 panels: enum vs strict ratio, interp_cov vs enum, pi_addressed vs pi_overall). [+0.5 d]
- [ ] Reward-hacking examples (qualitative, sampled rollouts from v5). [+0.5 d]
- [ ] Benchmark sample-question figure (5–8 examples). [+0.5 d]

### 7d. Submission prep
- [ ] Anonymize repo + remove org references. [+0.5 d]
- [ ] Format to ACL/ARR template. [+0.5 d]
- [ ] **Supplementary**: full prompts, hyperparameters, additional metrics, qualitative examples. [+1 d]
- [ ] **Data + code release plan**: HF hub for benchmark, GitHub for chains; redact API keys; add LICENSE. [+1 d]
- [ ] Final proofread + figure polish. [+1 d]

## 8. Optional / stretch (only if W-2 has buffer)
- [ ] Spatial ambiguity (already excluded per [[feedback-stemo-ambig-spatial-vs-temporal]]) — note as out-of-scope, not a TODO.
- [ ] Multi-modal probing: try image-only static frames vs full video — does it change failure modes? [+1 d]
- [ ] DPO baseline (paired STaR best/rejected) as a third method point alongside SFT / GRPO. [+2 d]
- [ ] Activation/attention probing on a single model to see if "ambiguous referent" is internally represented. [+3 d]

## Open decisions to make this week
1. **InternVL3.5-38B v4** — port the chain or use a smaller InternVL3-8B as the cross-family check? Memory says we picked 38B for fairness; reconsider only if porting is > 3 d.
2. **Human validation budget** — who annotates? Self + 1 collaborator suffices for κ; or recruit 2–3 annotators for stronger story?
3. **Black-box model API budget** — GPT-4o + Gemini-Pro eval on 1056 items × 2 = ~$200 each, acceptable?
4. **Whether to include v5 RL** in the main paper or as appendix-only — depends on whether reward-hacking is dramatic or subtle.

## Critical-path summary
The minimum viable paper at ARR cutoff requires:
1. v4 across all 3 Qwen models (~done)
2. Cross-family base evals (GPT-4o + Gemini + InternVL) — **must finish by W-4**
3. v5 RL at least on qwen35 (qwen36/qwen3vl32b optional) — **must finish by W-3**
4. Format-mimicry analysis of v3 + per-K curves — **easy, do in W-5**
5. Judge κ + prompt-sensitivity ablations — **W-5**
6. Paper draft — **W-3 to W-1**

Everything else is "nice to have"; defend the cuts vs reviewers via the limitations section.
