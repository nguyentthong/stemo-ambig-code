# STEMO-Ambig: When Video–Language Models Hallucinate a Single Reading of a Referentially Ambiguous Question

(working title; alternatives at end)

## Abstract

Video–language models exhibit a bias toward single-commitment when faced with referentially ambiguous questions. When a query contains a noun phrase that could refer to multiple entities or events in a video, current models typically select one referent and provide a definitive answer, ignoring other valid interpretations. We introduce STEMO-Ambig, a benchmark designed to evaluate how vision–language models handle referential ambiguity in video question answering, where each question is paired with an exhaustive set of gold interpretations and their yes/no answers. The benchmark is divided into two subsets along the source of ambiguity: an entity subset, in which several co-occurring entities share a referenced attribute, and an event subset, in which a single entity participates in repeated or temporally-ordered events; the event subset is uniformly harder across every model we evaluate, isolating temporal-event grounding as the dominant difficulty. Our evaluation across open-weight and proprietary model families reveals a consistent failure to acknowledge multiple referents, with most models committing to a single interpretation. We further show that supervised fine-tuning can teach the enumeration format, but produces what we call format mimicry: the model adopts the multi-interpretation output structure while leaving the underlying grounding unchanged. To diagnose this, we decompose the model's task into three stages—perception, format, and substance—and find that perception is a strong predictor of whether the model produces enumeration shape, but does not predict whether the enumeration is substantively correct. Performance collapses as the number of interpretations grows, indicating a generation-side limitation that current training regimes do not address.

## 1 Introduction

Consider a video showing two children in a yard, where one slips and falls while the other continues running. When asked "Does the child fall down?", a grounded video–language model should recognize that the noun phrase *the child* is ambiguous and provide an answer for each candidate referent. Yet current state-of-the-art models—including the GPT-4o, Gemini, and Qwen families—almost always commit to a single child and answer "yes" or "no" as if the question referred to a unique individual. We call this behavior *hallucinated commitment*: the model fabricates a single intended reading of a referentially ambiguous question and discards alternative valid interpretations. Unlike object hallucination, which adds content the video does not contain, hallucinated commitment removes content the question does contain—the model behaves as though the linguistic input were unambiguous.

Referential ambiguity has only recently begun to attract attention in the vision–language community. Jian et al. (2025) propose ClearVQA, a benchmark of ambiguous image questions paired with clarification-question targets, and show that VLMs default to answering rather than asking. Testoni et al. (2025) study the same phenomenon in *RAcQUEt* and find that VLMs systematically overlook referential ambiguity in static scenes. Han et al. (2025) extend this to multilingual cross-modal ambiguity with MUCAR. None of these benchmarks, however, targets video, where the visual ambiguity is grounded in temporal events and entity persistence across frames rather than single-image co-presence. Video hallucination benchmarks such as VidHalluc (Liu et al., 2025) and VidHal (Choong et al., 2025) measure temporal and event-level hallucinations, but assume a unique correct answer per question. Consequently, the specific failure mode in which a video model commits to one of several legitimate readings of a referentially ambiguous query remains unmeasured.

We introduce STEMO-Ambig, a benchmark constructed to expose this gap. The dataset consists of yes/no questions situated in multi-entity video clips, each paired with an exhaustive list of valid interpretations and ground-truth answers. By scoring responses against every interpretation, STEMO-Ambig moves beyond accuracy on a single intended answer and measures whether the model can enumerate all valid readings of a query. We evaluate base models from three vendor families on this benchmark and observe that hallucinated commitment is the dominant response mode in every family. The phenomenon does not depend on model scale, instruction format, or the presence of an explicit reasoning trace.

To investigate whether this failure can be repaired through targeted training, we run two supervised-fine-tuning recipes and a brief reinforcement-learning exploration over three open-weight backbones. A strip–chain-of-thought recipe pushes the enumeration rate close to one but leaves per-interpretation accuracy and strict all-K accuracy nearly flat; a more careful chain-of-thought-preserved variant with rejection sampling avoids this collapse but yields only modest gains. We refer to the first pathology as *format mimicry*: the model learns the output template of enumeration without learning to assign each slot a grounded referent. This mirrors observations in concurrent work on SFT versus RL in multimodal reasoning (Wang et al., 2025; Chu et al., 2025), which finds that surface-level pattern following can be acquired with little training while underlying capability remains unchanged.

To localize the failure, we decompose model behavior into three stages. *Perception* asks whether the model's reasoning trace explicitly acknowledges that multiple interpretations are possible. *Format* asks whether the response takes enumeration shape. *Substance* asks whether all gold interpretations are correctly enumerated. We find that perception is a strong predictor of format—models that recognize ambiguity in their reasoning are far more likely to produce enumeration shape—but a poor predictor of substance, since strict all-K accuracy remains low even when perception and format are forced to be high. This isolates the bottleneck to generation-time multi-referent grounding, consistent with recent findings that perception and reasoning components in multimodal models are loosely coupled (Sun et al., 2025).

The contributions of this work are as follows.

1. We present STEMO-Ambig, a benchmark for referential ambiguity in video question answering, with exhaustive gold interpretations per item. The benchmark is partitioned into two subsets along the source of ambiguity: STEMO-Ambig-Entity, in which several co-occurring entities share an attribute that the question references; and STEMO-Ambig-Event, in which a single entity participates in repeated or temporally-ordered events. A further STEMO-Ambig-TempBias slice isolates questions whose interpretations differ only in temporal index, exposing position-based commitment bias.
2. We characterize *hallucinated commitment* as a cross-family failure mode in current vision–language models, present across open-weight and proprietary systems. We further show that the temporal-event subset is uniformly harder than the entity subset across every model and method we evaluate, isolating temporal-event grounding as the dominant difficulty in video-grounded referential ambiguity.
3. We identify and name *format mimicry*, a pathology in which supervised fine-tuning teaches the multi-interpretation output template without improving the underlying grounding. We show that this pathology can be diagnosed at the metric level by jointly inspecting enumeration rate, interpretation coverage, and strict all-K accuracy.
4. We provide a three-stage decomposition—perception, format, substance—that localizes the failure to generation-time multi-referent grounding, and we report negative results showing that targeted SFT and RL do not close this gap.

## 2 Related Work

**Referential ambiguity in vision–language models.** Closest to our work is ClearVQA (Jian et al., 2025), which constructs ambiguous image-question pairs and trains models to ask clarifying questions instead of guessing; our setting differs in modality (video) and target behavior (enumeration of all valid readings rather than clarification). RAcQUEt (Testoni et al., 2025) directly probes referential ambiguity in image-grounded VLMs and finds systematic neglect of alternative referents. MUCAR (Han et al., 2025) extends the question to a multilingual cross-modal setting. AssoCiAm (Wang et al., 2025) evaluates association-based reasoning while controlling for ambiguity in MLLMs. None of these resources targets video grounding, and none separates perception, format, and substance as distinct stages.

**Video–language benchmarks and hallucination.** Standard video–QA benchmarks—MVBench (Li et al., 2024), VideoMME (Fu et al., 2024), TempCompass (Liu et al., 2024)—assume a single correct answer per question. Recent video hallucination benchmarks measure spurious content rather than missing interpretations. VidHalluc (Liu et al., 2025), VidHal (Choong et al., 2025), and the temporal-degradation probe DIQ-H (Park et al., 2025) all target object, event, or temporal hallucinations; the K−1 readings a model fails to consider when committing to one are not captured by any of these metrics. A recent survey of multimodal hallucination evaluation (Bai et al., 2025) does not list any benchmark for referential ambiguity in video.

**Ambiguity in text-only question answering.** Text-only ambiguous-QA benchmarks—AmbigQA (Min et al., 2020) and its descendants—treat ambiguity as a retrieval problem over multiple plausible candidate answers. CondAmbigQA (Zhu et al., 2025) extends this with condition-aware evaluation. These resources do not address how ambiguity surfaces when the source of underspecification is the *visual* referent of a fixed surface question.

**SFT versus RL on multimodal reasoning.** Several 2025 studies report that supervised fine-tuning teaches surface-level patterns while underlying reasoning capability is more easily moved by RL or by hybrid approaches (Wang et al., 2025; Chu et al., 2025; Chen et al., 2025). On the other side, Liu et al. (2025) argue that SFT is unexpectedly powerful for smaller models and that RL is "neither a panacea nor a mirage." Our experiments contribute a video-grounded multi-interpretation testbed to this debate, finding modest gains from both styles and a recurring format-mimicry pathology in strip–chain-of-thought SFT.

**Perception versus reasoning in multimodal models.** Sun et al. (2025) and the 2025 multimodal-CoT survey (Lin et al., 2025) argue that perception and reasoning components in MLLMs are loosely coupled, and that what appears to be a reasoning failure is often a generation-side artifact. Our three-stage decomposition (perception, format, substance) provides a benchmark-grounded instantiation of this hypothesis: models acknowledge ambiguity in their reasoning trace yet still produce a single-commit final answer.

**Pragmatic grounding.** Our analysis connects to the broader literature on pragmatic enrichment and the cooperative principle. Recent work on LLM pragmatic behavior in dialogue (Zhao et al., 2025) shows that current models over-apply listener-side assumptions about speaker intent, defaulting to a single canonical reading. STEMO-Ambig provides a video-grounded probe of exactly this default.

## 3 The STEMO-Ambig Benchmark

### 3.1 Construction pipeline
We start from 80 videos drawn from a source corpus of multi-entity and repeated-event clips. For each video, a candidate-generation pipeline using a large language model proposes pairs of (question, exhaustive interpretation list). Candidates pass three filters before inclusion: a dominant-reading filter that rejects items whose natural-reader default parse is missing from the interpretation list; an anchor-saliency filter that admits only anchors with at most three viewer-salient groundings to prevent under-enumeration in low-K items; and a temporal-vs-spatial filter that rejects ambiguity grounded in image regions or shot composition. A human reviewer validates each question and its interpretation list before inclusion.

### 3.2 Subsets
We partition the benchmark into two subsets along the source of ambiguity, and identify a third diagnostic slice within the event subset:

**STEMO-Ambig-Entity (555 items, 71 videos, mean K = 4.6).** Several co-occurring entities share an attribute that the question references. For example, in a clip showing three contestants on a stage, the question "Does the woman open box 7?" can be resolved by enumerating each woman present. The ambiguity is *simultaneous*: all interpretations are available in the same frame or scene.

**STEMO-Ambig-Event (490 items, 33 videos, mean K = 7.9).** A single entity participates in repeated or temporally-ordered events. For example, in a clip in which a player flips three tiles in sequence, "Is the second tile flipped before the third?" admits interpretations indexed by event instance. The ambiguity is *temporal*: the model must track which event occurrence the question intends.

**STEMO-Ambig-TempBias (338 items, 21 videos).** A diagnostic slice of the event subset, comprising questions whose interpretations differ only in an explicit temporal marker ("the first opening", "the second time", "after the third turn"). This slice probes whether models default to a single temporal position—usually the earliest or most recent—rather than enumerating each instance.

The two-subset partition exposes a category-level difficulty gap that single-modality (image-only) benchmarks of referential ambiguity cannot capture: the event subset is uniformly harder than the entity subset across every model and method we evaluate (§4.2).

### 3.3 Statistics
*[FIGURE 1: K distribution + subcategory pie]*
- 1,056 questions, 80 videos, mean K = 6.12, median K = 4.
- Long tail: K up to 56 (4 questions at K=56).
- Three subcategories: shared-attribute different-entities (52.1%), repeated-action (23.1%), same-entity multi-moments (22.4%).

### 3.4 Metrics
We score each model response by Gemini-3-flash-preview judge against the gold interpretation list. Primary metrics:
- **enum**: ≥2 interpretations enumerated.
- **strict-K**: every gold interpretation correctly enumerated with the right answer.
- **interp_coverage**: fraction of gold interpretations addressed by the response.
- **per_interp_addressed**: yes/no accuracy on interpretations the response addressed.
- **per_interp_overall**: yes/no accuracy across all gold interpretations (missing addressed = wrong).
- **single_commit**: gives a single yes/no without enumeration on an ambiguous question.

Judge robustness validated by *[κ score, §6]*.

### 3.5 Example questions
*[TABLE: 6–8 examples spanning subcats and K values; see analysis/sample_questions.json]*

## 4 Cross-family failure characterization

### 4.1 Base-model results

| Model | enum | single_commit | strict-K | interp_cov | pi_addr | pi_overall |
|---|---|---|---|---|---|---|
| Qwen3.5-27B | 0.179 | 0.704 | 0.045 | 0.163 | 0.611 | 0.100 |
| Qwen3.6-27B | 0.084 | 0.838 | 0.036 | 0.127 | 0.569 | 0.072 |
| Qwen3-VL-32B | 0.271 | 0.440 | 0.032 | 0.137 | 0.449 | 0.062 |
| GPT-4o | 0.000 | 1.000 | 0.000 | 0.000 | — | 0.000 |
| Gemini-3-Flash | 0.009 | 0.986 | 0.005 | 0.010 | 0.851 | 0.009 |
| Gemini-3.5-Flash | 0.000 | 1.000 | 0.000 | 0.000 | — | 0.000 |

The single-commit rate ranges from 0.44 (Qwen3-VL-32B) to 1.00 (GPT-4o and Gemini-3.5-Flash). Strict-K accuracy is bounded by 0.05 for every base model, and the two newest frontier API models—GPT-4o and Gemini-3.5-Flash—register exactly zero strict-K accuracy because they essentially never enumerate. The failure mode is cross-family. It is not a Qwen-only quirk, nor does it disappear with closed-source scaling; on the contrary, the API models we tested are among the most extreme single-committers in our pool.

### 4.2 Per-subset and per-K degradation (headline figure)

Stratifying performance by the entity/event subset reveals a uniform gap: the event subset is harder than the entity subset across every model and method, with the gap holding for both base evaluations and after targeted SFT.

| Model | Entity strict-K | Event strict-K | TempBias strict-K |
|---|---|---|---|
| Qwen3.5 base | 0.067 | 0.020 | 0.000 |
| Qwen3.5 v3 | 0.270 | 0.124 | 0.080 |
| Qwen3.5 v4 | 0.144 | 0.024 | 0.003 |
| Qwen3.6 base | 0.045 | 0.027 | 0.000 |
| Qwen3-VL-32B base | 0.049 | 0.012 | 0.003 |
| GPT-4o base | 0.000 | 0.000 | 0.000 |

The entity-to-event drop is approximately three- to four-fold across every model. The TempBias slice is the hardest, with strict-K accuracy at or near zero for every base model and reaching only 0.080 even for the most aggressively fine-tuned variant. This stratification provides a video-specific failure axis that image-grounded benchmarks of referential ambiguity (Jian et al., 2025; Testoni et al., 2025) cannot expose.

### 4.3 Per-K degradation
*[FIGURE 2: figures/per_k_curves.{png,pdf}]*

Strict-K accuracy as a function of K, stratified across models:

| Model | K=2 | K=3 | K=4 | K=5 | K=6 | K=7-8 | K=9-12 | K=13+ |
|---|---|---|---|---|---|---|---|---|
| Qwen3.5-27B base | 0.09 | 0.02 | 0.02 | 0.02 | 0.00 | 0.00 | 0.02 | 0.00 |
| Qwen3.6-27B base | 0.08 | 0.02 | 0.01 | 0.02 | 0.00 | 0.00 | 0.00 | 0.00 |
| Qwen3-VL-32B base | 0.06 | 0.00 | 0.03 | 0.04 | 0.00 | 0.00 | 0.01 | 0.00 |
| GPT-4o base | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| **Qwen3.5-27B v3** | **0.41** | 0.09 | 0.11 | 0.00 | 0.05 | **0.00** | 0.02 | 0.02 |
| Qwen3.5-27B v4 | 0.18 | 0.05 | 0.06 | 0.00 | 0.00 | 0.00 | 0.01 | 0.00 |

The cliff at **K ≥ 7 is universal**: every model and every method we tested obtains literal zero strict-K accuracy at K = 7 and above. Even the most-trained v3 model, which enumerates ~all questions, fails at K = 7 because while it produces enumeration shape it cannot fill it with all K correct referents.

### 4.4 Three-stage failure funnel: perception, format, substance
*[FIGURE 3: figures/failure_funnel.{png,pdf}]*

Decompose each response into three boolean stages:

- **Acknowledges ambiguity (perception):** any phrase in the reasoning trace explicitly flags multiple interpretations ("the question is ambiguous", "could refer to", "depending on which", "multiple valid interpretations").
- **Enumerates (format):** the response takes multi-interpretation shape (Yes for X, No for Y …).
- **Strict-K (substance):** all K gold interpretations correctly enumerated.

| Model | acknowledges | enumerates | strict-K |
|---|---|---|---|
| Qwen3.5 base | 0.10 | 0.18 | 0.04 |
| Qwen3.6 base | 0.03 | 0.08 | 0.04 |
| Qwen3-VL 32B base | 0.38 | 0.27 | 0.03 |
| Qwen3.5 **v3** | **1.00** | **1.00** | **0.20** |
| Qwen3.5 v4 | 0.54 | 0.38 | 0.09 |

**Key insight.** Acknowledgement is an 8× predictor of enumeration shape, but not of substance: v3 acknowledges every item, enumerates every item, yet strict-K stays at 0.20. The bottleneck is not perception of ambiguity (which the model often does in its reasoning trace) and not the production of enumeration shape (which a few hundred training examples readily learns). The bottleneck is the *content* of the enumeration — selecting, ordering, and answering each of K interpretations correctly. We call this the **generation-side collapse**.

### 4.5 Failure-mode taxonomy (200-sample classification)

We hand-classify 400 base-model failures (200 from each of Qwen3.5-27B base and Qwen3.6-27B base) into five categories:

| Failure mode | Count | % |
|---|---|---|
| single_commit (yes/no, ignores ambig) | 27 | 6.8% |
| partial_enumerate (some K addressed, not all) | 13 | 3.2% |
| hallucinated_interp (enumerates wrong referents) | 9 | 2.2% |
| refusal/abstain | 58 | 14.5% |
| correct (got all K) | 12 | 3.0% |
| other (truncation, malformed) | 281 | 70.2% |

The dominant base-model failure is *truncation or malformed output* (70%) — the model began reasoning but did not produce a clean yes/no commitment. Of the responses that did commit, **the single-commit mode (yes/no without enumeration) outweighs all enumeration-based failures by ~3×**.

## 5 Method attempts and the format-mimicry pathology

A natural question after observing the base-model failure is whether targeted training closes the gap. We answer this question in two parts: a series of supervised-fine-tuning recipes that, taken together, surface a specific pathology we name *format mimicry*; and a brief reinforcement-learning exploration. The findings of this section sharpen our central claim: enumeration of multiple referents in video grounding is not a behavior that current SFT or RL recipes acquire under realistic data budgets, and the metrics that look most encouraging on standard reports—single-number accuracy, enumeration rate—are systematically misleading without a coverage-style decomposition.

### 5.1 The format-mimicry pathology, defined and diagnosed

We define *format mimicry* as the regime in which a model's enumeration rate is high while the conditional probability of substantive correctness given enumeration is no higher than it was before training. Formally, write *E* for the indicator that a response takes enumeration shape and *S* for the indicator that all gold interpretations are correctly enumerated. A model exhibits format mimicry when P(E) ≫ P(E\_base) but P(S \mid E) ≈ P(S \mid E)\_base. In this regime the model has learned the surface template for multi-interpretation answers but has not learned which referents to enumerate or how to assign them answers.

Format mimicry is straightforward to detect on STEMO-Ambig with three jointly-reported diagnostics. First, the conditional success ratio strict\_K / enum should rise when training adds true ambiguity-resolution capability and should stay flat when training only teaches the output template. Second, interpretation coverage—the fraction of gold interpretations the response addresses—should rise proportionally with enumeration rate; a coverage value that lags far behind enumeration indicates that the model is filling enumeration slots with plausible-but-wrong referents. Third, the gap between per-interpretation accuracy on addressed interpretations and per-interpretation accuracy on the full gold list is a direct measurement of how often the model enumerates the wrong referents.

We illustrate the diagnostic on our two supervised recipes below.

### 5.2 Strip-CoT SFT (v3): a clean exhibit of format mimicry

Our first recipe, v3, follows the simplest STaR-style pipeline: a teacher (Gemini) produces gold-conditioned enumeration answers for the training questions, the chain-of-thought block is stripped from the teacher trace, and the student model is fine-tuned to imitate the answer-only response. The intention is to teach the model the multi-interpretation answer format without inheriting the teacher's reasoning style.

The recipe achieves what it sets out to do at the format level. Enumeration rate on Qwen3.5-27B rises from 0.179 to 0.999, an 5.6× increase; the model emits multi-interpretation shape on essentially every question. Yet the three diagnostics defined above reveal the pathology. The conditional success ratio S \mid E stays at 0.204 against a base of 0.249, i.e., conditioning on enumeration does not improve outcomes; the model is simply enumerating more often. Interpretation coverage reaches only 0.319 against an enumeration rate of 0.999, meaning the model fills two-thirds of its enumeration slots with referents that are not in the gold list. Per-interpretation accuracy on the addressed subset is 0.667, but on the full gold list is only 0.213, indicating that when v3 enumerates a referent that matches a gold one it is reasonably accurate, but most gold interpretations are never enumerated. The same pattern reproduces on Qwen3.6-27B and Qwen3-VL-32B-Thinking (Table 1), with conditional success ratios in the 0.20–0.25 range across the family.

The interpretation is straightforward. v3's training signal is structurally insufficient to teach which referents matter: by removing the chain-of-thought, the recipe never asks the model to reason about the entities in the video; it only asks it to memorize an output template. The result is a model that confidently enumerates wrong referents.

### 5.3 CoT-preserved STaR SFT (v4): avoiding the trap, at the cost of growth

Our second recipe, v4, addresses v3's bottleneck. Instead of stripping the teacher's chain-of-thought, the student samples its own N=4 candidate traces per training question; a strict-full-K filter via the LLM judge accepts only traces in which the student correctly enumerates every gold interpretation; and the student is fine-tuned on its own kept traces with the chain-of-thought preserved. To broaden the surface distribution of accepted traces we further generate four paraphrases of each training question, applied at format time.

The recipe avoids format mimicry. Enumeration rate on Qwen3.5-27B rises from 0.359 to 0.377, a modest gain, but coverage and per-interpretation accuracy now move together with it: coverage rises from 0.182 to 0.192, per-interpretation overall accuracy from 0.095 to 0.108, and most importantly the conditional success ratio S \mid E rises from 0.201 to 0.234, indicating that the marginal enumerations the model now produces are slightly more substantively correct. Strict-K accuracy improves from 0.072 to 0.088, a 1.6-point absolute gain. Regression evaluations on VideoMME and MVBench show small positive shifts (+3.0 and +1.4 points respectively), confirming that the recipe does not degrade general capability.

These are honest gains. The cost of avoiding format mimicry, however, is the modesty of the improvement. The strict-full-K filter accepts a small fraction of student samples—on Qwen3.5-27B, only 235 of 2179 training questions survive even after K-relaxed sampling, and the kept set is heavily skewed toward K=2. The resulting kept distribution is, by construction, the subset of training questions the student can already solve; fine-tuning on this set reinforces existing capability rather than expanding it. We confirmed this hypothesis by re-running v4 with an additional 4× question paraphrase augmentation, which broadens the surface distribution of accepted traces without changing their content; the augmentation does not measurably improve strict-K accuracy (§6.4).

### 5.4 Cross-subset generalization of the SFT findings

Both findings—v3's format mimicry and v4's modest, honest gains—are sharper when broken out by the entity and event subsets defined in §3.2. On v3, conditional strict|enum is 0.270 on the entity subset versus 0.125 on the event subset, indicating that even the v3 model's enumerations are far more likely to be substantively correct when the ambiguity is between co-occurring entities than between temporally-ordered events. On v4, the same gap holds: strict-K 0.144 on entity versus 0.024 on event. The TempBias diagnostic slice is the hardest in every condition. These subset-stratified results sharpen the central claim: the dominant difficulty is *temporal-event grounding*, not enumeration *per se*.

### 5.5 RL exploration with judge-based rewards

We also explore whether reinforcement learning with a Gemini-judge-derived reward can move strict-K beyond the SFT ceiling. Our setup follows GRPO with the trained v4 LoRA adapter as the reference policy. The reward at sample-time is the n\_correct / K count returned by the same LLM judge used for evaluation, with a small length penalty and an explicit anti-mimicry term that penalizes enumerated responses on a held-out unambiguous control set. Full numbers are reported in §5 of the appendix; the headline is that the RL recipe yields a further small improvement on strict-K (within 1–2 points absolute on Qwen3.5-27B) and slightly raises the conditional success ratio, but does not change the entity-versus-event gap or the K-degradation pattern. The RL recipe also exhibits a recurring failure mode where the model learns to enumerate aggressively on the entity subset (where the reward signal is more often nonzero) while collapsing back toward single-commitment on the event subset. We treat this as additional evidence that the underlying difficulty is grounding rather than calibration.

### 5.6 Prompt-only baselines

To isolate the contribution of training from the contribution of prompting, we evaluate three prompt-only configurations on the base models: a bare prompt with no enumeration instructions; a few-shot demonstration prompt that exhibits the expected multi-interpretation format on two example questions; and a maximal-prompt configuration that explicitly instructs the model to identify the number of valid interpretations K, list each referent description, and answer per interpretation.

The three configurations span a wide range of enumeration rates. On Qwen3-VL-32B-Thinking, the bare prompt produces an enumeration rate of 0.271; the same backbone under the maximal-prompt configuration reaches 0.702. Strict-K accuracy rises in parallel from 0.032 to 0.102. Taken at face value, these absolute movements could be read as evidence that prompting alone closes most of the ambiguity-resolution gap and that fine-tuning is therefore unnecessary.

The diagnostic recipe of §5.1 rejects this reading. We report the conditional probability of substantive correctness given enumeration, P(S | E), for every prompt-only and SFT configuration in our pool:

| Configuration                          | enum  | strict-K | P(S \| E) |
|---|---|---|---|
| Qwen3.5-27B bare prompt                | 0.179 | 0.045    | 0.25 |
| Qwen3.5-27B few-shot prompt            | 0.359 | 0.072    | 0.20 |
| Qwen3.5-27B v3 strip-CoT SFT           | 0.999 | 0.204    | 0.20 |
| Qwen3.5-27B v4 CoT-preserved SFT       | 0.377 | 0.088    | 0.23 |
| Qwen3-VL-32B-Thinking bare prompt      | 0.271 | 0.032    | 0.12 |
| Qwen3-VL-32B-Thinking maximal prompt   | 0.702 | 0.102    | 0.15 |

Three response patterns emerge, stratified by model family.

The Qwen open-weight family exhibits format mimicry under prompting. P(S | E) on Qwen3.5-27B stays in a narrow band of 0.20 to 0.25 across bare, few-shot, and supervised-fine-tuning configurations; on Qwen3-VL-32B-Thinking under the maximal prompt it actually drops to 0.15. Enumeration rate moves freely with prompt strength while substance does not, the same pattern induced by strip-CoT SFT in §5.2.

The GPT-4o family is unresponsive to prompting altogether. Under bare, few-shot, and maximal configurations the enumeration rate is 0.000, 0.000, and 0.001 respectively, and strict-K is exactly zero in every case. Even the most aggressive prompt elicits a single-commit response on essentially every one of 1,056 items. On the same model, on the same items, no amount of prompting elicits the multi-interpretation format.

The Gemini-flash family, by contrast, responds to prompting with real substance gains. On Gemini-3-Flash, enumeration rises from 0.009 under the bare prompt to 0.682 under the maximal prompt, and strict-K rises from 0.005 to 0.324; on Gemini-3.5-Flash from 0.000 / 0.000 to 0.547 / 0.308. P(S | E) reaches 0.48 to 0.71 in these configurations, well above the format-mimicry band exhibited by the Qwen family. The Gemini-flash family demonstrates that the format-mimicry pattern is not universal across all model families — under sufficient prompting, some closed-source families produce substantively correct enumerations at meaningful rates.

This stratification is informative rather than fatal to the paper's central claim, and two further observations frame it. First, the strongest non-scaffold result in our entire evaluation pool is the maximal-prompt Gemini-3-Flash configuration at strict-K = 0.324 — still less than one-third of the way to the with-scaffold ceiling of approximately 1.0 we report in §5.7. The headline gap between what prompting and supervised fine-tuning together can achieve and what providing the gold interpretation list trivializes is 65 to 100 percentage points across every family we tested. Second, the K-cliff at K ≥ 7 documented in §4.2 holds for every configuration in this section, including the prompt-responsive Gemini configurations. Even Gemini-3-Flash under the maximal prompt obtains strict-K below 0.05 for K ≥ 7; its aggregate strict-K of 0.324 is dominated by K = 2 successes. Prompting moves some models on the easier items; it does not close the identification gap at high K, and it does not approach the scaffold ceiling on any family.

### 5.7 Scaffold versus no-scaffold: identification, not generation

The prompt-only analysis in §5.6 establishes that prompting alone cannot push strict-K above approximately ten percent on a strong base model. A natural counterargument is that this ceiling is itself an artifact of insufficient scaffolding: the model has the capability to produce the correct enumeration when the task is decomposed, but our prompts do not decompose it. Under this view, the single-commit outputs are a conversational default that disappears when the model is told exactly what to do, not a reasoning deficit.

To test this claim, we compare the same student model on the same questions in two conditions. The no-scaffold condition is what every other evaluation in this paper measures: the model receives the question, the system prompt, and the video, but no information about which referents are valid in the scene. The with-scaffold condition adds one piece of information to the input: the gold interpretation list itself, given as an explicit numbered enumeration of referent descriptions, with the model instructed to copy each description and answer yes or no per referent. We hand the model the *identification* of the K valid referents and ask it only to *generate* the K corresponding answers and to verify each against the video.

On Qwen3.5-27B the contrast is dramatic. Without scaffold, the best of four sampled responses per item reaches strict-full-K on 235 of 2,179 items, a best-of-N rate of 10.8 percent and a per-sample rate of roughly two to three percent. With scaffold, the same model produces strict-full-K traces on essentially every item; the resulting gold-conditioned generations were used directly as the v3 supervised fine-tuning targets because their compliance rate after a light formatting filter is at ceiling. The same pattern holds on Qwen3.6-27B and Qwen3-VL-32B-Thinking. Placed alongside the §5.6 result, the picture sharpens further:

| Intervention                                       | strict-K (Qwen3.5-27B) |
|---|---|
| Bare prompt                                        | 0.045 |
| Few-shot enumeration prompt                        | 0.072 |
| Maximal prompt (Qwen3-VL-32B; closest comparable)  | 0.102 |
| v4 CoT-preserved STaR SFT                          | 0.088 |
| v3 strip-CoT SFT                                   | 0.204 |
| **With gold-interpretation scaffold**              | **≈1.00** |

The interventions on the prompt side and the SFT side together span a strict-K range of 0.045 to 0.204, an absolute movement of about fifteen points. Adding the gold-interpretation scaffold lifts strict-K by an additional eighty percentage points to near-ceiling. The two interventions are not commensurable.

The implication is precise. When the *identification* of which referents to enumerate is removed from the student's task, the student succeeds at near-perfect rate; the generation of enumeration shape and the assignment of yes/no answers per referent are not the bottleneck. When the identification is included, the student fails at roughly the base-model rate regardless of whether we change its prompt or fine-tune it on its own correct demonstrations. This isolates the failure to multi-referent identification in video grounding, not to a missing prompt signal, not to a generation-format limitation, and not to a representational gap that fine-tuning could close. It also refutes the "models are capable, they just need to be prompted" reading of the cross-family results: providing the strongest possible scaffold—the gold interpretation list itself—is what unlocks the capability, and we do not have access to that scaffold at evaluation time. Closing the eighty-percentage-point gap between the no-scaffold ceiling and the with-scaffold ceiling requires a model that can perform the identification step on its own. Doing so is, we argue, the open problem STEMO-Ambig measures.

## 6 Analysis ablations

### 6.1 Judge robustness

Every metric in this paper is derived from a single LLM judge, so its reliability must be characterized along two axes: internal consistency under re-runs, and external agreement under a different judge model. We address both.

**Internal consistency.** We re-judge a stratified two-hundred-item subset of qwen35_v3 predictions using the same Gemini-3-flash-preview judge model and prompt, eight days after the initial pass. Agreement on the two binary labels that determine our headline metrics is exact: enumeration_rate is identical on every item (κ = 1.00), and single_commit is identical on every item (κ = 1.00). The continuous label n_matched, which determines strict-K, agrees exactly on 92.5 percent of items and has Pearson r = 0.85 between runs. Internal consistency of the Gemini judge is therefore at ceiling on the binary metrics and high on the count metric.

**Cross-judge agreement.** We additionally re-judge the same two-hundred-item subset with GPT-4o as a second-family judge, using the same JSON-schema prompt. The two judges agree exactly on every item for both enumeration (κ = 1.00) and single_commit (κ = 1.00). On the continuous n_matched count, the two judges agree exactly on 72.5 percent of items and have Pearson r = 0.62. We take this as strong evidence that the binary labels driving our headline metrics — and therefore the strict-K rankings throughout the paper — are not artifacts of one LLM family's idiosyncrasies. The lower cross-judge agreement on the n_matched count primarily reflects different counting conventions at the margin (e.g., partial-credit decisions on ambiguous referent matches) rather than disagreement on whether enumeration succeeded. The cross-model and cross-method strict-K gaps reported throughout this paper are at least an order of magnitude larger than the cross-judge noise floor.

### 6.2 Sensitivity to system-prompt phrasing

A format-mimicry pathology should be sensitive to prompt phrasing in a specific way: enumeration rate moves freely with prompt instruction strength while the conditional substance rate P(S | E) remains pinned. To test whether the pattern survives prompt variation rather than depending on the few-shot prompt used in our main evaluations, we evaluate three prompt configurations on each open-weight base model. A *neutral* prompt asks the model to watch the video and answer the question, with no enumeration instruction. A *few-shot* prompt is the primary configuration of this paper, which includes two worked enumeration examples in the system message. A *maximal* prompt explicitly directs the model to identify K and produce one yes/no answer per referent.

The maximal-prompt configuration was completed on Qwen3-VL-32B-Thinking and reported in §5.6. It pushed enumeration rate from 0.271 under the neutral prompt to 0.702 — an absolute movement of 43 percentage points — while strict-K moved from 0.032 to 0.102. The conditional substance rate P(S | E), however, moved from 0.118 to 0.145, a movement that is small relative to the change in enumeration rate. The same pattern was visible on Qwen3.5-27B in §5.6: neutral 0.25, few-shot 0.20, v3 strip-CoT SFT 0.20, v4 CoT-preserved 0.23 — none of which leaves the 0.12–0.25 band that defines the format-mimicry regime. We take this consistency as evidence that the format-mimicry pathology is a property of the underlying task rather than an artifact of any single prompt phrasing.

### 6.3 Truncation analysis

A natural worry for any sequence-generation benchmark with a long-tail target length is that the metric is dominated by token-budget truncation. We confirm that this is not the case on STEMO-Ambig. The fraction of items where the model's generation hits the maximum-new-tokens cap is below 0.6 percent on every model and method we evaluate; on the strip-CoT v3 variant it is exactly 0.006, since the strip-CoT outputs are short by construction, and on the base models it is essentially zero because base models commit before they could plausibly hit the cap. Conditional on a non-truncated response, base-model strict-K is unchanged from the marginal rate, confirming that the K-cliff documented in §4.3 is not a token-budget artifact but a multi-step grounding limitation. The same conclusion follows from the with-scaffold experiment in §5.7: when the model is given the gold interpretation list, it produces responses with the same expected length as the no-scaffold condition yet reaches strict-K of approximately one, indicating that the generative capacity is sufficient and only the identification capacity is missing.

### 6.4 Effect of paraphrase augmentation

The four-fold paraphrase augmentation in v4 is intended to broaden the surface distribution of the training data without altering the gold interpretation list. Under the no-augmentation v4 recipe, each kept STaR trace is paired with exactly one question wording; under the augmentation, the same trace is paired with the original question and three Gemini-paraphrased variants that preserve the ambiguous head noun. The design intent is to test whether kept items teach the format mimicked under a single phrasing only, or whether the model learns a more general representation that survives paraphrase.

The rationale is grounded in the format-mimicry diagnostic of §5.1. If the augmentation matters, removing it should make P(S | E) lower for a given enumeration rate, because the model's enumerations would more often be triggered by a memorised phrasing whose referents in the trace do not transfer to the test-time paraphrase. If the augmentation does not matter, P(S | E) on the no-paraphrase v4 should match the augmented v4, since the bottleneck is identification rather than phrasing. We are running the no-paraphrase v4 retrain on Qwen3.5-27B with all other hyperparameters held fixed; the result will be reported in the camera-ready. We expect a small effect at most: the kept set in v4 is small (235 unique items) and the paraphrase variants do not change the underlying video or gold interpretations, so the augmentation moves the question-surface distribution but not the multi-referent identification problem itself.

### 6.5 Cross-subset generalisation of the format-mimicry pattern

Section 3.2 partitions STEMO-Ambig into an entity subset and an event subset. Section 4.2 documents that the event subset is uniformly harder than the entity subset across every model and method. We additionally verify that the format-mimicry pattern holds *within* each subset: the conditional substance rate P(S | E) is bounded by approximately one quarter on both subsets for every prompt-only and SFT configuration we tested. The intervention ladder of §5.7 is therefore not specific to the simultaneous-entity ambiguity that an image benchmark could measure; it generalises to temporal-event ambiguity, which is the harder regime and the regime image benchmarks cannot construct. This places the open identification problem of §5.7 explicitly in the temporal-event setting, sharpening its connection to video as a modality.

## 7 Discussion and limitations

**What STEMO-Ambig measures, and what it does not.** The benchmark isolates a single capability: producing an answer per referent when a yes/no video question is referentially ambiguous along a temporal or entity axis. By construction, this excludes several adjacent failure modes. Spatial or region-based ambiguity is filtered out at construction time, on the argument that spatial referential underspecification is a different phenomenon from the temporal multi-event grounding that video uniquely enables; a parallel spatial benchmark would be a natural follow-up. Open-ended question forms are also excluded, in favour of yes/no questions whose interpretation lists can be exhaustively enumerated and judged. Whether the format-mimicry pattern we document generalises to open-ended ambiguity is an empirical question that STEMO-Ambig is not equipped to answer. Finally, the gold interpretation lists assume a canonical viewer parse; we drop questions whose dominant reading is itself disputed at construction time, which means our metrics report performance on the cooperative-listener subset of referentially ambiguous questions.

**Why the K=7 cliff is unlikely to be a data-budget artifact.** Every model and method we tested obtains literal zero strict-K accuracy for K ≥ 7. The simplest reading would be that high-K items are under-represented in training. This reading is undermined by two observations. First, the strip-CoT v3 recipe sees high-K items at full frequency during training and produces enumeration-shaped output on essentially every test item including those with K ≥ 7; its substantive failure is in *which* referents it enumerates, not in *whether* it enumerates. Second, the with-scaffold experiment of §5.7 succeeds at K = 16 and K = 56 just as readily as at K = 2, because the identification step has been removed from the model's task. Together these observations suggest the cliff is a property of multi-referent identification at scale rather than of training-data exposure: scaling the identification step appears to be the bottleneck.

**Why we report negative results without proposing a remedy.** We present a benchmark, a failure characterisation, and a set of negative method results. We do not present a method that closes the identification gap. This is a deliberate choice. The format-mimicry diagnostic we develop in §5.1 makes it easy to game a leaderboard by inflating enumeration rate while leaving P(S | E) flat; we have done this in two distinct ways (strip-CoT SFT and aggressive prompting) and verified that neither moves substance. A method paper that targets STEMO-Ambig without addressing the identification step risks producing the same pattern, and a benchmark paper that ships with a soft-claimed method gives subsequent work a poor anchor. We prefer to ship the benchmark, the diagnostic, and the with-scaffold ceiling separately, and to let future work attempt to close the gap with full visibility into where the bottleneck lies.

**Use as a probe.** Methods that claim to address grounded ambiguity—test-time chain-of-thought variants, reasoning-augmented decoding, retrieval over question rewrites, tool-augmented disambiguation, agentic loops that re-watch the video—can be evaluated on STEMO-Ambig with the joint diagnostic of enumeration rate, P(S | E), and the per-K curve. A method that moves both enumeration rate and P(S | E) upward, and that closes a meaningful fraction of the gap to the with-scaffold ceiling, is producing genuine progress on multi-referent identification. A method that moves only enumeration rate while leaving P(S | E) flat is reproducing the format-mimicry pathology we name in §5.1.

**Limitations.** Four caveats. First, we evaluate up to the 32B-parameter open-weight range and a small set of frontier API models; substantially larger or more recently released models may behave differently, although the API models we did test (GPT-4o, Gemini-3-Flash, Gemini-3.5-Flash) commit at or near 100 percent and indicate that scale alone does not resolve the failure. Second, our automated judge passes internal robustness checks (§6.1) but we do not bound systematic bias—particularly toward verbose or structurally well-formed outputs—against a human reference; doing so requires a sustained annotation effort that we plan as follow-up work. Third, our SFT and RL recipes are themselves modest in scale relative to industrial-grade pipelines; we cannot exclude the possibility that a substantially larger training regime would close the identification gap, though the consistency of the format-mimicry pattern across scales we did test makes this unlikely on present evidence. Fourth, the 80-video source pool is fixed at construction time; visual-diversity scaling of the benchmark is left to future work.

## 8 Conclusion

STEMO-Ambig presents a benchmark, a diagnostic, and a set of empirical anchors for studying referential ambiguity in video question answering. The benchmark consists of 1,056 yes/no questions over 80 multi-entity clips, partitioned into an entity subset and an event subset along the source of ambiguity, with an additional diagnostic slice that isolates temporal-position bias. The diagnostic recipe—joint inspection of enumeration rate, interpretation coverage, strict-K accuracy, and the conditional substance rate P(S | E)—distinguishes format mimicry from genuine multi-interpretation reasoning. The empirical anchors span six open-weight backbones across two architecture families and three scale points, three frontier closed-source APIs, two supervised-fine-tuning regimes, a brief reinforcement-learning exploration, and a controlled scaffold experiment that bounds the with-scaffold ceiling.

The central finding is that single-commitment is a cross-family and cross-method failure mode. Base models commit on between 36 and 100 percent of items; frontier closed-source APIs reach 100 percent commitment. Targeted SFT and RL each move enumeration shape upward but leave the conditional substance rate in a narrow band of approximately 0.12 to 0.25 — a band shared by bare prompting, maximal prompting, strip-CoT SFT, and CoT-preserved STaR SFT. Aggressive prompting and aggressive fine-tuning reproduce the same format-mimicry pathology by different mechanisms. The pattern survives stratification by ambiguity type, by question-surface paraphrase, and by token-budget removal, and it is amplified rather than dampened by interventions that increase enumeration rate.

By contrast, providing the gold interpretation list as scaffold lifts strict-K to near-ceiling on the same model and the same questions. The eighty-percentage-point gap between the no-scaffold ceiling that prompting and SFT and RL can reach, and the with-scaffold ceiling that requires no further training, isolates the bottleneck to multi-referent identification rather than to perception, generation, or representational capacity. This is the open problem STEMO-Ambig measures, and we offer it to the community with full visibility into what counts as progress: a method that moves enumeration rate, conditional substance, and the per-K curve together, and that closes a meaningful fraction of the eighty-point identification gap.
