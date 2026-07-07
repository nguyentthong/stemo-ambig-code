# Prompt: final structure verification for ReQueST (paste into ChatGPT and Gemini)
# Compare their answers; treat agreement on question 1 as the key signal. A previous
# consultation round already led to descoping a second dataset and rebuilding
# section 5 around a selection-bias axis; this round verifies the final result.

---

You are a brutally harsh ACL area chair. Below are the final INTRODUCTION and the
final SECTION STRUCTURE of a submission. Your job is to find remaining reasons to
reject that the authors can still fix, and to verify the structure against the
template of the accepted RAcQUEt paper (arXiv 2412.13835: two investigations, where
the second studies WHAT models select and its harms, not merely whether they fail).

## THE INTRODUCTION AND ABSTRACT (verbatim, citations as [CITE])

ABSTRACT:

To answer a question, one must first determine what it asks. Questions about videos often admit several readings, since they may refer to multiple entities or moments. While humans respond to such referential ambiguity by seeking clarification or addressing each alternative, video-language models commit to a single reading, a behavior we term hallucinated commitment. We examine this failure by introducing ReQueST, a benchmark that pairs each question with exhaustive, human-validated readings and per-reading answers. Unlike in images, a video question's readings are distributed over time. Models rarely seek clarification even when invited to, and when they do ask, they often fail to use the reply. Fine-tuning induces format mimicry: models adopt the style of ambiguity-aware answers but fill them with incorrect readings. Once the validated readings are supplied, however, the same models answer nearly perfectly. The bottleneck is thus finding the readings, not answering them. Our results underscore the urgency of equipping models to determine what a question asks before answering it.

[FIGURE 1: three example question-video pairs with real frames, timestamps, per-attempt gold answers, and verbatim model responses (GPT-4o "Yes.", Gemini naming the wrong bag, Qwen "No")]

INTRODUCTION:

Imagine watching a video of a homemade chain-reaction machine (Figure 1A). A friend glances over your shoulder, catches a ball rolling toward a wall of numbered doors, and asks: "Does the ball go into the door?". The video, however, shows three attempts spaced minutes apart: the first ball stops just short, while the second and third roll in. You may ask which attempt your friend means, recall which attempt was on screen and answer for it, or answer for each attempt in turn.

None of these responses is unusual, since language is pervasively ambiguous and conversation repairs it routinely [CITE]. Every such repair begins by locating the candidate referents the question could pick out, each of which gives the question a distinct reading. Psycholinguistic studies show how listeners do this when the candidates are in view: they distribute their gaze across the candidates in the scene [CITE]. A viewer of a video has no single scene to scan, because ongoing activity is experienced as a sequence of episodes, segmented while watching and later retrieved from memory [CITE]. The readings of the friend's question belong to episodes minutes apart (Figure 1A) and must be recalled and compared in memory. Referential ambiguity in video is therefore largely ambiguity in time, a dimension a single image cannot have.

Current video-language models fail to resolve this temporal ambiguity. Asked the friend's question, GPT-4o returns the complete response "Yes." whereas Qwen3.5 returns "No.", each silently selecting a different attempt from the three available (Figure 1A). The commitment persists even when a model is more verbose: asked the question in Figure 1B, Gemini names one of the four candidate bags, restricts its answer to that bag, and even this answer is incorrect. We call this behavior hallucinated commitment, by analogy to object hallucination, which invents content absent from the video [CITE]. What the model invents is not content but the uniqueness of the referent: it silently accommodates the question's presupposition that exactly one is meant [CITE] instead of challenging it.

This temporal ambiguity is a specific form of referential ambiguity, previously studied in text and static images. In text, more than half of naturally occurring questions admit multiple valid answers [CITE]. In images, vision-language models describe a single salient referent [CITE] rather than request clarification [CITE]. Because such questions admit no single correct answer, existing benchmarks provide no gold labels and instead classify the form of a response as committing, hedging, or asking. For images, this design is well motivated: the alternatives are visible and often mentioned in models' reasoning traces [CITE], so a commitment reflects an unwillingness to report what the model has perceived. In video, where readings must be recalled rather than observed, the same commitment may instead reflect an inability to identify the readings. Separating the two accounts requires an evaluation that accepts clarification requests and can score any reading against a gold answer. The former rules out unwillingness, and the latter tests identification.

To provide such an evaluation, we introduce ReQueST: a benchmark of REferential QUEstions Spanning Time. ReQueST comprises 1,056 referentially ambiguous questions about 80 videos, divided into an entity subset, where co-occurring entities share the mentioned attribute (Figure 1B), and an event subset, where the same event or entity recurs over time (Figure 1A,C). In contrast to its image-based predecessors, ReQueST does include gold answers: every question carries an exhaustive, human-validated list of its readings, each answerable with yes or no, so that a validated automatic judge can score any response strategy. A model may therefore answer for every reading, or it may ask for clarification and receive a scripted, deterministic reply naming the intended reading, which keeps the evaluation reproducible.

We assess proprietary and open-weight video-language models on ReQueST. While a human respondent would ask or answer for each reading, every model family we test overwhelmingly exhibits hallucinated commitment, even when the prompt offers the option to ask. In a separate diagnostic, however, the same models answer nearly perfectly once the gold readings are supplied, so the failure lies in finding the readings rather than answering them. As this diagnosis predicts, the event subset, whose readings must be recalled across time, proves the more difficult of the two. The inability to find readings resists every remedy we test, from prompting to fine-tuning to interactive clarification, marking the problem as an open challenge. Fine-tuning fails most instructively, yielding format mimicry, multi-reading templates filled with wrong readings, and this pattern both localizes the difficulty and charts directions for future research. These results are a warning sign for the reliability of video assistants: as footage grows longer and people or events recur, a model that commits silently may deliver confident answers to readings the user never intended.

## THE FINAL SECTION STRUCTURE (8.0 content pages)

| 1 | Introduction (+ Figure 1) | 2.00 |
| 2 | Related Work | 0.75 |
| 3 | ReQueST: A Benchmark for Referential Ambiguity in Video | 1.45 |
| 4 | Investigating Hallucinated Commitment with ReQueST | 1.50 |
| 5 | Investigating Commitment Biases and Silent Misinformation with ReQueST | 0.95 |
| 6 | On the Difficulty of Mitigating Hallucinated Commitment | 1.05 |
| 7 | Conclusion (one paragraph) | 0.30 |
| - | Limitations / Ethics (uncounted) | ~1.0 |

Subsection contents:
3.1 Dataset construction (80 videos, 1,056 questions, exhaustive human-validated
    reading lists, entity/event subdivisions; statistics figure; benchmark-comparison
    table vs RAcQUEt/ClearVQA/MUCAR/AmbigQA).
3.2 Interactive protocol (response taxonomy; fixed-intention scripted interlocutor
    that answers any clarification request by naming a pre-selected reading; full
    credit for precise requests, partial for vague; Silent Misinformation Rate (SMR)
    defined here: share of ambiguous questions where the model confidently answers a
    valid but non-intended reading).
3.3 Automatic judge and its validation suite (cross-judge agreement done; human
    kappa, inter-annotator agreement, human-baseline collection pending, one session).
4.1 Setup (GPT-4o, Gemini, Qwen family; bare / option-to-ask / interactive conditions).
4.2 Main results (humans vs models; commitment dominance; per-K degradation).
4.3 Entity vs event subsets (temporal recall difficulty, prediction-confirmation).
4.4 Diagnostics (gold-reading scaffold isolating finding-vs-answering; clarification
    follow-through failure).
5.1 What do models commit to? Commitment-selection analysis: temporal position of the
    chosen reading (recency/primacy) and entity saliency vs a random-selection
    baseline with significance tests (the RAcQUEt "what do models describe" analog,
    transplanted to time). Opens by conceding SMR is the measured cost of commitment,
    not a new behavior, then shows selection is systematic, making the cost
    predictable.
5.2 SMR on the core benchmark, per model and subset, connected to 5.1: biased
    selection means specific asker intents are systematically betrayed.
5.3 Case study: 3-5 long-form videos only, plus duration and evidence-distance
    stratification computed on core data; explicitly no dataset-scale claim.
6.1 Prompting ladder (bounded gains).
6.2 Fine-tuning: format mimicry (enumeration rate rises, conditional correctness
    flat) + saturation demonstration (fine-tuned model scores ~99.9% "acknowledges
    ambiguity" under response-type classification vs ~20% strict correctness).
6.3 RL (GRPO) as preliminary evidence; reward hacking in appendix.

Key decisions already made: descoped a planned second hour-long-video dataset
(annotator time protects core validation instead); section order diagnosis -> biases
and harm -> mitigation difficulty -> conclusion; one-paragraph Conclusion; extensive
appendices (annotation protocol, judge prompts and validation, interlocutor spec,
case-study details, extended results, qualitative examples, extended related work).

## QUESTIONS (max 10 numbered points total, be specific and merciless)

1. THE CENTRAL CHECK: Do sections 4 and 5 now carry genuinely distinct content, or
   can a reviewer still argue "same phenomenon, two sections"? Section 4 asks whether
   models commit and why; section 5 asks which reading they select (temporal and
   saliency biases vs a random baseline) and what the selection costs users (SMR).
   Attack this distinction as hard as you can, then give a verdict on whether it
   holds.
2. CLAIMS COVERAGE: Check every substantive claim in the introduction against the
   structure. Name any claim without an owing section, and any section content the
   introduction fails to preview.
3. STRUCTURE: any remaining ordering, balance, or scoping weakness (thin sections,
   redundancies, misplaced content, missing defenses) a brutal reviewer would exploit.
4. RISKS: rank the three most likely remaining rejection causes for this paper as
   structured, each with the cheapest effective mitigation.
5. SCORE: X/5 against the bar of RAcQUEt's structure and introduction taken together
   (5 = equal or better craft and rigor). Calibrated, not kind.

OUTPUT ORDER: answers 1-5, then a one-line overall verdict.