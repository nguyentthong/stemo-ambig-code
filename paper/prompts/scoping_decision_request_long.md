# Prompt: ReQueST scoping decision + structure + introduction adjustment
# (paste the block below into ChatGPT and Gemini; compare their answers;
#  Task 1's verdict drives Tasks 2 and 3, so read all three as one package)

---

You are a brutally honest senior ACL researcher. Six weeks before the ARR deadline,
decide a scoping question for the paper below, then adjust its section structure AND
its introduction so that all three are consistent. Give verdicts, not surveys.

## THE PAPER SO FAR

Everything about the CORE benchmark is done: 1,056 referentially ambiguous yes/no
questions about 80 minutes-long videos, exhaustive human-validated reading lists with
per-reading gold answers, an interactive protocol (answer every reading, or ask for
clarification and receive a scripted deterministic reply naming the pre-selected
intended reading; precise requests earn full credit, vague ones partial), and an
automatic judge. Core findings are done: every model family overwhelmingly commits to
a single reading (hallucinated commitment); supplied with gold readings the same
models answer nearly perfectly, so the bottleneck is finding readings; the event
subset is harder; fine-tuning yields format mimicry (multi-reading templates filled
with wrong readings); RL is partially run. Pending regardless of this decision: human
baseline (~100 items x 4 annotators), human validation of the judge (kappa), and
inter-annotator agreement, all in one annotation session.

## THE CURRENT ABSTRACT AND INTRODUCTION (verbatim, citations as [CITE])

ABSTRACT:

To answer a question, one must first determine what it asks. Questions about videos often admit several readings, since they may refer to multiple entities or moments. While humans respond to such referential ambiguity by seeking clarification or addressing each alternative, video-language models commit to a single reading, a behavior we term hallucinated commitment. We examine this failure by introducing ReQueST, a benchmark that pairs each question with exhaustive, human-validated readings and per-reading answers. Unlike in images, a video question's readings are distributed over time. Models rarely seek clarification even when invited to, and when they do ask, they often fail to use the reply. Fine-tuning induces format mimicry: models adopt the style of ambiguity-aware answers but fill them with incorrect readings. Once the validated readings are supplied, however, the same models answer nearly perfectly. The bottleneck is thus finding the readings, not answering them. Our results underscore the urgency of equipping models to determine what a question asks before answering it.

[FIGURE 1: three example question-video pairs with timestamps, per-attempt gold answers, and verbatim model responses (GPT-4o "Yes.", Gemini naming the wrong bag, Qwen "No")]

INTRODUCTION:

Imagine watching a video of a homemade chain-reaction machine (Figure 1A). A friend glances over your shoulder, catches a ball rolling toward a wall of numbered doors, and asks: "Does the ball go into the door?". The video, however, shows three attempts spaced minutes apart: the first ball stops just short, while the second and third roll in. You may ask which attempt your friend means, recall which attempt was on screen and answer for it, or answer for each attempt in turn.

None of these responses is unusual, since language is pervasively ambiguous and conversation repairs it routinely [CITE]. Every such repair begins by locating the candidate referents the question could pick out, each of which gives the question a distinct reading. Psycholinguistic studies show how listeners do this when the candidates are in view: they distribute their gaze across the candidates in the scene [CITE]. A viewer of a video has no single scene to scan, because ongoing activity is experienced as a sequence of episodes, segmented while watching and later retrieved from memory [CITE]. The readings of the friend's question belong to episodes minutes apart (Figure 1A) and must be recalled and compared in memory. Referential ambiguity in video is therefore largely ambiguity in time, a dimension a single image cannot have.

Current video-language models fail to resolve this temporal ambiguity. Asked the friend's question, GPT-4o returns the complete response "Yes." whereas Qwen3.5 returns "No.", each silently selecting a different attempt from the three available (Figure 1A). The commitment persists even when a model is more verbose: asked the question in Figure 1B, Gemini names one of the four candidate bags, restricts its answer to that bag, and even this answer is incorrect. We call this behavior hallucinated commitment, by analogy to object hallucination, which invents content absent from the video [CITE]. What the model invents is not content but the uniqueness of the referent: it silently accommodates the question's presupposition that exactly one is meant [CITE] instead of challenging it.

This temporal ambiguity is a specific form of referential ambiguity, previously studied in text and static images. In text, more than half of naturally occurring questions admit multiple valid answers [CITE]. In images, vision-language models describe a single salient referent [CITE] rather than request clarification [CITE]. Because such questions admit no single correct answer, existing benchmarks provide no gold labels and instead classify the form of a response as committing, hedging, or asking. For images, this design is well motivated: the alternatives are visible and often mentioned in models' reasoning traces [CITE], so a commitment reflects an unwillingness to report what the model has perceived. In video, where readings must be recalled rather than observed, the same commitment may instead reflect an inability to identify the readings. Separating the two accounts requires an evaluation that accepts clarification requests and can score any reading against a gold answer. The former rules out unwillingness, and the latter tests identification.

To provide such an evaluation, we introduce ReQueST: a benchmark of REferential QUEstions Spanning Time. ReQueST comprises 1,056 referentially ambiguous questions about 80 videos, divided into an entity subset, where co-occurring entities share the mentioned attribute (Figure 1B), and an event subset, where the same event or entity recurs over time (Figure 1A,C). In contrast to its image-based predecessors, ReQueST does include gold answers: every question carries an exhaustive, human-validated list of its readings, each answerable with yes or no, so that a validated automatic judge can score any response strategy. A model may therefore answer for every reading, or it may ask for clarification and receive a scripted, deterministic reply naming the intended reading, which keeps the evaluation reproducible.

We assess proprietary and open-weight video-language models on ReQueST. While a human respondent would ask or answer for each reading, every model family we test overwhelmingly exhibits hallucinated commitment, even when the prompt offers the option to ask. In a separate diagnostic, however, the same models answer nearly perfectly once the gold readings are supplied, so the failure lies in finding the readings rather than answering them. As this diagnosis predicts, the event subset, whose readings must be recalled across time, proves the more difficult of the two. The inability to find readings resists every remedy we test, from prompting to fine-tuning to interactive clarification, marking the problem as an open challenge. Fine-tuning fails most instructively, yielding format mimicry, multi-reading templates filled with wrong readings, and this pattern both localizes the difficulty and charts directions for future research. These results are a warning sign for the reliability of video assistants: deployed over hour-long footage, where people and events recur by default, a model that commits silently delivers confident answers to readings the user never intended.

## THE CURRENT PLANNED STRUCTURE (8.0 content pages)

| 1 | Introduction (+ Figure 1) | 2.00 |
| 2 | Related Work | 0.75 |
| 3 | ReQueST: A Benchmark for Referential Ambiguity in Video | 1.25 |
| 4 | Investigating Hallucinated Commitment with ReQueST | 1.25 |
| 5 | On the Difficulty of Mitigating Hallucinated Commitment | 0.75 |
| 6 | ReQueST-Long | 0.75 |
| 7 | Investigating Silent Misinformation in Long-Form Video with ReQueST-Long | 1.00 |
| 8 | Conclusion (one paragraph) | 0.25 |
| - | Limitations / Ethics (uncounted) | ~1.0 |

Section notes: 3 = dataset + task/protocol (fixed-intention scripted interlocutor,
full/partial credit) + metrics incl. Silent Misinformation Rate (SMR) + validated
automatic judge. 4 = setup, human baseline (pending), main results (commitment
dominates; SMR on core; event subset harder; per-K), diagnosis (gold-readings scaffold;
follow-through failure). 5 = prompting, fine-tuning -> format mimicry + the saturation
demonstration, RL. 6 = NEW long-video dataset (15-25 videos, 30min-2h, two-pass
annotation), comparative statistics. 7 = commitment and SMR core-vs-long, widening
diagnostic gap, evidence-distance analysis, one qualitative exhibit. Decision gate at
W-4 with a fallback: single-dataset paper, sections 6-7 collapse into "4.5 Case study:
long-form video" (3-5 long videos, qualitative + duration stratification within core).

## THE SCOPING DECISION

Whether to build ReQueST-Long (sections 6-7): 15-25 hour-long videos (meetings,
lectures, sports, raw footage), 150-250 questions via the same pipeline plus two-pass
adversarial annotation, 1 proprietary + 1-2 open models, headline metric SMR compared
core-vs-long. It mirrors the accepted RAcQUEt paper (arXiv 2412.13835), whose second
stakes-raising dataset (RAcQUEt-BIAS) is widely seen as what elevated it. Risks:
exhaustive annotation of hour-long videos is very hard (readings can be missed, K may
exceed 20); frontier-model inference on hour-long video is expensive and hits context
limits; the annotator pool is shared with the pending core annotation session; a null
result on the commitment-rate trend weakens the climax; every hour on Long is an hour
not spent hardening the core paper.

## YOUR TASKS

TASK 1 (verdict): Would you (a) commit to Long now, (b) run the gated plan as designed
(gate at W-4: >=10 validated Long videos, >=100 judged questions, judge kappa >= 0.7;
otherwise fall back), or (c) descope to the fallback today? Answer for ACL acceptance
probability specifically, address the shared-annotator bottleneck explicitly, state
the minimum viable version of Long if relevant, and name the single most likely
rejection cause under your chosen path. One-line verdict at the end.

TASK 2 (structure): Output the full revised section structure (numbered titles + page
budgets summing to 8.0) consistent with your Task 1 verdict. Follow RAcQUEt's naming
pattern ("Investigating X with DATASET"; second investigation titled by the
CONSEQUENCE, like their "...and Social Biases"). If you chose (c), show where the
case-study subsection goes and how its content is scoped.

TASK 3 (introduction): Output the minimal edits to the introduction above that make it
consistent with your Task 2 structure. Mark every changed or inserted sentence in
bold; change nothing else. If your verdict includes Long, insert its mention into the
benchmark paragraph (RAcQUEt introduces both its subsets in one sentence) and convert
the final warning sentence into a measured-finding preview with bracketed number
placeholders like [X%]. If you chose (c), state explicitly whether the current
introduction already suffices unchanged, and if not, show the single softening edit.
CAREFUL: entity/event are subdivisions of the core; ReQueST-Long must not read as a
third sibling of entity/event.

HARD STYLE RULES for any new sentence (violations rejected): no semicolons; no
em-dashes or en-dashes; no sentence under 7 words; no sentence beginning with "Yet";
at most one colon per paragraph of new material; formal ACL register; no "moreover",
"furthermore", "crucial", "delve", "novel"; every sentence must follow from its
predecessor.

OUTPUT ORDER: Task 1 verdict, Task 2 structure table, Task 3 marked edits, then a
five-line summary of why the three are consistent.
