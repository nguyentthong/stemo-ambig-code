# Prompt: design the human experiment for ReQueST

You are an experimental-design expert for top-tier NLP/vision venues (ACL,
CVPR), with a background in psycholinguistics methodology. Design the human
experiment for the benchmark paper described below, from first principles.

## The benchmark and the paper's thesis

ReQueST is a video question answering benchmark: 1,056 referentially
ambiguous yes/no questions about 80 YouTube videos. Each question is
annotated with an exhaustive, human-validated list of its K readings
(K ranges from 2 to 56, mean 6.1; example: "Does the ball go into the
door?" over a video with three attempts has three readings with gold
answers no, yes, yes). One reading per question is pre-designated as the
intended one by a deterministic hash.

Evaluation protocol for models: a model may answer directly (silent
commitment, credit 0), answer per reading, singly or grouped (proportional
credit: readings answered correctly / K), or ask a clarifying question,
after which a scripted interlocutor names the intended reading and a correct
final answer earns full credit. An LLM judge, validated against re-run,
cross-family, and human checks, assigns each reading the answer a response
entails.

Findings so far: models silently commit on 86-100% of questions, clarify at
most 17%, and use the interlocutor's reply correctly at most 11% of the
time. A scaffold diagnostic shows models answer nearly perfectly when the
gold readings are handed to them. The paper's central thesis: the bottleneck
is FINDING the readings, not answering them. Two novel metrics: SMR (silent
misinformation rate: the model committed to a non-intended reading AND its
answer is wrong for the intended one) and a reading-selection analysis
(which readings models commit to, versus a random baseline).

## What the human experiment is for

The paper needs human evidence that is (a) not predictable in advance,
(b) load-bearing for the thesis or for the novel metrics, (c) obtainable
from natural, reasonable participant behavior, and (d) feasible within the
budget. A study whose outcome any reader can predict is decorative and
will be attacked in review.

## Hard constraints

- Exactly 4 volunteers, roughly 2-3 hours each, working remotely through a
  web form (Gradio) with embedded YouTube videos. Videos take minutes to
  watch and dominate task time.
- No payment platform, no eye-tracking, no lab equipment.
- Stratified samples of the 1,056 questions can be drawn freely; gold
  reading lists and gold answers are available for scoring.
- Free-text responses can be scored afterwards with the validated LLM judge.

## Designs already considered and REJECTED by the author, with reasons

1. Response-type menu (choose: answer directly / answer per reading / ask),
   then fill in the chosen form. Rejected: the menu is a demand
   characteristic; real respondents do not pre-classify their answer.
2. Free-form conversational baseline (reply naturally; a typed question
   triggers the scripted interlocutor; unambiguous control fillers included).
   Rejected: the outcome is predictable. Humans will clarify for large K and
   answer or enumerate for small K. Confirming Gricean cooperation wastes
   volunteer hours; RAcQUEt (Testoni et al., 2025) already showed 91%
   explicit responses on image ambiguity.
3. Forced commitment (answer yes/no, you cannot ask; then indicate which
   reading you answered) to anchor SMR and reading selection. Rejected:
   forcing humans to commit on a question they know is ambiguous is
   unnatural and unreasonable; the elicited behavior does not correspond to
   anything humans genuinely do, so its interpretation is contested.

Do not re-propose these three designs, and do not smuggle their mechanisms
back in under new names. The author's standard: participants must only ever
be asked to do things a reasonable person naturally does when watching a
video and discussing it.

## Your task

1. Propose two to three candidate human-experiment designs that satisfy the
   standard above. For each design specify: the participant-facing task
   (exact instructions in one or two sentences), what is measured, why the
   outcome is NOT predictable in advance, which paper claim or metric it
   anchors, the headline sentence section 4 would gain, the biggest
   methodological risk with a concrete mitigation, and the volunteer-hours
   accounting (be realistic: video watching dominates).
2. State explicitly whether any of the paper's needs (thesis anchor for
   finding-vs-answering, SMR anchor, selection-analysis anchor, human
   performance reference for Table 3) CANNOT be met by any natural human
   task, and what the paper should do instead for those (for example, cite
   prior work, reframe the claim, or drop the anchor).
3. Recommend exactly one design (or a combination fitting the budget) and
   defend it in at most five sentences against the strongest reviewer
   attack you can formulate.
4. If you conclude that NO human experiment passes the bar (every candidate
   is either predictable or unnatural), say so plainly and specify how the
   paper should be revised to stand without one.

Be decisive and concrete. Assume the reader will implement your design
verbatim in a Gradio form.
