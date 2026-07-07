# Prompt: sentence-level polish of Section 3 (paste into ChatGPT and Gemini)
# Scope: PROSE ONLY. Structure is settled (outline v8, adjudicated) and must not be
# relitigated. Compare the two responses and adopt only fixes they converge on, or
# fixes that are self-evidently right on inspection.

---

You are two people at once: a brutally harsh ACL area chair who has rejected papers
for sloppy benchmark sections, and a professional academic copyeditor. Below is
Section 3 of an ACL submission, the benchmark section. The paper's structure,
terminology, and content decisions are FINAL. Your job is exclusively
sentence-level: find prose that is awkward, unclear, ambiguous, repetitive, or
below the register of a top-tier ACL paper, and propose concrete rewrites.

## Context (do not critique this part)

The paper introduces ReQueST, a video question-answering benchmark for referential
ambiguity. A question like "Does the ball go into the door?" over a video with
three attempts has multiple valid readings. Humans clarify or answer per reading.
Video-language models silently commit to one reading, which the paper terms
"hallucinated commitment". Section 1 (frozen) has introduced: the term "readings",
the two example figures (Figure 1 with three video-question pairs, panels A/B/C),
and the claim that video readings are separated in time unlike image readings.
Section 4 will report results, including a selection-bias analysis and SMR.

Fixed constraints you must respect in every rewrite:
- No semicolons and no em-dashes in prose (author's style rule).
- Terminology is frozen: readings (not interpretations), hallucinated commitment,
  fixed-intention interlocutor, strict-K, IAA, SMR, misgrounding.
- The gray "pending" cells in Table 2 are deliberate (human validation is being
  collected) and will be filled before submission. Do not flag their existence,
  but DO critique how the text talks about them.
- British/American spelling: American.

## SECTION 3 (verbatim, LaTeX lightly stripped, citations as [CITE])

### 3.1 Dataset Construction

ReQueST pairs 80 videos with 1,056 referentially ambiguous yes/no questions. Each
question carries an exhaustive list of its K readings, and each reading names its
referent, restates the question with the reference resolved, and fixes a gold
answer. The question of Figure 1A thus carries three readings, one per attempt.
The first, "does the ball go into the door on the first attempt?", is answered no,
and the later two are answered yes.

The videos show footage in which similar entities co-occur or the same event
recurs, the setting where a short question naturally underdetermines its referent.
A vision-language model drafts many more candidate questions than we keep, each
with a proposed reading list, and a candidate survives only if it meets three
requirements. First, the list must contain the reading a viewer would reach first.
For example, a list for "does the winner celebrate?" over a two-round game that
holds one reading per round winner but omits the overall winner would score that
viewer's parse as ungrounded, so the candidate is discarded. Second, the mentioned
referent must admit a countable set of clearly visible groundings. For instance,
"the person" over a street of incidental passersby supports no exhaustive reading
list, hence no list a model could fairly be judged against. Third, the ambiguity
must lie in the video's entities and events rather than in the frame. One excluded
case is "the game on the left" of a split screen, which picks out a screen region,
a picture-level ambiguity that images already exhibit. Human annotators then
validate every surviving question, checking that the reading list is exhaustive
and that each gold answer holds (guidelines in Appendix A). We release a public
development split of 256 questions and withhold the 800-question test split.

Questions are short, 6.4 tokens on average, so the ambiguity arises from the video
rather than from convoluted phrasing. The number of readings ranges from 2 to 56
with a mean of 6.1 (Figure 2, left), and the readings are consequential rather
than interchangeable: in 92.0% of questions at least two readings carry different
gold answers, so a model that resolves the question to the wrong reading usually
delivers the wrong answer.

**Large K.** We deliberately keep a long tail of readings that image benchmarks
exclude, RAcQUEt for instance admits a question only when its target category has
at most ten referents [CITE]. The exclusion is forced by their evaluation, which
scores the form of a single response: no cooperative speaker enumerates dozens of
alternatives, so high-K questions would be unfair by construction. Two properties
make them fair here. They are combinatorial rather than vague, K reaches 56 when
the two events a question relates recur eight and seven times, so a reading is a
pair of occurrences and validating the list reduces to counting them. And our
protocol never demands enumeration: a clarifying question earns full credit at any
K, and the scripted reply names a single intended reading, so the follow-up
requires one answer, not 56 (Section 3.2). Large-K questions are thus exactly the
ones where asking is the only sensible strategy, the behavior the benchmark exists
to measure, and we stratify all results by K so that no conclusion leans on the
tail.

**Subsets.** Following the two sources of ambiguity in Figure 1, ReQueST divides
into two subsets. [footnote: The remaining 11 questions mix both sources and are
counted only in aggregate results.] In ReQueST-Entity (555 questions, 71 videos,
mean K of 4.6) several co-occurring entities match the mentioned attribute
(Figure 1B), so the readings are largely simultaneous: their gold evidence spans a
median of 15 seconds. In ReQueST-Event (490 questions, 33 videos, mean K of 7.9)
the mentioned event or entity recurs (Figure 1A,C), so the readings are indexed by
occurrence and separated in time: their evidence spans a median of one minute, and
a tenth of the questions spread their readings over more than five minutes of
footage (Figure 2, right). The two subsets thus operationalize the contrast drawn
in the introduction, readings that can be scanned in one scene against readings
that must be recalled across episodes, and Table 1 situates the resulting
resource: ReQueST is the only benchmark whose readings are separated in time, and
the only one to combine exhaustive reading lists, per-reading gold answers, and a
scored clarification path.

### 3.2 Interactive Protocol and Metrics

**Protocol.** A model receives the video and the question under a prompt that
explicitly permits three behaviors: answering directly, answering for every
reading, or asking a clarifying question. A validated classifier (Section 3.3)
assigns the first response to one of five types: enumeration, scope-anchored
clarification (a question that names the ambiguous phrase, "which man do you
mean?"), vague clarification (an appeal for more information that does not locate
the ambiguity), single commitment, or refusal. A clarification is answered by a
scripted interlocutor with a fixed intention: the intended reading of every
question is fixed in advance, and the interlocutor replies by naming that reading
verbatim from its gold description, whatever the form of the model's request. The
model must then answer for the named reading, with a cap of three turns. Letting
the interlocutor instead adopt whichever reading the model proposes would allow a
model to steer the dialogue toward readings it finds easy and would make scores
incomparable across models.

**Scoring.** An enumeration earns credit 1 when all K readings appear with correct
answers, the strict-K criterion. A scope-anchored clarification earns credit 1
when the follow-up answer is correct for the named reading, and a vague
clarification earns credit 0.5 under the same condition, since it repairs the
ambiguity at a higher conversational cost. Everything else, including any
commitment on an ambiguous question, earns 0. The mean of this credit is the
headline metric, interactive ambiguity-aware accuracy (IAA). Two diagnostics
decompose it: reading coverage, the fraction of gold readings a response
addresses, and conditional correctness, the accuracy of yes/no answers on the
readings a response does address. The gap between them separates finding readings
from answering them, the distinction at the center of this paper.

**Silent misinformation.** Commitments are not all equally harmful, so we quantify
the harm directly. Each committed response is mapped to the gold reading it
answers by a validated mapping step (Section 3.3). The silent misinformation rate
(SMR) is the fraction of questions where the model commits to a reading other than
the intended one AND delivers an answer that differs from the intended reading's
gold answer: the user receives a fluent, confident, and wrong answer with no
signal that anything went wrong. When the delivered answer happens to coincide
with the intended reading's gold answer we instead count misgrounding, a right
answer for the wrong reason, and report it separately.

**Reproducibility.** The intended readings, interlocutor replies, and gold answers
are fixed before evaluation, so every model faces the same deterministic dialogue.
Evaluation is single-episode with frozen models, so no policy about the fixed
reply can be learned during evaluation (details and worked dialogues in
Appendix C).

### 3.3 Judge and Validation Suite

Every metric above is computed by an automatic suite with three components: the
five-way response-type classifier, an answer-correctness judge that matches each
reading-answer pair in a response against the gold list regardless of order and
wording, and the reading-selection mapping that assigns a committed response to
the reading it answers. The mapping deserves emphasis because it silently
underwrites SMR and the selection analysis of Section 4: it is a classifier in its
own right, so we validate it as one rather than treating it as a preprocessing
step.

Table 2 reports the validation. The judge is a pinned Gemini-3-Flash configuration
at temperature 0, and we characterize it along three axes. Re-running the same
judge eight days apart reproduces both binary labels exactly (kappa = 1.00) and
agrees on the matched-reading count on 92.5% of responses (Pearson r = 0.85).
Replacing it with GPT-4o, a judge from a different model family, again reproduces
both binary labels exactly, while count agreement drops to 72.5% (r = 0.62), a
discrepancy that concentrates in partial-credit decisions at the margin rather
than in whether enumeration succeeded. The cross-model gaps we report in Section 4
exceed this noise floor by an order of magnitude. Human agreement for all five
components, together with inter-annotator agreement on reading-list exhaustiveness
and the error audit, is being collected in a single annotation session and will
complete the table.

### Captions (critique these too)

Table 1 caption: "ReQueST against ambiguity benchmarks in text [CITE], visual QA
[CITE], and cross-modal resolution [CITE]. Time: readings separated in time rather
than co-present. Exh.: exhaustive human-validated reading lists. Gold: a gold
answer per reading, so any response strategy is scorable. Inter.: a scored
interactive clarification path."

Table 2 caption: "Validation of the evaluation suite on a stratified 200-response
sample. Binary labels report Cohen's kappa, the matched-reading count reports
exact agreement / Pearson r. Re-run: same judge eight days apart. Cross: GPT-4o as
a second-family judge. Human entries (gray) are being collected in a single
annotation session with a 30-example adjudicated error audit (Appendix B). En
dashes mark combinations that do not apply."

Figure 2 caption: "ReQueST statistics. Left: questions per number of readings K,
by subset. Right: cumulative distribution of the temporal spread of a question's
readings, the span between the earliest and latest gold evidence timestamps. The
median event-subset question spreads its readings over a minute of footage, and
one question in ten over more than five minutes."

## Your output, in this exact format

1. A numbered list of findings, worst first. Each finding: the exact quoted
   phrase, what is wrong with it (be specific: ambiguous antecedent, garden-path,
   redundancy, register, unexplained jargon, claim a reviewer would poke), a
   severity tag (BLOCKING / MINOR), and a concrete rewrite that respects the fixed
   constraints above.
2. Then answer three targeted questions:
   a. Does the Large K paragraph convince you, as a reviewer inclined to say
      "questions with 56 readings are unnatural"? If not, what one sentence would
      fix it?
   b. Are the three filter examples (winner celebrate / the person / the game on
      the left) immediately understandable on first read? Which is weakest?
   c. Is any claim in 3.1-3.3 overclaimed relative to what a benchmark section can
      assert (e.g., "exhaustive", "only benchmark", "validated")?
3. End with the three highest-impact rewrites overall, restated in one line each.

Do NOT propose: restructuring, new experiments, new tables, terminology changes,
or cuts to the Large K defense. Sentence-level only.
