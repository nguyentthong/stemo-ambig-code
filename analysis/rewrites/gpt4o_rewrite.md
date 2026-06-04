```markdown
## Abstract

Video question answering systems often fail to handle referential ambiguity, where a question like "Does the boy fall down?" could refer to multiple boys in a video. We introduce **STEMO-Ambig**, a benchmark designed to evaluate this phenomenon with a collection of referentially ambiguous questions paired with exhaustive gold interpretations. Our analysis across various video-language models reveals a tendency to commit to a single interpretation, neglecting the ambiguity inherent in the questions. Despite attempts to improve model performance through supervised fine-tuning, results indicate that models learn to mimic the expected format without genuinely resolving the ambiguity. We decompose the failure into three stages: perception, format, and substance, and demonstrate that while models can be trained to recognize ambiguity, this recognition does not translate into accurately enumerating all possible interpretations. We argue that this issue represents a fundamental generation-side failure in current training paradigms for video grounding.

## 1 Introduction

Consider a scenario where two boys are playing in a yard, and a video question answering system is asked, "Does the boy fall down?" In this context, a robust system should acknowledge the ambiguity of the question, as either boy could be the referent. However, state-of-the-art video-language models, including GPT-4o and the Qwen3-VL family, often commit to a single interpretation, providing a yes or no answer without considering alternative readings.

This phenomenon, which we term *hallucinated commitment*, involves the model fabricating a single intended reading of a referentially ambiguous question and discarding other possible interpretations. Unlike object hallucination, which introduces content not present in the video, hallucinated commitment effectively removes content by treating an ambiguous question as if it were unambiguous.

To systematically investigate this issue, we introduce **STEMO-Ambig**, a benchmark comprising 1,056 referentially ambiguous questions across 80 videos, each paired with multiple gold-standard interpretations. This benchmark allows us to evaluate models on their ability to enumerate all possible interpretations and assess their performance using metrics such as enumeration rate, strict-K accuracy, and per-interpretation accuracy.

Our findings reveal a cross-family failure mode: models consistently exhibit low strict-K accuracy and high single-commit rates, indicating a pervasive inability to handle referential ambiguity. While supervised fine-tuning appears to improve enumeration rates, it often results in format mimicry without substantive gains in correctly identifying all interpretations. Even advanced methods like CoT-preserved STaR rejection sampling yield only modest improvements, underscoring the challenge of closing the gap in ambiguity resolution.

The failure is most pronounced when the number of interpretations (K) is large, with strict-K accuracy dropping to zero for K ≥ 7 across all tested models and methods. This suggests a fundamental limitation in current approaches to video grounding, which do not adequately address the generation-side challenges posed by referential ambiguity.

Our contributions are as follows:
1. We introduce the STEMO-Ambig benchmark to evaluate referential ambiguity in video question answering.
2. We demonstrate the prevalence of hallucinated commitment across various models and configurations.
3. We show that fine-tuning strategies often lead to format mimicry rather than genuine ambiguity resolution.
4. We analyze the decomposition of model failures into perception, format, and substance stages, highlighting the disconnect between ambiguity recognition and accurate interpretation enumeration.
```

## Notes

1. **Abstract:** Removed specific quantitative claims and focused on the motivation, benchmark introduction, findings, and claim. Avoided detailed enumeration of model performances.
2. **Introduction:** Reorganized to follow the canonical structure:
   - Started with a concrete example to frame the problem.
   - Explained why existing models and methods fail to address the issue.
   - Introduced our benchmark and its purpose.
   - Provided a continuous prose preview of findings.
   - Ended with a numbered list of contributions.
3. **Removed anti-patterns:** Eliminated bullet points and callouts in the introduction, except for the final contributions list. Removed all-caps headers and avoided hype words.

### Model Papers
- "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" (NAACL 2019)
- "Attention is All You Need" (NeurIPS 2017)
