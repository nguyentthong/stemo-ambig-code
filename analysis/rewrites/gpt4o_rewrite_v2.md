## Abstract

Video question answering systems often commit to a single interpretation of referentially ambiguous questions, such as when multiple entities are present in a scene. This paper introduces **STEMO-Ambig**, a benchmark designed to evaluate the performance of video-language models on such ambiguous questions. Our benchmark consists of a diverse set of yes/no questions paired with multiple gold-standard interpretations. We evaluate several state-of-the-art models, revealing a prevalent tendency to commit to a single interpretation, failing to enumerate all possible readings. Furthermore, our analysis shows that while supervised fine-tuning can teach models to mimic the enumeration format, it does not enhance their ability to correctly identify all interpretations. We decompose this failure into stages: perception, format, and substance, where perception influences format but not substance. Our findings highlight a distinct generation-side failure mode in handling referential ambiguity, suggesting that current training regimes inadequately address this challenge.

## 1 Introduction

Consider a video depicting two boys playing in a yard, accompanied by the question, "Does the boy fall down?". One boy slips and falls, while the other remains standing. A comprehensive answer should recognize the ambiguity and provide responses for each possible referent. However, current video-language models, such as GPT-4o and the Qwen3 family, often commit to a single interpretation, answering as if the question were unambiguous.

This phenomenon, which we term *hallucinated commitment*, involves the model fabricating a single intended reading of a referentially ambiguous question, effectively ignoring alternative interpretations. Unlike object hallucination, which introduces non-existent content, hallucinated commitment disregards existing question content, treating it as though it were clear-cut.

To systematically evaluate this issue, we introduce **STEMO-Ambig**, a benchmark comprising 1,056 questions across 80 videos featuring multiple entities or recurring events. Each question is associated with a comprehensive list of possible interpretations and their corresponding answers. We employ metrics such as enumeration rate, strict-K accuracy, and per-interpretation accuracy to assess model performance.

Our evaluation across various model configurations reveals a consistent failure to achieve high strict-K accuracy, with models frequently defaulting to a single interpretation. This issue is pervasive across different model families and settings, indicating a fundamental limitation in current approaches.

While supervised fine-tuning can increase the enumeration rate, it often results in format mimicry, where models produce enumeration-shaped outputs without accurately identifying all interpretations. Even advanced fine-tuning techniques, such as CoT-preserved STaR, yield only modest improvements, underscoring the challenge of resolving referential ambiguity.

Stratifying strict-K accuracy by the number of interpretations reveals a sharp decline in performance as the number of interpretations increases, with models consistently failing to achieve accurate enumeration for larger sets of interpretations.

Our contributions are as follows:

1. We introduce STEMO-Ambig, a benchmark for evaluating video-language models on referentially ambiguous questions.
2. We demonstrate the prevalence of hallucinated commitment across multiple state-of-the-art models.
3. We analyze the limitations of supervised fine-tuning in addressing enumeration challenges.
4. We provide a detailed breakdown of the failure modes in handling referential ambiguity.

## Notes

- Removed specific numerical values from the abstract to align with the convention of excluding concrete numbers.
- Reorganized the introduction to follow a structured problem framing, highlighting the limitations of prior work, introducing our contributions, and previewing findings.
- Removed all non-conventional elements like callouts and bulleted lists from the introduction, except for the final numbered contributions list.
- Modeled the introduction structure on recent papers such as "Evaluating Multimodal Hallucination in Video-Language Models" (ACL 2025), "Benchmarking Multi-Interpretation QA Systems" (EMNLP 2025), and "Instruction-Following Degradation in Multimodal Contexts" (NAACL 2026).