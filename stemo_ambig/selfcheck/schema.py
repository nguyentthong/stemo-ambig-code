"""Pydantic models for emitted candidates.

These also double as the strict-JSON validator for what Gemini returns. Malformed
candidates are rejected without retry per spec; the malformed payload is logged
upstream so prompt iteration can fix the root cause.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


AmbiguityType = Literal[
    "repeated_temporal_referent",
    "ambiguous_temporal_anchor",
    "fuzzy_event_boundary",
]
VLMConfidence = Literal["high", "medium", "low"]


class Interpretation(BaseModel):
    interpretation_id: str
    referent_description: str
    disambiguated_question: str
    vlm_proposed_evidence_spans: list[tuple[float, float]] = Field(default_factory=list)
    supporting_sub_question_ids: list[str] = Field(default_factory=list)
    predicted_answer: str
    vlm_confidence: VLMConfidence


class SubstrateAnchor(BaseModel):
    """Optional STEMO-substrate provenance. Seed-driven candidates skip this."""
    referenced_sub_question_ids: list[str] = Field(default_factory=list)
    referenced_target_question_ids: list[str] = Field(default_factory=list)
    ambiguity_source_ids: list[str] = Field(default_factory=list)
    anchor_rationale: str = ""


class Candidate(BaseModel):
    candidate_id: str
    video_id: str
    generator: str
    generator_prompt_version: str
    ambiguity_type: AmbiguityType
    ambiguity_subtype: str
    question: str
    # Optional: only used by the cat1/cat2/cat3 (substrate-bound) generators.
    # Seed-driven candidates leave this as the default empty SubstrateAnchor.
    substrate_anchor: SubstrateAnchor = Field(default_factory=SubstrateAnchor)
    interpretations: list[Interpretation]
    answer_changes_across_interpretations: bool
    evidence_changes_across_interpretations: bool
    generation_notes: str = ""

    @model_validator(mode="after")
    def _at_least_two_interpretations(self) -> "Candidate":
        if len(self.interpretations) < 2:
            raise ValueError("candidate must have >= 2 interpretations")
        ids = [i.interpretation_id for i in self.interpretations]
        if len(set(ids)) != len(ids):
            raise ValueError("interpretation_ids must be unique within a candidate")
        return self
