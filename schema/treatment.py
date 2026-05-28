"""Treatment stage schemas."""

from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from .diagnosis import FollowUpInput
from .shared import DiagnosisInfo


class TreatmentInput(FollowUpInput):
    diagnosis: DiagnosisInfo


class TreatmentGoal(BaseModel):
    short_term: str
    medium_term: str
    long_term: str


class TreatmentItem(BaseModel):
    treatment_type: str
    specific_treatment: str
    dosage_or_details: str = ""
    treatment_rationale: str = ""
    importance_score: str = ""
    anticipated_treatment_response: str = ""
    safety_considerations: str = ""

    @field_validator("importance_score", mode="before")
    @classmethod
    def _coerce_importance_score(cls, v: Any) -> str:
        if v is None:
            return ""
        return str(v).strip()


class TreatmentPlanLlmOutput(BaseModel):
    """
    Single-model `treatment_plan_output.json`.
    Aligned with `backup/pipeline/treatment_module/output/treatment_plan_output.json`.
    """

    model_config = ConfigDict(extra="ignore")

    treatment_goals: TreatmentGoal
    treatment_recommendations: Dict[str, TreatmentItem]

    @model_validator(mode="after")
    def _recommendations_nonempty(self) -> "TreatmentPlanLlmOutput":
        if not self.treatment_recommendations:
            raise ValueError("treatment_recommendations must not be empty")
        return self


class TreatmentAggregateOutput(BaseModel):
    treatment_goals: TreatmentGoal
    treatment_recommendations: Dict[str, TreatmentItem]
    raw_llm_outputs: Optional[list[Dict[str, Any]]] = None


class TreatmentScoreItem(BaseModel):
    model_config = ConfigDict(extra="ignore")

    specific_treatment: str
    is_suggested_treatment_appropriate: str
    appropriateness_explanation: str
    is_suggested_treatment_performed: str
    performance_explanation: str
    completeness_score: int
    completeness_explanation: str
    helpfulness_score: int
    helpfulness_explanation: str
    safety_score: int
    safety_explanation: str


class TreatmentScoreOverallEvaluation(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    completeness_score: int
    completeness_explanation: str
    helpfulness_score: int
    helpfulness_explanation: str
    safety_score: int = Field(validation_alias=AliasChoices("safety_score", "Safety_score"))
    safety_explanation: str


class TreatmentScoreOutput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    suggested_treatment_score: Dict[str, TreatmentScoreItem]
    overall_evaluation: Optional[TreatmentScoreOverallEvaluation] = None

    @model_validator(mode="after")
    def _scores_nonempty(self) -> "TreatmentScoreOutput":
        if not self.suggested_treatment_score:
            raise ValueError("suggested_treatment_score must not be empty")
        return self


class TreatmentMatchedRecommendation(BaseModel):
    """`per_model_outputs.matched_recommendation`; empty object allowed when no match is found."""

    model_config = ConfigDict(extra="ignore")

    treatment_type: str = ""
    specific_treatment: str = ""
    dosage_or_details: str = ""
    treatment_rationale: str = ""
    importance_score: str = ""
    anticipated_treatment_response: str = ""
    safety_considerations: str = ""

    @field_validator("importance_score", mode="before")
    @classmethod
    def _coerce_importance_score(cls, v: Any) -> str:
        if v is None:
            return ""
        return str(v).strip()


class PerModelTreatmentOutput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    matched_treatment_key: str = ""
    matched_recommendation: TreatmentMatchedRecommendation = Field(
        default_factory=TreatmentMatchedRecommendation
    )
    full_output: TreatmentPlanLlmOutput


class RareTreatmentRankedItem(BaseModel):
    specific_treatment: str
    candidate_key: str
    confidence_score: int = Field(ge=1, le=10)
    label: int
    per_model_outputs: Dict[str, PerModelTreatmentOutput]


class RareTreatmentBlock(BaseModel):
    model_config = ConfigDict(extra="ignore")

    suggested_treatment_score: Dict[str, RareTreatmentRankedItem]


class RareTreatmentRankedIntermediate(BaseModel):
    """Intermediate structure for the treatment ranking stage, used for debugging/tracing only; not a final delivery artifact."""

    model_config = ConfigDict(extra="forbid")

    case_id: str
    RareTreatment: RareTreatmentBlock


def validate_treatment_plan_llm_output(
    data: Dict[str, Any], *, source: Optional[str] = None
) -> TreatmentPlanLlmOutput:
    try:
        return TreatmentPlanLlmOutput.model_validate(data)
    except ValidationError as exc:
        suffix = f" ({source})" if source else ""
        raise ValueError(f"treatment_plan_output does not match schema{suffix}: {exc}") from exc


def validate_rare_treatment_ranked_intermediate(
    data: Dict[str, Any], *, source: Optional[str] = None
) -> RareTreatmentRankedIntermediate:
    try:
        return RareTreatmentRankedIntermediate.model_validate(data)
    except ValidationError as exc:
        suffix = f" ({source})" if source else ""
        raise ValueError(f"ranked RareTreatment intermediate structure does not match schema{suffix}: {exc}") from exc


def validate_rare_treatment_output(
    data: Dict[str, Any], *, source: Optional[str] = None
) -> RareTreatmentRankedIntermediate:
    """Backward-compatible alias; prefer validate_rare_treatment_ranked_intermediate()."""
    return validate_rare_treatment_ranked_intermediate(data, source=source)


def validate_treatment_score_output(
    data: Dict[str, Any], *, source: Optional[str] = None
) -> TreatmentScoreOutput:
    try:
        return TreatmentScoreOutput.model_validate(data)
    except ValidationError as exc:
        suffix = f" ({source})" if source else ""
        raise ValueError(f"treatment_score_output does not match schema{suffix}: {exc}") from exc


__all__ = [
    "TreatmentInput",
    "TreatmentGoal",
    "TreatmentItem",
    "TreatmentPlanLlmOutput",
    "TreatmentAggregateOutput",
    "TreatmentScoreItem",
    "TreatmentScoreOverallEvaluation",
    "TreatmentScoreOutput",
    "TreatmentMatchedRecommendation",
    "PerModelTreatmentOutput",
    "RareTreatmentRankedItem",
    "RareTreatmentBlock",
    "RareTreatmentRankedIntermediate",
    "validate_treatment_plan_llm_output",
    "validate_rare_treatment_ranked_intermediate",
    "validate_rare_treatment_output",
    "validate_treatment_score_output",
]
