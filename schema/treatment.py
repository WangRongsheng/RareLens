"""Treatment stage schemas."""

from __future__ import annotations

from typing import Any, Dict

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from .diagnosis import FollowUpInput
from .shared import DiagnosisInfo


# ── Input ───────────────────────────────────────────────────────────────────

class TreatmentInput(FollowUpInput):
    """Treatment stage input (FollowUpInput + confirmed diagnosis)."""
    diagnosis: DiagnosisInfo


# ── Building blocks ─────────────────────────────────────────────────────────

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


# ── Output ──────────────────────────────────────────────────────────────────

class TreatmentOutput(BaseModel):
    """LLM output for treatment planning (goals + recommendations)."""

    model_config = ConfigDict(extra="ignore")

    treatment_goals: TreatmentGoal
    treatment_recommendations: Dict[str, TreatmentItem]

    @model_validator(mode="after")
    def _recommendations_nonempty(self) -> "TreatmentOutput":
        if not self.treatment_recommendations:
            raise ValueError("treatment_recommendations must not be empty")
        return self


__all__ = [
    "TreatmentInput",
    "TreatmentGoal",
    "TreatmentItem",
    "TreatmentOutput",
]
