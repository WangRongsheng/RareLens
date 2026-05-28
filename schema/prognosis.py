"""Prognosis stage schemas."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, ValidationError

from .treatment import TreatmentInput
from .shared import TreatmentHistory


class PrognosisInput(TreatmentInput):
    treatment_information: List[TreatmentHistory]


class PrognosisTarget(BaseModel):
    confidence_score: int
    explanation: str


class OverallOutcome(PrognosisTarget):
    outcome_category: str


class FunctionalStatus(PrognosisTarget):
    status: str


class SymptomBurden(PrognosisTarget):
    burden: str


class ClinicalEvent(BaseModel):
    event_type: str
    time_period: str
    confidence_score: int
    explanation: str


class PrognosisPredictionOutput(BaseModel):
    """Canonical prognosis final output schema."""

    model_config = ConfigDict(extra="forbid")

    overall_outcome: OverallOutcome
    functional_status: FunctionalStatus
    symptom_burden: SymptomBurden
    clinical_events: List[ClinicalEvent]


class PrognosisAggregateOutput(BaseModel):
    overall_outcome: OverallOutcome
    functional_status: FunctionalStatus
    symptom_burden: SymptomBurden
    clinical_events: List[ClinicalEvent]
    raw_llm_outputs: Optional[List[Dict[str, Any]]] = None


def validate_prognosis_prediction_output(
    data: Dict[str, Any], *, source: Optional[str] = None
) -> PrognosisPredictionOutput:
    try:
        return PrognosisPredictionOutput.model_validate(data)
    except ValidationError as exc:
        suffix = f" ({source})" if source else ""
        raise ValueError(f"prognosis_prediction_output does not match schema{suffix}: {exc}") from exc


__all__ = [
    "PrognosisInput",
    "PrognosisTarget",
    "OverallOutcome",
    "FunctionalStatus",
    "SymptomBurden",
    "ClinicalEvent",
    "PrognosisPredictionOutput",
    "PrognosisAggregateOutput",
    "validate_prognosis_prediction_output",
]
