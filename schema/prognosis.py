"""Prognosis stage schemas."""

from __future__ import annotations

from typing import List

from pydantic import BaseModel, ConfigDict

from .treatment import TreatmentInput
from .shared import TreatmentHistory


# ── Input ───────────────────────────────────────────────────────────────────

class PrognosisInput(TreatmentInput):
    """Prognosis stage input (TreatmentInput + treatment history)."""
    treatment_information: List[TreatmentHistory]


# ── Building blocks ─────────────────────────────────────────────────────────

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


# ── Output ──────────────────────────────────────────────────────────────────

class PrognosisPredictionOutput(BaseModel):
    """LLM output for prognosis prediction."""

    model_config = ConfigDict(extra="forbid")

    overall_outcome: OverallOutcome
    functional_status: FunctionalStatus
    symptom_burden: SymptomBurden
    clinical_events: List[ClinicalEvent]


__all__ = [
    "PrognosisInput",
    "PrognosisTarget",
    "OverallOutcome",
    "FunctionalStatus",
    "SymptomBurden",
    "ClinicalEvent",
    "PrognosisPredictionOutput",
]
