"""Diagnosis stage schemas."""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel

from .risk import RiskInput
from .shared import ExaminationItem


# ── Inputs ──────────────────────────────────────────────────────────────────

class PrimaryInput(RiskInput):
    """Primary consultation input (same fields as RiskInput)."""
    pass


class FollowUpInput(RiskInput):
    """Follow-up consultation input (RiskInput + examination results)."""
    laboratory_examinations: List[ExaminationItem]
    radiographic_examinations: List[ExaminationItem]
    other_tests: List[ExaminationItem]


# ── Building blocks ─────────────────────────────────────────────────────────

class DiagnosisReason(BaseModel):
    diagnosis_name: str
    confidence_score: int
    diagnostic_reasoning: str
    orphacode: Optional[int] = None


class TestRecommendation(BaseModel):
    test_name: str
    necessity_score: int
    rationale: str


# ── Outputs ─────────────────────────────────────────────────────────────────

class FirstVisitDiagnosisOutput(BaseModel):
    """LLM output for primary consultation (top-5 diagnoses + recommended tests)."""
    most_likely_diagnosis: Dict[str, DiagnosisReason]
    further_diagnostic_test: Optional[Dict[str, TestRecommendation]] = None


class FollowUpDiagnosisOutput(BaseModel):
    """LLM output for follow-up consultation (top-5 diagnoses only)."""
    most_likely_diagnosis: Dict[str, DiagnosisReason]


__all__ = [
    "PrimaryInput",
    "FollowUpInput",
    "DiagnosisReason",
    "TestRecommendation",
    "FirstVisitDiagnosisOutput",
    "FollowUpDiagnosisOutput",
]
