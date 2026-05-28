"""Diagnosis stage schemas."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from .risk import RiskInput
from .shared import ExaminationItem


class FollowUpInput(RiskInput):
    laboratory_examinations: List[ExaminationItem]
    radiographic_examinations: List[ExaminationItem]
    other_tests: List[ExaminationItem]


class DiagnosisReason(BaseModel):
    diagnosis_name: str
    confidence_score: int
    diagnostic_reasoning: str
    orphacode: Optional[int] = None


class TestRecommendation(BaseModel):
    test_name: str
    necessity_score: int
    rationale: str


class FirstVisitDiagnosisOutput(BaseModel):
    most_likely_diagnosis: Dict[str, DiagnosisReason]
    further_diagnostic_test: Optional[Dict[str, TestRecommendation]]
    raw_llm_outputs: Optional[List[Dict[str, Any]]] = None


class FollowUpDiagnosisOutput(BaseModel):
    most_likely_diagnosis: Dict[str, DiagnosisReason]
    raw_llm_outputs: Optional[List[Dict[str, Any]]] = None


class DiagnosisPrimaryConsultationLLMOutput(BaseModel):
    """Single-model initial consultation LLM output, corresponding to `primary_consultation_output.json`."""

    most_likely_diagnosis: Dict[str, DiagnosisReason]
    further_diagnostic_test: Optional[Dict[str, TestRecommendation]] = None


class DiagnosisFollowUpConsultationLLMOutput(BaseModel):
    """Single-model follow-up consultation LLM output, corresponding to `follow_up_consultation_output.json`."""

    most_likely_diagnosis: Dict[str, DiagnosisReason]


class DiagnosisLlmModelOutput(BaseModel):
    """Single-model initial consultation parsed result, fields aligned with `primary_consultation_output*.json`."""

    most_likely_diagnosis: Dict[str, DiagnosisReason]
    further_diagnostic_test: Optional[Dict[str, TestRecommendation]] = None


class DiagnosisTestScoreItem(BaseModel):
    """Legacy nesting: the old pipeline stored test recommendations under `suggested_test_score`."""

    test_name: str = ""
    necessity_score: Optional[int] = None
    model_necessity_score: Optional[int] = None
    rationale: str = ""


class DiagnosisTestRecommendationModelOutput(BaseModel):
    suggested_test_score: Dict[str, DiagnosisTestScoreItem] = Field(default_factory=dict)


class DiagnosisLLMOutputs(BaseModel):
    """Multi-model aggregated view, suitable for post-processing / tracing."""

    case_id: str
    diagnosis: Dict[str, DiagnosisLlmModelOutput]
    test_recommendation: Optional[Dict[str, DiagnosisTestRecommendationModelOutput]] = None


class DiagnosisRerankBundle(BaseModel):
    """Runtime input provided to `DiagnosisRerankStage` (in-memory state)."""

    case_id: str
    patient_data: Dict[str, Any]
    llm_outputs: Dict[str, Dict[str, Any]]
    llm_test_outputs: Optional[Dict[str, Dict[str, Any]]] = None
    followup_outputs: Optional[Dict[str, Dict[str, Any]]] = None
    diagnostic_test_data: Optional[Dict[str, Any]] = None


class DiagnosisStagePrediction(BaseModel):
    rank: int
    orphacode: int
    diagnosis_name: str
    score: float


class DiagnosisRerankOutput(BaseModel):
    """ML rerank stage output."""

    top_predictions: List[DiagnosisStagePrediction]
    reasoning: Optional[Dict[str, Any]] = None
    tests: Optional[List[Dict[str, Any]]] = None
    further_diagnostic_test: Optional[Dict[str, TestRecommendation]] = None


__all__ = [
    "FollowUpInput",
    "DiagnosisReason",
    "TestRecommendation",
    "FirstVisitDiagnosisOutput",
    "FollowUpDiagnosisOutput",
    "DiagnosisPrimaryConsultationLLMOutput",
    "DiagnosisFollowUpConsultationLLMOutput",
    "DiagnosisLlmModelOutput",
    "DiagnosisTestScoreItem",
    "DiagnosisTestRecommendationModelOutput",
    "DiagnosisLLMOutputs",
    "DiagnosisRerankBundle",
    "DiagnosisStagePrediction",
    "DiagnosisRerankOutput",
]
