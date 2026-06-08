"""Stage-organized schema package — Input/Output format definitions for each pipeline stage."""

from .shared import (
    BasicInformation,
    DiagnosisInfo,
    ExaminationItem,
    MedicalHistory,
    PhysicalExamination,
    TreatmentHistory,
)
from .risk import InsightItem, RiskInput, RiskOutput
from .diagnosis import (
    DiagnosisReason,
    FirstVisitDiagnosisOutput,
    FollowUpDiagnosisOutput,
    FollowUpInput,
    PrimaryInput,
    TestRecommendation,
)
from .treatment import (
    TreatmentGoal,
    TreatmentInput,
    TreatmentItem,
    TreatmentOutput,
)
from .prognosis import (
    ClinicalEvent,
    FunctionalStatus,
    OverallOutcome,
    PrognosisInput,
    PrognosisPredictionOutput,
    PrognosisTarget,
    SymptomBurden,
)

__all__ = [
    # shared
    "BasicInformation",
    "MedicalHistory",
    "PhysicalExamination",
    "ExaminationItem",
    "DiagnosisInfo",
    "TreatmentHistory",
    # risk (alert)
    "RiskInput",
    "InsightItem",
    "RiskOutput",
    # diagnosis
    "PrimaryInput",
    "FollowUpInput",
    "DiagnosisReason",
    "TestRecommendation",
    "FirstVisitDiagnosisOutput",
    "FollowUpDiagnosisOutput",
    # treatment
    "TreatmentInput",
    "TreatmentGoal",
    "TreatmentItem",
    "TreatmentOutput",
    # prognosis
    "PrognosisInput",
    "PrognosisTarget",
    "OverallOutcome",
    "FunctionalStatus",
    "SymptomBurden",
    "ClinicalEvent",
    "PrognosisPredictionOutput",
]
