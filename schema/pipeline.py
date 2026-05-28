"""Pipeline-wide aggregate schemas."""

from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

from .diagnosis import FirstVisitDiagnosisOutput, FollowUpDiagnosisOutput, FollowUpInput
from .prognosis import PrognosisAggregateOutput, PrognosisInput
from .risk import RiskInput, RiskOutput
from .shared import (
    BasicInformation,
    DiagnosisInfo,
    ExaminationItem,
    MedicalHistory,
    PhysicalExamination,
    TreatmentHistory,
)
from .treatment import TreatmentAggregateOutput, TreatmentInput


class PatientCase(BaseModel):
    """Full-pipeline patient case: fields can be filled incrementally; aligned with the original per-stage input model fields."""

    case_id: str = ""
    basic_information: BasicInformation
    medical_history: MedicalHistory
    physical_examination: PhysicalExamination
    laboratory_examinations: list[ExaminationItem] = Field(default_factory=list)
    radiographic_examinations: list[ExaminationItem] = Field(default_factory=list)
    other_tests: list[ExaminationItem] = Field(default_factory=list)
    diagnosis: Optional[DiagnosisInfo] = None
    treatment_information: list[TreatmentHistory] = Field(default_factory=list)

    @classmethod
    def from_risk_input(cls, case_id: str, inp: RiskInput) -> "PatientCase":
        return cls(
            case_id=case_id,
            basic_information=inp.basic_information,
            medical_history=inp.medical_history,
            physical_examination=inp.physical_examination,
        )

    @classmethod
    def from_follow_up(cls, case_id: str, inp: FollowUpInput) -> "PatientCase":
        return cls(
            case_id=case_id,
            basic_information=inp.basic_information,
            medical_history=inp.medical_history,
            physical_examination=inp.physical_examination,
            laboratory_examinations=list(inp.laboratory_examinations or []),
            radiographic_examinations=list(inp.radiographic_examinations or []),
            other_tests=list(inp.other_tests or []),
        )

    @classmethod
    def from_treatment_input(cls, case_id: str, inp: TreatmentInput) -> "PatientCase":
        base = cls.from_follow_up(case_id, inp)
        return base.model_copy(update={"diagnosis": inp.diagnosis})

    @classmethod
    def from_prognosis_input(cls, case_id: str, inp: PrognosisInput) -> "PatientCase":
        base = cls.from_treatment_input(case_id, inp)
        return base.model_copy(
            update={"treatment_information": list(inp.treatment_information or [])}
        )

    def as_risk_input(self) -> RiskInput:
        return RiskInput(
            basic_information=self.basic_information,
            medical_history=self.medical_history,
            physical_examination=self.physical_examination,
        )

    def as_follow_up_input(self) -> FollowUpInput:
        return FollowUpInput(
            basic_information=self.basic_information,
            medical_history=self.medical_history,
            physical_examination=self.physical_examination,
            laboratory_examinations=self.laboratory_examinations,
            radiographic_examinations=self.radiographic_examinations,
            other_tests=self.other_tests,
        )

    def as_treatment_input(self) -> TreatmentInput:
        if self.diagnosis is None:
            raise ValueError("PatientCase.diagnosis is None; cannot construct TreatmentInput")
        return TreatmentInput(
            basic_information=self.basic_information,
            medical_history=self.medical_history,
            physical_examination=self.physical_examination,
            laboratory_examinations=self.laboratory_examinations,
            radiographic_examinations=self.radiographic_examinations,
            other_tests=self.other_tests,
            diagnosis=self.diagnosis,
        )

    def as_prognosis_input(self) -> PrognosisInput:
        ti = self.as_treatment_input()
        return PrognosisInput(
            **ti.model_dump(),
            treatment_information=self.treatment_information,
        )


class PipelineCase(BaseModel):
    case_id: str
    risk: RiskOutput
    first_visit_diagnosis: FirstVisitDiagnosisOutput
    follow_up_diagnosis: FollowUpDiagnosisOutput
    treatment: TreatmentAggregateOutput
    prognosis: PrognosisAggregateOutput
    evaluation: Optional[Dict[str, Any]] = None


__all__ = [
    "PatientCase",
    "PipelineCase",
]
