"""Shared clinical schema blocks."""

from __future__ import annotations

from pydantic import BaseModel


class BasicInformation(BaseModel):
    age: str
    gender: str
    weight: str
    height: str
    BMI: str
    ethnicity: str
    occupation: str


class MedicalHistory(BaseModel):
    chief_complaint: str
    history_of_present_illness: str
    past_medical_history: str
    family_history: str
    social_history: str


class PhysicalExamination(BaseModel):
    general_examination: str
    specialty_examination: str


class ExaminationItem(BaseModel):
    procedure_name: str
    date: str
    findings: str
    changes_over_time: str


class DiagnosisInfo(BaseModel):
    final_diagnosis: str
    diagnostic_reasoning: str


class TreatmentHistory(BaseModel):
    treatment_type: str
    specific_treatment: str
    start_date: str
    end_date: str
    dosage_or_details: str


__all__ = [
    "BasicInformation",
    "MedicalHistory",
    "PhysicalExamination",
    "ExaminationItem",
    "DiagnosisInfo",
    "TreatmentHistory",
]
