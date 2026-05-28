"""Risk stage schemas."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, ConfigDict, model_validator

from .shared import BasicInformation, MedicalHistory, PhysicalExamination


class RiskInput(BaseModel):
    basic_information: BasicInformation
    medical_history: MedicalHistory
    physical_examination: PhysicalExamination


class InsightItem(BaseModel):
    # Allow ignoring deprecated fields (e.g. padded) when reading old files; shape is still validated on construction.
    model_config = ConfigDict(extra="ignore")

    insight1: Optional[str] = None
    insight2: Optional[str] = None
    insight3: Optional[str] = None
    insight4: Optional[str] = None
    insight5: Optional[str] = None
    weight: float
    description: str

    @model_validator(mode="after")
    def _single_insight_key(self) -> "InsightItem":
        keys = ("insight1", "insight2", "insight3", "insight4", "insight5")
        present = [k for k in keys if str(getattr(self, k) or "").strip()]
        if len(present) != 1:
            raise ValueError("Each key_insight must contain exactly one key from insight1~insight5.")
        return self


class RiskOutput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    key_insights: List[InsightItem]
    risk_score: int
    risk_explanation: str

    @model_validator(mode="after")
    def _legacy_key_insights_shape(self) -> "RiskOutput":
        if len(self.key_insights) != 5:
            raise ValueError("key_insights must contain exactly 5 items.")
        expected = {"insight1", "insight2", "insight3", "insight4", "insight5"}
        actual = set()
        for item in self.key_insights:
            for key in expected:
                if str(getattr(item, key) or "").strip():
                    actual.add(key)
        if actual != expected:
            raise ValueError("key_insights must contain all of insight1~insight5, each appearing exactly once.")
        return self


__all__ = [
    "RiskInput",
    "InsightItem",
    "RiskOutput",
]
