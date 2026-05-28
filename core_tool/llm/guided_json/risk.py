"""Guided JSON schema for vLLM guided decoding (risk stage).

This schema is intentionally stricter than the legacy ``schema.risk.RiskOutput`` model.
It is used ONLY to constrain generation (guided_json), not for parsing old outputs.
"""

from __future__ import annotations

from typing import Tuple

from pydantic import BaseModel, ConfigDict, Field


class _Insight1(BaseModel):
    model_config = ConfigDict(extra="forbid")
    insight1: str = Field(..., min_length=1)
    weight: float = Field(..., gt=0)
    description: str = Field(..., min_length=10)


class _Insight2(BaseModel):
    model_config = ConfigDict(extra="forbid")
    insight2: str = Field(..., min_length=1)
    weight: float = Field(..., gt=0)
    description: str = Field(..., min_length=10)


class _Insight3(BaseModel):
    model_config = ConfigDict(extra="forbid")
    insight3: str = Field(..., min_length=1)
    weight: float = Field(..., gt=0)
    description: str = Field(..., min_length=10)


class _Insight4(BaseModel):
    model_config = ConfigDict(extra="forbid")
    insight4: str = Field(..., min_length=1)
    weight: float = Field(..., gt=0)
    description: str = Field(..., min_length=10)


class _Insight5(BaseModel):
    model_config = ConfigDict(extra="forbid")
    insight5: str = Field(..., min_length=1)
    weight: float = Field(..., gt=0)
    description: str = Field(..., min_length=10)


class RiskOutputGuided(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key_insights: Tuple[_Insight1, _Insight2, _Insight3, _Insight4, _Insight5]
    risk_score: int = Field(..., ge=0, le=100)
    risk_explanation: str = Field(..., min_length=1)


def risk_guided_json_schema() -> dict:
    """
    Return a JSON schema (dict) suitable for vLLM's ``guided_json`` parameter.
    """

    return RiskOutputGuided.model_json_schema()

