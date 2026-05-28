"""Reasoning trace schemas."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class TraceEvidence(BaseModel):
    model: str
    matched_item: str = ""
    model_confidence_score: Optional[int] = None
    reasoning: str = ""
    meta: Optional[Dict[str, Any]] = None


class TraceItem(BaseModel):
    item_key: str
    item_name: str
    final_confidence_score: Optional[int] = None
    final_reasoning: str = ""
    aggregation_method: str = ""
    evidences: List[TraceEvidence]
    meta: Optional[Dict[str, Any]] = None


class ReasoningTrace(BaseModel):
    case_id: str
    stage: str
    module: str
    items: List[TraceItem]


__all__ = [
    "TraceEvidence",
    "TraceItem",
    "ReasoningTrace",
]
