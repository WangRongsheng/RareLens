"""LLM payload and call metadata schemas."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class RawLlmTurn(BaseModel):
    """Single model call record (embeddable in raw_llm_outputs of each stage)."""

    model: str = ""
    role: str = "assistant"
    content: str = ""
    finish_reason: Optional[str] = None
    usage: Optional[Dict[str, Any]] = None


class ParsedLlmJson(BaseModel):
    """Container for parsed JSON objects (keys vary by task)."""

    data: Dict[str, Any] = Field(default_factory=dict)
    parse_ok: bool = True
    parse_error: Optional[str] = None


class RiskLlmPayload(BaseModel):
    """Optional structure aligned with rare_alert risk parsing results."""

    key_insights: Optional[List[Dict[str, Any]]] = None
    risk_score: Optional[int] = None
    risk_explanation: Optional[str] = None


class LLMCallRecord(BaseModel):
    """Record for each LLM call, used for token counting and latency tracking."""

    model: str
    stage: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    success: bool = True
    timestamp: Optional[str] = None
    error: Optional[str] = None


__all__ = [
    "RawLlmTurn",
    "ParsedLlmJson",
    "RiskLlmPayload",
    "LLMCallRecord",
]
