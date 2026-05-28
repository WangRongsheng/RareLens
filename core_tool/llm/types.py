"""LLM single-call result (shared with TokenLogger / adapters)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

# NOTE: LLMResponse is reserved for a richer return type (tokens + raw payload).
# core_tool.llm.client.LLMClient.call currently returns plain str only; wire-up is TODO.


@dataclass
class LLMResponse:
    text: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    raw: Dict[str, Any] = field(default_factory=dict)
