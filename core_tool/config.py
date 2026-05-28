"""Stage-specific configs (YAML + Pydantic). Shared LLM fields live on ``LLMCallConfig``."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field

from core_tool.io_utils import read_config


class LLMCallConfig(BaseModel):
    """Parameters shared by all stages that call ``LLMClient``."""

    model_config = ConfigDict(extra="ignore")

    model: str = ""
    base_url: str = ""
    api_key: str = ""
    max_tokens: int = 4096
    temperature: float = 0.0
    max_retries: int = 10
    retry_delay_sec: float = 2.0
    dry_run: bool = False
    stream: bool = False
    json_object_response: bool = True
    system_message: str = ""
    #: Extra fields passed through to the OpenAI-compatible API, e.g. {"enable_thinking": false} to disable Qwen3 thinking mode.
    extra_body: Optional[Dict[str, Any]] = None


class RiskStageConfig(LLMCallConfig):
    """RareAlert risk stage: LLM settings + CLI/runtime flags."""

    model: str = Field(default="/root/autodl-tmp/qwen3_32b/med_h1")
    verbose: bool = False
    early_stop_threshold: int = 30
    #: If true, inject vLLM guided decoding schema via ``extra_body.guided_json``.
    #: This constrains output to the strict legacy JSON shape (5 insights, weight>0, score 0-100).
    use_guided_json: bool = False
    #: True: fall back to regex/heuristic field extraction only when JSON parsing fails (prone to mis-parsing). False: trust JSON only; return a placeholder on failure.
    freeform_parse_fallback: bool = False
    #: Maximum number of parse retries on failure (excluding the first attempt); 0 means no retries.
    max_parse_retries: int = 2

    @classmethod
    def from_yaml(cls, path: str | Path) -> RiskStageConfig:
        data = read_config(path)
        return cls.model_validate(data)


__all__ = ["LLMCallConfig", "RiskStageConfig"]
