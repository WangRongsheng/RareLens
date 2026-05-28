"""Construct ``LLMClient`` from a typed ``LLMCallConfig`` (env vars fill missing URL/key)."""

from __future__ import annotations

import os

from core_tool.config import LLMCallConfig
from core_tool.llm.client import LLMClient, _load_dotenv_if_present


def build_llm_client(config: LLMCallConfig) -> LLMClient:
    _load_dotenv_if_present()
    api_key = (config.api_key or "").strip() or os.getenv("Alert_API_KEY", "") or os.getenv("QWEN_API_KEY", "")
    base_url = (config.base_url or "").strip() or os.getenv("Alert_URL", "") or os.getenv("QWEN_BASE_URL", "")
    return LLMClient(
        api_key=api_key,
        base_url=base_url,
        model=config.model,
        dry_run=config.dry_run,
        max_retries=config.max_retries,
        retry_delay_sec=config.retry_delay_sec,
    )
