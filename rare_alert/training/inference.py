"""RareAlert risk stage: input JSON -> OpenAI-compatible LLM -> parse -> RiskOutput."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Dict, Optional, Protocol, runtime_checkable

from core_tool.config import RiskStageConfig
from core_tool.llm.factory import build_llm_client
from core_tool.llm.guided_json.risk import risk_guided_json_schema
from core_tool.parser.risk_input_normalize import normalize_risk_input_dict
from core_tool.parser.risk_output_parser import parse_risk_response
from core_tool.prompt.templates.risk_assessment import build_risk_assessment_prompt
from schema import RiskInput

logger = logging.getLogger(__name__)

RISK_SYSTEM_MESSAGE = (
    "You are a helpful assistant designed to output the final answer in JSON."
)


@runtime_checkable
class LLMBackend(Protocol):
    """OpenAI-compatible sync backend; aligned with ``core_tool.llm.client.LLMClient.call``."""

    def call(
        self,
        prompt: str,
        max_tokens: int = 800,
        temperature: float = 0.0,
        extra_body: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None,
        response_format: Optional[Dict[str, Any]] = None,
        *,
        system: Optional[str] = None,
        stream: bool = False,
    ) -> str: ...


def _safe_json_loads(payload: str) -> Dict[str, Any]:
    if not isinstance(payload, str) or not payload.strip():
        return {}
    try:
        obj = json.loads(payload)
        return obj if isinstance(obj, dict) else {}
    except json.JSONDecodeError:
        return {}


def _normalize_risk_input(data: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(data, dict):
        return {}
    return normalize_risk_input_dict(data)


class RiskStage:
    """Risk stage: parse input, call LLM, return strict RiskOutput (no file I/O)."""

    def __init__(
        self,
        config: RiskStageConfig | None = None,
        *,
        llm_client: LLMBackend | None = None,
    ) -> None:
        self.cfg = config or RiskStageConfig()
        if llm_client is not None and not isinstance(llm_client, LLMBackend):
            raise TypeError(
                "llm_client must implement LLMBackend (a synchronous .call(prompt, **kwargs) -> str)."
            )
        self._llm_client = llm_client

    def _build_llm_client(self) -> LLMBackend:
        return build_llm_client(self.cfg)

    def _resolve_client(self) -> LLMBackend:
        client = self._llm_client if self._llm_client is not None else self._build_llm_client()
        self._last_client = client
        return client

    def _llm_rf_and_system(self) -> tuple[Optional[Dict[str, Any]], str]:
        rf: Optional[Dict[str, Any]] = (
            {"type": "json_object"} if self.cfg.json_object_response else None
        )
        sys_msg = (self.cfg.system_message or "").strip() or RISK_SYSTEM_MESSAGE
        return rf, sys_msg

    def _prepare_run(self, payload: str) -> tuple[str, LLMBackend]:
        t0 = time.perf_counter()
        if self.cfg.verbose:
            logger.info("[RiskStage] 1/5 Parsing and validating input")
        data = _normalize_risk_input(_safe_json_loads(payload))
        risk_input = RiskInput.model_validate(data).model_dump()
        if self.cfg.verbose:
            logger.info("[RiskStage] 1/5 Done, %.2f ms", (time.perf_counter() - t0) * 1000.0)

        t_prompt = time.perf_counter()
        if self.cfg.verbose:
            logger.info("[RiskStage] 2/5 Building prompt")
        prompt = build_risk_assessment_prompt(risk_input)
        # For debugging/raw capture
        self._last_prompt = prompt
        if self.cfg.verbose:
            logger.info("[RiskStage] 2/5 Done, prompt_len=%d, %.2f ms", len(prompt), (time.perf_counter() - t_prompt) * 1000.0)

        t_client = time.perf_counter()
        if self.cfg.verbose:
            logger.info("[RiskStage] 3/5 Resolving LLM client")
        client = self._resolve_client()
        if self.cfg.verbose:
            logger.info("[RiskStage] 3/5 Done, %.2f ms", (time.perf_counter() - t_client) * 1000.0)
        return prompt, client

    def _invoke_llm(self, client: LLMBackend, prompt: str) -> str:
        if self.cfg.verbose:
            logger.info("[RiskStage] 4/5 Calling model: %s", self.cfg.model)
        t_llm = time.perf_counter()
        rf, sys_msg = self._llm_rf_and_system()
        extra_body: Optional[Dict[str, Any]] = (
            dict(self.cfg.extra_body) if isinstance(self.cfg.extra_body, dict) else None
        )
        temperature = 0.0 if self.cfg.use_guided_json else self.cfg.temperature
        if self.cfg.use_guided_json:
            if extra_body is None:
                extra_body = {}
            extra_body.setdefault("guided_json", risk_guided_json_schema())
        raw_text = client.call(
            prompt,
            max_tokens=self.cfg.max_tokens,
            temperature=temperature,
            model=self.cfg.model,
            response_format=rf,
            system=sys_msg,
            stream=self.cfg.stream,
            extra_body=extra_body,
        )
        # For debugging/raw capture
        self._last_raw_text = raw_text
        if self.cfg.verbose:
            logger.info(
                "[RiskStage] 4/5 Done, response_len=%d, %.2f ms",
                len(raw_text or ""),
                (time.perf_counter() - t_llm) * 1000.0,
            )
            logger.info("[RiskStage] 4/5 Raw response:\n%s", raw_text or "(empty)")
        return raw_text

    @staticmethod
    def _is_parse_failure(result: dict) -> bool:
        """Return True if the result is a parse-failure placeholder (risk_score=0 and fixed sentinel text)."""
        insights = result.get("key_insights") or []
        if not insights:
            return True
        first = insights[0] if isinstance(insights[0], dict) else {}
        return str(first.get("insight1") or "").strip() == "(parse failure placeholder — not a clinical conclusion)"

    def _parse_and_finish(self, raw_text: str, t_pipeline_start: float) -> dict:
        t_parse = time.perf_counter()
        if self.cfg.verbose:
            logger.info("[RiskStage] 5/5 Parsing model output")
        parsed = parse_risk_response(
            raw_text,
            allow_freeform_fallback=self.cfg.freeform_parse_fallback,
        )
        result = parsed.model_dump(exclude_none=True)
        if self.cfg.verbose:
            logger.info("[RiskStage] 5/5 Done, %.2f ms", (time.perf_counter() - t_parse) * 1000.0)
            logger.info("[RiskStage] Total pipeline time: %.2f ms", (time.perf_counter() - t_pipeline_start) * 1000.0)
        return result

    def run_sync(self, payload: str) -> dict:
        """Full pipeline synchronously (for ThreadPoolExecutor batch; no nested asyncio.to_thread)."""
        t_pipeline_start = time.perf_counter()
        prompt, client = self._prepare_run(payload)
        for attempt in range(1 + self.cfg.max_parse_retries):
            raw_text = self._invoke_llm(client, prompt)
            result = self._parse_and_finish(raw_text, t_pipeline_start)
            if not self._is_parse_failure(result):
                return result
            if attempt < self.cfg.max_parse_retries:
                logger.warning(
                    "[RiskStage] Parse failed, retry %d/%d",
                    attempt + 1, self.cfg.max_parse_retries,
                )
        return result

    async def run(self, payload: str) -> dict:
        """Async entry: only the blocking LLM call runs in a worker thread."""
        t_pipeline_start = time.perf_counter()
        prompt, client = self._prepare_run(payload)
        result: dict = {}
        for attempt in range(1 + self.cfg.max_parse_retries):
            raw_text = await asyncio.to_thread(self._invoke_llm, client, prompt)
            result = self._parse_and_finish(raw_text, t_pipeline_start)
            if not self._is_parse_failure(result):
                return result
            if attempt < self.cfg.max_parse_retries:
                logger.warning(
                    "[RiskStage] Parse failed, retry %d/%d",
                    attempt + 1, self.cfg.max_parse_retries,
                )
        return result
