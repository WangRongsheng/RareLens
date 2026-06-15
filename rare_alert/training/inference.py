"""RareAlert risk stage: input JSON -> OpenAI-compatible LLM -> parse -> RiskOutput."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Tuple, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from schema import RiskInput

from .llm_client import LLMClient, build_llm_client
from .output_parser import parse_risk_response

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════════

def _read_config(path: str | Path) -> Dict[str, Any]:
    """Load JSON or YAML config file."""
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".json":
        data = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"Expected JSON object at root in file: {p}")
        return data
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except Exception as exc:
            raise RuntimeError("Reading YAML config requires pyyaml (pip install pyyaml).") from exc
        data = yaml.safe_load(p.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"Expected mapping at root in YAML config: {p}")
        return data
    # Unknown suffix: try JSON then YAML
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    try:
        import yaml  # type: ignore
    except Exception as exc:
        raise RuntimeError("Unknown config suffix and YAML parsing unavailable.") from exc
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping at root in config: {p}")
    return data


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
    extra_body: Optional[Dict[str, Any]] = None


class RiskStageConfig(LLMCallConfig):
    """RareAlert risk stage: LLM settings + CLI/runtime flags."""

    model: str = Field(default="rare_alert")
    verbose: bool = False
    early_stop_threshold: int = 30
    use_guided_json: bool = False
    freeform_parse_fallback: bool = False
    max_parse_retries: int = 2

    @classmethod
    def from_yaml(cls, path: str | Path) -> RiskStageConfig:
        data = _read_config(path)
        return cls.model_validate(data)


# ═══════════════════════════════════════════════════════════════════════════
# Input normalize
# ═══════════════════════════════════════════════════════════════════════════

def _normalize_risk_input_dict(data: Dict[str, Any]) -> Dict[str, Any]:
    """Map legacy / alternate keys to canonical RiskInput fields."""
    payload = dict(data)
    if "basic_information" not in payload and "basic_info" in payload:
        payload["basic_information"] = payload.get("basic_info")
    if "physical_examination" not in payload and "physical_exam" in payload:
        payload["physical_examination"] = payload.get("physical_exam")
    return payload


# ═══════════════════════════════════════════════════════════════════════════
# Prompt template
# ═══════════════════════════════════════════════════════════════════════════

RISK_ASSESSMENT_PROMPT = """
You are an expert clinician.
Your task is to
1. Read the patient's information;
2. Analyze if the patient may have a rare disease.
3. Output five most important key insights (signs and symptoms) that contribute to the risk of rare disease and assign weights to the key insights, where the weights should add up to 1 (for example, symptom xx weight 0.3, sign xxx weight 0.4).
4. Assign score for the risk that the patient may have a rare disease. The score is 0-100, where 0 indicates no risk and 100 indicates certainty that the patient has a rare disease.
5. Output the top five most possible rare disease diagnoses and explanation for each one.
6. Output explanation for your assessment of risk score.

Important alignment with the schema below:
- Do NOT add a separate "top five diagnoses" section outside this JSON. Any differential reasoning belongs inside risk_explanation as plain text only.

STRICT OUTPUT RULES (violations break downstream parsing):

You MUST respond with a single valid JSON object.
The top-level keys must be exactly: "key_insights" (array of 5 objects), "risk_score" (integer), "risk_explanation" (string).
Each object in key_insights must have keys: "insight<N>", "weight", "description" where <N> is 1 for the first object, 2 for the second, …, 5 for the fifth (never a generic "insight" key without a number).
Do NOT output any text outside the JSON object.

- Respond with that single JSON object only. No prose, markdown, or labels before or after it.
- Do not use markdown (no **headings**, no bullet lists outside JSON), no code fences (```), and no labels such as "Final Output" or "Answer:".
- key_insights MUST be a JSON array of exactly five objects. Each element MUST be a JSON object (never a bare string). Each object MUST include "weight" (JSON number) and "description" (string).
- You MUST provide exactly five key_insights entries (the array length must be 5). If fewer than five clinically distinct points exist, expand, split, or refine the most important findings so that all five slots are filled with substantive content. Never omit an entry or leave any slot empty.
- Insight keys (critical): The ONLY allowed insight field names are the strings "insight1", "insight2", "insight3", "insight4", and "insight5". Do NOT use a generic key named "insight" (without a number). The first array object MUST contain ONLY "insight1" (plus weight and description); the second object MUST contain ONLY "insight2"; … the fifth MUST contain ONLY "insight5". Never put "insight1" in more than one object.
- "risk_explanation" MUST be a single plain-language string. Do NOT embed a second JSON object or escaped JSON inside it.
- risk_score MUST appear once at the top level as a JSON integer (not quoted).
- Ensure the entire response is valid JSON: double quotes for keys/strings, commas between array elements, no trailing commas, no comments.

Output in the following json format:
{{
  "key_insights": [
    {{ "insight1": "string", "weight": "float", "description": "string" }},
    {{ "insight2": "string", "weight": "float", "description": "string" }},
    {{ "insight3": "string", "weight": "float", "description": "string" }},
    {{ "insight4": "string", "weight": "float", "description": "string" }},
    {{ "insight5": "string", "weight": "float", "description": "string" }}
  ],
  "risk_score": "integer",
  "risk_explanation": "string"
}}
Here is the patient's information:

{content}
"""


def _build_risk_assessment_prompt(risk_input: RiskInput | Dict[str, Any]) -> str:
    """Build prompt text from RiskInput (or a compatible dict)."""
    if isinstance(risk_input, RiskInput):
        payload = risk_input.model_dump()
    else:
        payload = RiskInput.model_validate(risk_input).model_dump()
    content = json.dumps(payload, ensure_ascii=False, indent=2)
    content = content.replace("___", "[REDACTED]")
    return RISK_ASSESSMENT_PROMPT.format(content=content)


# ═══════════════════════════════════════════════════════════════════════════
# Guided JSON schema (for vLLM guided decoding)
# ═══════════════════════════════════════════════════════════════════════════

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

class _RiskOutputGuided(BaseModel):
    model_config = ConfigDict(extra="forbid")
    key_insights: Tuple[_Insight1, _Insight2, _Insight3, _Insight4, _Insight5]
    risk_score: int = Field(..., ge=0, le=100)
    risk_explanation: str = Field(..., min_length=1)

def _risk_guided_json_schema() -> dict:
    return _RiskOutputGuided.model_json_schema()


# ═══════════════════════════════════════════════════════════════════════════
# RiskStage
# ═══════════════════════════════════════════════════════════════════════════

RISK_SYSTEM_MESSAGE = (
    "You are a helpful assistant designed to output the final answer in JSON."
)


@runtime_checkable
class LLMBackend(Protocol):
    """OpenAI-compatible sync backend."""

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
    return _normalize_risk_input_dict(data)


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
        prompt = _build_risk_assessment_prompt(risk_input)
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
            extra_body.setdefault("guided_json", _risk_guided_json_schema())
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
        insights = result.get("key_insights") or []
        if not insights:
            return True
        first = insights[0] if isinstance(insights[0], dict) else {}
        return str(first.get("insight1") or "").strip() == "(parse failure placeholder \u2014 not a clinical conclusion)"

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
        """Full pipeline synchronously (for ThreadPoolExecutor batch)."""
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
