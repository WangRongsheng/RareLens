"""Optional multi-model prognosis_prediction_output generation.

Aligned with rare_treatment / rare_diagnosis: per-model timeout, bounded concurrency,
optional total timeout, retries from config, partial success + failed_models meta.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from configs.models import PROGNOSIS_MODELS
from core_tool.io_utils import dump_json, read_json
from core_tool.llm.client import LLMClient, _load_dotenv_if_present
from core_tool.llm.credential_resolver import resolve_prognosis
from core_tool.llm.token_tracker import stats_from_client
from core_tool.parser.json_extractor import extract
from core_tool.prompt.templates.prognosis_generation import PROGNOSIS_LLM_PROMPT
from schema import validate_prognosis_prediction_output


def _default_oai_config_path() -> Path:
    return (Path(__file__).resolve().parents[1] / "OAI_Config_List.json").resolve()


@lru_cache(maxsize=1)
def _load_oai_model_mappings() -> Dict[str, str]:
    path = _default_oai_config_path()
    if not path.is_file():
        return {}
    try:
        obj = read_json(path)
    except Exception:
        return {}
    if not isinstance(obj, list):
        return {}
    mapping: Dict[str, str] = {}
    for item in obj:
        if not isinstance(item, dict):
            continue
        actual_model = str(item.get("model") or "").strip()
        if not actual_model:
            continue
        tags = item.get("tags", [])
        if isinstance(tags, list):
            for tag in tags:
                tag_text = str(tag or "").strip()
                if tag_text:
                    mapping[tag_text] = actual_model
        mapping[actual_model] = actual_model
    return mapping


def _resolve_prognosis_model_tag(tag: str) -> str:
    t = str(tag or "").strip()
    if not t:
        return ""
    return _load_oai_model_mappings().get(t, t)

logger = logging.getLogger(__name__)


def _resolve_prognosis_gen_api_key(model_name: str) -> str:
    return resolve_prognosis(model_name=model_name).api_key


def _resolve_prognosis_gen_base_url(model_name: str) -> str:
    return resolve_prognosis(model_name=model_name).base_url


def _dry_run_stub_prognosis_payload() -> Dict[str, Any]:
    """Valid minimal object for schema validation when API is skipped."""
    return {
        "overall_outcome": {
            "outcome_category": "stabilization",
            "confidence_score": 5,
            "explanation": "[dry_run] placeholder prognosis output",
        },
        "functional_status": {
            "status": "mild",
            "confidence_score": 5,
            "explanation": "[dry_run] placeholder prognosis output",
        },
        "symptom_burden": {
            "burden": "occasional",
            "confidence_score": 5,
            "explanation": "[dry_run] placeholder prognosis output",
        },
        "clinical_events": [],
    }


def _build_prognosis_prompt(*, aggregated_record: Dict[str, Any]) -> str:
    return PROGNOSIS_LLM_PROMPT.format(
        content=json.dumps(aggregated_record, ensure_ascii=False),
    )


def _call_prognosis_model_sync(
    *,
    aggregated_record: Dict[str, Any],
    model_name: str,
    dry_run: bool = False,
    max_tokens: int = 4096,
    temperature: float = 0.0,
    stream: bool = False,
    max_retries: int = 3,
    retry_delay_sec: float = 2.0,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Returns (payload, stats) where stats includes token counts and latency."""
    if dry_run:
        payload = validate_prognosis_prediction_output(
            _dry_run_stub_prognosis_payload(),
            source=f"dry_run_prognosis[{model_name}]",
        ).model_dump(exclude_none=True)
        return payload, {"model": model_name, "total_tokens": 0, "prompt_tokens": 0, "completion_tokens": 0, "latency_seconds": 0.0}

    api_key = _resolve_prognosis_gen_api_key(model_name)
    if not api_key:
        raise RuntimeError(
            f"Missing API key for prognosis generation "
            f"(set Prognosis_OPENAI_API_KEY or Qwen3_OPENAI_API_KEY for qwen3): {model_name}"
        )

    actual_model = _resolve_prognosis_model_tag(model_name) or model_name
    client = LLMClient(
        api_key=api_key,
        model=str(actual_model).strip(),
        dry_run=False,
        base_url=_resolve_prognosis_gen_base_url(model_name),
        max_retries=max(1, int(max_retries)),
        retry_delay_sec=float(retry_delay_sec),
    )
    model_norm = str(model_name or "").strip().lower()
    if model_norm.startswith("qwen3"):
        extra_body = {"enable_thinking": False}
    elif model_norm == "gpt-5":
        extra_body = {"reasoning_effort": "medium"}
    else:
        extra_body = None

    effective_max_tokens = 16384 if "gemini" in model_norm else max_tokens
    # Reasoning models that don't accept temperature: gpt-5, deepseek-r1-*.
    effective_temperature = None if model_norm in ("gpt-5",) or model_norm.startswith("deepseek-r1") else temperature
    t_start = time.monotonic()
    raw_text = client.call(
        prompt=_build_prognosis_prompt(aggregated_record=aggregated_record),
        max_tokens=effective_max_tokens,
        temperature=effective_temperature,
        extra_body=extra_body,
        stream=stream,
    )
    latency = time.monotonic() - t_start
    parsed = extract(raw_text or "")
    if not isinstance(parsed, dict):
        raise RuntimeError(
            f"Prognosis generation failed for model={model_name}: invalid JSON response"
        )
    validated = validate_prognosis_prediction_output(
        parsed,
        source=f"generate_prognosis[{model_name}]",
    )
    stats = stats_from_client(client, model_name=actual_model or model_name, latency=latency)
    return validated.model_dump(exclude_none=True), stats


def _resolve_generation_models(
    *,
    cfg: Dict[str, Any],
    generation_models: Optional[Iterable[str]] = None,
) -> List[str]:
    raw = generation_models if generation_models is not None else cfg.get("generation_models")
    if isinstance(raw, str):
        items = [part.strip() for part in raw.split(",")]
    elif isinstance(raw, (list, tuple, set)):
        items = [str(part).strip() for part in raw]
    else:
        items = list(PROGNOSIS_MODELS)
    seen = set()
    out: List[str] = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _load_aggregated_prognosis_input(case_bundle: Dict[str, Any]) -> Dict[str, Any]:
    raw = case_bundle.get("prognosis_prediction_input")
    if isinstance(raw, dict) and raw:
        return raw
    pin = case_bundle.get("_prognosis_prediction_input_path")
    if isinstance(pin, str) and pin.strip():
        p = Path(pin).resolve()
        if p.is_file():
            obj = read_json(p)
            return obj if isinstance(obj, dict) else {}
    return {}


async def _generate_one_prognosis_model_payload(
    *,
    aggregated_record: Dict[str, Any],
    model_name: str,
    dry_run_llm_generation: bool,
    max_tokens: int,
    gen_temperature: float,
    gen_stream: bool,
    max_retries: int,
    retry_delay_sec: float,
    per_model_timeout: float,
) -> Tuple[str, Optional[Dict[str, Any]], Optional[str], Dict[str, Any]]:
    """Async single-model call; returns (model_name, payload, error, stats)."""
    try:
        payload, stats = await asyncio.wait_for(
            asyncio.to_thread(
                _call_prognosis_model_sync,
                aggregated_record=aggregated_record,
                model_name=model_name,
                dry_run=bool(dry_run_llm_generation),
                max_tokens=max_tokens,
                temperature=gen_temperature,
                stream=gen_stream,
                max_retries=max_retries,
                retry_delay_sec=retry_delay_sec,
            ),
            timeout=per_model_timeout,
        )
        return model_name, payload, None, stats
    except asyncio.TimeoutError:
        return model_name, None, f"timeout after {per_model_timeout:.0f}s", {}
    except Exception as exc:
        logger.warning("Prognosis generation failed for model=%s: %s", model_name, exc)
        return model_name, None, str(exc), {}


async def _generate_case_bundle_prognosis_llm(
    *,
    aggregated_record: Dict[str, Any],
    model_list: List[str],
    dry_run_llm_generation: bool,
    max_tokens: int,
    gen_temperature: float,
    gen_stream: bool,
    max_retries: int,
    retry_delay_sec: float,
    per_model_timeout: float,
    max_concurrency: int,
    total_timeout: Optional[float],
    progress_hook: Optional[Callable[[str], None]] = None,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, str], List[str], Dict[str, Dict[str, Any]]]:
    """
    Multi-model concurrent generation; mirrors ``_generate_case_bundle_llm`` (diagnosis).

    Returns:
      (llm_outputs, failed_models, generated_models_order, model_token_stats)
    """
    llm_outputs: Dict[str, Dict[str, Any]] = {}
    failed_models: Dict[str, str] = {}
    generated_order: List[str] = []
    model_token_stats: Dict[str, Dict[str, Any]] = {}

    sem = asyncio.Semaphore(max(1, int(max_concurrency)))
    start = time.monotonic()

    async def _one_with_timeout(model_name: str) -> Tuple[str, Optional[Dict[str, Any]], Optional[str], Dict[str, Any]]:
        return await _generate_one_prognosis_model_payload(
            aggregated_record=aggregated_record,
            model_name=model_name,
            dry_run_llm_generation=dry_run_llm_generation,
            max_tokens=max_tokens,
            gen_temperature=gen_temperature,
            gen_stream=gen_stream,
            max_retries=max_retries,
            retry_delay_sec=retry_delay_sec,
            per_model_timeout=per_model_timeout,
        )

    async def _one_limited(model_name: str) -> Tuple[str, Optional[Dict[str, Any]], Optional[str], Dict[str, Any]]:
        async with sem:
            return await _one_with_timeout(model_name)

    tasks = [asyncio.create_task(_one_limited(str(m))) for m in model_list]
    try:
        for fut in asyncio.as_completed(tasks):
            if total_timeout is not None:
                remaining = float(total_timeout) - (time.monotonic() - start)
                if remaining <= 0:
                    raise asyncio.TimeoutError("total_timeout reached")
                model_name, payload, error, stats = await asyncio.wait_for(fut, timeout=remaining)
            else:
                model_name, payload, error, stats = await fut
            if payload is not None:
                llm_outputs[model_name] = payload
                generated_order.append(model_name)
                if stats:
                    model_token_stats[model_name] = stats
                if callable(progress_hook):
                    progress_hook(f"  ✓ {model_name}: prognosis_prediction_output 已生成")
            elif error:
                failed_models[model_name] = error
                if callable(progress_hook):
                    progress_hook(f"  ✗ {model_name}: {error[:120]}")
    except asyncio.TimeoutError:
        msg = (
            f"total_timeout after {float(total_timeout or 0):.0f}s"
            if total_timeout is not None
            else "timeout"
        )
        logger.warning("prognosis LLM generation timeout; cancelling remaining tasks (%s).", msg)
        for t in tasks:
            if not t.done():
                t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        if not llm_outputs:
            raise RuntimeError(f"Prognosis LLM generation failed: {msg}")
    finally:
        for t in tasks:
            if not t.done():
                t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    if not llm_outputs:
        raise RuntimeError("Prognosis LLM generation failed for all models")
    return llm_outputs, failed_models, generated_order, model_token_stats


async def ensure_bundle_with_prognosis_llm(
    *,
    case_bundle: Dict[str, Any],
    cfg: Optional[Dict[str, Any]] = None,
    generation_models: Optional[Iterable[str]] = None,
    auto_generate_llm_if_missing: bool = False,
    force_generate_llm: bool = False,
    dry_run_llm_generation: bool = False,
    progress_hook: Optional[Callable[[str], None]] = None,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    """
    Same role as rare_diagnosis ``_ensure_bundle_with_llm``: validate existing,
    optionally generate missing models, merge ``llm_outputs``.
    """
    cfg = dict(cfg or {})
    existing_raw = case_bundle.get("llm_outputs", {})
    existing: Dict[str, Dict[str, Any]] = {}
    if isinstance(existing_raw, dict):
        for model_name, payload in existing_raw.items():
            if not isinstance(payload, dict) or not payload:
                continue
            try:
                validate_prognosis_prediction_output(
                    payload,
                    source=f"llm_outputs[{model_name}]",
                )
                existing[str(model_name)] = payload
            except Exception as exc:
                logger.warning("skip invalid existing prognosis output %s: %s", model_name, exc)

    model_list = _resolve_generation_models(cfg=cfg, generation_models=generation_models)
    meta: Dict[str, Any] = {
        "enabled": bool(auto_generate_llm_if_missing),
        "force_generate": bool(force_generate_llm),
        "requested_models": list(model_list),
        "existing_count": len(existing),
        "generated_models": [],
        "failed_models": {},
        "mode": "reuse_existing",
    }

    if not auto_generate_llm_if_missing and not force_generate_llm:
        if existing:
            return existing, meta
        raise ValueError(
            "case_bundle 缺少 llm_outputs，且未开启在线生成。"
            "请提供磁盘上的各模型 prognosis_prediction_output.json，或在 CLI 加 --generate-llm-if-missing。"
        )

    aggregated = _load_aggregated_prognosis_input(case_bundle)
    target_models = [
        model_name
        for model_name in model_list
        if force_generate_llm or model_name not in existing
    ]
    if not target_models:
        return existing, meta

    if not aggregated:
        raise ValueError(
            "无法自动生成预后 LLM 输出：缺少 prognosis_prediction.json（或 prognosis_prediction_input）。"
            "请将其放在与 patient-json 同目录，或传入 --prognosis-input-json。"
        )

    meta["mode"] = "real" if not dry_run_llm_generation else "dry_run"
    max_tokens = int(cfg.get("generation_max_tokens", 4096))
    gen_temperature = float(cfg.get("generation_temperature", 0.0))
    gen_stream = bool(cfg.get("generation_stream", False))
    per_model_timeout = float(cfg.get("generation_per_model_timeout", 120.0))
    max_concurrency = max(1, int(cfg.get("generation_max_concurrency", 8)))
    total_timeout_raw = cfg.get("generation_total_timeout_sec")
    total_timeout: Optional[float] = None
    if total_timeout_raw is not None and str(total_timeout_raw).strip():
        try:
            total_timeout = float(total_timeout_raw)
        except (TypeError, ValueError):
            total_timeout = None
    max_retries = int(cfg.get("generation_max_retries", 3))
    retry_delay_sec = float(cfg.get("generation_retry_delay_sec", 2.0))

    meta["per_model_timeout_sec"] = per_model_timeout
    meta["max_concurrency"] = max_concurrency
    meta["total_timeout_sec"] = total_timeout
    meta["models_total"] = len(target_models)

    if callable(progress_hook):
        progress_hook(
            f"预后 LLM 生成：准备 {len(target_models)} 个模型 "
            f"(mode={meta['mode']}, per_model_timeout={per_model_timeout:.0f}s, concurrency={max_concurrency})",
        )

    gen_out, failed, gen_order, model_token_stats = await _generate_case_bundle_prognosis_llm(
        aggregated_record=aggregated,
        model_list=target_models,
        dry_run_llm_generation=dry_run_llm_generation,
        max_tokens=max_tokens,
        gen_temperature=gen_temperature,
        gen_stream=gen_stream,
        max_retries=max_retries,
        retry_delay_sec=retry_delay_sec,
        per_model_timeout=per_model_timeout,
        max_concurrency=max_concurrency,
        total_timeout=total_timeout,
        progress_hook=progress_hook,
    )
    meta["failed_models"] = failed
    meta["generated_models"] = gen_order
    meta["models_succeeded"] = len(gen_order)
    meta["models_failed"] = len(failed)
    meta["model_token_stats"] = model_token_stats

    outputs = {**existing, **gen_out}
    if not outputs:
        raise RuntimeError("Prognosis LLM generation failed for all models")
    return outputs, meta


# Backward-compatible name for run_prog_pipeline and external callers
ensure_prognosis_llm_outputs = ensure_bundle_with_prognosis_llm


def persist_prognosis_model_outputs(
    *,
    case_id: str,
    llm_outputs: Dict[str, Dict[str, Any]],
    case_output_root: Path,
) -> None:
    """Write per-model JSON under case_output_root/<case_id>/<model>/<case_id>/."""
    cid = str(case_id)
    root = Path(case_output_root).resolve()
    for model_name, payload in (llm_outputs or {}).items():
        if not isinstance(payload, dict) or not payload:
            continue
        nested = root / cid / str(model_name)
        nested.mkdir(parents=True, exist_ok=True)
        dump_json(nested / "prognosis_prediction_output.json", payload)
