"""
Centralized API credential resolution.

Routing is driven by MODEL_CREDENTIAL_SLOT in configs/models.py.
Each slot maps to a standard provider-level env var pair — the same
credentials are shared across all pipeline stages for the same model family.

Slot → env vars
────────────────────────────────────────────────────────────────
  "openai"    → OPENAI_API_KEY    / OPENAI_BASE_URL
  "gemini"    → GOOGLE_API_KEY    / GOOGLE_BASE_URL
  "qwen3"     → QWEN3_API_KEY     / QWEN3_BASE_URL
  "deepseek"  → DEEPSEEK_API_KEY  / DEEPSEEK_BASE_URL
  "anthropic" → ANTHROPIC_API_KEY / ANTHROPIC_BASE_URL
  "alert"     → ALERT_API_KEY     / ALERT_URL

Module-level overrides (optional, rarely needed):
  Diagnosis_OPENAI_API_KEY / BASE_URL   — overrides OPENAI_* for diagnosis only
  Prognosis_OPENAI_API_KEY / BASE_URL   — overrides OPENAI_* for prognosis only
  Treatment_OPENAI_API_KEY / BASE_URL   — overrides OPENAI_* for treatment only
────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from core_tool.llm.client import _load_dotenv_if_present


@dataclass(frozen=True)
class Credentials:
    api_key: str
    base_url: str


def _getenv(name: str) -> str:
    return str(os.getenv(name, "") or "").strip()


def _slot_for(model_name: str) -> str:
    """Return the credential slot for a model name (defaults to 'openai')."""
    from configs.models import MODEL_CREDENTIAL_SLOT
    return MODEL_CREDENTIAL_SLOT.get(str(model_name or "").strip(), "openai")


# Slot → (key_env, url_env)
_SLOT_ENVS = {
    "openai":    ("OPENAI_API_KEY",    "OPENAI_BASE_URL"),
    "gemini":    ("GOOGLE_API_KEY",    "GOOGLE_BASE_URL"),
    "qwen3":     ("QWEN3_API_KEY",     "QWEN3_BASE_URL"),
    "deepseek":  ("DEEPSEEK_API_KEY",  "DEEPSEEK_BASE_URL"),
    "anthropic": ("ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL"),
    "alert":     ("ALERT_API_KEY",     "ALERT_URL"),
}


def _resolve_slot(
    slot: str,
    *,
    module_key_env: str = "",
    module_url_env: str = "",
) -> Credentials:
    """
    Resolve credentials for a slot.

    For "openai": module-specific override takes precedence over shared OPENAI_*.
    For other slots: only the provider-level vars are used.
    """
    key_env, url_env = _SLOT_ENVS.get(slot, _SLOT_ENVS["openai"])

    if slot == "openai" and module_key_env:
        api_key = _getenv(module_key_env) or _getenv(key_env)
        base_url = _getenv(module_url_env) or _getenv(url_env)
    else:
        api_key = _getenv(key_env)
        base_url = _getenv(url_env)

    return Credentials(api_key, base_url)


def resolve_alert() -> Credentials:
    """RareAlert — self-hosted fine-tuned model."""
    _load_dotenv_if_present()
    return Credentials(
        api_key=_getenv("ALERT_API_KEY"),
        base_url=_getenv("ALERT_URL"),
    )


def resolve_diagnosis(*, model_name: str = "") -> Credentials:
    _load_dotenv_if_present()
    return _resolve_slot(
        _slot_for(model_name),
        module_key_env="Diagnosis_OPENAI_API_KEY",
        module_url_env="Diagnosis_OPENAI_BASE_URL",
    )


def resolve_prognosis(*, model_name: str = "") -> Credentials:
    _load_dotenv_if_present()
    return _resolve_slot(
        _slot_for(model_name),
        module_key_env="Prognosis_OPENAI_API_KEY",
        module_url_env="Prognosis_OPENAI_BASE_URL",
    )


def resolve_treatment(*, model_name: str = "") -> Credentials:
    _load_dotenv_if_present()
    return _resolve_slot(
        _slot_for(model_name),
        module_key_env="Treatment_OPENAI_API_KEY",
        module_url_env="Treatment_OPENAI_BASE_URL",
    )


def resolve_orpha_rag_embeddings(
    *,
    override_api_key: Optional[str] = None,
    override_base_url: Optional[str] = None,
    fallback: Optional[Credentials] = None,
) -> Credentials:
    """OrphaRAG embeddings endpoint."""
    _load_dotenv_if_present()
    api_key = str(override_api_key or "").strip() or _getenv("ORPHA_RAG_EMBEDDINGS_API_KEY")
    base_url = str(override_base_url or "").strip() or _getenv("ORPHA_RAG_EMBEDDINGS_BASE_URL")
    if (not api_key or not base_url) and fallback is not None:
        api_key = api_key or fallback.api_key
        base_url = base_url or fallback.base_url
    return Credentials(api_key=api_key, base_url=base_url)
