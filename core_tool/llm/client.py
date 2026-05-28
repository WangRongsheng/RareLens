"""
core/llm/client.py — Unified LLM call interface (shared across all modules)

Uses the official ``openai.OpenAI`` client; concurrent ``chat.completions.create`` calls
from multiple threads are supported (underlying HTTP client is thread-safe).

For now: OpenAI-compatible sync calls (non-stream and stream).
Future: async client variant if needed.
"""

import json
import logging
import os
import time
import threading
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)
_DOTENV_LOADED = False
_DOTENV_LOCK = threading.Lock()


def _load_dotenv_if_present(*, force: bool = False) -> None:
    """
    Best-effort .env loader with zero extra dependencies.
    - Searches current working directory first, then repository root.
    - Does not overwrite existing environment variables.
    - Use force=True in tests to re-read .env after changing the file or guard state.
    """
    global _DOTENV_LOADED
    # Allow disabling implicit .env loading in library code (recommended for services/tests).
    disable = str(os.getenv("CORE_TOOL_DOTENV_AUTOLOAD", "") or "").strip().lower() in {"0", "false", "no", "off"}
    if disable and not force:
        return

    with _DOTENV_LOCK:
        if _DOTENV_LOADED and not force:
            return
        if force:
            _DOTENV_LOADED = False

        candidates = [
            Path.cwd() / ".env",
            Path(__file__).resolve().parents[2] / ".env",
        ]
        dotenv_path = next((p for p in candidates if p.is_file()), None)
        if dotenv_path is None:
            _DOTENV_LOADED = True
            return

        try:
            for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
            logger.info("Loaded environment variables from %s", dotenv_path)
        except Exception as e:
            logger.warning("Failed to load .env from %s: %s", dotenv_path, e)
        finally:
            _DOTENV_LOADED = True


class LLMClient:
    """
    Minimal LLM client for reasoning synthesis.
    dry_run=True → returns concatenated text without calling any API.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        dry_run: bool = False,
        base_url: Optional[str] = None,
        debug: bool = False,
        max_retries: int = 10,
        retry_delay_sec: float = 2.0,
        http_timeout_sec: float = 180.0,
    ):
        _load_dotenv_if_present()
        self.api_key = (
            (api_key or "").strip()
            or os.getenv("Alert_API_KEY", "")
            or os.getenv("QWEN_API_KEY", "")
        )
        self.base_url = (
            (base_url or "").strip()
            or os.getenv("Alert_URL", "")
            or os.getenv("QWEN_BASE_URL", "")
        )
        self.model = model
        self.dry_run = dry_run
        self.debug = bool(debug)
        self.max_retries = max(1, int(max_retries))
        self.retry_delay_sec = float(retry_delay_sec)
        self.http_timeout_sec = float(http_timeout_sec)
        self._client = None
        self._client_lock = threading.Lock()
        self.call_count = 0
        self.total_tokens = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def _debug_print(self, msg: str) -> None:
        if self.debug:
            # Use print for "obvious" CLI visibility (logging config may be absent).
            print(msg)

    @staticmethod
    def _debug_allow_content() -> bool:
        """
        Whether debug output may include prompt/response raw text.
        Default: False (avoid leaking PHI/PII into logs/terminals).
        """
        v = str(os.getenv("LLM_DEBUG_CONTENT", "") or "").strip().lower()
        return v in {"1", "true", "yes", "y", "on"}

    @staticmethod
    def _text_fingerprint(text: str) -> str:
        s = (text or "").encode("utf-8", errors="ignore")
        return hashlib.sha256(s).hexdigest()[:12]

    def _get_client(self):
        if self._client is None:
            with self._client_lock:
                if self._client is None:
                    try:
                        from openai import OpenAI

                        kwargs: Dict[str, Any] = {
                            "api_key": self.api_key,
                            "timeout": self.http_timeout_sec,
                        }
                        if self.base_url:
                            kwargs["base_url"] = self.base_url
                        self._client = OpenAI(**kwargs)
                    except ImportError:
                        logger.warning("openai not installed, switching to dry_run")
                        self.dry_run = True
        return self._client

    @staticmethod
    def _messages_payload(prompt: str, system: Optional[str]) -> list:
        if system and str(system).strip():
            return [
                {"role": "system", "content": str(system).strip()},
                {"role": "user", "content": prompt},
            ]
        return [{"role": "user", "content": prompt}]

    def call(
        self,
        prompt: str,
        max_tokens: int = 800,
        temperature: Optional[float] = 0.0,
        extra_body: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None,
        response_format: Optional[Dict[str, Any]] = None,
        *,
        system: Optional[str] = None,
        stream: bool = False,
    ) -> str:
        """
        Send a prompt, return raw text response. Returns '' on failure.

        There is **no** instance-level default for ``stream`` beyond this parameter:
        each call site decides (``stream=False`` unless you pass ``True``). Stages that
        read YAML (e.g. rare_alert ``RiskStageConfig.stream``) pass it through explicitly.
        """
        if self.dry_run:
            return ""

        client = self._get_client()
        if client is None:
            return ""

        for attempt in range(1, self.max_retries + 1):
            start = time.monotonic()
            try:
                model_name = model or self.model
                if self.debug:
                    p = (prompt or "").replace("\r", "")
                    fp = self._text_fingerprint(p)
                    allow = self._debug_allow_content()
                    preview = ""
                    if allow:
                        preview = (p[:600] + ("... [truncated]" if len(p) > 600 else "")).replace("\n", "\\n")
                    self._debug_print(
                        f"[LLMClient] call attempt={attempt}/{self.max_retries} model={model_name!r} "
                        f"base_url={self.base_url!r} max_tokens={int(max_tokens)} temp={float(temperature)} "
                        f"stream={stream} prompt_len={len(p)} prompt_sha256_12={fp}"
                        + (f" preview={preview}" if preview else "")
                    )
                req: Dict[str, Any] = {
                    "model": model_name,
                    "messages": self._messages_payload(prompt, system),
                    "max_tokens": max_tokens,
                    "stream": stream,
                }
                if temperature is not None:
                    req["temperature"] = temperature
                if extra_body:
                    req["extra_body"] = extra_body
                if response_format:
                    req["response_format"] = response_format

                resp = client.chat.completions.create(**req)

                if stream:
                    full_content = ""
                    for chunk in resp:
                        if not chunk.choices:
                            continue
                        delta = chunk.choices[0].delta
                        if delta is not None and getattr(delta, "content", None):
                            full_content += delta.content or ""
                        if chunk.usage:
                            self.total_tokens += chunk.usage.total_tokens
                            self.prompt_tokens += getattr(chunk.usage, "prompt_tokens", 0) or 0
                            self.completion_tokens += getattr(chunk.usage, "completion_tokens", 0) or 0
                    text = full_content.strip()
                else:
                    if resp.usage:
                        self.total_tokens += resp.usage.total_tokens
                        self.prompt_tokens += getattr(resp.usage, "prompt_tokens", 0) or 0
                        self.completion_tokens += getattr(resp.usage, "completion_tokens", 0) or 0
                    text = (resp.choices[0].message.content or "").strip()

                self.call_count += 1

                latency = (time.monotonic() - start) * 1000
                logger.debug(f"LLM call: {model_name}, {latency:.0f}ms")
                if self.debug:
                    allow = self._debug_allow_content()
                    fp = self._text_fingerprint(text)
                    preview = self._response_preview(text, limit=600).replace("\n", "\\n") if allow else ""
                    self._debug_print(
                        f"[LLMClient] response model={model_name!r} latency_ms={latency:.0f} "
                        f"text_len={len(text)} text_sha256_12={fp}"
                        + (f" preview={preview}" if preview else "")
                    )
                return text
            except Exception as e:
                err_str = str(e)
                if len(err_str) > 300:
                    err_str = err_str[:300] + f"... [truncated, total {len(str(e))} chars]"
                logger.warning(
                    "LLM call failed (attempt %s/%s): %s",
                    attempt,
                    self.max_retries,
                    err_str,
                )
                if self.debug:
                    self._debug_print(f"[LLMClient] ERROR attempt={attempt}/{self.max_retries}: {err_str}")
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay_sec)
        return ""

    def embeddings_create(
        self,
        *,
        model: str,
        input: List[str],
    ) -> Optional[List[List[float]]]:
        """
        Call OpenAI-compatible /v1/embeddings.

        Behaviors aligned with ``call()``:
        - honors dry_run (returns None)
        - uses same underlying OpenAI client + base_url/api_key
        - retries with max_retries/retry_delay_sec
        - prints debug preview when debug=True
        """
        if self.dry_run:
            return None
        client = self._get_client()
        if client is None:
            return None
        text_list = list(input or [])
        if not text_list:
            return []

        for attempt in range(1, self.max_retries + 1):
            start = time.monotonic()
            try:
                if self.debug:
                    first = str(text_list[0] or "")
                    allow = self._debug_allow_content()
                    fp = self._text_fingerprint(first)
                    preview = first.replace("\n", "\\n")[:180] if allow else ""
                    self._debug_print(
                        f"[LLMClient] embeddings attempt={attempt}/{self.max_retries} model={model!r} "
                        f"base_url={self.base_url!r} n={len(text_list)} first={preview[:180]}"
                        + ("" if allow else f" first_sha256_12={fp}")
                    )
                resp = client.embeddings.create(model=str(model or "").strip(), input=text_list)
                out: List[List[float]] = []
                for item in (resp.data or []):
                    out.append(list(item.embedding))
                latency = (time.monotonic() - start) * 1000
                logger.debug("LLM embeddings: %s, %0.fms", model, latency)
                return out
            except Exception as e:
                err_str = str(e)
                if len(err_str) > 300:
                    err_str = err_str[:300] + f"... [truncated, total {len(str(e))} chars]"
                logger.warning(
                    "LLM embeddings failed (attempt %s/%s): %s",
                    attempt,
                    self.max_retries,
                    err_str,
                )
                if self.debug:
                    self._debug_print(f"[LLMClient] EMBEDDINGS ERROR attempt={attempt}/{self.max_retries}: {err_str}")
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay_sec)
        return None

    def _response_preview(self, raw: str, limit: int = 480) -> str:
        s = (raw or "").replace("\n", " ").strip()
        if len(s) <= limit:
            return s
        return s[: limit - 3] + "..."

    def _format_extracted_json(self, parsed: Dict[str, Any], key: Optional[str], preview: str) -> str:
        """Return string value for key, or full JSON string when key is None."""
        if not key:
            return json.dumps(parsed, ensure_ascii=False)
        val = parsed.get(key, "")
        if val is None or (isinstance(val, str) and not val.strip()):
            logger.warning(
                "call_and_parse_json: JSON parsed but field %r is missing or empty; actual keys=%s; response_preview=%r",
                key,
                list(parsed.keys()),
                preview,
            )
            return ""
        if isinstance(val, str):
            return val
        return json.dumps(val, ensure_ascii=False)

    def call_and_parse_json(
        self,
        prompt: str,
        key: Optional[str] = None,
        **call_kwargs: Any,
    ) -> str:
        """
        Call LLM, then parse JSON with ``json_extractor.extract`` only (single strategy chain).
        Extra keyword arguments are forwarded to ``call()`` (e.g. system=..., stream=...).
        """
        raw = self.call(prompt, **call_kwargs)
        if not raw:
            logger.warning(
                "call_and_parse_json: LLM returned empty string (model=%r, dry_run=%s, api_key_set=%s). "
                "Check QWEN_API_KEY / Alert_API_KEY, QWEN_BASE_URL / Alert_URL, quota, and any 'LLM call failed' warnings above.",
                self.model,
                self.dry_run,
                bool(self.api_key),
            )
            return ""
        preview = self._response_preview(raw)
        try:
            from core_tool.parser.json_extractor import extract

            parsed = extract(raw)
        except Exception as e:
            logger.warning("call_and_parse_json: extract raised an exception: %s", e)
            parsed = None

        if isinstance(parsed, dict):
            return self._format_extracted_json(parsed, key, preview)

        logger.warning(
            "call_and_parse_json: could not extract a JSON object from the response; preview=%r",
            preview,
        )
        if key:
            return ""
        return raw

    def get_stats(self) -> dict:
        return {
            "calls": self.call_count,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }
