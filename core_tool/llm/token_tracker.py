"""Utilities for collecting and aggregating LLM token usage statistics."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def stats_from_client(client: Any, *, model_name: str, latency: float) -> Dict[str, Any]:
    """Build a token stats dict from an LLMClient instance after a call."""
    return {
        **client.get_stats(),
        "model": model_name,
        "latency_seconds": round(latency, 3),
    }


def build_model_breakdown(model_token_stats: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert per-model token stats dict to a list suitable for telemetry output."""
    return [
        {
            "model": s.get("model", m_name),
            "tokens": s.get("total_tokens", 0),
            "prompt_tokens": s.get("prompt_tokens", 0),
            "completion_tokens": s.get("completion_tokens", 0),
            "latency_seconds": s.get("latency_seconds", 0),
            "status": "success",
        }
        for m_name, s in model_token_stats.items()
    ]


def merge_model_breakdown(
    model_token_stats: Dict[str, Dict[str, Any]],
    existing_telemetry_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Build model breakdown list, merging with a previous telemetry file if provided.

    Ensures partial re-runs don't erase prior model records already written to disk.
    """
    new_breakdown: Dict[str, Dict[str, Any]] = {
        s.get("model", m_name): {
            "model": s.get("model", m_name),
            "tokens": s.get("total_tokens", 0),
            "prompt_tokens": s.get("prompt_tokens", 0),
            "completion_tokens": s.get("completion_tokens", 0),
            "latency_seconds": s.get("latency_seconds", 0),
            "status": "success",
        }
        for m_name, s in model_token_stats.items()
    }
    if existing_telemetry_path is not None and Path(existing_telemetry_path).is_file():
        try:
            prev = json.loads(Path(existing_telemetry_path).read_text(encoding="utf-8"))
            for entry in prev.get("model_breakdown", []):
                key = entry.get("model") or ""
                if key and key not in new_breakdown:
                    new_breakdown[key] = entry
        except Exception:
            pass
    return list(new_breakdown.values())


def aggregate_token_stats(entries: List[Dict[str, Any]]) -> Tuple[int, int, int]:
    """Return (prompt_tokens, completion_tokens, total_tokens) summed across all entries."""
    prompt = sum(int(e.get("prompt_tokens", 0) or 0) for e in entries)
    completion = sum(int(e.get("completion_tokens", 0) or 0) for e in entries)
    total = sum(int(e.get("total_tokens", 0) or 0) for e in entries)
    return prompt, completion, total
