"""Shared key normalization for ``schema.RiskInput``-shaped dicts."""

from __future__ import annotations

from typing import Any, Dict


def normalize_risk_input_dict(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Map legacy / alternate keys (e.g. ``basic_info``) to canonical RiskInput fields.
    Callers should pass a shallow copy or expect this to return a new dict based on ``data``.
    """
    payload = dict(data)
    if "basic_information" not in payload and "basic_info" in payload:
        payload["basic_information"] = payload.get("basic_info")
    if "physical_examination" not in payload and "physical_exam" in payload:
        payload["physical_examination"] = payload.get("physical_exam")
    return payload
