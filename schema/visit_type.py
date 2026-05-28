"""Shared visit_type utilities to avoid stringly-typed branching bugs."""

from __future__ import annotations

from typing import Literal

VisitType = Literal["primary", "followup"]


def normalize_visit_type(value: str) -> VisitType:
    v = str(value or "").strip().lower()
    if v in {"primary", "first", "first_visit", "firstvisit"}:
        return "primary"
    if v in {"followup", "follow_up", "follow-up", "followup_visit"}:
        return "followup"
    raise ValueError(f"Unsupported visit_type: {value!r} (expected 'primary' or 'followup').")

