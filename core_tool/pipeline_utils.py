"""Shared helpers for runtime pipeline entry scripts."""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


def dump_feature_rows_csv(
    path: Path,
    rows: List[Dict[str, Any]],
    *,
    empty_fieldnames: List[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=empty_fieldnames)
            writer.writeheader()
        return

    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            if isinstance(row, dict):
                writer.writerow(row)


def cli_log(prefix: str, msg: str) -> None:
    print(f"[{prefix}] {msg}")


def cli_err(prefix: str, msg: str) -> None:
    print(f"[{prefix}][ERROR] {msg}", file=sys.stderr)


def emit_progress(progress_hook: Optional[Callable[[str], None]], msg: str) -> None:
    if callable(progress_hook):
        progress_hook(msg)
