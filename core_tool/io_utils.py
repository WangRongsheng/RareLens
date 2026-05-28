"""Shared JSON/config IO helpers for runtime pipelines."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def repo_root() -> Path:
    """Repository root directory (contains ``data/``, ``configs/``, etc.)."""
    return Path(__file__).resolve().parents[1]


def default_case_test_input_dir() -> Path:
    """Default batch risk input root directory: ``data/case_test_input``."""
    return repo_root() / "data" / "case_test_input"


def default_case_test_output_root() -> Path:
    """Default risk output root directory: ``data/case_test_output``."""
    return repo_root() / "data" / "case_test_output"


def read_json(path: str | Path) -> Dict[str, Any]:
    """Load a JSON file whose root value must be an object (mapping)."""
    p = Path(path)
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"Failed to load JSON file: {p}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object at root in file: {p}")
    return data


def dump_json(path: str | Path, payload: Dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def read_config(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".json":
        return read_json(p)
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("Reading YAML config requires pyyaml (pip install pyyaml).") from exc
        data = yaml.safe_load(p.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"Expected mapping at root in YAML config: {p}")
        return data

    try:
        return read_json(p)
    except Exception:
        try:
            import yaml  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("Unknown config suffix and YAML parsing unavailable (pyyaml not installed).") from exc
        data = yaml.safe_load(p.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"Expected mapping at root in YAML config: {p}")
        return data
