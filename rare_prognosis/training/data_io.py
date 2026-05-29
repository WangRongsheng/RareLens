"""
Shared I/O utilities and task configuration for prognosis training pipeline.

All task definitions, label normalization, JSON loading, and common helpers
are centralized here to avoid duplication across scripts.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Task configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TaskConfig:
    """Unified task configuration for all prognosis sub-tasks."""
    name: str
    # GT extraction
    gt_section: Optional[str]       # None => root-level key; "quality_of_life" => nested
    gt_key: str                     # key inside the section (or root)
    # Prediction extraction
    pred_section: str               # e.g. "overall_outcome"
    pred_key: str                   # e.g. "outcome_category"
    expl_key: str = "explanation"   # key for explanation text
    # Label synonyms  {raw_lower: canonical}
    synonyms: Dict[str, str] = field(default_factory=dict)
    # S1 CSV location  (subdir, filename)
    s1_csv: Tuple[str, str] = ("", "")
    # Model bundle filename
    bundle: str = ""

    @property
    def pred_path(self) -> Tuple[str, str]:
        return (self.pred_section, self.pred_key)

    @property
    def expl_path(self) -> Tuple[str, str]:
        return (self.pred_section, self.expl_key)


TASK_CONFIGS: Dict[str, TaskConfig] = {
    "overall_outcome": TaskConfig(
        name="overall_outcome",
        gt_section=None,
        gt_key="overall_outcome",
        pred_section="overall_outcome",
        pred_key="outcome_category",
        synonyms={"death": "terminal", "progressed": "progression"},
        s1_csv=("overall", "S1_stacking_gbdt.csv"),
        bundle="overall_outcome_C2_stacking_gbdt.pkl",
    ),
    "functional_status": TaskConfig(
        name="functional_status",
        gt_section="quality_of_life",
        gt_key="functional_status",
        pred_section="functional_status",
        pred_key="status",
        synonyms={},
        s1_csv=("funcational", "S1_stacking_gbdt.csv"),
        bundle="functional_status_C2_stacking_gbdt.pkl",
    ),
    "symptom_burden": TaskConfig(
        name="symptom_burden",
        gt_section="quality_of_life",
        gt_key="symptom_burden",
        pred_section="symptom_burden",
        pred_key="burden",
        synonyms={},
        s1_csv=("symptom", "S1_catboost.csv"),
        bundle="symptom_burden_C2_stacking_gbdt.pkl",
    ),
}

ENSEMBLE_SUBDIR: Dict[str, str] = {
    t: f"ensemble_output_{t}" for t in TASK_CONFIGS
}

DEFAULT_EXPL_KEYWORDS: List[str] = [
    "progression", "progress", "stable", "stabil", "terminal",
    "death", "metast", "response", "improve", "worsen",
]


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_json(path: Path) -> Any:
    """Load a JSON file, returning None on error."""
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def load_case_ids(path: Path) -> List[str]:
    """Load a JSON list of case IDs."""
    data = load_json(path)
    if data is None or not isinstance(data, list):
        raise SystemExit(f"case_ids file must be a JSON list: {path}")
    return [str(x) for x in data]


def get_nested(obj: Any, keys: Tuple[str, ...]) -> Any:
    """Traverse nested dicts by a sequence of keys."""
    cur = obj
    for k in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    return cur


# ---------------------------------------------------------------------------
# Label normalization
# ---------------------------------------------------------------------------

def normalize_label(label: Any, task: str) -> Optional[str]:
    """Normalize a label string using task-specific synonyms."""
    if not isinstance(label, str):
        return None
    key = label.strip().lower()
    if not key:
        return None
    return TASK_CONFIGS[task].synonyms.get(key, key)


# ---------------------------------------------------------------------------
# GT / prediction extraction
# ---------------------------------------------------------------------------

def get_gt_label(task: str, obj: dict) -> Optional[str]:
    """Extract ground-truth label from a prognosis_new.json object."""
    cfg = TASK_CONFIGS[task]
    section = obj if cfg.gt_section is None else obj.get(cfg.gt_section, {})
    if not isinstance(section, dict):
        return None
    return normalize_label(section.get(cfg.gt_key), task)


def get_pred_label(task: str, obj: dict) -> Optional[str]:
    """Extract prediction label from a prognosis_prediction_output.json object."""
    cfg = TASK_CONFIGS[task]
    section = obj.get(cfg.pred_section, {})
    if not isinstance(section, dict):
        return None
    return normalize_label(section.get(cfg.pred_key), task)


# ---------------------------------------------------------------------------
# Directory helpers
# ---------------------------------------------------------------------------

def list_model_dirs(
    models_root: Path,
    sample_case_id: str,
    ignore: Optional[set] = None,
) -> List[Path]:
    """Discover model directories that contain output for a sample case."""
    if ignore is None:
        ignore = {"RarePrognosis", "Output_prog", "Output_prog.zip"}
    out: List[Path] = []
    for p in sorted(models_root.iterdir(), key=lambda x: x.name):
        if not p.is_dir() or p.name in ignore:
            continue
        if (p / sample_case_id / "prognosis_prediction_output.json").is_file():
            out.append(p)
    return out
