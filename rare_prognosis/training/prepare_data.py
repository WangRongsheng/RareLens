#!/usr/bin/env python3
"""
Prepare data for the prognosis stacking pipeline.

Converts raw case outputs and LLM predictions into the directory structure
expected by build_features.py, train_models.py, and infer_models.py.

Inputs:
  - case_root/<case_id>/prognosis_new.json  (GT labels)
  - llm/<model>/<case_id>/prognosis_prediction_output.json

Outputs (into --out-dir):
  rareprognosis/{overall,functional,symptom}/S1_*.csv
  models/<model>/<case_id>/prognosis_prediction_output.json
  dataset/train_case_ids.json
  dataset/test_case_ids.json

Usage:
    python prepare_data.py \\
        --case-root data_500 \\
        --llm-root data_demo/pipeline_data/prognoisis/llm \\
        --out-dir data_demo/pipeline_data/prognoisis/prepared
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from data_io import TASK_CONFIGS


def _load_gt_from_prognosis_new(
    case_root: Path,
    case_ids: List[str],
) -> Dict[str, Dict[str, str]]:
    """Extract GT labels per task from <case_root>/<case_id>/prognosis_new.json.

    Structure:
      { "overall_outcome": "...",
        "quality_of_life": { "functional_status": "...", "symptom_burden": "..." }, ... }
    """
    gt: Dict[str, Dict[str, str]] = {t: {} for t in TASK_CONFIGS}
    for cid in case_ids:
        path = case_root / cid / "prognosis_new.json"
        if not path.is_file():
            continue
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        label = obj.get("overall_outcome")
        if isinstance(label, str) and label.strip():
            gt["overall_outcome"][cid] = label.strip()
        qol = obj.get("quality_of_life", {})
        if isinstance(qol, dict):
            for key in ("functional_status", "symptom_burden"):
                label = qol.get(key)
                if isinstance(label, str) and label.strip():
                    gt[key][cid] = label.strip()
    return gt


def _write_s1_csv(
    path: Path,
    case_ids: List[str],
    gt: Dict[str, str],
    split: str,
) -> None:
    """Write an S1-format CSV with case_id, split, prediction(empty), gt, correct, method."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case_id", "split", "prediction", "gt", "correct", "method"])
        for cid in case_ids:
            label = gt.get(cid, "")
            w.writerow([cid, split, "", label, "", ""])


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare data for prognosis stacking pipeline.")
    parser.add_argument("--case-root", required=True,
                        help="Root of case data (contains <case_id>/prognosis_new.json)")
    parser.add_argument("--llm-root", required=True,
                        help="Root of LLM outputs (contains <model>/<case_id>/prognosis_prediction_output.json)")
    parser.add_argument("--out-dir", required=True,
                        help="Output directory for prepared pipeline data")
    args = parser.parse_args()

    case_root = Path(args.case_root)
    llm_root = Path(args.llm_root)
    out_dir = Path(args.out_dir)

    # Discover case IDs from LLM output dirs
    model_dirs = sorted([p for p in llm_root.iterdir() if p.is_dir()])
    if not model_dirs:
        raise SystemExit(f"No model directories found under {llm_root}")

    case_ids_set: set = set()
    for md in model_dirs:
        for p in md.iterdir():
            if p.is_dir() and (p / "prognosis_prediction_output.json").is_file():
                case_ids_set.add(p.name)
    case_ids = sorted(case_ids_set)
    if not case_ids:
        raise SystemExit("No case IDs found")
    logger.info("Found %d cases: %s", len(case_ids), case_ids)
    logger.info("Found %d models: %s", len(model_dirs), [m.name for m in model_dirs])

    # 1. Extract GT labels from prognosis_new.json
    gt = _load_gt_from_prognosis_new(case_root, case_ids)

    for task in gt:
        logger.info("  [%s] GT labels: %d", task, len(gt[task]))

    # 2. Create S1 CSVs under rareprognosis/
    rare_root = out_dir / "rareprognosis"
    for task, cfg in TASK_CONFIGS.items():
        subdir, fname = cfg.s1_csv
        s1_path = rare_root / subdir / fname
        _write_s1_csv(s1_path, case_ids, gt[task], split="train")
        logger.info("  S1 CSV: %s", s1_path)

    # 3. Set up models dir: <out_dir>/models/<model>/<case_id>/prognosis_prediction_output.json
    models_out = out_dir / "models"
    for md in model_dirs:
        for cid in case_ids:
            src = md / cid / "prognosis_prediction_output.json"
            if not src.is_file():
                continue
            dst_dir = models_out / md.name / cid
            dst_dir.mkdir(parents=True, exist_ok=True)
            dst = dst_dir / "prognosis_prediction_output.json"
            if not dst.exists():
                shutil.copy2(src, dst)
    logger.info("Models output ready: %s", models_out)

    # 4. Create train/test case ID JSONs
    # For demo: all cases as both train and test (smoke test)
    dataset_dir = out_dir / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    with (dataset_dir / "train_case_ids.json").open("w", encoding="utf-8") as f:
        json.dump(case_ids, f, indent=2)
    with (dataset_dir / "test_case_ids.json").open("w", encoding="utf-8") as f:
        json.dump(case_ids, f, indent=2)
    logger.info("Dataset splits: %s (all %d cases as train+test)", dataset_dir, len(case_ids))
    logger.info("Finished. Prepared data saved to %s", out_dir)


if __name__ == "__main__":
    main()
