#!/usr/bin/env python3
"""
Prepare data for the prognosis stacking pipeline.

Converts raw case outputs and LLM predictions into the directory structure
expected by build_features.py, train_models.py, and infer_models.py.

Inputs:
  - case_output/<case_id>/6_prognosis/RarePrognosis_output.json  (GT labels)
  - llm/<model>/<case_id>/prognosis_prediction_output.json
  - RarePrognois/{overall,funcational,symptom}/*.csv  (optional fallback for GT)

Outputs (into --out-dir):
  rareprognosis/{overall,funcational,symptom}/S1_*.csv
  models/<model>/<case_id>/prognosis_prediction_output.json
  dataset/train_case_ids.json
  dataset/test_case_ids.json

Usage:
    python prepare_data.py \\
        --case-root data_demo/case_output \\
        --llm-root data_demo/pipeline_data/prognoisis/llm \\
        --rareprognois-root data_demo/pipeline_data/prognoisis/RarePrognois \\
        --out-dir data_demo/pipeline_data/prognoisis/prepared
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from data_io import TASK_CONFIGS


def _load_gt_from_case_output(
    case_root: Path,
    case_ids: List[str],
) -> Dict[str, Dict[str, str]]:
    """Extract GT labels per task from RarePrognosis_output.json."""
    gt: Dict[str, Dict[str, str]] = {t: {} for t in TASK_CONFIGS}
    for cid in case_ids:
        path = case_root / cid / "6_prognosis" / "RarePrognosis_output.json"
        if not path.is_file():
            continue
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        rp = obj.get("RarePrognosis", {})
        for task in TASK_CONFIGS:
            section = rp.get(task, {})
            label = section.get("gt")
            if isinstance(label, str) and label.strip():
                gt[task][cid] = label.strip()
    return gt


def _load_gt_from_s2_csv(
    rareprognois_root: Path,
) -> Dict[str, Dict[str, str]]:
    """Fallback: extract GT from existing S2 CSVs."""
    gt: Dict[str, Dict[str, str]] = {t: {} for t in TASK_CONFIGS}
    csv_map = {
        "overall_outcome":   "overall/S2_meta_weighted_acc.csv",
        "functional_status": "funcational/S2_meta_weighted_acc.csv",
        "symptom_burden":    "symptom/S2_meta_stacking_gbdt.csv",
    }
    for task, rel_path in csv_map.items():
        path = rareprognois_root / rel_path
        if not path.is_file():
            continue
        with path.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                cid = str(row.get("case_id", "")).strip()
                label = str(row.get("gt", "")).strip()
                if cid and label:
                    gt[task][cid] = label
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
                        help="Root of case_output/ (contains <case_id>/6_prognosis/RarePrognosis_output.json)")
    parser.add_argument("--llm-root", required=True,
                        help="Root of LLM outputs (contains <model>/<case_id>/prognosis_prediction_output.json)")
    parser.add_argument("--rareprognois-root", default="",
                        help="Root of RarePrognois/ with S2 CSVs (optional fallback for GT)")
    parser.add_argument("--out-dir", required=True,
                        help="Output directory for prepared pipeline data")
    args = parser.parse_args()

    case_root = Path(args.case_root)
    llm_root = Path(args.llm_root)
    out_dir = Path(args.out_dir)
    rareprognois_root = Path(args.rareprognois_root) if args.rareprognois_root else None

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
    print(f"Found {len(case_ids)} cases: {case_ids}")
    print(f"Found {len(model_dirs)} models: {[m.name for m in model_dirs]}")

    # 1. Extract GT labels
    gt = _load_gt_from_case_output(case_root, case_ids)
    # Fallback to S2 CSVs if case_output GT is incomplete
    if rareprognois_root and rareprognois_root.is_dir():
        gt_s2 = _load_gt_from_s2_csv(rareprognois_root)
        for task in gt:
            for cid, label in gt_s2[task].items():
                if cid not in gt[task]:
                    gt[task][cid] = label

    for task in gt:
        print(f"  [{task}] GT labels: {len(gt[task])}")

    # 2. Create S1 CSVs under rareprognosis/
    rare_root = out_dir / "rareprognosis"
    for task, cfg in TASK_CONFIGS.items():
        subdir, fname = cfg.s1_csv
        s1_path = rare_root / subdir / fname
        _write_s1_csv(s1_path, case_ids, gt[task], split="train")
        print(f"  S1 CSV: {s1_path}")

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
    print(f"Models output ready: {models_out}")

    # 4. Create train/test case ID JSONs
    # For demo: all cases as both train and test (smoke test)
    dataset_dir = out_dir / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    with (dataset_dir / "train_case_ids.json").open("w", encoding="utf-8") as f:
        json.dump(case_ids, f, indent=2)
    with (dataset_dir / "test_case_ids.json").open("w", encoding="utf-8") as f:
        json.dump(case_ids, f, indent=2)
    print(f"Dataset splits: {dataset_dir} (all {len(case_ids)} cases as train+test)")

    # Summary
    print(f"\n{'='*60}")
    print("Prepared data summary:")
    print(f"  rareprognosis root: {rare_root}")
    print(f"  models root:        {models_out}")
    print(f"  dataset:            {dataset_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
