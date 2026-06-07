#!/usr/bin/env python3
"""
Evaluate ML ensemble (stacking GBDT) prognosis prediction performance.

Reads predictions from S1 CSVs (written by infer_models.py) and computes
core + secondary metrics per task and all3.

Core:     accuracy, MCC, macro F1, balanced accuracy, per-class recall
Secondary: severe recall, false optimism rate, MAOE, OBI

Usage:
    python eval_ml.py \
        --rareprognosis-root prepared/rareprognosis \
        --split test

    # Or provide explicit prediction CSVs:
    python eval_ml.py \
        --input overall_outcome=path/to/S1.csv \
        --input functional_status=path/to/S1.csv \
        --input symptom_burden=path/to/S1.csv \
        --split test
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

THIS_DIR = Path(__file__).resolve().parent
TRAINING_DIR = THIS_DIR.parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

from data_io import TASK_CONFIGS, load_s1_csv, write_eval_csv
from metrics import (
    TASKS, ORDINAL_LABELS,
    evaluate_task, compute_all3_accuracy,
)


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

METRIC_KEYS = [
    "accuracy", "mcc", "macro_f1", "balanced_accuracy",
    "severe_recall", "false_optimism", "maoe", "obi",
]


def _fmt(v, width=10):
    if v is None:
        return "N/A".rjust(width)
    return f"{v:.4f}".rjust(width)


def _print_task_metrics(task: str, m: dict):
    print(f"\n  [{task}]  n={m.get('n_scored', 0)}")
    for k in METRIC_KEYS:
        print(f"    {k:<20s} {_fmt(m.get(k))}")
    recall = m.get("per_class_recall", {})
    if recall:
        print(f"    {'--- per-class recall ---'}")
        for label in ORDINAL_LABELS.get(task, []):
            r = recall.get(label)
            print(f"    {label:<24s} {_fmt(r)}")




# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate ML ensemble on prognosis tasks.")
    p.add_argument("--rareprognosis-root", default="", help="Root containing S1 CSVs with predictions")
    p.add_argument("--input", action="append", default=[], help="task=csv_path (explicit input per task)")
    p.add_argument("--split", choices=("train", "test", "all"), default="test")
    p.add_argument("--out-dir", default="", help="Output directory for CSV results (optional)")
    args = p.parse_args()

    # Resolve input CSVs
    task_csv: Dict[str, Path] = {}

    # From --input args: task=path
    for spec in args.input:
        if "=" not in spec:
            raise SystemExit(f"Invalid --input format: {spec}  (expected task=path)")
        task_name, csv_path = spec.split("=", 1)
        task_name = task_name.strip()
        if task_name not in TASK_CONFIGS:
            raise SystemExit(f"Unknown task: {task_name}")
        task_csv[task_name] = Path(csv_path.strip())

    # From --rareprognosis-root (auto-discover S1 CSVs)
    if args.rareprognosis_root:
        rare_root = Path(args.rareprognosis_root)
        for task, cfg in TASK_CONFIGS.items():
            if task not in task_csv:
                s1_path = rare_root / cfg.s1_csv[0] / cfg.s1_csv[1]
                if s1_path.is_file():
                    task_csv[task] = s1_path

    if not task_csv:
        raise SystemExit("No input CSVs found. Use --rareprognosis-root or --input.")

    print(f"Evaluating ML ensemble (split={args.split})")

    gt_by_task: Dict[str, Dict[str, str]] = {}
    preds_by_task: Dict[str, Dict[str, str]] = {}
    all_case_ids: Dict[str, List[str]] = {}
    csv_rows: List[dict] = []

    for task in TASKS:
        if task not in task_csv:
            continue
        path = task_csv[task]
        if not path.is_file():
            print(f"[{task}] CSV not found: {path}, skipping")
            continue

        s1 = load_s1_csv(path, task)
        if args.split == "all":
            case_ids = sorted(set(s1.train_ids + s1.test_ids))
        elif args.split == "train":
            case_ids = s1.train_ids
        else:
            case_ids = s1.test_ids
        gt = s1.gt_by_id
        preds = s1.pred_by_id
        if not case_ids:
            print(f"[{task}] no cases for split={args.split}")
            continue

        gt_by_task[task] = gt
        preds_by_task[task] = preds
        all_case_ids[task] = case_ids

        m = evaluate_task(task, case_ids, gt, preds)
        _print_task_metrics(task, m)

        row = {"task": task, "split": args.split, "n": m["n_scored"]}
        for k in METRIC_KEYS:
            row[k] = m.get(k)
        for label, rec in m.get("per_class_recall", {}).items():
            row[f"recall_{label}"] = rec
        csv_rows.append(row)

    # All3 accuracy
    if len(gt_by_task) == 3:
        common_ids = None
        for task in TASKS:
            ids = set(all_case_ids.get(task, []))
            common_ids = ids if common_ids is None else common_ids & ids
        common_list = sorted(common_ids) if common_ids else []
        a3 = compute_all3_accuracy(common_list, preds_by_task, gt_by_task)
        print(f"\n  [all3]  all-three-correct accuracy: {_fmt(a3)}  n={len(common_list)}")

    # Write CSV
    if args.out_dir and csv_rows:
        write_eval_csv(Path(args.out_dir) / f"eval_ml_{args.split}.csv", csv_rows)


if __name__ == "__main__":
    main()
