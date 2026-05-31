#!/usr/bin/env python3
"""
Evaluate individual LLM models' prognosis prediction performance.

Reads per-model prognosis_prediction_output.json and ground truth from S1 CSVs.
Computes core + secondary metrics for each LLM model, per task and all3.

Core:     accuracy, MCC, macro F1, balanced accuracy, per-class recall
Secondary: severe recall, false optimism rate, MAOE, OBI

Usage:
    python eval_llm.py \
        --models-root prepared/models \
        --rareprognosis-root prepared/rareprognosis \
        --train-ids prepared/dataset/train_case_ids.json \
        --test-ids prepared/dataset/test_case_ids.json \
        --split test
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

THIS_DIR = Path(__file__).resolve().parent
TRAINING_DIR = THIS_DIR.parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

from data_io import TASK_CONFIGS, load_json, normalize_label, list_model_dirs
from metrics import (
    TASKS, ORDINAL_LABELS,
    evaluate_task, compute_all3_accuracy,
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_gt_from_s1(path: Path, task: str):
    """Load GT and split info from S1 CSV."""
    train_ids, test_ids = [], []
    gt: Dict[str, str] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            cid = str(row.get("case_id", "")).strip()
            split = str(row.get("split", "")).strip().lower()
            if not cid:
                continue
            label = normalize_label(row.get("gt"), task)
            if label is not None:
                gt[cid] = label
            if split == "train":
                train_ids.append(cid)
            elif split == "test":
                test_ids.append(cid)
    return sorted(set(train_ids)), sorted(set(test_ids)), gt


def _load_llm_predictions(
    models_root: Path,
    model_name: str,
    case_ids: List[str],
    task: str,
) -> Dict[str, str]:
    """Load predictions for one LLM model across all cases."""
    cfg = TASK_CONFIGS[task]
    preds: Dict[str, str] = {}
    for cid in case_ids:
        obj = load_json(models_root / model_name / cid / "prognosis_prediction_output.json")
        if not isinstance(obj, dict):
            continue
        section = obj.get(cfg.pred_section, {})
        if not isinstance(section, dict):
            continue
        label = normalize_label(section.get(cfg.pred_key), task)
        if label is not None:
            preds[cid] = label
    return preds


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

METRIC_KEYS = [
    "accuracy", "mcc", "macro_f1", "balanced_accuracy",
    "severe_recall", "false_optimism", "maoe", "obi",
]


def _fmt(v, width=8):
    if v is None:
        return "N/A".rjust(width)
    return f"{v:.4f}".rjust(width)


def _print_header():
    cols = ["model".ljust(40)] + [k.rjust(12) for k in METRIC_KEYS] + ["n".rjust(6)]
    print("  " + " ".join(cols))
    print("  " + "-" * len(" ".join(cols)))


def _print_row(name: str, m: dict):
    cols = [name.ljust(40)]
    for k in METRIC_KEYS:
        cols.append(_fmt(m.get(k), 12))
    cols.append(str(m.get("n_scored", "")).rjust(6))
    print("  " + " ".join(cols))


def _write_csv(path: Path, rows: List[dict]):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    seen = set()
    fieldnames = []
    for row in rows:
        for k in row:
            if k not in seen:
                seen.add(k)
                fieldnames.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate LLM models on prognosis tasks.")
    p.add_argument("--models-root", required=True, help="Root containing <model>/<case_id>/prognosis_prediction_output.json")
    p.add_argument("--rareprognosis-root", required=True, help="Root containing S1 CSVs with GT labels")
    p.add_argument("--train-ids", default="", help="JSON list of train case IDs (optional)")
    p.add_argument("--test-ids", default="", help="JSON list of test case IDs (optional)")
    p.add_argument("--split", choices=("train", "test", "all"), default="test", help="Which split to evaluate")
    p.add_argument("--out-dir", default="", help="Output directory for CSV results (optional)")
    args = p.parse_args()

    models_root = Path(args.models_root)
    rare_root = Path(args.rareprognosis_root)

    # Discover models
    sample_case = None
    for md in sorted(models_root.iterdir()):
        if not md.is_dir():
            continue
        for sub in md.iterdir():
            if sub.is_dir() and (sub / "prognosis_prediction_output.json").is_file():
                sample_case = sub.name
                break
        if sample_case:
            break
    if not sample_case:
        raise SystemExit(f"No model outputs found under {models_root}")

    model_dirs = list_model_dirs(models_root, sample_case)
    model_names = [d.name for d in model_dirs]
    print(f"Found {len(model_names)} models")

    # Optional ID filters
    train_allow = set()
    test_allow = set()
    if args.train_ids and Path(args.train_ids).is_file():
        train_allow = set(str(x) for x in (load_json(Path(args.train_ids)) or []))
    if args.test_ids and Path(args.test_ids).is_file():
        test_allow = set(str(x) for x in (load_json(Path(args.test_ids)) or []))

    all_csv_rows: Dict[str, List[dict]] = {}

    for task in TASKS:
        cfg = TASK_CONFIGS[task]
        s1_path = rare_root / cfg.s1_csv[0] / cfg.s1_csv[1]
        if not s1_path.is_file():
            print(f"[{task}] S1 CSV not found: {s1_path}, skipping")
            continue

        s1_train, s1_test, gt = _load_gt_from_s1(s1_path, task)

        if args.split == "train":
            eval_ids = s1_train
        elif args.split == "test":
            eval_ids = s1_test if s1_test else s1_train
        else:
            eval_ids = sorted(set(s1_train + s1_test))

        if train_allow or test_allow:
            allowed = train_allow | test_allow
            eval_ids = [c for c in eval_ids if c in allowed]

        if not eval_ids:
            print(f"[{task}] no cases for split={args.split}, skipping")
            continue

        print(f"\n[{task}] evaluating {len(eval_ids)} cases (split={args.split})")
        _print_header()

        csv_rows = []
        for mn in model_names:
            preds = _load_llm_predictions(models_root, mn, eval_ids, task)
            m = evaluate_task(task, eval_ids, gt, preds)
            _print_row(mn, m)
            row = {"model": mn, "task": task, "split": args.split, "n": m["n_scored"]}
            for k in METRIC_KEYS:
                row[k] = m.get(k)
            for label, rec in m.get("per_class_recall", {}).items():
                row[f"recall_{label}"] = rec
            csv_rows.append(row)

        all_csv_rows[task] = csv_rows

    # All3 accuracy
    if len(all_csv_rows) == 3:
        print(f"\n[all3] all-three-correct accuracy (split={args.split})")
        # Need to get common case_ids and per-task preds/gt
        # Reload gt and preds for all3
        common_ids = None
        gt_by_task: Dict[str, Dict[str, str]] = {}
        for task in TASKS:
            cfg = TASK_CONFIGS[task]
            s1_path = rare_root / cfg.s1_csv[0] / cfg.s1_csv[1]
            _, _, gt = _load_gt_from_s1(s1_path, task)
            gt_by_task[task] = gt
            task_ids = set(gt.keys())
            common_ids = task_ids if common_ids is None else common_ids & task_ids

        if train_allow or test_allow:
            common_ids = common_ids & (train_allow | test_allow)
        common_list = sorted(common_ids) if common_ids else []

        print(f"  {'model'.ljust(40)} {'all3_acc'.rjust(10)} {'n'.rjust(6)}")
        print(f"  {'-' * 58}")
        for mn in model_names:
            preds_by_task: Dict[str, Dict[str, str]] = {}
            for task in TASKS:
                preds_by_task[task] = _load_llm_predictions(models_root, mn, common_list, task)
            a3 = compute_all3_accuracy(common_list, preds_by_task, gt_by_task)
            print(f"  {mn.ljust(40)} {_fmt(a3, 10)} {str(len(common_list)).rjust(6)}")

    # Write CSVs
    if args.out_dir:
        out_dir = Path(args.out_dir)
        for task, rows in all_csv_rows.items():
            _write_csv(out_dir / f"eval_llm_{task}_{args.split}.csv", rows)


if __name__ == "__main__":
    main()
