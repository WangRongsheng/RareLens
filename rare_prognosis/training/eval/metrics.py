"""
Shared metric functions for prognosis evaluation.

Core metrics:  accuracy, MCC, macro F1, balanced accuracy, per-class recall
Secondary metrics:  severe recall, false optimism rate, MAOE, OBI (ordinal bias index)
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Ordinal and severity definitions
# ---------------------------------------------------------------------------

ORDINAL_LABELS: Dict[str, List[str]] = {
    "overall_outcome": [
        "complete_recovery", "partial_recovery", "stabilization",
        "progression", "terminal",
    ],
    "functional_status": ["none", "mild", "moderate", "severe"],
    "symptom_burden": ["none", "occasional", "persistent_mild", "persistent_severe"],
}

SEVERE_LABELS: Dict[str, set] = {
    "overall_outcome": {"progression", "terminal"},
    "functional_status": {"severe"},
    "symptom_burden": {"persistent_severe"},
}

TASKS = ["overall_outcome", "functional_status", "symptom_burden"]


def ordinal_index(task: str, label: str) -> Optional[int]:
    order = ORDINAL_LABELS.get(task)
    if order is None or label not in order:
        return None
    return order.index(label)


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------

def build_confusion(
    case_ids: List[str],
    gt_by_id: Dict[str, str],
    pred_by_id: Dict[str, str],
    labels: List[str],
) -> dict:
    matrix = {g: {p: 0 for p in labels} for g in labels}
    total = correct = missing = 0
    for cid in case_ids:
        gt = gt_by_id.get(cid)
        pred = pred_by_id.get(cid)
        if gt is None or pred is None or gt not in labels or pred not in labels:
            missing += 1
            continue
        matrix[gt][pred] += 1
        total += 1
        if gt == pred:
            correct += 1
    return {"matrix": matrix, "total": total, "correct": correct, "missing": missing}


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------

def _safe_div(a, b):
    return a / b if b else None


def _f1(p, r):
    if p is None or r is None or (p + r) == 0:
        return None
    return 2 * p * r / (p + r)


def compute_core_metrics(
    case_ids: List[str],
    gt_by_id: Dict[str, str],
    pred_by_id: Dict[str, str],
    labels: List[str],
) -> dict:
    """Accuracy, MCC, macro F1, balanced accuracy, per-class recall."""
    conf = build_confusion(case_ids, gt_by_id, pred_by_id, labels)
    mat = conf["matrix"]
    total = conf["total"]
    correct = conf["correct"]
    if total == 0:
        return {"accuracy": None, "mcc": None, "macro_f1": None,
                "balanced_accuracy": None, "per_class_recall": {}, "n_scored": 0}

    recalls = []
    f1s = []
    per_class_recall = {}
    for label in labels:
        tp = mat[label][label]
        fn = sum(mat[label][p] for p in labels if p != label)
        fp = sum(mat[g][label] for g in labels if g != label)
        rec = _safe_div(tp, tp + fn)
        prec = _safe_div(tp, tp + fp)
        recalls.append(rec)
        f1s.append(_f1(prec, rec))
        per_class_recall[label] = rec

    valid_rec = [r for r in recalls if r is not None]
    valid_f1 = [f for f in f1s if f is not None]

    # MCC
    row_totals = [sum(mat[l].values()) for l in labels]
    col_totals = [sum(mat[g][l] for g in labels) for l in labels]
    num = (correct * total) - sum(c * r for c, r in zip(col_totals, row_totals))
    dl = (total ** 2) - sum(c * c for c in col_totals)
    dr = (total ** 2) - sum(r * r for r in row_totals)
    denom = math.sqrt(max(0.0, dl) * max(0.0, dr))

    return {
        "accuracy": correct / total,
        "mcc": num / denom if denom else None,
        "macro_f1": sum(valid_f1) / len(valid_f1) if valid_f1 else None,
        "balanced_accuracy": sum(valid_rec) / len(valid_rec) if valid_rec else None,
        "per_class_recall": per_class_recall,
        "n_scored": total,
    }


# ---------------------------------------------------------------------------
# Secondary metrics
# ---------------------------------------------------------------------------

def compute_severe_recall(
    task: str,
    case_ids: List[str],
    gt_by_id: Dict[str, str],
    pred_by_id: Dict[str, str],
) -> Optional[float]:
    severe = SEVERE_LABELS[task]
    total = correct = 0
    for cid in case_ids:
        gt = gt_by_id.get(cid)
        if gt not in severe:
            continue
        pred = pred_by_id.get(cid)
        if pred is None:
            continue
        total += 1
        if pred in severe:
            correct += 1
    return correct / total if total else None


def compute_false_optimism_rate(
    task: str,
    case_ids: List[str],
    gt_by_id: Dict[str, str],
    pred_by_id: Dict[str, str],
) -> Optional[float]:
    sr = compute_severe_recall(task, case_ids, gt_by_id, pred_by_id)
    return (1.0 - sr) if sr is not None else None


def compute_maoe_and_obi(
    task: str,
    case_ids: List[str],
    gt_by_id: Dict[str, str],
    pred_by_id: Dict[str, str],
) -> Tuple[Optional[float], Optional[float]]:
    """Mean Absolute Ordinal Error and Optimism Bias Index."""
    abs_errors = []
    signed_errors = []
    for cid in case_ids:
        gt = gt_by_id.get(cid)
        pred = pred_by_id.get(cid)
        if gt is None or pred is None:
            continue
        gi = ordinal_index(task, gt)
        pi = ordinal_index(task, pred)
        if gi is None or pi is None:
            continue
        diff = pi - gi
        abs_errors.append(abs(diff))
        signed_errors.append(diff)
    maoe = sum(abs_errors) / len(abs_errors) if abs_errors else None
    obi = sum(signed_errors) / len(signed_errors) if signed_errors else None
    return maoe, obi


# ---------------------------------------------------------------------------
# All-in-one per-task evaluation
# ---------------------------------------------------------------------------

def evaluate_task(
    task: str,
    case_ids: List[str],
    gt_by_id: Dict[str, str],
    pred_by_id: Dict[str, str],
) -> dict:
    labels = ORDINAL_LABELS[task]
    core = compute_core_metrics(case_ids, gt_by_id, pred_by_id, labels)
    maoe, obi = compute_maoe_and_obi(task, case_ids, gt_by_id, pred_by_id)
    sr = compute_severe_recall(task, case_ids, gt_by_id, pred_by_id)
    fo = (1.0 - sr) if sr is not None else None
    return {**core, "maoe": maoe, "obi": obi, "severe_recall": sr, "false_optimism": fo}


# ---------------------------------------------------------------------------
# All3 accuracy
# ---------------------------------------------------------------------------

def compute_all3_accuracy(
    case_ids: List[str],
    preds_by_task: Dict[str, Dict[str, str]],
    gt_by_task: Dict[str, Dict[str, str]],
) -> Optional[float]:
    """Fraction of cases where all 3 tasks are correct."""
    total = correct = 0
    for cid in case_ids:
        all_ok = True
        any_scored = False
        for task in TASKS:
            gt = gt_by_task.get(task, {}).get(cid)
            pred = preds_by_task.get(task, {}).get(cid)
            if gt is None or pred is None:
                all_ok = False
                break
            any_scored = True
            if gt != pred:
                all_ok = False
                break
        if any_scored:
            total += 1
            if all_ok:
                correct += 1
    return correct / total if total else None
