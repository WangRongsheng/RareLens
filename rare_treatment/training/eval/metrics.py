#!/usr/bin/env python3
"""
metrics.py -- Ranking evaluation metrics for treatment reranking.

Supports: Hit@K (Any Hit), NDCG, MRR, MAP, Precision@K, Recall@K.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Per-case metric functions
# ---------------------------------------------------------------------------

def ndcg_at_k(labels: np.ndarray, k: int) -> float:
    sel = labels[:k]
    if sel.size == 0:
        return 0.0
    gains = np.asarray(sel, dtype=float)
    discounts = 1.0 / np.log2(np.arange(2, gains.size + 2))
    dcg = float(np.sum(gains * discounts))

    ideal = np.sort(labels)[::-1][:k]
    ideal = np.asarray(ideal, dtype=float)
    if ideal.size == 0:
        return 0.0
    idcg = float(np.sum(ideal * discounts[: ideal.size]))
    if idcg <= 0:
        return 0.0
    return dcg / idcg


def mrr(labels: np.ndarray) -> float:
    pos = np.where(labels == 1)[0]
    if pos.size == 0:
        return 0.0
    return 1.0 / float(pos[0] + 1)


def hit_at_k(labels: np.ndarray, k: int) -> float:
    sel = labels[:k]
    if sel.size == 0:
        return 0.0
    return 1.0 if np.any(sel == 1) else 0.0


def precision_at_k(labels: np.ndarray, k: int) -> float:
    sel = labels[:k]
    if sel.size == 0:
        return 0.0
    return float(np.mean(sel))


def recall_at_k(labels: np.ndarray, k: int) -> float:
    total_pos = int(np.sum(labels))
    if total_pos <= 0:
        return 0.0
    sel = labels[:k]
    return float(np.sum(sel) / total_pos)


def map_at_k(labels: np.ndarray, k: int) -> float:
    sel = labels[:k]
    total_pos = int(np.sum(labels))
    if sel.size == 0 or total_pos == 0:
        return 0.0
    hits = 0
    ap_sum = 0.0
    for i, y in enumerate(sel, start=1):
        if y == 1:
            hits += 1
            ap_sum += hits / i
    denom = min(total_pos, k)
    if denom <= 0:
        return 0.0
    return ap_sum / denom


# ---------------------------------------------------------------------------
# Aggregation over all cases
# ---------------------------------------------------------------------------

def hit_success_rates(
    df: pd.DataFrame,
    score_col: str,
    label_col: str = "label",
    case_col: str = "case_id",
    k_max: int = 10,
) -> Dict[int, float]:
    """Compute Any-Hit rates (Hit@1..Hit@k) across cases. Returns {k: percent}."""
    metrics: Dict[int, List[float]] = {k: [] for k in range(1, k_max + 1)}
    for _, group in df.groupby(case_col):
        sorted_group = group.sort_values(by=score_col, ascending=False)
        labels = sorted_group[label_col].to_numpy()
        for k in range(1, k_max + 1):
            metrics[k].append(hit_at_k(labels, k))

    return {
        k: (float(np.mean(v)) * 100.0 if v else 0.0)
        for k, v in metrics.items()
    }


def calc_aux_metrics(
    df: pd.DataFrame,
    score_col: str,
    label_col: str = "label",
    case_col: str = "case_id",
    ndcg_ks: Sequence[int] = (1, 3, 5),
) -> Dict[str, float]:
    out: Dict[str, List[float]] = {"MRR": []}
    ks = sorted(set(max(1, int(k)) for k in ndcg_ks))
    for k in ks:
        out[f"NDCG@{k}"] = []

    for _, group in df.groupby(case_col):
        sorted_group = group.sort_values(by=score_col, ascending=False)
        labels = sorted_group[label_col].to_numpy(dtype=int)
        out["MRR"].append(mrr(labels))
        for k in ks:
            out[f"NDCG@{k}"].append(ndcg_at_k(labels, k))

    return {k: float(np.mean(v)) if v else 0.0 for k, v in out.items()}


def evaluate_all_metrics(
    df: pd.DataFrame,
    case_col: str,
    label_col: str,
    score_col: str,
    ks: List[int],
) -> Dict[str, float]:
    """Compute full suite of ranking metrics across all cases."""
    out_lists: Dict[str, List[float]] = {"MRR": [], "MAP": []}
    for k in ks:
        out_lists[f"Hit@{k}"] = []
        out_lists[f"Precision@{k}"] = []
        out_lists[f"Recall@{k}"] = []
        out_lists[f"nDCG@{k}"] = []
        out_lists[f"MAP@{k}"] = []

    for _, group in df.groupby(case_col):
        sorted_group = group.sort_values(by=score_col, ascending=False)
        labels = sorted_group[label_col].to_numpy(dtype=int)
        out_lists["MRR"].append(mrr(labels))
        out_lists["MAP"].append(map_at_k(labels, max(ks)))
        for k in ks:
            out_lists[f"Hit@{k}"].append(hit_at_k(labels, k))
            out_lists[f"Precision@{k}"].append(precision_at_k(labels, k))
            out_lists[f"Recall@{k}"].append(recall_at_k(labels, k))
            out_lists[f"nDCG@{k}"].append(ndcg_at_k(labels, k))
            out_lists[f"MAP@{k}"].append(map_at_k(labels, k))

    metrics: Dict[str, float] = {}
    for k, v in out_lists.items():
        metrics[k] = float(np.mean(v) * 100.0) if k != "MRR" and k != "MAP" else float(np.mean(v))
    return metrics


def format_hit_table(
    rates: Dict[int, float],
    model_name: str,
    n_cases: int,
) -> str:
    """Render two-line tables (Hit@1-5 and Hit@6-10)."""
    header1 = (
        f"{'Model Name':<25} | "
        f"{'Hit@1':>8} | {'Hit@2':>8} | {'Hit@3':>8} | {'Hit@4':>8} | {'Hit@5':>8} | "
        f"{'Ncase':>5}"
    )
    row1 = (
        f"{model_name:<25} | "
        f"{rates.get(1, 0.0):7.2f}% | {rates.get(2, 0.0):7.2f}% | "
        f"{rates.get(3, 0.0):7.2f}% | {rates.get(4, 0.0):7.2f}% | "
        f"{rates.get(5, 0.0):7.2f}% | {n_cases:>5}"
    )

    header2 = (
        f"{'Model Name':<25} | "
        f"{'Hit@6':>8} | {'Hit@7':>8} | {'Hit@8':>8} | {'Hit@9':>8} | {'Hit@10':>8} | "
        f"{'Ncase':>5}"
    )
    row2 = (
        f"{model_name:<25} | "
        f"{rates.get(6, 0.0):7.2f}% | {rates.get(7, 0.0):7.2f}% | "
        f"{rates.get(8, 0.0):7.2f}% | {rates.get(9, 0.0):7.2f}% | "
        f"{rates.get(10, 0.0):7.2f}% | {n_cases:>5}"
    )

    lines = [
        "=" * 80,
        "ANY HIT RATE (Hit@1 ~ Hit@5)",
        "(Rate = % of cases where at least one Top-K candidate has label=1)",
        "=" * 80,
        header1,
        "-" * len(header1),
        row1,
        "-" * len(header1),
        "",
        "=" * 80,
        "ANY HIT RATE (Hit@6 ~ Hit@10)",
        "=" * 80,
        header2,
        "-" * len(header2),
        row2,
        "-" * len(header2),
        "=" * 80,
    ]
    return "\n".join(lines)
