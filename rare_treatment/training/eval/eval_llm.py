#!/usr/bin/env python3
"""
Evaluate individual LLM models' treatment ranking performance.

Uses importance_score from each model's treatment_plan_output.json as the
ranking basis, and treatment_score.json as the ground truth label.

Computes full ranking metrics (Hit@K, NDCG, MRR, Precision, Recall,
MAP) per model — aligned with eval_ml.py for fair comparison.

Usage:
    python eval_llm.py \
        --score_root /data/scores \
        --train_ids dataset/train_cases.json \
        --test_ids dataset/test_cases.json
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
PARENT_DIR = THIS_DIR.parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

from data_io import read_json, to_yes
from metrics import (
    ndcg_at_k, mrr, hit_at_k,
    precision_at_k, recall_at_k, map_at_k,
)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def extract_last_int(s: str) -> Optional[int]:
    nums = re.findall(r"\d+", str(s))
    if not nums:
        return None
    return int(nums[-1])


def find_treatment_number_from_path(path_keys: List[str]) -> Optional[int]:
    for k in reversed(path_keys):
        num = extract_last_int(k)
        if num is not None:
            return num
    return None


# ---------------------------------------------------------------------------
# Label parsing (treatment_score.json)
# ---------------------------------------------------------------------------

def extract_labels_recursive(obj: Any, path_keys: List[str], out_rows: List[Dict[str, Any]]) -> None:
    if isinstance(obj, dict):
        if "is_suggested_treatment_appropriate" in obj:
            tnum = find_treatment_number_from_path(path_keys)
            out_rows.append({
                "treatment_number": tnum,
                "is_appropriate": to_yes(obj.get("is_suggested_treatment_appropriate")),
                "path_key": path_keys[-1] if path_keys else "",
            })
        for k, v in obj.items():
            extract_labels_recursive(v, path_keys + [str(k)], out_rows)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            extract_labels_recursive(v, path_keys + [f"[{i}]"], out_rows)


def parse_label_file(file_path: Path) -> List[Dict[str, Any]]:
    data = read_json(file_path)
    root = data.get("suggested_treatment_score", {})
    rows: List[Dict[str, Any]] = []
    extract_labels_recursive(root, ["suggested_treatment_score"], rows)
    return rows


# ---------------------------------------------------------------------------
# Score parsing (treatment_plan_output.json)
# ---------------------------------------------------------------------------

def extract_scores_recursive(obj: Any, path_keys: List[str], out_scores: Dict[int, float]) -> None:
    if isinstance(obj, dict):
        if "importance_score" in obj:
            tnum = find_treatment_number_from_path(path_keys)
            try:
                val = float(obj["importance_score"])
            except (ValueError, TypeError):
                val = -1.0
            if tnum is not None:
                out_scores[tnum] = val
        for k, v in obj.items():
            extract_scores_recursive(v, path_keys + [str(k)], out_scores)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            extract_scores_recursive(v, path_keys + [f"[{i}]"], out_scores)


def parse_plan_file(file_path: Path) -> Dict[int, float]:
    data = read_json(file_path)
    scores_map: Dict[int, float] = {}
    extract_scores_recursive(data, [], scores_map)
    return scores_map


# ---------------------------------------------------------------------------
# Per-case metric aggregation
# ---------------------------------------------------------------------------

def aggregate_case(sorted_rows: List[Dict[str, Any]], ks: List[int]) -> Dict[str, Any]:
    """Calculate full ranking metrics for a single case (rows already sorted by score)."""
    labels = np.array([r["is_appropriate"] for r in sorted_rows], dtype=int)
    total = len(labels)
    appr = int(np.sum(labels))

    metrics: Dict[str, Any] = {
        "total_treatments": total,
        "appropriate_count": appr,
        "inappropriate_count": total - appr,
        "at_least_one_appropriate": 1 if appr > 0 else 0,
        "MRR": mrr(labels),
    }
    for k in ks:
        metrics[f"nDCG@{k}"] = ndcg_at_k(labels, k)
        metrics[f"Hit@{k}"] = hit_at_k(labels, k)
        metrics[f"Precision@{k}"] = precision_at_k(labels, k)
        metrics[f"Recall@{k}"] = recall_at_k(labels, k)
        metrics[f"MAP@{k}"] = map_at_k(labels, k)

    return metrics


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

EVAL_KS = list(range(1, 11))


def evaluate_split(
    split_name: str,
    case_ids: List[str],
    score_root: Path,
    label_fname: str,
    plan_fname: str,
    models: Optional[List[str]] = None,
) -> None:
    case_ids = [str(x) for x in case_ids]
    print(f"\n{'='*20} Evaluating {split_name.upper()} (N={len(case_ids)}) {'='*20}")

    if models is None:
        active_models = [p.name for p in score_root.iterdir() if p.is_dir()]
        active_models.sort()
    else:
        active_models = [m for m in models if (score_root / m).is_dir()]

    if not active_models:
        print(f"Error: no model directories found under: {score_root}")
        return

    # Collect per-model, per-case metrics
    per_model: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    union_hits = 0

    metric_keys = ["MRR"]
    for k in EVAL_KS:
        metric_keys.extend([f"nDCG@{k}", f"Hit@{k}", f"Precision@{k}", f"Recall@{k}", f"MAP@{k}"])

    for case_id in case_ids:
        any_model_hit = False

        for model in active_models:
            model_dir = score_root / model / case_id
            path_label = model_dir / label_fname
            path_plan = model_dir / plan_fname

            if not path_label.exists():
                continue

            rows = parse_label_file(path_label)
            if not rows:
                continue

            scores_map = {}
            if path_plan.exists():
                scores_map = parse_plan_file(path_plan)

            for r in rows:
                tnum = r.get("treatment_number")
                r["score"] = scores_map.get(tnum, -1.0)

            rows.sort(key=lambda x: x["score"], reverse=True)
            agg = aggregate_case(rows, EVAL_KS)

            pm = per_model[model]
            pm["n_cases"].append(1)
            pm["case_appr"].append(float(agg["appropriate_count"]))

            for mk in metric_keys:
                pm[mk].append(float(agg[mk]))

            if agg["at_least_one_appropriate"] == 1:
                any_model_hit = True

        if any_model_hit:
            union_hits += 1

    # Sort models by Hit@1 descending
    def model_sort_key(m: str) -> float:
        vals = per_model[m]["Hit@1"]
        return (sum(vals) / len(vals)) if vals else 0.0

    sorted_models = sorted(active_models, key=model_sort_key, reverse=True)

    def get_mean(pm_dict: Dict[str, List[float]], key: str) -> float:
        vals = pm_dict[key]
        return (sum(vals) / len(vals)) if vals else 0.0

    # ── Table 1: Any Hit Rates (Hit@1 ~ Hit@5) ──
    print("\n[TABLE 1] Any Hit Rates (Hit@1 ~ Hit@5)")
    header1 = (
        f"{'Model Name':<30} | "
        f"{'Hit@1':>7} | {'Hit@2':>7} | {'Hit@3':>7} | {'Hit@4':>7} | {'Hit@5':>7} | "
        f"{'MeanAppr':>8} | {'Ncase':>5}"
    )
    print("-" * len(header1))
    print(header1)
    print("-" * len(header1))

    for model in sorted_models:
        pm = per_model[model]
        ncase = len(pm["n_cases"])
        if ncase == 0:
            continue
        appr_mean = get_mean(pm, "case_appr")
        print(
            f"{model:<30} | "
            f"{get_mean(pm, 'Hit@1') * 100:6.2f}% | {get_mean(pm, 'Hit@2') * 100:6.2f}% | "
            f"{get_mean(pm, 'Hit@3') * 100:6.2f}% | {get_mean(pm, 'Hit@4') * 100:6.2f}% | "
            f"{get_mean(pm, 'Hit@5') * 100:6.2f}% | "
            f"{appr_mean:8.2f} | {ncase:>5}"
        )
    print("-" * len(header1))

    # ── Table 2: Any Hit Rates (Hit@6 ~ Hit@10) ──
    print("\n[TABLE 2] Any Hit Rates (Hit@6 ~ Hit@10)")
    header2 = (
        f"{'Model Name':<30} | "
        f"{'Hit@6':>7} | {'Hit@7':>7} | {'Hit@8':>7} | {'Hit@9':>7} | {'Hit@10':>7} | "
        f"{'Ncase':>5}"
    )
    print("-" * len(header2))
    print(header2)
    print("-" * len(header2))

    for model in sorted_models:
        pm = per_model[model]
        ncase = len(pm["n_cases"])
        if ncase == 0:
            continue
        print(
            f"{model:<30} | "
            f"{get_mean(pm, 'Hit@6') * 100:6.2f}% | {get_mean(pm, 'Hit@7') * 100:6.2f}% | "
            f"{get_mean(pm, 'Hit@8') * 100:6.2f}% | {get_mean(pm, 'Hit@9') * 100:6.2f}% | "
            f"{get_mean(pm, 'Hit@10') * 100:6.2f}% | "
            f"{ncase:>5}"
        )
    print("-" * len(header2))

    # ── Table 3: Ranking Metrics (MRR, nDCG, Hit, Precision, Recall, MAP) ──
    display_ks = [1, 3, 5]
    print("\n[TABLE 3] Ranking Metrics")
    header3_parts = [f"{'Model Name':<30}"]
    header3_parts.append(f"{'MRR':>7}")
    for k in display_ks:
        header3_parts.extend([f"{'nDCG@'+str(k):>8}", f"{'Hit@'+str(k):>7}", f"{'P@'+str(k):>7}", f"{'R@'+str(k):>7}"])
    header3 = " | ".join(header3_parts) + f" | {'Ncase':>5}"
    print("-" * len(header3))
    print(header3)
    print("-" * len(header3))

    for model in sorted_models:
        pm = per_model[model]
        ncase = len(pm["n_cases"])
        if ncase == 0:
            continue
        parts = [f"{model:<30}"]
        parts.append(f"{get_mean(pm, 'MRR'):7.4f}")
        for k in display_ks:
            parts.extend([
                f"{get_mean(pm, f'nDCG@{k}') * 100:7.2f}%",
                f"{get_mean(pm, f'Hit@{k}') * 100:6.2f}%",
                f"{get_mean(pm, f'Precision@{k}') * 100:6.2f}%",
                f"{get_mean(pm, f'Recall@{k}') * 100:6.2f}%",
            ])
        row = " | ".join(parts) + f" | {ncase:>5}"
        print(row)
    print("-" * len(header3))

    union_rate = (union_hits / len(case_ids) * 100) if case_ids else 0.0
    print(f"\n>>> Union Coverage: {union_rate:.2f}% ({union_hits}/{len(case_ids)})")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def load_ids(path: str) -> List[str]:
    p = Path(path)
    if not p.exists():
        return []
    data = read_json(p)
    if isinstance(data, list):
        return [str(x) for x in data]
    return []


def main():
    ap = argparse.ArgumentParser(
        description="Evaluate individual LLM models' treatment ranking."
    )
    ap.add_argument("--score_root", type=str, required=True,
                    help="Directory containing per-model evaluation outputs")
    ap.add_argument("--train_ids", type=str, default="dataset/train_cases.json")
    ap.add_argument("--test_ids", type=str, default="dataset/test_cases.json")
    ap.add_argument("--label_fname", type=str, default="treatment_score.json",
                    help="File with ground truth labels")
    ap.add_argument("--plan_fname", type=str, default="treatment_plan_output.json",
                    help="File with importance_score for ranking")
    ap.add_argument("--models", type=str, default="",
                    help="Comma-separated model filter (empty = all)")
    args = ap.parse_args()

    score_root = Path(args.score_root)
    models = [m.strip() for m in args.models.split(",") if m.strip()] or None
    train_ids = load_ids(args.train_ids)
    test_ids = load_ids(args.test_ids)

    if train_ids:
        evaluate_split("Train", train_ids, score_root, args.label_fname, args.plan_fname, models)
    if test_ids:
        evaluate_split("Test", test_ids, score_root, args.label_fname, args.plan_fname, models)


if __name__ == "__main__":
    main()
