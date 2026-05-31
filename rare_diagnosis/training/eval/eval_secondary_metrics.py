#!/usr/bin/env python3
"""
Secondary diagnosis metrics: primary vs follow-up convergence and rescue analysis.

Compares ML ranking outputs between primary and follow-up stages, computing:
  - Per-stage Acc@1/3/5, MRR, NDCG metrics
  - Convergence flags: gained/lost/stable correct/stable wrong
  - Rescue rate: cases where follow-up corrects primary failures

Usage:
    python -m rare_diagnosis.training.eval.eval_secondary_metrics \\
        --primary-csv /data/models/primary/test_predictions_ranked.csv \\
        --followup-csv /data/models/followup/test_predictions_ranked.csv \\
        --out-dir /data/results/secondary_metrics

    # LLM rescue: compare ML vs best LLM
    python -m rare_diagnosis.training.eval.eval_secondary_metrics \\
        --primary-csv /data/models/primary/test_predictions_ranked.csv \\
        --llm-score-root /data/scores \\
        --test-ids dataset/test_cases.json \\
        --out-dir /data/results/secondary_metrics
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from rare_diagnosis.training.eval.eval_llm import process_case as llm_process_case, read_json, SCORE_FILES
from rare_diagnosis.training.eval.metrics import case_metrics


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_csv_metrics(csv_path: str, stage_label: str) -> pd.DataFrame:
    """Load CSV and compute per-case metrics."""
    df = pd.read_csv(csv_path, low_memory=False)
    df["rank"] = pd.to_numeric(df["rank"], errors="coerce")
    rel_col = "is_correct" if "is_correct" in df.columns else "label"

    rows = []
    for case_id, group in df.groupby("case_id", sort=False):
        group = group.sort_values("rank")
        rels = group[rel_col].fillna(0).astype(int).tolist()[:5]
        m = case_metrics(rels)
        m["case_id"] = str(case_id)
        rows.append(m)

    result = pd.DataFrame(rows)
    print(f"  [{stage_label}] {len(result)} cases loaded")
    return result


# ---------------------------------------------------------------------------
# Convergence analysis (primary vs follow-up)
# ---------------------------------------------------------------------------

def convergence_analysis(primary_df: pd.DataFrame, follow_df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze how predictions change from primary to follow-up."""
    merged = pd.merge(primary_df, follow_df, on="case_id", suffixes=("_primary", "_follow"), how="inner")
    n = len(merged)

    results: Dict[str, Any] = {"paired_cases": n}

    for k_name in ["Acc@1", "Acc@3", "Acc@5"]:
        p_col = f"{k_name}_primary"
        f_col = f"{k_name}_follow"

        gained = int(((merged[p_col] == 0) & (merged[f_col] == 1)).sum())
        lost = int(((merged[p_col] == 1) & (merged[f_col] == 0)).sum())
        stable_correct = int(((merged[p_col] == 1) & (merged[f_col] == 1)).sum())
        stable_wrong = int(((merged[p_col] == 0) & (merged[f_col] == 0)).sum())

        primary_wrong = gained + stable_wrong
        rescue_rate = gained / primary_wrong if primary_wrong > 0 else 0.0

        results[f"{k_name}_gained"] = gained
        results[f"{k_name}_lost"] = lost
        results[f"{k_name}_stable_correct"] = stable_correct
        results[f"{k_name}_stable_wrong"] = stable_wrong
        results[f"{k_name}_rescue_rate"] = rescue_rate
        results[f"{k_name}_net_gain"] = gained - lost

    # MRR improvement
    results["MRR_primary_mean"] = float(merged["MRR_primary"].mean())
    results["MRR_follow_mean"] = float(merged["MRR_follow"].mean())
    results["MRR_delta"] = results["MRR_follow_mean"] - results["MRR_primary_mean"]

    return results


# ---------------------------------------------------------------------------
# ML vs LLM rescue analysis
# ---------------------------------------------------------------------------

def load_llm_case_acc(score_root: Path, case_ids: List[str]) -> Dict[str, Dict[str, int]]:
    """Load per-model Acc@1 for each case, reusing eval_llm score parsing."""
    model_dirs = sorted(p.name for p in score_root.iterdir() if p.is_dir())
    model_acc: Dict[str, Dict[str, int]] = {}

    for model in model_dirs:
        model_path = score_root / model
        acc_map: Dict[str, int] = {}
        for cid in case_ids:
            m = llm_process_case(model_path, cid)
            acc_map[cid] = int(m["Acc@1"]) if m else 0
        model_acc[model] = acc_map

    return model_acc


def ml_vs_llm_rescue(
    ml_csv: str, score_root: Path, case_ids: List[str],
) -> Dict[str, Any]:
    """Compare ML ranking vs best LLM on rescue metrics."""
    ml_df = load_csv_metrics(ml_csv, "ML")
    ml_acc = {str(row["case_id"]): int(row["Acc@1"]) for _, row in ml_df.iterrows()}

    llm_acc = load_llm_case_acc(score_root, case_ids)

    # Find best LLM by overall accuracy
    model_scores = {}
    for model, acc_map in llm_acc.items():
        vals = [acc_map.get(cid, 0) for cid in case_ids if cid in acc_map]
        model_scores[model] = np.mean(vals) if vals else 0.0

    best_llm = max(model_scores, key=model_scores.get) if model_scores else None
    if not best_llm:
        return {"error": "No LLM models found"}

    best_llm_acc = llm_acc[best_llm]

    # Compute rescue metrics
    shared = [cid for cid in case_ids if cid in ml_acc and cid in best_llm_acc]
    n = len(shared)
    rescued = sum(1 for cid in shared if best_llm_acc[cid] == 0 and ml_acc[cid] == 1)
    regressed = sum(1 for cid in shared if best_llm_acc[cid] == 1 and ml_acc[cid] == 0)
    llm_wrong = sum(1 for cid in shared if best_llm_acc[cid] == 0)
    llm_correct = sum(1 for cid in shared if best_llm_acc[cid] == 1)

    return {
        "best_llm": best_llm,
        "best_llm_acc": model_scores[best_llm],
        "ml_acc": float(np.mean([ml_acc[cid] for cid in shared])),
        "shared_cases": n,
        "rescued": rescued,
        "regressed": regressed,
        "rescue_rate": rescued / llm_wrong if llm_wrong > 0 else 0.0,
        "regression_rate": regressed / llm_correct if llm_correct > 0 else 0.0,
        "net_rescue": (rescued - regressed) / n if n > 0 else 0.0,
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def fmt(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return f"{v:.6f}"
    return str(v)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Secondary diagnosis metrics: convergence and rescue analysis."
    )
    ap.add_argument("--primary-csv", help="Primary stage ML predictions CSV")
    ap.add_argument("--followup-csv", default=None, help="Follow-up stage ML predictions CSV")
    ap.add_argument("--llm-score-root", default=None, help="LLM score root for rescue analysis")
    ap.add_argument("--test-ids", default=None, help="Test case IDs JSON")
    ap.add_argument("--out-dir", default="results/secondary_metrics", help="Output directory")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Convergence analysis (primary vs follow-up)
    if args.primary_csv and args.followup_csv:
        print("\n=== Primary vs Follow-up Convergence ===")
        primary_df = load_csv_metrics(args.primary_csv, "primary")
        follow_df = load_csv_metrics(args.followup_csv, "followup")
        conv = convergence_analysis(primary_df, follow_df)

        print(f"\nPaired cases: {conv['paired_cases']}")
        for k_name in ["Acc@1", "Acc@3", "Acc@5"]:
            print(f"  {k_name}: gained={conv[f'{k_name}_gained']} lost={conv[f'{k_name}_lost']} "
                  f"rescue_rate={conv[f'{k_name}_rescue_rate']:.4f} net={conv[f'{k_name}_net_gain']}")
        print(f"  MRR: primary={conv['MRR_primary_mean']:.4f} follow={conv['MRR_follow_mean']:.4f} "
              f"delta={conv['MRR_delta']:.4f}")

        conv_path = out_dir / "convergence.csv"
        with conv_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["metric", "value"])
            for k, v in conv.items():
                w.writerow([k, fmt(v)])
        print(f"  -> {conv_path}")

    # ML vs LLM rescue analysis
    if args.primary_csv and args.llm_score_root and args.test_ids:
        print("\n=== ML vs Best LLM Rescue ===")
        with Path(args.test_ids).open("r", encoding="utf-8") as f:
            test_ids = [str(x) for x in json.load(f)]

        rescue = ml_vs_llm_rescue(args.primary_csv, Path(args.llm_score_root), test_ids)
        for k, v in rescue.items():
            print(f"  {k}: {fmt(v)}")

        rescue_path = out_dir / "ml_vs_llm_rescue.csv"
        with rescue_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["metric", "value"])
            for k, v in rescue.items():
                w.writerow([k, fmt(v)])
        print(f"  -> {rescue_path}")

    print(f"\nDone. Results in {out_dir}")


if __name__ == "__main__":
    main()
