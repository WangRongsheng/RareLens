#!/usr/bin/env python3
"""
Evaluate individual LLM models on diagnosis ranking.

Reads evaluation score JSONs (primary_diagnosis_score.json or
suggested_test_score.json) from each model directory, computes
Acc@1/3/5, MRR, NDCG@1/3/5 per model.

Outputs:
  - Primary metrics: Acc@1/3/5 (%) per model
  - Secondary metrics: MRR, NDCG@1/3/5 per model
  - Optional Excel output matching reference format

Usage:
    python -m rare_diagnosis.training.eval.eval_llm \\
        --score-root /data/scores \\
        --test-ids dataset/test_cases.json \\
        --out-dir /data/results/llm_metrics

    # With Excel output
    python -m rare_diagnosis.training.eval.eval_llm \\
        --score-root /data/scores \\
        --train-ids dataset/train_cases.json \\
        --test-ids dataset/test_cases.json \\
        --out-dir /data/results/llm_metrics \\
        --excel
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from rare_diagnosis.training.eval.metrics import (
    METRIC_KEYS,
    PRIMARY_METRIC_KEYS,
    SECONDARY_METRIC_KEYS,
    case_metrics,
)


# ---------------------------------------------------------------------------
# Score parsing
# ---------------------------------------------------------------------------

def parse_score(score: Any) -> int:
    try:
        if isinstance(score, str):
            return int(float(score))
        return int(score) if isinstance(score, (int, float)) else -1
    except Exception:
        return -1


def extract_scores_from_json(data: dict) -> List[int]:
    """Extract evaluation scores from diagnosis score JSON."""
    if not data:
        return []

    root_key = "most_likely_diagnosis" if "most_likely_diagnosis" in data else "suggested_test_score"
    if root_key not in data:
        return []

    content = data[root_key]
    scores = []

    def get_score(item):
        if isinstance(item, dict):
            return parse_score(item.get("evaluation_score", -1))
        elif isinstance(item, list) and len(item) > 0 and isinstance(item[0], dict):
            return parse_score(item[0].get("evaluation_score", -1))
        return parse_score(item)

    if isinstance(content, dict):
        def safe_digit(k):
            m = re.search(r"\d+", k)
            return int(m.group()) if m else 999
        for k in sorted(content.keys(), key=safe_digit):
            s = get_score(content[k])
            if s != -1:
                scores.append(s)
    elif isinstance(content, list):
        for item in content:
            s = get_score(item)
            if s != -1:
                scores.append(s)

    return scores


def read_json(path: Path) -> Optional[dict]:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

SCORE_FILES = ["primary_diagnosis_score.json", "suggested_test_score.json"]


def process_case(model_path: Path, case_id: str) -> Optional[Dict[str, float]]:
    """Process a single case for a model, return metrics or None."""
    data = None
    for fname in SCORE_FILES:
        p = model_path / case_id / fname
        if p.exists():
            data = read_json(p)
            if data:
                break
    if not data:
        return None

    scores = extract_scores_from_json(data)
    if not scores:
        return None

    # Score >= 5 is a hit (perfect match)
    rels = [1 if s >= 5 else 0 for s in scores]
    return case_metrics(rels)


def evaluate_split(
    split_name: str,
    case_ids: List[str],
    score_root: Path,
    models: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Evaluate all models on a split. Returns list of model summary dicts."""
    if models is None:
        active_models = sorted(p.name for p in score_root.iterdir() if p.is_dir())
    else:
        active_models = [m for m in models if (score_root / m).is_dir()]

    if not active_models:
        print(f"  No model directories found under: {score_root}")
        return []

    print(f"\n{'='*20} {split_name.upper()} (N={len(case_ids)}) {'='*20}")

    results = []
    for model in active_models:
        model_path = score_root / model
        per_case = []

        for cid in case_ids:
            m = process_case(model_path, cid)
            if m:
                per_case.append(m)

        if not per_case:
            continue

        n = len(per_case)
        summary: Dict[str, Any] = {
            "model": model,
            "split": split_name,
            "n_cases": n,
        }
        for k in METRIC_KEYS:
            summary[k] = float(np.mean([m[k] for m in per_case]))
        results.append(summary)

        print(
            f"  {model:<45} Acc@1={summary['Acc@1']:.2%} Acc@3={summary['Acc@3']:.2%} "
            f"Acc@5={summary['Acc@5']:.2%} MRR={summary['MRR']:.4f} N={n}"
        )

    # Sort by Acc@1 descending
    results.sort(key=lambda r: -r["Acc@1"])
    return results


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def save_primary_metrics(results: List[Dict[str, Any]], out_path: Path, as_excel: bool = False):
    """Save main metrics: model, n_cases, Acc@1/3/5 (%)."""
    rows = []
    for r in results:
        rows.append({
            "模型名称": r["model"],
            "Cases总数": r["n_cases"],
            "Acc@1 (%)": round(r["Acc@1"] * 100, 2),
            "Acc@3 (%)": round(r["Acc@3"] * 100, 2),
            "Acc@5 (%)": round(r["Acc@5"] * 100, 2),
        })
    df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if as_excel:
        df.to_excel(out_path.with_suffix(".xlsx"), index=False)
    else:
        df.to_csv(out_path.with_suffix(".csv"), index=False)
    print(f"  -> Primary metrics: {out_path.with_suffix('.xlsx' if as_excel else '.csv')}")


def save_secondary_metrics(results: List[Dict[str, Any]], out_path: Path, as_excel: bool = False):
    """Save secondary metrics: model, MRR, NDCG@1/3/5."""
    rows = []
    for r in results:
        rows.append({
            "模型名称": r["model"],
            "MRR": round(r["MRR"], 4),
            "NDCG@1": round(r["NDCG@1"], 4),
            "NDCG@3": round(r["NDCG@3"], 4),
            "NDCG@5": round(r["NDCG@5"], 4),
        })
    df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if as_excel:
        df.to_excel(out_path.with_suffix(".xlsx"), index=False)
    else:
        df.to_csv(out_path.with_suffix(".csv"), index=False)
    print(f"  -> Secondary metrics: {out_path.with_suffix('.xlsx' if as_excel else '.csv')}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def load_ids(path: str) -> List[str]:
    p = Path(path)
    if not p.exists():
        return []
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return [str(x) for x in data] if isinstance(data, list) else []


def main():
    ap = argparse.ArgumentParser(
        description="Evaluate individual LLM models on diagnosis ranking."
    )
    ap.add_argument("--score-root", required=True,
                    help="Root directory with per-model evaluation JSONs")
    ap.add_argument("--train-ids", default="", help="JSON list of train case IDs")
    ap.add_argument("--test-ids", default="", help="JSON list of test case IDs")
    ap.add_argument("--models", default="",
                    help="Comma-separated model filter (empty = all)")
    ap.add_argument("--out-dir", default=None, help="Output directory for metric files")
    ap.add_argument("--out-csv", default=None, help="Save all results to a single CSV")
    ap.add_argument("--excel", action="store_true",
                    help="Output Excel files (.xlsx) matching reference format")
    args = ap.parse_args()

    score_root = Path(args.score_root)
    models = [m.strip() for m in args.models.split(",") if m.strip()] or None

    all_results = []
    for split_name, ids_path in [("train", args.train_ids), ("test", args.test_ids)]:
        if not ids_path:
            continue
        case_ids = load_ids(ids_path)
        if not case_ids:
            continue
        results = evaluate_split(split_name, case_ids, score_root, models)
        all_results.extend(results)

        # Save per-split primary and secondary metrics
        if args.out_dir and results:
            out_dir = Path(args.out_dir)
            save_primary_metrics(
                results,
                out_dir / f"{split_name}_Top{len(case_ids)}",
                as_excel=args.excel,
            )
            save_secondary_metrics(
                results,
                out_dir / f"{split_name}_ndcg_mrr",
                as_excel=args.excel,
            )

    if not all_results:
        print("No results generated.")
        return

    # Save combined CSV
    if args.out_csv:
        df = pd.DataFrame(all_results)
        Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out_csv, index=False)
        print(f"\nSaved: {args.out_csv}")


if __name__ == "__main__":
    main()
