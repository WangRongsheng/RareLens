#!/usr/bin/env python3
"""
Evaluate ML diagnosis ranking from CSV or JSON predictions.

Computes Acc@1/3/5, MRR, NDCG@1/3/5. Supports grouping by visit_type
(primary/followup/overall).

Usage:
    # From ranked CSV (output of train_ranker.py)
    python -m rare_diagnosis.training.eval.eval_ml \
        --csv /data/models/test_predictions_ranked.csv

    # From ranked JSON
    python -m rare_diagnosis.training.eval.eval_ml \
        --json /data/models/test_predictions_ranked.json

    # With visit_type grouping
    python -m rare_diagnosis.training.eval.eval_ml \
        --csv /data/models/test_predictions_ranked.csv \
        --visit-map /data/visit_types.json \
        --out-csv /data/results/ml_metrics.csv
"""
from __future__ import annotations

import argparse
import csv
import json
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
# Data loading
# ---------------------------------------------------------------------------

def load_from_csv(csv_path: Path) -> Dict[str, List[int]]:
    """Load per-case relevance lists from CSV. Returns {case_id: [rels...]}."""
    df = pd.read_csv(csv_path, low_memory=False)
    df["rank"] = pd.to_numeric(df["rank"], errors="coerce")

    rel_col = "is_correct" if "is_correct" in df.columns else "label"

    case_rels: Dict[str, List[int]] = {}
    for case_id, group in df.groupby("case_id"):
        group = group.sort_values("rank")
        rels = group[rel_col].fillna(0).astype(int).tolist()
        case_rels[str(case_id)] = rels
    return case_rels


def load_from_json(json_path: Path) -> Dict[str, List[int]]:
    """Load per-case relevance lists from JSON predictions."""
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    case_rels: Dict[str, List[int]] = {}
    for case_id, case_data in data.items():
        preds = case_data.get("predictions", [])
        preds_sorted = sorted(preds, key=lambda p: p.get("rank", p.get("score", 0)))
        if preds_sorted and "rank" not in preds_sorted[0]:
            preds_sorted = sorted(preds, key=lambda p: -p.get("score", 0))
        rels = [int(p.get("is_correct", p.get("label", 0))) for p in preds_sorted]
        case_rels[str(case_id)] = rels
    return case_rels


def load_visit_map(path: str) -> Dict[str, str]:
    """Load case_id -> visit_type mapping from JSON or CSV."""
    p = Path(path)
    if p.suffix.lower() == ".json":
        with p.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        out: Dict[str, str] = {}
        for k, v in raw.items():
            if isinstance(v, dict):
                out[str(k)] = str(v.get("visit_type", "")).strip().lower()
            else:
                out[str(k)] = str(v).strip().lower()
        return out

    out = {}
    with p.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cid = str(row.get("case_id", "")).strip()
            vt = str(row.get("visit_type", "")).strip().lower()
            if cid and vt:
                out[cid] = vt
    return out


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def aggregate_metrics(per_case: List[Dict[str, float]], name: str) -> Dict[str, Any]:
    if not per_case:
        return {"group": name, "n_cases": 0, **{k: 0.0 for k in METRIC_KEYS}}
    return {
        "group": name,
        "n_cases": len(per_case),
        **{k: float(np.mean([m[k] for m in per_case])) for k in METRIC_KEYS},
    }


def evaluate(
    case_rels: Dict[str, List[int]],
    visit_map: Optional[Dict[str, str]] = None,
) -> List[Dict[str, Any]]:
    """Evaluate all cases, optionally grouped by visit_type."""
    grouped: Dict[str, List[Dict[str, float]]] = {"overall": []}

    for case_id, rels in case_rels.items():
        m = case_metrics(rels)
        grouped["overall"].append(m)

        if visit_map:
            vt = visit_map.get(case_id, "unknown")
            grouped.setdefault(vt, []).append(m)

    results = []
    for group_name in sorted(grouped.keys()):
        agg = aggregate_metrics(grouped[group_name], group_name)
        results.append(agg)

    return results


def print_results(results: List[Dict[str, Any]]) -> None:
    header = f"{'Group':<20} {'Acc@1':>8} {'Acc@3':>8} {'Acc@5':>8} {'MRR':>8} {'NDCG@3':>8} {'N':>6}"
    print(f"\n{header}")
    print("-" * len(header))
    for r in results:
        print(
            f"{r['group']:<20} {r['Acc@1']:>7.2%} {r['Acc@3']:>7.2%} {r['Acc@5']:>7.2%} "
            f"{r['MRR']:>8.4f} {r['NDCG@3']:>8.4f} {r['n_cases']:>6}"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Evaluate ML diagnosis ranking predictions."
    )
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument("--csv", help="Path to ranked predictions CSV")
    grp.add_argument("--json", help="Path to ranked predictions JSON")

    ap.add_argument("--visit-map", default=None,
                    help="JSON/CSV with case_id -> visit_type mapping")
    ap.add_argument("--out-csv", default=None, help="Save results to CSV")
    args = ap.parse_args()

    if args.csv:
        case_rels = load_from_csv(Path(args.csv))
    else:
        case_rels = load_from_json(Path(args.json))

    visit_map = load_visit_map(args.visit_map) if args.visit_map else None
    results = evaluate(case_rels, visit_map)

    print_results(results)

    if args.out_csv:
        df = pd.DataFrame(results)
        Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out_csv, index=False)
        print(f"\nSaved: {args.out_csv}")


if __name__ == "__main__":
    main()
