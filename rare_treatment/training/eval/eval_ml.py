#!/usr/bin/env python3
"""
Evaluate ML ensemble treatment ranking performance.

Computes full ranking metrics (Hit@K, Precision@K, Recall@K,
nDCG@K, MAP@K, MRR) on ensemble predictions or per-model rank columns.

Usage:
    # Evaluate ensemble predictions
    python eval_ml.py \
        --input ensemble=/data/results/test_predictions_ensemble.csv:ensemble_score

    # Evaluate individual model rankings from feature CSV
    python eval_ml.py \
        --rank-input gpt5=/data/features/features_test.csv:rank__gpt-5

    # Compare multiple inputs
    python eval_ml.py \
        --input ens=/data/results/test_predictions_ensemble.csv:ensemble_score \
        --rank-input gpt5=/data/features/features_test.csv:rank__gpt-5 \
        --rank-input qwen3=/data/features/features_test.csv:rank__qwen3-32b \
        --ks 1,2,3,5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
if "metrics" in sys.modules and not sys.modules["metrics"].__file__.startswith(str(THIS_DIR)):
    del sys.modules["metrics"]

from metrics import evaluate_all_metrics


def parse_ks(text: str) -> List[int]:
    ks: List[int] = []
    for x in str(text).split(","):
        x = x.strip()
        if not x:
            continue
        ks.append(max(1, int(x)))
    if not ks:
        raise SystemExit("No valid ks parsed.")
    return sorted(set(ks))


def parse_score_input(spec: str, default_score_col: str) -> Tuple[str, Path, str]:
    """Format: name=csv_path[:score_col]"""
    if "=" not in spec:
        raise SystemExit(f"Invalid --input format: {spec}")
    name, rest = spec.split("=", 1)
    if ":" in rest:
        path_s, score_col = rest.rsplit(":", 1)
    else:
        path_s, score_col = rest, default_score_col
    return name.strip(), Path(path_s.strip()), score_col.strip()


def parse_rank_input(spec: str) -> Tuple[str, Path, str]:
    """Format: name=features_csv:rank_col"""
    if "=" not in spec or ":" not in spec:
        raise SystemExit(f"Invalid --rank-input format: {spec}")
    name, rest = spec.split("=", 1)
    path_s, rank_col = rest.rsplit(":", 1)
    return name.strip(), Path(path_s.strip()), rank_col.strip()


def load_score_df(path: Path, case_col: str, label_col: str, score_col: str) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"Missing file: {path}")
    df = pd.read_csv(path, low_memory=False)
    needed = {case_col, label_col, score_col}
    missing = needed - set(df.columns)
    if missing:
        raise SystemExit(f"{path} missing columns: {sorted(missing)}")
    out = df[[case_col, label_col, score_col]].copy()
    out[case_col] = out[case_col].astype(str)
    out[label_col] = pd.to_numeric(out[label_col], errors="coerce").fillna(0).astype(int)
    out[score_col] = pd.to_numeric(out[score_col], errors="coerce").fillna(0.0)
    return out


def load_rank_df(path: Path, case_col: str, label_col: str, rank_col: str) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"Missing file: {path}")
    df = pd.read_csv(path, low_memory=False)
    needed = {case_col, label_col, rank_col}
    missing = needed - set(df.columns)
    if missing:
        raise SystemExit(f"{path} missing columns: {sorted(missing)}")
    out = df[[case_col, label_col, rank_col]].copy()
    out[case_col] = out[case_col].astype(str)
    out[label_col] = pd.to_numeric(out[label_col], errors="coerce").fillna(0).astype(int)
    rank = pd.to_numeric(out[rank_col], errors="coerce").fillna(999.0).clip(lower=1.0)
    out["score"] = 1.0 / (rank + 0.5)
    return out[[case_col, label_col, "score"]]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate ML ensemble ranking metrics for treatment outputs."
    )
    parser.add_argument("--input", action="append", default=[],
                        help="Score input: name=csv_path[:score_col]. Can repeat.")
    parser.add_argument("--rank-input", action="append", default=[],
                        help="Rank input: name=features_csv:rank_col. Can repeat.")
    parser.add_argument("--case-col", default="case_id")
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--default-score-col", default="ensemble_score")
    parser.add_argument("--ks", default="1,2,3,5")
    parser.add_argument("--out-csv", default=None,
                        help="Save results to CSV (optional)")
    args = parser.parse_args()

    ks = parse_ks(args.ks)
    rows: List[Dict] = []

    for spec in args.input:
        name, path, score_col = parse_score_input(spec, args.default_score_col)
        df = load_score_df(path, args.case_col, args.label_col, score_col)
        m = evaluate_all_metrics(df, args.case_col, args.label_col, score_col, ks)
        row: Dict = {
            "name": name,
            "source": str(path),
            "score_col": score_col,
            "Ncase": int(df[args.case_col].nunique()),
        }
        row.update(m)
        rows.append(row)

    for spec in args.rank_input:
        name, path, rank_col = parse_rank_input(spec)
        df = load_rank_df(path, args.case_col, args.label_col, rank_col)
        m = evaluate_all_metrics(df, args.case_col, args.label_col, "score", ks)
        row = {
            "name": name,
            "source": str(path),
            "score_col": f"derived_from_{rank_col}",
            "Ncase": int(df[args.case_col].nunique()),
        }
        row.update(m)
        rows.append(row)

    if not rows:
        raise SystemExit("No inputs provided. Use --input and/or --rank-input.")

    out = pd.DataFrame(rows)
    sort_cols = [f"Hit@{k}" for k in ks if f"Hit@{k}" in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols, ascending=[False] * len(sort_cols))

    if args.out_csv:
        out_path = Path(args.out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(out_path, index=False)
        print(f"Saved: {out_path}")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
