#!/usr/bin/env python3
"""
Step 3b: Run inference with trained XGBoost models on test features.

Loads per-fold model files from a directory and ensembles predictions.

Usage:
    python infer_ranker.py \\
        --model-dir /data/models/treatment/models \\
        --test-csv /data/features/features_test.csv \\
        --out-dir /data/results
"""
from __future__ import annotations

import argparse
import sys
from glob import glob
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import xgboost as xgb

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
if "data_io" in sys.modules and not sys.modules["data_io"].__file__.startswith(str(THIS_DIR)):
    del sys.modules["data_io"]

from data_io import load_features_csv, export_ranked_json, save_predictions_csv
from eval.metrics import hit_success_rates, format_hit_table


def get_expected_feature_names(model: xgb.XGBRanker, fallback: List[str]) -> List[str]:
    booster = model.get_booster()
    names = booster.feature_names
    if names:
        return list(names)
    return list(fallback)


def align_features_for_model(
    df_test: pd.DataFrame,
    expected_feature_names: List[str],
) -> pd.DataFrame:
    x = df_test.reindex(columns=expected_feature_names, fill_value=0.0).copy()
    for c in expected_feature_names:
        x[c] = pd.to_numeric(x[c], errors="coerce").fillna(0.0)
    return x


def load_models(model_path: str | None, model_dir: str | None) -> List[xgb.XGBRanker]:
    models: List[xgb.XGBRanker] = []
    if model_path:
        model = xgb.XGBRanker()
        model.load_model(model_path)
        models.append(model)
        return models

    if not model_dir:
        raise SystemExit("Provide either --model-path or --model-dir")

    model_dir_path = Path(model_dir)
    model_files = sorted(glob(str(model_dir_path / "*.json"))) + sorted(glob(str(model_dir_path / "*.ubj")))
    if not model_files:
        raise SystemExit(f"No model files found in {model_dir_path}")

    for path in model_files:
        model = xgb.XGBRanker()
        model.load_model(path)
        models.append(model)
    return models


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference for treatment ranker")
    parser.add_argument("--model-path", default=None, help="Single model path (.json/.ubj)")
    parser.add_argument("--model-dir", default=None, help="Directory with models to ensemble")
    parser.add_argument("--test-csv", required=True, help="features_test.csv path")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    args = parser.parse_args()

    df_test, feat_cols = load_features_csv(Path(args.test_csv))
    models = load_models(args.model_path, args.model_dir)

    preds = np.zeros(len(df_test), dtype=float)
    for model in models:
        expected = get_expected_feature_names(model, feat_cols)
        x_test = align_features_for_model(df_test, expected)
        preds += np.asarray(model.predict(x_test), dtype=float)
    preds /= len(models)

    out_dir = Path(args.out_dir)
    save_predictions_csv(df_test, preds, out_dir / "test_predictions.csv", score_col="score")
    export_ranked_json(df_test, preds, out_dir / "ranked_results.json", score_col="score")

    if "label" in df_test.columns:
        eval_df = df_test[["case_id", "label"]].copy()
        eval_df["score"] = preds
        rates = hit_success_rates(eval_df, "score")
        print(format_hit_table(rates, "Ensemble" if len(models) > 1 else "Model", eval_df["case_id"].nunique()))


if __name__ == "__main__":
    main()
