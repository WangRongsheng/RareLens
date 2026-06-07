#!/usr/bin/env python3
"""
Compute secondary treatment metrics (M2-M11) for paper tables.

Compares RareTreatment (ML reranker) against individual LLM baselines on:
  M2:  MRR, NDCG@{1,2,3,5}
  M3:  Average number of appropriate treatments per case
  M4:  Average number of inappropriate treatments per case
  M5:  Case positive probability (>= 1 performed treatment)
  M6:  High risk ratio (safety score <= 2)
  M7:  Overall completeness
  M8:  Overall helpfulness
  M9:  Overall safety
  M10: Performed fraction
  M11: Rescue rate (cases where RareTreatment succeeds but baseline fails)

Usage:
    python eval_secondary_metrics.py \
        --n11-pred-csv /data/results/test_predictions_ensemble.csv \
        --features-test-csv /data/features/features_test.csv \
        --rag-root /data/scores \
        --out-dir /data/results/secondary_metrics
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
PARENT_DIR = THIS_DIR.parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))
for _mod in ("data_io", "metrics"):
    if _mod in sys.modules and not sys.modules[_mod].__file__.startswith(str(PARENT_DIR)):
        del sys.modules[_mod]

from data_io import (
    normalize_text, safe_float, clip_score_1_to_5, to_yes, extract_idx, read_json,
)
from metrics import mrr, ndcg_at_k, hit_at_k

RERANK_NAME = "RareTreatment"
RERANK_TOPK = 10


def mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return 0.0
    return float(sum(values) / len(values))


# ---------------------------------------------------------------------------
# Treatment evaluation dataclass
# ---------------------------------------------------------------------------

@dataclass
class TreatmentEval:
    appropriate: int
    performed: int
    completeness: float
    helpfulness: float
    safety: float


def parse_model_eval(path: Path) -> tuple[List[TreatmentEval], Dict[str, TreatmentEval]]:
    obj = read_json(path)
    root = obj.get("suggested_treatment_score", {})
    ordered: List[TreatmentEval] = []
    dedup: Dict[str, TreatmentEval] = {}
    if not isinstance(root, dict):
        return ordered, dedup

    for _, item in sorted(root.items(), key=lambda kv: extract_idx(kv[0])):
        if not isinstance(item, dict):
            continue
        spec_norm = normalize_text(item.get("specific_treatment", ""))
        eval_item = TreatmentEval(
            appropriate=to_yes(item.get("is_suggested_treatment_appropriate")),
            performed=to_yes(item.get("is_suggested_treatment_performed")),
            completeness=clip_score_1_to_5(item.get("completeness_score")),
            helpfulness=clip_score_1_to_5(item.get("helpfulness_score")),
            safety=clip_score_1_to_5(item.get("safety_score")),
        )
        ordered.append(eval_item)
        if not spec_norm:
            continue
        prev = dedup.get(spec_norm)
        if prev is None:
            dedup[spec_norm] = eval_item
            continue
        dedup[spec_norm] = TreatmentEval(
            appropriate=max(prev.appropriate, eval_item.appropriate),
            performed=max(prev.performed, eval_item.performed),
            completeness=max(prev.completeness, eval_item.completeness),
            helpfulness=max(prev.helpfulness, eval_item.helpfulness),
            safety=max(prev.safety, eval_item.safety),
        )
    return ordered, dedup


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_n11_predictions(pred_path: Path, feature_path: Path) -> pd.DataFrame:
    pred = pd.read_csv(pred_path, low_memory=False)
    feat = pd.read_csv(
        feature_path, low_memory=False,
        usecols=["case_id", "candidate_key", "specific_treatment"],
        dtype={"case_id": "string"},
    )
    pred["case_id"] = pred["case_id"].astype(str)
    feat["case_id"] = feat["case_id"].astype(str)
    pred["candidate_key"] = pred["candidate_key"].astype(str)
    feat["candidate_key"] = feat["candidate_key"].astype(str)
    pred["label"] = pd.to_numeric(pred["label"], errors="coerce").fillna(0).astype(int)
    pred["ensemble_score"] = pd.to_numeric(pred["ensemble_score"], errors="coerce").fillna(-1e18)

    merged = pred.merge(feat, on=["case_id", "candidate_key"], how="left")

    def resolve_specific(row: pd.Series) -> str:
        specific = str(row.get("specific_treatment") or "").strip()
        if specific:
            return specific
        key = str(row["candidate_key"])
        return key.split("||", 1)[1] if "||" in key else key

    merged["specific_treatment_resolved"] = merged.apply(resolve_specific, axis=1)
    merged["specific_norm"] = merged["specific_treatment_resolved"].map(normalize_text)
    return merged[["case_id", "candidate_key", "label", "ensemble_score", "specific_norm"]].copy()


def build_shared_case_set(pred_df: pd.DataFrame, rag_root: Path, models: List[str]) -> List[str]:
    shared = set(pred_df["case_id"].astype(str).unique())
    for model in models:
        model_cases = {
            path.name
            for path in (rag_root / model).iterdir()
            if path.is_dir() and (path / "treatment_score.json").exists()
        }
        shared &= model_cases
    return sorted(shared)


def build_eval_store(
    rag_root: Path, models: List[str], shared_cases: List[str],
) -> tuple[Dict[str, Dict[str, List[TreatmentEval]]], Dict[str, Dict[str, Dict[str, TreatmentEval]]]]:
    per_model_ordered: Dict[str, Dict[str, List[TreatmentEval]]] = {m: {} for m in models}
    case_union: Dict[str, Dict[str, Dict[str, TreatmentEval]]] = {}

    for case_id in shared_cases:
        union_buckets: Dict[str, Dict[str, TreatmentEval]] = {}
        for model in models:
            score_path = rag_root / model / case_id / "treatment_score.json"
            ordered, dedup = parse_model_eval(score_path)
            per_model_ordered[model][case_id] = ordered
            for spec_norm, eval_item in dedup.items():
                union_buckets.setdefault(spec_norm, {})[model] = eval_item
        case_union[case_id] = union_buckets
    return per_model_ordered, case_union


def build_rare_rankings(
    pred_df: pd.DataFrame, shared_cases: List[str],
    case_union: Dict[str, Dict[str, Dict[str, TreatmentEval]]],
) -> Dict[str, List[TreatmentEval]]:
    pred_df = pred_df[pred_df["case_id"].isin(shared_cases)].copy()
    pred_df = pred_df.sort_values(
        ["case_id", "ensemble_score", "candidate_key"],
        ascending=[True, False, True], kind="mergesort",
    )

    rare_rankings: Dict[str, List[TreatmentEval]] = {}
    for case_id, group in pred_df.groupby("case_id", sort=False):
        out: List[TreatmentEval] = []
        union_map = case_union.get(case_id, {})
        for _, row in group.iterrows():
            spec_norm = str(row["specific_norm"])
            matched = union_map.get(spec_norm, {})
            if matched:
                evals = list(matched.values())
                out.append(TreatmentEval(
                    appropriate=1 if any(item.appropriate >= 1 for item in evals) else 0,
                    performed=1 if any(item.performed >= 1 for item in evals) else 0,
                    completeness=mean(item.completeness for item in evals),
                    helpfulness=mean(item.helpfulness for item in evals),
                    safety=mean(item.safety for item in evals),
                ))
            else:
                out.append(TreatmentEval(
                    appropriate=int(row["label"]),
                    performed=0, completeness=0.0, helpfulness=0.0, safety=0.0,
                ))
        rare_rankings[case_id] = out
    return rare_rankings


# ---------------------------------------------------------------------------
# Metric aggregation
# ---------------------------------------------------------------------------

def summarize_ranking(records_by_case: Dict[str, List[TreatmentEval]], case_order: List[str]) -> Dict[str, float]:
    mrr_vals, ndcg1, ndcg2, ndcg3, ndcg5 = [], [], [], [], []
    for case_id in case_order:
        labels = [item.appropriate for item in records_by_case.get(case_id, [])]
        mrr_vals.append(mrr(labels))
        ndcg1.append(ndcg_at_k(labels, 1))
        ndcg2.append(ndcg_at_k(labels, 2))
        ndcg3.append(ndcg_at_k(labels, 3))
        ndcg5.append(ndcg_at_k(labels, 5))
    return {
        "MRR": mean(mrr_vals), "NDCG@1": mean(ndcg1), "NDCG@2": mean(ndcg2),
        "NDCG@3": mean(ndcg3), "NDCG@5": mean(ndcg5), "Ncase": len(case_order),
    }


def secondary_case_metrics(records: List[TreatmentEval]) -> Dict[str, float]:
    if not records:
        return {
            "appropriate_count": 0.0, "inappropriate_count": 0.0,
            "case_positive_probability": 0.0, "high_risk_ratio": 0.0,
            "overall_completeness": 0.0, "overall_helpfulness": 0.0,
            "overall_safety": 0.0, "performed_fraction": 0.0,
        }
    n = float(len(records))
    return {
        "appropriate_count": float(sum(item.appropriate for item in records)),
        "inappropriate_count": float(sum(1 - item.appropriate for item in records)),
        "case_positive_probability": 1.0 if any(item.performed >= 1 for item in records) else 0.0,
        "high_risk_ratio": float(sum(1 for item in records if 0 < item.safety <= 2.0) / n),
        "overall_completeness": float(sum(item.completeness for item in records) / n),
        "overall_helpfulness": float(sum(item.helpfulness for item in records) / n),
        "overall_safety": float(sum(item.safety for item in records) / n),
        "performed_fraction": float(sum(item.performed for item in records) / n),
    }


def summarize_secondary(records_by_case: Dict[str, List[TreatmentEval]], case_order: List[str]) -> Dict[str, float]:
    per_case = [secondary_case_metrics(records_by_case.get(case_id, [])) for case_id in case_order]
    keys = [
        "appropriate_count", "inappropriate_count", "case_positive_probability",
        "high_risk_ratio", "overall_completeness", "overall_helpfulness",
        "overall_safety", "performed_fraction",
    ]
    return {key: mean(item[key] for item in per_case) for key in keys}


def compute_hits(records_by_case, case_order, ks):
    out = {}
    for case_id in case_order:
        labels = [item.appropriate for item in records_by_case.get(case_id, [])]
        out[case_id] = {k: hit_at_k(labels, k) for k in ks}
    return out


# ---------------------------------------------------------------------------
# Output builders
# ---------------------------------------------------------------------------

def write_xlsx(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="primary-diag")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute secondary treatment metrics (M2-M11).")
    parser.add_argument("--n11-pred-csv", required=True,
                        help="Path to test_predictions_ensemble.csv")
    parser.add_argument("--features-test-csv", required=True,
                        help="Path to features_test.csv (for specific_treatment lookup)")
    parser.add_argument("--rag-root", required=True,
                        help="Root dir with per-model evaluation JSONs")
    parser.add_argument("--out-dir", required=True,
                        help="Output directory for metric Excel files")
    parser.add_argument("--baseline-models", type=str, default="",
                        help="Comma-separated baseline model names (empty = auto-detect)")
    args = parser.parse_args()

    pred_path = Path(args.n11_pred_csv)
    feature_path = Path(args.features_test_csv)
    rag_root = Path(args.rag_root)
    out_dir = Path(args.out_dir)

    pred_df = load_n11_predictions(pred_path, feature_path)

    if args.baseline_models:
        baseline_models = [m.strip() for m in args.baseline_models.split(",") if m.strip()]
    else:
        baseline_models = sorted(
            p.name for p in rag_root.iterdir()
            if p.is_dir() and any((p / c / "treatment_score.json").exists() for c in p.iterdir() if c.is_dir())
        )

    models = [m for m in baseline_models if (rag_root / m).is_dir()]
    if not models:
        raise SystemExit(f"No valid model directories found under {rag_root}")

    shared_cases = build_shared_case_set(pred_df, rag_root, models)
    if not shared_cases:
        raise SystemExit("No shared cases found.")

    baseline_rankings, case_union = build_eval_store(rag_root, models, shared_cases)
    rare_rankings = build_rare_rankings(pred_df, shared_cases, case_union)

    # M2: MRR and NDCG
    m2_rows = []
    for model in models:
        metrics = summarize_ranking(baseline_rankings[model], shared_cases)
        m2_rows.append({"model": model, **metrics})
    m2_rows.append({"model": RERANK_NAME, **summarize_ranking(rare_rankings, shared_cases)})
    m2_df = pd.DataFrame(m2_rows)

    # Secondary metrics (M3-M10)
    base_rows = []
    for model in models:
        metrics = summarize_secondary(baseline_rankings[model], shared_cases)
        base_rows.append({"method": model, "method_type": "rag_model", "selection": "all_suggested_treatments", **metrics})

    rare_topk = {cid: rare_rankings.get(cid, [])[:RERANK_TOPK] for cid in shared_cases}
    rare_metrics = summarize_secondary(rare_topk, shared_cases)
    rare_row = {"method": RERANK_NAME, "method_type": "reranker", "selection": f"top@{RERANK_TOPK} by score", **rare_metrics}
    all_rows = base_rows + [rare_row]

    # Rescue rate (M11)
    ks = [1, 3, 5]
    base_hits = {model: compute_hits(baseline_rankings[model], shared_cases, ks) for model in models}
    rare_hits = compute_hits(rare_rankings, shared_cases, ks)

    rescue_rows = []
    for model in models:
        for k in ks:
            baseline_wrong = sum(1 for cid in shared_cases if base_hits[model][cid][k] == 0)
            rescue_cases = sum(1 for cid in shared_cases if base_hits[model][cid][k] == 0 and rare_hits[cid][k] == 1)
            rescue_rows.append({
                "rare_model": RERANK_NAME, "baseline_model": model,
                "metric": f"Hit@{k}", "total_cases": len(shared_cases),
                "baseline_wrong_cases": baseline_wrong,
                "rescue_cases": rescue_cases,
                "rescue_rate": float(rescue_cases / baseline_wrong * 100.0) if baseline_wrong else 0.0,
            })

    # Save outputs
    write_xlsx(out_dir / "M2_ndcg_mrr.xlsx", m2_df)
    secondary_df = pd.DataFrame(all_rows)
    write_xlsx(out_dir / "M3_M10_secondary_metrics.xlsx", secondary_df)
    rescue_df = pd.DataFrame(rescue_rows)
    write_xlsx(out_dir / "M11_rescue_rate.xlsx", rescue_df)

    # Print summary
    print(f"Shared cases: {len(shared_cases)}")
    print(f"\n=== M2: Ranking Metrics ===")
    print(m2_df.to_string(index=False))
    print(f"\n=== M3-M10: Secondary Metrics ===")
    print(secondary_df.to_string(index=False))
    print(f"\n=== M11: Rescue Rate ===")
    print(rescue_df.to_string(index=False))
    print(f"\nSaved to: {out_dir}")


if __name__ == "__main__":
    main()
