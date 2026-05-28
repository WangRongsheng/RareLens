#!/usr/bin/env python3
"""
One-click training for treatment XGBoost ranker.

Reads llm_outputs.json files (format: {case_id, models: {model_name: {suggested_treatment_score: {...}}}})
from <data-dir>/<case_id>/5_treatment/llm_outputs.json, builds ranking features directly,
trains an XGBoost ranker with GroupKFold, saves per-fold models, and prints Hit@K metrics.

Default paths are pre-set for the data_demo layout so you can just run:
    python run_training.py

Or override:
    python run_training.py \\
        --data-dir D:/research/github/RareLens/data_demo/case_output \\
        --out-dir  D:/research/github/RareLens/rare_treatment/models_treat
"""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

try:
    import xgboost as xgb
except ImportError:
    raise SystemExit("xgboost is not installed. Run: pip install xgboost")

# ─── Text / numeric helpers ───────────────────────────────────────────────────

RE_DIGIT = re.compile(r"\d+")
RE_ALPHANUM = re.compile(r"[^a-z0-9\s]+")
YES_SET = {"yes", "y", "true", "1"}

CERTAINTY_WORDS = ["definitely", "confirms", "standard of care", "recommended",
                   "strong evidence", "guideline"]
UNCERTAINTY_WORDS = ["possible", "probable", "might", "could", "uncertain",
                     "investigational", "limited evidence", "not established"]


def normalize_text(s: str) -> str:
    if not s:
        return ""
    t = s.lower()
    t = RE_ALPHANUM.sub(" ", t)
    return " ".join(t.split())


def safe_float(x, default: float = 0.0) -> float:
    try:
        v = float(str(x).strip())
        return v if math.isfinite(v) else default
    except Exception:
        return default


def to_yes(v) -> int:
    return 1 if str(v).strip().lower() in YES_SET else 0


def extract_rank(key: str) -> int:
    """'treatment1' → 1, 'treatment12' → 12, unknown → 999."""
    m = RE_DIGIT.findall(str(key))
    return int(m[0]) if m else 999


def text_certainty(text: str) -> float:
    t = text.lower()
    c = sum(t.count(w) for w in CERTAINTY_WORDS)
    u = sum(t.count(w) for w in UNCERTAINTY_WORDS)
    return float(c - u)


def token_entropy(tokens: list[str]) -> float:
    if not tokens:
        return 0.0
    counts = Counter(tokens)
    total = sum(counts.values())
    return -sum((v / total) * math.log(v / total + 1e-12) for v in counts.values())


# ─── Parse llm_outputs.json ──────────────────────────────────────────────────

def parse_llm_outputs(path: Path) -> tuple[str, dict]:
    """
    Returns (case_id, model_data) where
    model_data = {model_name: {spec_norm: {rank, appropriate, performed,
                                           completeness, helpfulness, safety,
                                           rationale_text, specific_treatment}}}
    """
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)

    case_id = str(obj.get("case_id", path.parent.parent.name))
    models_raw = obj.get("models", {})
    if not isinstance(models_raw, dict):
        return case_id, {}

    result: dict = {}
    for model_name, model_data in models_raw.items():
        score_dict = model_data.get("suggested_treatment_score", {})
        if not isinstance(score_dict, dict):
            continue

        entries: dict = {}
        # Sort by key so treatment1 < treatment2 < ... → rank matches position
        for key, v in sorted(score_dict.items(), key=lambda kv: extract_rank(kv[0])):
            if not isinstance(v, dict):
                continue
            spec = str(v.get("specific_treatment", "")).strip()
            spec_norm = normalize_text(spec)
            if not spec_norm:
                continue

            explanations = " ".join([
                str(v.get("appropriateness_explanation", "")),
                str(v.get("completeness_explanation", "")),
                str(v.get("helpfulness_explanation", "")),
                str(v.get("safety_explanation", "")),
            ])

            comp = min(5.0, max(0.0, safe_float(v.get("completeness_score", 0))))
            help_ = min(5.0, max(0.0, safe_float(v.get("helpfulness_score", 0))))
            safe = min(5.0, max(0.0, safe_float(v.get("safety_score", 0))))

            entries[spec_norm] = {
                "specific_treatment": spec,
                "rank": extract_rank(key),
                "appropriate": float(to_yes(v.get("is_suggested_treatment_appropriate"))),
                "performed": float(to_yes(v.get("is_suggested_treatment_performed"))),
                "completeness": comp,
                "helpfulness": help_,
                "safety": safe,
                "rationale_text": explanations,
            }

        if entries:
            result[model_name] = entries

    return case_id, result


# ─── Build feature rows for one case ─────────────────────────────────────────

def build_case_features(
    case_id: str,
    model_data: dict,
    model_weights: dict,
    top_models: list[str],
    all_models: list[str],
) -> list[dict]:
    """Returns one feature-dict per candidate treatment."""

    # ── Pool unique candidates across all models ──
    pool: dict[str, dict] = {}
    for model_name, entries in model_data.items():
        for spec_norm, info in entries.items():
            if spec_norm not in pool:
                pool[spec_norm] = {
                    "specific_treatment": info["specific_treatment"],
                    "per_model": {},
                }
            elif len(info["specific_treatment"]) > len(pool[spec_norm]["specific_treatment"]):
                pool[spec_norm]["specific_treatment"] = info["specific_treatment"]
            pool[spec_norm]["per_model"][model_name] = info

    if not pool:
        return []

    # ── Positive label: appropriate == yes from any model ──
    pos_set = {
        sn for sn, cand in pool.items()
        if any(info["appropriate"] >= 1.0 for info in cand["per_model"].values())
    }

    total_model_weight = sum(model_weights.get(m, 0.0) for m in all_models)

    rows: list[dict] = []
    for spec_norm, cand in pool.items():
        pm = cand["per_model"]
        names = list(pm.keys())

        ranks = [pm[mn]["rank"] for mn in names]
        w_list = [float(model_weights.get(mn, 0.0)) for mn in names]
        total_w = sum(w_list)

        # Weighted rank score
        weighted_rank_score = sum(
            float(model_weights.get(mn, 0.0)) * (1.0 / (pm[mn]["rank"] + 0.5))
            for mn in names
        )

        # King (top-model) consensus
        king_names = [mn for mn in names if mn in top_models]
        king_votes = len(king_names)
        king_top1 = int(any(pm[mn]["rank"] == 1 for mn in king_names)) if king_names else 0
        kings_consensus = int(bool(top_models) and king_votes == len(top_models))

        appear = len(pm)
        agreement_ratio = appear / max(1, len(all_models))

        rank_arr = np.array(ranks, dtype=float)
        rank_std = float(np.std(rank_arr)) if len(ranks) > 1 else 0.0
        rank_min = float(rank_arr.min())
        rank_max = float(rank_arr.max())
        rank_mean = float(rank_arr.mean())
        rank_median = float(np.median(rank_arr))
        rank_top1_count = float((rank_arr <= 1).sum())
        rank_top3_count = float((rank_arr <= 3).sum())
        rank_top5_count = float((rank_arr <= 5).sum())
        w_mean_rank = (sum(float(model_weights.get(mn, 0.0)) * pm[mn]["rank"] for mn in names) / total_w
                       if total_w > 0 else rank_mean)

        support_weight_sum = float(sum(float(model_weights.get(mn, 0.0)) for mn in names))
        support_weight_mean = support_weight_sum / float(appear) if appear > 0 else 0.0
        support_weight_ratio = (support_weight_sum / total_model_weight) if total_model_weight > 0 else 0.0
        support_weight_max = float(max((float(model_weights.get(mn, 0.0)) for mn in names), default=0.0))
        topk_support_count = float(sum(1 for mn in names if mn in top_models))
        topk_support_ratio = (topk_support_count / float(len(top_models))) if top_models else 0.0

        # Rationale text features
        rat_texts = [pm[mn].get("rationale_text", "") for mn in names]
        rat_lens = [float(len(t.split())) for t in rat_texts]
        rat_certs = [text_certainty(t) for t in rat_texts]
        rationale_len_mean = float(np.mean(rat_lens)) if rat_lens else 0.0
        rationale_certainty_mean = float(np.mean(rat_certs)) if rat_certs else 0.0

        # Stage-3 eval aggregate features (from the score fields directly)
        evals = list(pm.values())
        eval_appr = [e["appropriate"] for e in evals]
        eval_perf = [e["performed"] for e in evals]
        eval_comp = [e["completeness"] for e in evals]
        eval_help = [e["helpfulness"] for e in evals]
        eval_safe = [e["safety"] for e in evals]
        eval_qual = [(c + h + s) / 3.0 for c, h, s in zip(eval_comp, eval_help, eval_safe)]

        eval_scored_models = float(len(evals))
        eval_coverage_ratio = eval_scored_models / max(1.0, float(appear))
        eval_appr_ratio = float(np.mean(eval_appr)) if evals else 0.0
        eval_perf_ratio = float(np.mean(eval_perf)) if evals else 0.0
        eval_perf_any = 1.0 if any(v >= 1.0 for v in eval_perf) else 0.0
        eval_comp_mean = float(np.mean(eval_comp)) if evals else 0.0
        eval_help_mean = float(np.mean(eval_help)) if evals else 0.0
        eval_safe_mean = float(np.mean(eval_safe)) if evals else 0.0
        eval_qual_mean = float(np.mean(eval_qual)) if evals else 0.0
        eval_qual_std = float(np.std(eval_qual)) if len(eval_qual) > 1 else 0.0

        row: dict = {
            "case_id": case_id,
            "candidate_key": spec_norm,
            "specific_treatment": cand["specific_treatment"],
            "label": 1 if spec_norm in pos_set else 0,
            # Consensus / rank stats
            "case_type_entropy": 0.0,           # no treatment_type in score files
            "pos_count_union": float(len(pos_set)),
            "weighted_rank_score": weighted_rank_score,
            "rank_std": rank_std,
            "appear_count": float(appear),
            "agreement_ratio": agreement_ratio,
            "kings_consensus": float(kings_consensus),
            "is_king_top1": float(king_top1),
            "mean_importance": 0.0,             # importance_score not in score files
            "weighted_mean_importance": 0.0,
            # Model support
            "model_support_weight_sum": support_weight_sum,
            "model_support_weight_ratio": support_weight_ratio,
            "model_support_weight_mean": support_weight_mean,
            "model_support_weight_max": support_weight_max,
            "topk_support_count": topk_support_count,
            "topk_support_ratio": topk_support_ratio,
            "rank_min": rank_min,
            "rank_max": rank_max,
            "rank_mean": rank_mean,
            "rank_median": rank_median,
            "rank_top1_count": rank_top1_count,
            "rank_top3_count": rank_top3_count,
            "rank_top5_count": rank_top5_count,
            "imp_min": 0.0,
            "imp_max": 0.0,
            "imp_std": 0.0,
            "w_mean_rank": w_mean_rank,
            # Semantic / NLI (no GPU/embedding model in demo → 0)
            "sem_sim_candidate": 0.0,
            "feat_nli_entailment": 0.0,
            "feat_soft_consensus": 0.0,
            # Rationale text
            "rationale_len_mean": rationale_len_mean,
            "rationale_certainty_mean": rationale_certainty_mean,
            # Stage-3 eval features (excluded from model inputs, used only as aux supervision)
            "feat_eval_scored_models": eval_scored_models,
            "feat_eval_coverage_ratio": eval_coverage_ratio,
            "feat_eval_appropriate_ratio": eval_appr_ratio,
            "feat_eval_performed_ratio": eval_perf_ratio,
            "feat_eval_performed_any": eval_perf_any,
            "feat_eval_completeness_mean": eval_comp_mean,
            "feat_eval_helpfulness_mean": eval_help_mean,
            "feat_eval_safety_mean": eval_safe_mean,
            "feat_eval_quality_mean": eval_qual_mean,
            "feat_eval_quality_std": eval_qual_std,
        }

        # Per-model rank / hit features (imp always 0 — not in score files)
        for mn in all_models:
            if mn in pm:
                row[f"rank__{mn}"] = float(pm[mn]["rank"])
                row[f"imp__{mn}"] = 0.0
                row[f"hit__{mn}"] = 1.0
            else:
                row[f"rank__{mn}"] = 999.0
                row[f"imp__{mn}"] = 0.0
                row[f"hit__{mn}"] = 0.0

        rows.append(row)

    return rows


# ─── XGBoost training ─────────────────────────────────────────────────────────

# Columns to exclude from model inputs
_EXCLUDE = {
    "case_id", "candidate_key", "specific_treatment", "label",
    # Stage-3 eval signals: supervision only, never model inputs (matches train_ranker.py design)
    "feat_eval_scored_models", "feat_eval_coverage_ratio",
    "feat_eval_appropriate_ratio", "feat_eval_performed_ratio",
    "feat_eval_performed_any", "feat_eval_completeness_mean",
    "feat_eval_helpfulness_mean", "feat_eval_safety_mean",
    "feat_eval_quality_mean", "feat_eval_quality_std",
}


def get_groups(df: pd.DataFrame) -> np.ndarray:
    return df.groupby("case_id", sort=False).size().to_numpy()


def train_xgb_fold(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    feat_cols: list[str],
) -> xgb.XGBRanker:
    """Train one XGBoost fold with the same hyperparameters as models_treat."""
    params = {
        "objective": "rank:ndcg",
        "learning_rate": 0.05,
        "n_estimators": 1000,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.9,
        "early_stopping_rounds": 50,
        "tree_method": "hist",
        "eval_metric": ["ndcg@3"],
        "device": "cpu",
    }
    model = xgb.XGBRanker(**params)
    model.fit(
        df_train[feat_cols],
        df_train["label"].to_numpy(dtype=np.int32),
        group=get_groups(df_train),
        eval_set=[(df_val[feat_cols], df_val["label"].to_numpy(dtype=np.int32))],
        eval_group=[get_groups(df_val)],
        verbose=False,
    )
    return model


# ─── Evaluation helpers ───────────────────────────────────────────────────────

def hit_at_k(labels: np.ndarray, k: int) -> float:
    return 1.0 if np.any(labels[:k] == 1) else 0.0


def ndcg_at_k(labels: np.ndarray, k: int) -> float:
    sel = labels[:k].astype(float)
    if sel.size == 0:
        return 0.0
    gains = sel
    discounts = 1.0 / np.log2(np.arange(2, gains.size + 2))
    dcg = float((gains * discounts).sum())
    ideal = np.sort(labels)[::-1][:k].astype(float)
    idcg = float((ideal * discounts[: ideal.size]).sum())
    return (dcg / idcg) if idcg > 0 else 0.0


def evaluate(df: pd.DataFrame, scores: np.ndarray, ks: tuple[int, ...] = (1, 3, 5)) -> dict:
    work = df[["case_id", "label"]].copy()
    work["score"] = scores
    metrics: dict[str, list[float]] = {f"hit@{k}": [] for k in ks}
    metrics.update({f"ndcg@{k}": [] for k in ks})
    for _, grp in work.groupby("case_id"):
        sorted_labels = grp.sort_values("score", ascending=False)["label"].to_numpy()
        for k in ks:
            metrics[f"hit@{k}"].append(hit_at_k(sorted_labels, k))
            metrics[f"ndcg@{k}"].append(ndcg_at_k(sorted_labels, k))
    return {m: float(np.mean(v)) for m, v in metrics.items()}


# ─── Main ────────────────────────────────────────────────────────────────────

_THIS = Path(__file__).resolve()
_DEFAULT_DATA = str(_THIS.parent.parent.parent / "data_demo" / "case_output")
_DEFAULT_OUT = str(_THIS.parent.parent / "models_treat")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="One-click training: llm_outputs.json → XGBoost ranker"
    )
    parser.add_argument(
        "--data-dir", default=_DEFAULT_DATA,
        help=f"Root of case_output dir (default: {_DEFAULT_DATA})"
    )
    parser.add_argument(
        "--out-dir", default=_DEFAULT_OUT,
        help=f"Output directory for models and CSVs (default: {_DEFAULT_OUT})"
    )
    parser.add_argument("--n-splits", type=int, default=5,
                        help="GroupKFold splits (auto-capped to n_cases)")
    parser.add_argument("--top-models", type=int, default=3,
                        help="Number of top models for king-consensus features")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Discover and parse llm_outputs.json files ──────────────────────────
    output_files = sorted(data_dir.glob("*/5_treatment/llm_outputs.json"))
    if not output_files:
        raise SystemExit(
            f"No llm_outputs.json files found under {data_dir}/*/5_treatment/\n"
            f"Expected path: <data-dir>/<case_id>/5_treatment/llm_outputs.json"
        )

    print(f"Found {len(output_files)} llm_outputs.json file(s)")

    all_case_data: dict[str, dict] = {}
    for f in output_files:
        case_id, model_data = parse_llm_outputs(f)
        if model_data:
            all_case_data[case_id] = model_data
        else:
            print(f"  [WARN] No model data in {f}, skipping")

    if not all_case_data:
        raise SystemExit("No valid case data found. Check JSON format.")

    print(f"Parsed {len(all_case_data)} case(s) successfully")

    # ── 2. Determine active models and compute coverage weights ───────────────
    all_models_set: set[str] = set()
    for md in all_case_data.values():
        all_models_set.update(md.keys())
    all_models = sorted(all_models_set)

    n_cases = len(all_case_data)
    model_weights = {
        m: sum(1 for md in all_case_data.values() if m in md) / n_cases
        for m in all_models
    }
    top_models = sorted(all_models, key=lambda m: -model_weights[m])[: args.top_models]

    print(f"\nActive models ({len(all_models)}):")
    for m in all_models:
        mark = "*" if m in top_models else " "
        print(f"  {mark} {m}  (coverage={model_weights[m]:.2f})")
    print(f"  (* = top-{args.top_models} models used for king-consensus)")

    # ── 3. Build feature rows ─────────────────────────────────────────────────
    all_rows: list[dict] = []
    for case_id, model_data in all_case_data.items():
        rows = build_case_features(case_id, model_data, model_weights, top_models, all_models)
        all_rows.extend(rows)
        pos = sum(r["label"] for r in rows)
        print(f"  case {case_id}: {len(rows)} candidates, {pos} positive")

    total_pos = sum(r["label"] for r in all_rows)
    print(f"\nTotal rows: {len(all_rows)} | positive: {total_pos} ({100*total_pos/len(all_rows):.1f}%)")

    # ── 4. Build DataFrame, save features CSV ─────────────────────────────────
    df = (pd.DataFrame(all_rows)
          .sort_values("case_id")
          .reset_index(drop=True))
    feat_csv = out_dir / "features.csv"
    df.to_csv(feat_csv, index=False)
    print(f"Features saved → {feat_csv}")

    feat_cols = [c for c in df.columns if c not in _EXCLUDE]
    print(f"Feature columns used for training: {len(feat_cols)}")

    # ── 5. Train XGBoost with GroupKFold ─────────────────────────────────────
    n_unique = df["case_id"].nunique()
    n_splits = min(args.n_splits, n_unique)
    if n_splits < 2:
        raise SystemExit(
            f"Need at least 2 cases for cross-validation, found {n_unique}."
        )
    print(f"\nTraining XGBoost: {n_splits}-fold GroupKFold on {n_unique} cases")

    gkf = GroupKFold(n_splits=n_splits)
    models_dir = out_dir / "models"
    models_dir.mkdir(exist_ok=True)

    fold_models: list[xgb.XGBRanker] = []
    test_pred_sum = np.zeros(len(df), dtype=float)

    for fold, (train_idx, val_idx) in enumerate(
        gkf.split(df, df["label"], df["case_id"])
    ):
        df_train = df.iloc[train_idx].copy().sort_values("case_id")
        df_val = df.iloc[val_idx].copy().sort_values("case_id")

        train_cases = df_train["case_id"].nunique()
        val_cases = df_val["case_id"].nunique()
        val_pos = df_val["label"].sum()
        print(f"  Fold {fold+1}: train={train_cases} cases, val={val_cases} cases "
              f"({val_pos} positives in val)")

        model = train_xgb_fold(df_train, df_val, feat_cols)
        fold_models.append(model)

        # Accumulate ensemble predictions
        test_pred_sum += model.predict(df[feat_cols])

        model_path = models_dir / f"model_fold{fold+1}.json"
        model.save_model(model_path)
        print(f"    Saved → {model_path}")

    # ── 6. Evaluate ensemble on full dataset ──────────────────────────────────
    ensemble_scores = test_pred_sum / n_splits
    metrics = evaluate(df, ensemble_scores)

    print("\n" + "=" * 55)
    print(f"  Ensemble Evaluation ({n_splits}-fold, {n_unique} cases)")
    print("=" * 55)
    for k in (1, 3, 5):
        h = metrics.get(f"hit@{k}", 0.0)
        n = metrics.get(f"ndcg@{k}", 0.0)
        print(f"  Hit@{k} = {h:.4f}   NDCG@{k} = {n:.4f}")
    print("=" * 55)
    print(f"\nModels saved → {models_dir}")
    print("Done.")


if __name__ == "__main__":
    main()
