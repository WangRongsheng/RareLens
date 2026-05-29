"""
Train prognosis stacking models (GBDT) from feature CSVs built by build_features.py.

Training uses 5-fold StratifiedKFold OOF (out-of-fold) for evaluation,
then fits a final model on ALL training data for the deployment pkl bundle.

Best parameters (fixed, no search):
  overall_outcome   -> GradientBoostingClassifier(random_state=seed)
  functional_status -> GradientBoostingClassifier(random_state=seed)
  symptom_burden    -> GradientBoostingClassifier(random_state=seed)

Usage:
    python -m rare_prognosis.training.train_models \\
        --features-dir rare_prognosis/training/features \\
        --out-dir rare_prognosis/models \\
        --seed 42
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.model_selection import StratifiedKFold

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

TASK_MODEL_KIND = {
    "overall_outcome": "gbdt",
    "functional_status": "gbdt",
    "symptom_burden": "gbdt",
}

TASK_BUNDLE_NAME = {
    "overall_outcome": "overall_outcome_C2_stacking_gbdt.pkl",
    "functional_status": "functional_status_C2_stacking_gbdt.pkl",
    "symptom_burden": "symptom_burden_C2_stacking_gbdt.pkl",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_feature_csv(path: Path) -> Tuple[List[str], List[str], List[List[float]]]:
    """Returns (case_ids, gt_labels, X)."""
    case_ids: List[str] = []
    gt_labels: List[str] = []
    X: List[List[float]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        feat_cols = [c for c in (reader.fieldnames or []) if c not in ("case_id", "gt_label")]
        for row in reader:
            case_ids.append(row["case_id"])
            gt_labels.append(row["gt_label"])
            X.append([float(row[c]) for c in feat_cols])
    return case_ids, gt_labels, X


# ---------------------------------------------------------------------------
# OOF CV prediction (aligned with root-level train scripts)
# ---------------------------------------------------------------------------

def _cv_predict(
    X_train: np.ndarray,
    y_train: List[str],
    X_test: np.ndarray | None,
    n_splits: int,
    seed: int,
    model_factory,
) -> Tuple[List[str], List[str], List[Any], List[str]]:
    """5-fold StratifiedKFold OOF for train, averaged proba for test.

    Returns (oof_pred, test_pred, fold_models, class_list).
    """
    class_list = sorted(set(y_train))
    class_index = {c: i for i, c in enumerate(class_list)}
    n_classes = len(class_list)

    counts = Counter(y_train)
    min_count = min(counts.values()) if counts else 0
    n_splits = min(n_splits, min_count) if min_count else 0

    if n_splits < 2:
        clf = model_factory()
        clf.fit(X_train, y_train)
        train_pred = [str(p) for p in clf.predict(X_train)]
        test_pred = [str(p) for p in clf.predict(X_test)] if X_test is not None and len(X_test) else []
        return train_pred, test_pred, [clf], class_list

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof_proba = np.zeros((len(y_train), n_classes), dtype=float)
    test_proba_sum = np.zeros((len(X_test), n_classes), dtype=float) if X_test is not None and len(X_test) else None
    fold_models: List[Any] = []

    for train_idx, val_idx in skf.split(X_train, y_train):
        clf = model_factory()
        clf.fit(X_train[train_idx], [y_train[i] for i in train_idx])
        fold_models.append(clf)

        val_proba = clf.predict_proba(X_train[val_idx])
        aligned_val = np.zeros((len(val_idx), n_classes), dtype=float)
        for i, cls in enumerate(clf.classes_):
            aligned_val[:, class_index[cls]] = val_proba[:, i]
        oof_proba[val_idx] = aligned_val

        if test_proba_sum is not None:
            test_proba = clf.predict_proba(X_test)
            aligned_test = np.zeros((len(X_test), n_classes), dtype=float)
            for i, cls in enumerate(clf.classes_):
                aligned_test[:, class_index[cls]] = test_proba[:, i]
            test_proba_sum += aligned_test

    oof_pred = [class_list[int(i)] for i in np.argmax(oof_proba, axis=1)]

    test_pred: List[str] = []
    if test_proba_sum is not None:
        test_avg = test_proba_sum / float(n_splits)
        test_pred = [class_list[int(i)] for i in np.argmax(test_avg, axis=1)]

    return oof_pred, test_pred, fold_models, class_list


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

def _make_gbdt_factory(seed: int):
    from sklearn.ensemble import GradientBoostingClassifier
    return lambda: GradientBoostingClassifier(random_state=seed)


def _write_bundle(path: Path, model: Any, meta: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump({"model": model, "meta": meta}, f, protocol=pickle.HIGHEST_PROTOCOL)


# ---------------------------------------------------------------------------
# Per-task training
# ---------------------------------------------------------------------------

def train_task(
    *,
    task: str,
    features_dir: Path,
    out_dir: Path,
    seed: int,
    cv_folds: int,
) -> None:
    task_dir = features_dir / task
    train_csv = task_dir / "features.train.csv"
    test_csv = task_dir / "features.test.csv"
    meta_json = task_dir / "meta.json"

    if not train_csv.is_file():
        raise SystemExit(f"[{task}] missing feature CSV: {train_csv} -- run build_features.py first")
    if not meta_json.is_file():
        raise SystemExit(f"[{task}] missing meta.json: {meta_json}")

    meta = json.loads(meta_json.read_text(encoding="utf-8"))
    train_ids, y_train, X_train_list = _load_feature_csv(train_csv)

    if not X_train_list:
        raise SystemExit(f"[{task}] empty training data")

    X_train = np.asarray(X_train_list)

    # Load test set if available
    X_test = None
    test_ids: List[str] = []
    y_test: List[str] = []
    if test_csv.is_file():
        test_ids, y_test, X_test_list = _load_feature_csv(test_csv)
        if X_test_list:
            X_test = np.asarray(X_test_list)

    model_factory = _make_gbdt_factory(seed)
    print(f"[{task}] train={len(y_train)} test={len(test_ids)} features={X_train.shape[1]}")

    # --- OOF training: save all fold models ---
    oof_pred, test_pred, fold_models, class_list = _cv_predict(
        X_train, y_train, X_test, n_splits=cv_folds, seed=seed,
        model_factory=model_factory,
    )

    # OOF train accuracy
    oof_correct = sum(1 for p, g in zip(oof_pred, y_train) if p == g)
    oof_acc = oof_correct / len(y_train)
    print(f"  OOF train acc: {oof_acc:.6f} ({oof_correct}/{len(y_train)})")

    # Test accuracy (if available)
    if test_pred and y_test:
        test_correct = sum(1 for p, g in zip(test_pred, y_test) if p == g)
        test_acc = test_correct / len(y_test)
        print(f"  Test acc:      {test_acc:.6f} ({test_correct}/{len(y_test)})")

    print(f"  Saving {len(fold_models)} fold model(s)")

    from datetime import datetime
    bundle_meta = {
        "base_model_names": meta["base_model_names"],
        "labels": meta["labels"],
        "expl_keywords": meta["expl_keywords"],
        "expl_standardize": meta["expl_standardize"],
        "class_list": class_list,
        "model_type": type(fold_models[0]).__name__,
        "trained_with": "train_models.py",
        "cv_n_splits": cv_folds,
        "timestamp": datetime.now().isoformat(),
    }
    out_path = out_dir / TASK_BUNDLE_NAME[task]
    _write_bundle(out_path, model=fold_models, meta=bundle_meta)
    print(f"[OK] {task} -> {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Train prognosis stacking models from feature CSVs.")
    p.add_argument("--features-dir", default="rare_prognosis/training/features",
                   help="Output of build_features.py")
    p.add_argument("--out-dir", default="rare_prognosis/models")
    p.add_argument("--task", choices=("overall_outcome", "functional_status", "symptom_burden", "all"), default="all")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cv-folds", type=int, default=5)
    args = p.parse_args()

    tasks = list(TASK_MODEL_KIND) if args.task == "all" else [args.task]
    for task in tasks:
        train_task(
            task=task,
            features_dir=Path(args.features_dir),
            out_dir=Path(args.out_dir),
            seed=args.seed,
            cv_folds=args.cv_folds,
        )


if __name__ == "__main__":
    main()
