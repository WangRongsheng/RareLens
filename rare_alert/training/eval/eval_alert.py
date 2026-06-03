"""
RareAlert evaluation script.

Inputs : pipeline output directory + ground-truth case directories (rare / non-rare)
Outputs: AUC, Accuracy (fixed threshold + optimal Youden/F1/F2), Sensitivity,
         Specificity, Balanced Accuracy, F1, F2, MCC, PPV, NPV, TP/FP/FN/TN

Accuracy calculation follows the reference implementation in
eval_gt/evaluation_code/risk/diagnosis_with_TPFPFNTN.py:
  - AUC         : sklearn roc_auc_score  (equivalent to roc_curve + auc)
  - Acc (fixed) : (TP+TN)/total at a caller-specified threshold
  - Acc (opt.)  : three auto-search methods — Youden / F1 / F2

Standalone usage:
    python eval_alert.py \\
        --output-root /data/pipeline_output \\
        --rare-dir    /data/gt/rare \\
        --nonrare-dir /data/gt/nonrare \\
        [--threshold 30] \\
        [--out-json   /data/results/alert_metrics.json]
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _load_risk_output(
    output_root: Path,
    case_id: str,
    *,
    filenames: Tuple[str, ...],
) -> Optional[float]:
    """Return risk_score from the pipeline output JSON for a given case, or None on failure."""
    for name in filenames:
        candidate = output_root / case_id / name
        if candidate.exists():
            try:
                with open(candidate, encoding="utf-8") as f:
                    d = json.load(f)
                score = d.get("risk_score")
                return float(score) if score is not None else None
            except Exception:
                return None
    return None


def _collect_cases(
    output_root: Path,
    rare_dir: Path,
    nonrare_dir: Path,
    *,
    filenames: Tuple[str, ...],
) -> List[Tuple[str, float, int]]:
    """
    Walk rare_dir and nonrare_dir to collect (case_id, risk_score, true_label) triples.
    true_label: 1 = rare disease, 0 = non-rare.
    """
    records: List[Tuple[str, float, int]] = []
    for label, directory in ((1, rare_dir), (0, nonrare_dir)):
        for case_dir in sorted(directory.iterdir()):
            if not case_dir.is_dir():
                continue
            score = _load_risk_output(output_root, case_dir.name, filenames=filenames)
            if score is not None:
                records.append((case_dir.name, score, label))
    return records


# ---------------------------------------------------------------------------
# Threshold search
# ---------------------------------------------------------------------------

def _find_optimal_threshold_youden(fpr_arr, tpr_arr, thresholds_arr) -> float:
    """Return the threshold that maximises Youden's index (TPR - FPR)."""
    import numpy as np
    idx = int(np.argmax(np.array(tpr_arr) - np.array(fpr_arr)))
    return float(thresholds_arr[idx])


def _find_optimal_threshold_f1(y_true: List[int], y_scores: List[float]) -> float:
    """Return the threshold that maximises F1 score (grid search over 99 candidates)."""
    import numpy as np
    best_f1, best_t = -1.0, 0.5
    for t in np.linspace(0, 1, 101)[1:-1]:
        preds = [1 if s >= t else 0 for s in y_scores]
        tp = sum(1 for a, b in zip(y_true, preds) if a == 1 and b == 1)
        fp = sum(1 for a, b in zip(y_true, preds) if a == 0 and b == 1)
        fn = sum(1 for a, b in zip(y_true, preds) if a == 1 and b == 0)
        denom = 2 * tp + fp + fn
        f1 = (2 * tp / denom) if denom > 0 else 0.0
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t


def _find_optimal_threshold_f2(y_true: List[int], y_scores: List[float]) -> float:
    """Return the threshold that maximises F2 score (beta=2, recall-weighted)."""
    import numpy as np
    best_f2, best_t = -1.0, 0.5
    for t in np.linspace(0, 1, 101)[1:-1]:
        preds = [1 if s >= t else 0 for s in y_scores]
        tp = sum(1 for a, b in zip(y_true, preds) if a == 1 and b == 1)
        fp = sum(1 for a, b in zip(y_true, preds) if a == 0 and b == 1)
        fn = sum(1 for a, b in zip(y_true, preds) if a == 1 and b == 0)
        denom = 5 * tp + fp + 4 * fn
        f2 = (5 * tp / denom) if denom > 0 else 0.0
        if f2 > best_f2:
            best_f2, best_t = f2, float(t)
    return best_t


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def _calc_metrics_at_threshold(
    y_true: List[int],
    y_scores: List[float],
    threshold: float,
) -> Dict:
    """Compute the full metric suite at a given classification threshold."""
    preds = [1 if s >= threshold else 0 for s in y_scores]
    tp = sum(1 for a, b in zip(y_true, preds) if a == 1 and b == 1)
    tn = sum(1 for a, b in zip(y_true, preds) if a == 0 and b == 0)
    fp = sum(1 for a, b in zip(y_true, preds) if a == 0 and b == 1)
    fn = sum(1 for a, b in zip(y_true, preds) if a == 1 and b == 0)

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv         = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv         = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    accuracy    = (tp + tn) / len(y_true) if y_true else 0.0
    balanced_acc = (sensitivity + specificity) / 2

    precision, recall = ppv, sensitivity
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    f2_denom = 4 * precision + recall
    f2 = (5 * precision * recall / f2_denom) if f2_denom > 0 else 0.0

    denom_sq = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    mcc = (tp * tn - fp * fn) / math.sqrt(denom_sq) if denom_sq > 0 else 0.0

    return dict(
        threshold=threshold,
        tp=tp, tn=tn, fp=fp, fn=fn,
        sensitivity=round(sensitivity, 4),
        specificity=round(specificity, 4),
        ppv=round(ppv, 4),
        npv=round(npv, 4),
        accuracy=round(accuracy, 4),
        balanced_accuracy=round(balanced_acc, 4),
        f1=round(f1, 4),
        f2=round(f2, 4),
        mcc=round(mcc, 4),
        fnr=round(fn / (tp + fn), 4) if (tp + fn) > 0 else 0.0,
        fpr=round(fp / (tn + fp), 4) if (tn + fp) > 0 else 0.0,
    )


# ---------------------------------------------------------------------------
# Main evaluation function
# ---------------------------------------------------------------------------

def evaluate_alert(
    output_root: str | Path,
    rare_dir: str | Path,
    nonrare_dir: str | Path,
    threshold: float = 30.0,
    risk_output_filenames: Tuple[str, ...] = ("risk_output.json", "riskoutput.json"),
) -> Dict:
    """
    Evaluate the RareAlert risk module.

    Returns a dict containing:
      - fixed_threshold  : full metrics at the specified threshold
      - optimal_youden   : full metrics at the Youden-optimal threshold
      - optimal_f1       : full metrics at the F1-optimal threshold
      - optimal_f2       : full metrics at the F2-optimal threshold
      - auc              : ROC AUC
      - details          : per-case predictions (at fixed threshold)
      - top-level shorthands: acc, sensitivity, specificity, tp, tn, fp, fn

    Parameters
    ----------
    output_root : Pipeline output root — expects <output_root>/<case_id>/risk_output.json
    rare_dir    : Ground-truth rare-disease case directory
    nonrare_dir : Ground-truth non-rare case directory
    threshold   : Fixed risk score threshold for binary classification (default 30)
    """
    output_root = Path(output_root)
    rare_dir    = Path(rare_dir)
    nonrare_dir = Path(nonrare_dir)

    records = _collect_cases(output_root, rare_dir, nonrare_dir, filenames=risk_output_filenames)
    if not records:
        return {"error": "No risk_output.json files found — run the pipeline first."}

    case_ids = [r[0] for r in records]
    y_scores = [r[1] for r in records]
    y_true   = [r[2] for r in records]

    # ROC AUC
    try:
        from sklearn.metrics import roc_auc_score, roc_curve
        auc_val = round(float(roc_auc_score(y_true, y_scores)), 4)
        fpr_arr, tpr_arr, thresh_arr = roc_curve(y_true, y_scores)
    except Exception:
        auc_val = None
        fpr_arr = tpr_arr = thresh_arr = []

    # Optimal thresholds
    opt_youden = _find_optimal_threshold_youden(fpr_arr, tpr_arr, thresh_arr) if auc_val is not None else threshold
    opt_f1     = _find_optimal_threshold_f1(y_true, y_scores)
    opt_f2     = _find_optimal_threshold_f2(y_true, y_scores)

    n_rare    = sum(y_true)
    n_nonrare = len(y_true) - n_rare

    # Per-case details at fixed threshold
    fixed_preds = [1 if s >= threshold else 0 for s in y_scores]
    details = [
        {
            "case_id":    cid,
            "risk_score": score,
            "true_label": label,
            "pred_label": pred,
            "correct":    label == pred,
        }
        for cid, score, label, pred in zip(case_ids, y_scores, y_true, fixed_preds)
    ]
    correct_rare_case_ids = [d["case_id"] for d in details if d["true_label"] == 1 and d["pred_label"] == 1]

    return {
        "n_total":   len(records),
        "n_rare":    n_rare,
        "n_nonrare": n_nonrare,
        "n_found":   len(records),
        "auc": auc_val,
        "fixed_threshold": _calc_metrics_at_threshold(y_true, y_scores, threshold),
        "optimal_youden":  _calc_metrics_at_threshold(y_true, y_scores, opt_youden),
        "optimal_f1":      _calc_metrics_at_threshold(y_true, y_scores, opt_f1),
        "optimal_f2":      _calc_metrics_at_threshold(y_true, y_scores, opt_f2),
        # Top-level shorthands for backward-compatible callers
        "threshold":   threshold,
        "acc":         round(sum(1 for t, p in zip(y_true, fixed_preds) if t == p) / len(y_true), 4),
        "sensitivity": round(sum(1 for t, p in zip(y_true, fixed_preds) if t == 1 and p == 1) / max(n_rare, 1), 4),
        "specificity": round(sum(1 for t, p in zip(y_true, fixed_preds) if t == 0 and p == 0) / max(n_nonrare, 1), 4),
        "tp": sum(1 for t, p in zip(y_true, fixed_preds) if t == 1 and p == 1),
        "tn": sum(1 for t, p in zip(y_true, fixed_preds) if t == 0 and p == 0),
        "fp": sum(1 for t, p in zip(y_true, fixed_preds) if t == 0 and p == 1),
        "fn": sum(1 for t, p in zip(y_true, fixed_preds) if t == 1 and p == 0),
        "correct_rare_n":        len(correct_rare_case_ids),
        "correct_rare_case_ids": correct_rare_case_ids,
        "details": details,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _print_results(results: Dict) -> None:
    """Pretty-print evaluation results to stdout."""
    if "error" in results:
        print(f"[ERROR] {results['error']}")
        return

    print(f"\n{'='*60}")
    print(f"  RareAlert Evaluation Results")
    print(f"{'='*60}")
    print(f"  Total cases : {results['n_total']}  "
          f"(rare={results['n_rare']}, non-rare={results['n_nonrare']})")
    print(f"  AUC         : {results['auc']}")
    print()

    for key, label in (
        ("fixed_threshold", f"Fixed threshold = {results['threshold']}"),
        ("optimal_youden",  "Optimal threshold (Youden)"),
        ("optimal_f1",      "Optimal threshold (F1)"),
        ("optimal_f2",      "Optimal threshold (F2)"),
    ):
        m = results[key]
        print(f"  [{label}]  threshold={m['threshold']}")
        print(f"    Acc={m['accuracy']}  BalancedAcc={m['balanced_accuracy']}")
        print(f"    Sens={m['sensitivity']}  Spec={m['specificity']}  "
              f"PPV={m['ppv']}  NPV={m['npv']}")
        print(f"    F1={m['f1']}  F2={m['f2']}  MCC={m['mcc']}")
        print(f"    TP={m['tp']}  TN={m['tn']}  FP={m['fp']}  FN={m['fn']}")
        print()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate the RareAlert risk module.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output-root", required=True,
        help="Pipeline output root directory (contains <case_id>/risk_output.json).",
    )
    parser.add_argument(
        "--rare-dir", required=True,
        help="Ground-truth rare-disease case directory.",
    )
    parser.add_argument(
        "--nonrare-dir", required=True,
        help="Ground-truth non-rare case directory.",
    )
    parser.add_argument(
        "--threshold", type=float, default=30.0,
        help="Fixed risk score threshold for binary classification.",
    )
    parser.add_argument(
        "--out-json", default=None,
        help="Optional path to save full results as JSON.",
    )
    args = parser.parse_args()

    results = evaluate_alert(
        output_root=args.output_root,
        rare_dir=args.rare_dir,
        nonrare_dir=args.nonrare_dir,
        threshold=args.threshold,
    )

    _print_results(results)

    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # Exclude per-case details from saved JSON to keep file size small
        save = {k: v for k, v in results.items() if k != "details"}
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(save, f, ensure_ascii=False, indent=2)
        print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
