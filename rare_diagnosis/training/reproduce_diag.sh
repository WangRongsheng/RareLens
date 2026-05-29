#!/usr/bin/env bash
#
# One-click reproduction script for RareDiagnosis XGBoost ranking pipeline.
#
# Pipeline steps:
#   Step 1: Build features  (build_features_primary / build_features_followup)
#   Step 2: Train ranker    (train_ranker — XGBoost GroupKFold)
#   Step 3: Evaluate ML     (eval.eval_ml)
#   Step 4: Evaluate LLMs   (eval.eval_llm)
#
# Usage:
#   bash rare_diagnosis/training/reproduce_diag.sh                        # primary stage
#   VISIT_TYPE=followup bash rare_diagnosis/training/reproduce_diag.sh    # followup stage
#
# Requirements: Python with xgboost, scikit-learn, pandas, numpy, sentence-transformers
#
set -euo pipefail

# Use pwd -W on MSYS/Git-Bash (Windows) so Python receives valid paths
_pwd() { pwd -W 2>/dev/null || pwd; }
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && _pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && _pwd)"

# ── Python interpreter ───────────────────────────────────────────────────
PYTHON="${PYTHON:-python}"

# ── Defaults ─────────────────────────────────────────────────────────────
VISIT_TYPE="${VISIT_TYPE:-primary}"

# ── Parse arguments ──────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --visit-type)       VISIT_TYPE="$2";        shift 2 ;;
        --query-root)       QUERY_ROOT="$2";        shift 2 ;;
        --gt-root)          GT_ROOT="$2";           shift 2 ;;
        --models-root)      MODELS_ROOT="$2";       shift 2 ;;
        --train-ids)        TRAIN_IDS="$2";         shift 2 ;;
        --test-ids)         TEST_IDS="$2";          shift 2 ;;
        --score-root)       SCORE_ROOT="$2";        shift 2 ;;
        --work-dir)         WORK_DIR="$2";          shift 2 ;;
        --out-dir)          OUT_DIR="$2";           shift 2 ;;
        --python)           PYTHON="$2";            shift 2 ;;
        --no-gpu)           USE_GPU="";             shift ;;
        -h|--help)
            echo "Usage: bash reproduce_diag.sh [OPTIONS]"
            echo ""
            echo "  --visit-type TYPE     primary (default) or followup"
            echo "  --query-root DIR      Root of query data (primary_consultation.json)"
            echo "  --gt-root DIR         Root of ground-truth scores"
            echo "  --models-root DIR     Root of per-model diagnosis outputs"
            echo "  --train-ids PATH      JSON list of train case IDs"
            echo "  --test-ids PATH       JSON list of test case IDs"
            echo "  --score-root DIR      Root of evaluation scores (for LLM eval)"
            echo "  --work-dir DIR        Feature output directory"
            echo "  --out-dir DIR         Model/eval output directory"
            echo "  --python PATH         Python interpreter (default: python)"
            echo "  --no-gpu              Disable GPU training"
            exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ── Input paths (defaults) ───────────────────────────────────────────────
QUERY_ROOT="${QUERY_ROOT:-${REPO_ROOT}/dataset/diagnosis/query_data}"
GT_ROOT="${GT_ROOT:-${REPO_ROOT}/dataset/diagnosis/diagnosis_score}"
MODELS_ROOT="${MODELS_ROOT:-${REPO_ROOT}/dataset/diagnosis/diagnosis_output}"
TRAIN_IDS="${TRAIN_IDS:-${REPO_ROOT}/dataset/diagnosis/splits/train.json}"
TEST_IDS="${TEST_IDS:-${REPO_ROOT}/dataset/diagnosis/splits/test.json}"
SCORE_ROOT="${SCORE_ROOT:-${GT_ROOT}}"

# ── Config (stage-specific hyperparameters) ──────────────────────────────
if [[ "${VISIT_TYPE}" == "primary" ]]; then
    CONFIG_PATH="${REPO_ROOT}/rare_diagnosis/training/best_hyperopt_config_primary.json"
elif [[ "${VISIT_TYPE}" == "followup" ]]; then
    CONFIG_PATH="${REPO_ROOT}/rare_diagnosis/training/best_hyperopt_config_followup.json"
else
    echo "[ERROR] VISIT_TYPE must be primary or followup, got: ${VISIT_TYPE}" >&2
    exit 1
fi

# ── Output paths ─────────────────────────────────────────────────────────
WORK_DIR="${WORK_DIR:-${REPO_ROOT}/outputs/diagnosis_features_${VISIT_TYPE}}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/outputs/diagnosis_model_${VISIT_TYPE}}"
EVAL_DIR="${OUT_DIR}/eval_results"
USE_GPU="${USE_GPU:-"--use-gpu"}"

mkdir -p "${WORK_DIR}" "${OUT_DIR}" "${EVAL_DIR}"

echo "============================================================"
echo "RareDiagnosis Pipeline Reproduction"
echo "============================================================"
echo "  visit_type:   ${VISIT_TYPE}"
echo "  query_root:   ${QUERY_ROOT}"
echo "  gt_root:      ${GT_ROOT}"
echo "  models_root:  ${MODELS_ROOT}"
echo "  train_ids:    ${TRAIN_IDS}"
echo "  test_ids:     ${TEST_IDS}"
echo "  config:       ${CONFIG_PATH}"
echo "  work_dir:     ${WORK_DIR}"
echo "  out_dir:      ${OUT_DIR}"
echo "  python:       ${PYTHON}"
echo "============================================================"
echo ""

# ── Step 1: Build features ───────────────────────────────────────────────
echo "[Step 1/4] Building features (${VISIT_TYPE})..."
if [[ "${VISIT_TYPE}" == "primary" ]]; then
    "${PYTHON}" -m rare_diagnosis.training.build_features_primary \
        --query_root "${QUERY_ROOT}" \
        --out_dir "${WORK_DIR}" \
        --gt_root "${GT_ROOT}" \
        --primary_models_root "${MODELS_ROOT}" \
        --train_ids "${TRAIN_IDS}" \
        --test_ids "${TEST_IDS}" \
        --primary_fname "most_likely_diagnosis_orphacode.json" \
        --gt_fname "primary_diagnosis_score.json"
else
    "${PYTHON}" -m rare_diagnosis.training.build_features_followup \
        --query_root "${QUERY_ROOT}" \
        --out_dir "${WORK_DIR}" \
        --gt_root "${GT_ROOT}" \
        --primary_models_root "${MODELS_ROOT}" \
        --train_ids "${TRAIN_IDS}" \
        --test_ids "${TEST_IDS}" \
        --primary_fname "most_likely_diagnosis_orphacode.json" \
        --gt_fname "primary_diagnosis_score.json"
fi
echo ""

# ── Step 2: Train XGBoost ranker ─────────────────────────────────────────
echo "[Step 2/4] Training XGBoost ranker..."
"${PYTHON}" -m rare_diagnosis.training.train_ranker \
    --input-dir "${WORK_DIR}" \
    --config "${CONFIG_PATH}" \
    --out-dir "${OUT_DIR}" \
    ${USE_GPU}
echo ""

# ── Step 3: Evaluate ML ranking ─────────────────────────────────────────
echo "[Step 3/4] Evaluating ML ranking..."
"${PYTHON}" -m rare_diagnosis.training.eval.eval_ml \
    --json "${OUT_DIR}/test_predictions_ranked.json" \
    --out-csv "${EVAL_DIR}/ml_metrics.csv"
echo ""

# ── Step 4: Evaluate LLM baselines ──────────────────────────────────────
echo "[Step 4/4] Evaluating LLM baselines..."
"${PYTHON}" -m rare_diagnosis.training.eval.eval_llm \
    --score-root "${SCORE_ROOT}" \
    --test-ids "${TEST_IDS}" \
    --out-dir "${EVAL_DIR}"
echo ""

echo "============================================================"
echo "Pipeline complete!"
echo "  Features:      ${WORK_DIR}"
echo "  Model/preds:   ${OUT_DIR}"
echo "  Eval results:  ${EVAL_DIR}"
echo "============================================================"
