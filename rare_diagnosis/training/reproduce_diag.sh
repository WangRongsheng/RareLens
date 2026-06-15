#!/usr/bin/env bash
#
# One-click reproduction script for RareDiagnosis XGBoost ranking pipeline.
#
# Pipeline steps:
#   Step 1: Build features  (build_features_primary / build_features_followup)
#   Step 2: Train ranker    (train_ranker — XGBoost GroupKFold)
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
        --case-root)        QUERY_ROOT="$2";        shift 2 ;;
        --score-root)       GT_ROOT="$2";           shift 2 ;;
        --llm-root)         MODELS_ROOT="$2";       shift 2 ;;
        --train-ids)        TRAIN_IDS="$2";         shift 2 ;;
        --test-ids)         TEST_IDS="$2";          shift 2 ;;
        --primary-fname)    PRIMARY_FNAME="$2";     shift 2 ;;
        --gt-fname)         GT_FNAME="$2";          shift 2 ;;
        --num-gpus)         FEAT_NUM_GPUS="$2";     shift 2 ;;
        --workers)          FEAT_WORKERS="$2";      shift 2 ;;
        --work-dir)         WORK_DIR="$2";          shift 2 ;;
        --out-dir)          OUT_DIR="$2";           shift 2 ;;
        --python)           PYTHON="$2";            shift 2 ;;
        --no-gpu)           USE_GPU="";             shift ;;
        -h|--help)
            echo "Usage: bash reproduce_diag.sh [OPTIONS]"
            echo ""
            echo "  --visit-type TYPE     primary (default) or followup"
            echo "  --case-root DIR       Root of case data (primary_consultation.json)"
            echo "  --score-root DIR      Root of ground-truth scores"
            echo "  --llm-root DIR        Root of per-model diagnosis outputs"
            echo "  --train-ids PATH      JSON list of train case IDs"
            echo "  --test-ids PATH       JSON list of test case IDs"
            echo "  --primary-fname NAME  Per-model output filename (default depends on visit-type)"
            echo "  --gt-fname NAME       Per-case GT score filename (default: primary_diagnosis_score.json)"
            echo "  --num-gpus N          GPUs for feature extraction (default: 1; use 0 for CPU)"
            echo "  --workers N           Feature-extraction worker processes (default: 4; use 1-2 on CPU)"
            echo "  --work-dir DIR        Feature output directory"
            echo "  --out-dir DIR         Model output directory"
            echo "  --python PATH         Python interpreter (default: python)"
            echo "  --no-gpu              Disable GPU training (train_ranker)"
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

# ── Config (stage-specific hyperparameters) ──────────────────────────────
if [[ "${VISIT_TYPE}" == "primary" ]]; then
    CONFIG_PATH="${REPO_ROOT}/rare_diagnosis/training/best_hyperopt_config_primary.json"
elif [[ "${VISIT_TYPE}" == "followup" ]]; then
    CONFIG_PATH="${REPO_ROOT}/rare_diagnosis/training/best_hyperopt_config_followup.json"
else
    echo "[ERROR] VISIT_TYPE must be primary or followup, got: ${VISIT_TYPE}" >&2
    exit 1
fi

# ── Per-model output filename (depends on visit-type; override with --primary-fname) ──
# primary  : <models-root>/<model>/<case>/most_likely_diagnosis_orphacode.json
# followup : <models-root>/<model>/<case>/follow_up_consultation_output_unified_orphacode.json
if [[ "${VISIT_TYPE}" == "primary" ]]; then
    PRIMARY_FNAME="${PRIMARY_FNAME:-most_likely_diagnosis_orphacode.json}"
else
    PRIMARY_FNAME="${PRIMARY_FNAME:-follow_up_consultation_output_unified_orphacode.json}"
fi
GT_FNAME="${GT_FNAME:-primary_diagnosis_score.json}"

# ── Feature-extraction compute (override with --num-gpus / --workers) ──────
# GPU worker pools can crash on some setups (CUDA-in-subprocess / OOM). Use
# --num-gpus 0 to run feature extraction on CPU (fine for the demo).
FEAT_NUM_GPUS="${FEAT_NUM_GPUS:-1}"
FEAT_WORKERS="${FEAT_WORKERS:-4}"

# ── Output paths ─────────────────────────────────────────────────────────
WORK_DIR="${WORK_DIR:-${REPO_ROOT}/outputs/diagnosis_features_${VISIT_TYPE}}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/outputs/diagnosis_model_${VISIT_TYPE}}"
USE_GPU="${USE_GPU:-"--use-gpu"}"

mkdir -p "${WORK_DIR}" "${OUT_DIR}"

echo "============================================================"
echo "RareDiagnosis Pipeline Reproduction"
echo "============================================================"
echo "  visit_type:   ${VISIT_TYPE}"
echo "  query_root:   ${QUERY_ROOT}"
echo "  gt_root:      ${GT_ROOT}"
echo "  models_root:  ${MODELS_ROOT}"
echo "  train_ids:    ${TRAIN_IDS}"
echo "  test_ids:     ${TEST_IDS}"
echo "  primary_fname:${PRIMARY_FNAME}"
echo "  gt_fname:     ${GT_FNAME}"
echo "  feat_gpus:    ${FEAT_NUM_GPUS}  (workers=${FEAT_WORKERS})"
echo "  config:       ${CONFIG_PATH}"
echo "  work_dir:     ${WORK_DIR}"
echo "  out_dir:      ${OUT_DIR}"
echo "  python:       ${PYTHON}"
echo "============================================================"
echo ""

# ── Step 1: Build features ───────────────────────────────────────────────
echo "[Step 1/2] Building features (${VISIT_TYPE})..."
if [[ "${VISIT_TYPE}" == "primary" ]]; then
    "${PYTHON}" -m rare_diagnosis.training.build_features_primary \
        --query_root "${QUERY_ROOT}" \
        --out_dir "${WORK_DIR}" \
        --gt_root "${GT_ROOT}" \
        --primary_models_root "${MODELS_ROOT}" \
        --train_ids "${TRAIN_IDS}" \
        --test_ids "${TEST_IDS}" \
        --primary_fname "${PRIMARY_FNAME}" \
        --gt_fname "${GT_FNAME}" \
        --num_gpus "${FEAT_NUM_GPUS}" \
        --workers "${FEAT_WORKERS}"
else
    "${PYTHON}" -m rare_diagnosis.training.build_features_followup \
        --query_root "${QUERY_ROOT}" \
        --out_dir "${WORK_DIR}" \
        --gt_root "${GT_ROOT}" \
        --primary_models_root "${MODELS_ROOT}" \
        --train_ids "${TRAIN_IDS}" \
        --test_ids "${TEST_IDS}" \
        --primary_fname "${PRIMARY_FNAME}" \
        --gt_fname "${GT_FNAME}" \
        --num_gpus "${FEAT_NUM_GPUS}" \
        --workers "${FEAT_WORKERS}"
fi
echo ""

# ── Step 2: Train XGBoost ranker ─────────────────────────────────────────
echo "[Step 2/2] Training XGBoost ranker..."
"${PYTHON}" -m rare_diagnosis.training.train_ranker \
    --input-dir "${WORK_DIR}" \
    --config "${CONFIG_PATH}" \
    --out-dir "${OUT_DIR}" \
    ${USE_GPU}
echo ""

echo "============================================================"
echo "Pipeline complete!"
echo "  Features:      ${WORK_DIR}"
echo "  Model/preds:   ${OUT_DIR}"
echo "============================================================"
