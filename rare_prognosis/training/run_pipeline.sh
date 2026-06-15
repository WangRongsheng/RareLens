#!/usr/bin/env bash
#
# One-click reproduction script for RarePrognosis stacking pipeline.
#
# Usage:
#   bash run_pipeline.sh                          # use demo data
#   bash run_pipeline.sh --case-root /data/cases --llm-root /data/llm  # full data
#
# Requirements: Python with scikit-learn, numpy
#
set -euo pipefail

# Use pwd -W on MSYS/Git-Bash (Windows) so Python receives valid paths
_pwd() { pwd -W 2>/dev/null || pwd; }
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && _pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && _pwd)"

# ── Python interpreter (override with --python or PYTHON env var) ─────────
PYTHON="${PYTHON:-python}"

# ── Defaults (demo data) ──────────────────────────────────────────────────
CASE_ROOT="${REPO_ROOT}/data_500"
LLM_ROOT="${REPO_ROOT}/data_demo/pipeline_data/prognoisis/llm"
OUT_ROOT="${REPO_ROOT}/data_demo/pipeline_data/prognoisis/prepared"
CV_FOLDS=3         # use 3 for demo (6 cases), 5 for full data
SEED=42
TASK="all"
# Optional explicit splits. Empty = use prepare_data's all-cases-as-train+test (smoke).
TRAIN_IDS_ARG=""
TEST_IDS_ARG=""

# ── Parse arguments ───────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --case-root)         CASE_ROOT="$2";         shift 2 ;;
        --llm-root)          LLM_ROOT="$2";          shift 2 ;;
        --out-root)          OUT_ROOT="$2";           shift 2 ;;
        --cv-folds)          CV_FOLDS="$2";           shift 2 ;;
        --seed)              SEED="$2";               shift 2 ;;
        --task)              TASK="$2";               shift 2 ;;
        --train-ids)         TRAIN_IDS_ARG="$2";     shift 2 ;;
        --test-ids)          TEST_IDS_ARG="$2";      shift 2 ;;
        --python)            PYTHON="$2";             shift 2 ;;
        -h|--help)
            echo "Usage: bash run_pipeline.sh [OPTIONS]"
            echo "  --case-root DIR          Root of case data with prognosis_new.json (default: data_500)"
            echo "  --llm-root DIR           Root of LLM prognosis outputs (default: demo data)"
            echo "  --out-root DIR           Output root directory"
            echo "  --cv-folds N             StratifiedKFold splits (default: 3 for demo, use 5 for full)"
            echo "  --seed N                 Random seed (default: 42)"
            echo "  --task TASK              Task to run: overall_outcome|functional_status|symptom_burden|all"
            echo "  --train-ids PATH         JSON list of train case IDs (default: all cases, smoke mode)"
            echo "  --test-ids PATH          JSON list of test case IDs (default: all cases, smoke mode)"
            echo "  --python PATH            Python interpreter (default: python, or \$PYTHON env var)"
            exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

PREPARED="${OUT_ROOT}"
RARE_ROOT="${PREPARED}/rareprognosis"
MODELS_ROOT="${PREPARED}/models"
DATASET_DIR="${PREPARED}/dataset"
FEATURES_DIR="${PREPARED}/features"
MODELS_OUT="${PREPARED}/trained_models"
# Use explicit splits when given, else fall back to prepare_data's all-cases JSONs.
TRAIN_IDS="${TRAIN_IDS_ARG:-${DATASET_DIR}/train_case_ids.json}"
TEST_IDS="${TEST_IDS_ARG:-${DATASET_DIR}/test_case_ids.json}"

echo "============================================================"
echo "RarePrognosis Pipeline Reproduction"
echo "============================================================"
echo "  case_root:         ${CASE_ROOT}"
echo "  llm_root:          ${LLM_ROOT}"
echo "  out_root:          ${OUT_ROOT}"
echo "  cv_folds:          ${CV_FOLDS}"
echo "  seed:              ${SEED}"
echo "  task:              ${TASK}"
echo "  python:            ${PYTHON}"
echo "============================================================"
echo ""

# ── Step 0: Prepare data ─────────────────────────────────────────────────
echo "[Step 0/3] Preparing data..."
"${PYTHON}" "${SCRIPT_DIR}/prepare_data.py" \
    --case-root "${CASE_ROOT}" \
    --llm-root "${LLM_ROOT}" \
    --out-dir "${PREPARED}"
echo ""

# ── Step 1: Build features ───────────────────────────────────────────────
echo "[Step 1/3] Building features..."
"${PYTHON}" "${SCRIPT_DIR}/build_features.py" \
    --rareprognosis-root "${RARE_ROOT}" \
    --models-root "${MODELS_ROOT}" \
    --train-ids "${TRAIN_IDS}" \
    --test-ids "${TEST_IDS}" \
    --out-dir "${FEATURES_DIR}" \
    --task "${TASK}"
echo ""

# ── Step 2: Train models (OOF GBDT) ─────────────────────────────────────
echo "[Step 2/3] Training OOF GBDT stacking models..."
"${PYTHON}" "${SCRIPT_DIR}/train_models.py" \
    --features-dir "${FEATURES_DIR}" \
    --out-dir "${MODELS_OUT}" \
    --task "${TASK}" \
    --seed "${SEED}" \
    --cv-folds "${CV_FOLDS}"
echo ""

# ── Step 3: Inference with trained models ────────────────────────────────
echo "[Step 3/3] Running inference..."
"${PYTHON}" "${SCRIPT_DIR}/infer_models.py" \
    --rareprognosis-root "${RARE_ROOT}" \
    --models-root "${MODELS_ROOT}" \
    --train-ids "${TRAIN_IDS}" \
    --test-ids "${TEST_IDS}" \
    --models-dir "${MODELS_OUT}" \
    --task "${TASK}"
echo ""

echo "============================================================"
echo "Pipeline complete!"
echo "  Features:       ${FEATURES_DIR}"
echo "  Trained models: ${MODELS_OUT}"
echo "  S1 CSVs:        ${RARE_ROOT}"
echo "============================================================"
