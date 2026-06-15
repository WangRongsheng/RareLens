#!/usr/bin/env bash
#
# One-click reproduction script for RareTreatment ranking pipeline.
#
# Usage:
#   bash run_pipeline.sh                          # use demo data (6 cases)
#   bash run_pipeline.sh --case-root /data/cases --llm-root /data/llm_outputs  # use full data
#
# Requirements: Python with xgboost, pandas, scikit-learn, numpy
#               Optional: sentence-transformers, torch (for semantic/NLI features)
#
set -euo pipefail

# Use pwd -W on MSYS/Git-Bash (Windows) so Python receives valid paths
_pwd() { pwd -W 2>/dev/null || pwd; }
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && _pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && _pwd)"

# ── Python interpreter (override with --python or PYTHON env var) ─────────
PYTHON="${PYTHON:-python}"

# ── Defaults (demo data) ──────────────────────────────────────────────────
CASE_ROOT="${REPO_ROOT}/data_demo/case_output"
LLM_OUTPUT_ROOT="${REPO_ROOT}/data_demo/pipeline_data/treatment/treatment_llm"
OUT_ROOT="${REPO_ROOT}/data_demo/pipeline_data/treatment/prepared"
# Optional: pre-split per-model judge scores (<model>/<case>/treatment_score.json).
# When set, build_features reads them directly — no llm_outputs.json splitting
# needed. Empty = legacy behaviour.
SCORE_ROOT_ARG=""
# Optional explicit splits. Empty = use prepare_data's all-cases-as-train+test (smoke).
TRAIN_IDS_ARG=""
TEST_IDS_ARG=""
NUM_GPUS=1
N_SPLITS=5          # GroupKFold splits; lower it if a stage has very few cases
TARGET_K=3
OBJECTIVE="rank:ndcg"
DROP_GROUPS="stage3_eval"

# ── Parse arguments ───────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --case-root)       CASE_ROOT="$2";       shift 2 ;;
        --llm-root)        LLM_OUTPUT_ROOT="$2"; shift 2 ;;
        --score-root)      SCORE_ROOT_ARG="$2";  shift 2 ;;
        --train-ids)       TRAIN_IDS_ARG="$2";   shift 2 ;;
        --test-ids)        TEST_IDS_ARG="$2";    shift 2 ;;
        --out-root)        OUT_ROOT="$2";         shift 2 ;;
        --num-gpus)        NUM_GPUS="$2";         shift 2 ;;
        --n-splits)        N_SPLITS="$2";         shift 2 ;;
        --target-k)        TARGET_K="$2";         shift 2 ;;
        --objective)       OBJECTIVE="$2";        shift 2 ;;
        --drop-groups)     DROP_GROUPS="$2";      shift 2 ;;
        --python)          PYTHON="$2";           shift 2 ;;
        -h|--help)
            echo "Usage: bash run_pipeline.sh [OPTIONS]"
            echo "  --case-root DIR        Root of case directories (default: demo data)"
            echo "  --llm-root DIR         Root of LLM treatment outputs (default: demo data)"
            echo "  --score-root DIR       Pre-split per-model judge scores"
            echo "                         (<model>/<case>/treatment_score.json). Read directly, no splitting."
            echo "  --train-ids PATH       JSON list of train case IDs (default: all cases, smoke mode)"
            echo "  --test-ids PATH        JSON list of test case IDs (default: all cases, smoke mode)"
            echo "  --out-root DIR         Output root directory"
            echo "  --num-gpus N           GPUs for build_features.py (default: 1)"
            echo "  --n-splits N           GroupKFold splits (default: 5; lower it if very few cases)"
            echo "  --target-k K           Target cutoff K (default: 3)"
            echo "  --objective OBJ        XGBoost objective (default: rank:ndcg)"
            echo "  --drop-groups GROUPS   Feature groups to drop (default: stage3_eval,model_support)"
            echo "  --python PATH          Python interpreter (default: python, or \$PYTHON env var)"
            exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

PREPARED="${OUT_ROOT}"
FEATURES_DIR="${PREPARED}/features"
MODELS_DIR="${PREPARED}/models"
RESULTS_DIR="${PREPARED}/results"
# When --score-root is given, build_features reads it directly (diagnosis-style);
# otherwise use the scores prepare_data splits into the prepared dir.
SCORE_ROOT="${SCORE_ROOT_ARG:-${PREPARED}/treatment_score}"
DATASET_DIR="${PREPARED}/dataset"
# Use explicit splits when given, else fall back to prepare_data's all-cases JSONs.
TRAIN_IDS="${TRAIN_IDS_ARG:-${DATASET_DIR}/train_cases.json}"
TEST_IDS="${TEST_IDS_ARG:-${DATASET_DIR}/test_cases.json}"

echo "============================================================"
echo "RareTreatment Pipeline Reproduction"
echo "============================================================"
echo "  case_root:       ${CASE_ROOT}"
echo "  llm_output_root: ${LLM_OUTPUT_ROOT}"
echo "  score_root:      ${SCORE_ROOT}"
echo "  out_root:        ${OUT_ROOT}"
echo "  num_gpus:        ${NUM_GPUS}"
echo "  n_splits:        ${N_SPLITS}"
echo "  target_k:        ${TARGET_K}"
echo "  objective:       ${OBJECTIVE}"
echo "  drop_groups:     ${DROP_GROUPS}"
echo "  python:          ${PYTHON}"
echo "============================================================"
echo ""

# ── Step 0: Prepare data ─────────────────────────────────────────────────
echo "[Step 0/3] Preparing data..."
"${PYTHON}" "${SCRIPT_DIR}/prepare_data.py" \
    --case-root "${CASE_ROOT}" \
    --llm-root "${LLM_OUTPUT_ROOT}" \
    ${SCORE_ROOT_ARG:+--score-root "${SCORE_ROOT_ARG}"} \
    --out-dir "${PREPARED}"
echo ""

# ── Step 1: Build features ───────────────────────────────────────────────
echo "[Step 1/3] Building features..."
"${PYTHON}" "${SCRIPT_DIR}/build_features.py" \
    --plan_root "${PREPARED}/plan_root" \
    --treatment_output_root "${PREPARED}/treatment_output" \
    --treatment_score_root "${SCORE_ROOT}" \
    --train_ids "${TRAIN_IDS}" \
    --test_ids "${TEST_IDS}" \
    --out_dir "${FEATURES_DIR}" \
    --num_gpus "${NUM_GPUS}"
echo ""

# ── Step 2: Train ranker ─────────────────────────────────────────────────
echo "[Step 2/3] Training XGBoost ranker..."
"${PYTHON}" "${SCRIPT_DIR}/train_ranker.py" \
    --data-dir "${FEATURES_DIR}" \
    --out-dir "${MODELS_DIR}" \
    --objective "${OBJECTIVE}" \
    --n-splits "${N_SPLITS}" \
    --target-k "${TARGET_K}" \
    --drop-feature-groups "${DROP_GROUPS}" \
    --save-models
echo ""

# ── Step 3: Inference ────────────────────────────────────────────────────
echo "[Step 3/3] Running inference..."
"${PYTHON}" "${SCRIPT_DIR}/infer_ranker.py" \
    --model-dir "${MODELS_DIR}/models" \
    --test-csv "${FEATURES_DIR}/features_test.csv" \
    --out-dir "${RESULTS_DIR}"
echo ""

echo "============================================================"
echo "Pipeline complete!  (Test Hit@1/3/5 + MRR printed in Step 2)"
echo "  Features:    ${FEATURES_DIR}"
echo "  Models:      ${MODELS_DIR}/models"
echo "  Predictions: ${RESULTS_DIR}"
echo "============================================================"
