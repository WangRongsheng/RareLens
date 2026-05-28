#!/usr/bin/env bash
#
# One-click reproduction script for RareTreatment ranking pipeline.
#
# Usage:
#   bash run_pipeline.sh                          # use demo data (6 cases)
#   bash run_pipeline.sh --case-root /data/cases --llm-output-root /data/llm_outputs  # use full data
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
NUM_GPUS=1
N_SPLITS=3          # use 3 for demo (6 cases), 5 for full data
TARGET_K=3
OBJECTIVE="rank:ndcg"
DROP_GROUPS="stage3_eval,model_support"

# ── Parse arguments ───────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --case-root)       CASE_ROOT="$2";       shift 2 ;;
        --llm-output-root) LLM_OUTPUT_ROOT="$2"; shift 2 ;;
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
            echo "  --llm-output-root DIR  Root of LLM treatment outputs (default: demo data)"
            echo "  --out-root DIR         Output root directory"
            echo "  --num-gpus N           GPUs for build_features.py (default: 1)"
            echo "  --n-splits N           GroupKFold splits (default: 3 for demo, use 5 for full)"
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
SCORE_ROOT="${PREPARED}/treatment_score"
DATASET_DIR="${PREPARED}/dataset"

echo "============================================================"
echo "RareTreatment Pipeline Reproduction"
echo "============================================================"
echo "  case_root:       ${CASE_ROOT}"
echo "  llm_output_root: ${LLM_OUTPUT_ROOT}"
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
echo "[Step 0/5] Preparing data..."
"${PYTHON}" "${SCRIPT_DIR}/prepare_demo_data.py" \
    --case-root "${CASE_ROOT}" \
    --llm-output-root "${LLM_OUTPUT_ROOT}" \
    --out-dir "${PREPARED}"
echo ""

# ── Step 1: Build features ───────────────────────────────────────────────
echo "[Step 1/5] Building features..."
"${PYTHON}" "${SCRIPT_DIR}/build_features.py" \
    --plan_root "${PREPARED}/plan_root" \
    --treatment_output_root "${PREPARED}/treatment_output" \
    --treatment_score_root "${SCORE_ROOT}" \
    --train_ids "${DATASET_DIR}/train_cases.json" \
    --test_ids "${DATASET_DIR}/test_cases.json" \
    --out_dir "${FEATURES_DIR}" \
    --num_gpus "${NUM_GPUS}"
echo ""

# ── Step 2: Train ranker ─────────────────────────────────────────────────
echo "[Step 2/5] Training XGBoost ranker..."
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
echo "[Step 3/5] Running inference..."
"${PYTHON}" "${SCRIPT_DIR}/infer_ranker.py" \
    --model-dir "${MODELS_DIR}/models" \
    --test-csv "${FEATURES_DIR}/features_test.csv" \
    --out-dir "${RESULTS_DIR}"
echo ""

# ── Step 4: Evaluate LLM baselines ───────────────────────────────────────
echo "[Step 4/5] Evaluating LLM baselines..."
"${PYTHON}" "${SCRIPT_DIR}/eval/eval_llm.py" \
    --score_root "${SCORE_ROOT}" \
    --train_ids "${DATASET_DIR}/train_cases.json" \
    --test_ids "${DATASET_DIR}/test_cases.json"
echo ""

# ── Step 5: Evaluate ML ensemble ─────────────────────────────────────────
echo "[Step 5/5] Evaluating ML ensemble..."
"${PYTHON}" "${SCRIPT_DIR}/eval/eval_ml.py" \
    --input "ensemble=${RESULTS_DIR}/test_predictions.csv:score" \
    --ks "1,2,3,5"
echo ""

echo "============================================================"
echo "Pipeline complete!"
echo "  Features:    ${FEATURES_DIR}"
echo "  Models:      ${MODELS_DIR}/models"
echo "  Predictions: ${RESULTS_DIR}"
echo "============================================================"
