# Treatment Ranking - Reproduction Pipeline

This directory contains self-contained scripts to reproduce the treatment ranking experiments.

## Quick Start

```bash
# One-click reproduction with demo data (6 cases)
bash run_pipeline.sh --python D:/Anaconda/envs/raredis/python.exe

# With full data
bash run_pipeline.sh \
    --case-root /data/case_output \
    --llm-output-root /data/treatment_llm \
    --n-splits 5 \
    --python D:/Anaconda/envs/raredis/python.exe
```

## Pipeline Overview

```
Step 0: Data Preparation    ->  plan_root/, treatment_output/, treatment_score/, dataset/
Step 1: LLM Generation      ->  treatment_plan_output.json (per model, per case)
Step 2: Feature Engineering  ->  features_train.csv, features_test.csv
Step 3: ML Training          ->  trained XGBoost models + test_predictions_ensemble.csv
Step 4: Inference            ->  test_predictions.csv
Step 5: Evaluation           ->  ranking metrics (Hit@K, nDCG, MRR, MAP, etc.)
```

## Directory Structure

```
rare_treatment/training/
├── run_pipeline.sh             # One-click reproduction script (bash)
├── run_training.py             # One-click standalone training (Python, from llm_outputs.json)
├── prepare_demo_data.py        # Step 0: Convert demo data into pipeline-expected format
├── generate_llm_outputs.py     # Step 1: Call LLMs to generate treatment plans
├── build_features.py           # Step 2: Build 50+ L2R features from multi-model outputs
├── train_ranker.py             # Step 3: Train XGBoost ranker with GroupKFold CV
├── infer_ranker.py             # Step 4: Run inference with trained models
├── data_io.py                  # Shared: data loading/saving/utility functions
├── eval/
│   ├── metrics.py              # Shared: ranking metric functions (Hit@K, nDCG, MRR, MAP, etc.)
│   ├── eval_llm.py             # Step 5a: Evaluate individual LLM models
│   ├── eval_ml.py              # Step 5b: Evaluate ML ensemble
│   └── eval_secondary_metrics.py  # Step 5c: Secondary metrics for paper (M2-M11)
└── README.md
```

## Step-by-Step Usage

### Step 0: Prepare Data

Convert the compact demo/raw format into the directory structure expected by downstream scripts.

```bash
python prepare_demo_data.py \
    --case-root ../data_demo/case_output \
    --llm-output-root ../data_demo/pipeline_data/treatment/treatment_llm \
    --out-dir ../data_demo/pipeline_data/treatment/prepared
```

**Input:**
- `case_output/<case_id>/1_raw_data/treatment_plan.json` (patient case data)
- `case_output/<case_id>/5_treatment/llm_outputs.json` (per-model eval scores)
- `treatment_llm/<model>/<case_id>/treatment_plan_output.json` (LLM outputs)

**Output:**
- `plan_root/<case_id>/treatment_plan.json`
- `treatment_output/<model>/<case_id>/treatment_plan_output.json`
- `treatment_score/<model>/<case_id>/treatment_score.json`
- `dataset/train_cases.json`, `dataset/test_cases.json`

### Step 1: Generate LLM Outputs

Call LLMs to generate treatment recommendations for each patient case.

```bash
# Direct API mode (e.g., Qwen via DashScope)
python generate_llm_outputs.py \
    /path/to/input /path/to/output \
    --model qwen3-32b \
    --base-url https://dashscope.aliyuncs.com/compatible-mode/v1 \
    --api-key YOUR_KEY \
    --num-workers 10

# Config file mode (e.g., GPT/Claude/DeepSeek)
python generate_llm_outputs.py \
    /path/to/input /path/to/output \
    --model gpt-5 \
    --config configs/OAI_Config_List.json
```

**Input:** `input/<case_id>/treatment_plan.json`
**Output:** `output/<model>/<case_id>/treatment_plan_output.json`

### Step 2: Build Features

Construct learning-to-rank features from multi-model LLM outputs.

```bash
python build_features.py \
    --plan_root /data/plan_root \
    --treatment_output_root /data/treatment_output \
    --treatment_score_root /data/treatment_score \
    --train_ids dataset/train_cases.json \
    --test_ids dataset/test_cases.json \
    --out_dir /data/features \
    --num_gpus 1
```

**Features include:** consensus features, per-model rank/importance/hit, semantic similarity (SentenceTransformer), NLI entailment (DeBERTa), rationale text features, Stage-3 evaluation auxiliaries.

**Output:** `features_train.csv`, `features_test.csv`, `groups_train.csv`, `groups_test.csv`

### Step 3: Train XGBoost Ranker

Train an XGBoost learning-to-rank model with GroupKFold cross-validation.

```bash
python train_ranker.py \
    --data-dir /data/features \
    --out-dir /data/models \
    --objective rank:ndcg \
    --n-splits 5 \
    --target-k 3 \
    --drop-feature-groups stage3_eval,model_support \
    --save-models

# Feature ablation example
python train_ranker.py \
    --data-dir /data/features \
    --out-dir /data/models/ablation_no_nli \
    --drop-feature-groups semantic_nli,stage3_eval,model_support
```

**Output:** `test_predictions_ensemble.csv`, `ranked_results.json`, `feature_importance_ensemble.csv`, `feature_columns_used.txt`, `models/model_fold*.json`

### Step 4: Inference

Run inference with trained models on test features.

```bash
python infer_ranker.py \
    --model-dir /data/models/models \
    --test-csv /data/features/features_test.csv \
    --out-dir /data/results
```

**Output:** `test_predictions.csv`, `ranked_results.json`

### Step 5: Evaluation

#### 5a. Evaluate individual LLM models

```bash
python eval/eval_llm.py \
    --score_root /data/treatment_score \
    --train_ids dataset/train_cases.json \
    --test_ids dataset/test_cases.json
```

#### 5b. Evaluate ML ensemble

```bash
python eval/eval_ml.py \
    --input ensemble=/data/results/test_predictions.csv:score \
    --ks 1,2,3,5
```

#### 5c. Secondary metrics for paper tables (M2-M11)

```bash
python eval/eval_secondary_metrics.py \
    --n11-pred-csv /data/results/test_predictions.csv \
    --features-test-csv /data/features/features_test.csv \
    --rag-root /data/treatment_score \
    --out-dir /data/results/secondary_metrics
```

## Standalone Training (run_training.py)

A self-contained script that reads `llm_outputs.json` directly, builds features, trains XGBoost, and evaluates — without separate steps.

```bash
# Demo data (default)
python run_training.py

# Full data
python run_training.py \
    --data-dir /data/case_output \
    --out-dir /data/models \
    --n-splits 5
```

## Dependencies

```
numpy
pandas
scikit-learn
xgboost
openai            # for LLM API calls (Step 1 only)
tqdm
torch             # for semantic/NLI features (Step 2)
sentence-transformers
openpyxl          # for Excel output in secondary metrics
```
