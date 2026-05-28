# Treatment Ranking - Reproduction Pipeline

This directory contains self-contained scripts to reproduce the treatment ranking experiments. The pipeline has 4 steps:

## Pipeline Overview

```
Step 1: LLM Generation     ->  treatment_plan_output.json (per model, per case)
Step 2: Feature Engineering ->  features_train.csv, features_test.csv
Step 3: ML Training         ->  trained models + test_predictions_ensemble.csv
Step 4: Evaluation          ->  ranking metrics (P@T, NDCG, MRR, etc.)
```

## Directory Structure

```
rare_treatment/training/
├── generate_llm_outputs.py     # Step 1: Call LLMs to generate treatment plans
├── build_features.py           # Step 2: Build 50+ L2R features from multi-model outputs
├── train_ranker.py             # Step 3: Train XGBoost/LightGBM/CatBoost rankers
├── infer_ranker.py             # Step 3b: Run inference with trained models
├── data_io.py                  # Shared: data loading/saving utilities
├── eval/
│   ├── metrics.py              # Shared: ranking metric functions
│   ├── eval_llm.py             # Step 4a: Evaluate individual LLM models (Hit@K, nDCG, MRR)
│   ├── eval_ml.py              # Step 4b: Evaluate ML ensemble (Hit@K, nDCG, MRR, MAP)
│   └── eval_secondary_metrics.py  # Step 4c: Secondary metrics for paper (M2-M11)
└── README.md
```

## Step 1: Generate LLM Outputs

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

## Step 2: Build Features

Construct learning-to-rank features from multi-model LLM outputs.

```bash
python build_features.py \
    --plan_root /data/raw \
    --treatment_output_root /data/output \
    --treatment_score_root /data/scores \
    --train_ids dataset/train_cases.json \
    --test_ids dataset/test_cases.json \
    --out_dir /data/features \
    --num_gpus 4
```

**Features include:** consensus features, per-model rank/importance/hit, semantic similarity (SentenceTransformer), NLI entailment (DeBERTa), rationale text features, Stage-3 evaluation auxiliaries.

**Output:** `features_train.csv`, `features_test.csv`, `groups_train.csv`, `groups_test.csv`

## Step 3: Train ML Ranker

Train learning-to-rank models with GroupKFold cross-validation.

```bash
# XGBoost (default)
python train_ranker.py \
    --data-dir /data/features \
    --out-dir /data/models/treatment \
    --backend xgboost \
    --objective rank:ndcg \
    --save-models

# LightGBM
python train_ranker.py \
    --data-dir /data/features \
    --out-dir /data/models/treatment \
    --backend lightgbm \
    --objective lambdarank

# CatBoost
python train_ranker.py \
    --data-dir /data/features \
    --out-dir /data/models/treatment \
    --backend catboost \
    --objective stochastic_yetirank

# Feature ablation example
python train_ranker.py \
    --data-dir /data/features \
    --out-dir /data/models/ablation_no_nli \
    --drop-feature-groups semantic_nli
```

**Output:** `test_predictions_ensemble.csv`, `ranked_results.json`, `feature_importance_ensemble.csv`, `feature_columns_used.txt`

## Step 4: Evaluation

### 4a. Evaluate individual LLM models (P@T metrics)

```bash
python eval/eval_llm.py \
    --score_root /data/scores \
    --test_ids dataset/test_cases.json
```

### 4b. Evaluate ML ensemble (ranking metrics)

```bash
python eval/eval_ml.py \
    --input ensemble=/data/models/treatment/test_predictions_ensemble.csv:ensemble_score \
    --ks 1,2,3,5
```

### 4c. Secondary metrics for paper tables (M2-M11)

```bash
python eval/eval_secondary_metrics.py \
    --n11-pred-csv /data/models/treatment/test_predictions_ensemble.csv \
    --features-test-csv /data/features/features_test.csv \
    --rag-root /data/scores \
    --out-dir /data/results/secondary_metrics
```

## Dependencies

```
numpy
pandas
scikit-learn
xgboost          # for XGBoost backend
lightgbm         # for LightGBM backend (optional)
catboost          # for CatBoost backend (optional)
openai            # for LLM API calls
tqdm
torch             # for semantic/NLI features
sentence-transformers
openpyxl          # for Excel output in secondary metrics
```
