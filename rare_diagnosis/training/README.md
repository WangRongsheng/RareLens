# Diagnosis Ranking - Reproduction Pipeline

This directory contains scripts to reproduce the diagnosis ranking experiments. The pipeline covers primary consultation and follow-up visit diagnosis, using LLM outputs + XGBoost reranking.

## Pipeline Overview

```
Step 0: RAG Cache (optional)  ->  orphacode_rag_cache/ (FAISS vector index)
Step 1: LLM Generation        ->  primary_consultation_output.json + most_likely_diagnosis_orphacode.json
Step 2: Feature Engineering    ->  features.{train,test}.csv (per visit type)
Step 3: XGBoost Training       ->  GroupKFold CV ranker (rank:ndcg)
Step 4: Evaluation             ->  Acc@1/3/5, MRR, NDCG@1/3/5
```

## Directory Structure

```
rare_diagnosis/training/
├── reproduce_diag.sh                      # One-click reproduction (Steps 2-4)
├── generate_llm_outputs.py                # Step 1: Batch LLM diagnosis generation
├── orphacode_rag.py                       # OrphaCode RAG enrichment module
├── build_features_primary.py              # Step 2: Feature engineering (primary)
├── build_features_followup.py             # Step 2: Feature engineering (followup)
├── train_ranker.py                        # Step 3: XGBoost ranker training
├── infer_ranker.py                        # Standalone inference with trained models
├── best_hyperopt_config_primary.json      # Hyperparameters (primary stage)
├── best_hyperopt_config_followup.json     # Hyperparameters (followup stage)
├── orphanet_hierarchy.json                # Disease ontology with parent relationships
├── eval/
│   ├── metrics.py                         # Shared metric functions (DCG, NDCG, MRR)
│   ├── eval_llm.py                        # Step 4a: Per-model LLM evaluation
│   ├── eval_ml.py                         # Step 4b: ML ranking evaluation
│   └── eval_secondary_metrics.py          # Step 4c: Convergence & rescue analysis
└── README.md
```

## Quick Start

```bash
# Full pipeline (primary stage)
bash rare_diagnosis/training/reproduce_diag.sh \
    --python /path/to/python \
    --visit-type primary \
    --query-root /data/query \
    --gt-root /data/scores \
    --models-root /data/llm_outputs \
    --train-ids /data/splits/train.json \
    --test-ids /data/splits/test.json \
    --out-dir /data/output

# Follow-up stage
bash rare_diagnosis/training/reproduce_diag.sh \
    --visit-type followup \
    --query-root /data/query \
    --gt-root /data/scores \
    --models-root /data/llm_outputs \
    --train-ids /data/splits/train.json \
    --test-ids /data/splits/test.json \
    --out-dir /data/output
```

## Expected Data Layout

```
query_root/
  {case_id}/primary_consultation.json       # Patient data (basic_information, medical_history, ...)

models_root/                                # LLM diagnosis outputs
  {model_name}/
    {case_id}/most_likely_diagnosis_orphacode.json

gt_root/                                    # Ground-truth evaluation scores
  {case_id}/primary_diagnosis_score.json    # Direct path, OR:
  {model_name}/{case_id}/primary_diagnosis_score.json  # Per-model sub-dirs (auto-detected)

splits/
  train.json                                # JSON list of train case IDs
  test.json                                 # JSON list of test case IDs
```

## Step 0: RAG Cache (Optional, Pre-built)

The OrphaCode RAG cache (`rare_diagnosis/orphacode_rag_cache/`) maps diagnosis names to Orphanet codes:
- 15,772 disease entries from Orphanet
- FAISS vector index with BAAI/bge-base-en-v1.5 embeddings (768-dim)

To rebuild: `python rare_diagnosis/tools/build_orphacode_rag_cache.py`

## Step 1: Generate LLM Outputs

```bash
# Direct API mode
python -m rare_diagnosis.training.generate_llm_outputs \
    --input-root /data/query \
    --case-ids /data/splits/train.json \
    --base-url https://api.openai.com/v1 \
    --api-key $OPENAI_API_KEY \
    --models gpt-5 o3-mini gpt-3.5-turbo gpt-4o-mini \
    --visit-type primary \
    --out-dir /data/llm_outputs \
    --workers 8

# Config file mode (multiple endpoints)
python -m rare_diagnosis.training.generate_llm_outputs \
    --input-root /data/query \
    --case-ids /data/splits/train.json \
    --config llm_config.json \
    --visit-type primary \
    --out-dir /data/llm_outputs

# With OrphaCode RAG enrichment
python -m rare_diagnosis.training.generate_llm_outputs \
    ... \
    --enable-orphacode-rag \
    --rag-ontology-path rare_diagnosis/orphacode_rag_cache/orphanet_rare_diseases.json
```

**Elite models and weights (used in feature engineering):**

| Model | Weight | Role |
|-------|--------|------|
| gpt-5 | 10.0 | King |
| o3-mini | 8.0 | King |
| gpt-3.5-turbo | 6.0 | Knight |
| gemini-2.5-flash-preview-05-20-nothinking | 6.0 | Knight |
| deepseek-r1-0528 | 5.5 | Knight |
| qwen3-235b-a22b-instruct-2507 | 4.5 | Knight |
| claude-haiku-4-5-20251001 | 3.0 | Knight |
| qwen3-8b | 1.0 | Pawn |
| qwen3-14b | 1.0 | Pawn |
| qwen3-32b | 1.0 | Pawn |
| gpt-4o-mini | 1.0 | Pawn |

## Step 2: Build Features

56 features per diagnosis candidate, grouped into:
- **Consensus:** weighted_score, agreement_ratio, kings_consensus, appear_count_elite
- **Semantic:** SentenceTransformer cosine similarity (name, reasoning)
- **Ontology:** depth, is_leaf, ancestor_match, num_parents
- **Per-model:** rank, conf, hit, z_conf, r_sim (per elite model)
- **Text:** input_word_len, negation_count, certainty_score, reasoning_len

```bash
# Primary stage
python -m rare_diagnosis.training.build_features_primary \
    --query_root /data/query \
    --primary_models_root /data/llm_outputs \
    --gt_root /data/scores \
    --train_ids /data/splits/train.json \
    --test_ids /data/splits/test.json \
    --out_dir /data/features/primary

# Follow-up stage
python -m rare_diagnosis.training.build_features_followup \
    --query_root /data/query \
    --primary_models_root /data/llm_outputs \
    --gt_root /data/scores \
    --train_ids /data/splits/train.json \
    --test_ids /data/splits/test.json \
    --out_dir /data/features/followup
```

Output: `features.{train,test}.csv` + `groups.{train,test}.csv`

## Step 3: Train XGBoost Ranker

```bash
python -m rare_diagnosis.training.train_ranker \
    --input-dir /data/features/primary \
    --config rare_diagnosis/training/best_hyperopt_config_primary.json \
    --out-dir /data/models/primary \
    --use-gpu
```

- Objective: `rank:ndcg` with GroupKFold (5-fold)
- Monotonicity constraints auto-inferred from feature names
- Stage-specific hyperparameters (`best_hyperopt_config_{primary,followup}.json`)

Output: `models/xgboost_fold_*.json`, `test_predictions_ranked.{json,csv}`, `feature_importance.csv`

### Standalone Inference

```bash
python -m rare_diagnosis.training.infer_ranker \
    --input-dir /data/features/primary \
    --model-dir /data/models/primary/models \
    --config rare_diagnosis/training/best_hyperopt_config_primary.json \
    --out-dir /data/inference_output
```

Supports both `.json` (Booster) and `.pkl` (XGBRanker, legacy) model formats.

## Step 4: Evaluation

### 4a. Evaluate LLM Models

```bash
python -m rare_diagnosis.training.eval.eval_llm \
    --score-root /data/scores \
    --test-ids /data/splits/test.json \
    --out-dir /data/results

# Excel output
python -m rare_diagnosis.training.eval.eval_llm \
    --score-root /data/scores \
    --test-ids /data/splits/test.json \
    --out-dir /data/results \
    --excel
```

Output: `{split}_Top{N}.csv` (Acc@1/3/5 %) + `{split}_ndcg_mrr.csv` (MRR, NDCG@1/3/5)

### 4b. Evaluate ML Ranking

```bash
python -m rare_diagnosis.training.eval.eval_ml \
    --json /data/models/primary/test_predictions_ranked.json \
    --out-csv /data/results/ml_metrics.csv
```

### 4c. Secondary Metrics (Convergence & Rescue)

```bash
# Primary vs follow-up convergence
python -m rare_diagnosis.training.eval.eval_secondary_metrics \
    --primary-csv /data/models/primary/test_predictions_ranked.csv \
    --followup-csv /data/models/followup/test_predictions_ranked.csv \
    --out-dir /data/results/secondary_metrics

# ML vs best LLM rescue
python -m rare_diagnosis.training.eval.eval_secondary_metrics \
    --primary-csv /data/models/primary/test_predictions_ranked.csv \
    --llm-score-root /data/scores \
    --test-ids /data/splits/test.json \
    --out-dir /data/results/secondary_metrics
```

## Dependencies

```
numpy
pandas
scikit-learn
xgboost
openai            # LLM API calls
tqdm
torch             # Semantic features (optional, degrades gracefully)
sentence-transformers
faiss-cpu         # RAG vector index (or faiss-gpu, optional)
openpyxl          # Excel output (optional)
```
