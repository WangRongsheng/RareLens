# RareDiagnosis

## Overview

We provide the training pipeline for rare disease diagnosis ranking, supporting both primary consultation and follow-up visit stages. For methodological details, please refer to the paper.

## Pipeline

```
Step 0: RAG Cache (optional)   →  FAISS vector index for OrphaCode resolution
Step 1: LLM Generation         →  per-model diagnosis outputs + OrphaCode mapping
Step 2: Feature Engineering     →  features.{train,test}.csv (56+ features per candidate)
Step 3: XGBoost Training        →  GroupKFold CV ranker (rank:ndcg)
```

## Quick Start

```bash
# Full pipeline (primary stage): build features → train
bash rare_diagnosis/training/reproduce_diag.sh \
    --python /path/to/python \
    --visit-type primary \
    --query-root /data/query \
    --gt-root /data/scores \
    --models-root /data/llm_outputs \
    --train-ids /data/splits/train.json \
    --test-ids /data/splits/test.json

# Follow-up stage
bash rare_diagnosis/training/reproduce_diag.sh \
    --visit-type followup \
    --query-root /data/query \
    --gt-root /data/scores \
    --models-root /data/llm_outputs \
    --train-ids /data/splits/train.json \
    --test-ids /data/splits/test.json
```

## Step-by-Step Usage

### Step 1: LLM Generation

[`generate_llm_outputs.py`](generate_llm_outputs.py) queries LLMs to produce top-5 diagnoses per case. Each diagnosis is optionally enriched with an OrphaCode via semantic retrieval ([`orphacode_rag.py`](orphacode_rag.py)).

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

# With OrphaCode RAG enrichment
python -m rare_diagnosis.training.generate_llm_outputs \
    ... \
    --enable-orphacode-rag \
    --rag-ontology-path rare_diagnosis/orphacode_rag_cache/orphanet_rare_diseases.json
```

### Step 2: Feature Engineering

[`build_features_primary.py`](build_features_primary.py) and [`build_features_followup.py`](build_features_followup.py) construct ranking features per candidate from multi-model outputs.

```bash
# Primary stage
python -m rare_diagnosis.training.build_features_primary \
    --query_root /data/query \
    --primary_models_root /data/llm_outputs \
    --gt_root /data/scores \
    --train_ids /data/splits/train.json \
    --test_ids /data/splits/test.json \
    --out_dir /data/features/primary

# Follow-up stage (includes diagnostic test results)
python -m rare_diagnosis.training.build_features_followup \
    --query_root /data/query \
    --primary_models_root /data/llm_outputs \
    --gt_root /data/scores \
    --train_ids /data/splits/train.json \
    --test_ids /data/splits/test.json \
    --out_dir /data/features/followup
```

### Step 3: XGBoost Training

[`train_ranker.py`](train_ranker.py) trains an XGBoost LTR model with GroupKFold (5-fold) cross-validation. Monotonicity constraints are auto-inferred from feature names.

```bash
python -m rare_diagnosis.training.train_ranker \
    --input-dir /data/features/primary \
    --config rare_diagnosis/training/best_hyperopt_config_primary.json \
    --out-dir /data/models/primary \
    --use-gpu
```

Standalone inference with trained models:

```bash
python -m rare_diagnosis.training.infer_ranker \
    --input-dir /data/features/primary \
    --model-dir /data/models/primary/models \
    --config rare_diagnosis/training/best_hyperopt_config_primary.json \
    --out-dir /data/inference_output
```

