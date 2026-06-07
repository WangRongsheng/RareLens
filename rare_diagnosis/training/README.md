# RareDiagnosis — Diagnosis Ranking Module

## Overview

We provide the training and evaluation pipeline for rare disease diagnosis ranking, supporting both primary consultation and follow-up visit stages. For methodological details, please refer to the paper.

## Pipeline

```
Step 0: RAG Cache (optional)   →  FAISS vector index for OrphaCode resolution
Step 1: LLM Generation         →  per-model diagnosis outputs + OrphaCode mapping
Step 2: Feature Engineering     →  features.{train,test}.csv (56+ features per candidate)
Step 3: XGBoost Training        →  GroupKFold CV ranker (rank:ndcg)
Step 4: Evaluation              →  Acc@1/3/5, MRR, NDCG@1/3/5
```

## Quick Start

```bash
# Full pipeline (primary stage): build features → train → eval ML → eval LLM
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

### Step 4: Evaluation

**LLM-as-judge scoring** ([`eval/run_judge.py`](eval/run_judge.py)): a judge LLM scores each diagnosis prediction (0–5) against ground truth.

```bash
python -m rare_diagnosis.training.eval.run_judge \
    --pred-root /data/llm_outputs/qwen3-32b \
    --gt-root /data/gt/diag \
    --out-root /data/scores/qwen3-32b \
    --tasks primary_diag follow_diag diag_test \
    --base-url https://api.openai.com/v1 \
    --api-key $OPENAI_API_KEY
```

**LLM baseline evaluation** ([`eval/eval_llm.py`](eval/eval_llm.py)):

```bash
python -m rare_diagnosis.training.eval.eval_llm \
    --score-root /data/scores \
    --test-ids /data/splits/test.json \
    --out-dir /data/results --excel
```

**ML ranking evaluation** ([`eval/eval_ml.py`](eval/eval_ml.py)):

```bash
python -m rare_diagnosis.training.eval.eval_ml \
    --json /data/models/primary/test_predictions_ranked.json \
    --out-csv /data/results/ml_metrics.csv
```

**Secondary metrics** ([`eval/eval_secondary_metrics.py`](eval/eval_secondary_metrics.py)): convergence analysis (primary → follow-up) and rescue analysis (ML vs best LLM).

## Evaluation Metrics

| Metric | Description |
|---|---|
| Acc@1 / Acc@3 / Acc@5 | Fraction of cases with a correct diagnosis in top-K |
| MRR | Mean Reciprocal Rank of first correct diagnosis |
| NDCG@1 / NDCG@3 / NDCG@5 | Normalized Discounted Cumulative Gain |
| Rescue rate | Cases where ML succeeds but best LLM baseline fails |
| Convergence gain | Cases improved from primary to follow-up stage |
