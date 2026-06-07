# RarePrognosis

## Overview

We provide the training and evaluation pipeline for rare disease prognosis prediction, covering three sub-tasks: overall outcome, functional status, and symptom burden. For methodological details, please refer to the paper.

## Pipeline

```
Step 0: Data Preparation       →  S1 CSVs, model directories, train/test splits
Step 1: Feature Engineering    →  features.{train,test}.csv per sub-task
Step 2: Training               →  GBDT stacking models (.pkl)
Step 3: Inference              →  Predictions written to S1 CSVs
Step 4: Evaluation             →  Core + secondary metrics (LLM baselines and ML ensemble)
```

## Quick Start

```bash
# One-click reproduction
bash rare_prognosis/training/run_pipeline.sh \
    --python /path/to/python \
    --case-root /data/case_output \
    --llm-root /data/llm \
    --rareprognois-root /data/RarePrognois \
    --cv-folds 5
```

## Step-by-Step Usage

### Step 0: Data Preparation

[`prepare_data.py`](prepare_data.py) converts raw case outputs and LLM predictions into S1-format CSVs and the directory structure expected by downstream scripts.

```bash
python -m rare_prognosis.training.prepare_data \
    --case-root /data/case_output \
    --llm-root /data/llm \
    --out-dir /data/prepared
```

### Step 1: Feature Engineering

[`build_features.py`](build_features.py) constructs stacking features from multi-model outputs.

```bash
python -m rare_prognosis.training.build_features \
    --rareprognosis-root /data/prepared/rareprognosis \
    --models-root /data/prepared/models \
    --train-ids /data/prepared/dataset/train_case_ids.json \
    --test-ids /data/prepared/dataset/test_case_ids.json \
    --out-dir /data/prepared/features
```

### Step 2: Training

[`train_models.py`](train_models.py) trains a `GradientBoostingClassifier` per sub-task with 5-fold StratifiedKFold OOF evaluation. 

```bash
python -m rare_prognosis.training.train_models \
    --features-dir /data/prepared/features \
    --out-dir /data/prepared/trained_models \
    --seed 42 \
    --cv-folds 5
```

### Step 3: Inference

[`infer_models.py`](infer_models.py) loads trained model bundles, builds per-case features, and writes averaged ensemble predictions to S1 CSVs.

```bash
python -m rare_prognosis.training.infer_models \
    --rareprognosis-root /data/prepared/rareprognosis \
    --models-root /data/prepared/models \
    --train-ids /data/prepared/dataset/train_case_ids.json \
    --test-ids /data/prepared/dataset/test_case_ids.json \
    --models-dir /data/prepared/trained_models
```

### Step 4: Evaluation

**LLM baseline evaluation** ([`eval/eval_llm.py`](../eval/eval_llm.py)):

```bash
python -m rare_prognosis.training.eval.eval_llm \
    --models-root /data/prepared/models \
    --rareprognosis-root /data/prepared/rareprognosis \
    --train-ids /data/prepared/dataset/train_case_ids.json \
    --test-ids /data/prepared/dataset/test_case_ids.json \
    --split test \
    --out-dir /data/prepared/eval_results
```

**ML ensemble evaluation** ([`eval/eval_ml.py`](../eval/eval_ml.py)):

```bash
python -m rare_prognosis.training.eval.eval_ml \
    --rareprognosis-root /data/prepared/rareprognosis \
    --split test \
    --out-dir /data/prepared/eval_results
```

## Evaluation Metrics

**Core metrics:**

| Metric | Description |
|---|---|
| Accuracy | Overall classification accuracy |
| MCC | Matthews Correlation Coefficient |
| Macro F1 | Macro-averaged F1 score |
| Balanced Accuracy | Mean of per-class recalls |
| Per-class Recall | Recall breakdown by label |

**Secondary metrics:**

| Metric | Description |
|---|---|
| Severe Recall | Recall on severe-outcome labels (progression/terminal, severe, persistent_severe) |
| False Optimism | 1 − severe recall (rate of missed severe cases) |
| MAOE | Mean Absolute Ordinal Error (ordinal distance between prediction and ground truth) |
| OBI | Optimism Bias Index (signed ordinal error; positive = over-optimistic prediction) |
| All3 Accuracy | Fraction of cases where all three sub-tasks are predicted correctly |
