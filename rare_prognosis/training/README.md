# Prognosis Prediction - Reproduction Pipeline

This directory contains self-contained scripts to reproduce the prognosis prediction experiments. The pipeline covers 3 sub-tasks: `overall_outcome`, `functional_status`, `symptom_burden`.

## Pipeline Overview

```
Step 0: Data Preparation        ->  S1 CSVs, model dirs, train/test IDs
Step 1: Feature Engineering     ->  features.train.csv, features.test.csv (per task)
Step 2: Train OOF GBDT Models   ->  5-fold stacking models (.pkl)
Step 3: Inference               ->  S1 CSV predictions
Step 4: Evaluate LLM Baselines  ->  per-model core + secondary metrics
Step 5: Evaluate ML Ensemble    ->  ensemble core + secondary metrics
```

## Quick Start

```bash
# One-click pipeline (demo data, 6 cases)
bash run_pipeline.sh --python D:/Anaconda/envs/raredis/python.exe

# Full data with 5-fold CV
bash run_pipeline.sh \
    --case-root /data/case_output \
    --llm-root /data/llm \
    --rareprognois-root /data/RarePrognois \
    --cv-folds 5 \
    --python python
```

## Directory Structure

```
rare_prognosis/
├── training/
│   ├── run_pipeline.sh             # One-click reproduction script
│   ├── prepare_data.py             # Step 0: Prepare data layout
│   ├── build_features.py           # Step 1: Build stacking features
│   ├── train_models.py             # Step 2: Train OOF GBDT models
│   ├── infer_models.py             # Step 3: Inference with trained models
│   ├── generate_llm_outputs.py     # Batch LLM output generation
│   ├── llm_generation.py           # Core LLM generation logic
│   ├── data_io.py                  # Shared: task config, I/O, label normalization
│   ├── ensemble_utils.py           # Shared: voting, ML, CV, feature encoding
│   └── README.md
├── eval/
│   ├── eval_llm.py                 # Step 4: Evaluate individual LLM models
│   ├── eval_ml.py                  # Step 5: Evaluate ML ensemble
│   └── metrics.py                  # Shared: metric computation
└── models/
    ├── overall_outcome_C2_stacking_gbdt.pkl
    ├── functional_status_C2_stacking_gbdt.pkl
    └── symptom_burden_C2_stacking_gbdt.pkl
```

## Step 0: Prepare Data

Converts raw case outputs and LLM predictions into the expected directory layout.

```bash
python prepare_data.py \
    --case-root data_demo/case_output \
    --llm-root data_demo/pipeline_data/prognoisis/llm \
    --rareprognois-root data_demo/pipeline_data/prognoisis/RarePrognois \
    --out-dir prepared
```

**Output:**
- `rareprognosis/{overall,funcational,symptom}/S1_*.csv` (GT labels + split)
- `models/<model>/<case_id>/prognosis_prediction_output.json`
- `dataset/train_case_ids.json`, `dataset/test_case_ids.json`

## Step 1: Build Features

```bash
python build_features.py \
    --rareprognosis-root prepared/rareprognosis \
    --models-root prepared/models \
    --train-ids prepared/dataset/train_case_ids.json \
    --test-ids prepared/dataset/test_case_ids.json \
    --out-dir prepared/features
```

**Output per task:** `features.train.csv`, `features.test.csv`, `meta.json`

Feature structure: `n_models x n_labels` one-hot base features + 15 explanation text features (5 stats + 10 keyword counts), standardized using train statistics.

## Step 2: Train OOF GBDT Models

All 3 tasks use `GradientBoostingClassifier` with `random_state=42`, default sklearn params. Training uses 5-fold StratifiedKFold OOF for evaluation, saving all fold models for deployment.

```bash
python train_models.py \
    --features-dir prepared/features \
    --out-dir prepared/trained_models \
    --seed 42 \
    --cv-folds 5
```

**Output:** `{task}_C2_stacking_gbdt.pkl` per task

**Bundle format:** `{"model": [fold_0, ..., fold_4], "meta": {base_model_names, labels, class_list, expl_keywords, expl_standardize, ...}}`

Inference uses averaged `predict_proba` across all fold models, matching the OOF evaluation behavior.

## Step 3: Inference

```bash
python infer_models.py \
    --rareprognosis-root prepared/rareprognosis \
    --models-root prepared/models \
    --train-ids prepared/dataset/train_case_ids.json \
    --test-ids prepared/dataset/test_case_ids.json \
    --models-dir prepared/trained_models
```

Writes predictions back to S1 CSVs.

## Step 4-5: Evaluation

### Evaluate LLM Baselines

```bash
python ../eval/eval_llm.py \
    --models-root prepared/models \
    --rareprognosis-root prepared/rareprognosis \
    --train-ids prepared/dataset/train_case_ids.json \
    --test-ids prepared/dataset/test_case_ids.json \
    --split test \
    --out-dir prepared/eval_results
```

### Evaluate ML Ensemble

```bash
python ../eval/eval_ml.py \
    --rareprognosis-root prepared/rareprognosis \
    --split test \
    --out-dir prepared/eval_results
```

### Metrics Computed

**Core metrics:** accuracy, MCC, macro F1, balanced accuracy, per-class recall

**Secondary metrics:**
| Metric | Description |
|--------|-------------|
| Severe recall | Recall for severe classes (progression/terminal, severe, persistent_severe) |
| False optimism | 1 - severe recall (missed severe cases) |
| MAOE | Mean Absolute Ordinal Error (ordinal distance between pred and GT) |
| OBI | Optimism Bias Index (signed ordinal error, positive = over-optimistic) |
| All3 accuracy | Fraction of cases where all 3 tasks are correct simultaneously |

## Dependencies

```
numpy
scikit-learn
```
