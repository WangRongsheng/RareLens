# RareTreatment

## Overview

We provide the training pipeline for treatment plan ranking. For methodological details, please refer to the paper.

## Pipeline

```
Step 0: Data Preparation     →  Organized directory structure for downstream steps
Step 1: LLM Generation       →  treatment_plan_output.json per model per case
Step 2: Feature Engineering   →  features_{train,test}.csv 
Step 3: Training + Inference  →  XGBoost GroupKFold ranker → ensemble predictions
```

## Quick Start

Prepare → build features → train → infer. The per-model LLM plans
(`--llm-root`) come from Step 1 below; the judge scores (`--score-root`, the
ground truth) follow the paper's method (see the note under the table). Both must
exist before running.

```bash
bash rare_treatment/training/run_pipeline.sh \
    --python /path/to/python \
    --case-root /data/case_output \
    --llm-root /data/treatment_llm \
    --score-root /data/treatment_scores \
    --train-ids /data/splits/train.json \
    --test-ids /data/splits/test.json \
    --n-splits 5
```

| Flag | Holds |
| --- | --- |
| `--case-root`        | raw cases (`<case>/treatment_plan.json`, flat or `1_raw_data/`) |
| `--llm-root`  | per-model LLM plans (`<model>/<case>/treatment_plan_output.json`) |
| `--score-root`       | per-model judge scores (`<model>/<case>/treatment_score.json`) |

> The judge scores (the ground truth used here) are produced by LLM-as-judge
> evaluation following the method described in the paper. The scoring code is not
> included in this repository — please refer to the paper to reproduce them.

Without `--score-root`, scores are split from the legacy
`<case>/5_treatment/llm_outputs.json` instead. Without `--train-ids/--test-ids`,
all cases are used as both train and test (smoke mode). Hit@1/3/5 + MRR are printed
by the training step. Use `--num-gpus 0` if GPU feature workers fail.

## Step-by-Step Usage

### Step 0: Data Preparation

[`prepare_data.py`](prepare_data.py) converts raw case outputs and LLM predictions into the directory structure expected by downstream scripts.

```bash
python -m rare_treatment.training.prepare_data \
    --case-root /data/case_output \
    --llm-root /data/treatment_llm \
    --out-dir /data/prepared
```

### Step 1: LLM Generation

[`generate_llm_outputs.py`](generate_llm_outputs.py) calls LLMs to generate treatment recommendations per case. 

```bash
# Direct API mode
python -m rare_treatment.training.generate_llm_outputs \
    /data/input /data/output \
    --model qwen3-32b \
    --base-url https://dashscope.aliyuncs.com/compatible-mode/v1 \
    --api-key $API_KEY \
    --num-workers 10

# Config file mode
python -m rare_treatment.training.generate_llm_outputs \
    /data/input /data/output \
    --model gpt-5 \
    --config configs/OAI_Config_List.json
```

### Step 2: Feature Engineering

[`build_features.py`](build_features.py) constructs features from multi-model outputs.

> Downloads `pritamdeka/S-PubMedBert-MS-MARCO` (embeddings) and
> `cross-encoder/nli-deberta-v3-large` (NLI; needs `sentencepiece`) from HuggingFace on
> first run — see the main README's *Feature-engineering models* note for the mirror /
> pre-cache. If GPU feature workers crash, run on CPU with `--num-gpus 0`.

```bash
python -m rare_treatment.training.build_features \
    --plan_root /data/plan_root \
    --treatment_output_root /data/treatment_output \
    --treatment_score_root /data/treatment_score \
    --train_ids /data/dataset/train_cases.json \
    --test_ids /data/dataset/test_cases.json \
    --out_dir /data/features \
    --num_gpus 1
```

### Step 3: Training + Inference

[`train_ranker.py`](train_ranker.py) trains an XGBoost LTR model with GroupKFold (5-fold) cross-validation.

```bash
python -m rare_treatment.training.train_ranker \
    --data-dir /data/features \
    --out-dir /data/models \
    --objective rank:ndcg \
    --n-splits 5 \
    --target-k 3 \
    --drop-feature-groups stage3_eval \
    --save-models
```

Standalone inference with trained models ([`infer_ranker.py`](infer_ranker.py)):

```bash
python -m rare_treatment.training.infer_ranker \
    --model-dir /data/models/models \
    --test-csv /data/features/features_test.csv \
    --out-dir /data/results
```

