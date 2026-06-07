# RareTreatment — Treatment Ranking Module

## Overview

We provide the training and evaluation pipeline for treatment plan ranking. For methodological details, please refer to the paper.

## Pipeline

```
Step 0: Data Preparation     →  Organized directory structure for downstream steps
Step 1: LLM Generation       →  treatment_plan_output.json per model per case
Step 2: Feature Engineering   →  features_{train,test}.csv 
Step 3: Training + Inference  →  XGBoost GroupKFold ranker → ensemble predictions
Step 4: Evaluation            →  Hit@K, nDCG, MRR, MAP + secondary metrics (M2–M11)
```

## Quick Start

```bash
# One-click reproduction
bash rare_treatment/training/run_pipeline.sh \
    --python /path/to/python \
    --case-root /data/case_output \
    --llm-output-root /data/treatment_llm \
    --n-splits 5
```

## Step-by-Step Usage

### Step 0: Data Preparation

[`prepare_data.py`](prepare_data.py) converts raw case outputs and LLM predictions into the directory structure expected by downstream scripts.

```bash
python -m rare_treatment.training.prepare_data \
    --case-root /data/case_output \
    --llm-output-root /data/treatment_llm \
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
    --drop-feature-groups stage3_eval,model_support \
    --save-models
```

Standalone inference with trained models ([`infer_ranker.py`](infer_ranker.py)):

```bash
python -m rare_treatment.training.infer_ranker \
    --model-dir /data/models/models \
    --test-csv /data/features/features_test.csv \
    --out-dir /data/results
```

### Step 4: Evaluation

**LLM-as-judge scoring** ([`eval/run_judge.py`](eval/run_judge.py)): a judge LLM evaluates each treatment for appropriateness, completeness, helpfulness, and safety.

```bash
python -m rare_treatment.training.eval.run_judge \
    --pred-root /data/llm_outputs/qwen3-32b \
    --gt-root /data/gt/treat \
    --out-root /data/scores/qwen3-32b \
    --base-url https://api.openai.com/v1 \
    --api-key $OPENAI_API_KEY
```

**LLM baseline evaluation** ([`eval/eval_llm.py`](eval/eval_llm.py)):

```bash
python -m rare_treatment.training.eval.eval_llm \
    --score_root /data/treatment_score \
    --train_ids /data/dataset/train_cases.json \
    --test_ids /data/dataset/test_cases.json
```

**ML ranking evaluation** ([`eval/eval_ml.py`](eval/eval_ml.py)):

```bash
python -m rare_treatment.training.eval.eval_ml \
    --input ensemble=/data/results/test_predictions.csv:score \
    --ks 1,2,3,5
```

**Secondary metrics** ([`eval/eval_secondary_metrics.py`](eval/eval_secondary_metrics.py)): M2–M11 metrics comparing the ML reranker against individual LLM baselines.

## Evaluation Metrics

| Metric | Description |
|---|---|
| Hit@K | Fraction of cases with at least one appropriate treatment in top-K |
| nDCG@K | Normalized Discounted Cumulative Gain |
| MRR | Mean Reciprocal Rank |
| MAP | Mean Average Precision |
| M3–M6 | Appropriate/inappropriate counts, case positive probability, high-risk ratio |
| M7–M9 | Completeness, helpfulness, safety scores |
| M10 | Performed fraction |
| M11 | Rescue rate (reranker succeeds where baseline fails) |
