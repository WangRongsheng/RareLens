# RareAlert

## Overview

We provide the training and evaluation pipeline for rare disease risk scoring from structured patient consultation data. For methodological details, please refer to the paper.

## Pipeline

```
Step 1: Fine-tuning    →  LoRA SFT on Qwen3-32B via LLaMA-Factory
Step 2: Inference      →  Online serving with fine-tuned model
Step 3: Evaluation     →  AUC, Sensitivity, Specificity, F1, F2, MCC, …
```

## Quick Start

```bash
# Evaluate
python -m rare_alert.training.eval.eval_alert \
    --output-root /data/output \
    --rare-dir    /data/gt/rare \
    --nonrare-dir /data/gt/nonrare \
    --threshold 30 \
    --out-json    /data/results/alert_metrics.json
```

## Step-by-Step Usage

### Step 1: Fine-tuning

We fine-tune Qwen3-32B using [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) (LoRA SFT). Fine-tuning hyperparameters and model weights are not released.

### Step 2: Inference

[`inference.py`](inference.py) implements `RiskStage`, a stateless pipeline for API serving (no file I/O).

```python
from core_tool.config import RiskStageConfig
from rare_alert.training.inference import RiskStage

cfg = RiskStageConfig(
    base_url="http://localhost:8000/v1",
    model="rare_alert",
    api_key="EMPTY",
)
stage = RiskStage(config=cfg)

result = stage.run_sync(payload_json_str)
# result: dict with keys risk_score, key_insights, risk_explanation
```

Supports guided JSON (vLLM), streaming, and parse-failure retries. For async usage: `await stage.run(payload_json_str)`.

### Step 3: Evaluation

[`eval/eval_alert.py`](eval/eval_alert.py) evaluates predictions by comparing `risk_score` against ground-truth rare/non-rare labels.

```bash
python -m rare_alert.training.eval.eval_alert \
    --output-root /data/output \
    --rare-dir    /data/gt/rare \
    --nonrare-dir /data/gt/nonrare \
    --threshold 30 \
    --out-json    /data/results/alert_metrics.json
```

## Evaluation Metrics

We compute metrics at four thresholds: fixed (user-specified), and optimal thresholds selected via Youden's J, F1, and F2 grid search.

| Metric | Description |
|---|---|
| AUC | ROC Area Under the Curve |
| Accuracy | (TP+TN) / total |
| Balanced Accuracy | (Sensitivity + Specificity) / 2 |
| Sensitivity (Recall) | TP / (TP+FN) |
| Specificity | TN / (TN+FP) |
| PPV (Precision) | TP / (TP+FP) |
| NPV | TN / (TN+FN) |
| F1 | Harmonic mean of Precision and Recall |
| F2 | Recall-weighted F-score (beta=2) |
| MCC | Matthews Correlation Coefficient |
