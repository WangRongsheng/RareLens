# RareAlert

## Overview

We provide the training pipeline for rare disease risk scoring from structured patient consultation data. For methodological details, please refer to the paper.

## Pipeline

```
Step 1: Fine-tuning    →  LoRA SFT on Qwen3-32B via LLaMA-Factory
Step 2: Inference      →  Online serving with fine-tuned model
```

## Step-by-Step Usage

### Step 1: Fine-tuning

We fine-tune Qwen3-32B using [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) (LoRA SFT). Fine-tuning hyperparameters and model weights are not released.

### Step 2: Inference

[`inference.py`](inference.py) implements `RiskStage`, a stateless pipeline for API serving (no file I/O).

```python
from rare_alert.training.inference import RiskStage, RiskStageConfig

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

## File Structure

| File | Description |
|------|-------------|
| [`inference.py`](inference.py) | `RiskStage` pipeline, config classes, prompt template, guided JSON schema |
| [`llm_client.py`](llm_client.py) | OpenAI-compatible LLM client with retry, streaming, and `.env` loading |
| [`output_parser.py`](output_parser.py) | Multi-strategy JSON extraction and risk output parsing |
| [`data_template.md`](data_template.md) | SFT data format spec (single- / multi-hypothesis) |
| [`data_template.jsonl`](data_template.jsonl) | Copy-paste-ready training example (1 and 10 hypotheses) |
