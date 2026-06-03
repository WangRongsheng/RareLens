# RareAlert — Risk Alerting Module

This directory contains the inference pipeline for the RareAlert module, which assesses the risk that a patient may have a rare disease.

## Task

Given a patient's structured clinical data (basic information, medical history, physical examination), the model produces:

| Field | Type | Description |
|---|---|---|
| `risk_score` | int (0–100) | Rare disease risk score; 0 = no risk, 100 = near certainty |
| `key_insights` | list of 5 items | Top contributing signs/symptoms with weights (sum to 1) and descriptions |
| `risk_explanation` | str | Plain-language explanation of the risk assessment |

Output schema is defined in [`schema/risk.py`](../../schema/risk.py).

## Model

The RareAlert module is powered by a **fine-tuned Qwen3-32B** model. Fine-tuning was performed using [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) on domain-specific rare disease clinical data with supervised fine-tuning (SFT). The fine-tuned model is served via an OpenAI-compatible API endpoint (e.g., vLLM).

> Fine-tuning hyperparameters and model weights are not released.

## Inference

The inference entry point is [`inference.py`](inference.py), which implements `RiskStage` — a stateless pipeline that accepts a JSON payload and returns a structured `RiskOutput`.

```python
from core_tool.config import RiskStageConfig
from rare_alert.training.inference import RiskStage

cfg = RiskStageConfig(
    base_url="http://localhost:8000/v1",  # vLLM or any OpenAI-compatible endpoint
    model="rare_alert",
    api_key="EMPTY",
)
stage = RiskStage(config=cfg)

# payload: JSON string of RiskInput fields
result = stage.run_sync(payload_json_str)
# result: dict with keys risk_score, key_insights, risk_explanation
```

For async usage, use `await stage.run(payload_json_str)`.

## Directory Structure

```
rare_alert/
├── training/
│   ├── inference.py    # RiskStage inference pipeline
│   └── README.md
└── (model weights not included)
```

## Dependencies

```
openai      # OpenAI-compatible API client
pydantic    # Schema validation
```
