# RareLens

<!-- TODO[badges]: Add links once paper/dataset/project page are finalized.
<div style='display:flex; gap: 0.6rem; '>
<a href='ARXIV_LINK'><img src='https://img.shields.io/badge/Paper-PDF-red'></a>
<a href='PROJECT_HOMEPAGE'><img src='https://img.shields.io/badge/Project-Homepage-pink'></a>
<a href='LICENSE'><img src='https://img.shields.io/badge/License-Apache--2.0-lightgrey'></a>
</div>
-->

## Overview

<!-- TODO[overview]: Brief description of RareLens.
![](./figs/architecture.png)
-->

## Web Application

<!-- TODO[webapp]: Live URL, backend stack, deployed modules. -->

## Demo

<!-- TODO[demo]:
![Demo](video/rarelens_demo.gif)
-->
---

The following sections provide instructions for reproducing the model training and evaluation reported in the paper. For the full end-to-end clinical pipeline, see the [Web Application](#web-application) above.

## Modules

RareLens spans the four canonical stages of the rare-disease clinical workflow — risk alerting, diagnosis, treatment selection, and prognosis — each instantiated as an independently trainable and evaluable module.

| Module | Path | Task | Approach |
| --- | --- | --- | --- |
| Alert | `rare_alert/` | Rare disease risk scoring | End-to-end scoring via fine-tuned LLM (Qwen3-32B, LoRA SFT) |
| Diagnosis | `rare_diagnosis/` | Candidate disease ranking (primary / follow-up) | Learning-to-rank over multi-LLM-generated candidates (XGBoost, GroupKFold) |
| Treatment | `rare_treatment/` | Treatment plan ranking | Learning-to-rank over multi-LLM-generated candidates (XGBoost, GroupKFold) |
| Prognosis | `rare_prognosis/` | Outcome / functional status / symptom burden prediction | Stacking ensemble over multi-LLM predictions (GBDT, 5-fold CV) |

The Diagnosis, Treatment, and Prognosis modules share a common paradigm: **multi-LLM generation → feature engineering → ML training → evaluation**, differing in feature design, label schema, and metrics.

## System Requirements

### Hardware
- **RAM**: Minimum 16GB (32GB recommended)
- **Storage**: ~5GB for code, dataset, and model artifacts; additional ~80GB for local Qwen3-32B deployment (Alert module)
- **GPU**: Optional for ML modules; required for local Alert model deployment (48GB+ VRAM recommended)
- **CPU**: Any modern 64-bit processor

**Note:** The Alert module can also be accessed via API without local GPU. ML modules (Diagnosis, Treatment, Prognosis) run on CPU.

### Software
- **OS**: Any 64-bit operating system
- **Python**: 3.10+

## LLM API Key Requirements

Each module's `generate_llm_outputs.py` calls LLMs via the OpenAI-compatible API, configured either by command-line flags (`--base-url`, `--api-key`, `--models`) or a JSON config file (`--config llm_config.json`).

Supported providers:

| Provider | Models | Sign up |
| --- | --- | --- |
| OpenAI | gpt-5, gpt-4o-mini, gpt-3.5-turbo, o3-mini | [platform.openai.com](https://platform.openai.com) |
| Google | gemini-2.5-flash-preview-05-20-nothinking | [ai.google.dev](https://ai.google.dev) |
| Qwen | qwen3-8b, qwen3-14b, qwen3-32b, qwen3-235b-a22b-instruct-2507 | [dashscope.aliyun.com](https://dashscope.aliyun.com) |
| DeepSeek | deepseek-r1-0528, deepseek-v3 | [platform.deepseek.com](https://platform.deepseek.com) |
| Anthropic | claude-haiku-4-5-20251001 | [console.anthropic.com](https://console.anthropic.com) |

All providers use the OpenAI-compatible protocol and can be pointed to a local vLLM / Ollama / SGLang server.

## Installation

```bash
git clone <REPO_URL>
cd RareLens
pip install -r requirements.txt
```

## Dataset

For reproducibility, we release a 500-case demo subset ([`data_500/`](data_500/)) of the full RareBench cohort used in our experiments. See [`data_500/README.md`](data_500/README.md) for format details.

## Reproduction Instructions

Each module provides a one-click training pipeline and a standalone inference script.

### Alert

Fine-tuned Qwen3-32B via [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) (LoRA SFT). 

**Inference:**

```bash
python -c "
from rare_alert.training.inference import RiskStage, RiskStageConfig
cfg = RiskStageConfig(base_url='http://localhost:8000/v1', api_key='EMPTY', model='rare_alert')
stage = RiskStage(cfg)
result = stage.run_sync(open('/data/case/primary_consultation.json').read())
print(result)
"
```

See [`rare_alert/training/README.md`](rare_alert/training/README.md) for details.

### Diagnosis

**Training:**

```bash
bash rare_diagnosis/training/reproduce_diag.sh \
    --python /path/to/python \
    --visit-type primary \
    --query-root /data/query \
    --gt-root /data/scores \
    --models-root /data/llm_outputs \
    --train-ids /data/splits/train.json \
    --test-ids /data/splits/test.json
```

**Inference:**

```bash
python -m rare_diagnosis.training.infer_ranker \
    --input-dir /data/features \
    --model-dir /data/models/primary/models \
    --config rare_diagnosis/training/best_hyperopt_config_primary.json \
    --out-dir /data/inference_output
```

See [`rare_diagnosis/training/README.md`](rare_diagnosis/training/README.md) for details.

### Treatment

**Training:**

```bash
bash rare_treatment/training/run_pipeline.sh \
    --python /path/to/python \
    --case-root /data/case_output \
    --llm-output-root /data/treatment_llm
```

**Inference:**

```bash
python -m rare_treatment.training.infer_ranker \
    --model-dir /data/models/models \
    --test-csv /data/features/features_test.csv \
    --out-dir /data/inference_output
```

See [`rare_treatment/training/README.md`](rare_treatment/training/README.md) for details.

### Prognosis

**Training :**

```bash
bash rare_prognosis/training/run_pipeline.sh \
    --python /path/to/python \
    --case-root /data/case_output \
    --llm-root /data/llm
```

**Inference:**

```bash
python -m rare_prognosis.training.infer_models \
    --rareprognosis-root /data/rareprognosis \
    --models-root /data/models \
    --train-ids /data/dataset/train_case_ids.json \
    --test-ids /data/dataset/test_case_ids.json \
    --models-dir /data/trained_models \
    --task all
```

See [`rare_prognosis/training/README.md`](rare_prognosis/training/README.md) for details.

## Citation

<!-- TODO[citation]:
```bibtex
@article{TODO,
  title   = {TODO},
  author  = {TODO},
  journal = {TODO},
  year    = {TODO}
}
```
-->

## License

This project is released under the Apache License 2.0. See [`LICENSE`](LICENSE) for details.

## Acknowledgement

<!-- TODO[ack]: Acknowledge public datasets, foundation models, and tools used. -->
