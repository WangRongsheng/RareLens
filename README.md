# RareLens

<!-- TODO[badges]: Add links once paper/dataset/project page are finalized.
<a href='ARXIV_LINK'><img src='https://img.shields.io/badge/Paper-PDF-red'></a>
<a href='HUGGINGFACE_DATASET_LINK'><img src='https://img.shields.io/badge/RareBench-Dataset-blue'></a>
<a href='PROJECT_HOMEPAGE'><img src='https://img.shields.io/badge/Project-Homepage-pink'></a>
<a href='LICENSE'><img src='https://img.shields.io/badge/License-Apache--2.0-lightgrey'></a>
-->

## Overview

<!-- TODO[overview]: One-line positioning of RareLens. Should cover:
1) Target clinical tasks for rare diseases (diagnosis / prognosis / treatment)
2) Core method (heterogeneous LLM reasoning alignment + ML calibration)
3) How it differs from prior work (not an agent; multi-task, reproducible, locally deployable training pipelines)
-->

<!-- TODO[architecture-figure]: Optional system architecture diagram.
![](./figs/architecture.png)
-->

## Demo

<!-- TODO[demo]: Record a GIF and place it under figs/ or video/.
![Demo](video/rarelens_demo.gif)
(Suggested: show a minimal end-to-end flow from case input to module output on the web UI)
-->

## Web Application

<!-- TODO[webapp]: Confirm with team and fill in:
1) Live URL for the web demo
2) Authentication / target audience (researchers / clinical collaborators)
3) Backend stack (e.g. FastAPI + local/cloud LLM)
4) Which modules are currently deployed
-->

## Modules

RareLens is organized into four independent modules along the clinical workflow. Each module has its own training pipeline and model artifacts:

| Module | Path | Task | Method |
| --- | --- | --- | --- |
| Alert | `rare_alert/` | Rare disease risk scoring from primary consultation data | Fine-tuned Qwen3-32B (LLaMA-Factory SFT) |
| Diagnosis | `rare_diagnosis/` | Candidate disease ranking and final diagnosis inference (primary / followup) | XGBoost LTR (`rank:ndcg`) + GroupKFold CV |
| Prognosis | `rare_prognosis/` | Overall outcome / functional status / symptom burden prediction | GBDT 5-fold stacking ensemble |
| Treatment | `rare_treatment/` | Treatment plan candidate ranking | XGBoost LTR + GroupKFold CV |

The Alert module is the first stage in the pipeline and uses a fine-tuned LLM directly for end-to-end risk scoring. The Diagnosis, Prognosis, and Treatment modules follow the same paradigm: **multi-LLM generation → feature engineering → ML training → evaluation**, but differ in feature design, label schema, and evaluation metrics.

Shared components:

- `core_tool/`: LLM client, prompt templates, JSON parser
- `schema/`: Pydantic I/O schemas shared across all modules

## System Requirements

### Hardware

<!-- TODO[hardware]: Confirm values with team. -->

- **RAM**: Minimum 16GB (32GB recommended)
- **Storage**: <!-- TODO: e.g. 100GB+ free disk space (including RareBench data) -->
- **GPU**: Optional. XGBoost training supports `--use-gpu`; SentenceTransformer feature extraction benefits from GPU
- **CPU**: Any modern 64-bit processor

### Software

- **OS**: Any 64-bit operating system (Linux / macOS / Windows)
- **Python**: 3.10+
- **Dependencies**: see `requirements.txt`

## LLM API Key Requirements

The LLM generation step in each module uses the OpenAI-compatible API. Each module's `generate_llm_outputs.py` supports two configuration modes:

1. **Direct**: `--base-url` + `--api-key` + `--models`
2. **Config file**: `--config llm_config.json` (JSON list, each entry contains `base_url`, `api_key`, `models`)

Supported LLM providers:

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

Key dependencies:

```
openai            # LLM API calls
numpy, pandas     # Data processing
xgboost           # ML ranking models
scikit-learn      # Cross-validation, metrics
sentence-transformers  # Semantic features (Diagnosis module)
faiss-cpu         # OrphaCode RAG vector retrieval (optional)
openpyxl          # Excel output (optional)
tqdm
```

## Dataset

<!-- TODO[dataset]: Confirm RareBench release format with team. Should cover:
1) Dataset homepage / HuggingFace link
2) Download instructions (example command)
3) Directory structure (dataset/{diagnosis,prognosis,treatment}/...)
4) Data license
Example:
huggingface-cli download <ORG>/RareBench --repo-type=dataset --local-dir ./dataset
-->

A demo dataset is provided in `data_500/` with 500 cases (primary consultation, diagnosis, and treatment outcome records), suitable for small-scale evaluation and reproducibility checks.

## Usage

Each module provides two types of entry points:

- **Reproducible Training** — `rare_<task>/training/`: Builds features and trains with fixed best hyperparameters to reproduce the models reported in the paper.
- **Standalone Inference** — `rare_<task>/training/infer_*.py`: Loads trained models and runs inference on new data.

> End-to-end chaining (diagnosis → treatment → prognosis) is not yet released.

### Reproducible Training

#### Alert

The Alert module uses a fine-tuned Qwen3-32B model trained via [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) (LoRA SFT). Fine-tuning hyperparameters and model weights are not released.

Evaluation:

```bash
python -m rare_alert.training.eval.eval_alert \
    --output-root /data/pipeline_output \
    --rare-dir    /data/gt/rare \
    --nonrare-dir /data/gt/nonrare \
    --threshold 30 \
    --out-json    /data/results/alert_metrics.json
```

Metrics: AUC, Accuracy, Sensitivity, Specificity, Balanced Accuracy, F1, F2, MCC, PPV, NPV (at fixed + optimal thresholds via Youden / F1 / F2).

See [`rare_alert/training/README.md`](rare_alert/training/README.md) for details.

#### Diagnosis

```bash
# Primary visit — full pipeline (build features → train XGBoost → eval ML → eval LLM)
bash rare_diagnosis/training/reproduce_diag.sh \
    --python /path/to/python \
    --visit-type primary \
    --query-root /data/query \
    --gt-root /data/scores \
    --models-root /data/llm_outputs \
    --train-ids /data/splits/train.json \
    --test-ids /data/splits/test.json

# Follow-up visit
bash rare_diagnosis/training/reproduce_diag.sh --visit-type followup ...
```

Pipeline: build features (primary/followup) → XGBoost `rank:ndcg` with GroupKFold → eval ML + LLM baselines.

See [`rare_diagnosis/training/README.md`](rare_diagnosis/training/README.md) for details.

#### Prognosis

```bash
# One-click pipeline (3 sub-tasks: overall_outcome, functional_status, symptom_burden)
bash rare_prognosis/training/run_pipeline.sh \
    --python /path/to/python \
    --case-root /data/case_output \
    --llm-root /data/llm \
    --rareprognois-root /data/RarePrognois
```

Pipeline: prepare data → build stacking features → train OOF GBDT models → inference → eval LLM + ML.

See [`rare_prognosis/training/README.md`](rare_prognosis/training/README.md) for details.

#### Treatment

```bash
# One-click pipeline
bash rare_treatment/training/run_pipeline.sh \
    --python /path/to/python \
    --case-root /data/case_output \
    --llm-output-root /data/treatment_llm
```

Pipeline: prepare data → build L2R features → XGBoost ranker with GroupKFold → inference → eval.

See [`rare_treatment/training/README.md`](rare_treatment/training/README.md) for details.

### Standalone Inference

After training (or using pre-trained weights), run inference on new data:

```bash
# Alert: call fine-tuned model via OpenAI-compatible endpoint
python -c "
from rare_alert.training.inference import RiskStage, RiskStageConfig
cfg = RiskStageConfig(base_url='http://localhost:8000/v1', api_key='EMPTY', model='rare_alert')
stage = RiskStage(cfg)
result = stage.run_sync(open('/data/case/primary_consultation.json').read())
print(result)
"

# Diagnosis: load fold models, predict on feature CSV
python -m rare_diagnosis.training.infer_ranker \
    --input-dir /data/features \
    --model-dir /data/models/primary/models \
    --config rare_diagnosis/training/best_hyperopt_config_primary.json \
    --out-dir /data/inference_output

# Prognosis: load trained models, predict
python -m rare_prognosis.training.infer_models \
    --input-dir /data/features \
    --model-dir /data/models \
    --out-dir /data/inference_output

# Treatment: load fold models, predict on feature CSV
python -m rare_treatment.training.infer_ranker \
    --input-dir /data/features \
    --model-dir /data/models \
    --out-dir /data/inference_output
```

## Evaluation

Each module provides independent evaluation scripts under `eval/`:

| Module | Eval Script | Metrics |
| --- | --- | --- |
| Alert | `eval_alert.py` | AUC, Acc, Sensitivity, Specificity, Balanced Acc, F1, F2, MCC, PPV, NPV |
| Diagnosis | `eval_llm.py`, `eval_ml.py` | Acc@1/3/5, MRR, NDCG@1/3/5 |
| Prognosis | `eval_llm.py`, `eval_ml.py` | Core + secondary metrics (per sub-task) |
| Treatment | `eval_llm.py`, `eval_ml.py` | Hit@K, nDCG, MRR, MAP |

```bash
# Example: Alert eval
python -m rare_alert.training.eval.eval_alert \
    --output-root /data/pipeline_output \
    --rare-dir /data/gt/rare \
    --nonrare-dir /data/gt/nonrare \
    --threshold 30

# Example: Diagnosis ML eval
python -m rare_diagnosis.training.eval.eval_ml \
    --json /data/models/primary/test_predictions_ranked.json \
    --out-csv /data/results/ml_metrics.csv

# Example: Diagnosis LLM eval (with Excel output)
python -m rare_diagnosis.training.eval.eval_llm \
    --score-root /data/scores \
    --test-ids /data/splits/test.json \
    --out-dir /data/results --excel
```

## Project Layout

```
RareLens/
├── rare_diagnosis/                  # Diagnosis module
│   ├── training/
│   │   ├── reproduce_diag.sh        #   One-click reproduction script
│   │   ├── generate_llm_outputs.py  #   LLM generation (openai API)
│   │   ├── orphacode_rag.py         #   OrphaCode RAG enrichment
│   │   ├── build_features_primary.py#   Feature engineering (primary)
│   │   ├── build_features_followup.py#  Feature engineering (followup)
│   │   ├── train_ranker.py          #   XGBoost ranker training
│   │   ├── infer_ranker.py          #   Standalone inference
│   │   └── eval/                    #   Evaluation (eval_llm, eval_ml, metrics)
│   ├── orphacode_rag_cache/         #   Pre-built FAISS index
│   └── tools/                       #   Utilities (build_orphacode_rag_cache.py)
├── rare_prognosis/                  # Prognosis module
│   └── training/
│       ├── run_pipeline.sh          #   One-click reproduction script
│       ├── generate_llm_outputs.py  #   LLM generation
│       ├── prepare_data.py          #   Data preparation
│       ├── build_features.py        #   Stacking feature engineering
│       ├── train_models.py          #   OOF GBDT training
│       ├── infer_models.py          #   Inference
│       └── eval/                    #   Evaluation (eval_llm, eval_ml, metrics)
├── rare_treatment/                  # Treatment module
│   └── training/
│       ├── run_pipeline.sh          #   One-click reproduction script
│       ├── generate_llm_outputs.py  #   LLM generation
│       ├── prepare_data.py          #   Data preparation
│       ├── build_features.py        #   L2R feature engineering
│       ├── train_ranker.py          #   XGBoost ranker training
│       ├── infer_ranker.py          #   Inference
│       └── eval/                    #   Evaluation (eval_llm, eval_ml, metrics)
├── rare_alert/                      # Alert module
│   └── training/
│       ├── inference.py             #   RiskStage inference pipeline (fine-tuned Qwen3-32B)
│       └── eval/
│           └── eval_alert.py        #   Evaluation (AUC, Sensitivity, Specificity, F1, F2, MCC, …)
├── core_tool/                       # Shared: LLM client, prompt templates, JSON parser
├── schema/                          # Shared: Pydantic I/O schemas
├── data_500/                        # Demo data (500 cases, primary consultation + diagnosis + treatment outcome)
├── requirements.txt
├── LICENSE
└── README.md
```

## Citation

<!-- TODO[citation]: Replace with correct BibTeX once the paper is published / on arXiv.
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

<!-- TODO[ack]: Acknowledge public datasets (Orphanet, MIMIC-IV-Ext, etc.), foundation models, and tools used. -->
