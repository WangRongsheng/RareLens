# RareDiagnosis

## Overview

We provide the training pipeline for rare disease diagnosis ranking, supporting both primary consultation and follow-up visit stages. For methodological details, please refer to the paper.

## Pipeline

```
Step 0: RAG Cache (optional)   →  FAISS vector index for OrphaCode resolution
Step 1: LLM Generation         →  per-model diagnosis outputs + OrphaCode mapping
Step 2: Feature Engineering     →  features.{train,test}.csv (56+ features per candidate)
Step 3: XGBoost Training        →  GroupKFold CV ranker (rank:ndcg)
```

## Quick Start

Build features → train (feature engineering + ranking). The per-model LLM outputs
(`--models-root`) come from Step 1 below; the judge scores (`--gt-root`, the ground
truth) follow the paper's method (see the note under the table). Both must exist
before running.

```bash
# Primary stage
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

| Flag | Holds |
| --- | --- |
| `--query-root`  | raw cases (`<case>/primary_consultation.json`) |
| `--models-root` | per-model LLM outputs (`<model>/<case>/…`) |
| `--gt-root`     | per-model judge scores (GT) |

> The judge scores (the ground truth used here) are produced by LLM-as-judge
> evaluation following the method described in the paper. The scoring code is not
> included in this repository — please refer to the paper to reproduce them.

The per-model output filename differs by stage and is handled automatically; override
with `--primary-fname` / `--gt-fname` if needed. Run feature extraction on CPU with
`--num-gpus 0` if GPU workers fail.

## Step-by-Step Usage

### Step 1: LLM Generation

[`generate_llm_outputs.py`](generate_llm_outputs.py) queries LLMs to produce top-5 diagnoses per case. Each diagnosis is optionally enriched with an OrphaCode via semantic retrieval ([`orphacode_rag.py`](orphacode_rag.py)).

The first two arguments are positional (`input_folder` `output_folder`), and `--model` runs **one** model per invocation — loop over models to produce the multi-LLM outputs.

```bash
# Direct API mode (one model per run)
python -m rare_diagnosis.training.generate_llm_outputs \
    /data/query  /data/llm_outputs \
    --model gpt-4o-mini \
    --base-url https://api.openai.com/v1 \
    --api-key $OPENAI_API_KEY \
    --visit-type primary \
    --num-workers 8

# Config-file mode (multiple providers in one JSON list; still one --model per run)
for m in gpt-5 o3-mini gpt-3.5-turbo gpt-4o-mini; do
    python -m rare_diagnosis.training.generate_llm_outputs \
        /data/query  /data/llm_outputs \
        --model "$m" --config llm_config.json \
        --visit-type primary --num-workers 8
done

# With OrphaCode RAG enrichment
python -m rare_diagnosis.training.generate_llm_outputs \
    /data/query  /data/llm_outputs \
    --model gpt-4o-mini --config llm_config.json \
    --visit-type primary --num-workers 8 \
    --enable-orphacode-rag \
    --rag-ontology-path rare_diagnosis/training/orphanet_hierarchy.json
```

### Step 2: Feature Engineering

[`build_features_primary.py`](build_features_primary.py) and [`build_features_followup.py`](build_features_followup.py) construct ranking features per candidate from multi-model outputs.

> Downloads `pritamdeka/S-PubMedBert-MS-MARCO` (semantic features) from HuggingFace on
> first run — see the main README's *Feature-engineering models* note for the mirror /
> pre-cache and the `--num_gpus 0 --workers 2` CPU fallback if GPU workers crash.

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

