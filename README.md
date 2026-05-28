# RareLens

<!-- TODO[badges]: 论文/数据集/项目主页/License 等链接确定后再填。
示例占位：
<a href='ARXIV_LINK'><img src='https://img.shields.io/badge/Paper-PDF-red'></a>
<a href='HUGGINGFACE_DATASET_LINK'><img src='https://img.shields.io/badge/RareBench-Dataset-blue'></a>
<a href='PROJECT_HOMEPAGE'><img src='https://img.shields.io/badge/Project-Homepage-pink'></a>
<a href='LICENSE'><img src='https://img.shields.io/badge/License-Apache--2.0-lightgrey'></a>
-->

## Overview

<!-- TODO[overview]: 一句话定位 RareLens 是什么。
建议覆盖三点：
1) 项目目标（针对罕见病的哪些临床任务，例如早期风险筛查 / 诊断 / 预后 / 治疗）
2) 方法核心（异质 LLM 推理对齐 + 机器学习校准的整体范式）
3) 与已有工作的差异（不是 agent，是多任务、可复现、可本地部署的训练管线）-->

<!-- TODO[architecture-figure]: 放一张系统总图（可选）
![](./figs/architecture.png)
-->

## Demo

<!-- TODO[demo]: 录一段 gif 放到 figs/ 或 video/ 下后引用，例如：
![Demo](video/rarelens_demo.gif)
（建议：gif 内展示一次完整的 Web 端病例输入 → 任一模块输出的最小流程）
-->

## Web Application

<!-- TODO[webapp]: 跟组员确认网页地址、是否对外开放、技术栈描述。建议覆盖：
1) 在线试用入口（URL）— For easy access without local setup, try our web demo at <WEB_URL>.
2) 是否需要登录 / 适用人群（科研用户 / 临床合作方）
3) 后端部署形式（如 FastAPI + 本地 LLM / 云端 LLM）
4) 当前已上线的模块（alert / diagnosis / prognosis / treatment 中哪几个）
-->

## Modules

RareLens 按临床流程拆分为四个独立模块，每个模块都包含独立的训练管线和模型产物：

| Module | Path | Task |
| --- | --- | --- |
| Alert | `rare_alert/` | <!-- TODO[alert-task]: 一句话描述（如：基于初诊信息的罕见病风险筛查） --> |
| Diagnosis | `rare_diagnosis/` | <!-- TODO[diagnosis-task]: 一句话描述（如：候选病种排序 / 最终诊断推断） --> |
| Prognosis | `rare_prognosis/` | <!-- TODO[prognosis-task]: 一句话描述（如：整体预后 / 功能状态 / 症状负担预测） --> |
| Treatment | `rare_treatment/` | <!-- TODO[treatment-task]: 一句话描述（如：治疗方案候选排序与生成） --> |

公共组件：

- `core_tool/`：LLM 客户端、prompt 模板、JSON 解析器
- `configs/`：每个模块的 YAML 配置
- `pipeline/`：模块间的 adapter 与端到端编排
- `eval/`：评估与对比脚本
- `schema/`：Pydantic 输入输出 schema

## System Requirements

### Hardware

<!-- TODO[hardware]: 跟组员确认后保留 / 修改下面的数值。 -->

- **RAM**: Minimum 16GB (32GB recommended)
- **Storage**: <!-- TODO: e.g. 100GB+ free disk space (含 RareBench 数据规模) -->
- **GPU**: <!-- TODO: 训练 / 特征抽取 / LLM 推理各阶段是否必需，建议显存 -->
- **CPU**: Any modern 64-bit processor

### Software

- **OS**: Any 64-bit operating system (Linux / macOS / Windows)
- **Python**: 3.10+
- **Dependencies**: see `requirements.txt`

## LLM API Key Requirements

RareLens supports multiple LLM providers. Credentials are organized by **model family** — the same API account is shared across all pipeline stages for the same family. You need at least one provider from each family you intend to use.

Copy the credential template first:

```bash
cp .env.example .env
```

> `.env` is git-ignored. Never commit real keys to the repository.

Model-to-provider routing and API name overrides are configured in [`configs/models.py`](configs/models.py). See that file for details.

#### OpenAI (gpt-5, gpt-4o-mini, gpt-3.5-turbo, o3-mini)
- **How to obtain**: Sign up at [platform.openai.com](https://platform.openai.com)
- **Environment variable**: `OPENAI_API_KEY` / `OPENAI_BASE_URL`

#### Google Gemini (gemini-2.5-flash, …)
- **How to obtain**: Sign up at [ai.google.dev](https://ai.google.dev)
- **Environment variable**: `GOOGLE_API_KEY` / `GOOGLE_BASE_URL`

#### Qwen3 (qwen3-8b, qwen3-14b, qwen3-32b, qwen3-235b, …)
- **How to obtain**: Sign up at [dashscope.aliyun.com](https://dashscope.aliyun.com)
- **Environment variable**: `QWEN3_API_KEY` / `QWEN3_BASE_URL`

#### DeepSeek (deepseek-r1, deepseek-v3, …)
- **How to obtain**: Sign up at [platform.deepseek.com](https://platform.deepseek.com)
- **Environment variable**: `DEEPSEEK_API_KEY` / `DEEPSEEK_BASE_URL`

#### Anthropic Claude (claude-haiku-4-5, …)
- **How to obtain**: Sign up at [console.anthropic.com](https://console.anthropic.com) or [openrouter.ai](https://openrouter.ai)
- **Environment variable**: `ANTHROPIC_API_KEY` / `ANTHROPIC_BASE_URL`

#### RareAlert — Self-hosted / Fine-tuned Model
- **How to obtain**: Deploy your own OpenAI-compatible inference server (e.g. vLLM, SGLang)
- **Environment variable**: `ALERT_API_KEY` / `ALERT_URL`

#### OrphaRAG Embeddings
- **How to obtain**: Any OpenAI-compatible `/v1/embeddings` endpoint
- **Environment variable**: `ORPHA_RAG_EMBEDDINGS_API_KEY` / `ORPHA_RAG_EMBEDDINGS_BASE_URL`

#### Local / Custom LLM (Optional)
- **Custom LLM Integration**: All providers above use the OpenAI-compatible protocol — point any `*_BASE_URL` to a local vLLM / Ollama / SGLang server and set `*_API_KEY=EMPTY`
- **Setup**: Change the model's slot in `configs/models.py` → `MODEL_CREDENTIAL_SLOT`, then set the corresponding env vars
- **Custom protocol**: If your provider is not OpenAI-compatible, extend `LLMClient` in `core_tool/llm/client.py` and add a `resolve_*` function in `core_tool/llm/credential_resolver.py`

## Installation

```bash
git clone <REPO_URL>
cd RareLens
pip install -r requirements.txt

# Configure credentials
cp .env.example .env
# then edit `.env` to fill in keys / endpoints you actually use
```

<!-- TODO[install-extra]: 如果有额外依赖（例如 sentence-transformers 模型权重、faiss、Orphanet 本体 JSON 缓存），在这里加一段说明 -->

## Dataset

<!-- TODO[dataset]: 与组员对齐 RareBench 的发布形式后填写。建议覆盖：
1) 数据集主页 / HuggingFace 链接
2) 下载方式（示例命令）
3) 数据目录结构（dataset/{alert,diagnosis,prognosis,treatment}/...）
4) 数据使用许可
示例命令：
huggingface-cli download <ORG>/RareBench --repo-type=dataset --local-dir ./dataset
-->

## Usage

每个模块包含两类入口：

- **Reproducible Training** — `rare_<task>/training/`：从特征构建到固定最优参数训练（不做超参搜索），用于复现论文报告的最终模型。
- **Single-Module Inference** — `rare_<task>/run_*_pipeline.py`：加载训练好的模型对新病例做单模块推理，输出该任务的结构化结果。

> 端到端串联（alert → diagnosis → treatment → prognosis）暂未发布。

### 1. Reproducible Training

#### Alert

<!-- TODO[alert-train-cmd]: 与组员确认 rare_alert/training 的复现入口（脚本名/参数）后补充。建议参考 diagnosis 的 sh 风格统一为：
bash rare_alert/training/reproduce_alert.sh
-->

#### Diagnosis

```bash
# Primary visit
bash rare_diagnosis/training/reproduce_diag.sh

# Follow-up visit
VISIT_TYPE=followup bash rare_diagnosis/training/reproduce_diag.sh
```

可通过环境变量覆盖路径，例如 `QUERY_ROOT`、`GT_ROOT`、`MODELS_ROOT`、`TRAIN_IDS`、`TEST_IDS`、`CONFIG_PATH`、`OUT_DIR`。

#### Prognosis

<!-- TODO[prog-train-cmd]: 与 rare_prognosis/training 对齐后填写。可参考 diagnosis 的脚本风格补一个 reproduce_prog.sh -->

#### Treatment

<!-- TODO[treat-train-cmd]: 与 rare_treatment/training 对齐后填写。可参考 diagnosis 的脚本风格补一个 reproduce_treat.sh -->

### 2. Single-Module Inference

训练完成（或直接使用 `rare_<task>/models/` 中的预置权重）后，可对单个病例独立运行某个模块：

#### Alert

```bash
python -m rare_alert.run_risk_pipeline --help
```

<!-- TODO[alert-infer-cmd]: 给出最小可跑示例（输入病例路径 / 输出路径 / 必要参数） -->

#### Diagnosis

```bash
python -m rare_diagnosis.run_diag_pipeline --help
```

<!-- TODO[diag-infer-cmd]: 给出最小可跑示例（区分 primary / followup visit） -->

#### Prognosis

```bash
python -m rare_prognosis.run_prog_pipeline --help
```

<!-- TODO[prog-infer-cmd]: 给出最小可跑示例 -->

#### Treatment

```bash
python -m rare_treatment.run_treat_pipeline --help
```

<!-- TODO[treat-infer-cmd]: 给出最小可跑示例 -->

## Evaluation

<!-- TODO[eval]: 与组员对齐发布的评估脚本范围（eval/ 下脚本众多）。建议只保留最终用到的几个：
示例：
python -m rare_diagnosis.training.evaluate --json_path <pred.json>
python -m eval.script.eval_diagnosis ...
-->

## Project Layout

```
RareLens/
├── rare_alert/                # 早期风险筛查
│   ├── run_risk_pipeline.py   #   单模块推理入口
│   └── training/              #   复现训练脚本
├── rare_diagnosis/            # 诊断
│   ├── run_diag_pipeline.py
│   └── training/
├── rare_prognosis/            # 预后
│   ├── run_prog_pipeline.py
│   └── training/
├── rare_treatment/            # 治疗
│   ├── run_treat_pipeline.py
│   └── training/
├── core_tool/                 # LLM / prompt / parser 公共能力
├── configs/                   # 各模块 YAML 配置
├── pipeline/                  # 模块间 adapter（端到端串联暂未发布）
├── eval/                      # 评估脚本
├── schema/                    # Pydantic schema
└── requirements.txt
```

## Citation

<!-- TODO[citation]: 论文正式发表/挂 arXiv 后替换为正确的 BibTeX。
示例占位：
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

<!-- TODO[license]: 与组员确认对外发布使用的 License。当前仓库内 LICENSE 文件是 Apache 2.0，如确认沿用即可写：
This project is released under the Apache License 2.0. See `LICENSE` for details.
-->

## Acknowledgement

<!-- TODO[ack]: 致谢使用的公共数据集（Orphanet / MIMIC-IV-Ext 等）、基础模型与工具。 -->
