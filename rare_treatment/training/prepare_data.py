#!/usr/bin/env python3
"""
Prepare data for the treatment ranking pipeline.

Converts raw case outputs and LLM predictions into the directory structure expected by
build_features.py, train_ranker.py, and eval scripts.

Inputs (from data_demo/):
  - case_output/<case_id>/1_raw_data/treatment_plan.json  (patient case data)
  - case_output/<case_id>/5_treatment/llm_outputs.json     (per-model eval scores)
  - pipeline_data/treatment/treatment_llm/<model>/<case_id>/treatment_plan_output.json

Outputs (into --out-dir):
  plan_root/<case_id>/treatment_plan.json          (symlinked or copied)
  treatment_output/<model>/<case_id>/treatment_plan_output.json (symlinked or copied)
  treatment_score/<model>/<case_id>/treatment_score.json        (split from llm_outputs.json)
  dataset/train_cases.json
  dataset/test_cases.json

Usage:
    python prepare_demo_data.py \
        --case-root ../data_demo/case_output \
        --llm-output-root ../data_demo/pipeline_data/treatment/treatment_llm \
        --out-dir ../data_demo/pipeline_data/treatment/prepared
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare demo data for treatment ranking pipeline.")
    parser.add_argument("--case-root", required=True,
                        help="Root of case_output/ (contains <case_id>/1_raw_data/treatment_plan.json)")
    parser.add_argument("--llm-output-root", required=True,
                        help="Root of treatment_llm/ (contains <model>/<case_id>/treatment_plan_output.json)")
    parser.add_argument("--out-dir", required=True,
                        help="Output directory for prepared pipeline data")
    args = parser.parse_args()

    case_root = Path(args.case_root)
    llm_output_root = Path(args.llm_output_root)
    out_dir = Path(args.out_dir)

    # Discover case IDs — support both layouts:
    #   <case_id>/1_raw_data/treatment_plan.json  (case_output format)
    #   <case_id>/treatment_plan.json             (flat format)
    case_ids = sorted([
        p.name for p in case_root.iterdir()
        if p.is_dir() and (
            (p / "1_raw_data" / "treatment_plan.json").exists()
            or (p / "treatment_plan.json").exists()
        )
    ])
    if not case_ids:
        raise SystemExit(f"No cases found under {case_root}")
    logger.info("Found %d cases: %s", len(case_ids), case_ids)

    # 1. Set up plan_root: <out_dir>/plan_root/<case_id>/treatment_plan.json
    plan_root = out_dir / "plan_root"
    for case_id in case_ids:
        src = case_root / case_id / "1_raw_data" / "treatment_plan.json"
        if not src.exists():
            src = case_root / case_id / "treatment_plan.json"
        dst_dir = plan_root / case_id
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / "treatment_plan.json"
        if not dst.exists():
            shutil.copy2(src, dst)
    logger.info("Plan root ready: %s", plan_root)

    # 2. Set up treatment_output: <out_dir>/treatment_output/<model>/<case_id>/treatment_plan_output.json
    treatment_output = out_dir / "treatment_output"
    models = sorted([p.name for p in llm_output_root.iterdir() if p.is_dir()])
    for model in models:
        for case_id in case_ids:
            src = llm_output_root / model / case_id / "treatment_plan_output.json"
            if not src.exists():
                continue
            dst_dir = treatment_output / model / case_id
            dst_dir.mkdir(parents=True, exist_ok=True)
            dst = dst_dir / "treatment_plan_output.json"
            if not dst.exists():
                shutil.copy2(src, dst)
    logger.info("Treatment output ready: %s (%d models)", treatment_output, len(models))

    # 3. Split llm_outputs.json into per-model treatment_score.json
    treatment_score = out_dir / "treatment_score"
    for case_id in case_ids:
        llm_outputs_path = case_root / case_id / "5_treatment" / "llm_outputs.json"
        if not llm_outputs_path.exists():
            logger.warning("  %s not found, skipping", llm_outputs_path)
            continue

        with llm_outputs_path.open("r", encoding="utf-8") as f:
            llm_data = json.load(f)

        models_data = llm_data.get("models", {})
        for model_name, model_eval in models_data.items():
            score_obj = {
                "suggested_treatment_score": model_eval.get("suggested_treatment_score", {})
            }
            dst_dir = treatment_score / model_name / case_id
            dst_dir.mkdir(parents=True, exist_ok=True)
            dst = dst_dir / "treatment_score.json"
            with dst.open("w", encoding="utf-8") as f:
                json.dump(score_obj, f, indent=2, ensure_ascii=False)

    logger.info("Treatment scores ready: %s", treatment_score)

    # 4. Create train/test split JSON files
    dataset_dir = out_dir / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    # For demo: use all cases as both train and test (smoke test)
    with (dataset_dir / "train_cases.json").open("w", encoding="utf-8") as f:
        json.dump(case_ids, f, indent=2)
    with (dataset_dir / "test_cases.json").open("w", encoding="utf-8") as f:
        json.dump(case_ids, f, indent=2)

    logger.info("Dataset splits ready: %s (all %d cases as train+test)", dataset_dir, len(case_ids))
    logger.info("Finished. Prepared data saved to %s", out_dir)


if __name__ == "__main__":
    main()
