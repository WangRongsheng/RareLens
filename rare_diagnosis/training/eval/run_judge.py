#!/usr/bin/env python3
"""
Score LLM diagnosis outputs using an LLM-as-judge.

Reads prediction JSONs from --pred-root, compares against ground truth
from --gt-root, and writes evaluation score JSONs to --out-root.

Supports three scoring tasks:
  - primary_diag:  Score most_likely_diagnosis against diagnosis.json
  - follow_diag:   Same prompt, output as followup_diagnosis_score.json
  - diag_test:     Score further_diagnostic_test against diagnostic_test.json + diagnosis.json

Output files are consumed by eval_llm.py for metric computation.

Usage:
    # Score primary diagnosis predictions
    python -m rare_diagnosis.training.eval.run_judge \
        --pred-root /data/llm_outputs/qwen3-32b/primary_visit \
        --gt-root /data/gt/diag \
        --out-root /data/scores/qwen3-32b \
        --tasks primary_diag \
        --base-url https://api.openai.com/v1 \
        --api-key $OPENAI_API_KEY \
        --judge-model gpt-5-nano

    # Score followup diagnosis + diagnostic tests
    python -m rare_diagnosis.training.eval.run_judge \
        --pred-root /data/llm_outputs/qwen3-32b/follow_visit \
        --gt-root /data/gt/diag \
        --out-root /data/scores/qwen3-32b \
        --tasks follow_diag diag_test \
        --config configs/judge_config.json

    # Score all tasks with direct API credentials
    python -m rare_diagnosis.training.eval.run_judge \
        --pred-root /data/llm_outputs/qwen3-32b \
        --gt-root /data/gt/diag \
        --out-root /data/scores/qwen3-32b \
        --tasks primary_diag follow_diag diag_test \
        --base-url https://api.openai.com/v1 \
        --api-key $OPENAI_API_KEY
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ── Prompt templates ──────────────────────────────────────────────────────

PROMPT_DIAG = """
You are a medical expert, please evaluate the following diagnoses provided by the student.
The student's answer contains a list of five diagnoses in JSON format.
The correct answer is provided, please evaluate each diagnosis against the correct diagnosis.

Your evaluation should be based on the correct diagnosis and according to the scoring criteria.
Scoring Criteria:
- 5: The actual diagnosis was suggested, or the alternative name of the actual diagnosis was suggested
- 4: The suggestions included something very close, but not exact
- 3: The suggestions included something closely related that might have been helpful
- 2: The suggestions included something related, but unlikely to be helpful
- 0: No suggestions close
If a diagnosis is not valid, the score would be 0.

Important note:
Your evaluation score should be based on the core diagnosis:
1. If the student's diagnosis and the correct answer refer to the same disease (core diagnosis), but use different naming conventions such as different synonyms, abbreviations, or expressions, this is not a reason to lower the evaluation score.
2. If the student's diagnosis and the correct answer refer to the same disease (core diagnosis), but the correct answer includes additional descriptive information such as subtype, acute/chronic status, staging, or anatomical location that the student's diagnosis lacks, this is also not a reason to lower the evaluation score.
3. Evaluate each of the five diagnoses in the student's answer, provide an evaluation score and brief explanation.


Output your evaluation in the following JSON format:
{{
    "most_likely_diagnosis": {{
        "diagnosis1": {{
            "diagnosis_name": "[diagnosis name from student's answer]",
            "evaluation_score": integer,
            "brief_explanation": "string"
        }},
        "diagnosis2":{{
            "diagnosis_name": "[diagnosis name from student's answer]",
            "evaluation_score": integer,
            "brief_explanation": "string"
        }},
        "diagnosis3": {{
            "diagnosis_name": "[diagnosis name from student's answer]",
            "evaluation_score": integer,
            "brief_explanation": "string"
        }},
        "diagnosis4": {{
            "diagnosis_name": "[diagnosis name from student's answer]",
            "evaluation_score": integer,
            "brief_explanation": "[diagnosis name from student's answer]"
        }},
        "diagnosis5": {{
            "diagnosis_name": "[diagnosis name from student's answer]",
            "evaluation_score": integer,
            "brief_explanation": "string"
        }}
    }}
}}

Student's Answer:
{consultation_output}

Correct Answer:
{correct_answer}
"""

PROMPT_TEST = """
You are a medical expert tasked with evaluating a student's selection of diagnostic tests for a patient case. The student has suggested five diagnostic tests (in JSON format) they believe would be most helpful in reaching the correct diagnosis for a given patient.

Your Evaluation Tasks
Review the student's suggested diagnostic tests and determine if each test was actually performed according to the patient's clinical record
Assign an appropriateness score for each test based on its usefulness in reaching the final diagnosis
Provide a brief explanation for each score
Scoring Criteria
5: The suggested test is confirmatory and would directly lead to the correct diagnosis
4: The suggested test is not confirmatory but is helpful in leading to the correct diagnosis
3: The suggested test may help slightly in leading to the correct diagnosis
2: The suggested test is not helpful in leading to the correct diagnosis
1: The suggested test is not helpful and may mislead to an incorrect diagnosis or pose harm to the patient
If the student proposes an invalid test, the score should be 0.

Important Guidelines
1.When determining if the student's suggested test appears in the actual clinical record, focus on the test's meaning rather than exact terminology. Consider the test as performed if the conceptual meaning matches, regardless of differences in wording, spelling, abbreviations, or expression forms.
2.Your evaluation score should be based solely on whether the test can help reach the core final diagnosis.
3.When scoring, whether the test appears in the patient's actual clinical record does not affect the appropriateness score. The score depends only on whether the test would help reach the correct final diagnosis.
4.Evaluate each test independently.

Please output in the following JSON format:
{{
    "suggested_test_score": {{
        "test1": {{
            "is_suggested_test_performed": "yes/no",
            "test_name": "[test1 name from student's answer]",
            "evaluation_score": integer,
            "brief_explanation": "string"
        }},
        "test2": {{
            "is_suggested_test_performed": "yes/no",
            "test_name": "[test2 name from student's answer]",
            "evaluation_score": integer,
            "brief_explanation": "string"
        }},
        "test3": {{
            "is_suggested_test_performed": "yes/no",
            "test_name": "[test3 name from student's answer]",
            "evaluation_score": integer,
            "brief_explanation": "string"
        }},
        "test4": {{
            "is_suggested_test_performed": "yes/no",
            "test_name": "[test4 name from student's answer]",
            "evaluation_score": integer,
            "brief_explanation": "string"
        }},
        "test5": {{
            "is_suggested_test_performed": "yes/no",
            "test_name": "[test5 name from student's answer]",
            "evaluation_score": integer,
            "brief_explanation": "string"
        }}
    }}
}}

Student's Answer:
{consultation_output}

Patient's original clinical record:
{correct_answer}
"""

ALL_TASKS = ["primary_diag", "follow_diag", "diag_test"]

# Task → output score filename (must match what eval_llm.py reads)
TASK_OUTPUT_FILE = {
    "primary_diag": "primary_diagnosis_score.json",
    "follow_diag": "followup_diagnosis_score.json",
    "diag_test": "suggested_test_score.json",
}


# ── Config loading ────────────────────────────────────────────────────────

def load_model_config(config_path: str, model_tag: str) -> dict:
    """Load model config (base_url, api_key, model) from a JSON config list."""
    with open(config_path, "r", encoding="utf-8") as f:
        config_list = json.load(f)
    for cfg in config_list:
        if model_tag in cfg.get("tags", []):
            return cfg
    for cfg in config_list:
        if cfg.get("model") == model_tag:
            return cfg
    raise ValueError(f"Model '{model_tag}' not found in {config_path}.")


# ── LLM call ─────────────────────────────────────────────────────────────

def get_completion(
    client: OpenAI,
    prompt: str,
    model: str,
    *,
    max_retries: int = 10,
    delay: float = 2.0,
) -> str | None:
    """Call the LLM judge API with retry logic."""
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant designed to output the final answer in JSON."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
            )
            return resp.choices[0].message.content
        except Exception as e:
            logger.warning("API error (attempt %d/%d): %s", attempt + 1, max_retries, e)
            if attempt < max_retries - 1:
                time.sleep(delay)
    return None


# ── JSON parsing ─────────────────────────────────────────────────────────

def parse_json(text: str) -> dict:
    """Parse JSON from LLM text output, handling markdown code blocks."""
    text = re.sub(r"```(?:json)?\s*|\s*```", "", text).strip("\"'")
    for m in re.findall(r"(\{[\s\S]*\})", text, re.DOTALL):
        try:
            return json.loads(m.strip())
        except json.JSONDecodeError:
            continue
    return json.loads(text)


def read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text("utf-8"))
    except Exception:
        return None


# ── Job collection ────────────────────────────────────────────────────────

def find_pred_file(case_dir: Path, pred_file: str) -> Path | None:
    """Search for prediction file in case_dir or one level deeper."""
    direct = case_dir / pred_file
    if direct.exists():
        return direct
    # Search subdirectories (e.g. rarediagnosis_output/RareDiagnosis_output.json)
    for child in case_dir.iterdir():
        if child.is_dir():
            nested = child / pred_file
            if nested.exists():
                return nested
    return None


def collect_jobs(
    task: str,
    pred_root: Path,
    gt_root: Path,
    out_root: Path,
    pred_file: str,
) -> list[dict]:
    """Collect pending scoring jobs for a task."""
    out_fname = TASK_OUTPUT_FILE[task]
    jobs = []

    for case_dir in sorted(pred_root.iterdir()):
        if not case_dir.is_dir() or case_dir.name.startswith(("_", ".")):
            continue
        case_id = case_dir.name

        pred_path = find_pred_file(case_dir, pred_file)
        if pred_path is None:
            continue

        gt_diag = gt_root / case_id / "diagnosis.json"
        if not gt_diag.exists():
            continue

        out_path = out_root / case_id / out_fname
        if out_path.exists():
            continue

        job = {
            "task": task,
            "case_id": case_id,
            "pred": pred_path,
            "gt_diag": gt_diag,
            "out": out_path,
        }

        if task == "diag_test":
            gt_test = gt_root / case_id / "diagnostic_test.json"
            if not gt_test.exists():
                continue
            job["gt_test"] = gt_test

        jobs.append(job)

    return jobs


# ── Job execution ─────────────────────────────────────────────────────────

def run_diag_job(job: dict, client: OpenAI, model: str, max_retries: int) -> bool:
    """Score a diagnosis prediction (primary_diag or follow_diag)."""
    pred = read_json(job["pred"])
    gt = read_json(job["gt_diag"])
    if not pred or not gt:
        return False

    diagnosis = pred.get("most_likely_diagnosis", pred)
    prompt = PROMPT_DIAG.format(
        consultation_output=json.dumps(diagnosis, ensure_ascii=False, indent=2),
        correct_answer=json.dumps(gt, ensure_ascii=False, indent=2),
    )
    return _score_and_save(job, prompt, client, model, max_retries)


def run_test_job(job: dict, client: OpenAI, model: str, max_retries: int) -> bool:
    """Score a diagnostic test prediction."""
    pred = read_json(job["pred"])
    if not pred:
        return False

    tests = pred.get("further_diagnostic_test")
    if not tests:
        # Write sentinel so the case is not re-queued
        out = job["out"]
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps({"skipped": "further_diagnostic_test is null"}, ensure_ascii=False),
            encoding="utf-8",
        )
        return True

    gt_test = read_json(job["gt_test"]) or {}
    gt_diag = read_json(job["gt_diag"]) or {}
    correct = {"diagnostic_test": gt_test, "final_diagnosis": gt_diag}

    prompt = PROMPT_TEST.format(
        consultation_output=json.dumps(tests, ensure_ascii=False, indent=2),
        correct_answer=json.dumps(correct, ensure_ascii=False, indent=2),
    )
    return _score_and_save(job, prompt, client, model, max_retries)


def _score_and_save(
    job: dict, prompt: str, client: OpenAI, model: str, max_retries: int,
) -> bool:
    """Call judge LLM, parse response, and save to disk."""
    resp = get_completion(client, prompt, model, max_retries=max_retries)
    if resp is None:
        logger.error("No response for %s/%s", job["task"], job["case_id"])
        return False
    try:
        data = parse_json(resp)
    except Exception as e:
        logger.error("JSON parse failed for %s/%s: %s", job["task"], job["case_id"], e)
        return False

    out = job["out"]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return True


def dispatch(job: dict, client: OpenAI, model: str, max_retries: int) -> tuple[str, bool]:
    """Route a job to the correct handler."""
    label = f"{job['task']}/{job['case_id']}"
    if job["task"] in ("primary_diag", "follow_diag"):
        ok = run_diag_job(job, client, model, max_retries)
    elif job["task"] == "diag_test":
        ok = run_test_job(job, client, model, max_retries)
    else:
        ok = False
    return label, ok


# ── CLI ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Score LLM diagnosis outputs using an LLM-as-judge."
    )
    parser.add_argument("--pred-root", required=True,
                        help="Root directory with per-case prediction JSONs")
    parser.add_argument("--gt-root", required=True,
                        help="Root directory with per-case ground truth "
                             "(diagnosis.json, diagnostic_test.json)")
    parser.add_argument("--out-root", required=True,
                        help="Output directory for score JSONs")
    parser.add_argument("--tasks", nargs="+", default=ALL_TASKS,
                        choices=ALL_TASKS,
                        help="Which scoring tasks to run (default: all)")
    parser.add_argument("--pred-file", default="RareDiagnosis_output.json",
                        help="Prediction filename to look for in each case dir")
    parser.add_argument("--judge-model", default="gpt-5-nano",
                        help="Model name for the LLM judge")
    parser.add_argument("--base-url", default=None,
                        help="OpenAI-compatible API base URL")
    parser.add_argument("--api-key", default=None,
                        help="API key")
    parser.add_argument("--config", default=None,
                        help="Path to JSON config list (alternative to --base-url/--api-key)")
    parser.add_argument("--workers", type=int, default=20,
                        help="Number of concurrent threads (default: 20)")
    parser.add_argument("--max-retries", type=int, default=10,
                        help="Max retries per LLM call")
    parser.add_argument("--max-cases", type=int, default=None,
                        help="Limit cases per task (for testing)")
    args = parser.parse_args()

    # Initialize OpenAI client
    if args.config:
        model_cfg = load_model_config(args.config, args.judge_model)
        client = OpenAI(
            base_url=model_cfg.get("base_url"),
            api_key=model_cfg.get("api_key"),
        )
        judge_model = model_cfg.get("model", args.judge_model)
    elif args.base_url and args.api_key:
        client = OpenAI(base_url=args.base_url, api_key=args.api_key)
        judge_model = args.judge_model
    else:
        logger.error("Provide either --config or both --base-url and --api-key.")
        sys.exit(1)

    pred_root = Path(args.pred_root)
    gt_root = Path(args.gt_root)
    out_root = Path(args.out_root)

    # Collect all pending jobs
    all_jobs: list[dict] = []
    for task in args.tasks:
        jobs = collect_jobs(task, pred_root, gt_root, out_root, args.pred_file)
        if args.max_cases:
            jobs = jobs[:args.max_cases]
        logger.info("[%s] total=%d pending", task, len(jobs))
        all_jobs.extend(jobs)

    if not all_jobs:
        logger.info("Nothing to do — all scores already exist.")
        return

    logger.info("Total pending jobs: %d  workers=%d", len(all_jobs), args.workers)

    n_ok = n_fail = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(dispatch, job, client, judge_model, args.max_retries): job
            for job in all_jobs
        }
        for fut in tqdm(as_completed(futures), total=len(futures), unit="job"):
            label, ok = fut.result()
            if ok:
                n_ok += 1
            else:
                n_fail += 1
                logger.warning("FAIL: %s", label)

    logger.info("Done. ok=%d  fail=%d", n_ok, n_fail)
    logger.info("Scores written to: %s", out_root)


if __name__ == "__main__":
    main()
