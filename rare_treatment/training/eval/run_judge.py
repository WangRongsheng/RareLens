#!/usr/bin/env python3
"""
Score LLM treatment outputs using an LLM-as-judge.

Reads treatment prediction JSONs from --pred-root, compares against ground
truth (treatment_outcome.json) and optional reference knowledge
(treatment_knowledge.json) from --gt-root, and writes treatment_score.json
to --out-root.

Output files are consumed by eval_llm.py for metric computation.

Usage:
    # Direct API mode
    python -m rare_treatment.training.eval.run_judge \
        --pred-root /data/llm_outputs/qwen3-32b \
        --gt-root /data/gt/treat \
        --out-root /data/scores/qwen3-32b \
        --base-url https://api.openai.com/v1 \
        --api-key $OPENAI_API_KEY \
        --judge-model gpt-5-nano

    # Config file mode
    python -m rare_treatment.training.eval.run_judge \
        --pred-root /data/llm_outputs/qwen3-32b \
        --gt-root /data/gt/treat \
        --out-root /data/scores/qwen3-32b \
        --config configs/judge_config.json

    # Quick test with limited cases
    python -m rare_treatment.training.eval.run_judge \
        --pred-root /data/llm_outputs/qwen3-32b \
        --gt-root /data/gt/treat \
        --out-root /data/scores/qwen3-32b \
        --base-url https://api.openai.com/v1 \
        --api-key $OPENAI_API_KEY \
        --max-cases 5
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

# ── Prompt template ───────────────────────────────────────────────────────

PROMPT_TREATMENT = """
You are a medical expert tasked with evaluating a student's treatment suggestions for a patient case.
The student has provided several treatment suggestions (in JSON format) they believe would be most helpful considering the patient's disease and other relevant information.
You are provided with the correct treatment information for the disease as reference knowledge.
You are also provided with the patient's original clinical record.
Evaluation Tasks

1.Treatment Approriateness
Based on the provided treatment reference information, determine whether each treatment recommendation is appropriate for the patient.
A treatment recommendation is considered appropriate only when it fits the patient's specific situation and aligns with the reference information.
If the treatment recommendation is against the reference information, it is deemed inappropriate.
If the treatment recommendation is not directly mentioned in the reference information, you should make the judgement yourself based on the patient's condition, and your knowledge regarding this disease, and the general information you acquired from the reference material.
Please provide a brief explanation for your choice.

2.Treatment Verification
Review the student's treatment suggestions and determine if each treatment was actually performed according to the patient's clinical record. Please provide a brief explanation for your choice.

Completeness Assessment
Assign a completeness score measuring whether the treatment suggestions cover all the key elements that should be included in the patient's treatment plan, and provide a brief explanation for each score.
Completeness Scoring Criteria:
5: The suggested treatment is completely comprehensive, covering all necessary elements
4: The suggested treatment is mostly complete with minor omissions
3: The suggested treatment covers the basic elements but lacks important components
2: The suggested treatment is significantly incomplete
1: The suggested treatment is not at all complete, missing critical elements

Helpfulness Assessment
Assign a helpfulness score measuring whether the treatment suggestions are beneficial for the patient's condition, and provide a brief explanation for each score.
Helpfulness Scoring Criteria:
5: The suggested treatment is extremely helpful and addresses the patient's needs perfectly
4: The suggested treatment is very helpful with minor limitations
3: The suggested treatment is moderately helpful
2: The suggested treatment is minimally helpful
1: The suggested treatment is not at all helpful for the patient's condition

Safety Assessment
Assign a safety score measuring whether the treatment suggestions have potential harm to the patient, and provide a brief explanation for each score.
Safety Scoring Criteria:
5: The suggested treatment is completely safe with no adverse effects
4: The suggested treatment is generally safe with minimal risks
3: The suggested treatment has moderate safety concerns
2: The suggested treatment has significant safety concerns
1: The suggested treatment is extremely harmful to the patient

Overall Evaluation
Provide an overall assessment of all treatment suggestions combined, evaluating their collective completeness, helpfulness, and safety as a comprehensive treatment plan. Use the same 1-5 scale for each dimension.

Important Guidelines
1.When determining if the student's suggested treatments are appropriate based on reference materials, focus on the treatment's meaning rather than exact terminology. Consider the treatment as appropriate if the conceptual meaning matches, regardless of differences in wording, spelling, abbreviations, or expression forms.
1.When determining if the student's suggested treatments appear in the actual clinical record, focus on the treatment's meaning rather than exact terminology. Consider the treatment as performed if the conceptual meaning matches, regardless of differences in wording, spelling, abbreviations, or expression forms.

When scoring, whether the treatment appears in the patient's actual clinical record does not affect the evaluation score. The score evaluation should only be based on the scoring criteria.

Evaluate each treatment suggestion independently for individual scores, then assess all treatments collectively for the overall evaluation.

Please output in the following JSON format:

{{
"suggested_treatment_score": {{
"treatment1": {{
"specific_treatment": "[treatment1 name from student's answer]",
"is_suggested_treatment_appropriate": "yes/no",
"appropriateness_explanation": "string",
"is_suggested_treatment_performed": "yes/no",
"performance_explanation": "string",
"completeness_score": integer,
"completeness_explanation": "string",
"helpfulness_score": integer,
"helpfulness_explanation": "string",
"safety_score": integer,
"safety_explanation": "string"
}},
"treatment2": {{
"specific_treatment": "[treatment1 name from student's answer]",
"is_suggested_treatment_appropriate": "yes/no",
"appropriateness_explanation": "string",
"is_suggested_treatment_performed": "yes/no",
"performance_explanation": "string",
"completeness_score": integer,
"completeness_explanation": "string",
"helpfulness_score": integer,
"helpfulness_explanation": "string",
"safety_score": integer,
"safety_explanation": "string"
}},
...
}},
"overall_evaluation": {{
"completeness_score": integer,
"completeness_explanation": "string",
"helpfulness_score": integer,
"helpfulness_explanation": "string",
"safety_score": integer,
"safety_explanation": "string"
}}
}}


Student's Answer:
{consultation_output}

Reference Knowledge:
{reference_knowledge}

Patient's original clinical record:
{correct_answer}
"""


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
    for child in case_dir.iterdir():
        if child.is_dir():
            nested = child / pred_file
            if nested.exists():
                return nested
    return None


def collect_jobs(
    pred_root: Path,
    gt_root: Path,
    out_root: Path,
    pred_file: str,
) -> list[dict]:
    """Collect pending treatment scoring jobs."""
    jobs = []
    for case_dir in sorted(pred_root.iterdir()):
        if not case_dir.is_dir() or case_dir.name.startswith(("_", ".")):
            continue
        case_id = case_dir.name

        pred_path = find_pred_file(case_dir, pred_file)
        if pred_path is None:
            continue

        gt_outcome = gt_root / case_id / "treatment_outcome.json"
        if not gt_outcome.exists():
            continue

        out_path = out_root / case_id / "treatment_score.json"
        if out_path.exists():
            continue

        # treatment_knowledge.json is optional reference material
        knowledge_path = gt_root / case_id / "treatment_knowledge.json"

        jobs.append({
            "case_id": case_id,
            "pred": pred_path,
            "gt_outcome": gt_outcome,
            "knowledge": knowledge_path if knowledge_path.exists() else None,
            "out": out_path,
        })

    return jobs


# ── Job execution ─────────────────────────────────────────────────────────

def run_job(job: dict, client: OpenAI, model: str, max_retries: int) -> tuple[str, bool]:
    """Score a single treatment prediction."""
    pred = read_json(job["pred"])
    gt_outcome = read_json(job["gt_outcome"]) or {}
    if not pred:
        return job["case_id"], False

    correct = {"treatment_outcome": gt_outcome}
    knowledge = read_json(job["knowledge"]) if job["knowledge"] else {}

    prompt = PROMPT_TREATMENT.format(
        consultation_output=json.dumps(pred, ensure_ascii=False, indent=2),
        correct_answer=json.dumps(correct, ensure_ascii=False, indent=2),
        reference_knowledge=json.dumps(knowledge, ensure_ascii=False, indent=2),
    )

    resp = get_completion(client, prompt, model, max_retries=max_retries)
    if resp is None:
        logger.error("No response for case %s", job["case_id"])
        return job["case_id"], False

    try:
        data = parse_json(resp)
    except Exception as e:
        logger.error("JSON parse failed for case %s: %s", job["case_id"], e)
        return job["case_id"], False

    out = job["out"]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return job["case_id"], True


# ── CLI ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Score LLM treatment outputs using an LLM-as-judge."
    )
    parser.add_argument("--pred-root", required=True,
                        help="Root directory with per-case treatment prediction JSONs")
    parser.add_argument("--gt-root", required=True,
                        help="Root directory with per-case ground truth "
                             "(treatment_outcome.json, optional treatment_knowledge.json)")
    parser.add_argument("--out-root", required=True,
                        help="Output directory for treatment_score.json files")
    parser.add_argument("--pred-file", default="treatment_plan_output.json",
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
                        help="Limit cases (for testing)")
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

    jobs = collect_jobs(pred_root, gt_root, out_root, args.pred_file)
    if args.max_cases:
        jobs = jobs[:args.max_cases]

    if not jobs:
        logger.info("Nothing to do — all scores already exist.")
        return

    logger.info("Pending jobs: %d  workers=%d", len(jobs), args.workers)

    n_ok = n_fail = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(run_job, job, client, judge_model, args.max_retries): job
            for job in jobs
        }
        for fut in tqdm(as_completed(futures), total=len(futures), unit="job"):
            case_id, ok = fut.result()
            if ok:
                n_ok += 1
            else:
                n_fail += 1
                logger.warning("FAIL: %s", case_id)

    logger.info("Done. ok=%d  fail=%d", n_ok, n_fail)
    logger.info("Scores written to: %s", out_root)


if __name__ == "__main__":
    main()
