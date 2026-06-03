#!/usr/bin/env python3
"""
Adapter: generate treatment pipeline inputs from data_demo_500 + LLM500.

Creates the case_output directory structure expected by
rare_treatment/training/prepare_data.py, with synthetic scoring derived from
treatment_outcome.json (treatments actually performed = positive labels).

Specifically generates:
  <out_dir>/case_output/<case_id>/1_raw_data/treatment_plan.json   (copied)
  <out_dir>/case_output/<case_id>/5_treatment/llm_outputs.json     (synthetic)

Scoring logic (no human eval needed):
  - Treatment recommended by the LLM AND found in treatment_outcome.json:
      is_suggested_treatment_appropriate = true
      is_suggested_treatment_performed   = true
      completeness_score = 4, helpfulness_score = 4, safety_score = 4
  - Treatment recommended but NOT in treatment_outcome.json:
      is_suggested_treatment_appropriate = false
      is_suggested_treatment_performed   = false
      completeness_score = 2, helpfulness_score = 2, safety_score = 2

Usage (from repo root):
    python scripts/prep_treatment_500.py

Then run the pipeline (eval steps auto-skipped when score data is thin):
    bash rare_treatment/training/run_pipeline.sh \\
        --case-root       <out_dir>/case_output \\
        --llm-output-root LLM500/treatment_output \\
        --n-splits        3
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


_RE_NONALNUM = re.compile(r"[^a-z0-9\s]")


def normalize(text: str) -> str:
    t = text.lower()
    t = _RE_NONALNUM.sub(" ", t)
    return " ".join(t.split())


def extract_performed_treatments(treatment_outcome: dict) -> set[str]:
    """Return normalized names of treatments actually performed (from treatment_outcome.json)."""
    performed: set[str] = set()
    for item in treatment_outcome.get("treatment_information", []):
        name = item.get("specific_treatment", "").strip()
        if name:
            performed.add(normalize(name))
    return performed


def build_llm_outputs(
    model_names: list[str],
    case_id: str,
    llm_root: Path,
    performed: set[str],
) -> dict:
    """
    Build synthetic llm_outputs.json for one case.
    Each model's treatment recommendations are scored against performed treatments.
    """
    models_block: dict = {}
    for model in model_names:
        plan_path = llm_root / model / case_id / "treatment_plan_output.json"
        if not plan_path.is_file():
            continue

        plan_obj = json.loads(plan_path.read_text(encoding="utf-8"))
        recs = plan_obj.get("treatment_recommendations", {})
        if isinstance(recs, list):
            recs = {str(i + 1): v for i, v in enumerate(recs)}

        score_entries: dict = {}
        for key, rec in recs.items():
            spec = rec.get("specific_treatment", "").strip()
            if not spec:
                continue
            is_performed = normalize(spec) in performed
            score_entries[key] = {
                "specific_treatment": spec,
                "is_suggested_treatment_appropriate": is_performed,
                "is_suggested_treatment_performed":   is_performed,
                "completeness_score": 4 if is_performed else 2,
                "helpfulness_score":  4 if is_performed else 2,
                "safety_score":       4 if is_performed else 2,
            }

        if score_entries:
            models_block[model] = {"suggested_treatment_score": score_entries}

    return {"models": models_block}


def main() -> None:
    ap = argparse.ArgumentParser(description="Prep treatment case_output for data_demo_500.")
    ap.add_argument("--demo-root",  default="data_demo_500",
                    help="data_demo_500 root (default: data_demo_500)")
    ap.add_argument("--llm-root",   default="LLM500/treatment_output",
                    help="LLM treatment output root (default: LLM500/treatment_output)")
    ap.add_argument("--out-dir",    default="outputs/treat500_prep",
                    help="Output directory (default: outputs/treat500_prep)")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    demo_root = (repo_root / args.demo_root).resolve()
    llm_root  = (repo_root / args.llm_root).resolve()
    out_dir   = (repo_root / args.out_dir).resolve()

    # ── Discover model names ───────────────────────────────────────────────
    model_names = sorted([p.name for p in llm_root.iterdir() if p.is_dir()])
    if not model_names:
        raise SystemExit(f"No model directories found under {llm_root}")
    print(f"LLM models found: {model_names}")

    # ── Discover overlapping case IDs ──────────────────────────────────────
    demo_cases: set[str] = set()
    for p in demo_root.iterdir():
        if (p.is_dir()
                and (p / "treatment_plan.json").is_file()
                and (p / "treatment_outcome.json").is_file()):
            demo_cases.add(p.name)

    llm_cases: set[str] = set()
    for model in model_names:
        for case_dir in (llm_root / model).iterdir():
            if case_dir.is_dir() and (case_dir / "treatment_plan_output.json").is_file():
                llm_cases.add(case_dir.name)

    overlap = sorted(demo_cases & llm_cases)
    print(f"data_demo_500 cases with treatment data:  {len(demo_cases)}")
    print(f"LLM500 treatment cases:                   {len(llm_cases)}")
    print(f"Overlap (usable):                         {len(overlap)}")

    if not overlap:
        raise SystemExit("No overlapping cases — check paths.")

    # ── Generate case_output structure ─────────────────────────────────────
    case_out_root = out_dir / "case_output"
    for case_id in overlap:
        # 1. Copy treatment_plan.json -> 1_raw_data/
        src_plan = demo_root / case_id / "treatment_plan.json"
        dst_plan_dir = case_out_root / case_id / "1_raw_data"
        dst_plan_dir.mkdir(parents=True, exist_ok=True)
        dst_plan = dst_plan_dir / "treatment_plan.json"
        if not dst_plan.exists():
            dst_plan.write_bytes(src_plan.read_bytes())

        # 2. Load performed treatments from treatment_outcome.json
        outcome_path = demo_root / case_id / "treatment_outcome.json"
        outcome_obj  = json.loads(outcome_path.read_text(encoding="utf-8"))
        performed    = extract_performed_treatments(outcome_obj)

        # 3. Build and write synthetic llm_outputs.json -> 5_treatment/
        llm_outputs  = build_llm_outputs(model_names, case_id, llm_root, performed)
        dst_score_dir = case_out_root / case_id / "5_treatment"
        dst_score_dir.mkdir(parents=True, exist_ok=True)
        (dst_score_dir / "llm_outputs.json").write_text(
            json.dumps(llm_outputs, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        n_pos = sum(
            1
            for m in llm_outputs["models"].values()
            for v in m.get("suggested_treatment_score", {}).values()
            if v.get("is_suggested_treatment_performed")
        )
        print(f"  {case_id}: {len(performed)} performed treatments, {n_pos} matched in LLM outputs")

    print(f"\ncase_output root: {case_out_root}")

    # ── Print run command ──────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Run the pipeline with:")
    print(f"  bash rare_treatment/training/run_pipeline.sh \\")
    print(f"    --case-root       {case_out_root} \\")
    print(f"    --llm-output-root {llm_root} \\")
    print(f"    --n-splits        3")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
