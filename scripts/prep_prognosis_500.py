#!/usr/bin/env python3
"""
Adapter: generate prognosis pipeline inputs from data_demo_500 + LLM500.

Converts data_demo_500/<case_id>/prognosis_new.json into the format
expected by rare_prognosis/training/prepare_data.py:
  <out_dir>/case_output/<case_id>/6_prognosis/RarePrognosis_output.json

Usage (from repo root):
    python scripts/prep_prognosis_500.py

Then run the pipeline:
    bash rare_prognosis/training/run_pipeline.sh \\
        --case-root <out_dir>/case_output \\
        --llm-root  LLM500/prognosis_output \\
        --cv-folds  5
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


# Maps prognosis_new.json fields -> RarePrognosis_output.json task keys
# task_key: (json_path_in_prognosis_new, fallback)
TASK_MAP = {
    "overall_outcome":   (["overall_outcome"],                   "unknown"),
    "functional_status": (["quality_of_life", "functional_status"], "unknown"),
    "symptom_burden":    (["quality_of_life", "symptom_burden"],    "unknown"),
}


def get_nested(obj: dict, path: list[str], fallback: str = "") -> str:
    cur = obj
    for key in path:
        if not isinstance(cur, dict):
            return fallback
        cur = cur.get(key, fallback)
    return str(cur) if cur else fallback


def convert_prognosis(prognosis_new: dict) -> dict:
    """Convert prognosis_new.json structure to RarePrognosis_output.json format."""
    rare_prognosis: dict = {}
    for task, (path, fallback) in TASK_MAP.items():
        label = get_nested(prognosis_new, path, fallback)
        rare_prognosis[task] = {"gt": label}
    return {"RarePrognosis": rare_prognosis}


def main() -> None:
    ap = argparse.ArgumentParser(description="Prep prognosis case_output dir for data_demo_500.")
    ap.add_argument("--demo-root", default="data_demo_500",
                    help="data_demo_500 root (default: data_demo_500)")
    ap.add_argument("--llm-root",  default="LLM500/prognosis_output",
                    help="LLM prognosis output root (default: LLM500/prognosis_output)")
    ap.add_argument("--out-dir",   default="outputs/prog500_prep",
                    help="Output directory (default: outputs/prog500_prep)")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    demo_root = (repo_root / args.demo_root).resolve()
    llm_root  = (repo_root / args.llm_root).resolve()
    out_dir   = (repo_root / args.out_dir).resolve()

    # ── Discover overlapping case IDs ──────────────────────────────────────
    demo_cases: set[str] = set()
    for p in demo_root.iterdir():
        if p.is_dir() and (p / "prognosis_new.json").is_file():
            demo_cases.add(p.name)

    llm_cases: set[str] = set()
    for model_dir in llm_root.iterdir():
        if not model_dir.is_dir():
            continue
        for case_dir in model_dir.iterdir():
            if case_dir.is_dir() and (case_dir / "prognosis_prediction_output.json").is_file():
                llm_cases.add(case_dir.name)

    overlap = sorted(demo_cases & llm_cases)
    print(f"data_demo_500 cases with prognosis_new.json: {len(demo_cases)}")
    print(f"LLM500 prognosis cases:                       {len(llm_cases)}")
    print(f"Overlap (usable):                             {len(overlap)}")

    if not overlap:
        raise SystemExit("No overlapping cases — check paths.")

    # ── Convert and write RarePrognosis_output.json ────────────────────────
    case_out_root = out_dir / "case_output"
    converted = 0
    skipped   = 0
    for case_id in overlap:
        src = demo_root / case_id / "prognosis_new.json"
        prognosis_new = json.loads(src.read_text(encoding="utf-8"))
        converted_obj = convert_prognosis(prognosis_new)

        # Validate: skip if all labels are unknown
        labels = [v.get("gt", "") for v in converted_obj["RarePrognosis"].values()]
        if all(lbl in ("", "unknown") for lbl in labels):
            print(f"  WARN: {case_id} has no usable prognosis labels — skipping")
            skipped += 1
            continue

        dst_dir = case_out_root / case_id / "6_prognosis"
        dst_dir.mkdir(parents=True, exist_ok=True)
        (dst_dir / "RarePrognosis_output.json").write_text(
            json.dumps(converted_obj, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        converted += 1

    print(f"Converted: {converted}  |  Skipped: {skipped}")
    print(f"case_output root: {case_out_root}")

    # ── Print run command ──────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Run the pipeline with:")
    print(f"  bash rare_prognosis/training/run_pipeline.sh \\")
    print(f"    --case-root {case_out_root} \\")
    print(f"    --llm-root  {llm_root} \\")
    print(f"    --cv-folds  5")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
