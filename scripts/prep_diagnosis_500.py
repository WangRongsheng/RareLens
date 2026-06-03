#!/usr/bin/env python3
"""
Adapter: generate diagnosis pipeline inputs from data_demo_500 + LLM500.

Creates:
  <out_dir>/diagnosis_gt/<case_id>/primary_diagnosis_score.json
      -- derived from data_demo_500/<case_id>/diagnosis.json
         final_diagnosis  ->  evaluation_score = 5
  <out_dir>/splits/train.json
  <out_dir>/splits/test.json
      -- 80/20 split over cases that exist in both data_demo_500 and LLM500

Usage (from repo root):
    python scripts/prep_diagnosis_500.py
    python scripts/prep_diagnosis_500.py --visit-type followup
    python scripts/prep_diagnosis_500.py --out-dir my_outputs/diag500 --test-ratio 0.2

Then run the pipeline:
    bash rare_diagnosis/training/reproduce_diag.sh \\
        --query-root  data_demo_500 \\
        --gt-root     <out_dir>/diagnosis_gt \\
        --models-root LLM500/primary_diagnosis_output \\
        --train-ids   <out_dir>/splits/train.json \\
        --test-ids    <out_dir>/splits/test.json
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def normalize(text: str) -> str:
    return " ".join(text.lower().split())


def make_gt_score(final_diagnosis: str, differential: str = "") -> dict:
    """
    Build a synthetic primary_diagnosis_score.json.
    final_diagnosis -> evaluation_score 5 (exact GT)
    Differential diagnoses (semicolon-separated) -> evaluation_score 3
    """
    items: dict = {}
    idx = 1

    if final_diagnosis:
        items[str(idx)] = {
            "diagnosis_name": final_diagnosis.strip(),
            "evaluation_score": 5,
        }
        idx += 1

    if differential:
        for name in differential.split(","):
            name = name.strip()
            if name and normalize(name) != normalize(final_diagnosis):
                items[str(idx)] = {
                    "diagnosis_name": name,
                    "evaluation_score": 3,
                }
                idx += 1

    return {"most_likely_diagnosis": items}


def main() -> None:
    ap = argparse.ArgumentParser(description="Prep diagnosis GT + splits for data_demo_500.")
    ap.add_argument("--demo-root",    default="data_demo_500",
                    help="data_demo_500 root (default: data_demo_500)")
    ap.add_argument("--llm-root",     default="LLM500/primary_diagnosis_output",
                    help="LLM output root (default: LLM500/primary_diagnosis_output)")
    ap.add_argument("--visit-type",   default="primary", choices=["primary", "followup"],
                    help="primary (default) or followup — determines which LLM output dir is used")
    ap.add_argument("--out-dir",      default="outputs/diag500_prep",
                    help="Output directory (default: outputs/diag500_prep)")
    ap.add_argument("--test-ratio",   type=float, default=0.2,
                    help="Fraction of cases to hold out as test (default: 0.2)")
    ap.add_argument("--seed",         type=int, default=42)
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    demo_root = (repo_root / args.demo_root).resolve()
    out_dir   = (repo_root / args.out_dir).resolve()

    # Use follow_up_out for followup visit type
    if args.visit_type == "followup":
        llm_dir = (repo_root / "LLM500" / "follow_up_out").resolve()
        llm_fname = "most_likely_diagnosis_orphacode.json"
    else:
        llm_dir = (repo_root / args.llm_root).resolve()
        llm_fname = "most_likely_diagnosis_orphacode.json"

    # ── Discover overlapping case IDs ──────────────────────────────────────
    demo_cases: set[str] = set()
    for p in demo_root.iterdir():
        if p.is_dir() and (p / "diagnosis.json").is_file():
            demo_cases.add(p.name)

    llm_cases: set[str] = set()
    for model_dir in llm_dir.iterdir():
        if not model_dir.is_dir():
            continue
        for case_dir in model_dir.iterdir():
            if case_dir.is_dir() and (case_dir / llm_fname).is_file():
                llm_cases.add(case_dir.name)

    overlap = sorted(demo_cases & llm_cases)
    print(f"data_demo_500 cases with diagnosis.json:   {len(demo_cases)}")
    print(f"LLM500 cases with {llm_fname}: {len(llm_cases)}")
    print(f"Overlap (usable):                          {len(overlap)}")

    if not overlap:
        raise SystemExit("No overlapping cases — check paths.")

    # ── Generate GT score files ────────────────────────────────────────────
    gt_root = out_dir / "diagnosis_gt"
    skipped = 0
    for case_id in overlap:
        src = demo_root / case_id / "diagnosis.json"
        data = json.loads(src.read_text(encoding="utf-8"))
        diag_block = data.get("diagnosis", {})
        final = diag_block.get("final_diagnosis", "").strip()
        diff  = diag_block.get("differential_diagnosis", "").strip()

        if not final:
            skipped += 1
            continue

        score_obj = make_gt_score(final, diff)
        dst_dir = gt_root / case_id
        dst_dir.mkdir(parents=True, exist_ok=True)
        (dst_dir / "primary_diagnosis_score.json").write_text(
            json.dumps(score_obj, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    print(f"GT score files written: {len(overlap) - skipped} (skipped {skipped} with empty diagnosis)")

    # ── Generate train/test splits ─────────────────────────────────────────
    valid = [c for c in overlap if (gt_root / c / "primary_diagnosis_score.json").is_file()]
    rng = random.Random(args.seed)
    rng.shuffle(valid)
    n_test = max(1, int(len(valid) * args.test_ratio))
    test_ids  = sorted(valid[:n_test])
    train_ids = sorted(valid[n_test:])

    splits_dir = out_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    (splits_dir / "train.json").write_text(json.dumps(train_ids, indent=2), encoding="utf-8")
    (splits_dir / "test.json").write_text(json.dumps(test_ids,  indent=2), encoding="utf-8")

    print(f"Train: {len(train_ids)} cases  |  Test: {len(test_ids)} cases")
    print(f"Splits written to: {splits_dir}")

    # ── Print run command ──────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Run the pipeline with:")
    print(f"  bash rare_diagnosis/training/reproduce_diag.sh \\")
    print(f"    --visit-type  {args.visit_type} \\")
    print(f"    --query-root  {demo_root} \\")
    print(f"    --gt-root     {gt_root} \\")
    print(f"    --models-root {llm_dir} \\")
    print(f"    --train-ids   {splits_dir / 'train.json'} \\")
    print(f"    --test-ids    {splits_dir / 'test.json'}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
