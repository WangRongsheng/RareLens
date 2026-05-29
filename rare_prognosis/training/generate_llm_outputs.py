"""
Batch generate prognosis_prediction_output.json for training cases.

Reads prognosis input JSON from input_root/<case_id>/ and saves per-model outputs to:
  out_dir/<model>/<case_id>/prognosis_prediction_output.json

Usage:
    python -m rare_prognosis.training.generate_llm_outputs \\
        --input-root dataset/prognosis/prognosis_input \\
        --case-ids prog_out/RarePrognosis/train_case_ids.json \\
        --models qwen3-32b gpt-4o \\
        --out-dir prog_out
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from data_io import load_json
from llm_generation import _call_prognosis_model_sync

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

_INPUT_FNAMES = ("prognosis_prediction.json", "prognosis_prediction_input.json")


def _find_prognosis_input(case_dir: Path) -> Optional[Dict]:
    for fname in _INPUT_FNAMES:
        obj = load_json(case_dir / fname)
        if isinstance(obj, dict) and obj:
            return obj
    return None


def _process_one(
    case_id: str,
    model: str,
    *,
    input_root: Path,
    out_dir: Path,
    overwrite: bool,
    max_tokens: int,
    temperature: float,
    max_retries: int,
    retry_delay_sec: float,
    dry_run: bool,
) -> bool:
    out_path = out_dir / model / case_id / "prognosis_prediction_output.json"
    if not overwrite and out_path.is_file():
        return True

    aggregated = _find_prognosis_input(input_root / case_id)
    if not aggregated:
        logger.warning("[%s] no prognosis input found under %s", case_id, input_root / case_id)
        return False

    try:
        payload = _call_prognosis_model_sync(
            aggregated_record=aggregated,
            model_name=model,
            dry_run=dry_run,
            max_tokens=max_tokens,
            temperature=temperature,
            max_retries=max_retries,
            retry_delay_sec=retry_delay_sec,
        )
    except Exception as e:
        logger.warning("[%s/%s] failed: %s", case_id, model, e)
        return False

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return True


def main() -> None:
    p = argparse.ArgumentParser(description="Batch generate prognosis LLM outputs for training.")
    p.add_argument("--input-root", required=True, help="Root dir with <case_id>/ subdirs containing prognosis input JSON")
    p.add_argument("--case-ids", required=True, help="JSON list of case IDs")
    p.add_argument("--models", nargs="+", required=True, help="Model names to run")
    p.add_argument("--out-dir", required=True, help="Output root: <out-dir>/<model>/<case_id>/prognosis_prediction_output.json")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-retries", type=int, default=3)
    p.add_argument("--retry-delay", type=float, default=2.0)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    with open(args.case_ids) as f:
        case_ids: List[str] = [str(x) for x in json.load(f)]
    logger.info("cases=%d models=%s", len(case_ids), args.models)

    input_root = Path(args.input_root)
    out_dir = Path(args.out_dir)
    total_ok = total_fail = 0

    for model in args.models:
        logger.info("=== model: %s ===", model)
        ok = fail = 0
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {
                ex.submit(
                    _process_one, cid, model,
                    input_root=input_root,
                    out_dir=out_dir,
                    overwrite=args.overwrite,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    max_retries=args.max_retries,
                    retry_delay_sec=args.retry_delay,
                    dry_run=args.dry_run,
                ): cid
                for cid in case_ids
            }
            for fut in as_completed(futs):
                try:
                    if fut.result():
                        ok += 1
                    else:
                        fail += 1
                except Exception as e:
                    logger.warning("case %s: %s", futs[fut], e)
                    fail += 1
        logger.info("[%s] ok=%d fail=%d", model, ok, fail)
        total_ok += ok
        total_fail += fail

    logger.info("done: total ok=%d fail=%d", total_ok, total_fail)


if __name__ == "__main__":
    main()
