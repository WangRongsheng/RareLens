"""Build validated RiskInput payloads from raw case JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

from core_tool.io_utils import read_json
from core_tool.parser.risk_input_normalize import normalize_risk_input_dict
from schema import RiskInput


DEFAULT_PATIENT_FILE_CANDIDATES: Tuple[str, ...] = (
    "primary_consultation.json",
    "primary_input.json",
    "risk_input.json",
)


def build_risk_input_from_dict(data: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize + validate a dict as schema.RiskInput."""
    normalized = normalize_risk_input_dict(data)
    return RiskInput.model_validate(normalized).model_dump()


def resolve_patient_file(case_dir: Path, candidates: Iterable[str]) -> Path:
    for name in candidates:
        p = case_dir / str(name)
        if p.is_file():
            return p
    expected = ", ".join(str(x) for x in candidates)
    raise FileNotFoundError(f"No patient input file found in {case_dir}; expected one of: {expected}")


def build_risk_input_from_case_dir(
    case_dir: Path,
    *,
    patient_file: str | None = None,
    candidates: Iterable[str] = DEFAULT_PATIENT_FILE_CANDIDATES,
) -> Dict[str, Any]:
    src = case_dir / patient_file if patient_file else resolve_patient_file(case_dir, candidates)
    data = read_json(src)
    return build_risk_input_from_dict(data)


def dump_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build and validate schema.RiskInput from raw case JSON.")
    parser.add_argument(
        "--case-dir",
        type=Path,
        required=True,
        help="Case directory, e.g. data/test_cases/diag_primary/raw_case/15163878",
    )
    parser.add_argument(
        "--patient-file",
        default=None,
        help="Optional explicit patient file name in case-dir. Defaults to auto-detect.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output file path. Defaults to <case-dir>/risk_input.json",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    case_dir = Path(args.case_dir).resolve()
    if not case_dir.is_dir():
        raise SystemExit(f"Case directory not found: {case_dir}")

    risk_input = build_risk_input_from_case_dir(case_dir, patient_file=args.patient_file)
    output_path = Path(args.output).resolve() if args.output else (case_dir / "risk_input.json")
    dump_json(output_path, risk_input)
    print(json.dumps({"case_dir": str(case_dir), "output": str(output_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
