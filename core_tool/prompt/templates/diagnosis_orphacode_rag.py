"""Prompt for selecting Orpha code from retrieved diagnosis candidates."""

from __future__ import annotations

from typing import Any, Dict, List


DIAGNOSIS_ORPHACODE_RAG_PROMPT = """
You are an Orphanet medical coding expert.
You will be given a disease name and semantically retrieved candidates (from a knowledge-base Excel file).

Select the OrphaCode that best matches the disease from the candidates.
If no match is found, output Not Found.

You may only judge based on the provided candidates; do not guess.

Output strict JSON only, for example:
{{
  "predicted_orphacode": "558"
}}

Disease name:
{diagnosis_name}

Candidates (sorted by semantic similarity):
{rag_context}
""".strip()


def build_diagnosis_orphacode_rag_prompt(
    *,
    diagnosis_name: str,
    candidates: List[Dict[str, Any]],
) -> str:
    lines = []
    for idx, cand in enumerate(candidates, start=1):
        lines.append(
            f"{idx}. Disease: {cand.get('disease_name', '')} | "
            f"OrphaCode: {cand.get('orphacode', '')} | "
            f"Similarity: {float(cand.get('score', 0.0)):.4f}"
        )
    rag_context = "\n".join(lines) if lines else "(no candidates)"
    return DIAGNOSIS_ORPHACODE_RAG_PROMPT.format(
        diagnosis_name=str(diagnosis_name or "").strip(),
        rag_context=rag_context,
    )
