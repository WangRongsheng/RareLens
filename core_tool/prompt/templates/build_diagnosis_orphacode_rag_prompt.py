"""Prompt builder for OrphaCode selection from retrieved candidates."""

from __future__ import annotations

from typing import Any, Dict, List


PROMPT_TEMPLATE = """
You are an Orphanet medical coding expert.

Task:
Given one diagnosis name and a short candidate list, choose the single best matching OrphaCode.

Rules:
- You MUST choose ONLY from the provided candidates.
- Prefer exact disease match over partial lexical overlap.
- If none of the candidates is appropriate, return "not found".
- Do not explain your reasoning.

Diagnosis name:
{diagnosis_name}

Candidates:
{candidate_block}

Return STRICT JSON only:
{{
  "predicted_orphacode": "<orphacode or not found>"
}}
""".strip()


def build_diagnosis_orphacode_rag_prompt(
    diagnosis_name: str,
    candidates: List[Dict[str, Any]],
) -> str:
    lines: List[str] = []
    for idx, cand in enumerate(candidates or [], start=1):
        vector_score = cand.get("score")
        rerank_score = cand.get("rerank_score")
        parts = [
            f"{idx}. disease_name: {str(cand.get('disease_name') or '').strip()}",
            f"orphacode: {cand.get('orphacode', '')}",
        ]
        if isinstance(vector_score, (int, float)):
            parts.append(f"vector_score: {float(vector_score):.4f}")
        if isinstance(rerank_score, (int, float)):
            parts.append(f"rerank_score: {float(rerank_score):.4f}")
        lines.append(" | ".join(parts))

    candidate_block = "\n".join(lines) if lines else "(no candidates)"
    return PROMPT_TEMPLATE.format(
        diagnosis_name=str(diagnosis_name or "").strip(),
        candidate_block=candidate_block,
    )