"""Classification/voting trace helpers (used by prognosis post-process)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from schema import ReasoningTrace, TraceEvidence, TraceItem


def build_classifier_trace(
    case_id: str,
    *,
    stage: str = "prognosis",
    module: str = "RarePrognosis",
    item_key: str = "prediction",
    prediction: str = "",
    final_confidence_score: Optional[int] = None,
    final_reasoning: str = "",
    aggregation_method: str = "vote",
    model_preds: Optional[Dict[str, str]] = None,
    model_confidences: Optional[Dict[str, int]] = None,
    model_explanations: Optional[Dict[str, str]] = None,
) -> ReasoningTrace:
    """Build a single-item ``ReasoningTrace`` from model-level evidence."""
    model_preds = model_preds or {}
    model_confidences = model_confidences or {}
    model_explanations = model_explanations or {}
    evidences: List[TraceEvidence] = []
    for m, pred in model_preds.items():
        evidences.append(
            TraceEvidence(
                model=m,
                matched_item=pred,
                model_confidence_score=model_confidences.get(m),
                reasoning=model_explanations.get(m, ""),
            )
        )
    item = TraceItem(
        item_key=item_key,
        item_name=prediction,
        final_confidence_score=final_confidence_score,
        final_reasoning=final_reasoning,
        evidences=evidences,
        aggregation_method=aggregation_method,
    )
    return ReasoningTrace(case_id=case_id, stage=stage, module=module, items=[item])
