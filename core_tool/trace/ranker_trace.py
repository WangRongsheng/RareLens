"""Build shared ReasoningTrace items for ranked outputs."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional


def build_ranker_trace(
    top_predictions: List[Dict[str, Any]],
    llm_section: Dict[str, Any],
    active_models: List[str],
    match_fn: Callable,
    model_selector: Optional[Callable[[Dict[str, Any]], List[str]]] = None,
    on_miss_fn: Optional[Callable[[str, Dict[str, Any]], Optional[Dict[str, Any]]]] = None,
) -> List[Dict[str, Any]]:
    """
    For each ML-ranked candidate, find which models support it.

    Args:
        top_predictions: [{rank, score, ...}] — any ranked candidate list (diagnosis or treatment)
        llm_section: The relevant section of llm_outputs (e.g. llm_outputs["diagnosis"])
        active_models: Model name list
        match_fn: (model_name, model_output_dict, prediction_dict) -> Optional[dict]
                  Returns the matched item from the model's output, or None.

    Returns:
        List of trace dicts, one per candidate.
    """
    traces = []
    for pred in top_predictions:
        models = model_selector(pred) if callable(model_selector) else active_models
        if not isinstance(models, list):
            models = list(active_models)
        supporting = []
        for model in models:
            model_output = llm_section.get(model, {})
            matched = match_fn(model, model_output, pred)
            if matched is not None:
                supporting.append({
                    "model": model,
                    **matched,
                })
                continue
            if callable(on_miss_fn):
                fallback = on_miss_fn(model, pred)
                if isinstance(fallback, dict):
                    supporting.append({
                        "model": model,
                        **fallback,
                    })

        traces.append({
            **pred,
            "appear_count": len(supporting),
            "supporting_models": supporting,
        })
    return traces
