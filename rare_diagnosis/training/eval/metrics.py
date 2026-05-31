"""
Shared ranking metrics for diagnosis evaluation.

Used by eval_ml.py, eval_llm.py, and eval_secondary_metrics.py.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np


def dcg(scores: List[float]) -> float:
    """Discounted Cumulative Gain."""
    return sum(
        (2**rel - 1) / np.log2(idx + 2)
        for idx, rel in enumerate(scores)
        if rel >= 0
    )


def ndcg_at_k(rels: List[int], k: int) -> float:
    """Normalized DCG at position k."""
    rels_k = rels[:k]
    ideal = sorted(rels_k, reverse=True)
    if not ideal or sum(ideal) == 0:
        return 0.0
    idcg = dcg(ideal)
    return dcg(rels_k) / idcg if idcg > 0 else 0.0


def mrr_for_case(rels: List[int]) -> float:
    """Mean Reciprocal Rank for a single case."""
    for idx, rel in enumerate(rels):
        if rel >= 1:
            return 1.0 / (idx + 1)
    return 0.0


def case_metrics(rels: List[int]) -> Dict[str, float]:
    """Compute all ranking metrics for a single case (top-5 truncated)."""
    rels = list(rels)[:5]
    return {
        "Acc@1": 1 if any(r >= 1 for r in rels[:1]) else 0,
        "Acc@3": 1 if any(r >= 1 for r in rels[:3]) else 0,
        "Acc@5": 1 if any(r >= 1 for r in rels[:5]) else 0,
        "MRR": mrr_for_case(rels),
        "NDCG@1": ndcg_at_k(rels, 1),
        "NDCG@3": ndcg_at_k(rels, 3),
        "NDCG@5": ndcg_at_k(rels, 5),
    }


# Canonical list of metric keys (used for table headers and aggregation)
METRIC_KEYS = ["Acc@1", "Acc@3", "Acc@5", "MRR", "NDCG@1", "NDCG@3", "NDCG@5"]
PRIMARY_METRIC_KEYS = ["Acc@1", "Acc@3", "Acc@5"]
SECONDARY_METRIC_KEYS = ["MRR", "NDCG@1", "NDCG@3", "NDCG@5"]
