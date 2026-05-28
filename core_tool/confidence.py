"""
core/confidence.py — Confidence score normalization (shared across all modules)
"""

# Consistent with the backup script soft_min_max: when the range is too small, use an effective range of at least 2 as the denominator to avoid amplifying noise.
_MIN_EFFECTIVE_RANGE = 2.0


def _soft_anchor_int(value: float, *, hi: float, lo: float, scale: int) -> int:
    """hi → scale, lo is mapped via effective range; result is rounded and clamped to [1, scale]."""
    real_range = hi - lo
    effective_range = max(float(real_range), _MIN_EFFECTIVE_RANGE)
    raw = scale - (hi - value) * (scale / effective_range)
    return max(1, min(scale, int(round(raw))))


def normalize_scores(scores: list[float], scale: int = 10) -> list[int]:
    """Map a list of ML scores to integers in [1, scale]; denominator is max(range, 2)."""
    if not scores:
        return []
    mn, mx = min(scores), max(scores)
    if mx == mn:
        return [scale] * len(scores)
    return [_soft_anchor_int(s, hi=mx, lo=mn, scale=scale) for s in scores]


def normalize_scores_with_bounds(scores: list[float], *, lo: float, hi: float, scale: int = 10) -> list[int]:
    """
    Same mapping as ``normalize_scores`` but uses precomputed bounds.

    Useful when the caller keeps only top-K candidates but wants to normalize
    against the full candidate distribution (lo/hi computed on full list).
    """
    if not scores:
        return []
    if hi == lo:
        return [scale] * len(scores)
    return [_soft_anchor_int(float(s), hi=float(hi), lo=float(lo), scale=int(scale)) for s in scores]


def normalize_avg_confidence(confidences: list[int]) -> int:
    """Average confidence scores → integer."""
    if not confidences:
        return 5
    return round(sum(confidences) / len(confidences))
