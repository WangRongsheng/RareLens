#!/usr/bin/env python3
"""
Shared feature-encoding utilities for prognosis stacking pipeline.

Provides one-hot encoding of base model predictions and text feature
extraction from LLM explanation strings.
"""
from __future__ import annotations

from typing import Dict, List, Optional


def encode_features(
    case_ids: List[str],
    preds: Dict[str, Dict[str, Optional[str]]],
    label_list: List[str],
) -> List[List[float]]:
    """One-hot encode base model predictions into a feature matrix.

    Each model contributes ``len(label_list)`` binary columns (one per label).
    """
    label_index = {l: i for i, l in enumerate(label_list)}
    model_names = sorted(preds.keys())
    n_models = len(model_names)
    n_labels = len(label_list)
    X: List[List[float]] = []
    for cid in case_ids:
        row = [0.0] * (n_models * n_labels)
        for m_idx, mn in enumerate(model_names):
            pred = preds[mn].get(cid)
            if pred is not None and pred in label_index:
                row[m_idx * n_labels + label_index[pred]] = 1.0
        X.append(row)
    return X


def extract_text_features(text: str, keywords: List[str]) -> Dict[str, float]:
    """Extract text statistics and keyword counts from explanation text."""
    text_l = text.lower()
    words = text_l.split()
    feats: Dict[str, float] = {
        "expl_chars": float(len(text)),
        "expl_words": float(len(words)),
        "expl_sentences": float(text.count(".") + text.count("\u3002") + text.count("!")),
        "expl_negations": float(sum(text_l.count(k) for k in ["no ", "not ", "without ", "\u5426\u8ba4", "\u65e0", "\u672a\u89c1"])),
        "expl_numbers": float(sum(ch.isdigit() for ch in text)),
    }
    for k in keywords:
        feats[f"kw_{k}"] = float(text_l.count(k))
    return feats
