"""Diagnosis pipeline config schema (typed view of configs/diagnosis_config.yaml)."""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class DiagnosisConfig(BaseModel):
    version: int = 1

    # RAG / ontology
    ontology_path: str = ""
    enable_orphacode_rag: bool = True
    orphacode_rag_embedding_model: str = ""
    orphacode_rag_top_k: int = 5
    orphacode_rag_llm_model: str = ""
    orphacode_rag_embedding_batch: int = 256
    orphacode_rag_vector_cache_dir: str = ""
    orphacode_rag_dry_run: bool = False
    orphacode_rag_base_url: str = ""
    orphacode_rag_embedding_base_url: str = ""
    orphacode_rag_embedding_api_key: str = ""
    # Optional: path to orphacode_acc.xlsx (or .json) containing the trusted set of orphacodes.
    # When set, RAG retrieval is constrained to only return codes present in this file.
    orphacode_acc_path: str = ""
    # FAISS cosine-similarity thresholds for name resolution (range ~0–1).
    # New semantics:
    #   retrieval_score_floor: score below this is discarded.
    #   llm_trigger_score:     score below this will NOT trigger LLM fallback.
    #   auto_accept_score:     score >= this uses top-1 directly, skipping LLM.
    # Legacy fields are kept for backward compatibility with older configs/call sites.
    orphacode_rag_retrieval_score_floor: float = 0.72
    orphacode_rag_llm_trigger_score: float = 0.93
    orphacode_rag_auto_accept_score: float = 0.94
    orphacode_rag_high_confidence_score: float = 0.94
    orphacode_rag_min_retrieval_score: float = 0.72

    # Semantic model (sentence-transformers)
    semantic_model: str = ""

    # Feature schema/release metadata
    feature_schema_version: str = ""
    release_manifest_path: Optional[str] = None

    # LLM recommended knobs (not always enforced by code paths, but useful for visibility)
    llm_recommended_max_tokens: int = 4096
    llm_recommended_temperature: float = 0.0
    llm_recommended_stream: bool = False

    # LLM generation orchestration controls (optional; fallback to code defaults if unset)
    llm_per_model_timeout_sec: float = Field(default=120.0, ge=1.0)
    llm_total_timeout_sec: Optional[float] = Field(default=None, ge=1.0)
    llm_max_concurrency: int = Field(default=8, ge=1)

    # ------------------------------------------------------------------
    # Diagnosis rerank: elite weights / kings / tag aliases (optional)
    # ------------------------------------------------------------------
    # When set, merged in feature_core.make_diagnosis_elite_runtime for this run.
    # - elite_model_weights: full replacement dict (canonical tag -> weight) if non-empty.
    # - kings: canonical tags for kings_consensus / is_king_top1; if omitted, code defaults apply.
    # - diagnosis_elite_tag_aliases: raw llm_outputs key -> canonical tag (merged over configs/models).
    elite_model_weights: Optional[Dict[str, float]] = None
    kings: Optional[List[str]] = None
    diagnosis_elite_tag_aliases: Optional[Dict[str, str]] = None


__all__ = ["DiagnosisConfig"]

