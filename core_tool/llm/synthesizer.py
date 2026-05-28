"""
core/llm/synthesizer.py — Merge multiple reasoning texts into one.
Uses core/llm/client + core/prompt/templates/reasoning_synthesis.
"""

import logging
from typing import List

from core_tool.llm.client import LLMClient
from core_tool.prompt.templates.reasoning_synthesis import (
    REASONING_SYNTHESIS_PROMPT,
    resolve_synthesis_fields,
)

logger = logging.getLogger(__name__)

# Align with rare_alert risk stage: JSON-oriented system message for structured synthesis output.
_SYNTHESIS_SYSTEM_MESSAGE = (
    "You are a helpful assistant designed to output the final answer in JSON."
)

# Keep postprocess prompts aligned with full consultation JSON; cap avoids runaway payloads.
_MAX_PATIENT_CONTEXT_CHARS = 50_000


class ReasoningSynthesizer:
    """
    Merges multiple LLM reasoning texts into one integrated paragraph.
    Calls GPT via LLMClient; falls back to concatenation in dry_run or on failure.
    """

    def __init__(self, llm_client: LLMClient):
        self.client = llm_client

    def synthesize(
        self,
        reasoning_texts: List[str],
        task_type: str = "diagnosis",
        patient_context: str = "",
    ) -> str:
        """
        Args:
            reasoning_texts: List of reasoning strings from different models
            task_type: Key from TASK_SYNTHESIS_CONFIG (e.g. "diagnosis_primary")
            patient_context: Patient info for the prompt
        Returns:
            Integrated reasoning string
        """
        debug = bool(getattr(self.client, "debug", False))
        if not reasoning_texts:
            if debug:
                print(f"[ReasoningSynthesizer] skip: empty reasoning_texts task_type={task_type!r}")
            return ""
        if len(reasoning_texts) == 1:
            if debug:
                t = (reasoning_texts[0] or "").strip()
                print(
                    f"[ReasoningSynthesizer] skip: single reasoning (no LLM call) "
                    f"task_type={task_type!r} text_len={len(t)}"
                )
            return reasoning_texts[0]
        if self.client.dry_run:
            if debug:
                print(
                    f"[ReasoningSynthesizer] skip: client.dry_run=True (no LLM call) "
                    f"task_type={task_type!r} n_reasoning={len(reasoning_texts)}"
                )
            return " ; ".join(reasoning_texts)

        reasoning_block = ""
        for i, text in enumerate(reasoning_texts, 1):
            reasoning_block += f"\n[Model {i}]: {text}\n"

        task_phrase, patient_info_note = resolve_synthesis_fields(task_type)
        prompt = REASONING_SYNTHESIS_PROMPT.format(
            task_type=task_phrase,
            patient_info_note=patient_info_note,
            patient_context=(
                patient_context[:_MAX_PATIENT_CONTEXT_CHARS]
                if len(patient_context) > _MAX_PATIENT_CONTEXT_CHARS
                else patient_context
            ),
            reasoning_texts=reasoning_block,
        ) + "\n\nImportant: Write the integrated clinical reasoning in English."

        if getattr(self.client, "debug", False):
            logger.info(
                "[ReasoningSynthesizer] synthesize task_type=%r n_reasoning=%s patient_context_len=%s prompt_len=%s",
                task_type,
                len(reasoning_texts),
                len(patient_context or ""),
                len(prompt),
            )
            # Keep preview short; full prompt is printed by LLMClient when debug=True.
            logger.info("[ReasoningSynthesizer] prompt_preview=%r", prompt[:280])

        result = self.client.call_and_parse_json(
            _SYNTHESIS_SYSTEM_MESSAGE + "\n\n" + prompt,
            key="integrated_clinical_reasoning",
            max_tokens=4000,
        )
        if result:
            return result

        logger.warning(
            "ReasoningSynthesizer: synthesis failed, falling back to concatenation (task_type=%r, n_reasoning=%s, model=%r). "
            "Check WARNING messages from call_and_parse_json / LLM call failed above for the root cause.",
            task_type,
            len(reasoning_texts),
            getattr(self.client, "model", None),
        )
        return " ; ".join(reasoning_texts)

    def generate_from_context(
        self,
        *,
        diagnosis_name: str,
        task_type: str = "diagnosis",
        patient_context: str = "",
    ) -> str:
        """
        Generate a reasoning paragraph even when upstream models provided no reasoning.

        Returns an empty string in dry_run or on hard failure.
        """
        debug = bool(getattr(self.client, "debug", False))
        if self.client.dry_run:
            if debug:
                print(f"[ReasoningSynthesizer] generate_from_context skip: client.dry_run=True task_type={task_type!r}")
            return ""
        name = str(diagnosis_name or "").strip()
        if not name:
            if debug:
                print(f"[ReasoningSynthesizer] generate_from_context skip: empty diagnosis_name task_type={task_type!r}")
            return ""

        task_phrase, patient_info_note = resolve_synthesis_fields(task_type)
        prompt = (
            "You are a clinical reasoning assistant.\n"
            "Based only on the patient information provided, generate a concise, readable clinical reasoning paragraph for the given diagnosis.\n\n"
            "Requirements:\n"
            "- Do NOT invent tests, findings, or history not present in the patient info.\n"
            "- It is OK to express uncertainty (e.g., 'requires further confirmation').\n"
            "- Write in English.\n"
            "- Output strict JSON with the key integrated_clinical_reasoning.\n\n"
            f"Task type: {task_phrase}\n"
            f"Patient block note: {patient_info_note}\n\n"
            "Patient information:\n"
            f"{(patient_context[:_MAX_PATIENT_CONTEXT_CHARS] if len(patient_context) > _MAX_PATIENT_CONTEXT_CHARS else patient_context)}\n\n"
            f"Candidate diagnosis: {name}\n\n"
            'Return JSON only: {"integrated_clinical_reasoning":"..."}'
        )
        if debug:
            logger.info(
                "[ReasoningSynthesizer] generate_from_context task_type=%r diagnosis_name=%r patient_context_len=%s prompt_len=%s",
                task_type,
                name[:80],
                len(patient_context or ""),
                len(prompt),
            )
        result = self.client.call_and_parse_json(
            _SYNTHESIS_SYSTEM_MESSAGE + "\n\n" + prompt,
            key="integrated_clinical_reasoning",
            max_tokens=4000,
        )
        return result or ""
