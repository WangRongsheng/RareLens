"""Parse free-form risk LLM output into legacy structured RiskOutput fields."""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Tuple

from pydantic import ValidationError

from core_tool.parser.json_extractor import extract as extract_json_object
from schema import InsightItem, RiskOutput

logger = logging.getLogger(__name__)

# Stable warning codes for monitoring; values are confidence penalties (sum capped in _finalize_parse_meta).
_WARNING_PENALTIES: Dict[str, float] = {
    "json_used_prefix_strip": 0.12,
    "risk_score_from_text_fragment": 0.08,
    "key_insights_padded": 0.1,
    "key_insights_from_root_keys": 0.08,
    "key_insights_merged_root_tail": 0.06,
    "key_insights_flat_reassembled": 0.07,
    "key_insights_flat_lifted": 0.06,
    "risk_explanation_nested_unwrap": 0.06,
    "parse_fallback_freeform": 0.4,
    "key_insights_fallback_heuristic": 0.12,
}


def _finalize_parse_meta(warnings: List[str]) -> Tuple[float, List[str]]:
    dedup = list(dict.fromkeys(warnings))
    penalty = sum(_WARNING_PENALTIES.get(w, 0.05) for w in dedup)
    confidence = max(0.12, round(1.0 - min(penalty, 0.88), 3))
    return confidence, dedup


def _clip_score(score: int) -> int:
    return max(0, min(100, int(score)))


def _extract_json_blob(text: str) -> Dict[str, Any] | None:
    """Delegate to ``json_extractor.extract`` (with balanced-brace support); no need to re-implement greedy regex."""
    shared = extract_json_object(text)
    return shared if isinstance(shared, dict) else None


def _strip_common_llm_prefixes(text: str) -> str:
    """Remove wrappers like **Final Output**: before JSON / code fences."""
    s = text.strip()
    s = re.sub(r"(?is)^\s*\*\*final output\*\*:\s*", "", s)
    return s.strip()


def _recover_risk_score_from_jsonish(text: str) -> int:
    """
    When the model prints JSON but it is invalid / not fully parsed, recover
    score from fragments like \"risk_score\": 90 (last occurrence wins).
    """
    matches = list(re.finditer(r'["\']risk_score["\']\s*:\s*(\d{1,3})', text))
    if not matches:
        return 0
    return _clip_score(int(matches[-1].group(1)))


def _strip_opening_code_fence(text: str) -> str:
    """Remove a leading ```json / ``` fence without requiring a closing fence."""
    s = text.strip()
    return re.sub(r"^```(?:json)?\s*\n?", "", s, count=1, flags=re.IGNORECASE)


def _decode_json_string_content(raw: str) -> str:
    """Decode escapes inside a JSON string fragment (handles truncated tail)."""
    i = 0
    out: List[str] = []
    while i < len(raw):
        if raw[i] == "\\" and i + 1 < len(raw):
            esc = raw[i + 1]
            if esc == '"':
                out.append('"')
                i += 2
                continue
            if esc == "\\":
                out.append("\\")
                i += 2
                continue
            if esc == "/":
                out.append("/")
                i += 2
                continue
            if esc == "n":
                out.append("\n")
                i += 2
                continue
            if esc == "r":
                out.append("\r")
                i += 2
                continue
            if esc == "t":
                out.append("\t")
                i += 2
                continue
            if esc == "u" and i + 6 <= len(raw):
                hexpart = raw[i + 2 : i + 6]
                try:
                    out.append(chr(int(hexpart, 16)))
                    i += 6
                    continue
                except ValueError:
                    pass
            out.append(raw[i])
            i += 1
            continue
        out.append(raw[i])
        i += 1
    return "".join(out)


def _extract_risk_explanation_string_value(text: str) -> str | None:
    """
    When outer JSON is invalid or truncated (no closing fence / brace), pull the
    value of \"risk_explanation\" from the text.
    Handles both quoted and unquoted values.
    """
    m = re.search(r'"risk_explanation"\s*:\s*"', text)
    if m:
        i = m.end()
        raw_parts: List[str] = []
        while i < len(text):
            if text[i] == "\\" and i + 1 < len(text):
                raw_parts.append(text[i : i + 2])
                i += 2
                continue
            if text[i] == '"':
                break
            raw_parts.append(text[i])
            i += 1
        raw = "".join(raw_parts)
        if raw:
            return _decode_json_string_content(raw)

    # Fallback: unquoted value — "risk_explanation": The text here...
    m2 = re.search(r'"risk_explanation"\s*:\s*(?!")(.+?)(?=\s*\n\s*[}\]]|\Z)', text, re.DOTALL)
    if m2:
        val = m2.group(1).strip().rstrip("}").strip()
        if val:
            return val
    return None


def _extract_insights_from_colon_format(text: str) -> List[Tuple[str, float, str]]:
    """
    Handle model output format: "insightN": "label": description_text
    where description is unquoted and weight is missing.
    Example:
        "insight1": "Progressive dysphagia": The patient has a chronic...
    """
    rows: List[Tuple[str, float, str]] = []
    pattern = re.compile(
        r'"insight(\d)"\s*:\s*"([^"]+)"\s*:\s*([^\n"{}[\]]+)',
        re.IGNORECASE,
    )
    for m in sorted(pattern.finditer(text), key=lambda x: int(x.group(1))):
        label = m.group(2).strip()
        desc = m.group(3).strip().rstrip(",")
        rows.append((label, 0.2, desc))
    return rows


def _unwrap_nested_json_explanation(explanation: str) -> str:
    """
    If risk_explanation itself contains a full JSON (or ```json ... ```) with a
    real human-readable risk_explanation field, return that inner string.
    """
    if not explanation or not str(explanation).strip():
        return explanation
    original = str(explanation)
    s = _strip_common_llm_prefixes(original)
    s = _strip_opening_code_fence(s)
    s = re.sub(r"\n?```\s*$", "", s).strip()

    blob = extract_json_object(s)
    if isinstance(blob, dict):
        inner = blob.get("risk_explanation")
        if isinstance(inner, str) and inner.strip():
            return inner.strip()

    inner_frag = _extract_risk_explanation_string_value(s)
    if inner_frag is not None:
        return inner_frag.strip()

    if s != original.strip():
        return s
    return explanation


def _extract_score(text: str) -> int:
    m = re.search(r"RISK_SCORE\s*[:：]\s*(\d{1,3})", text, re.IGNORECASE)
    if m:
        return _clip_score(int(m.group(1)))

    m = re.search(r"risk\s*score[^0-9]{0,20}(\d{1,3})", text, re.IGNORECASE)
    if m:
        return _clip_score(int(m.group(1)))

    m = re.search(r"\b(\d{1,3})\s*/\s*100\b", text)
    if m:
        return _clip_score(int(m.group(1)))
    return 0


def _extract_explanation(text: str) -> str:
    m = re.search(r"RISK_EXPLANATION\s*[:：]\s*([\s\S]+)$", text, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    parts = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    return parts[-1] if parts else ""


def _parse_key_insight_line(line: str) -> Tuple[str, float, str] | None:
    clean = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line).strip()
    if not clean:
        return None
    fields = [p.strip() for p in clean.split("|")]
    if not fields:
        return None
    insight = fields[0]
    weight = 0.2
    description = ""
    for f in fields[1:]:
        w = re.search(r"weight\s*=\s*([0-9]*\.?[0-9]+)", f, re.IGNORECASE)
        if w:
            weight = float(w.group(1))
            continue
        d = re.search(r"description\s*=\s*(.+)", f, re.IGNORECASE)
        if d:
            description = d.group(1).strip()
    if not description:
        description = insight
    return insight, weight, description


def _insight_text_from_json_item(item: Dict[str, Any], slot_index: int) -> str:
    """
    Prefer the key matching this array position (insight1 for index 0, …) so we do not
    pick generic \"insight\" when insight1 also exists. Bare \"insight\" is last resort.
    """
    preferred = f"insight{slot_index + 1}"
    v = item.get(preferred)
    if v is not None and str(v).strip():
        return str(v).strip()
    for n in range(1, 6):
        key = f"insight{n}"
        if key == preferred:
            continue
        v = item.get(key)
        if v is not None and str(v).strip():
            return str(v).strip()
    v = item.get("name")
    if v is not None and str(v).strip():
        return str(v).strip()
    v = item.get("insight")
    if v is not None and str(v).strip():
        return str(v).strip()
    return ""


def _coerce_weight(v: Any, default: float = 0.2) -> float:
    if v is None:
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _reassemble_flat_insights(raw_list: List[Any]) -> List[Dict[str, Any]]:
    """
    Pattern A: the model sometimes expands each insight's fields into separate string elements, e.g.:
        ["insight4", "weight", "description", "Barium enema...", "0.2", "Microcolon..."]
    Reassemble adjacent insightN labels and the following text/weight/description triplet back into a dict.
    Elements that are already dicts are left unchanged.
    """
    result: List[Dict[str, Any]] = []
    i = 0
    while i < len(raw_list):
        item = raw_list[i]
        if isinstance(item, dict):
            result.append(item)
            i += 1
            continue
        _im = re.match(r"^(insight\d+)(?::\s*(.*))?$", item.strip(), re.IGNORECASE) if isinstance(item, str) else None
        if _im:
            tag = _im.group(1).lower()
            _inline = (_im.group(2) or "").strip()
            pending: Dict[str, Any] = {tag: _inline if _inline else None, "weight": 0.2, "description": ""}
            j = i + 1
            filled = 0
            while j < len(raw_list) and filled < 3:
                nxt = raw_list[j]
                if isinstance(nxt, str):
                    if nxt.strip().lower() in ("weight", "description"):
                        j += 1
                        continue
                    try:
                        pending["weight"] = float(nxt.strip())
                        filled += 1
                        j += 1
                        continue
                    except ValueError:
                        pass
                    if pending[tag] is None:
                        pending[tag] = nxt.strip()
                        filled += 1
                    elif not pending["description"]:
                        pending["description"] = nxt.strip()
                        filled += 1
                elif isinstance(nxt, (int, float)):
                    pending["weight"] = float(nxt)
                    filled += 1
                elif isinstance(nxt, dict):
                    break
                j += 1
            if pending[tag] is None:
                pending[tag] = tag
            if not pending["description"]:
                pending["description"] = str(pending[tag])
            result.append(pending)
            i = j
            continue
        i += 1
    return result


def _lift_flat_insight_dict(obj: Dict[str, Any]) -> Dict[str, Any]:
    """
    Pattern B: top-level key_insights is missing or not an array, but insight appears as flat keys in the same dict.

    B1 — only one insightN key: treat the whole dict as a single insight (with weight/description at the same level).
    B2 — multiple insightN keys: split into multiple items, each containing only the corresponding insightN;
         share weight/description, or prefer matching numeric suffixes like weight2, description3.
    """
    ki = obj.get("key_insights")
    if isinstance(ki, list) and len(ki) > 0:
        return obj

    insight_keys = sorted(
        [k for k in obj if re.match(r"^insight\d+$", str(k), re.IGNORECASE)],
        key=lambda k: int(re.search(r"\d+", str(k)).group()),
    )
    if not insight_keys:
        return obj

    risk_score = obj.get("risk_score", 0)
    risk_explanation = obj.get("risk_explanation", "")

    if len(insight_keys) == 1:
        items: List[Dict[str, Any]] = [obj]
    else:
        items = []
        shared_weight = obj.get("weight", 0.2)
        shared_desc = obj.get("description", "")
        for k in insight_keys:
            n = re.search(r"\d+", str(k)).group()
            w = obj.get(f"weight{n}", obj.get(f"weight_{n}", shared_weight))
            d = obj.get(f"description{n}", obj.get(f"description_{n}", shared_desc))
            val = obj[k]
            desc = d or val
            items.append(
                {
                    str(k): val,
                    "weight": _coerce_weight(w, 0.2),
                    "description": str(desc).strip() if desc is not None else "",
                }
            )

    return {
        "key_insights": items,
        "risk_score": risk_score,
        "risk_explanation": risk_explanation,
    }


def _is_placeholder_insight_row(row: Tuple[str, float, str], slot_1based: int) -> bool:
    insight, _, desc = row
    return (
        insight == f"insight_{slot_1based}_missing"
        and desc == "model response missing"
    )


def _has_any_root_insight(obj: Dict[str, Any]) -> bool:
    """Still a valid signal when the model flattened insights at the root level (no key_insights array)."""
    for slot in range(5):
        key = f"insight{slot + 1}"
        val = obj.get(key)
        if isinstance(val, dict) and _insight_text_from_json_item(val, slot):
            return True
        if isinstance(val, str) and val.strip():
            return True
    return False


def _rows_from_root_insight_keys(obj: Dict[str, Any]) -> List[Tuple[str, float, str]]:
    """
    Build one row per insight1~insight5 slot (order consistent with _to_legacy_insight_items).
    Missing slots use a placeholder tuple so results can be merged with array-parsed rows by tail segment.
    """
    rows: List[Tuple[str, float, str]] = []
    for slot in range(5):
        key = f"insight{slot + 1}"
        val = obj.get(key)
        if isinstance(val, dict):
            insight = _insight_text_from_json_item(val, slot)
            if insight:
                w = _coerce_weight(val.get("weight"), 0.2)
                desc = str(val.get("description") or insight).strip() or insight
                rows.append((insight, w, desc))
                continue
        if isinstance(val, str) and val.strip():
            s = val.strip()
            rows.append((s, 0.2, s))
            continue
        sb = slot + 1
        rows.append((f"insight_{sb}_missing", 0.0, "model response missing"))
    return rows


def _failure_risk_output(reason: str = "") -> RiskOutput:
    """Return a schema-compliant placeholder when parsing fails, to avoid polluting fields with heuristic text fragments."""
    msg = (reason or "").strip() or (
        "Failed to parse model output into a schema-compliant JSON. "
        "Recommended: keep json_object_response: true, set stream: false, and increase max_tokens as needed."
    )
    short = "(parse failure placeholder — not a clinical conclusion)"
    items: List[InsightItem] = []
    for idx in range(1, 6):
        items.append(
            InsightItem.model_validate(
                {
                    f"insight{idx}": short,
                    "weight": 0.2,
                    "description": short,
                }
            )
        )
    return RiskOutput(key_insights=items, risk_score=0, risk_explanation=msg)


def _risk_output_from_loose_dict(obj: Dict[str, Any], text: str, warnings: List[str]) -> RiskOutput:
    """A dict was extracted from the text but does not validate as RiskOutput: extract fields using legacy logic and pad to 5 items."""
    pre_lift = dict(obj)
    obj = _lift_flat_insight_dict(obj)
    _ki_before = pre_lift.get("key_insights")
    _had_nonempty_ki = isinstance(_ki_before, list) and len(_ki_before) > 0
    lift_applied = (not _had_nonempty_ki) and bool(obj.get("key_insights"))
    if lift_applied and _has_any_root_insight(pre_lift):
        warnings.append("key_insights_flat_lifted")

    rows: List[Tuple[str, float, str]] = []
    raw_insights = list(obj.get("key_insights", []) or [])
    if any(not isinstance(x, dict) for x in raw_insights):
        warnings.append("key_insights_flat_reassembled")
    raw_insights = _reassemble_flat_insights(raw_insights)
    for slot_index, item in enumerate(raw_insights):
        if not isinstance(item, dict):
            continue
        insight = _insight_text_from_json_item(item, slot_index)
        if not insight:
            continue
        weight = _coerce_weight(item.get("weight"), 0.2)
        description = str(item.get("description") or insight).strip() or insight
        rows.append((insight, weight, description))
        if len(rows) == 5:
            break

    root_rows = _rows_from_root_insight_keys(pre_lift)
    if len(rows) == 0 and _has_any_root_insight(pre_lift):
        rows = root_rows
        warnings.append("key_insights_from_root_keys")
    elif 0 < len(rows) < 5:
        if not lift_applied and any(
            not _is_placeholder_insight_row(root_rows[i], i + 1) for i in range(len(rows), 5)
        ):
            for i in range(len(rows), 5):
                rows.append(root_rows[i])
            warnings.append("key_insights_merged_root_tail")

    # If still short, try colon-format extraction from raw text as last resort
    if len(rows) < 5:
        colon_rows = _extract_insights_from_colon_format(text)
        if len(colon_rows) > len(rows):
            rows = colon_rows
            warnings.append("key_insights_padded")
        else:
            warnings.append("key_insights_padded")
    elif len(rows) == 5 and any(
        _is_placeholder_insight_row(rows[i], i + 1) for i in range(5)
    ):
        warnings.append("key_insights_padded")

    base_score = _clip_score(int(obj.get("risk_score", 0) or 0))
    recovered = _recover_risk_score_from_jsonish(text)
    score = max(base_score, recovered)
    if recovered > base_score:
        warnings.append("risk_score_from_text_fragment")

    explanation_raw = str(obj.get("risk_explanation", "") or "")
    # If dict yielded no explanation, try recovering from raw text (handles unquoted values)
    if not explanation_raw.strip():
        recovered_expl = _extract_risk_explanation_string_value(text)
        if recovered_expl:
            explanation_raw = recovered_expl
    explanation = _unwrap_nested_json_explanation(explanation_raw)
    if explanation.strip() != explanation_raw.strip():
        warnings.append("risk_explanation_nested_unwrap")

    confidence, wcodes = _finalize_parse_meta(warnings)
    if wcodes:
        logger.warning(
            "[parse_risk_response] confidence=%s codes=%s (loose_dict)",
            confidence,
            wcodes,
        )
    return RiskOutput(
        key_insights=_to_legacy_insight_items(rows),
        risk_score=score,
        risk_explanation=explanation,
    )


def _to_legacy_insight_items(rows: List[Tuple[str, float, str]]) -> List[InsightItem]:
    out: List[InsightItem] = []
    normalized = list(rows[:5])
    while len(normalized) < 5:
        idx = len(normalized) + 1
        normalized.append((f"insight_{idx}_missing", 0.0, "model response missing"))

    for idx, (insight, wt, desc) in enumerate(normalized, start=1):
        payload: Dict[str, Any] = {
            f"insight{idx}": insight,
            "weight": wt,
            "description": desc,
        }
        out.append(InsightItem.model_validate(payload))
    return out


def _extract_key_insights(text: str, warnings: List[str]) -> List[InsightItem]:
    block_match = re.search(
        r"KEY_INSIGHTS\s*[:：]\s*([\s\S]*?)(?:\n\s*RISK_EXPLANATION\s*[:：]|\Z)",
        text,
        re.IGNORECASE,
    )
    candidate = block_match.group(1) if block_match else text
    lines = [ln for ln in candidate.splitlines() if ln.strip()]
    rows: List[Tuple[str, float, str]] = []
    for ln in lines:
        parsed = _parse_key_insight_line(ln)
        if parsed is None:
            continue
        rows.append(parsed)
        if len(rows) == 5:
            break
    if rows:
        return _to_legacy_insight_items(rows)

    warnings.append("key_insights_fallback_heuristic")
    chunks = [c.strip() for c in re.split(r"[。\n;；]+", text) if c.strip()]
    fallback_rows: List[Tuple[str, float, str]] = []
    for c in chunks[:5]:
        fallback_rows.append((c[:120], 0.2, c[:300]))
    return _to_legacy_insight_items(fallback_rows)


def parse_risk_response(raw_text: str, *, allow_freeform_fallback: bool = False) -> RiskOutput:
    """
    Convert model output to ``RiskOutput``.

    ``allow_freeform_fallback`` is **disabled** by default: only JSON (full or balanced-brace extraction)
    is accepted; on failure a placeholder is returned to avoid sentence-splitting "parsing" of raw text.
    """
    warnings: List[str] = []
    text = (raw_text or "").strip()

    obj = _extract_json_blob(text)
    if not obj:
        obj = _extract_json_blob(_strip_common_llm_prefixes(text))
        if obj:
            warnings.append("json_used_prefix_strip")

    if obj is not None:
        try:
            return RiskOutput.model_validate(obj)
        except ValidationError as e:
            logger.warning(
                "[parse_risk_response] Direct JSON→RiskOutput validation failed, attempting loose field extraction: %s",
                e,
            )
        try:
            return _risk_output_from_loose_dict(obj, text, warnings)
        except (ValidationError, ValueError) as e:
            logger.warning("[parse_risk_response] Loose extraction failed: %s", e)
            return _failure_risk_output(f"JSON structure cannot be mapped to RiskOutput: {e}")

    if not allow_freeform_fallback:
        return _failure_risk_output(
            "Failed to extract a JSON object (freeform_parse_fallback=false). "
            "Ensure json_object_response is enabled and prefer stream=false to receive complete JSON."
        )

    warnings.append("parse_fallback_freeform")
    score = max(_extract_score(text), _recover_risk_score_from_jsonish(text))
    # Try colon-format insights ("insightN": "label": description) before generic line parse
    colon_rows = _extract_insights_from_colon_format(text)
    if colon_rows:
        key_insights = _to_legacy_insight_items(colon_rows)
    else:
        key_insights = _extract_key_insights(text, warnings)
    # Try unquoted risk_explanation before generic paragraph extraction
    explanation_raw = _extract_risk_explanation_string_value(text) or _extract_explanation(text)
    explanation = _unwrap_nested_json_explanation(explanation_raw)
    if explanation.strip() != explanation_raw.strip():
        warnings.append("risk_explanation_nested_unwrap")

    confidence, wcodes = _finalize_parse_meta(warnings)
    logger.warning(
        "[parse_risk_response] confidence=%s codes=%s (freeform_fallback)",
        confidence,
        wcodes,
    )
    return RiskOutput(
        key_insights=key_insights,
        risk_score=score,
        risk_explanation=explanation,
    )