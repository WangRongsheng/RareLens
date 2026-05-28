"""Extract JSON objects from LLM free-form text (multi-strategy)."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional, Tuple, Type

from pydantic import BaseModel


def _first_balanced_object(text: str, start: int = 0) -> Optional[str]:
    """
    Starting from the first ``{``, extract by bracket depth to the matching ``}``,
    avoiding greedy ``{...}`` spanning multiple objects or truncation errors.
    Respects quotes and escape sequences inside strings.
    """
    i = text.find("{", start)
    if i < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    n = len(text)
    j = i
    while j < n:
        c = text[j]
        if in_string:
            if escape:
                escape = False
            elif c == "\\":
                escape = True
            elif c == '"':
                in_string = False
            j += 1
            continue
        if c == '"':
            in_string = True
            j += 1
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[i : j + 1]
        j += 1
    return None


def _strip_think_blocks(text: str) -> str:
    """Strip <think>...</think> reasoning blocks from Qwen3 and similar thinking model outputs."""
    return re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE).strip()


def _repair_missing_open_braces(text: str) -> str:
    """
    In a JSON array context, complete object members that are missing an opening brace {.

    Common LLM output error:
        [
          { "a": 1 },
          "b": 2 }      ← missing opening brace
        ]

    Approach: scan character by character, tracking a { / [ stack. When inside [,
    the previous non-whitespace character is , or [, and a token of the form "key":
    (a string followed immediately by :) is encountered, insert { .
    """
    out: list[str] = []
    i = 0
    n = len(text)
    stack: list[str] = []  # '[' or '{'
    in_str = False
    esc = False
    prev_non_ws = ""

    while i < n:
        c = text[i]

        if esc:
            out.append(c)
            esc = False
            i += 1
            continue

        if in_str:
            if c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            out.append(c)
            i += 1
            continue

        if c == '"':
            # Check if we are in an array context and the previous non-whitespace char is , or [ (start of a new array element).
            if stack and stack[-1] == "[" and prev_non_ws in (",", "["):
                # Look ahead: find the matching closing quote, then check if : follows immediately (i.e., this is a key).
                j = i + 1
                while j < n:
                    if text[j] == "\\":
                        j += 2
                        continue
                    if text[j] == '"':
                        break
                    j += 1
                k = j + 1
                while k < n and text[k] in " \t":
                    k += 1
                if k < n and text[k] == ":":
                    out.append("{")
                    out.append(" ")
                    stack.append("{")  # Sync context so inner object keys are not mistaken for array elements.
            in_str = True
            out.append(c)
            prev_non_ws = c
            i += 1
            continue

        if c == "{":
            stack.append("{")
            prev_non_ws = c
        elif c == "[":
            stack.append("[")
            prev_non_ws = c
        elif c == "}":
            if stack and stack[-1] == "{":
                stack.pop()
            prev_non_ws = c
        elif c == "]":
            if stack and stack[-1] == "[":
                stack.pop()
            prev_non_ws = c
        elif c not in " \t\n\r":
            prev_non_ws = c

        out.append(c)
        i += 1

    return "".join(out)


def _close_unclosed_last_array_item(text: str) -> str:
    """
    If the last element (object) of an array is missing a closing brace },
    insert one before ].
    Pattern: a char that is not } or ] + optional whitespace + newline + optional whitespace + ]
    Replaces only once (the nearest ]).
    """
    return re.sub(r'([^}\]\s])(\s*\n\s*\])', r'\1 }\2', text, count=1)


def extract(raw_text: str) -> Optional[Dict[str, Any]]:
    """
    Strategy order: full json.loads → ```json``` code block → first balanced-brace JSON object → last-resort greedy ``{...}``.
    <think>...</think> blocks from Qwen3 and similar thinking models are stripped before all strategies.
    """
    if not raw_text or not str(raw_text).strip():
        return None

    s = _strip_think_blocks(str(raw_text).strip())

    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", s, re.IGNORECASE)
    if fence:
        chunk = fence.group(1).strip()
        try:
            obj = json.loads(chunk)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass

    balanced = _first_balanced_object(s, 0)
    if balanced:
        try:
            obj = json.loads(balanced)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass

    m = re.search(r"\{[\s\S]*\}", s, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass

    # Strategy 5: repair missing opening braces { in array elements, then retry.
    # Case: the model wrote "insightN": ... } but omitted the leading {, causing orphan } misalignment.
    # After repair, typically only the last element is missing }, which json_repair below can handle.
    patched = _repair_missing_open_braces(s)
    if patched != s:
        try:
            obj = json.loads(patched)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
        # Strategy 5b: close the unclosed last array element }, then retry once more.
        patched2 = _close_unclosed_last_array_item(patched)
        if patched2 != patched:
            try:
                obj = json.loads(patched2)
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                pass
        # Run balanced-brace extraction again on the repaired text.
        balanced2 = _first_balanced_object(patched2 if patched2 != patched else patched, 0)
        if balanced2:
            try:
                obj = json.loads(balanced2)
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                pass

    # Strategy 6 (final fallback): json_repair handles remaining syntax errors (e.g. last element missing }).
    # Prefer running on text where opening braces are already fixed, to prevent repair from incorrectly nesting top-level fields into an array.
    try:
        from json_repair import repair_json  # type: ignore

        candidate = patched if patched != s else (m.group(0) if m else s)
        repaired = repair_json(candidate, return_objects=True)
        if isinstance(repaired, dict) and repaired:
            return repaired
    except Exception:
        pass

    return None


def extract_json_object(raw_text: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Backward-compatible shim for the old ``pipeline.parser`` interface."""
    obj = extract(raw_text)
    if obj is not None:
        return obj, None
    return None, "no_json_object"


def extract_with_fallback(
    raw_text: str,
    schema_cls: Type[BaseModel],
    *,
    fallback_llm=None,
) -> Dict[str, Any]:
    """Reserved: on failure, a second LLM structuring call can be made here (returns empty dict when fallback_llm is not configured)."""
    obj = extract(raw_text)
    if obj is not None:
        return obj
    if fallback_llm is None:
        return {}
    # Extension point: call fallback_llm here.
    return {}
