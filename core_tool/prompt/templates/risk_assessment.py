"""Shared prompt template for rare disease risk assessment."""

from __future__ import annotations

import json
from typing import Any, Dict

from schema import RiskInput


RISK_ASSESSMENT_PROMPT = """
You are an expert clinician.
Your task is to 
1. Read the patient’s information; 
2. Analyze if the patient may have a rare disease.
3. Output five most important key insights (signs and symptoms) that contribute to the risk of rare disease and assign weights to the key insights, where the weights should add up to 1 (for example, symptom xx weight 0.3, sign xxx weight 0.4).
4. Assign score for the risk that the patient may have a rare disease. The score is 0-100, where 0 indicates no risk and 100 indicates certainty that the patient has a rare disease.
5. Output the top five most possible rare disease diagnoses and explanation for each one.
6. Output explanation for your assessment of risk score.

Important alignment with the schema below:
- Do NOT add a separate "top five diagnoses" section outside this JSON. Any differential reasoning belongs inside risk_explanation as plain text only.

STRICT OUTPUT RULES (violations break downstream parsing):

You MUST respond with a single valid JSON object.
The top-level keys must be exactly: "key_insights" (array of 5 objects), "risk_score" (integer), "risk_explanation" (string).
Each object in key_insights must have keys: "insight<N>", "weight", "description" where <N> is 1 for the first object, 2 for the second, …, 5 for the fifth (never a generic "insight" key without a number).
Do NOT output any text outside the JSON object.

- Respond with that single JSON object only. No prose, markdown, or labels before or after it.
- Do not use markdown (no **headings**, no bullet lists outside JSON), no code fences (```), and no labels such as "Final Output" or "Answer:".
- key_insights MUST be a JSON array of exactly five objects. Each element MUST be a JSON object (never a bare string). Each object MUST include "weight" (JSON number) and "description" (string).
- You MUST provide exactly five key_insights entries (the array length must be 5). If fewer than five clinically distinct points exist, expand, split, or refine the most important findings so that all five slots are filled with substantive content. Never omit an entry or leave any slot empty.
- Insight keys (critical): The ONLY allowed insight field names are the strings "insight1", "insight2", "insight3", "insight4", and "insight5". Do NOT use a generic key named "insight" (without a number). The first array object MUST contain ONLY "insight1" (plus weight and description); the second object MUST contain ONLY "insight2"; … the fifth MUST contain ONLY "insight5". Never put "insight1" in more than one object.
- "risk_explanation" MUST be a single plain-language string. Do NOT embed a second JSON object or escaped JSON inside it.
- risk_score MUST appear once at the top level as a JSON integer (not quoted).
- Ensure the entire response is valid JSON: double quotes for keys/strings, commas between array elements, no trailing commas, no comments.

Output in the following json format:
{{
  "key_insights": [
    {{ "insight1": "string", "weight": "float", "description": "string" }},
    {{ "insight2": "string", "weight": "float", "description": "string" }},
    {{ "insight3": "string", "weight": "float", "description": "string" }},
    {{ "insight4": "string", "weight": "float", "description": "string" }},
    {{ "insight5": "string", "weight": "float", "description": "string" }}
  ],
  "risk_score": "integer",
  "risk_explanation": "string"
}}
Here is the patient’s information:

{content}
"""


def build_risk_assessment_prompt(risk_input: RiskInput | Dict[str, Any]) -> str:
    """Build prompt text from RiskInput (or a compatible dict)."""
    if isinstance(risk_input, RiskInput):
        payload = risk_input.model_dump()
    else:
        payload = RiskInput.model_validate(risk_input).model_dump()
    content = json.dumps(payload, ensure_ascii=False, indent=2)
    # The MIMIC dataset uses ___ to represent PHI-redacted fields; the model returns empty output when it encounters them.
    # Replace with [REDACTED] so the model can process anonymised records normally.
    content = content.replace("___", "[REDACTED]")
    return RISK_ASSESSMENT_PROMPT.format(content=content)
