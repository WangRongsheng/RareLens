"""
Prompt templates for prognosis LLM generation (per-model prognosis_prediction_output).

Moved verbatim from: backup/reproduce_module/prog/universal_prognosis.py.py (prompt_template, lines 292–352).
"""

PROGNOSIS_LLM_PROMPT = """
You are an expert medical consultant analyzing a hypothetical patient scenario. Your task is to predict the patient's prognosis based on the presented patient record.

Important Timeframe Context: The patient information provided represents the point of confirmed diagnosis and treatment planning, but before treatment implementation. Your predictions should be based on the expected outcomes after this point. For timing references: short-term (<3 months), mid-term (3 months to 1 year), and long-term (>1 year).

Instructions for Prognosis Analysis
1. Overall Outcome: Predict the patient’s long-term (>12 months) overall outcome
1.1 Outcome Category: Select one of the following: complete_recovery, partial_recovery, stabilization, progression, terminal. Here is information for each category: Complete Recovery: Complete resolution of the condition with return to pre-illness baseline functioning, where the patient regains full health with no residual symptoms or limitations. Partial Recovery: Significant improvement from the initial condition, but with some persistent symptoms, functional limitations, or sequelae that do not fully resolve despite appropriate treatment. Stabilization: A plateau state where the condition neither significantly improves nor deteriorates, with symptoms and functional status remaining relatively constant under current management strategies. Progression: Ongoing deterioration of the condition despite therapeutic interventions, characterized by worsening symptoms, increasing functional limitations, or advancing disease burden over time. Terminal: An irreversible condition that has advanced to a stage where death is expected within a relatively short timeframe (typically 6 months or less). 
1.2Confidence Score: Rate your prediction confidence from 0-10 (10 = completely confident, 0 = not at all confident)
1.3Explanation: Provide a concise medical rationale for your prediction based on provided information

2. Functional Status: Predict the long-term (>12months) functional status of the patient
2.1 Functional status should reflect the patient’s ability to perform daily or age-appropriate activities. Categorize the patient’s long term functional status into the following categories: none: no functional limitation; mild: minor limitation, activities largely preserved; moderate: clear limitation affecting normal activities; severe: major impairment or dependence
2.2Confidence Score: Rate your prediction confidence from 0-10 (10 = completely confident, 0 = not at all confident)
2.3Explanation: Provide a concise medical rationale for your prediction based on provided information

3. Symptom Burden: Predict the long-term (>12months) functional status of the patient
3.1 Symptom burden: Symptom burden should reflect persistence and impact of symptoms. Categorize the patient’s long term symptom burden into the following categories: none | occasional | persistent_mild | persistent_severe|.
3.2Confidence Score: Rate your prediction confidence from 0-10 (10 = completely confident, 0 = not at all confident)
3.3Explanation: Provide a concise medical rationale for your prediction based on provided information

4.Predict key clinical events with its occurring time period patient may encounter during follow-up
4.1 List all clinical events and categorize them into the following event types: symptom_improvement | functional_improvement | deterioration | readmission | re_intervention | major_complication | death
4.2 For each clinical event, predict the time point and categorize the time point of the event into the following time period short_term | mid_term | long_term | unknown
4.3 For each clinical event, rate your prediction confidence from 0-10 (10 = completely confident, 0 = not at all confident) 
4.4 For each clinical event, provide a concise medical rationale for your prediction based on provided information

Guideline:
1.Base your analysis only on the provided patient information - avoid assumptions beyond what is stated. 
2.Consider age, comorbidity, and treatment adherence in your assessment.

Output in the following json format:
{{
  "overall_outcome": {{
    "outcome_category": "complete_recovery | partial_recovery | stabilization | progression | terminal",
    "confidence_score": integer,
    "explanation": "string"
  }},
  "functional_status": {{
    "status": "none | mild | moderate | severe",
    "confidence_score": integer,,
    "explanation": ""
  }},
  "symptom_burden": {{
    "burden": "none | occasional | persistent_mild | persistent_severe",
    "confidence_score": integer,,
    "explanation": "string"
  }}, 
  "clinical_events": [
    {{
      "event_type": "symptom_improvement | functional_improvement | deterioration | readmission | re_intervention | major_complication | death",
      "time_period": "short_term | mid_term | long_term | unknown",
      "confidence_score": integer,,
      "explanation": "string"
    }}
  ]
}}

Here is the patient's information:
{content}
"""
