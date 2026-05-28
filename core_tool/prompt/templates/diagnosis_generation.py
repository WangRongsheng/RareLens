"""
Prompt templates for diagnosis candidate generation.
"""

DIAGNOSIS_PRIMARY_PROMPT = """
You are an expert clinician.

Your task is to:

1. Read the patient's information carefully.

2. Formulate your own diagnostic hypotheses through robust medical reasoning.

3. Determine the 5 most likely diagnoses:
   3.1 Follow your own reasoning path to identify the five most likely diagnoses
   3.2 Provide brief diagnostic reasoning for each diagnosis
   3.3 Assign a confidence score for each diagnosis (0-10 scale, where 0 = no confidence and 10 = absolute confidence)
   3.4 Sort the diagnoses from highest to lowest confidence score

4. Determine the top 5 crucial diagnostic tests required to reach a final diagnosis:
   4.1 Recommend only tests that are crucial and confirmatory for reaching a final diagnosis
   4.2 Provide brief rationale for why this test is essential
   4.2 Assign a necessity score (0-10) for each diagnostic test based on its importance for confirming the diagnosis(0-10 scale, where 0 = not necessary at all and 10 = absolutely necessary)


5.Output your analysis in the following JSON format:

Output in the following json format:
{{
  "most_likely_diagnosis": {{
    "diagnosis1": {{
      "diagnosis_name": "string",
      "confidence_score": integer,
      "diagnostic_reasoning": "string"
    }},
    "diagnosis2": {{
      "diagnosis_name": "string",
      "confidence_score": integer,
      "diagnostic_reasoning": "string"
    }},
    "diagnosis3": {{
      "diagnosis_name": "string",
      "confidence_score": integer,
      "diagnostic_reasoning": "string"
    }},
    "diagnosis4": {{
      "diagnosis_name": "string",
      "confidence_score": integer,
      "diagnostic_reasoning": "string"
    }},
    "diagnosis5": {{
      "diagnosis_name": "string",
      "confidence_score": integer,
      "diagnostic_reasoning": "string"
    }}
  }},
  "further_diagnostic_test": {{
    "test1": {{
      "test_name": "string",
      "necessity_score": integer,
      "rationale": "string"
    }},
    "test2": {{
      "test_name": "string",
      "necessity_score": integer,
      "rationale": "string"
    }},
    "test3": {{
      "test_name": "string",
      "necessity_score": integer,
      "rationale": "string"
    }},
    "test4": {{
      "test_name": "string",
      "necessity_score": integer,
      "rationale": "string"
    }},
    "test5": {{
      "test_name": "string",
      "necessity_score": integer,
      "rationale": "string"
    }}
  }}
}}

Here is the patient’s information:

{content}
""".strip()


DIAGNOSIS_FOLLOWUP_PROMPT = """
You are an expert clinician.

Your task is to:

1. Read the patient's information carefully.

2. Formulate your own diagnostic hypotheses through robust medical reasoning.

3. Determine the 5 most likely diagnoses:
   3.1 Follow your own reasoning path to identify the five most likely diagnoses
   3.2 Provide brief diagnostic reasoning for each diagnosis
   3.3 Assign a confidence score for each diagnosis (0-10 scale, where 0 = no confidence and 10 = absolute confidence)
   3.4 Sort the diagnoses from highest to lowest confidence score

4.Output your analysis in the following JSON format:

Output in the following json format:
{{
  "most_likely_diagnosis": {{
    "diagnosis1": {{
      "diagnosis_name": "string",
      "confidence_score": integer,
      "diagnostic_reasoning": "string"
    }},
    "diagnosis2": {{
      "diagnosis_name": "string",
      "confidence_score": integer,
      "diagnostic_reasoning": "string"
    }},
    "diagnosis3": {{
      "diagnosis_name": "string",
      "confidence_score": integer,
      "diagnostic_reasoning": "string"
    }},
    "diagnosis4": {{
      "diagnosis_name": "string",
      "confidence_score": integer,
      "diagnostic_reasoning": "string"
    }},
    "diagnosis5": {{
      "diagnosis_name": "string",
      "confidence_score": integer,
      "diagnostic_reasoning": "string"
    }}
  }}
}}


Here is the patient’s information:
{content}
""".strip()
