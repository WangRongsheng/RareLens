"""
Prompt templates for treatment plan generation.
"""

TREATMENT_PLAN_PROMPT = """
You are an expert medical consultant analyzing a hypothetical patient scenario. 
Your task is to provide comprehensive, evidence-based treatment recommendations, predict treatment response.

Instructions
1. Treatment Goals
Establish comprehensive treatment goals across different timeframes. Include short-term goals (immediate priorities within days/weeks), medium-term goals (priorities within months), and long-term goals (priorities for 1+ years). For each timeframe, specify the objective type, which may include death, cure, symptom relief, functional improvement, disease stabilization, or others.

2. Treatment Recommendations
List all necessary treatments in descending order of importance score (10→0). For each treatment, include:
2.1 Treatment type: Select from surgery, medication, radiation therapy, chemotherapy, immunotherapy, targeted therapy, hormone therapy, physical/occupational therapy, psychological intervention, nutritional support, palliative care, preventive measures, monitoring/follow-up, lifestyle modifications, or alternative/complementary therapy.
2.2 Specific treatment: Provide the precise name of drug/procedure/intervention.
2.3 Dosage or details: For medications, include dosage, route, frequency, and duration. For procedures, detail the technique, approach, and extent. For therapies, specify intensity, schedule, and duration.
2.4 Treatment rationale: Provide evidence-based justification, including clinical guideline references, research evidence supporting efficacy, and mechanism of action addressing the patient's condition.
2.5 Importance score: Assign an integer from 0-10, where 10 is life-saving and absolutely essential core treatment, 7-9 is highly important for outcome, 4-6 is moderately important, 1-3 is adjunctive/supportive, and 0 is optional with minimal impact.
2.6 Anticipated treatment response: Specify expected primary effect (e.g., pathogen elimination, tumor reduction), symptom improvement, timeline for response, and response measurement method.
2.7Safety considerations: Include potential adverse effects (common and severe), contraindications



Guidelines for Response Quality
1.Priority: Rank treatment recommendations based on the importance score of the treatment, with the highest-scoring, most important treatment options listed first.
2.Completeness: List all necessary treatment recommendations.
3.Evidence-based: Base recommendations strictly on established clinical guidelines and peer-reviewed evidence.
4.Patient-centered: Consider the patient's specific characteristics, comorbidities, and risk factors.
5.Safety-focused: Explicitly address contraindications, interactions, and necessary precautions.
6.Practical: Consider availability, cost, and implementation challenges.
7. Very important: output no more than 10 treatments.
Output Format
Please provide your response in the following JSON format:

{{
  "treatment_goals": {{
    "short_term": "cure|symptom_relief|functional_improvement|disease_stabilization|complication_prevention",
    "medium_term": "cure|symptom_relief|functional_improvement|disease_stabilization|complication_prevention",
    "long_term": "cure|symptom_relief|functional_improvement|disease_stabilization|complication_prevention"
  }},
  "treatment_recommendations": {{
    "treatment1": {{
      "treatment_type": "surgery|medication|radiation|chemotherapy|immunotherapy|targeted_therapy|hormone_therapy|physical_therapy|psychological_intervention|nutritional_support|palliative_care|preventive_measures|monitoring|lifestyle_modification|alternative_therapy",
      "specific_treatment": "string",
      "dosage_or_details": "string",
      "treatment_rationale": "string",
      "importance_score": "0-10",
      "anticipated_treatment_response": "string",
      "safety_considerations": "string"
    }},
    "treatment2": {{
      "treatment_type": "surgery|medication|radiation|chemotherapy|immunotherapy|targeted_therapy|hormone_therapy|physical_therapy|psychological_intervention|nutritional_support|palliative_care|preventive_measures|monitoring|lifestyle_modification|alternative_therapy",
      "specific_treatment": "string",
      "dosage_or_details": "string",
      "treatment_rationale": "string",
      "importance_score": "0-10",
      "anticipated_treatment_response": "string",
      "safety_considerations": "string"
    }},
    "...": "..."
  }},
}}


Here is the patient’s information:
{content}
""".strip()

