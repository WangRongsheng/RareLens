# Demo Dataset

500 cases for small-scale evaluation and reproducibility checks.

## Directory Structure

Each case directory (`<case_id>/`) contains model input files and ground-truth labels.

### Model Input Files

| File | Used by |
| --- | --- |
| `primary_consultation.json` | Diagnosis (primary) |
| `follow_up_consultation.json` | Diagnosis (follow-up) |
| `risk_input.json` | Alert |
| `treatment_plan.json` | Treatment |
| `prognosis_prediction.json` | Prognosis |

### Ground-Truth Files (for evaluation)

| File | Used by |
| --- | --- |
| `diagnosis.json` | Diagnosis |
| `rare_or_not_final.json` | Alert |
| `treatment_outcome.json` | Treatment |
| `prognosis_new.json` | Prognosis |

See [`schema/`](../schema/) for data schema definitions.
