# RareAlert — Fine-tuning Data Template

SFT data format for the `rare_alert` risk-scoring model (LoRA SFT via
[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)).

Each line of the dataset is one independent training example. The dataset comes
in two interchangeable forms, differing only in how many reasoning hypotheses
are stacked inside the `assistant` turn:

| Form | Hypotheses per answer |
|------|-----------------------|
| single-hypothesis | 1  (`hypothesis_1`) |
| multi-hypothesis  | 10 (`hypothesis_1` … `hypothesis_10`) |

The `system` and `user` turns are identical between the two forms; only the
`assistant` turn changes.

---

## 1. Line format (LLaMA-Factory `messages` / OpenAI chat format)

Each line is a single JSON object with one key, `messages`, holding exactly
**three** turns in order: `system`, `user`, `assistant`.

```json
{
  "messages": [
    { "role": "system",    "content": "<SYSTEM_PROMPT>" },
    { "role": "user",      "content": "<PATIENT_INFO>" },
    { "role": "assistant", "content": "<ANSWER>" }
  ]
}
```

> Register this with LLaMA-Factory `dataset_info.json` using the `sharegpt`
> formatting with OpenAI-style roles, e.g.:
> ```json
> "rare_alert": {
>   "file_name": "<your_dataset>.jsonl",
>   "formatting": "sharegpt",
>   "columns": { "messages": "messages" },
>   "tags": {
>     "role_tag": "role", "content_tag": "content",
>     "user_tag": "user", "assistant_tag": "assistant", "system_tag": "system"
>   }
> }
> ```

---

## 2. `system` content

```
You are a helpful assistant specialized in detecting rare disease. Your task is to: 1.Analyze if the patient may have a rare disease. 2.Output five most important key insights (signs and symptoms) that contribute to the risk of rare disease and assign weights to the key insights, where the weights should add up to 1 (for example, symptom xx weight 0.3, sign xxx weight 0.4). 3.Assign score for the risk that the patient may have a rare disease. The score is 0-100, where 0 indicates no risk and 100 indicates certainty that the patient has a rare disease. 4.Output the top five most possible rare disease diagnoses and explanation for each one. 5.Output explanation for your assessment of risk score.
```

---

## 3. `user` content 

Template (`{...}` = fill in; omit a value with an empty string or "Unknown"):

```
Here is the patient's information Patient is a {age} year old {sex}. Weight: {weight_kg} kg. Height: {height_cm} cm. BMI: {bmi}. Ethnicity: {ethnicity}. Chief complaint: {chief_complaint}. History of present illness: {hpi}. Past medical history: {pmh}. Family history: {family_history}. General examination: {general_exam}. Specialty examination: {specialty_exam}.
```


---

## 4. `assistant` content

The answer is **free text** (not JSON). It starts with `Here is the answer: `
followed by one or more `hypothesis_{i}` blocks, and ends with a single
aggregated `Final RISK SCORE: {N}`.

### 4.1 One hypothesis block

```
hypothesis_{i}: 1.Identify key clinical presentations:
{insight_1}: {explanation_1}
{insight_2}: {explanation_2}
{insight_3}: {explanation_3}
{insight_4}: {explanation_4}
{insight_5}: {explanation_5}

2.Identify possible diagnosis
{disease_1}: {why_1}
{disease_2}: {why_2}
{disease_3}: {why_3}
{disease_4}: {why_4}
{disease_5}: {why_5}

3.Identify risk of rare disease: {free-text justification of the score}
4.Identify risk score: {score_0_100}
```

### 4.2 Full assistant string

- Prefix once: `Here is the answer: `
- Then block `hypothesis_1` … `hypothesis_K` (K = 1 or 10), separated by a blank
  line (`\n\n`).
- Append once at the very end: ` Final RISK SCORE: {final_score}`
  (the aggregated/ensemble score across all hypotheses).

```
Here is the answer: hypothesis_1: ...4.Identify risk score: {s1}

hypothesis_2: ...4.Identify risk score: {s2}

...

hypothesis_{K}: ...4.Identify risk score: {sK} Final RISK SCORE: {final_score}
```

### Field notes

| Field | Constraint |
|-------|------------|
| key insights | exactly **5** per hypothesis, format `name: explanation` |
| weights | the system prompt asks weights to sum to 1 (stated in prose) |
| diagnoses | exactly **5** per hypothesis, format `disease: explanation` |
| `risk score` (per hypothesis) | integer 0–100 |
| `Final RISK SCORE` | integer 0–100, appears **once**, at the end of the whole answer |

---

## 5. Minimal valid example (1 hypothesis)

See `data_template.jsonl` for a copy-paste-ready single-line example.
