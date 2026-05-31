# Data Statement

> **Warning:** This benchmark uses synthetic patient profiles and draft LLM-reviewed labels.
> It is not clinical gold-standard data and is not for clinical use or medical decision-making.

---

## Dataset Purpose

This dataset was created to support development and evaluation of a rule-based clinical trial eligibility matcher. It provides structured patient–trial pairs with eligibility labels, enabling reproducible benchmarking of matching logic against a consistent set of cases.

The benchmark is a research and portfolio artifact. It is not intended for deployment in clinical workflows.

---

## Data Sources

The benchmark combines three types of data:

1. Synthetic patient profiles generated for this project
2. Trial eligibility criteria derived from public ClinicalTrials.gov records
3. Draft eligibility labels produced through an LLM-assisted review process

---

## Synthetic Patient Cases

All patient profiles are fully synthetic. They were generated to cover a range of clinically relevant dimensions for Parkinson's disease trial eligibility, including age, diagnosis subtype, disease duration, motor severity, cognitive status, medication history, device history, and comorbidities.

No real patient records, protected health information (PHI), or clinical databases were used. No IRB approval was sought or required.

Patient profiles are stored in `data/processed/patient_cases.json`.

---

## Trial Criteria

Trial eligibility criteria were derived from publicly available records on ClinicalTrials.gov. Trials were selected based on relevance to Parkinson's disease and the presence of structured inclusion and exclusion criteria in English.

Criteria text may have been cleaned or reformatted for consistency. No proprietary or restricted data sources were used.

Trial cases are stored in `data/processed/trial_cases.json`.

---

## Labeling Process

Eligibility labels (`eligible`, `not_eligible`, `unclear`) were produced through an LLM-assisted review process and stored in `data/processed/labels_llm_reviewed.json`.

Labels represent a best-effort assessment of each patient–trial pair given the available information. They have not been validated by clinicians, reviewed against clinical gold standards, or audited for completeness.

Labels should be treated as draft research labels. Known issues include inconsistencies in `unclear` label assignment and cases where protocol ambiguity makes the correct label genuinely uncertain.

---

## Known Limitations

- Patient profiles are synthetic and may not reflect the full complexity of real clinical presentations.
- Trial criteria summaries may omit protocol details that would affect real eligibility decisions.
- Labels are draft and have not undergone expert clinical review.
- The `unclear` class is particularly difficult to label consistently; unclear recall is low in v0.1 (0.545).
- The benchmark covers Parkinson's disease trials only and does not generalise to other therapeutic areas.
- No temporal reasoning over real clinical dates is currently implemented.

---

## Intended Use

- Benchmarking and iterative development of the rule-based eligibility matcher
- Evaluating matcher performance across label classes and criterion types
- Research into clinical NLP, eligibility screening logic, and uncertainty handling
- Portfolio demonstration of benchmark design and evaluation methodology

---

## Non-Intended Use

- Clinical decision support or patient triage
- Medical advice or treatment recommendations
- Regulatory or compliance use
- Any application involving real patients or real eligibility decisions
- Training production machine learning models intended for clinical deployment
