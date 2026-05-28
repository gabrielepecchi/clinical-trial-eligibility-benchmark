# Project Specification

## Goal

Build a local benchmark that evaluates how accurately a system can match synthetic patients to Parkinson disease clinical trials based on eligibility criteria from ClinicalTrials.gov.

## Scope

### In scope
- Download and parse public trial data from ClinicalTrials.gov (Parkinson disease trials only)
- Define schemas for trials, eligibility criteria, and synthetic patients
- Generate synthetic patient profiles (no real data)
- Define expected eligibility labels (`eligible` / `not_eligible` / `unclear`) per patient-trial pair
- Run and score matching logic against expected labels
- Produce benchmark metrics (accuracy, precision, recall, F1)

### Out of scope
- Real patient data
- Clinical validation or regulatory use
- Medical advice or clinical decision support
- Any UI (initial phases)
- External APIs beyond ClinicalTrials.gov public REST API
- Database or Docker setup (initial phases)

## Disease Focus

Parkinson disease — ICD-10 G20, search term `"Parkinson disease"` on ClinicalTrials.gov.

## Data

| Source | Type | Notes |
|---|---|---|
| ClinicalTrials.gov v2 API | Trial metadata + eligibility text | Public, no auth required |
| Synthetic patients | JSON files | Generated locally, no real patients |
| Ground-truth labels | JSON files | LLM-reviewed draft labels; not clinical gold truth |

### Current dataset (complete)

- Real Parkinson disease trials downloaded from ClinicalTrials.gov
- Synthetic patient cases stored in `data/processed/patient_cases.json`
- 150 labeled patient-trial pairs (LLM-reviewed draft labels)

### Data pipeline outputs

| File | Description |
|---|---|
| `data/raw/parkinson_trials_raw.json` | Raw trial JSON from ClinicalTrials.gov |
| `data/processed/eligibility_criteria.json` | Extracted eligibility criteria |
| `data/processed/trial_cases.json` | Selected trial cases |
| `data/processed/patient_cases.json` | Synthetic patient profiles |
| `data/processed/label_candidates.json` | Candidate labels |
| `data/processed/labels_seed.json` | Seed labels |
| `data/processed/labels_seed_review.csv` | Labels exported for review |
| `data/processed/labels_llm_reviewed.json` | LLM-reviewed draft benchmark labels |
| `data/processed/results_llm_reviewed.json` | Benchmark results |
| `data/processed/error_analysis_llm_reviewed.json` | Error analysis output |

> **Note:** `labels_llm_reviewed.json` contains LLM-reviewed draft benchmark labels. These are not clinical gold truth and have not been clinically validated.

## Label Values

| Label | Meaning |
|---|---|
| `eligible` | Patient meets all criteria for the trial |
| `not_eligible` | Patient fails one or more criteria |
| `unclear` | Insufficient information to decide |

## Matching Logic

Rule-based deterministic matcher implemented in `app/eligibility/rule_matcher.py`.

Covers:
- Age range inclusion/exclusion
- Parkinson disease diagnosis requirements
- DBS (deep brain stimulation) criteria
- Cognitive exclusion criteria
- Medication uncertainty (unclear regimen, missing pharmacy records)
- Disease-stage and severity uncertainty (missing UPDRS, unclear Hoehn and Yahr stage, unknown duration)
- Atypical or unclear parkinsonism vs. idiopathic PD requirements
- Active cancer treatment in safety-sensitive trials
- Recent interventional trial participation and washout requirements
- Protocol-risk uncertainty (frailty, falls, pacemaker, orthostatic hypotension)

## Current Benchmark Results (LLM-reviewed labels)

- Evaluated pairs: 150
- Accuracy: 0.440
- Macro F1: 0.439
- Eligible F1: 0.484
- Not-eligible F1: 0.436
- Unclear F1: 0.397

## Tests

267 tests passing.

## Metrics

- Per-criterion accuracy
- Per-trial eligibility accuracy
- Overall benchmark score (accuracy, precision, recall, F1)

## Limitations

- No medical advice. No clinical decision support.
- No real patient data. No clinical validation.
- Synthetic patients do not represent any real individuals.
- Benchmark labels are LLM-reviewed drafts and do not constitute clinical ground truth.
- Benchmark scores reflect system performance only, not clinical accuracy.
