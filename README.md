# clinical-trial-eligibility-benchmark

A local benchmark for evaluating patient-to-trial eligibility matching using synthetic patients and public Parkinson disease clinical trial data from ClinicalTrials.gov.

## Purpose

- Download and parse public clinical trial eligibility criteria from ClinicalTrials.gov
- Generate synthetic patient profiles
- Score eligibility matching logic against known expected outcomes
- Benchmark matching accuracy with reproducible test cases

## Limitations

> **This project is for research and benchmarking purposes only.**
>
> - No medical advice. No clinical decision support.
> - No real patient data is used or stored.
> - No clinical validation has been performed.
> - Results must not be used for any real-world clinical decisions.
> - `labels_llm_reviewed.json` is an LLM-reviewed draft benchmark, not clinical gold truth.

## Requirements

- Python 3.12
- See `requirements.txt`

## Status

Real trial dataset pipeline complete. LLM-reviewed benchmark complete.

Local tests currently pass with 267 tests.

## Data Pipeline

The full pipeline produces the following files:

| Step | Output file |
|---|---|
| Raw trial download | `data/raw/parkinson_trials_raw.json` |
| Eligibility extraction | `data/processed/eligibility_criteria.json` |
| Trial case selection | `data/processed/trial_cases.json` |
| Synthetic patients | `data/processed/patient_cases.json` |
| Candidate label generation | `data/processed/label_candidates.json` |
| Seed labels | `data/processed/labels_seed.json` |
| Review CSV | `data/processed/labels_seed_review.csv` |
| LLM-reviewed draft labels | `data/processed/labels_llm_reviewed.json` |
| Benchmark results | `data/processed/results_llm_reviewed.json` |
| Error analysis | `data/processed/error_analysis_llm_reviewed.json` |

## Current benchmark results (LLM-reviewed labels)

- Evaluated pairs: 150
- Accuracy: 0.440
- Macro F1: 0.439
- Eligible F1: 0.484
- Not-eligible F1: 0.436
- Unclear F1: 0.397

## Running the pipeline

```
# Download raw trials from ClinicalTrials.gov
python -m scripts.download_trials

# Extract eligibility criteria
python -m scripts.extract_eligibility

# Select trial cases
python -m scripts.select_trial_cases

# Generate candidate labels
python -m scripts.generate_label_candidates

# Generate seed labels
python -m scripts.generate_labels_seed

# Export seed labels for review
python -m scripts.export_labels_seed_review

# Run LLM-reviewed benchmark
python -m eval.run_llm_reviewed_benchmark

# Summarize errors
python -m eval.summarize_llm_reviewed_errors

# Run full test suite
python -m pytest -q
```

## Sample benchmark (legacy)

The sample benchmark uses 3 hand-authored trial cases, 5 synthetic patients, and 10 labeled pairs.

```
python -m eval.run_sample_benchmark
python -m eval.summarize_error_analysis
```

## Data Sources

- [ClinicalTrials.gov](https://clinicaltrials.gov) — public data only
- Synthetic patient profiles — generated locally, no real patients
