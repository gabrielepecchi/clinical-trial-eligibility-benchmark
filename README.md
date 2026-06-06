# clinical-trial-eligibility-benchmark

A local benchmark for evaluating patient-to-trial eligibility matching using synthetic patients and public Parkinson disease clinical trial data from ClinicalTrials.gov.

---

## Project Overview

This project builds and evaluates a full AI reasoning pipeline for clinical trial eligibility matching. It fetches real eligibility criteria from the public ClinicalTrials.gov API, generates synthetic patient profiles locally, and runs a structured reasoning pipeline to classify each patient as **Eligible**, **Ineligible**, or **Uncertain** against trial criteria. Results are evaluated with accuracy and macro F1, and failures are analyzed against a structured error taxonomy.

All execution is local. All patients are fully synthetic. Trial data is public. Labels are LLM-reviewed draft benchmark labels — not clinical gold truth.

## What It Demonstrates

- **Benchmark design** — end-to-end pipeline from raw public data to scored evaluation results, with reproducible local execution
- **AI evaluation** — structured measurement of model reasoning quality using accuracy, macro F1, and per-class scores across 150 labeled pairs
- **Eligibility reasoning** — multi-criterion classification with explicit handling of inclusion/exclusion logic and the Uncertain label
- **Error analysis** — taxonomy-driven investigation of failure modes including negation errors, threshold misreads, temporal gaps, and hallucinated criteria
- **Label quality control** — consistent labeling rules, reviewer guidance, and principled Uncertain assignment for ambiguous or data-incomplete cases
- **Synthetic data discipline** — controlled patient generation that isolates reasoning quality from data noise, with no real patient data at any stage

## Why This Matters

Clinical trial eligibility criteria are written in complex natural language with nested logic, numerical thresholds, and implicit domain knowledge. This makes eligibility matching a genuinely hard structured reasoning task — and a strong testbed for evaluating where and why AI pipelines fail. This project is a diagnostic tool for understanding model behavior, not a leaderboard entry.

## Common Failure Modes

- **Numeric threshold errors** — wrong direction, wrong unit, or off-by-one at an exact boundary value
- **Negation / exclusion clause errors** — misreading "no history of" or "must not have" as a positive match
- **Temporal constraint errors** — ignoring washout periods, recency requirements, or minimum diagnosis duration
- **Underspecified comorbidity or medication history** — asserting Eligible or Ineligible when a required value is absent from the profile
- **Multi-step criteria interaction errors** — failing when correct eligibility requires chaining two or more reasoning steps rather than a direct attribute lookup

---

## Benchmark Documentation

| File | Purpose |
|---|---|
| [`BENCHMARK_CARD.md`](docs/BENCHMARK_CARD.md) | Task definition, dataset, metrics, known failure modes, and current results |
| [`ERROR_TAXONOMY.md`](docs/ERROR_TAXONOMY.md) | Categorized breakdown of reasoning failure modes with examples |
| [`CURATED_EXAMPLES.md`](docs/CURATED_EXAMPLES.md) | Hand-selected easy, ambiguous, and hard cases illustrating the reasoning challenge |
| [`BASELINES.md`](docs/BASELINES.md) | Majority-class, keyword, and simple LLM baselines for interpreting pipeline scores |
| [`PIPELINE_FLOW.md`](docs/PIPELINE_FLOW.md) | Stage-by-stage walkthrough of the full benchmark pipeline |
| [`LABELING_GUIDE.md`](docs/LABELING_GUIDE.md) | Reviewer guide for assigning and validating draft eligibility labels |
| [`PROJECT_SPEC.md`](docs/PROJECT_SPEC.md) | Project scope, data sources, labels, matching logic, metrics, and limitations |
| [`REPO_STRUCTURE.md`](docs/REPO_STRUCTURE.md) | Detailed repository tree and file organization |
| [`IMPLEMENTATION_PLAN.md`](docs/IMPLEMENTATION_PLAN.md) | Completed phases, current status, test count, and future work |

---

## Task

Given a synthetic patient profile and the eligibility criteria text of a clinical trial, predict whether the patient is **eligible**, **not eligible**, or **unclear**.

This is a structured clinical reasoning task. Eligibility criteria are written in complex natural language, often with nested logic, numerical thresholds, and implicit domain knowledge. Performance on this benchmark reflects how well a model handles that reasoning — not a clinical deployment.

## Data

- **Trials**: Public Parkinson disease trials from ClinicalTrials.gov. Eligibility criteria are extracted as structured text.
- **Patients**: Fully synthetic profiles generated locally. No real patient data is used.
- **Labels**: LLM-reviewed draft labels (`labels_llm_reviewed.json`). These are a best-effort benchmark baseline, not clinical gold truth.

### Label types

| Label | Meaning |
|---|---|
| `eligible` | Patient meets all inclusion criteria and no exclusion criteria |
| `not_eligible` | Patient fails at least one hard exclusion or inclusion criterion |
| `unclear` | Criteria are ambiguous or patient data is insufficient to decide |

### Example pairs

**Example 1 — not eligible**
Patient: 45-year-old with Parkinson disease, currently taking a MAO-B inhibitor.
Trial criterion: "No concurrent MAO-B inhibitor use."
Label: `not_eligible` — explicit exclusion criterion matched.

**Example 2 — eligible**
Patient: 62-year-old with idiopathic Parkinson disease, UPDRS III score 28, no deep brain stimulation history.
Trial criterion: "Diagnosis of idiopathic PD, UPDRS III 20–40, no prior DBS."
Label: `eligible` — all stated criteria satisfied.

**Example 3 — unclear**
Patient: 58-year-old with Parkinson disease, history of "cardiac arrhythmia" (type unspecified).
Trial criterion: "No clinically significant cardiac disease."
Label: `unclear` — arrhythmia type is unspecified; eligibility cannot be determined from the available data.

## Current benchmark results (LLM-reviewed labels)

This is a local draft benchmark using synthetic patients, ClinicalTrials.gov trial criteria, and LLM-reviewed draft labels. Results are not clinical gold truth.

- Evaluated pairs: 150
- Accuracy: 0.687
- Macro F1: 0.686
- Eligible F1: 0.659
- Not-eligible F1: 0.703
- Unclear F1: 0.696

**Error Pairs (gold → predicted)**
- not_eligible → eligible: 2
- eligible → not_eligible: 4
- unclear → eligible: 18
- unclear → not_eligible: 12
- eligible → unclear: 7
- not_eligible → unclear: 4
- Total errors: 47

**Robustness checks**
- Minimal pairs: 8/8 passed
- Counterfactual pairs: 8/8 passed

These results reflect the genuine difficulty of eligibility reasoning from free-text criteria. Criteria are often verbose, ambiguous, or require multi-step inference.

## Example Benchmark Case

Below is an illustrative synthetic benchmark case showing the structure of patient, trial criteria, expected label, and matcher reasoning.

**Synthetic Patient Profile:**
```json
{
  "patient_id": "P_demo_031",
  "age": 58,
  "sex": "male",
  "diagnosis": "Parkinson disease",
  "diagnosis_duration_years": 3,
  "hoehn_yahr_stage": null,
  "updrs_iii_score": null,
  "medications": ["levodopa/carbidopa 250mg daily"],
  "cardiac_history": "arrhythmia (type unspecified)",
  "dbs_history": false,
  "cognitive_status": "not documented",
  "recent_trial_participation": false
}
```

**Trial Eligibility Criteria (excerpt):**
```
Inclusion:
  - Diagnosis of idiopathic Parkinson disease
  - Hoehn-Yahr stage 2–4
  - Stable medication regimen for at least 4 weeks

Exclusion:
  - Clinically significant cardiac disease
  - Montreal Cognitive Assessment (MoCA) score < 24
```

**Expected Label (Gold):** `unclear`  
**Matcher Prediction:** `unclear`  
**Correct:** Yes

**Criterion-Level Reasoning (illustrative):**
- ? Hoehn-Yahr stage — not documented in profile → cannot confirm inclusion criterion
- ? Cardiac exclusion — arrhythmia type is unspecified; whether it constitutes "clinically significant cardiac disease" cannot be determined
- ? MoCA — cognitive status not documented; cannot confirm exclusion does not apply
- ✓ Diagnosis — Parkinson disease noted, though idiopathic subtype not confirmed
- ✓ DBS history — false

**Analysis:** This case illustrates the dominant remaining challenge: when key fields are absent or ambiguous, the correct label is `unclear`, and the matcher correctly defers rather than over-committing. The largest error category in the current benchmark is `unclear` cases predicted as `eligible` (18 of 47 total errors), driven by incomplete patient profiles where the matcher lacks sufficient signal to block eligibility.

---

## How to Read the Results

- **Accuracy and macro F1 of ~0.69** reflects the matcher's current performance across all three labels; the majority-class baseline scores only 0.51 accuracy / 0.23 macro F1, so the gap is meaningful
- **Unclear cases matter** — the Uncertain label exists because missing or ambiguous information should not be guessed; a model that over-commits on unclear cases is failing in a meaningful way
- **Robustness is verified** — minimal pairs (8/8) and counterfactual pairs (8/8) all pass, confirming that single-variable changes to patient profiles produce the expected prediction changes
- **Future improvements should focus on** parser coverage for complex free-text criteria, uncertainty handling for incomplete patient profiles, and systematic reduction of the failure modes identified in the error taxonomy
- **Results are draft** — labels are LLM-reviewed and not clinical gold truth; this benchmark is a diagnostic tool for model reasoning, not a clinical deployment evaluation

## Key takeaway

This project demonstrates end-to-end local benchmark construction: ClinicalTrials.gov data download and eligibility extraction, synthetic patient generation, draft label creation with LLM review, benchmark evaluation, and structured error analysis — without any real patient data.

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

Local tests currently pass with 579 tests.

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
| Benchmark predictions CSV | `data/processed/results_llm_reviewed.csv` |
| Criterion-level results CSV | `data/processed/criterion_level_results.csv` |
| Error analysis | `data/processed/error_analysis_llm_reviewed.json` |
| Error analysis CSV | `data/processed/error_analysis_llm_reviewed.csv` |
| Sample benchmark predictions CSV | `data/processed/results_sample_predictions.csv` |
| Hard-case subsets (tagged records) | `data/processed/hard_case_subsets.json` |
| Hard-case subsets CSV | `data/processed/hard_case_subsets.csv` |
| Hard-case per-tag metrics | `data/processed/hard_case_metrics.json` |
| Hard-case per-tag metrics CSV | `data/processed/hard_case_metrics.csv` |
| HTML benchmark report | `reports/benchmark_report.html` |

## Local demo (offline, fast)

Run the full local demo pipeline without a network connection (reuses an existing raw trials file):

```
PYTHONPATH=. python scripts/run_local_demo.py
```

Optional variants:

```
# Run the full sample benchmark instead of the quick subset
PYTHONPATH=. python scripts/run_local_demo.py --full-sample

# Download fresh trial data (requires network) and run the quick demo
PYTHONPATH=. python scripts/run_local_demo.py --online --max-trials 20
```

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

### Quick demo

Run a small deterministic subset (first 3 pairs) to verify the pipeline works without waiting for the full benchmark:

```
PYTHONPATH=. python eval/run_sample_benchmark.py --quick-demo
```

To control the subset size:

```
PYTHONPATH=. python eval/run_sample_benchmark.py --quick-demo --limit 5
PYTHONPATH=. python eval/run_sample_benchmark.py --limit 10
```

## Hard-case subset tagging

Tags each benchmark record with one or more difficulty labels — `hard_negative`, `hard_positive`, `ambiguous_clinical_severity` — using deterministic text and metadata rules. No model inference required.

```
PYTHONPATH=. python eval/tag_hard_cases.py
```

Generated files:

- `data/processed/hard_case_subsets.json` — full tagged records with summary counts and per-tag metrics
- `data/processed/hard_case_subsets.csv` — flat CSV for inspection
- `data/processed/hard_case_metrics.json` — classification metrics (accuracy, macro F1, per-class scores) computed separately for each hard-case tag
- `data/processed/hard_case_metrics.csv` — flat CSV version of per-tag metrics

## HTML benchmark report

Generates a self-contained local HTML report from all benchmark result files:

```
PYTHONPATH=. python eval/generate_benchmark_report.py
```

The report is written to `reports/benchmark_report.html` and includes global metrics, confusion matrix, error analysis, criterion-type summary, and — when the hard-case files are present — hard-case subset summaries and per-tag classification metrics.

## Data Sources

- [ClinicalTrials.gov](https://clinicaltrials.gov) — public data only
- Synthetic patient profiles — generated locally, no real patients
