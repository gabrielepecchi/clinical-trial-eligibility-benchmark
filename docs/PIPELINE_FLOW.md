# Pipeline Flow

This document describes the local benchmark pipeline for evaluating AI-assisted clinical trial eligibility reasoning. All execution is local, patients are synthetic, and trial data is sourced from public records. The pipeline is designed for reproducibility.

---

## Stage 1 — Fetch Trial Criteria

| | |
|---|---|
| **Input** | NCT IDs or a search query against ClinicalTrials.gov |
| **Output** | Structured inclusion/exclusion criteria per trial |
| **Purpose** | Obtain real, publicly available eligibility criteria as the benchmark's source reference text |

---

## Stage 2 — Generate Synthetic Patient Profiles

| | |
|---|---|
| **Input** | Configurable parameters (age range, condition categories, demographics) |
| **Output** | A set of synthetic patient records (demographics, diagnoses, medications, lab values) |
| **Purpose** | Produce realistic but entirely fictional patients to avoid any use of real or identifiable health data |

---

## Stage 3 — Create Patient–Trial Pairs

| | |
|---|---|
| **Input** | Synthetic patient profiles + fetched trial criteria |
| **Output** | A list of (patient, trial) pairs to be evaluated |
| **Purpose** | Define the evaluation scope — each pair is one eligibility decision the pipeline must reason about |

---

## Stage 4 — Produce Draft Eligibility Labels

| | |
|---|---|
| **Input** | Patient–trial pairs |
| **Output** | Draft Eligible / Ineligible / Uncertain labels with criterion-level annotations |
| **Purpose** | Create a reference label set for evaluating the reasoning pipeline's predictions against |

---

## Stage 5 — Run the Eligibility Reasoning Pipeline

| | |
|---|---|
| **Input** | Patient–trial pairs (same set as Stage 3) |
| **Output** | Predicted Eligible / Ineligible / Uncertain decisions with reasoning traces |
| **Purpose** | Apply the AI reasoning pipeline to each pair and record its decision and supporting rationale |

---

## Stage 6 — Evaluate Predictions

| | |
|---|---|
| **Input** | Pipeline predictions (Stage 5) + draft labels (Stage 4) |
| **Output** | Accuracy and macro F1 scores across the benchmark set |
| **Purpose** | Quantify how well the pipeline's decisions align with the reference labels |

---

## Stage 7 — Analyze Errors

| | |
|---|---|
| **Input** | Mispredicted pairs + reasoning traces + `ERROR_TAXONOMY.md` |
| **Output** | Error counts and patterns grouped by taxonomy category |
| **Purpose** | Identify systematic failure modes (e.g. misread negations, missing lab comparisons) to guide targeted improvements |

---

## Notes

- **No clinical validation** — this pipeline benchmarks AI reasoning on synthetic data only.
- **Fully local** — no data leaves the local environment during evaluation.
- **Reproducible** — fixed synthetic generation parameters and public trial data allow exact benchmark reruns.
