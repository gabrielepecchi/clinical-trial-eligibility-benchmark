# Implementation Plan

> **Research and benchmarking only. Not medical advice. Not clinical decision support.**

---

## Phase 1 — Core Schemas, Parser, and Matcher ✅

- [x] `app/models.py` — Pydantic models for `Trial`, `Patient`, `EligibilityLabel`
- [x] `app/eligibility/criteria_parser.py` — free-text eligibility criteria parser
- [x] `app/eligibility/rule_matcher.py` — deterministic rule-based matcher

---

## Phase 2 — ClinicalTrials.gov Data Ingestion ✅

- [x] `scripts/download_trials.py` — fetch raw Parkinson disease trials from ClinicalTrials.gov v2 API → `data/raw/parkinson_trials_raw.json`
- [x] `scripts/extract_eligibility.py` — extract eligibility criteria → `data/processed/eligibility_criteria.json`
- [x] `scripts/select_trial_cases.py` — select trial cases for benchmark → `data/processed/trial_cases.json`
- [x] `scripts/audit_trial_cases.py` — audit selected trial cases

---

## Phase 3 — Synthetic Patient and Label Workflow ✅

- [x] `data/processed/patient_cases.json` — synthetic patient profiles (no real patients)
- [x] `scripts/generate_label_candidates.py` → `data/processed/label_candidates.json`
- [x] `scripts/audit_label_candidates.py` — audit candidate labels
- [x] `scripts/generate_labels_seed.py` → `data/processed/labels_seed.json`
- [x] `scripts/audit_labels_seed.py` — audit seed labels
- [x] `scripts/export_labels_seed_review.py` → `data/processed/labels_seed_review.csv`
- [x] `scripts/import_reviewed_labels.py` — import reviewed labels back into pipeline

---

## Phase 4 — LLM-Reviewed Draft Benchmark ✅

- [x] `data/processed/labels_llm_reviewed.json` — LLM-reviewed draft benchmark labels
- [x] `eval/run_llm_reviewed_benchmark.py` → `data/processed/results_llm_reviewed.json`
- [x] `eval/summarize_llm_reviewed_errors.py` → `data/processed/error_analysis_llm_reviewed.json`

> **Note:** `labels_llm_reviewed.json` contains LLM-reviewed draft labels. These are not clinical gold truth and have not been clinically validated.

### Current benchmark results

- Evaluated pairs: 150
- Accuracy: 0.440
- Macro F1: 0.439
- Eligible F1: 0.484
- Not-eligible F1: 0.436
- Unclear F1: 0.397

---

## Phase 5 — Rule Matcher Improvements ✅

- [x] Age parsing fix — stage ranges like `1-3` are no longer misread as age ranges
- [x] Uncertainty logic for medication history (unclear regimen, missing pharmacy records)
- [x] Uncertainty logic for disease severity and stage (missing UPDRS, unclear Hoehn and Yahr, unknown duration)
- [x] Uncertainty logic for atypical or unclear parkinsonism vs. idiopathic PD requirements
- [x] Uncertainty logic for active cancer treatment in safety-sensitive trials
- [x] Uncertainty logic for recent interventional trial participation and washout requirements
- [x] Uncertainty logic for protocol risk: device/stimulation/imaging/rehab/cognitive/gait criteria

---

## Phase 6 — Testing ✅

- [x] Unit tests for schema validation
- [x] Unit tests for parser on example criteria strings
- [x] Unit tests for fetcher (mocked HTTP)
- [x] Data validation tests
- [x] Pipeline script tests
- [x] LLM-reviewed labels tests
- [x] Unclear matcher logic tests

**Current status: 267 tests passing.**

---

## Future Work (not yet done)

- [ ] Human expert review of benchmark labels to replace LLM-reviewed draft
- [ ] Stronger eligibility criteria parser (handle more free-text patterns)
- [ ] Improved matcher scoring (partial match, confidence weights)
- [ ] Expand synthetic patient set beyond current cases
- [ ] Optional: FastAPI or Streamlit interface (later phase)
- [ ] Optional: database or Docker setup (later phase)

---

## Guiding Principles

- No real patient data at any phase.
- No medical advice or clinical decision support at any phase.
- Keep the baseline simple and deterministic before adding complexity.
