# Benchmark Card — Clinical Trial Eligibility Reasoning

## Purpose

Evaluate whether an LLM pipeline can correctly match synthetic patients to clinical trials based on structured eligibility criteria, and reliably explain why a patient qualifies or does not qualify.

---

## Task Definition

Given a synthetic patient profile and a set of trial eligibility criteria, the model must:

1. Classify the patient as **Eligible**, **Ineligible**, or **Uncertain**
2. Provide a **criterion-level rationale** linking specific patient attributes to specific inclusion/exclusion criteria

---

## Data Sources

| Source | Role |
|---|---|
| ClinicalTrials.gov (public API) | Real trial eligibility criteria text |
| Synthetically generated patient profiles | Patient attribute inputs |
| LLM-assisted label review | Draft benchmark labels |

All trial data is sourced from the public ClinicalTrials.gov registry under open access. No real patient data is used at any stage.

---

## Synthetic Patient Policy

Patients are fully synthetic. Profiles are generated programmatically to cover:

- Edge cases near eligibility boundaries (e.g. age = cutoff ± 1 year)
- Conflicting criteria combinations
- Missing or ambiguous attribute scenarios

No real patient records, EHRs, or clinical databases are used. The pipeline runs entirely locally.

---

## Label Set

| Label | Meaning |
|---|---|
| `Eligible` | Patient meets all inclusion criteria and no exclusion criteria |
| `Ineligible` | Patient fails at least one hard exclusion or misses a required inclusion |
| `Uncertain` | Insufficient information to make a definitive determination |

---

## Label Quality Caveat

> **Labels are LLM-reviewed draft benchmark labels, not clinical gold standard truth.**
>
> They have not been validated by clinicians or medical professionals. This benchmark is intended to evaluate AI reasoning quality and pipeline behavior, not to support any clinical decision-making.

---

## Evaluation Metrics

| Metric | What It Measures |
|---|---|
| **Classification accuracy** | Correct Eligible / Ineligible / Uncertain calls |
| **Criterion-level recall** | Whether the model identifies all relevant criteria |
| **Rationale precision** | Whether cited criteria actually apply to the patient |
| **Uncertain rate** | How often the model appropriately hedges vs. over-commits |
| **Failure mode frequency** | Rate of known error patterns (see below) |

---

## Current Results

> Preliminary draft benchmark results. Numbers will update as the benchmark matures.

| Metric | Value |
|---|---|
| Patient-trial pairs evaluated | 150 |
| Classification accuracy | 0.440 |
| Macro F1 | 0.439 |

*All figures are computed on a synthetic held-out set with LLM-reviewed draft labels, not clinical gold standard truth.*

---

## Hard-Case Subset Analysis

Each benchmark record is tagged with one or more difficulty labels — `hard_negative`, `hard_positive`, `ambiguous_clinical_severity` — using deterministic text and metadata rules. Per-tag classification metrics (accuracy, macro F1, per-class scores) are computed separately for each subset and stored in:

- `data/processed/hard_case_subsets.json` — full tagged records with summary counts and per-tag metrics
- `data/processed/hard_case_metrics.json` — per-tag classification metrics

The HTML benchmark report (`reports/benchmark_report.html`) includes hard-case subset summaries and per-tag metrics when these files are present.

---

## Known Limitations

- Labels are LLM-generated drafts; no clinician review has occurred
- Synthetic patients may not reflect realistic co-morbidity distributions
- Eligibility criteria text from ClinicalTrials.gov is often ambiguous or inconsistently structured
- Benchmark size is currently small; results may not generalize
- No multi-model comparison has been run yet
- This remains a local draft benchmark; no clinical validation has been performed

---

## Known Failure Modes

| Failure Mode | Description |
|---|---|
| **Criteria hallucination** | Model cites a criterion not present in the trial text |
| **Boundary misclassification** | Model gets the wrong label when patient is at an exact cutoff value |
| **Exclusion omission** | Model correctly identifies inclusion match but misses a disqualifying exclusion |
| **Uncertain over-use** | Model defaults to Uncertain when a clear call is possible |
| **Criteria misattribution** | Model applies a criterion from a different trial section than intended |

---

## Why This Benchmark Matters for AI Evaluation

This benchmark was designed to test skills that are directly relevant to applied AI evaluation work:

- **Structured reasoning evaluation** — not just output accuracy, but criterion-level traceability
- **Error taxonomy** — explicit failure modes rather than a single aggregate score
- **Realistic ambiguity** — eligibility text is messy, which surfaces real model weaknesses
- **Synthetic data discipline** — controlled inputs that isolate the reasoning task from data noise
- **Local, reproducible pipeline** — no external API calls at inference time; results are auditable
- **Honest uncertainty handling** — the Uncertain label forces the model to signal when it cannot commit

The goal is not a leaderboard score. The goal is a diagnostic tool for understanding *where* and *why* a model fails on a structured medical reasoning task.

---

*Last updated: 2026-05 | Data: ClinicalTrials.gov public API | Patients: fully synthetic | Labels: LLM-reviewed draft, not clinical truth*
