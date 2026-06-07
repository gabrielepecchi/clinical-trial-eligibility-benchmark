# Clinical Trial Eligibility Benchmark

A local benchmark evaluating AI reasoning quality on a structured classification task: predicting patient eligibility for clinical trials from free-text eligibility criteria.

The pipeline covers data acquisition, synthetic patient generation, label construction, model evaluation, and error analysis — end to end, fully reproducible locally.

---

## What This Project Demonstrates

- **Structured benchmark design** — end-to-end pipeline from raw public data to scored evaluation results, with consistent labeling rules, reproducible execution, and no real patient data at any stage
- **Rigorous evaluation methodology** — three-class classification (Eligible / Not Eligible / Unclear) scored with accuracy and macro F1; per-class F1 reported separately to surface class-specific failure patterns
- **Systematic error analysis** — failures categorized by a structured taxonomy (negation errors, threshold misreads, temporal constraint errors, hallucinated criteria, underspecified data) rather than treated as undifferentiated noise
- **Principled uncertainty handling** — explicit Uncertain label for ambiguous or data-incomplete cases; treating over-commitment on unclear cases as a meaningful failure mode, not a minor variant
- **Robustness verification** — minimal pairs and counterfactual pairs confirm that single-variable changes to patient profiles produce the expected prediction changes (8/8 passed on each)
- **Careful limitation framing** — labels are documented as LLM-reviewed draft labels, not clinical gold truth; results are presented as a diagnostic tool for model reasoning, not a deployment benchmark

---

## Problem

Clinical trial eligibility criteria are written in complex natural language — often with nested logic, numerical thresholds, washout periods, and implicit domain knowledge. Determining whether a patient satisfies all inclusion criteria and violates no exclusion criteria is a genuinely hard structured reasoning task that exposes failure modes (negation errors, threshold direction errors, missing-data handling) that simpler NLP benchmarks do not.

This project uses that task as a testbed for evaluating where and why an AI reasoning pipeline fails.

---

## Approach

**Data sources**

- Trial eligibility criteria extracted from public Parkinson disease trials via the ClinicalTrials.gov API
- Fully synthetic patient profiles generated locally — no real patient data at any stage

**Labels**

Eligibility labels (`eligible`, `not_eligible`, `unclear`) were generated as draft candidates and then reviewed with LLM assistance against a documented labeling guide. Labels are a best-effort benchmark baseline; the guide specifies when the `unclear` label is required (ambiguous criteria, missing patient fields, unspecified severity).

**Matching pipeline**

The pipeline runs structured criterion-level reasoning for each patient–trial pair, then aggregates to a case-level prediction. Criterion-level output is retained for error analysis.

**Evaluation**

- Primary metrics: accuracy and macro F1 across 150 labeled pairs
- Secondary: per-class F1, confusion matrix, criterion-level results
- Robustness: minimal pairs (single-variable patient edits) and counterfactual pairs

---

## Evaluation Design

### Label schema

| Label | Condition |
|---|---|
| `eligible` | Patient satisfies all inclusion criteria and no exclusion criteria |
| `not_eligible` | Patient fails at least one hard inclusion or exclusion criterion |
| `unclear` | Criteria are ambiguous or patient data is insufficient to decide |

### Error taxonomy

Failures are categorized rather than aggregated:

| Category | Description |
|---|---|
| Numeric threshold errors | Wrong direction, wrong unit, or boundary-value misread |
| Negation / exclusion errors | Misreading "no history of" or "must not have" as a positive match |
| Temporal constraint errors | Ignoring washout periods, recency requirements, or minimum diagnosis duration |
| Underspecified data errors | Asserting Eligible or Ineligible when a required value is absent |
| Multi-step interaction errors | Failing when correct eligibility requires chaining two or more reasoning steps |

### Robustness checks

Minimal pairs verify that changing a single patient attribute (e.g., flipping a medication field) produces the expected change in prediction. Counterfactual pairs apply the same logic to inverted eligibility scenarios. Both sets passed 8/8.

---

## Results

Evaluated on 150 labeled patient–trial pairs using LLM-reviewed draft labels.

| Metric | Score |
|---|---|
| Accuracy | 0.687 |
| Macro F1 | 0.686 |
| Eligible F1 | 0.659 |
| Not-eligible F1 | 0.703 |
| Unclear F1 | 0.696 |

Majority-class baseline: 0.51 accuracy / 0.23 macro F1.

**Error breakdown (gold → predicted)**

| Error type | Count |
|---|---|
| unclear → eligible | 18 |
| unclear → not_eligible | 12 |
| eligible → unclear | 7 |
| eligible → not_eligible | 4 |
| not_eligible → unclear | 4 |
| not_eligible → eligible | 2 |
| **Total** | **47** |

The dominant failure mode is `unclear` cases predicted as `eligible` (18 of 47 errors), driven by incomplete patient profiles where the pipeline lacks sufficient signal to block eligibility. This pattern is consistent with the underspecified-data error category and points to the next improvement area: tighter uncertainty handling when required fields are absent.

**Test suite:** 579 tests currently passing.

---

## Limitations

- Labels are LLM-reviewed draft labels, not clinical gold truth. Results measure reasoning quality against a best-effort benchmark, not clinical accuracy.
- Trial data is limited to Parkinson disease. Performance on other indications is untested.
- Synthetic patient profiles are controlled for reasoning isolation; they may not reflect the full variability of real patient records.
- This project is for research and benchmarking purposes only. No medical advice. No real patient data. No clinical validation. Results must not be used for any real-world clinical decisions.

---

## Transferable Skills Demonstrated

This project involves skills that transfer directly to AI evaluation and quality roles outside the biomedical domain:

- Designing structured benchmarks with well-defined label schemas, labeling guides, and inter-rater consistency checks
- Building and interpreting multi-class evaluations including accuracy, macro F1, and per-class breakdowns
- Constructing taxonomy-driven error analyses that distinguish failure modes rather than reporting aggregate error rates
- Reasoning carefully about uncertainty as a first-class label, not a residual category
- Designing robustness checks (minimal pairs, counterfactuals) to validate that predictions respond correctly to controlled input changes
- Framing limitations honestly and distinguishing between benchmark performance and real-world validity

---

## Tech Stack

- Python 3.12
- ClinicalTrials.gov public API
- pytest (579 tests)
- See `requirements.txt` for full dependencies

---

## How to Run

```bash
# Full local demo (no network required, reuses existing raw trials file)
PYTHONPATH=. python scripts/run_local_demo.py

# Full sample benchmark
PYTHONPATH=. python scripts/run_local_demo.py --full-sample

# Download fresh trial data and run demo
PYTHONPATH=. python scripts/run_local_demo.py --online --max-trials 20
```

**Full pipeline steps**

```bash
python -m scripts.download_trials          # Download raw trials from ClinicalTrials.gov
python -m scripts.extract_eligibility      # Extract eligibility criteria
python -m scripts.select_trial_cases       # Select trial cases
python -m scripts.generate_label_candidates
python -m scripts.generate_labels_seed
python -m scripts.export_labels_seed_review
python -m eval.run_llm_reviewed_benchmark  # Run LLM-reviewed benchmark
python -m eval.summarize_llm_reviewed_errors
python -m pytest -q                        # Run full test suite
```

**Hard-case subset tagging**

```bash
PYTHONPATH=. python eval/tag_hard_cases.py
```

Tags each record with difficulty labels (`hard_negative`, `hard_positive`, `ambiguous_clinical_severity`) using deterministic rules. Outputs per-tag classification metrics.

**HTML benchmark report**

```bash
PYTHONPATH=. python eval/generate_benchmark_report.py
# Output: reports/benchmark_report.html
```

---

## Documentation

| File | Purpose |
|---|---|
| `docs/BENCHMARK_CARD.md` | Task definition, dataset, metrics, failure modes, results |
| `docs/ERROR_TAXONOMY.md` | Categorized failure modes with examples |
| `docs/CURATED_EXAMPLES.md` | Easy, ambiguous, and hard cases illustrating the reasoning challenge |
| `docs/BASELINES.md` | Majority-class, keyword, and simple LLM baselines |
| `docs/PIPELINE_FLOW.md` | Stage-by-stage pipeline walkthrough |
| `docs/LABELING_GUIDE.md` | Reviewer guide for assigning and validating draft labels |
| `docs/PROJECT_SPEC.md` | Scope, data sources, matching logic, metrics, limitations |
| `docs/REPO_STRUCTURE.md` | Repository tree and file organization |
| `docs/IMPLEMENTATION_PLAN.md` | Completed phases, current status, future work |
