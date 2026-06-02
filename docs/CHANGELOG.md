# Changelog

## Unreleased / Current

Active development. All items below reflect completed work in the current project state.

---

## Benchmark Data Pipeline

- Built data pipeline sourcing publicly available ClinicalTrials.gov trial eligibility criteria.
- Generated synthetic patient cases covering a range of Parkinson disease presentations.
- Produced LLM-reviewed draft benchmark labels (`labels_llm_reviewed.json`) with gold label, rationale, evidence, and label status per patient–trial pair.
- Implemented patient case enrichment script (`enrich_patients.py`) to extract and normalize clinical metadata fields from existing patient text.
- Implemented trial eligibility enrichment script (`enrich_trials.py`) to extract eligibility-focused metadata from existing trial criteria.
- Added narrative patient profile generator (`generate_narrative_profiles.py`) producing a `narrative_profile` field from existing structured fields.
- Added trial metadata normalization script (`enrich_trial_metadata.py`) covering phase, status, intervention type, and condition.

---

## Evaluation and Metrics

- Executed benchmark over LLM-reviewed labels producing `results_llm_reviewed.json` and `results_llm_reviewed.csv`.
- Generated criterion-level results (`criterion_level_results.csv`) with per-criterion decisions and reasons.
- Implemented confusion matrix printing and per-class metrics (precision, recall, F1, macro F1, accuracy).
- Added confidence calibration report (`report_calibration.py`) grouping predictions by confidence band.
- Added confidence threshold sweep report (`report_confidence_thresholds.py`) computing coverage, accuracy, and macro F1 at each threshold cutoff.
- Added criterion type classifier (`classify_criterion_types.py`) assigning each criterion a deterministic type label.
- Added criterion type metrics aggregation producing per-type summaries (accuracy, error rate, label distribution).
- Added prediction distribution report (`report_prediction_distribution.py`) covering gold vs predicted distributions, per-patient and per-trial breakdowns, and confidence statistics.
- Added uncertainty and safety evaluation metrics: unsafe eligible errors, overly conservative errors, uncertainty errors, unclear recall/precision, overcommitment rate, critical/major/minor error rates.

---

## Error Analysis and Audit Reports

- Implemented error analysis pipeline (`summarize_llm_reviewed_errors.py`) producing `error_analysis_llm_reviewed.json` and `.csv` with error type, severity, gold/predicted pairs, and counts.
- Added error pattern analysis by criterion type (`analyze_errors_by_criterion_type.py`) showing error rates per criterion category.
- Added cross-trial patient consistency report (`report_cross_trial_consistency.py`) grouping predictions by patient and surfacing repeated error patterns.
- Added cross-patient trial consistency report (`report_cross_patient_consistency.py`) grouping predictions by trial and surfacing repeated error patterns.
- Added human-review friendly CSV export (`human_review_queue.csv`) with review priority scoring (high / medium / low) and reviewer notes column.
- Added label coverage and completeness report (`report_label_coverage.py`) checking required field presence per label record.
- Added prediction distribution report covering top error patients and trials.

---

## Robustness and Contrastive Checks

- Added counterfactual pair checks verifying that changing a single patient field changes the prediction outcome where expected.
- Added minimal pair checks for near-identical patients differing on one eligibility-relevant attribute.
- Added noise robustness / sensitivity analysis checking prediction stability under minor input perturbations.
- Added capability tag report grouping predictions by tagged capability area.
- Added abstention analysis treating `unclear` predictions as abstentions and reporting precision/recall over kept predictions.
- Added baseline comparison script (`run_baselines.py`) computing majority-class and random baselines for reference.

---

## Validation and Quality Checks

- Added patient cases schema validation script.
- Added trial cases schema validation script.
- Added processed output schema validation script.
- Added duplicate check script verifying no repeated patient–trial pairs.
- Added inter-rater / label consistency analysis for labels with multiple review passes.
- Added label disagreement and ambiguity report.
- Added label noise analysis report.
- Added criterion length and complexity analysis.
- Added patient field coverage analysis.
- Added trial metadata coverage and phase/status analysis.

---

## Per-Case and Criterion-Level Summaries

- Generated HTML benchmark report with per-example anchors and navigation.
- Added per-patient and per-trial summary sections to benchmark outputs.
- Added reasoning trace (`reasoning_trace`) field to each prediction record with ordered steps.
- Added structured `criterion_results` output with per-criterion decision, reason, and matched facts.
- Added `criterion_type_classified.csv` with deterministic criterion type labels.
- Added `patient_cases_enriched.json` and `patient_cases_narrative.json` as enriched dataset variants.
- Added `trial_cases_enriched.json` and `trial_cases_eligibility_enriched.json` as enriched trial variants.

---

## Documentation

- `docs/BENCHMARK_CARD.md` — benchmark card summarising dataset, task, metrics, and limitations.
- `docs/KNOWN_LIMITATIONS.md` — documented known matcher and dataset limitations.
- `docs/FUTURE_WORK.md` — documented planned improvements and open research questions.
- `docs/EXPECTED_REASONING_STEPS.md` — documented expected matcher reasoning steps per criterion type.
- `docs/CHANGELOG.md` — this file.
- README updated with example benchmark case and usage instructions.

---

## Notes

- Reports are generated artifacts and can be regenerated by running the scripts in `eval/` and `scripts/`.
- Labels are LLM-reviewed draft benchmark labels, not clinician-adjudicated gold labels.
- This project is intended for research and portfolio benchmarking purposes only, and is not validated or intended for real clinical use.
