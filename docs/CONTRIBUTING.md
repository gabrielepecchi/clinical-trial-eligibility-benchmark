# Contributing

This is a research and portfolio benchmark project using synthetic patient cases and publicly available clinical trial eligibility criteria. It is **not** a clinical decision-support system and is not intended for real clinical use.

---

## Development Principles

- Keep scripts small, deterministic, and auditable.
- Prefer the Python standard library; avoid new external dependencies unless already used by the project.
- Do not hardcode `patient_id` or `trial_id` values to improve benchmark results (benchmark cheating).
- Do not modify `rule_matcher.py` solely to chase individual known benchmark errors.
- Do not use real patient data or any personally identifiable information (PHI).
- Do not add clinical deployment claims or overstate benchmark performance.

---

## Adding New Synthetic Patient Cases

- Add cases to `data/cases/patient_cases.json` (or the relevant source file).
- Each case must be fully synthetic — no real patient data or PHI.
- Include a range of clinical presentations to improve coverage; do not add cases that target known benchmark errors.
- Required fields per case: `patient_id`, `condition`, `age`, and any relevant clinical fields.
- After adding cases, run enrichment scripts if applicable:
  ```
  PYTHONPATH=. python scripts/enrich_patients.py
  PYTHONPATH=. python scripts/generate_narrative_profiles.py
  ```
- Run validation:
  ```
  PYTHONPATH=. python eval/validate_patient_cases.py
  ```

---

## Adding New Trial Cases

- Add cases to `data/cases/trial_cases.json` (or the relevant source file).
- Use only publicly available trial eligibility criteria (e.g. from ClinicalTrials.gov).
- Include a clear `trial_id`, `eligibility_criteria` (inclusion and exclusion), and basic trial metadata.
- After adding cases, run enrichment and validation:
  ```
  PYTHONPATH=. python eval/enrich_trial_metadata.py
  PYTHONPATH=. python scripts/enrich_trials.py
  PYTHONPATH=. python eval/validate_trial_cases.py
  ```

---

## Adding or Reviewing Labels

- Labels live in `data/processed/labels_llm_reviewed.json`.
- Each label record should include: `patient_id`, `trial_id`, `label`, `label_status`, `rationale`, and `evidence`.
- Valid label values: `eligible`, `not_eligible`, `unclear`.
- Labels are LLM-reviewed draft benchmark labels — do not claim they are clinician-adjudicated gold labels.
- When adding or correcting labels, include a clear `rationale` and supporting `evidence` from the patient/trial text.
- Do not adjust labels to match current matcher predictions.
- After updating labels, run the label coverage report:
  ```
  PYTHONPATH=. python eval/report_label_coverage.py
  ```

---

## Adding Evaluation Scripts

- Place new evaluation scripts in `eval/`.
- Place new data-processing scripts in `scripts/`.
- Scripts should read from `data/processed/` and write to `data/processed/` or `reports/`.
- Scripts must be runnable with:
  ```
  PYTHONPATH=. python eval/your_script.py
  ```
- Use only existing fields from source files; do not invent labels, metrics, or clinical facts.
- Provide small, pure functions with clear docstrings.
- Print a short terminal summary of records read, records written, and output path.
- Exit non-zero only if required input files are missing or malformed.

---

## Adding Tests

- Place new tests in `tests/`.
- Tests must be runnable with:
  ```
  PYTHONPATH=. python -m pytest
  ```
- Use small, self-contained synthetic fixtures — do not depend on benchmark data files.
- Do not hardcode `patient_id` or `trial_id` from the benchmark dataset.
- For tests covering behaviour the matcher does not yet implement, use:
  ```python
  @pytest.mark.xfail(strict=False, reason="Matcher does not yet support X.")
  ```
- Passing tests should assert only stable, current behaviour.

---

## Regenerating Reports

Most reports are generated artifacts and should be regenerated from scripts rather than edited manually:

```
PYTHONPATH=. python -m pytest
PYTHONPATH=. python eval/run_llm_reviewed_benchmark.py
PYTHONPATH=. python eval/summarize_llm_reviewed_errors.py
PYTHONPATH=. python eval/classify_criterion_types.py
PYTHONPATH=. python eval/analyze_errors_by_criterion_type.py
PYTHONPATH=. python eval/report_calibration.py
PYTHONPATH=. python eval/report_confidence_thresholds.py
PYTHONPATH=. python eval/report_cross_trial_consistency.py
PYTHONPATH=. python eval/report_cross_patient_consistency.py
PYTHONPATH=. python eval/report_label_coverage.py
PYTHONPATH=. python eval/report_prediction_distribution.py
```

If a report is manually edited, note the change clearly in a comment or a PR description.

---

## Documentation Updates

- Documentation lives in `docs/`.
- Keep documentation factual; do not overstate benchmark performance or matcher capabilities.
- Do not add clinical validation claims, expert review claims, IRB approval claims, or citations that are not verifiable from the project files.
- Do not claim labels are gold-standard or clinician-adjudicated.
- Update `docs/CHANGELOG.md` when a meaningful feature or script is added.
- Update `docs/KNOWN_LIMITATIONS.md` when a new limitation is identified.

---

## Safety and Scope Rules

- **No real patient data or PHI** — all patients are synthetic.
- **No clinical deployment claims** — this project is for research and portfolio purposes only.
- **No benchmark cheating** — do not use `patient_id` or `trial_id` to special-case matcher logic.
- **No overstated claims** — accuracy and F1 figures are benchmark results on synthetic data, not clinical performance guarantees.
- **No invented facts** — enrichment and report scripts must extract values only from existing fields; they must not invent clinical data.

---

## Pull Request Checklist

Before submitting a pull request:

- [ ] All tests pass: `PYTHONPATH=. python -m pytest`
- [ ] New scripts run cleanly: `PYTHONPATH=. python path/to/script.py`
- [ ] No real patient data or PHI introduced.
- [ ] No hardcoded `patient_id` or `trial_id` in matcher or evaluation logic.
- [ ] No clinical deployment or validation claims added.
- [ ] Labels include rationale and evidence where applicable.
- [ ] Documentation is factual and does not overstate performance.
- [ ] Generated reports are regenerated from scripts, not manually patched.
- [ ] `docs/CHANGELOG.md` updated if a meaningful feature was added.
