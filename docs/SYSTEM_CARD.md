# System Card: Clinical Trial Eligibility Benchmark

## System Overview

This system is a local, rule-based clinical trial eligibility matcher built for research and portfolio benchmarking purposes. It evaluates synthetic Parkinson disease patient profiles against public trial eligibility criteria and produces structured eligibility predictions with explanations.

The system consists of:
- A rule-based matcher (`rule_matcher.py`) that evaluates patient profiles against trial inclusion and exclusion criteria.
- A benchmark pipeline that runs the matcher over reviewed patient–trial pairs and computes evaluation metrics.
- A suite of analysis and reporting scripts for error analysis, calibration, and audit.

---

## Intended Use

- Portfolio demonstration of benchmark engineering, NLP evaluation, and clinical data pipeline skills.
- Research into rule-based and LLM-assisted clinical trial eligibility matching.
- Structured error analysis, capability documentation, and matcher evaluation.
- Educational exploration of clinical NLP evaluation methodology.

---

## Out-of-Scope Use

This system is **not** intended for:

- Real patient care or clinical trial eligibility screening.
- Regulatory submissions or compliance claims.
- Replacing clinical, legal, or regulatory review of trial eligibility.
- Any deployment in a healthcare or clinical operations setting.
- Use as a medical device or clinical decision-support tool.

---

## Inputs and Outputs

**Inputs:**
- Synthetic patient profiles (structured JSON with clinical fields and free-text narrative).
- Trial eligibility criteria (inclusion and exclusion criteria from public ClinicalTrials.gov records).

**Outputs per patient–trial pair:**
- Predicted eligibility label: `eligible`, `not_eligible`, or `unclear`.
- Confidence score (matcher-internal estimate, not a calibrated probability).
- Matched facts, blocking criteria, and uncertain criteria.
- Structured reasoning trace and criterion-level decisions.
- Plain-text matcher explanation.

---

## What the Matcher Does

- Applies deterministic keyword and pattern rules to match patient fields against trial criteria text.
- Identifies blocking criteria (exclusion signals) and uncertain criteria (cannot evaluate from available data).
- Returns `eligible` when no blocking criteria are found and key inclusion criteria are met.
- Returns `not_eligible` when one or more blocking criteria are identified.
- Returns `unclear` when critical criteria cannot be evaluated from the available patient data.

---

## What the Matcher Does Not Do

- Does not use real patient data or any personally identifiable information.
- Does not perform probabilistic clinical reasoning or diagnosis.
- Does not access external databases, medical records, or real-time data.
- Does not handle all clinical edge cases, temporal constraints, or complex multi-criterion interactions.
- Does not compute derived values such as BMI from height and weight, or unit conversions between lab value formats.
- Does not claim to replicate the reasoning of a trained clinician.

---

## Known Limitations

- **Synthetic patients only:** Patient profiles are fully fictional and do not represent real individuals. Benchmark results reflect performance on synthetic data, not real-world clinical populations.
- **Draft labels:** Benchmark labels are LLM-reviewed draft labels, not clinician-adjudicated gold labels. Label quality is variable and suitable for research benchmarking only.
- **Limited criterion coverage:** The matcher handles a subset of criterion types reliably (age, diagnosis, DBS history, some medications). Lab values, temporal windows, and numeric thresholds are incompletely supported.
- **Conservative on missing data:** When patient data is insufficient to evaluate a criterion, the matcher returns `unclear`, which may produce higher-than-expected unclear rates.
- **No unit conversion:** Numeric thresholds require values in the expected unit; lb/kg and g/L/g/dL conversions are not performed.
- **Rule brittleness:** Matcher rules are pattern-based and may miss paraphrased or unusual phrasings of eligibility criteria.

For a full list of known limitations, see `docs/KNOWN_LIMITATIONS.md`.

---

## Safety and Clinical-Use Disclaimer

**This system is not a medical device and is not validated for clinical use.**

- All patient profiles are synthetic. No real patient data was used at any stage.
- Benchmark labels are draft outputs of an LLM-assisted review process. They have not been adjudicated by licensed clinicians.
- Benchmark accuracy and F1 figures reflect performance on synthetic data only. They do not indicate real-world clinical performance.
- This system must not be used to make or inform real clinical trial eligibility decisions, patient care decisions, or any decisions affecting real individuals.
