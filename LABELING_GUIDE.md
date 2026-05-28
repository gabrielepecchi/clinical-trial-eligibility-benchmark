# Labeling Guide

This guide supports consistent review of draft eligibility labels in the benchmark pipeline. Labels are assigned to synthetic patient–trial pairs for the purpose of evaluating AI reasoning quality. They are **not clinical determinations** and have not been validated by medical professionals.

---

## Label Definitions

| Label | Meaning |
|---|---|
| **Eligible** | The patient meets all inclusion criteria and does not trigger any exclusion criteria based on the available synthetic profile. |
| **Ineligible** | The patient fails at least one inclusion criterion or triggers at least one exclusion criterion. |
| **Uncertain** | The available profile is insufficient to make a confident determination, or the criterion wording is ambiguous. |

---

## Basic Review Rules

- Evaluate each criterion independently before reaching an overall label.
- Base the label only on information present in the synthetic profile — do not infer missing data.
- Apply criteria as written; do not interpret beyond the text.
- Assign **Uncertain** rather than guessing when information is missing or ambiguous.
- Be consistent: the same profile + the same criteria should always produce the same label.

---

## Common Reviewer Checks

- Does the patient's age fall within the trial's required range?
- Are all required diagnoses or conditions explicitly present in the profile?
- Do any listed medications or conditions match exclusion criteria?
- Are relevant lab values present and within or outside the required thresholds?
- Does the criterion use negation (e.g. "no prior history of…")? Confirm it is applied correctly.

---

## When to Choose Uncertain

Use **Uncertain** when:

- A required data field (e.g. a lab value or diagnosis date) is absent from the synthetic profile.
- A criterion references a condition that is partially but not definitively described.
- The criterion wording is genuinely ambiguous and cannot be resolved from the text alone.
- Two criteria conflict and the profile does not provide enough detail to resolve the conflict.

Do **not** use Uncertain to avoid a difficult decision when the information needed is actually present.

---

## Short Examples

**Example 1 — Eligible**
- Trial requires: age 18–65, Type 2 diabetes diagnosis, no prior insulin use.
- Patient: age 52, Type 2 diabetes, no insulin recorded.
- Label: **Eligible** — all criteria are met.

**Example 2 — Ineligible**
- Trial requires: no prior chemotherapy.
- Patient: prior chemotherapy recorded.
- Label: **Ineligible** — exclusion criterion is triggered.

**Example 3 — Uncertain**
- Trial requires: HbA1c ≥ 7.5% within the last 6 months.
- Patient: HbA1c value present but no date recorded.
- Label: **Uncertain** — recency cannot be confirmed from the available data.

---

## Important Reminder

All labels produced by this pipeline are **draft benchmark labels** assigned to **synthetic patients** against **public trial criteria**. They are intended solely for evaluating and improving AI eligibility reasoning. They do not constitute clinical decisions, medical advice, or validated ground truth.
