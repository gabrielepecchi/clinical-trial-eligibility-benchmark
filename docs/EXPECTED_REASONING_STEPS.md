# Expected Reasoning Steps

This document describes idealized reasoning patterns for each criterion type
used in the clinical-trial-eligibility-benchmark.

> **Important:** This is not clinical guidance. It is intended to help
> interpret reasoning traces, error analysis, and criterion-level reports.
> All patient examples are synthetic.

---

## Age

**What the reasoning should check:**
- Read the patient's numeric age value.
- Compare against the criterion's minimum and maximum bounds.
- Respect boundary inclusivity ("≥ 18" includes 18; "> 18" excludes 18).
- Check both lower and upper bounds if both are specified.

**Common failure modes:**
- Wrong direction: treating a lower-bound criterion as an upper bound.
- Off-by-one at an exact boundary (e.g., age 40 for "age > 40").
- Ignoring the upper bound when both are specified.

**Expected uncertainty behavior:**
- If age is missing from the patient profile, predict `unclear`.
- Do not assume a default age.

---

## Diagnosis

**What the reasoning should check:**
- Confirm whether the patient has the specific required diagnosis (e.g., idiopathic Parkinson disease).
- Distinguish idiopathic PD from Parkinsonism, atypical parkinsonism, secondary parkinsonism, or healthy controls.
- Check whether the criterion requires a specific subtype, duration, or onset pattern.
- Verify that the diagnosis terminology in the profile matches or clearly maps to the criterion.

**Common failure modes:**
- Accepting "Parkinson disease" when the criterion requires "idiopathic Parkinson disease."
- Ignoring subtype restrictions (e.g., early-onset vs. late-onset).
- Assuming diagnosis is confirmed when only symptoms are mentioned.

**Expected uncertainty behavior:**
- If the diagnosis type is ambiguous or unspecified, predict `unclear`.
- Do not infer a specific subtype from a vague profile entry.

---

## Medication

**What the reasoning should check:**
- Identify whether the patient is currently taking the relevant drug or drug class.
- Recognize common drug class synonyms (e.g., MAO-B inhibitor → rasagiline, selegiline, safinamide).
- Respect stability requirements: "stable dose for ≥ 4 weeks" requires checking medication duration.
- Respect washout requirements: "no MAO-B inhibitor within 30 days" requires checking recency.

**Common failure modes:**
- Missing a MAO-B inhibitor because the drug is listed by brand name or partial name.
- Ignoring stability duration when the criterion specifies a minimum stable period.
- Assuming medication is absent if the profile is silent rather than explicitly negative.
- Missing COMT inhibitor or dopamine agonist exclusions when synonyms are used.

**Expected uncertainty behavior:**
- If the medication list is incomplete or missing, prefer `unclear` over assuming no medications.
- If stability duration is unspecified in the profile, prefer `unclear` rather than assuming stability.

---

## Procedure / DBS History

**What the reasoning should check:**
- Determine whether the patient has a history of the excluded procedure.
- Distinguish prior procedure history (past event) from active implanted device (current state).
- Recognize synonyms: "deep brain stimulation," "DBS," "brain stimulation surgery."
- Check whether the exclusion applies to a specific procedure type (e.g., DBS only, or any prior brain surgery).

**Common failure modes:**
- Missing a DBS history entry because it is phrased indirectly (e.g., "previously implanted").
- Confusing prior history with current device status.
- Failing to detect ablation, lesioning, or focused ultrasound when the criterion says "prior surgical procedure."

**Expected uncertainty behavior:**
- If procedure history is absent from the profile, prefer `unclear` if the criterion is an exclusion.
- Do not assume the patient has no prior procedures just because it is not mentioned.

---

## Cognitive Status / MoCA

**What the reasoning should check:**
- Read the numeric MoCA or MMSE score from the patient profile.
- Compare against the criterion threshold (e.g., "MoCA ≥ 26").
- Respect boundary inclusivity.
- If a general "cognitive status" field is present but no numeric score, do not treat it as equivalent to a validated threshold score.

**Common failure modes:**
- Treating "normal cognitive status" as equivalent to "MoCA ≥ 26" without a score.
- Missing the score because it is stored in a nested or non-standard field.
- Ignoring the lower end of a cognitive range ("MoCA between 20 and 26").

**Expected uncertainty behavior:**
- If the MoCA or MMSE score is absent, predict `unclear`.
- Do not infer a passing score from "cognitively normal" or similar qualitative entries.

---

## Device / Pacemaker

**What the reasoning should check:**
- Identify whether the patient has an implanted cardiac or neural device.
- Recognize synonyms: pacemaker, ICD, defibrillator, implanted cardioverter-defibrillator, IVCD, cardiac device.
- Check whether the exclusion is for any implanted device or a specific device type.
- Distinguish currently implanted (active exclusion) from historical (past device, now removed).

**Common failure modes:**
- Missing a pacemaker exclusion because the device is described with a variant term.
- Assuming no device history if the profile is silent on the topic.
- Failing to detect a device mentioned indirectly (e.g., "cardiac procedure" instead of "pacemaker").

**Expected uncertainty behavior:**
- If device history is unspecified, prefer `unclear` when the criterion is an exclusion requiring confirmed absence.

---

## Severity / Disease Stage

**What the reasoning should check:**
- Read the UPDRS III score, Hoehn-Yahr stage, or other severity measure.
- Compare against the criterion's allowed range (e.g., "UPDRS III 20–40," "Hoehn-Yahr 1–3").
- Respect both lower and upper bounds when a range is specified.

**Common failure modes:**
- Using UPDRS total instead of UPDRS III (or vice versa) when the criterion specifies a subscale.
- Missing an upper bound on severity (e.g., excluding patients with UPDRS III > 40).
- Confusing Hoehn-Yahr stages with other scales.

**Expected uncertainty behavior:**
- If the severity score is absent, predict `unclear`.
- Do not infer a severity value from disease duration or medication dose alone.

---

## Temporal Requirements

**What the reasoning should check:**
- Identify the temporal constraint: recency ("within 30 days"), duration ("for at least 6 months"), or washout ("no use in the past 4 weeks").
- Compare the patient's relevant date, duration, or period against the criterion threshold.
- Respect direction: "within" means at most; "for at least" means a minimum.

**Common failure modes:**
- Ignoring the temporal dimension entirely and treating the criterion as a simple yes/no.
- Confusing "washout period" (time since last use) with "stability period" (time on current dose).
- Incorrectly computing durations when only a start date or approximate period is provided.

**Expected uncertainty behavior:**
- If the relevant date or duration is absent from the profile, predict `unclear`.
- Do not assume the temporal requirement is satisfied when the profile is silent on timing.

---

## Comorbidities / Safety Exclusions

**What the reasoning should check:**
- Identify the specific comorbidity or safety exclusion (e.g., "no clinically significant cardiac disease").
- Determine whether the patient's listed conditions match the exclusion.
- Avoid over-claiming: if a condition is mentioned but its severity or clinical significance is not specified, the exclusion may not be met.
- Recognize common synonyms for cardiac, renal, hepatic, and other exclusions.

**Common failure modes:**
- Treating any mention of a cardiac condition as satisfying "clinically significant cardiac disease" without severity information.
- Missing an exclusion because the condition is named differently in the profile.
- Asserting eligibility when relevant comorbidities are ambiguously described.

**Expected uncertainty behavior:**
- If a condition is mentioned but clinical significance is unspecified, prefer `unclear` rather than assuming eligibility or ineligibility.
- Do not map vague symptom descriptions to specific excluded diagnoses without clear evidence.

---

## Missing or Ambiguous Information

**What the reasoning should check:**
- Identify which required patient fields are absent or insufficiently specified.
- For each missing required field, the correct prediction is typically `unclear`.
- Do not assume a default value for missing fields.
- Do not assert eligibility or ineligibility based on incomplete data.

**Common failure modes:**
- Predicting `eligible` or `not_eligible` when a required criterion field is absent.
- Treating "not mentioned" as equivalent to "confirmed absent."
- Failing to flag ambiguous phrasing as a source of uncertainty.

**Expected uncertainty behavior:**
- Missing required patient data → `unclear`.
- Ambiguous criterion language with no patient data to resolve it → `unclear`.
- Sufficient information for some criteria but not all → may still result in `not_eligible`
  if a definitive exclusion is already confirmed, otherwise `unclear`.

---

## How to Use This Document

This document can support:

- **Reviewing reasoning traces**: Check whether a reasoning trace addresses each
  expected step for its criterion type. Missing steps suggest incomplete reasoning.

- **Classifying error types**: When a prediction is wrong, compare the actual
  reasoning against the expected steps to identify the specific failure mode
  (e.g., ignored temporal constraint, missed synonym, assumed absent data).

- **Designing regression tests**: Use the common failure modes as a checklist
  when writing new test cases for `rule_matcher.py` or downstream scripts.

- **Interpreting criterion-level reports**: Use the criterion types and failure
  modes to contextualize low-coverage or high-error criterion categories
  in `criterion_level_results.csv` and related analysis reports.
