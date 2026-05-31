# Failure Gallery

> **Warning:** This benchmark uses synthetic patient profiles and draft LLM-reviewed labels.
> Labels have not been validated against clinical gold standards and are not for clinical use.
> This document is for development and portfolio purposes only.

Ten representative prediction errors from the LLM-reviewed benchmark (v0.1, 150 evaluated pairs, 47 total errors).
Cases are selected to cover the main recurring failure modes. Examples are summarized from generated benchmark artifacts and should be treated as draft analysis.

---

## Case 1 — Missed uncertainty: unrecognised comorbidity ambiguity

| Field       | Value |
|-------------|-------|
| Gold        | `unclear` |
| Predicted   | `eligible` |
| Error type  | `missed_uncertainty_missing_detail` |
| Severity    | major |

**What went wrong:** The patient had REM sleep behavior disorder. The matcher confirmed age and PD diagnosis, found no blocking criteria, and predicted `eligible`. It did not flag that the comorbidity's relevance to a neuropsychiatric protocol was unresolved.

**Improvement direction:** Teach the matcher to flag comorbidities that are neither explicitly excluded nor clearly irrelevant. Unresolved comorbidity relevance should surface as an `uncertain_criterion`.

---

## Case 2 — Overcalled not_eligible: healthy control vs. comparator arm ambiguity

| Field       | Value |
|-------------|-------|
| Gold        | `unclear` |
| Predicted   | `not_eligible` |
| Error type  | `overcalled_not_eligible_instead_of_unclear` |
| Severity    | major |

**What went wrong:** The patient was a healthy control. The matcher applied the PD diagnosis inclusion rule and blocked the patient as `not_eligible`. The trial summary did not clarify whether a healthy comparator arm existed, so the gold label was `unclear`.

**Improvement direction:** When a patient is a healthy control and trial scope is ambiguous, the matcher should prefer `unclear` over a hard `not_eligible` block.

---

## Case 3 — Overcalled not_eligible: DBS misclassification in a DBS-focused trial

| Field       | Value |
|-------------|-------|
| Gold        | `unclear` |
| Predicted   | `not_eligible` |
| Error type  | `overcalled_not_eligible_instead_of_unclear` |
| Severity    | major |

**What went wrong:** The patient had prior bilateral DBS. The matcher treated DBS history as a generic exclusion and blocked the patient. The trial was a DBS international study where prior implantation may be a requirement, not an exclusion — the protocol details were insufficient to determine this.

**Improvement direction:** DBS rules must be context-sensitive. A rule that excludes DBS in drug trials should not fire identically in device or DBS-specific trials.

---

## Case 4 — Overcalled not_eligible: atypical parkinsonism in a diagnostic imaging trial

| Field       | Value |
|-------------|-------|
| Gold        | `eligible` |
| Predicted   | `not_eligible` |
| Error type  | `overcalled_not_eligible` / `overstrict_age_or_stage_rule` |
| Severity    | minor |

**What went wrong:** The patient had suspected parkinsonism with an unconfirmed diagnosis. The matcher blocked them because the rule requires idiopathic PD. The trial was specifically an imaging study targeting patients with uncertain parkinsonian diagnoses — diagnostic uncertainty was an inclusion feature, not a barrier.

**Improvement direction:** Trials focused on differential diagnosis or early/uncertain PD should not trigger the idiopathic PD inclusion rule.

---

## Case 5 — Undercalled not_eligible: healthy control in a PD-only study

| Field       | Value |
|-------------|-------|
| Gold        | `not_eligible` |
| Predicted   | `unclear` |
| Error type  | `undercalled_not_eligible_as_unclear` |
| Severity    | major/minor |

**What went wrong:** The patient was a healthy control. The matcher was uncertain whether PD was required and produced `unclear`. For a PD enteric nervous system study with no confirmed control arm, the gold label was `not_eligible`.

**Improvement direction:** If no comparator arm is mentioned and the trial targets PD patients, the absence of PD should resolve to `not_eligible` rather than `unclear`. The matcher needs a stronger default for disease-specific trials.

---

## Case 6 — Missed safety uncertainty: autonomic dysfunction in a rehabilitation trial

| Field       | Value |
|-------------|-------|
| Gold        | `unclear` |
| Predicted   | `eligible` |
| Error type  | `missed_safety_uncertainty` |
| Severity    | major |

**What went wrong:** The patient had orthostatic hypotension. The matcher confirmed PD and found no blocks, predicting `eligible`. The gold label acknowledged that autonomic dysfunction raises unresolved safety questions for a rehabilitation quality-of-life protocol.

**Improvement direction:** Autonomic dysfunction, cardiac comorbidities, and fall-risk indicators should be treated as candidate uncertain criteria when a trial involves physical activity or unspecified safety screening.

---

## Case 7 — Overcalled unclear: over-conservative gait/FoG requirement

| Field       | Value |
|-------------|-------|
| Gold        | `eligible` |
| Predicted   | `unclear` |
| Error type  | `overcalled_unclear` |
| Severity    | major |

**What went wrong:** The matcher flagged that gait or freezing of gait (FoG) features were not documented in the patient profile, producing `unclear`. The trial was a general PD study (somatosensory stimulation device) where PD diagnosis was the operative criterion, not a specific gait requirement.

**Improvement direction:** The matcher should not require documented gait features unless the trial explicitly mandates them. Absence of documentation ≠ absence of the feature.

---

## Case 8 — Missed device uncertainty: implicit DBS hardware requirement

| Field       | Value |
|-------------|-------|
| Gold        | `unclear` |
| Predicted   | `eligible` |
| Error type  | `missed_device_uncertainty` |
| Severity    | major |

**What went wrong:** The patient had PD with imaging workup but no confirmed DBS. The trial was an fMRI-DBS study that likely required active DBS candidacy or hardware. The matcher confirmed PD and produced `eligible` without flagging the device requirement gap.

**Improvement direction:** Trials with DBS or other device keywords in the title or summary should trigger an uncertain criterion when the patient's device status is undocumented.

---

## Case 9 — Missed uncertainty: missing disease-specific symptom for specialised trial

| Field       | Value |
|-------------|-------|
| Gold        | `unclear` |
| Predicted   | `eligible` |
| Error type  | `missed_uncertainty_missing_detail` |
| Severity    | major |

**What went wrong:** The patient had mild tremor but no documented freezing of gait (FoG). The trial was a FoG-specific study. The matcher matched on PD diagnosis and produced `eligible` without checking whether FoG was present or documented.

**Improvement direction:** Specialised symptom-focused trials (FoG, dyskinesia, non-motor symptoms) should require confirmation of the target symptom. Absence of confirmation should produce `unclear`, not `eligible`.

---

## Case 10 — Overcalled not_eligible: frailty rule too broad

| Field       | Value |
|-------------|-------|
| Gold        | `eligible` |
| Predicted   | `not_eligible` |
| Error type  | `overcalled_not_eligible` |
| Severity    | minor |

**What went wrong:** The patient had PD with gait impairment. The matcher applied a frailty/fall-risk blocking rule and excluded the patient from a treadmill and agility training trial. The gold label was `eligible` because motor dysfunction is precisely what these rehabilitation trials target.

**Improvement direction:** Frailty and fall-risk exclusions should only fire when the trial explicitly excludes them. Rehabilitation and exercise trials often target patients with motor impairment; the current rule over-generalises.

---

## Summary of recurring failure patterns

| Pattern | Frequency | Direction |
|---------|-----------|-----------|
| `missed_uncertainty_missing_detail` | common | Matcher needs broader uncertain-criteria detection |
| `overcalled_not_eligible_instead_of_unclear` | common | Rules firing too hard; prefer `unclear` on ambiguous criteria |
| `overcalled_unclear` | recurring | Matcher over-flags absent documentation as uncertainty |
| `undercalled_not_eligible_as_unclear` | observed | Needs stronger default for disease-specific trials |
| `overcalled_not_eligible` (minor) | observed | Context-sensitive rules needed for trial type |

The dominant failure mode is the matcher predicting a confident label (`eligible` or `not_eligible`) when the correct answer is `unclear`. This drives the overcommitment rate of 0.455 and unclear recall of 0.545 seen in v0.1.
