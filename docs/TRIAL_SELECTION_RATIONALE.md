# Trial Selection Rationale

> **Note:** Trials were selected for benchmark development purposes only.
> Inclusion of a trial does not imply endorsement.
> This document is not for clinical use or medical decision-making.

---

## Purpose

This document explains the principles used to select clinical trials for the eligibility matching benchmark. The goal of trial selection was to produce a diverse, tractable set of patient–trial pairs that stress-test the rule-based matcher across a range of eligibility criteria types.

---

## Inclusion Criteria for Trial Selection

Trials were included if they met all of the following:

- **Parkinson's disease relevance:** The trial targets Parkinson's disease or a closely related parkinsonian condition.
- **Public availability:** The trial record is publicly available on ClinicalTrials.gov.
- **English eligibility text:** Inclusion and/or exclusion criteria are written in English.
- **Non-empty criteria:** Both inclusion and exclusion sections contain meaningful text, not placeholder or unavailable entries.
- **Sufficient detail for rule-based matching:** Criteria contain at least some structured constraints — age ranges, diagnosis requirements, medication restrictions, or procedure history — that a rule-based system can evaluate.

---

## Exclusion Criteria for Trial Selection

Trials were excluded if any of the following applied:

- Eligibility criteria were absent, empty, or listed as "not provided."
- Criteria were written in a language other than English.
- The trial was unrelated to Parkinson's disease or parkinsonian syndromes.
- Criteria text was too vague or entirely narrative to support structured matching.
- The trial was a duplicate of an already-selected record.

---

## Why Parkinson's Disease Trials

Parkinson's disease was chosen as the therapeutic focus for several practical reasons:

- It has a well-defined core diagnosis with recognisable inclusion patterns (idiopathic PD, UPDRS scores, Hoehn and Yahr stage, levodopa response).
- Trials span a wide range of types — pharmacological, device-based, rehabilitation, imaging, and observational — providing natural diversity in eligibility criteria structure.
- Common exclusion patterns (prior DBS, MAO-B inhibitor use, cognitive impairment, recent trial participation) are concrete enough to implement as rules.
- It is a domain where synthetic patient profiles can be constructed with clinically plausible variation without requiring real patient data.

Restricting to a single disease area also makes benchmark evaluation more interpretable: performance differences are more likely to reflect matcher logic than domain shift.

---

## Why Public ClinicalTrials.gov Records

ClinicalTrials.gov was chosen as the sole source of trial criteria because:

- Records are publicly available and freely usable for research purposes.
- Eligibility criteria are structured as plain text, accessible without proprietary access.
- The registry covers a wide range of trial phases, intervention types, and study designs.
- Using a single, well-known public registry makes the benchmark reproducible and auditable.

No proprietary, institutional, or restricted trial databases were used.

---

## Eligibility Criteria Requirements

Trials were required to have eligibility criteria that include at least one of the following types of constraint:

- Age range or age threshold
- Diagnosis confirmation (idiopathic PD, atypical parkinsonism, specific subtypes)
- Medication inclusion or exclusion (e.g. levodopa stability, MAO-B inhibitor washout)
- Prior procedure history (e.g. DBS implantation, surgical history)
- Cognitive or motor severity thresholds (e.g. MoCA score, UPDRS, Hoehn and Yahr stage)
- Device contraindications (e.g. pacemaker, implanted stimulator)

Trials with criteria limited to administrative or logistical constraints only (e.g. ability to attend clinic visits, informed consent capacity) were deprioritised unless richer clinical criteria were also present.

---

## Diversity of Trial Types

Where possible, trials were selected to represent different study designs and intervention categories:

- Pharmacological interventions
- Device-based interventions (DBS, stimulation devices)
- Rehabilitation and exercise studies
- Imaging and biomarker studies
- Observational and quality-of-life studies
- Neuropsychiatric and cognitive intervention studies

This diversity ensures the benchmark tests the matcher against varied criteria structures rather than a narrow slice of eligibility patterns.

---

## Known Limitations

- The number of trials in the benchmark is limited by the manual effort required to construct synthetic patient profiles and produce draft labels.
- Trial criteria summaries may have been simplified or reformatted for consistency; edge cases in the original text may not be fully preserved.
- Trials were selected to support rule-based matching and may not represent the full complexity of real eligibility screening.
- Selection was not systematic or exhaustive; it reflects pragmatic choices made during benchmark development.
- The benchmark does not cover trials outside Parkinson's disease.

---

## Relationship to the Benchmark Labels

Trial selection directly shapes what eligibility decisions appear in the benchmark. Trials with ambiguous or underspecified criteria tend to produce more `unclear` labels. Trials with strict device or medication exclusions tend to produce more `not_eligible` labels for the synthetic patient population used.

Label distribution is therefore partly a function of trial selection, not only of matcher performance. This should be taken into account when interpreting benchmark metrics.
