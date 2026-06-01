# Patient Generation Rationale

> **Note:** All patient profiles in this benchmark are fully synthetic.
> No real patient records, protected health information (PHI), or clinical databases were used.
> This document is not for clinical use or medical decision-making.

---

## Purpose

This document explains the design choices behind the synthetic patient profiles used in the eligibility matching benchmark. The goal was to produce a set of profiles that collectively stress-test the matcher across the eligibility dimensions most commonly encountered in Parkinson's disease clinical trials.

---

## Why Synthetic Patients

Synthetic profiles were chosen over real or de-identified patient data for several reasons:

- **No PHI risk:** Fully synthetic profiles eliminate any possibility of re-identification or privacy harm.
- **Controlled coverage:** Synthetic generation allows deliberate inclusion of edge cases, borderline values, and missing-data scenarios that may be rare in real datasets.
- **Reproducibility:** The benchmark can be shared and reproduced without data access agreements or IRB oversight.
- **Flexibility:** Profiles can be extended or modified as the benchmark evolves without ethical or regulatory constraints.

No IRB approval was sought or required. No clinical validation was performed.

---

## Patient Profile Dimensions

Each synthetic patient profile was designed to include variation across dimensions that directly map to common eligibility criteria in Parkinson's disease trials. The dimensions covered are:

- **Age** — a primary inclusion/exclusion criterion in most trials; profiles span a range from early-onset to older adult presentations.
- **Sex / gender** — included where relevant to trial-specific criteria or demographic completeness.
- **Parkinson's diagnosis subtype** — idiopathic PD, atypical parkinsonism, suspected parkinsonism with unconfirmed diagnosis, and healthy controls are represented to test diagnosis-based inclusion rules.
- **Disease duration** — relevant to trials requiring a minimum or maximum duration of PD diagnosis.
- **Motor severity** — represented through Hoehn and Yahr stage, UPDRS scores where available, and narrative descriptions of motor function.
- **Cognitive status** — MoCA or MMSE scores where available; narrative descriptions of cognitive impairment or intact cognition where scores are absent.
- **Medication exposure** — current medications (levodopa, dopamine agonists, MAO-B inhibitors, amantadine), medication stability, and washout history relevant to trial exclusion criteria.
- **DBS or device history** — prior or active deep brain stimulation implantation, a frequent exclusion criterion in drug trials and a potential inclusion criterion in device trials.
- **Pacemaker and device contraindications** — cardiac pacemakers and other implanted devices relevant to trials involving electrical stimulation or MRI procedures.
- **Comorbidities and safety signals** — including autonomic dysfunction, orthostatic hypotension, psychiatric history, cardiac conditions, and fall risk.
- **Recent trial participation** — where relevant to exclusion criteria prohibiting concurrent or recent investigational drug exposure.

---

## Diagnosis Variation

Diagnosis variation is one of the most important sources of eligibility complexity in this benchmark. Profiles include:

- Confirmed idiopathic Parkinson's disease (the most common inclusion target)
- Atypical parkinsonism with diagnostic uncertainty (tests over-strict idiopathic PD rules)
- Suspected parkinsonism undergoing diagnostic workup (relevant to imaging and diagnostic trials)
- Healthy controls (relevant to trials with comparator arms, or as clear exclusions in PD-only trials)

This variation is intentional. It exposes cases where the matcher must distinguish between a hard exclusion and a genuine uncertainty.

---

## Disease Severity and Duration

Motor severity and disease duration were varied to cover:

- Early-stage PD with mild symptoms
- Mid-stage PD with moderate motor impairment
- Advanced PD with significant disability
- Long disease duration (relevant to duration minimums)
- Short or undocumented disease duration (relevant to uncertainty cases)

Where UPDRS or Hoehn and Yahr values are present, they reflect plausible synthetic values for the described presentation. Where they are absent, the profile tests whether the matcher correctly flags missing severity data.

---

## Medication History

Medication profiles were designed to cover:

- Stable levodopa/carbidopa regimens (standard PD treatment, often an inclusion signal)
- MAO-B inhibitor use (rasagiline, selegiline — a common exclusion in trials involving investigational drugs)
- Dopamine agonist use
- Amantadine use
- Polypharmacy profiles with multiple agents
- Absent or undocumented medication history (tests unclear handling)

Medication variation directly targets one of the most common failure modes identified in benchmark error analysis.

---

## Procedure and Device History

Procedure and device history was varied to cover:

- No prior DBS (most common baseline)
- Prior bilateral DBS implantation (exclusion in drug trials; potential inclusion in device trials)
- DBS candidacy under evaluation (ambiguous for many trial types)
- Cardiac pacemaker present (relevant to MRI-based and stimulation trials)
- No documented procedure history (tests whether the matcher assumes absence or flags uncertainty)

---

## Cognitive and Psychiatric Variation

Cognitive and psychiatric variation was included because many trials exclude patients with dementia or significant cognitive impairment. Profiles include:

- Normal cognitive status
- Mild cognitive impairment (MCI)
- Dementia or significant cognitive decline
- Psychiatric comorbidities such as depression and anxiety
- REM sleep behavior disorder (a common non-motor PD feature with ambiguous eligibility implications)
- Undocumented cognitive status (tests unclear handling)

---

## Comorbidities and Safety Signals

Comorbidities were selected to reflect realistic safety concerns in PD trial eligibility:

- Autonomic dysfunction and orthostatic hypotension (relevant to rehabilitation and physical intervention trials)
- Hypertension (common comorbidity; usually not an exclusion unless severe or unstable)
- Cardiac arrhythmia (relevant to trials with cardiovascular safety criteria)
- Fall risk and frailty indicators (relevant to exercise and device trials)
- Renal or hepatic conditions where relevant to pharmacological trials

These comorbidities were chosen to test whether the matcher applies exclusion rules too broadly or correctly flags ambiguity.

---

## Relationship to Trial Eligibility Criteria

Patient profiles were not generated independently of the trial set. Dimensions were chosen specifically because they map to eligibility criteria present in the selected trials. This alignment ensures that the benchmark produces a meaningful distribution of `eligible`, `not_eligible`, and `unclear` labels rather than trivially easy or trivially impossible cases.

Some profiles were designed to be borderline or ambiguous by construction — for example, patients with atypical parkinsonism matched against trials requiring idiopathic PD, or patients with undocumented disease duration matched against trials with duration requirements.

---

## Known Limitations

- Synthetic profiles cannot fully capture the complexity and internal consistency of real clinical records.
- Some dimensions (e.g. UPDRS scores, MoCA values) are present in only a subset of profiles, reflecting realistic data sparsity but limiting rule coverage for those criteria.
- Profiles were not reviewed by clinicians and may contain combinations of features that would be clinically unusual.
- The patient population is limited to Parkinson's disease and does not cover other therapeutic areas.
- No temporal data (exact dates, washout windows, diagnosis timestamps) is systematically present across all profiles in v0.1.

---

## Intended and Non-Intended Use

**Intended use:**
- Benchmarking and development of the rule-based eligibility matcher
- Testing matcher behaviour across diverse patient presentations
- Research into eligibility screening logic and uncertainty handling
- Portfolio demonstration of synthetic dataset design

**Non-intended use:**
- Clinical decision support or patient triage
- Medical advice or treatment recommendations
- Any application involving real eligibility decisions for real patients
- Training production models for clinical deployment
