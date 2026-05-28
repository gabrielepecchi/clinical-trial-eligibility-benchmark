# Curated Examples — Clinical Trial Eligibility Reasoning

> All patients are fully synthetic. This file is for benchmark analysis and AI evaluation only.
> Not intended for clinical decision-making. Labels are draft benchmark labels, not clinical gold standard truth.

---

## Easy Cases

These cases have clear, direct matches or mismatches between the patient profile and a single criterion. A well-functioning model should get these right reliably.

---

### E-1 · Clear Eligible

**Patient:** 45-year-old with confirmed Type 2 diabetes diagnosis, no prior insulin use, eGFR 72.

**Criterion pattern:** Age 30–60 · Confirmed Type 2 diabetes · No prior insulin · eGFR ≥ 60

**Expected label:** `Eligible`

**Rationale:** Patient satisfies all four criteria directly. No inference required.

**Likely failure mode:** None expected. If the model fails, check for negation misreading on "no prior insulin."

---

### E-2 · Clear Ineligible — Age Exclusion

**Patient:** 72-year-old with confirmed hypertension, no comorbidities, stable on lisinopril.

**Criterion pattern:** Age 18–65 · Confirmed hypertension · No concurrent ACE inhibitor use

**Expected label:** `Ineligible`

**Rationale:** Patient is 72, outside the upper age bound. Also currently taking lisinopril, an ACE inhibitor — a second independent exclusion.

**Likely failure mode:** Model may catch one exclusion and stop without checking the second (medication exclusion error).

---

### E-3 · Clear Ineligible — Hard Exclusion

**Patient:** 50-year-old with rheumatoid arthritis, currently taking methotrexate.

**Criterion pattern:** Confirmed RA diagnosis · No concurrent use of disease-modifying antirheumatic drugs (DMARDs)

**Expected label:** `Ineligible`

**Rationale:** Methotrexate is a DMARD. Patient is excluded by a hard medication exclusion.

**Likely failure mode:** Medication exclusion error — model may not map methotrexate to the DMARD drug class.

---

### E-4 · Clear Eligible — Simple Threshold

**Patient:** 58-year-old, BMI 27, non-smoker, fasting glucose 118 mg/dL.

**Criterion pattern:** Age ≥ 40 · BMI 25–35 · Non-smoker · Fasting glucose 100–125 mg/dL

**Expected label:** `Eligible`

**Rationale:** All four values fall within the required ranges. Direct attribute lookup, no inference needed.

**Likely failure mode:** Numeric threshold error if model treats any range as exclusive at the boundary.

---

## Ambiguous / Uncertain Cases

These cases have missing information, ambiguous language, or conditions that cannot be resolved from the patient profile alone. The correct call is `Uncertain`.

---

### A-1 · Missing Lab Value

**Patient:** 61-year-old with chronic kidney disease. Creatinine last recorded 8 months ago at 1.4 mg/dL. No current eGFR on file.

**Criterion pattern:** Adequate renal function · eGFR ≥ 45 at screening

**Expected label:** `Uncertain`

**Rationale:** No current eGFR is available. An 8-month-old creatinine value cannot reliably confirm current renal function meets the threshold.

**Likely failure mode:** Missing information error — model may extrapolate from the old creatinine and assert Eligible or Ineligible overconfidently.

---

### A-2 · Ambiguous Diagnosis Language

**Patient:** 39-year-old, chart note reads "possible early-stage depression." No formal DSM diagnosis recorded.

**Criterion pattern:** Confirmed diagnosis of major depressive disorder (MDD)

**Expected label:** `Uncertain`

**Rationale:** "Possible early-stage depression" does not constitute a confirmed MDD diagnosis. The patient may or may not qualify depending on whether a formal evaluation is completed.

**Likely failure mode:** Model may treat "possible depression" as equivalent to confirmed MDD (negation / qualification handling error).

---

### A-3 · Temporal Gap Unclear

**Patient:** 54-year-old, completed a course of oral corticosteroids "earlier this year." No exact date recorded.

**Criterion pattern:** No systemic corticosteroid use within the past 30 days

**Expected label:** `Uncertain`

**Rationale:** "Earlier this year" is too vague to determine whether 30 days have elapsed. Cannot confirm or rule out the exclusion.

**Likely failure mode:** Temporal condition error — model may assume the washout period has passed and return Eligible.

---

### A-4 · Conflicting Signals

**Patient:** 47-year-old with a prior cancer diagnosis 6 years ago, currently in remission with no active treatment.

**Criterion pattern:** No history of malignancy in the past 5 years · Currently cancer-free

**Expected label:** `Uncertain`

**Rationale:** The diagnosis was 6 years ago, which clears the 5-year lookback — but the profile does not confirm whether the patient is formally declared cancer-free by a physician. One criterion appears met; the other cannot be verified.

**Likely failure mode:** Multi-step inference error — model may conflate "in remission" with "cancer-free" and return Eligible without flagging the unverified condition.

---

## Hard Cases

These cases require multi-step inference, close boundary reasoning, or distinguishing inclusion from exclusion logic under realistic ambiguity.

---

### H-1 · Derived Value Required

**Patient:** 68-year-old female. Serum creatinine 1.6 mg/dL. No eGFR recorded.

**Criterion pattern:** eGFR ≥ 60 required

**Expected label:** `Uncertain`

**Rationale:** No eGFR is directly provided. eGFR cannot be reliably determined from creatinine alone without a validated formula and confirmed inputs. The model should flag the missing value and return Uncertain rather than inferring eligibility from a proxy measure.

**Likely failure mode:** Multi-step inference error — model uses raw creatinine as a direct proxy and asserts Eligible or Ineligible without acknowledging that eGFR has not been measured.

---

### H-2 · Boundary Eligible

**Patient:** 65-year-old male, no comorbidities, meets all other criteria.

**Criterion pattern:** Age 18–65 (inclusive)

**Expected label:** `Eligible`

**Rationale:** Age 65 is at the exact upper boundary. If the criterion is inclusive (≤ 65), the patient qualifies.

**Likely failure mode:** Numeric threshold error — model treats the boundary as exclusive and returns Ineligible.

---

### H-3 · Inclusion / Exclusion Confusion Under Complexity

**Patient:** 52-year-old with confirmed HER2-positive breast cancer, no prior targeted therapy, ECOG score 1.

**Criterion pattern:**
- Inclusion: Confirmed HER2-positive breast cancer · ECOG ≤ 2
- Exclusion: Prior treatment with any HER2-targeted agent

**Expected label:** `Eligible`

**Rationale:** Patient satisfies both inclusion criteria. The exclusion does not apply because the patient has no prior targeted therapy. The model must hold three criteria simultaneously and not conflate the HER2-positive diagnosis with the HER2-targeted therapy exclusion.

**Likely failure mode:** Inclusion/exclusion confusion — model may flag the HER2-positive status as triggering the HER2-targeted therapy exclusion.

---

### H-4 · Chained Temporal + Medication Reasoning

**Patient:** 49-year-old, completed immunotherapy 5 weeks ago, currently taking low-dose aspirin (81 mg) for cardiac prophylaxis.

**Criterion pattern:**
- No immunotherapy within the past 4 weeks
- No concurrent antiplatelet agents

**Expected label:** `Uncertain`

**Rationale:** The immunotherapy washout is satisfied (5 weeks > 4 weeks). Whether the patient is excluded depends on whether the trial protocol defines low-dose prophylactic aspirin as a prohibited antiplatelet agent — not all protocols do. The model must clear the first criterion and flag the aspirin question rather than assuming a definitive exclusion.

**Likely failure mode:** Medication exclusion error — model may either dismiss low-dose aspirin as irrelevant and return Eligible, or treat it as a definitive exclusion and return Ineligible, without acknowledging that the protocol definition is ambiguous.

---

*12 examples total: 4 easy · 4 ambiguous · 4 hard*
*All patients synthetic · Labels are draft benchmark labels · Last updated: 2026-05*
