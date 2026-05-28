# Error Taxonomy — Clinical Trial Eligibility Reasoning

> This taxonomy is for benchmark analysis and AI evaluation purposes only.
> It is not intended for clinical decision-making and has not been clinically validated.
> All patient examples are fully synthetic.

---

## 1. Negation Handling Errors

**Definition:** Model misreads a negated condition as a positive one, or vice versa.

**Why it matters:** Eligibility criteria frequently use negation ("no history of", "must not have"). Misreading negation flips the entire eligibility decision.

**Example pattern:**
- Criterion: *"No prior diagnosis of diabetes"*
- Patient: *Has diabetes*
- Model output: Eligible *(fails to process negation)*

**Review note:** Flag cases where the model's rationale does not explicitly address negated criteria. Check whether the word "no", "not", "without", or "absence of" appears in the criterion but is absent from the model's reasoning.

---

## 2. Numeric Threshold Errors

**Definition:** Model applies a numeric cutoff incorrectly — wrong direction, wrong unit, or off-by-one at a boundary.

**Why it matters:** Many criteria are defined by exact thresholds (age, lab values, BMI, dosage). Small numeric errors directly change eligibility outcomes.

**Example pattern:**
- Criterion: *"Age 18–65"*
- Patient: *Age 65*
- Model output: Ineligible *(treats boundary as exclusive when it is inclusive)*

**Review note:** Check boundary cases carefully. Verify whether the model distinguishes `<` vs `≤` and `>` vs `≥`. Flag cases where patient value equals the threshold exactly.

---

## 3. Temporal Condition Errors

**Definition:** Model ignores or misapplies time-based conditions such as washout periods, recency requirements, or duration of diagnosis.

**Why it matters:** Trial criteria often require conditions to have been present for a minimum duration, or treatments to have ended a minimum number of weeks prior. Missing this changes the eligibility call.

**Example pattern:**
- Criterion: *"No chemotherapy within the past 4 weeks"*
- Patient: *Last chemotherapy 3 weeks ago*
- Model output: Eligible *(ignores recency constraint)*

**Review note:** Flag any criterion containing time words ("within", "prior", "at least X weeks", "duration of"). Verify the model's rationale explicitly computes or references the time gap.

---

## 4. Medication Exclusion Errors

**Definition:** Model fails to identify a patient's current or prior medication as a disqualifying exclusion.

**Why it matters:** Drug interaction and contraindication exclusions are common and safety-critical in trial design. Missing them produces false Eligible calls.

**Example pattern:**
- Criterion: *"No concurrent use of immunosuppressants"*
- Patient: *Currently taking methotrexate*
- Model output: Eligible *(does not identify methotrexate as immunosuppressant)*

**Review note:** Check whether the model correctly maps generic and brand drug names to their drug class. Flag cases where a medication is present in the patient profile but not mentioned in the rationale.

---

## 5. Missing Information / Uncertainty Errors

**Definition:** Model asserts Eligible or Ineligible when required information is absent from the patient profile, rather than returning Uncertain.

**Why it matters:** Overconfident calls on incomplete data are a meaningful failure mode. A well-calibrated model should hedge appropriately.

**Example pattern:**
- Criterion: *"ECOG performance status ≤ 2"*
- Patient profile: *No ECOG score recorded*
- Model output: Eligible *(assumes score is acceptable rather than flagging missing data)*

**Review note:** For each Eligible or Ineligible call, verify the patient profile actually contains values for every criterion the model cites. Flag rationales that assume values not present in the input.

---

## 6. Multi-Step Inference Errors

**Definition:** Model fails when correct eligibility requires chaining two or more reasoning steps rather than a direct attribute lookup.

**Why it matters:** Real eligibility decisions often require inference (e.g. deriving organ function from a lab value, inferring disease stage from treatment history). Single-step models break on these.

**Example pattern:**
- Criterion: *"Adequate renal function (eGFR ≥ 60)"*
- Patient: *Creatinine 1.9 mg/dL, age 72, female*
- Model output: Eligible *(uses raw creatinine rather than computing eGFR)*

**Review note:** Flag criteria that require derivation or inference rather than direct matching. Check whether the model's rationale shows intermediate steps or jumps directly to a conclusion.

---

## 7. Criteria Hallucination Errors

**Definition:** Model cites a criterion in its rationale that does not appear in the actual trial eligibility text.

**Why it matters:** Hallucinated criteria produce rationales that cannot be audited or traced back to the source. This is a direct failure of faithfulness.

**Example pattern:**
- Trial criteria: *Age, diagnosis, prior therapy*
- Model rationale: *"Patient is excluded due to elevated liver enzymes"* *(no such criterion exists in the trial)*

**Review note:** For each criterion cited in the model's rationale, verify it exists verbatim or by clear paraphrase in the original eligibility text. Any citation with no matching source text is a hallucination.

---

## 8. Inclusion / Exclusion Confusion

**Definition:** Model applies an inclusion criterion as if it were an exclusion, or vice versa.

**Why it matters:** Inclusion and exclusion criteria have opposite logical roles. Confusing them inverts the eligibility decision entirely.

**Example pattern:**
- Inclusion criterion: *"Must have confirmed diagnosis of Type 2 diabetes"*
- Patient: *Has Type 2 diabetes*
- Model output: Ineligible *(treats the inclusion requirement as a disqualifying condition)*

**Review note:** Check whether the model's rationale explicitly labels each criterion as inclusion or exclusion before applying it. Flag any case where a satisfied inclusion criterion is cited as a reason for ineligibility, or a triggered exclusion is cited as a reason for eligibility.

---

*This taxonomy is a living document for benchmark development. Categories and examples will expand as evaluation coverage grows.*

*Last updated: 2026-05 | Scope: AI benchmark analysis only | All patient examples: fully synthetic*
