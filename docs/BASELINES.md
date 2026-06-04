# Baselines — Clinical Trial Eligibility Reasoning

> Baselines provide reference points for interpreting benchmark scores.
> No results are reported here until each baseline has been run.
> This file is for AI evaluation and benchmark design purposes only. Not for clinical decision-making.

---

## 1. Majority-Class Baseline

**What it does:** Predicts the most frequent label in the benchmark dataset for every input, regardless of the patient profile or trial criteria.

**Why it is useful:** Establishes the floor. Any model that does not clearly beat this baseline is not learning anything meaningful from the input. It also reveals class imbalance in the benchmark — if one label dominates, a trivial predictor scores deceptively high.

**Expected weakness:** Completely ignores all reasoning. Will fail every case where the correct label is not the majority class. Produces zero criterion-level rationale.

**What it helps compare against:** The main pipeline model. If the pipeline score is close to the majority-class score, the model is not generalizing — it may be defaulting to a dominant label rather than reasoning about criteria.

**Results:** *To be added after running the baseline on the benchmark dataset.*

---

## 2. Keyword / Rule Baseline

**What it does:** Applies a small set of hand-written rules to the patient profile and criterion text — for example, flagging Ineligible if any exclusion keyword (e.g. "pregnancy", "prior chemotherapy") is found in the patient profile, and Eligible otherwise.

**Why it is useful:** Tests whether surface-level string matching can achieve competitive accuracy without any real reasoning. If a simple keyword pass performs nearly as well as the LLM pipeline, the benchmark may not be testing reasoning depth effectively.

**Expected weakness:** Brittle on negation, numeric thresholds, and temporal conditions. Cannot handle cases where a keyword appears in context that changes its meaning (e.g. "no prior chemotherapy"). Will miss multi-step inferences entirely.

**What it helps compare against:** The LLM pipeline on ambiguous and hard cases specifically. A strong pipeline should outperform keyword matching most clearly on the cases where surface matching breaks down.

**Results:** *To be added after running the baseline on the benchmark dataset.*

---

## 3. Simple LLM Prompting Baseline

**What it does:** Sends the patient profile and eligibility criteria to the model with a minimal zero-shot prompt — no chain-of-thought instruction, no output format specification, no examples. Records whatever label the model produces.

**Why it is useful:** Isolates the contribution of prompt engineering and pipeline design. Comparing this against the full pipeline shows how much of the performance gain comes from structured prompting, output constraints, and rationale elicitation rather than raw model capability.

**Expected weakness:** Inconsistent output format, higher hallucination rate, weaker handling of ambiguity and the Uncertain label. The model may conflate inclusion and exclusion logic without structured guidance.

**What it helps compare against:** The full structured pipeline. Quantifies the value of prompt design as an engineering decision — a core AI evaluation skill.

**Results:** *To be added after running the baseline on the benchmark dataset.*

---

---

## 4. Strict Missing → Unclear Baseline

**What it does:** Predicts `unclear` for any record that contains structured missingness signals — non-empty `unknown_fields`, `missing_information`, `uncertain_criteria`, `missing_reason_type`, or any `missing_information_details` entry with `status == "unknown"`. Predicts `eligible` otherwise.

**Why it is useful:** Tests the most conservative missing-information policy: any gap in the patient record is treated as a reason to withhold a confident prediction. Quantifies how much of the benchmark is affected by incomplete data.

**Expected behavior:** Will produce a high rate of `unclear` predictions wherever the matcher flags missing data. Accuracy will depend heavily on what fraction of gold labels are also `unclear`. Over-predicts `unclear` on cases where the correct label is `eligible` or `not_eligible` despite partial missing information.

**What it helps compare against:** The full pipeline and the conservative policy baseline. Shows whether the pipeline is being more decisive than warranted given data gaps.

**Note:** This is a diagnostic policy baseline for evaluating how incomplete clinical information is handled. It is **not a clinical recommendation**.

**Results:** *To be added after running the baseline on the benchmark dataset.*

---

## 5. Optimistic Missing → Eligible Baseline

**What it does:** Predicts `not_eligible` only when clear blocking evidence exists (`blocking_criteria` or `blocked_by` non-empty). Otherwise predicts `eligible` even when structured missingness signals are present — missing information is ignored in favor of an optimistic assumption.

**Why it is useful:** Represents the most permissive policy. Any patient without an explicit hard block is assumed eligible. Useful for estimating false-positive eligibility rates and understanding how often missing data leads the pipeline to unnecessary caution.

**Expected behavior:** High recall for `eligible`, high false-positive rate overall. Will miss cases where missingness is the correct reason to withhold eligibility.

**What it helps compare against:** The strict and conservative policy baselines. Shows the cost of optimistic assumptions when data is incomplete.

**Note:** This is a diagnostic policy baseline. It is **not a clinical recommendation**.

**Results:** *To be added after running the baseline on the benchmark dataset.*

---

## 6. Conservative Missing → Unclear or Not Eligible Baseline

**What it does:** A three-tier policy. If blocking evidence exists, predicts `not_eligible`. Else if structured missingness signals are present, predicts `unclear`. Otherwise predicts `eligible`. Separates hard blocks from soft uncertainty.

**Why it is useful:** Represents a balanced missing-information policy that distinguishes between definite disqualification and informational gaps. Tests whether separating blocking from missing yields better calibration than the strict or optimistic alternatives.

**Expected behavior:** Better precision/recall balance than either strict or optimistic alone. Most useful for understanding how often the pipeline's `unclear` predictions are driven by missingness versus genuine ambiguity.

**What it helps compare against:** The full pipeline. If the pipeline does not clearly outperform this simple three-tier policy, it suggests the pipeline is not adding reasoning value beyond blocking detection and missingness flagging.

**Note:** This is a diagnostic policy baseline. It is **not a clinical recommendation**.

**Results:** *To be added after running the baseline on the benchmark dataset.*

---

## How to Use These Baselines

A well-designed benchmark should show a clear ordering:

```
Majority-class  <  Keyword / rule  <  Simple LLM  <  Full pipeline
```

For missing-information policy baselines, the expected ordering is:

```
Optimistic  ≤  Conservative  ≤  Strict  ≤  Full pipeline
```

If that ordering does not hold, it is a signal to investigate — either the benchmark has a class imbalance problem, the pipeline is underperforming, or the easy cases are dominating the score. These baselines are diagnostic tools, not just reference numbers.

The missing-policy baselines (`strict_missing_unclear`, `optimistic_missing_eligible`, `conservative_missing_unclear_or_not_eligible`) require structured prediction records with fields such as `unknown_fields`, `blocking_criteria`, and `missing_information_details`. When those records are not available, all three fall back to predicting `eligible` for all records. They are **not clinical decision rules**.

---

*Last updated: 2026-05 | Scope: AI benchmark design and evaluation | No clinical validation*
