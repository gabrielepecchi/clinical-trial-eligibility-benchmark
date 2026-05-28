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

## How to Use These Baselines

A well-designed benchmark should show a clear ordering:

```
Majority-class  <  Keyword / rule  <  Simple LLM  <  Full pipeline
```

If that ordering does not hold, it is a signal to investigate — either the benchmark has a class imbalance problem, the pipeline is underperforming, or the easy cases are dominating the score. These baselines are diagnostic tools, not just reference numbers.

---

*Last updated: 2026-05 | Scope: AI benchmark design and evaluation | No clinical validation*
