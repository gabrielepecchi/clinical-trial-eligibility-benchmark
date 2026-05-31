# Benchmark Results History

Metrics recorded after each significant change to the matcher or dataset.
Run with: `PYTHONPATH=. python eval/run_llm_reviewed_benchmark.py`

> **Warning:** All results use synthetic patient cases and draft LLM-reviewed labels
> that have not been fully validated. Not for clinical use.

---

## v0.1 — Current baseline

**Label source:** `data/processed/labels_llm_reviewed.json`
**Evaluated pairs:** 150
**Skipped pairs:** 0

| Metric                  | Value  |
|-------------------------|--------|
| Accuracy                | 0.687  |
| Macro F1                | 0.694  |
| F1 — eligible           | 0.667  |
| F1 — not_eligible       | 0.753  |
| F1 — unclear            | 0.661  |
| Total errors            | 47     |
| Critical errors         | 0      |
| Major errors            | 43     |
| Minor errors            | 7      |
| Critical error rate     | 0.000  |
| Major error rate        | 0.287  |
| Minor error rate        | 0.047  |

### Safety & Uncertainty

| Metric                  | Value  |
|-------------------------|--------|
| Total predictions       | 150    |
| Unsafe eligible errors  | 0      |
| Overly conservative errors | 4   |
| Uncertainty errors      | 35     |
| Unclear recall          | 0.545  |
| Unclear precision       | 0.840  |
| Overcommitment rate     | 0.455  |

### Criterion Type Summary

| Type      | Total | Met | Not Met | Unclear | Pair Acc |
|-----------|-------|-----|---------|---------|----------|
| exclusion | 1002  | 13  | 27      | 0       | 0.687    |
| inclusion | 798   | 221 | 11      | 0       | 0.687    |

### Validation

| Source         | Validated | Errors | Warnings |
|----------------|-----------|--------|----------|
| Patient cases  | 20        | 0      | 0        |
| Trial cases    | 60        | 0      | 9        |

### Notes

- Baseline entry filled after running the full benchmark.
- Structural improvements completed before this baseline:
  - Task 3: structured reasoning trace per prediction
  - Task 19: criterion-type aggregate metrics
  - Task 21: navigable HTML report with per-example anchors
  - Task 26: patient cases quality validation script
  - Task 27: trial cases quality validation script
- Decision in place: do not modify `rule_matcher.py` to chase individual benchmark errors.

---

## How to update this file

After any change that affects benchmark metrics, append a new entry:

```
## vX.Y — Short description

**Label source:** data/processed/labels_llm_reviewed.json
**Evaluated pairs:** N
**Skipped pairs:** N

| Metric            | Value  |
|-------------------|--------|
| Accuracy          | 0.000  |
| Macro F1          | 0.000  |
| Total errors      | N      |
| Critical errors   | N      |
| Major errors      | N      |
| Minor errors      | N      |

### Notes
- What changed and why.
```

**Rules:**
- Never edit past entries.
- Only record after running the full benchmark, not after individual unit tests.
- Do not hardcode patient or trial IDs in notes.
- Do not record results from patched or temporary matcher changes.
- Do not invent metric values; run the benchmark and copy the printed output.
