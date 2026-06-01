# Future Work

This document outlines possible research and engineering directions for the
clinical-trial-eligibility-benchmark project. These are **exploratory ideas**,
not completed features or formal commitments.

---

## Current project context

The current project consists of:

- **Synthetic patients**: Fully local generation with no real patient data
- **Public trial criteria**: Extracted from ClinicalTrials.gov or similar public sources
- **LLM-reviewed draft labels**: A best-effort benchmark baseline, not clinician-adjudicated gold labels
- **Rule-based matcher**: Heuristic eligibility classification with known blind spots
- **Audit and reporting scripts**: Analysis tools to understand matcher behavior and label quality

Future work could extend any of these components, but should maintain the
synthetic-data discipline and avoid clinical deployment claims.

---

## Better criterion parsing

The current rule-based matcher may miss complex criteria phrasing. Possible improvements:

- Structured parsing of numeric ranges ("40 to 80 years", "between 20 and 40")
- Temporal duration extraction ("within 30 days", "for at least 6 months", "4-week washout")
- Negation and exclusion detection ("no history of", "must not have", "contraindication")
- Medical synonym normalization (MAO-B inhibitor → rasagiline/selegiline/safinamide)
- Procedure and device terminology (DBS, deep brain stimulation, focused ultrasound)

These would be incremental improvements to rule-based matching, not requiring new labels.

---

## Criterion-level entailment

Reformulate individual criterion evaluation as an entailment task:

**Premise**: Patient profile  
**Hypothesis**: "Patient is eligible for this criterion" (e.g., "Age is 40–80")  
**Label**: `entailed` | `contradicted` | `unknown`

This would provide fine-grained reasoning scores at criterion level rather than
trial level, helping identify which criterion types are hardest.

Requires: New criterion-level labels for each patient-trial pair.

---

## Missing-information detection

For each case marked `unclear`, generate a checklist of what patient information
would be needed to resolve eligibility:

**Output**: `["MoCA score", "medication stability duration", "DBS history"]`

This helps explain unclear predictions and guides data collection.

Requires: Analysis of current results; no new labels needed.

---

## Evidence span extraction

For each criterion, extract the specific patient profile field or text that
supports or refutes eligibility:

**Output**: `{"evidence_field": "medications", "evidence_value": "rasagiline", "criterion": "No current MAO-B inhibitor use", "supports": false}`

This moves toward interpretable retrieval-style workflows.

Requires: New annotation of evidence spans, or heuristic extraction rules.

---

## Error-type prediction

Instead of computing error types only post-hoc, ask a system to predict which
reasoning challenges a case poses:

**Input**: Patient profile, trial criteria  
**Output**: `["numeric_threshold", "temporal_reasoning", "negation"]`

This could guide model design and ensemble strategies.

Requires: Labels associating error types with cases before evaluation.

---

## Rationale generation

Generate natural-language explanations of eligibility decisions:

**Input**: Patient profile, trial criteria, predicted label  
**Output**: "Patient is ineligible because rasagiline (a MAO-B inhibitor) is listed, and the trial excludes current MAO-B inhibitor use."

Quality evaluation requires either ground-truth rationales or heuristic checks
(e.g., mentions relevant criterion? hallucinates? misses exclusions?).

Requires: Either generation model and rationale labels, or robust evaluation heuristics.

---

## Stronger synthetic patient generation

Current synthetic patients are simple JSON profiles. Improvements could include:

- Richer clinical detail (lab values, imaging results, procedure histories)
- Narrative patient notes (clinical vignettes in free text)
- Temporal information (dates of events, durations of conditions)
- Uncertainty markers (approximate values, "last known", "not recently checked")
- Realistic noise (conflicting information, ambiguous phrasing)

This would better test real-world robustness.

Requires: Domain knowledge and careful validation to avoid introducing unrealistic artifacts.

---

## More robust label review

Current labels are LLM-reviewed once. Stronger validation could include:

- Multiple independent LLM reviews with disagreement analysis
- Structured rubrics for disagreement resolution
- Domain expert spot-checks on hard cases
- Cross-check of label consistency within trials or across patients

This would increase label confidence without requiring full expert annotation.

Requires: Repeated label generation and systematic disagreement tracking.

---

## External validation requirements

For a strong benchmark, consider:

- Reproducibility validation: Do alternative implementations of the matcher on the same data produce similar results?
- Generalization testing: Do results on Parkinson trials transfer to other disease domains?
- Stability analysis: Are the benchmark cases robust to small perturbations in patient profile or criteria phrasing?
- Fairness analysis: Does accuracy vary by patient age, gender, or disease severity?

These would strengthen the benchmark's credibility.

Requires: Additional data, validation scripts, or domain expertise.

---

## Safer uncertainty handling

The current `unclear` label is a catch-all. Future work could:

- Categorize unclear cases by reason (missing required field, ambiguous clinical judgment, conflicting information)
- Implement abstention modes (system can refuse to predict, with coverage/accuracy trade-off analysis)
- Distinguish patient-side uncertainty (missing info) from criterion-side uncertainty (ambiguous rule)
- Test decision rules for when uncertain information should propagate to a not-eligible prediction

This would make the `unclear` label more actionable.

Requires: Richer annotation and decision-rule experimentation.

---

## Out of scope for the current project

The following are **not** appropriate directions for this benchmark:

- **Clinical deployment**: This is not a clinical decision-support tool and should never be deployed in patient care
- **Real patient data**: This project is intentionally synthetic to avoid PHI and real-world liability
- **Regulatory claims**: No part of this project should be used to support FDA submissions, clinical trial operations, or regulatory evidence
- **Replacing clinician review**: Eligibility decisions must always involve qualified human reviewers; this benchmark is for research only

Any extension of this work toward production use would require separate clinical validation, IRB review, and regulatory oversight — far beyond the scope of a research benchmark.
