# Benchmark Format

> **Note:** This document describes the intended unified schema for benchmark records.
> It reflects the target format, not necessarily the current state of all data files.
> The dataset is not yet published or submitted to HuggingFace Datasets.

---

## Purpose

This document defines the unified JSON schema for patient–trial eligibility benchmark records. The schema is designed to make individual records self-contained, auditable, and compatible with future upload to HuggingFace Datasets or similar sharing platforms.

---

## Record Schema Overview

Each benchmark record represents a single patient–trial pair and contains five top-level fields:

```json
{
  "patient": { ... },
  "trial": { ... },
  "criteria": { ... },
  "label": "eligible | not_eligible | unclear",
  "metadata": { ... }
}
```

---

## Field Descriptions

### `patient`

A structured synthetic patient profile. Contains demographic, clinical, and medication information relevant to trial eligibility screening.

| Field | Type | Description |
|-------|------|-------------|
| `patient_id` | string | Unique synthetic identifier |
| `age` | integer | Patient age in years |
| `sex` | string | Sex or gender |
| `diagnosis` | string | Primary diagnosis |
| `disease_duration_years` | number | Years since diagnosis |
| `medications` | list of strings | Current medications |
| `dbs_history` | boolean or string | Prior DBS implantation |
| `cognitive_status` | string | Cognitive status description |
| `comorbidities` | list of strings | Active comorbidities |
| `clinical_summary` | string | Free-text clinical narrative (optional) |

Not all fields are required to be present in every record. Missing fields indicate data not documented for that patient.

---

### `trial`

A summary of the clinical trial derived from a public ClinicalTrials.gov record.

| Field | Type | Description |
|-------|------|-------------|
| `trial_id` | string | Unique trial identifier (maps to NCT ID where available) |
| `nct_id` | string | ClinicalTrials.gov NCT identifier |
| `title` | string | Trial title or brief title |
| `phase` | string | Trial phase (e.g. Phase 2, Observational) |
| `status` | string | Recruitment status at time of record extraction |
| `intervention_type` | string | Type of intervention (e.g. Drug, Device, Behavioral) |
| `condition` | string | Target condition (e.g. Parkinson Disease) |

---

### `criteria`

The eligibility criteria text extracted from the trial record.

| Field | Type | Description |
|-------|------|-------------|
| `inclusion_criteria` | string or list | Inclusion criteria text |
| `exclusion_criteria` | string or list | Exclusion criteria text |
| `criteria_text` | string | Combined raw criteria text (fallback) |

At least one of these fields should be present for each record. Records with no criteria text are excluded from the benchmark.

---

### `label`

A string indicating the eligibility decision for this patient–trial pair.

| Value | Meaning |
|-------|---------|
| `eligible` | Patient meets all known inclusion criteria and no exclusion criteria are violated |
| `not_eligible` | At least one required inclusion criterion is unmet or at least one exclusion criterion is violated |
| `unclear` | Eligibility cannot be determined due to missing, ambiguous, or underspecified information |

Labels are draft benchmark labels produced through an LLM-assisted review process. They have not been validated by clinicians.

---

### `metadata`

Provenance and annotation fields for the record.

| Field | Type | Description |
|-------|------|-------------|
| `label_source` | string | How the label was produced (e.g. `llm_reviewed`) |
| `label_status` | string | Confidence or review status of the label |
| `gold_rationale` | string | Free-text explanation of the label decision |
| `difficulty` | string | Optional difficulty tag (e.g. `easy`, `medium`, `hard`, `ambiguous`) |
| `error_type` | string | Error category if the record was mispredicted (populated post-evaluation) |
| `severity` | string | Error severity if applicable (e.g. `critical`, `major`, `minor`) |

---

## Label Values and Definitions

The benchmark uses a three-class label scheme:

- **`eligible`** — the patient satisfies all evaluable inclusion criteria and no exclusion criterion is triggered by the available patient information.
- **`not_eligible`** — the patient fails at least one hard inclusion criterion or triggers at least one hard exclusion criterion.
- **`unclear`** — the available patient information is insufficient to determine eligibility with confidence; the correct answer requires information not present in the profile.

The `unclear` class is intentional and clinically important. It represents genuine epistemic uncertainty rather than a catch-all for difficult cases.

---

## Example Record

The following is a minimal illustrative example. Fields are synthetic and do not correspond to any real patient or trial.

```json
{
  "patient": {
    "patient_id": "P_EXAMPLE",
    "age": 67,
    "sex": "female",
    "diagnosis": "idiopathic Parkinson disease",
    "disease_duration_years": 5,
    "medications": ["levodopa/carbidopa"],
    "dbs_history": false,
    "cognitive_status": "normal",
    "comorbidities": ["hypertension"]
  },
  "trial": {
    "trial_id": "T_EXAMPLE",
    "nct_id": "NCT00000000",
    "title": "Example PD Intervention Study",
    "phase": "Phase 2",
    "status": "Recruiting",
    "intervention_type": "Drug",
    "condition": "Parkinson Disease"
  },
  "criteria": {
    "inclusion_criteria": "Diagnosis of idiopathic Parkinson disease. Age 40 to 80.",
    "exclusion_criteria": "Prior deep brain stimulation. Current MAO-B inhibitor use."
  },
  "label": "eligible",
  "metadata": {
    "label_source": "llm_reviewed",
    "label_status": "draft",
    "gold_rationale": "Patient meets age and diagnosis criteria. No DBS history. No MAO-B inhibitor use.",
    "difficulty": "easy"
  }
}
```

---

## Relationship to Current Data Files

The benchmark currently stores data across separate files rather than as unified records:

| Schema field | Current file |
|--------------|-------------|
| `patient` | `data/processed/patient_cases.json` |
| `trial` | `data/processed/trial_cases.json` |
| `criteria` | embedded in `trial_cases.json` |
| `label` | `data/processed/labels_llm_reviewed.json` |
| `metadata` | `data/processed/labels_llm_reviewed.json` (partial) |

A future export script could merge these files into the unified per-record format described here.

---

## Notes on HuggingFace Datasets Compatibility

The unified schema is designed to be compatible with HuggingFace Datasets conventions:

- Each record is a flat or lightly nested JSON object.
- Label values are a fixed string enum.
- All fields use consistent types across records.
- Missing optional fields use `null` or empty string rather than being omitted inconsistently.

To upload to HuggingFace Datasets, a dataset card (`README.md` with YAML front matter) and a `dataset_infos.json` would also be required. These are not yet prepared.

---

## Known Current Gaps

The following schema fields are not yet fully or consistently populated across all records:

- `phase`, `status`, `intervention_type` — partially present; enrichment script available (`eval/enrich_trial_metadata.py`)
- `difficulty` — not yet assigned systematically
- `clinical_summary` / narrative profile — present for some patients only
- Unified per-record export — not yet implemented; data lives in separate files
