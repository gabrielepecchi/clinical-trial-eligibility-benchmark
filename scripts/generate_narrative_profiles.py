"""
generate_narrative_profiles.py — Task 11: Generate narrative patient profiles.

Reads data/processed/patient_cases_enriched.json, adds a `narrative_profile`
field to each patient generated from existing structured fields only, and
writes data/processed/patient_cases_narrative.json.

Usage:
    PYTHONPATH=. python scripts/generate_narrative_profiles.py
    PYTHONPATH=. python scripts/generate_narrative_profiles.py --input PATH --output PATH
"""

import json
import sys
import argparse

DEFAULT_INPUT = "data/processed/patient_cases_enriched.json"
DEFAULT_OUTPUT = "data/processed/patient_cases_narrative.json"


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_patient_cases(path: str) -> list:
    """Load and return patient cases list from a JSON file."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"ERROR: File not found: {path}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"ERROR: Malformed JSON in {path}: {e}", file=sys.stderr)
        sys.exit(1)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "patients" in data:
        return data["patients"]
    print("ERROR: Unexpected JSON structure; expected a list or dict with 'patients'.", file=sys.stderr)
    sys.exit(1)


def write_patient_cases(patients: list, path: str) -> None:
    """Write the patient list to a JSON file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(patients, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Narrative generator
# ---------------------------------------------------------------------------

def _val(patient: dict, *keys):
    """Return the first non-empty value found among the given keys, or None."""
    for k in keys:
        v = patient.get(k)
        if v is not None and v != "" and v != [] and v is not False:
            return v
    return None


def generate_narrative(patient: dict) -> str:
    """
    Generate a short free-text clinical narrative from existing structured
    fields. No facts are invented; missing fields are omitted or phrased
    generically.
    """
    parts = []

    # --- Opening: age, sex, diagnosis ---
    age = _val(patient, "age")
    sex = _val(patient, "sex", "gender")
    diagnosis = _val(patient, "diagnosis") or "Parkinson disease"
    subtype = _val(patient, "diagnosis_subtype")
    disease_stage = _val(patient, "disease_stage")

    age_str = f"{age}-year-old" if age else "A patient"
    sex_str = ""
    if sex and str(sex).lower() not in ("any", ""):
        sex_str = f" {sex.lower()}"

    diag_str = diagnosis
    if subtype and str(subtype).lower() not in ("", "unknown"):
        diag_str = f"{subtype} {diagnosis}"

    opening = f"A {age_str}{sex_str} with {diag_str}"
    if disease_stage:
        opening += f" ({disease_stage} stage)"
    parts.append(opening + ".")

    # --- Disease duration ---
    duration = _val(patient, "disease_duration_years")
    if duration is not None:
        try:
            d = float(duration)
            years_str = "1 year" if d == 1.0 else f"{d:g} years"
            parts.append(f"Disease duration approximately {years_str}.")
        except (TypeError, ValueError):
            pass

    # --- Motor severity ---
    hy = _val(patient, "hoehn_yahr_stage")
    updrs = _val(patient, "updrs_iii")
    motor_parts = []
    if hy is not None:
        try:
            motor_parts.append(f"Hoehn and Yahr stage {float(hy):g}")
        except (TypeError, ValueError):
            pass
    if updrs is not None:
        try:
            motor_parts.append(f"UPDRS-III score {int(updrs)}")
        except (TypeError, ValueError):
            pass
    if motor_parts:
        parts.append("Motor severity: " + ", ".join(motor_parts) + ".")

    # --- Key features (if present and not already captured) ---
    features = _val(patient, "key_features")
    if isinstance(features, list) and features:
        parts.append("Notable features: " + "; ".join(features) + ".")

    # --- Cognitive status ---
    cognitive = _val(patient, "cognitive_status")
    moca = _val(patient, "moca_score")
    mmse = _val(patient, "mmse_score")
    cog_parts = []
    if cognitive:
        cog_parts.append(str(cognitive))
    if moca is not None:
        try:
            cog_parts.append(f"MoCA {int(moca)}")
        except (TypeError, ValueError):
            pass
    if mmse is not None:
        try:
            cog_parts.append(f"MMSE {int(mmse)}")
        except (TypeError, ValueError):
            pass
    if cog_parts:
        parts.append("Cognitive status: " + ", ".join(cog_parts) + ".")

    # --- DBS / procedure history ---
    dbs = _val(patient, "dbs_history")
    procedures = _val(patient, "procedure_history")
    proc_list = []
    if dbs is True:
        proc_list.append("deep brain stimulation")
    if isinstance(procedures, list):
        for p in procedures:
            label = str(p).lower().strip()
            if not label:
                continue
            is_dbs_entry = label in ("deep brain stimulation", "dbs")
            if is_dbs_entry and dbs is True:
                continue  # already added above
            if label not in proc_list:
                proc_list.append(label)
    if proc_list:
        parts.append("Procedure history: " + "; ".join(proc_list) + ".")

    # --- Medications ---
    meds = _val(patient, "medication_summary", "medications")
    if isinstance(meds, list) and meds:
        parts.append("Current medications: " + "; ".join(str(m) for m in meds) + ".")
    elif isinstance(meds, str) and meds:
        parts.append(f"Current medications: {meds}.")

    # --- Labs ---
    labs = patient.get("labs")
    if isinstance(labs, dict) and labs:
        lab_parts = [f"{k}: {v}" for k, v in labs.items() if v not in (None, "")]
        if lab_parts:
            parts.append("Laboratory values: " + "; ".join(lab_parts) + ".")

    # --- Comorbidities ---
    comorbidities = _val(patient, "comorbidities")
    if isinstance(comorbidities, list) and comorbidities:
        parts.append("Comorbidities: " + ", ".join(comorbidities) + ".")

    # --- Recent trial participation ---
    trial_part = _val(patient, "recent_trial_participation")
    if trial_part is True:
        parts.append("Recent clinical trial participation noted.")

    # --- Exclusion flags ---
    exclusions = _val(patient, "exclusions")
    if isinstance(exclusions, list) and exclusions:
        parts.append("Known exclusions: " + "; ".join(str(e) for e in exclusions) + ".")

    return " ".join(parts)


# ---------------------------------------------------------------------------
# Enrichment and main
# ---------------------------------------------------------------------------

def enrich_with_narrative(patients: list) -> list:
    """Return a new list with `narrative_profile` added to each patient."""
    result = []
    for patient in patients:
        enriched = dict(patient)
        enriched["narrative_profile"] = generate_narrative(patient)
        result.append(enriched)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate narrative patient profiles.")
    parser.add_argument("--input", default=DEFAULT_INPUT, help=f"Input path (default: {DEFAULT_INPUT})")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help=f"Output path (default: {DEFAULT_OUTPUT})")
    args = parser.parse_args()

    patients = load_patient_cases(args.input)
    enriched = enrich_with_narrative(patients)
    write_patient_cases(enriched, args.output)

    print(f"Records read   : {len(patients)}")
    print(f"Records written: {len(enriched)}")
    print(f"Output path    : {args.output}")


if __name__ == "__main__":
    main()
