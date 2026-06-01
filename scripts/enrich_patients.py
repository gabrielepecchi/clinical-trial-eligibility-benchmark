"""
enrich_patients.py — Task 10: Richer synthetic patient fields enrichment.

Reads data/processed/patient_cases.json, adds or normalizes clinical metadata
fields from existing content only, and writes to patient_cases_enriched.json.

Usage:
    PYTHONPATH=. python scripts/enrich_patients.py
    PYTHONPATH=. python scripts/enrich_patients.py --input PATH --output PATH
"""

import json
import re
import sys
import argparse

DEFAULT_INPUT = "data/processed/patient_cases.json"
DEFAULT_OUTPUT = "data/processed/patient_cases_enriched.json"

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
    """Write the enriched patient list to a JSON file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(patients, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Text collection
# ---------------------------------------------------------------------------

def collect_text_values(obj) -> str:
    """
    Recursively collect all string values from a dict/list/str into one
    lowercase string for pattern matching.
    """
    parts = []
    if isinstance(obj, str):
        parts.append(obj.lower())
    elif isinstance(obj, list):
        for item in obj:
            parts.append(collect_text_values(item))
    elif isinstance(obj, dict):
        for v in obj.values():
            parts.append(collect_text_values(v))
    return " ".join(p for p in parts if p)


# ---------------------------------------------------------------------------
# Field extractors — each returns the extracted value or a sentinel
# ---------------------------------------------------------------------------

def _extract_sex(text: str, patient: dict):
    """Return 'male', 'female', or existing value; else empty string."""
    existing = patient.get("sex") or patient.get("gender") or ""
    if existing:
        return str(existing).lower()
    if re.search(r"\b(male|man|men|his\b|he\b)", text):
        return "male"
    if re.search(r"\b(female|woman|women|her\b|she\b)", text):
        return "female"
    return ""


def _extract_diagnosis_subtype(text: str, patient: dict):
    """Return diagnosis subtype string or empty string."""
    existing = patient.get("diagnosis_subtype", "")
    if existing:
        return existing
    if re.search(r"idiopathic\s+park", text):
        return "idiopathic"
    if re.search(r"atypical\s+park", text):
        return "atypical"
    if re.search(r"young.onset|young onset|yopd", text):
        return "young-onset"
    return ""


def _extract_disease_duration(text: str, patient: dict):
    """Return disease duration in years as float, or None."""
    existing = patient.get("disease_duration_years")
    if existing is not None:
        try:
            return float(existing)
        except (TypeError, ValueError):
            pass
    # "X-year history", "diagnosed X years ago", "for X years"
    patterns = [
        r"(\d+(?:\.\d+)?)\s*[-\u2013]?\s*year(?:s)?\s+history",
        r"diagnosed\s+(\d+(?:\.\d+)?)\s+years?\s+ago",
        r"pd\s+for\s+(\d+(?:\.\d+)?)\s+years?",
        r"parkinson(?:'?s?| disease)?\s+for\s+(\d+(?:\.\d+)?)\s+years?",
        r"(\d+(?:\.\d+)?)\s*[-\u2013]?\s*year(?:s)?\s+(?:history\s+of\s+)?(?:idiopathic\s+)?pd",
    ]
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                pass
    return None


def _extract_hoehn_yahr(text: str, patient: dict):
    """Return H&Y stage as float, or None."""
    existing = patient.get("hoehn_yahr_stage")
    if existing is not None:
        try:
            return float(existing)
        except (TypeError, ValueError):
            pass
    m = re.search(r"h(?:oehn)?(?:\s*[&and]+\s*)y(?:ahr)?\s+(?:stage\s+)?(\d(?:\.\d)?)", text)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    m = re.search(r"hy\s*(?:stage\s+)?(\d(?:\.\d)?)", text)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    return None


def _extract_updrs_iii(text: str, patient: dict):
    """Return UPDRS-III score as int, or None."""
    existing = patient.get("updrs_iii")
    if existing is not None:
        try:
            return int(existing)
        except (TypeError, ValueError):
            pass
    m = re.search(r"updrs(?:\s*[-–]?\s*iii|\s+part\s+iii|\s+motor)?\s+(?:score\s+(?:of\s+)?)?(\d+)", text)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            pass
    return None


def _extract_dbs_history(text: str, patient: dict):
    """Return True only if DBS history is explicitly confirmed; else None (unknown)."""
    existing = patient.get("dbs_history")
    if existing is True:
        return True
    # Explicit denial — leave as None (absence not inferred from silence)
    if re.search(r"no\s+(?:prior\s+|history\s+of\s+)?(?:deep\s+brain\s+stim|dbs)", text):
        return None
    if re.search(r"(?:prior|previous|underwent|has)\s+(?:deep\s+brain\s+stim|dbs)", text):
        return True
    if re.search(r"dbs\s+(?:electrode|implant|surgery|device|history)", text):
        return True
    return None


def _extract_cognitive_status(text: str, patient: dict):
    """Return descriptive string or empty string."""
    existing = patient.get("cognitive_status", "")
    if existing:
        return existing
    if re.search(r"\b(dementia|demented)\b", text):
        return "dementia"
    if re.search(r"\b(mci|mild\s+cognitive\s+impairment)\b", text):
        return "mild cognitive impairment"
    if re.search(r"\b(cognitively\s+intact|normal\s+cognition|no\s+cognitive\s+impairment)\b", text):
        return "normal"
    return ""


def _extract_moca(text: str, patient: dict):
    """Return MoCA score as int, or None."""
    existing = patient.get("moca_score")
    if existing is not None:
        try:
            return int(existing)
        except (TypeError, ValueError):
            pass
    m = re.search(r"moca\s+(?:score\s+(?:of\s+)?)?(\d+)", text)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            pass
    return None


def _extract_mmse(text: str, patient: dict):
    """Return MMSE score as int, or None."""
    existing = patient.get("mmse_score")
    if existing is not None:
        try:
            return int(existing)
        except (TypeError, ValueError):
            pass
    m = re.search(r"mmse\s+(?:score\s+(?:of\s+)?)?(\d+)", text)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            pass
    return None


def _extract_comorbidities(text: str, patient: dict) -> list:
    """Return list of mentioned comorbidities; preserve existing list."""
    existing = patient.get("comorbidities")
    if isinstance(existing, list) and existing:
        return existing
    found = []
    _conditions = [
        (r"\bhypertension\b", "hypertension"),
        (r"\bdiabetes\b", "diabetes"),
        (r"\batrial\s+fibrillation\b", "atrial fibrillation"),
        (r"\bheart\s+failure\b", "heart failure"),
        (r"\bcoronary\s+artery\s+disease\b", "coronary artery disease"),
        (r"\bosteoporosis\b", "osteoporosis"),
        (r"\bchronic\s+kidney\s+disease\b", "chronic kidney disease"),
        (r"\bckd\b", "chronic kidney disease"),
        (r"\bcopd\b", "COPD"),
        (r"\bdepression\b", "depression"),
        (r"\banxiety\b", "anxiety"),
        (r"\bpsychosis\b", "psychosis"),
        (r"\barrhythmia\b", "arrhythmia"),
        (r"\bfrailty\b", "frailty"),
        (r"\bfalls?\b", "falls"),
    ]
    seen = set()
    for pattern, label in _conditions:
        if re.search(pattern, text) and label not in seen:
            found.append(label)
            seen.add(label)
    return found


def _extract_recent_trial(text: str, patient: dict):
    """Return True only if recent trial participation explicitly mentioned; else None."""
    existing = patient.get("recent_trial_participation")
    if existing is True:
        return True
    if re.search(r"(?:currently\s+enrolled|participating\s+in|enrolled\s+in)\s+(?:a\s+)?(?:clinical\s+)?trial", text):
        return True
    if re.search(r"recent\s+(?:trial|study)\s+participation", text):
        return True
    return None


def _extract_medication_summary(text: str, patient: dict) -> list:
    """Return list of mentioned medications; preserve existing."""
    existing = patient.get("medication_summary") or patient.get("medications")
    if isinstance(existing, list) and existing:
        return existing
    if isinstance(existing, str) and existing:
        return [existing]
    found = []
    _meds = [
        (r"\blevodopa(?:/carbidopa)?\b", "levodopa"),
        (r"\bcarbidopa\b", "carbidopa"),
        (r"\brasagiline\b", "rasagiline"),
        (r"\bselegiline\b", "selegiline"),
        (r"\bpramipexole\b", "pramipexole"),
        (r"\bropinirole\b", "ropinirole"),
        (r"\brotigotine\b", "rotigotine"),
        (r"\bapomorphine\b", "apomorphine"),
        (r"\bentacapone\b", "entacapone"),
        (r"\btolcapone\b", "tolcapone"),
        (r"\bamantadine\b", "amantadine"),
        (r"\btrihexyphenidyl\b", "trihexyphenidyl"),
        (r"\bclonazepam\b", "clonazepam"),
        (r"\bquetiapine\b", "quetiapine"),
        (r"\brivastigmine\b", "rivastigmine"),
        (r"\bdonepezil\b", "donepezil"),
        (r"\bmao.b\s+inhibitor\b", "MAO-B inhibitor"),
        (r"\bdopamine\s+agonist\b", "dopamine agonist"),
    ]
    seen = set()
    for pattern, label in _meds:
        if re.search(pattern, text) and label not in seen:
            found.append(label)
            seen.add(label)
    return found


def _extract_procedure_history(text: str, patient: dict) -> list:
    """Return list of mentioned procedures; preserve existing."""
    existing = patient.get("procedure_history")
    if isinstance(existing, list) and existing:
        return existing
    found = []
    _procs = [
        (r"\bdeep\s+brain\s+stimulation\b|\bdbs\b", "deep brain stimulation"),
        (r"\bpallidotomy\b", "pallidotomy"),
        (r"\bthalamotomy\b", "thalamotomy"),
        (r"\bfocused\s+ultrasound\b", "focused ultrasound"),
        (r"\bgamma\s+knife\b", "gamma knife"),
        (r"\bduopa\b|\bduodopa\b", "duodopa/duopa infusion"),
    ]
    seen = set()
    for pattern, label in _procs:
        if re.search(pattern, text) and label not in seen:
            found.append(label)
            seen.add(label)
    return found


# ---------------------------------------------------------------------------
# Core enrichment
# ---------------------------------------------------------------------------

def extract_patient_metadata(patient: dict) -> dict:
    """
    Extract all enrichable fields from a patient record.
    Returns a dict of new/normalized field values only.
    """
    text = collect_text_values(patient)

    return {
        "sex": _extract_sex(text, patient),
        "diagnosis_subtype": _extract_diagnosis_subtype(text, patient),
        "disease_duration_years": _extract_disease_duration(text, patient),
        "hoehn_yahr_stage": _extract_hoehn_yahr(text, patient),
        "updrs_iii": _extract_updrs_iii(text, patient),
        "dbs_history": _extract_dbs_history(text, patient),
        "cognitive_status": _extract_cognitive_status(text, patient),
        "moca_score": _extract_moca(text, patient),
        "mmse_score": _extract_mmse(text, patient),
        "comorbidities": _extract_comorbidities(text, patient),
        "recent_trial_participation": _extract_recent_trial(text, patient),
        "medication_summary": _extract_medication_summary(text, patient),
        "procedure_history": _extract_procedure_history(text, patient),
    }


def enrich_patient_case(patient: dict) -> dict:
    """
    Return a new dict with original fields preserved and enriched fields
    added or normalized. Existing non-null structured values take precedence.
    """
    enriched = dict(patient)
    metadata = extract_patient_metadata(patient)
    for field, value in metadata.items():
        # Only write if field is absent or empty in original
        original = patient.get(field)
        if original is None or original == "" or original == [] or original is False:
            enriched[field] = value
        # If original already has a real value, keep it untouched
    return enriched


def enrich_patient_cases(patients: list) -> list:
    """Return a new list of enriched patient dicts."""
    return [enrich_patient_case(p) for p in patients]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Enrich synthetic patient cases with richer metadata.")
    parser.add_argument("--input", default=DEFAULT_INPUT, help=f"Input path (default: {DEFAULT_INPUT})")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help=f"Output path (default: {DEFAULT_OUTPUT})")
    args = parser.parse_args()

    if args.output == args.input:
        print("ERROR: Output path must differ from input path to avoid overwriting source data.", file=sys.stderr)
        sys.exit(1)

    patients = load_patient_cases(args.input)
    enriched = enrich_patient_cases(patients)
    write_patient_cases(enriched, args.output)

    print(f"Records read   : {len(patients)}")
    print(f"Records written: {len(enriched)}")
    print(f"Output path    : {args.output}")


if __name__ == "__main__":
    main()
