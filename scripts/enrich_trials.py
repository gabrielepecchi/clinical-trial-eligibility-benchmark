"""
enrich_trials.py — Task 12: Eligibility-focused trial metadata enrichment.

Complements enrich_trial_metadata.py (which handles nct_id, title, phase,
status, intervention_type, condition). This script focuses on eligibility
constraints extracted from existing trial content only.

Reads  : data/processed/trial_cases.json
Writes : data/processed/trial_cases_eligibility_enriched.json

Usage:
    PYTHONPATH=. python scripts/enrich_trials.py
    PYTHONPATH=. python scripts/enrich_trials.py --input PATH --output PATH
"""

import json
import re
import sys
import argparse

DEFAULT_INPUT = "data/processed/trial_cases.json"
DEFAULT_OUTPUT = "data/processed/trial_cases_eligibility_enriched.json"


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_trial_cases(path: str) -> list:
    """Load and return trial cases list from a JSON file."""
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
    if isinstance(data, dict) and "trials" in data:
        return data["trials"]
    print("ERROR: Expected a JSON list or dict with 'trials'.", file=sys.stderr)
    sys.exit(1)


def write_trial_cases(trials: list, path: str) -> None:
    """Write the trial list to a JSON file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(trials, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

def collect_text_values(obj) -> str:
    """Recursively collect all string values into one lowercase string."""
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


def extract_criteria_text(trial: dict) -> tuple:
    """
    Return (inclusion_text, exclusion_text, full_text) as lowercase strings.
    Draws from inclusion_criteria, exclusion_criteria, and raw_eligibility.
    """
    def _join(val) -> str:
        if isinstance(val, list):
            return " ".join(str(v) for v in val).lower()
        if isinstance(val, str):
            return val.lower()
        return ""

    inc = _join(trial.get("inclusion_criteria", ""))
    exc = _join(trial.get("exclusion_criteria", ""))
    raw = _join(trial.get("raw_eligibility", ""))
    full = " ".join([inc, exc, raw, collect_text_values(trial)])
    return inc, exc, full


# ---------------------------------------------------------------------------
# Field extractors
# ---------------------------------------------------------------------------

def _parse_age_years(value) -> int | None:
    """Parse '45 Years', 45, '45', etc. to int or None."""
    if value is None or value == "":
        return None
    try:
        return int(float(str(value).split()[0]))
    except (ValueError, IndexError):
        return None


def _extract_min_age(full: str, trial: dict) -> int | None:
    existing = _parse_age_years(
        trial.get("minimum_age") or trial.get("min_age")
    )
    if existing is not None:
        return existing
    patterns = [
        r"(?:age(?:d)?|participants?)\s*(?:≥|>=|at\s+least|minimum\s+of?)\s*(\d+)",
        r"(\d+)\s*years?\s+(?:of\s+age\s+)?or\s+older",
        r"must\s+be\s+(\d+)\s*[-–]\s*\d+\s*years?\s+old",
        r"aged?\s+(\d+)\s*[-–]",
    ]
    for pat in patterns:
        m = re.search(pat, full)
        if m:
            try:
                return int(m.group(1))
            except ValueError:
                pass
    return None


def _extract_max_age(full: str, trial: dict) -> int | None:
    existing = _parse_age_years(
        trial.get("maximum_age") or trial.get("max_age")
    )
    if existing is not None:
        return existing
    patterns = [
        r"(?:age(?:d)?|participants?)\s*(?:≤|<=|no\s+more\s+than|at\s+most|maximum\s+of?)\s*(\d+)",
        r"(\d+)\s*years?\s+(?:of\s+age\s+)?or\s+(?:younger|below|under)",
        r"must\s+be\s+\d+\s*[-–]\s*(\d+)\s*years?\s+old",
        r"aged?\s+\d+\s*[-–]\s*(\d+)",
    ]
    for pat in patterns:
        m = re.search(pat, full)
        if m:
            try:
                return int(m.group(1))
            except ValueError:
                pass
    return None


def _extract_age_unit(trial: dict) -> str:
    """Return 'Years' or empty string from existing age fields."""
    for field in ("minimum_age", "maximum_age"):
        val = trial.get(field, "")
        if isinstance(val, str) and "year" in val.lower():
            return "Years"
    return "Years"  # default for clinical trials


def _extract_sex_restriction(full: str, trial: dict) -> str:
    existing = trial.get("sex", "") or trial.get("sex_restriction", "")
    if existing:
        s = str(existing).upper()
        if s in ("ALL", "BOTH"):
            return "all"
        if s in ("MALE", "M"):
            return "male"
        if s in ("FEMALE", "F"):
            return "female"
    if re.search(r"\bmale(?:s)?\s+only\b|\bmen\s+only\b", full):
        return "male"
    if re.search(r"\bfemale(?:s)?\s+only\b|\bwomen\s+only\b", full):
        return "female"
    if re.search(r"\ball\s+(?:sexes|genders)\b|\bboth\s+(?:sexes|genders)\b", full):
        return "all"
    return ""


def _extract_healthy_volunteers(full: str, trial: dict) -> bool | None:
    existing = trial.get("healthy_volunteers")
    if existing is not None:
        return bool(existing)
    if re.search(r"no\s+healthy\s+volunteers|healthy\s+volunteers?\s+(?:not\s+accepted|excluded|not\s+eligible)", full):
        return False
    if re.search(r"healthy\s+volunteers?\s+(?:accepted|welcome|eligible|allowed|included)", full):
        return True
    return None


def _extract_requires_parkinson(inc: str, full: str) -> bool | None:
    if re.search(r"(?:diagnosis|diagnosed)\s+(?:of\s+)?(?:idiopathic\s+)?(?:parkinson|pd)\b", inc):
        return True
    if re.search(r"parkinson(?:'?s?|\s+disease)?\s+(?:diagnosis|diagnosed|confirmed)", inc):
        return True
    return None


def _extract_excludes_atypical(exc: str) -> bool | None:
    if re.search(r"(?:atypical|secondary)\s+parkinson|parkinson(?:ian)?\s+syndrome", exc):
        return True
    return None


def _extract_excludes_dbs(exc: str) -> bool | None:
    if re.search(r"(?:deep\s+brain\s+stimulat|dbs|implanted\s+deep)", exc):
        return True
    return None


def _extract_requires_dbs(inc: str) -> bool | None:
    if re.search(r"(?:deep\s+brain\s+stimulat|dbs)\s+(?:required|implanted|recipients?|eligible|candidates?)", inc):
        return True
    if re.search(r"(?:requires?|must\s+have)\s+(?:deep\s+brain\s+stimulat|dbs)", inc):
        return True
    return None


def _extract_excludes_pacemaker(exc: str) -> bool | None:
    if re.search(r"pacemaker|cardiac\s+implant|implanted\s+(?:cardiac|metal|electronic)", exc):
        return True
    if re.search(r"mri.incompatible\s+implant|metallic\s+implant", exc):
        return True
    return None


def _extract_medication_exclusions(exc: str) -> list:
    found = []
    _meds = [
        (r"\bmaoi\b|monoamine\s+oxidase\s+inhibitor", "MAO inhibitor"),
        (r"\bclozapine\b", "clozapine"),
        (r"\bhaloperidol\b", "haloperidol"),
        (r"\bantipsychotic", "antipsychotics"),
        (r"\bwarfarin\b", "warfarin"),
        (r"\banticoagulant", "anticoagulants"),
        (r"\binvestigational\s+(?:drug|product|medication|treatment)", "investigational drug"),
        (r"\bimmunosuppressant", "immunosuppressants"),
        (r"\bchemotherap", "chemotherapy"),
        (r"\bcorticosteroid", "corticosteroids"),
    ]
    seen = set()
    for pattern, label in _meds:
        if re.search(pattern, exc) and label not in seen:
            found.append(label)
            seen.add(label)
    return found


def _extract_procedure_exclusions(exc: str) -> list:
    found = []
    _procs = [
        (r"(?:deep\s+brain\s+stimulat|dbs)", "deep brain stimulation"),
        (r"pallidotomy", "pallidotomy"),
        (r"thalamotomy", "thalamotomy"),
        (r"focused\s+ultrasound", "focused ultrasound"),
        (r"brain\s+surgery|neurosurgery|pd.related\s+(?:brain|surgical)", "brain surgery"),
    ]
    seen = set()
    for pattern, label in _procs:
        if re.search(pattern, exc) and label not in seen:
            found.append(label)
            seen.add(label)
    return found


def _extract_cognitive_exclusions(exc: str) -> list:
    found = []
    _cog = [
        (r"\bdementia\b", "dementia"),
        (r"mild\s+cognitive\s+impairment|\bmci\b", "mild cognitive impairment"),
        (r"cognitive\s+impairment|neurocognitive\s+impairment", "cognitive impairment"),
        (r"psychosis|active\s+hallucination", "psychosis/hallucinations"),
        (r"\bschizophrenia\b", "schizophrenia"),
    ]
    seen = set()
    for pattern, label in _cog:
        if re.search(pattern, exc) and label not in seen:
            found.append(label)
            seen.add(label)
    return found


def _extract_required_scores(inc: str) -> list:
    found = []
    _scores = [
        (r"hoehn\s+(?:and|&|y(?:ahr)?)\s+(?:stage\s+)?(?:≥|>=|at\s+least|i+|[1-5])", "Hoehn-Yahr stage"),
        (r"(?:mds.)?updrs\s+(?:part\s+)?iii\s*(?:≥|>=|score\s+(?:of\s+)?(?:≥|>=))?\s*\d+", "UPDRS-III"),
        (r"mmse\s*(?:score)?\s*(?:≥|>=|>|<|≤|<=)\s*\d+", "MMSE"),
        (r"moca\s*(?:score)?\s*(?:≥|>=|>|<|≤|<=)\s*\d+", "MoCA"),
        (r"hamd\s*(?:score)?\s*(?:≥|>=|>|<|≤|<=)\s*\d+", "HAMD"),
        (r"bmi\s*(?:of\s+)?\d+\s*[-–]\s*\d+", "BMI range"),
        (r"body\s+(?:weight|mass)\s+\d+\s*[-–]\s*\d+\s*kg", "weight range"),
    ]
    seen = set()
    for pattern, label in _scores:
        if re.search(pattern, inc) and label not in seen:
            found.append(label)
            seen.add(label)
    return found


def _build_eligibility_summary(trial: dict, meta: dict) -> str:
    """Build a short human-readable eligibility summary from extracted fields."""
    parts = []
    min_a = meta.get("min_age")
    max_a = meta.get("max_age")
    if min_a is not None and max_a is not None:
        parts.append(f"Age {min_a}–{max_a}")
    elif min_a is not None:
        parts.append(f"Age ≥{min_a}")
    elif max_a is not None:
        parts.append(f"Age ≤{max_a}")

    sex = meta.get("sex_restriction", "")
    if sex and sex != "all":
        parts.append(sex.capitalize() + " only")

    if meta.get("requires_parkinson_diagnosis"):
        parts.append("Parkinson diagnosis required")
    if meta.get("excludes_atypical_parkinsonism"):
        parts.append("No atypical parkinsonism")
    if meta.get("excludes_dbs"):
        parts.append("No prior DBS")
    if meta.get("requires_dbs"):
        parts.append("DBS required")
    if meta.get("excludes_pacemaker_or_implant"):
        parts.append("No pacemaker/implant")

    cog = meta.get("cognitive_exclusions", [])
    if cog:
        parts.append("Excludes: " + ", ".join(cog))

    med_exc = meta.get("medication_exclusions", [])
    if med_exc:
        parts.append("Medication exclusions: " + ", ".join(med_exc))

    return "; ".join(parts) if parts else ""


# ---------------------------------------------------------------------------
# Core enrichment
# ---------------------------------------------------------------------------

def extract_trial_eligibility_metadata(trial: dict) -> dict:
    """Extract eligibility-focused metadata from a trial record."""
    inc, exc, full = extract_criteria_text(trial)

    min_age = _extract_min_age(full, trial)
    max_age = _extract_max_age(full, trial)

    meta = {
        "min_age": min_age,
        "max_age": max_age,
        "age_unit": _extract_age_unit(trial),
        "sex_restriction": _extract_sex_restriction(full, trial),
        "accepts_healthy_volunteers": _extract_healthy_volunteers(full, trial),
        "requires_parkinson_diagnosis": _extract_requires_parkinson(inc, full),
        "excludes_atypical_parkinsonism": _extract_excludes_atypical(exc),
        "excludes_dbs": _extract_excludes_dbs(exc),
        "requires_dbs": _extract_requires_dbs(inc),
        "excludes_pacemaker_or_implant": _extract_excludes_pacemaker(exc),
        "medication_exclusions": _extract_medication_exclusions(exc),
        "procedure_exclusions": _extract_procedure_exclusions(exc),
        "cognitive_exclusions": _extract_cognitive_exclusions(exc),
        "required_scores_or_thresholds": _extract_required_scores(inc),
        "eligibility_summary": "",
    }
    meta["eligibility_summary"] = _build_eligibility_summary(trial, meta)
    return meta


def enrich_trial_case(trial: dict) -> dict:
    """Return a new dict with original fields preserved and eligibility fields added."""
    enriched = dict(trial)
    metadata = extract_trial_eligibility_metadata(trial)
    for field, value in metadata.items():
        original = trial.get(field)
        if original is None or original == "" or original == []:
            enriched[field] = value
    return enriched


def enrich_trial_cases(trials: list) -> list:
    """Return a new list of enriched trial dicts."""
    return [enrich_trial_case(t) for t in trials]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Enrich trial cases with eligibility-focused metadata."
    )
    parser.add_argument(
        "--input", default=DEFAULT_INPUT,
        help=f"Input path (default: {DEFAULT_INPUT})"
    )
    parser.add_argument(
        "--output", default=DEFAULT_OUTPUT,
        help=f"Output path (default: {DEFAULT_OUTPUT})"
    )
    args = parser.parse_args()

    if args.output == args.input:
        print("ERROR: Output path must differ from input path.", file=sys.stderr)
        sys.exit(1)

    trials = load_trial_cases(args.input)
    enriched = enrich_trial_cases(trials)
    write_trial_cases(enriched, args.output)

    print(f"Records read   : {len(trials)}")
    print(f"Records written: {len(enriched)}")
    print(f"Output path    : {args.output}")


if __name__ == "__main__":
    main()
