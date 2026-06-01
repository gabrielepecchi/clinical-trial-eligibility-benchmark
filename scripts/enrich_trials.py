"""
enrich_trials.py — Task 12: Trial metadata enrichment.

Reads data/processed/trial_cases.json, adds or normalizes metadata fields
extracted from existing trial content only, and writes
data/processed/trial_cases_enriched.json.

Usage:
    PYTHONPATH=. python scripts/enrich_trials.py
    PYTHONPATH=. python scripts/enrich_trials.py --input PATH --output PATH
"""

import json
import re
import sys
import argparse

DEFAULT_INPUT = "data/processed/trial_cases.json"
DEFAULT_OUTPUT = "data/processed/trial_cases_enriched.json"


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
    print("ERROR: Unexpected JSON structure; expected a list or dict with 'trials'.", file=sys.stderr)
    sys.exit(1)


def write_trial_cases(trials: list, path: str) -> None:
    """Write the trial list to a JSON file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(trials, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Text collection
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


# ---------------------------------------------------------------------------
# Field extractors
# ---------------------------------------------------------------------------

def _extract_phase(text: str, trial: dict):
    """Return trial phase string or empty string."""
    existing = trial.get("phase", "")
    if existing:
        return str(existing)
    m = re.search(r"phase\s+([1-4](?:[a-b])?|i{1,3}v?|iv)", text)
    if m:
        raw = m.group(1).upper()
        mapping = {"I": "1", "II": "2", "III": "3", "IV": "4",
                   "IIA": "2a", "IIB": "2b", "IIIA": "3a", "IIIB": "3b"}
        return mapping.get(raw, raw)
    return ""


def _extract_condition(text: str, trial: dict):
    """Return primary condition string or empty string."""
    existing = trial.get("condition", "") or trial.get("primary_condition", "")
    if existing:
        return str(existing)
    if re.search(r"parkinson", text):
        return "Parkinson disease"
    return ""


def _extract_sponsor_type(text: str, trial: dict):
    """Return 'industry', 'academic', or empty string."""
    existing = trial.get("sponsor_type", "")
    if existing:
        return str(existing)
    if re.search(r"\b(pharma|biotech|inc\.|ltd\.|corp\.|industry|sponsor)\b", text):
        return "industry"
    if re.search(r"\b(university|hospital|academic|institute|nih|nih-funded|foundation)\b", text):
        return "academic"
    return ""


def _extract_min_age(text: str, trial: dict):
    """Return minimum age as int, or None."""
    existing = trial.get("min_age") or trial.get("minimum_age")
    if existing is not None:
        try:
            return int(existing)
        except (TypeError, ValueError):
            pass
    patterns = [
        r"(?:age(?:d)?|participants?)\s+(?:≥|>=|at\s+least|minimum\s+of?)\s+(\d+)",
        r"(\d+)\s*(?:years?\s+(?:of\s+age\s+)?or\s+older)",
        r"minimum\s+age[:\s]+(\d+)",
        r"age\s+range[:\s]+(\d+)\s*[-–]",
    ]
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            try:
                return int(m.group(1))
            except ValueError:
                pass
    return None


def _extract_max_age(text: str, trial: dict):
    """Return maximum age as int, or None."""
    existing = trial.get("max_age") or trial.get("maximum_age")
    if existing is not None:
        try:
            return int(existing)
        except (TypeError, ValueError):
            pass
    patterns = [
        r"(?:age(?:d)?|participants?)\s+(?:≤|<=|no\s+more\s+than|at\s+most|maximum\s+of?)\s+(\d+)",
        r"(\d+)\s*(?:years?\s+(?:of\s+age\s+)?or\s+(?:younger|below|under))",
        r"maximum\s+age[:\s]+(\d+)",
        r"age\s+range[:\s]+\d+\s*[-–]\s*(\d+)",
        r"aged?\s+\d+\s*[-–]\s*(\d+)",
    ]
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            try:
                return int(m.group(1))
            except ValueError:
                pass
    return None


def _extract_sex_restriction(text: str, trial: dict):
    """Return 'male', 'female', 'all', or empty string."""
    existing = trial.get("sex_restriction", "") or trial.get("sex", "")
    if existing and str(existing).lower() not in ("any", ""):
        return str(existing).lower()
    if re.search(r"\bmale(?:s)?\s+only\b|\bmen\s+only\b", text):
        return "male"
    if re.search(r"\bfemale(?:s)?\s+only\b|\bwomen\s+only\b", text):
        return "female"
    if re.search(r"\ball\s+(?:sexes|genders)\b|\bboth\s+(?:sexes|genders)\b", text):
        return "all"
    return ""


def _extract_healthy_volunteers(text: str, trial: dict):
    """Return True if healthy volunteers accepted, False if excluded, None if unknown."""
    existing = trial.get("healthy_volunteers")
    if existing is not None:
        return existing
    if re.search(r"no\s+healthy\s+volunteers|healthy\s+volunteers?\s+(?:not\s+accepted|excluded|not\s+eligible)", text):
        return False
    if re.search(r"healthy\s+volunteers?\s+(?:accepted|welcome|eligible|allowed|included)", text):
        return True
    return None


def _extract_intervention_type(text: str, trial: dict) -> list:
    """Return list of intervention types found in text."""
    existing = trial.get("intervention_types") or trial.get("intervention_type")
    if isinstance(existing, list) and existing:
        return existing
    if isinstance(existing, str) and existing:
        return [existing]
    found = []
    _types = [
        (r"\b(drug|medication|pharmacolog)", "drug"),
        (r"\b(device|implant|stimulat)", "device"),
        (r"\b(behavioral|exercise|physical\s+therapy|rehabilitation)", "behavioral"),
        (r"\b(surgical|surgery|procedure)", "surgical"),
        (r"\b(gene\s+therapy|genetic)", "gene therapy"),
        (r"\b(stem\s+cell|cell\s+therapy)", "cell therapy"),
        (r"\b(dietary|supplement|nutritional)", "dietary supplement"),
    ]
    seen = set()
    for pattern, label in _types:
        if re.search(pattern, text) and label not in seen:
            found.append(label)
            seen.add(label)
    return found


def _extract_primary_outcome(text: str, trial: dict):
    """Return primary outcome string or empty string."""
    existing = trial.get("primary_outcome", "")
    if existing:
        return str(existing)
    m = re.search(r"primary\s+(?:outcome|endpoint)[:\s]+([^.]{5,80})", text)
    if m:
        return m.group(1).strip()
    return ""


# ---------------------------------------------------------------------------
# Core enrichment
# ---------------------------------------------------------------------------

def extract_trial_metadata(trial: dict) -> dict:
    """Extract all enrichable fields from a trial record."""
    text = collect_text_values(trial)
    return {
        "phase": _extract_phase(text, trial),
        "condition": _extract_condition(text, trial),
        "sponsor_type": _extract_sponsor_type(text, trial),
        "min_age": _extract_min_age(text, trial),
        "max_age": _extract_max_age(text, trial),
        "sex_restriction": _extract_sex_restriction(text, trial),
        "healthy_volunteers": _extract_healthy_volunteers(text, trial),
        "intervention_types": _extract_intervention_type(text, trial),
        "primary_outcome": _extract_primary_outcome(text, trial),
    }


def enrich_trial_case(trial: dict) -> dict:
    """Return a new dict with original fields preserved and enriched fields added."""
    enriched = dict(trial)
    metadata = extract_trial_metadata(trial)
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
    parser = argparse.ArgumentParser(description="Enrich trial cases with normalized metadata.")
    parser.add_argument("--input", default=DEFAULT_INPUT, help=f"Input path (default: {DEFAULT_INPUT})")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help=f"Output path (default: {DEFAULT_OUTPUT})")
    args = parser.parse_args()

    if args.output == args.input:
        print("ERROR: Output path must differ from input path to avoid overwriting source data.", file=sys.stderr)
        sys.exit(1)

    trials = load_trial_cases(args.input)
    enriched = enrich_trial_cases(trials)
    write_trial_cases(enriched, args.output)

    print(f"Records read   : {len(trials)}")
    print(f"Records written: {len(enriched)}")
    print(f"Output path    : {args.output}")


if __name__ == "__main__":
    main()
