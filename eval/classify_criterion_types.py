"""
classify_criterion_types.py — Task 14: Criterion-type classifier.

Reads data/processed/criterion_level_results.csv, classifies each criterion
by type using deterministic keyword/regex rules, and writes
data/processed/criterion_type_classified.csv with a new column:
classified_criterion_type.

Usage:
    PYTHONPATH=. python eval/classify_criterion_types.py
    PYTHONPATH=. python eval/classify_criterion_types.py --input PATH --output PATH
"""

import csv
import re
import sys
import argparse
from collections import Counter

DEFAULT_INPUT = "data/processed/criterion_level_results.csv"
DEFAULT_OUTPUT = "data/processed/criterion_type_classified.csv"

# ---------------------------------------------------------------------------
# Classification rules — ordered by priority (first match wins).
# Each entry: (type_label, compiled_pattern)
# ---------------------------------------------------------------------------

CLASSIFICATION_PRIORITY = [
    "age",
    "reproductive",
    "cognitive",
    "device",
    "procedure",
    "temporal",
    "lab",
    "medication",
    "severity",
    "diagnosis",
    "safety",
    "administrative",
    "other",
]

_RAW_RULES = [
    ("age", [
        r"\bage\b",
        r"\byears?\s+of\s+age\b",
        r"\byears?\s+old\b",
        r"\bminimum\s+age\b",
        r"\bmaximum\s+age\b",
        r"\baged?\s+\d+",
        r"\b\d+\s*[-–]\s*\d+\s+years?\b",
    ]),
    ("reproductive", [
        r"\bpregnancy\b",
        r"\bpregnant\b",
        r"\bbreastfeed",
        r"\bnursing\b",
        r"\blactati",
        r"\bcontraception\b",
        r"\bcontraceptive\b",
        r"\bfertility\b",
        r"\bfertile\b",
        r"\bchildbearing\b",
    ]),
    ("cognitive", [
        r"\bmmse\b",
        r"\bmoca\b",
        r"\bdementia\b",
        r"\bdemented\b",
        r"\bcognitive\s+impairment\b",
        r"\bneurocognitive\b",
        r"\bcognitive\s+decline\b",
        r"\bcognitively\s+intact\b",
        r"\bmemory\s+(?:impairment|disorder|loss)\b",
        r"\bpsychosis\b",
        r"\bhallucination",
        r"\bconfusion\b",
    ]),
    ("device", [
        r"\bpacemaker\b",
        r"\bcardiac\s+(?:device|implant)\b",
        r"\bimplanted\s+(?:metal|electronic|cardiac|neural)\b",
        r"\bmri.incompatible\b",
        r"\bmetallic\s+implant\b",
        r"\bimplantable\s+(?:device|pulse\s+generator)\b",
        r"\bcochlear\s+implant\b",
        r"\bneuromodulation\s+device\b",
    ]),
    ("procedure", [
        r"\bdeep\s+brain\s+stimulat",
        r"\bdbs\b",
        r"\bpallidotomy\b",
        r"\bthalamotomy\b",
        r"\bfocused\s+ultrasound\b",
        r"\bgamma\s+knife\b",
        r"\bbrain\s+surgery\b",
        r"\bneurosurgery\b",
        r"\bsurgical\b",
        r"\bsurgery\b",
        r"\bimplant(?:ation|ed)?\b",
        r"\bablation\b",
        r"\bduopa\b",
        r"\bduodopa\b",
        r"\bapomorphine\s+pump\b",
    ]),
    ("temporal", [
        r"\bwithin\s+\d+\s*(?:day|week|month|year)",
        r"\bprior\s+(?:\w+\s+){0,3}(?:treatment|use|therapy|enrollment|participation)",
        r"\brecent(?:ly)?\b",
        r"\bwashout\b",
        r"\bhistory\s+of\b",
        r"\bdiagnosed\s+(?:more\s+than|at\s+least|within)\b",
        r"\bcourse\s+of\s+disease\b",
        r"\bduration\s+of\b",
        r"\bat\s+least\s+\d+\s*(?:day|week|month|year)",
        r"\bsince\s+(?:diagnosis|onset)\b",
        r"\bonset\b",
    ]),
    ("lab", [
        r"\bhemoglobin\b",
        r"\bhaemoglobin\b",
        r"\bcreatinine\b",
        r"\balt\b",
        r"\bast\b",
        r"\balbumin\b",
        r"\bbilirubin\b",
        r"\bplatelet\b",
        r"\bneutrophil\b",
        r"\beGFR\b",
        r"\bglomerular\s+filtration\b",
        r"\bliver\s+function\b",
        r"\brenal\s+function\b",
        r"\bhematolog",
        r"\blaboratory\b",
        r"\bblood\s+(?:test|count|pressure|glucose)\b",
        r"\bvital\s+signs\b",
        r"\bbmi\b",
        r"\bbody\s+mass\s+index\b",
        r"\bbody\s+weight\b",
        r"\bweight\b.*\bkg\b",
    ]),
    ("medication", [
        r"\blevodopa\b",
        r"\bcarbidopa\b",
        r"\brasagiline\b",
        r"\bselegiline\b",
        r"\bpramipexole\b",
        r"\bropinirole\b",
        r"\brotigotine\b",
        r"\bamantadine\b",
        r"\bentacapone\b",
        r"\btolcapone\b",
        r"\bapomorphine\b",
        r"\bclozapine\b",
        r"\bhaloperidol\b",
        r"\bwarfarin\b",
        r"\banticoagulant\b",
        r"\bantipsychotic\b",
        r"\bmaoi?\b",
        r"\bmao.b\b",
        r"\bdopamine\s+agonist\b",
        r"\binvestigational\s+(?:drug|product|medication|treatment)\b",
        r"\bdrug\b",
        r"\bmedication\b",
        r"\bpharmacolog",
        r"\btreatment\s+(?:with|using)\b",
        r"\btherapy\b",
        r"\bwashout\b",
    ]),
    ("severity", [
        r"\bhoehn\s+(?:and|&|y(?:ahr)?)\b",
        r"\bhy\s+stage\b",
        r"\bupdrs\b",
        r"\bmotor\s+fluctuation",
        r"\bdyskinesia\b",
        r"\boff\s+period\b",
        r"\bmotor\s+stage\b",
        r"\bstage\s+[1-5i]+\b",
        r"\bdisease\s+stage\b",
        r"\bseverity\b",
        r"\bmild\b|\bmoderate\b|\badvanced\b",
    ]),
    ("diagnosis", [
        r"\bparkinson(?:'?s?)?\s+disease\b",
        r"\bidiopathic\s+pd\b",
        r"\bidiopathic\s+parkinson",
        r"\batypical\s+parkinson",
        r"\bpd\s+diagnosis\b",
        r"\bdiagnosis\s+of\s+(?:idiopathic\s+)?(?:parkinson|pd)\b",
        r"\bdiagnosed\s+with\s+(?:parkinson|pd)\b",
        r"\bneurological\s+disease\b",
        r"\bprimary\s+diagnosis\b",
        r"\bclinically\s+diagnosed\b",
    ]),
    ("safety", [
        r"\bcardiovascular\b",
        r"\bcardiac\b",
        r"\bunstable\b",
        r"\bmalignancy\b",
        r"\bcancer\b",
        r"\btumor\b",
        r"\binfection\b",
        r"\binfectious\b",
        r"\bcontraindication\b",
        r"\ballerg",
        r"\bhypersensitivit",
        r"\bseizure\b",
        r"\bstroke\b",
        r"\bsuicid",
        r"\bsubstance\s+abuse\b",
        r"\balcohol\s+(?:abuse|dependence|use\s+disorder)\b",
        r"\bliver\s+disease\b",
        r"\brenal\s+(?:failure|disease|impairment)\b",
        r"\bpulmonary\b",
        r"\brespiratory\b",
        r"\bimmunosuppress",
    ]),
    ("administrative", [
        r"\binformed\s+consent\b",
        r"\bcaRegiver\b",
        r"\bcaregiver\b",
        r"\bability\s+to\s+comply\b",
        r"\bcomplian",
        r"\bfollow.up\b",
        r"\blanguage\b",
        r"\bliteracy\b",
        r"\bwifi\b",
        r"\binternet\b",
        r"\bwireless\s+network\b",
        r"\bability\s+to\s+(?:read|understand|sign)\b",
        r"\bvoluntari",
        r"\bwilling\b",
        r"\bable\s+to\b",
    ]),
]

# Compile all patterns once at module load
_COMPILED_RULES: list[tuple[str, list]] = [
    (label, [re.compile(pat, re.IGNORECASE) for pat in patterns])
    for label, patterns in _RAW_RULES
]

# Build lookup: label -> compiled patterns (for classify function)
_RULES_BY_LABEL: dict[str, list] = {label: pats for label, pats in _COMPILED_RULES}


# ---------------------------------------------------------------------------
# Core classifier
# ---------------------------------------------------------------------------

def classify_criterion_text(text: str) -> str:
    """
    Classify a criterion string into one of the supported types.
    Uses CLASSIFICATION_PRIORITY order; first matching category wins.
    Returns 'other' if no pattern matches.
    """
    if not text or not text.strip():
        return "other"
    for label in CLASSIFICATION_PRIORITY:
        patterns = _RULES_BY_LABEL.get(label, [])
        for pat in patterns:
            if pat.search(text):
                return label
    return "other"


# ---------------------------------------------------------------------------
# CSV I/O
# ---------------------------------------------------------------------------

def load_criterion_rows(path: str) -> tuple:
    """
    Load CSV rows from path.
    Returns (fieldnames: list, rows: list[dict]).
    Exits with error if file is missing or malformed.
    """
    try:
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            if not fieldnames:
                print(f"ERROR: No columns found in {path}", file=sys.stderr)
                sys.exit(1)
            rows = list(reader)
    except FileNotFoundError:
        print(f"ERROR: File not found: {path}", file=sys.stderr)
        sys.exit(1)
    except csv.Error as e:
        print(f"ERROR: CSV error in {path}: {e}", file=sys.stderr)
        sys.exit(1)
    return list(fieldnames), rows


def classify_rows(rows: list) -> list:
    """
    Return a new list of row dicts with classified_criterion_type added.
    Uses the 'criterion' column for classification text.
    """
    result = []
    for row in rows:
        new_row = dict(row)
        text = row.get("criterion", "") or ""
        new_row["classified_criterion_type"] = classify_criterion_text(text)
        result.append(new_row)
    return result


def write_classified_rows(rows: list, path: str, fieldnames: list) -> None:
    """Write classified rows to a CSV file."""
    out_fields = list(fieldnames)
    if "classified_criterion_type" not in out_fields:
        out_fields.append("classified_criterion_type")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=out_fields)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Classify criterion types in benchmark results.")
    parser.add_argument(
        "--input", default=DEFAULT_INPUT,
        help=f"Input CSV path (default: {DEFAULT_INPUT})"
    )
    parser.add_argument(
        "--output", default=DEFAULT_OUTPUT,
        help=f"Output CSV path (default: {DEFAULT_OUTPUT})"
    )
    args = parser.parse_args()

    fieldnames, rows = load_criterion_rows(args.input)
    classified = classify_rows(rows)
    write_classified_rows(classified, args.output, fieldnames)

    counts = Counter(r["classified_criterion_type"] for r in classified)
    print(f"Rows read      : {len(rows)}")
    print(f"Rows written   : {len(classified)}")
    print(f"Output path    : {args.output}")
    print("")
    print("Counts by classified_criterion_type:")
    for label in CLASSIFICATION_PRIORITY:
        count = counts.get(label, 0)
        if count > 0:
            print(f"  {label:<16}: {count}")
    if counts.get("other", 0) > 0:
        print(f"  {'other':<16}: {counts['other']}")


if __name__ == "__main__":
    main()
