"""
Task 90: criterion type co-occurrence analysis.

Usage:
    PYTHONPATH=. python eval/analyze_criterion_co_occurrence.py
"""

import csv
import os
import sys
from collections import defaultdict
from itertools import combinations


INPUT_PATH = "data/processed/criterion_level_results.csv"
REPORT_PATH = "reports/criterion_co_occurrence.md"

TOP_PAIRS = 20
MAX_EXAMPLES = 3

_TYPE_CANDIDATES = [
    "classified_criterion_type",
    "criterion_type",
    "type",
    "category",
]

_INFER_KEYWORDS: list[tuple[str, list[str]]] = [
    ("age",          ["age", "years old", "year-old", "aged", "older", "younger"]),
    ("diagnosis",    ["diagnosis", "diagnosed", "parkinson", "atypical", "malignancy",
                       "cancer", "disease"]),
    ("medication",   ["medication", "drug", "levodopa", "rasagiline", "dopamine",
                       "carbidopa", "amantadine", "mao-b", "comt", "inhibitor",
                       "current use", "concomitant"]),
    ("procedure",    ["surgery", "procedure", "transplant", "infusion pump", "electrode"]),
    ("device",       ["device", "pacemaker", "dbs", "deep brain stimulation", "implanted",
                       "stimulation"]),
    ("cognitive",    ["cognitive", "dementia", "moca", "mmse", "memory",
                       "neuropsychological"]),
    ("severity",     ["severity", "stage", "hoehn", "yahr", "updrs", "mild", "moderate",
                       "severe", "advanced"]),
    ("temporal",     ["within", "prior to", "at least", "days", "weeks", "months",
                       "washout", "history of", "recent", "past", "last", "duration"]),
    ("comorbidity",  ["comorbidity", "cardiac", "renal", "hepatic", "orthostatic",
                       "hypotension", "seizure", "unstable"]),
    ("reproductive", ["pregnancy", "pregnant", "breastfeeding", "lactating",
                       "contraception", "childbearing"]),
    ("lab",          ["creatinine", "hemoglobin", "hematocrit", "bilirubin", "platelet",
                       "albumin", "glucose", "lab", "serum", "blood value"]),
    ("exclusion",    ["exclusion", "exclude", "excluded", "not eligible",
                       "contraindication"]),
    ("inclusion",    ["inclusion", "must have", "required", "eligible if",
                       "must be"]),
]


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_csv_rows(path: str) -> list[dict]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input file not found: '{path}'")
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise ValueError(f"No data rows in '{path}'.")
    return rows


def write_text(text: str, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


# ---------------------------------------------------------------------------
# Field helpers
# ---------------------------------------------------------------------------

def find_first_existing_column(rows: list[dict], candidates: list[str]) -> str | None:
    if not rows:
        return None
    keys = set(rows[0].keys())
    for c in candidates:
        if c in keys:
            return c
    return None


def normalize_type(value: str) -> str:
    return value.strip().lower() if value else ""


def infer_type_from_text(row: dict) -> str:
    text = ""
    for f in ("criterion", "criterion_text", "text"):
        v = row.get(f, "")
        if v:
            text = v.strip().lower()
            break
    if not text:
        return "other"
    for category, keywords in _INFER_KEYWORDS:
        if any(kw in text for kw in keywords):
            return category
    return "other"


_type_col_cache: str | None = None
_type_col_resolved = False


def get_criterion_type(row: dict) -> str:
    for col in _TYPE_CANDIDATES:
        v = row.get(col, "")
        if v and v.strip():
            return normalize_type(v)
    return infer_type_from_text(row)


def pair_key(row: dict) -> tuple[str, str] | None:
    pid = str(row.get("patient_id", "")).strip()
    tid = str(row.get("trial_id", "")).strip()
    if not pid or not tid:
        return None
    return (pid, tid)


# ---------------------------------------------------------------------------
# Grouping
# ---------------------------------------------------------------------------

def group_types_by_pair(rows: list[dict]) -> dict[tuple, set[str]]:
    pair_types: dict[tuple, set[str]] = defaultdict(set)
    for row in rows:
        k = pair_key(row)
        if k is None:
            continue
        ctype = get_criterion_type(row)
        if ctype:
            pair_types[k].add(ctype)
    return dict(pair_types)


# ---------------------------------------------------------------------------
# Co-occurrence
# ---------------------------------------------------------------------------

def compute_co_occurrence_matrix(
    pair_types: dict[tuple, set[str]]
) -> dict[str, dict[str, int]]:
    matrix: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for types in pair_types.values():
        sorted_types = sorted(types)
        for t in sorted_types:
            matrix[t][t] += 1  # self-count = pair frequency
        for a, b in combinations(sorted_types, 2):
            matrix[a][b] += 1
            matrix[b][a] += 1
    return {k: dict(v) for k, v in matrix.items()}


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyze_criterion_co_occurrence(rows: list[dict]) -> dict:
    pair_types = group_types_by_pair(rows)

    all_types: set[str] = set()
    for types in pair_types.values():
        all_types.update(types)
    sorted_types = sorted(all_types)

    type_freq: dict[str, int] = defaultdict(int)
    for types in pair_types.values():
        for t in types:
            type_freq[t] += 1

    pair_size_dist: dict[int, int] = defaultdict(int)
    for types in pair_types.values():
        pair_size_dist[len(types)] += 1

    matrix = compute_co_occurrence_matrix(pair_types)

    # Top co-occurring pairs (off-diagonal)
    pair_counts: dict[tuple[str, str], int] = {}
    for a in sorted_types:
        for b in sorted_types:
            if a < b:
                cnt = matrix.get(a, {}).get(b, 0)
                if cnt:
                    pair_counts[(a, b)] = cnt
    top_pairs = sorted(pair_counts.items(), key=lambda x: -x[1])[:TOP_PAIRS]

    # Examples per top pair
    examples: dict[tuple[str, str], list[dict]] = {}
    for (a, b), _ in top_pairs[:10]:
        ex = []
        for (pid, tid), types in pair_types.items():
            if a in types and b in types and len(ex) < MAX_EXAMPLES:
                ex.append({
                    "patient_id": pid,
                    "trial_id": tid,
                    "types": sorted(types),
                })
        examples[(a, b)] = ex

    return {
        "total_rows": len(rows),
        "total_pairs": len(pair_types),
        "unique_types": sorted_types,
        "type_freq": dict(sorted(type_freq.items(), key=lambda x: -x[1])),
        "pair_size_dist": dict(sorted(pair_size_dist.items())),
        "matrix": matrix,
        "top_pairs": top_pairs,
        "examples": examples,
    }


# ---------------------------------------------------------------------------
# Markdown formatting
# ---------------------------------------------------------------------------

def format_markdown_report(summary: dict) -> str:
    sorted_types = summary["unique_types"]
    lines = [
        "# Criterion Type Co-occurrence Analysis",
        "",
        f"**Total criterion rows:** {summary['total_rows']}  ",
        f"**Total patient-trial pairs:** {summary['total_pairs']}  ",
        f"**Unique criterion types:** {len(sorted_types)}",
        "",
        "---",
        "",
        "### Criterion Type Frequency",
        "",
        "| Criterion Type | Pair Count |",
        "| --- | --- |",
    ]
    for t, cnt in summary["type_freq"].items():
        lines.append(f"| {t} | {cnt} |")
    lines.append("")

    lines += [
        "---", "",
        "### Pair Count by Number of Criterion Types", "",
        "| # Types | # Pairs |", "| --- | --- |",
    ]
    for size, cnt in summary["pair_size_dist"].items():
        lines.append(f"| {size} | {cnt} |")
    lines.append("")

    lines += [
        "---", "",
        f"### Top {TOP_PAIRS} Co-occurring Criterion Type Pairs", "",
        "| Type A | Type B | Pairs |",
        "| --- | --- | --- |",
    ]
    for (a, b), cnt in summary["top_pairs"]:
        lines.append(f"| {a} | {b} | {cnt} |")
    lines.append("")

    # Co-occurrence matrix
    lines += ["---", "", "### Co-occurrence Matrix", "",
              "*(cell = number of patient-trial pairs where both types appear)*", ""]
    header = "| |" + "".join(f" {t} |" for t in sorted_types)
    sep = "| --- |" + " --- |" * len(sorted_types)
    lines += [header, sep]
    for row_type in sorted_types:
        cells = "".join(
            f" {summary['matrix'].get(row_type, {}).get(col_type, 0)} |"
            for col_type in sorted_types
        )
        lines.append(f"| {row_type} |{cells}")
    lines.append("")

    # Examples
    if summary["examples"]:
        lines += ["---", "", "### Examples for Common Co-occurrences", ""]
        for (a, b), ex_list in summary["examples"].items():
            lines.append(f"#### `{a}` + `{b}`")
            lines.append("")
            for e in ex_list:
                type_str = ", ".join(e["types"])
                lines.append(
                    f"- patient={e['patient_id']} · trial={e['trial_id']} · "
                    f"types present: {type_str}"
                )
            lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    try:
        rows = load_csv_rows(INPUT_PATH)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    summary = analyze_criterion_co_occurrence(rows)
    report = format_markdown_report(summary)

    try:
        write_text(report, REPORT_PATH)
    except OSError as exc:
        print(f"ERROR writing report: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Criterion rows  : {summary['total_rows']}")
    print(f"Patient-trial pairs: {summary['total_pairs']}")
    print(f"Unique types    : {len(summary['unique_types'])}")
    print(f"Report          : {REPORT_PATH}")


if __name__ == "__main__":
    main()
