"""
Task 81: analyze trial phase, status, intervention type, and condition distributions.

Usage:
    PYTHONPATH=. python eval/analyze_trial_phases.py
"""

import json
import os
import sys
from collections import defaultdict
from typing import Any


PREFERRED_PATH = "data/processed/trial_cases_enriched.json"
FALLBACK_PATH = "data/processed/trial_cases.json"
REPORT_PATH = "reports/trial_phase_status_analysis.md"


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def choose_input_path(preferred: str, fallback: str) -> str:
    if os.path.exists(preferred):
        return preferred
    if os.path.exists(fallback):
        return fallback
    raise FileNotFoundError(
        f"Neither '{preferred}' nor '{fallback}' found."
    )


def load_trial_cases(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array in '{path}'.")
    return data


def write_text(text: str, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


# ---------------------------------------------------------------------------
# Value normalisation
# ---------------------------------------------------------------------------

def normalize_value(value: Any) -> str:
    """Return a clean string; missing/empty/null → 'unknown'."""
    if value is None:
        return "unknown"
    s = str(value).strip()
    return s if s else "unknown"


# ---------------------------------------------------------------------------
# Counting helpers
# ---------------------------------------------------------------------------

def _get_nested(record: dict, field_names: list[str]) -> list[str]:
    """Try each field name in order; return list of normalized values."""
    for name in field_names:
        raw = record.get(name)
        if raw is None:
            continue
        if isinstance(raw, list):
            return [normalize_value(v) for v in raw] if raw else ["unknown"]
        return [normalize_value(raw)]
    return ["unknown"]


def count_field(records: list[dict], field_names: list[str]) -> dict[str, int]:
    """Count occurrences of each normalized value for the first matching field."""
    counts: dict[str, int] = defaultdict(int)
    for rec in records:
        for val in _get_nested(rec, field_names):
            counts[val] += 1
    return dict(sorted(counts.items(), key=lambda x: -x[1]))


def crosstab(
    records: list[dict],
    row_fields: list[str],
    col_fields: list[str],
) -> dict[str, dict[str, int]]:
    """Build a row × col cross-tabulation."""
    table: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for rec in records:
        row_vals = _get_nested(rec, row_fields)
        col_vals = _get_nested(rec, col_fields)
        for rv in row_vals:
            for cv in col_vals:
                table[rv][cv] += 1
    return {r: dict(cols) for r, cols in table.items()}


# ---------------------------------------------------------------------------
# Markdown formatting
# ---------------------------------------------------------------------------

def format_count_table(title: str, counts: dict[str, int]) -> str:
    lines = [f"### {title}", "", "| Value | Count |", "| --- | --- |"]
    for val, n in counts.items():
        lines.append(f"| {val} | {n} |")
    lines.append("")
    return "\n".join(lines)


def format_crosstab_table(title: str, table: dict[str, dict[str, int]]) -> str:
    all_cols: list[str] = []
    for cols in table.values():
        for c in cols:
            if c not in all_cols:
                all_cols.append(c)
    all_cols.sort()

    header = "| |" + "".join(f" {c} |" for c in all_cols)
    separator = "| --- |" + " --- |" * len(all_cols)

    lines = [f"### {title}", "", header, separator]
    for row in sorted(table.keys()):
        cells = "".join(f" {table[row].get(c, 0)} |" for c in all_cols)
        lines.append(f"| {row} |{cells}")
    lines.append("")
    return "\n".join(lines)


def format_markdown_report(summary: dict) -> str:
    parts = [
        "# Trial Phase & Status Analysis",
        "",
        f"**Input file:** `{summary['input_path']}`  ",
        f"**Total trials:** {summary['total']}",
        "",
        "---",
        "",
        format_count_table("Counts by Phase", summary["by_phase"]),
        format_count_table("Counts by Status", summary["by_status"]),
        format_count_table("Counts by Intervention Type", summary["by_intervention_type"]),
        format_count_table("Counts by Condition", summary["by_condition"]),
        format_crosstab_table("Phase × Status", summary["phase_x_status"]),
        format_crosstab_table("Phase × Intervention Type", summary["phase_x_intervention_type"]),
    ]
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    try:
        input_path = choose_input_path(PREFERRED_PATH, FALLBACK_PATH)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    try:
        records = load_trial_cases(input_path)
    except (json.JSONDecodeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    total = len(records)

    summary = {
        "input_path": input_path,
        "total": total,
        "by_phase": count_field(records, ["phase"]),
        "by_status": count_field(records, ["overall_status", "status"]),
        "by_intervention_type": count_field(records, ["intervention_types", "study_type"]),
        "by_condition": count_field(records, ["condition", "conditions"]),
        "phase_x_status": crosstab(
            records, ["phase"], ["overall_status", "status"]
        ),
        "phase_x_intervention_type": crosstab(
            records, ["phase"], ["intervention_types", "study_type"]
        ),
    }

    report = format_markdown_report(summary)

    try:
        write_text(report, REPORT_PATH)
    except OSError as exc:
        print(f"ERROR writing report: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Input  : {input_path}")
    print(f"Trials : {total}")
    print(f"Report : {REPORT_PATH}")


if __name__ == "__main__":
    main()
