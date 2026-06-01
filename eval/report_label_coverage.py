"""
report_label_coverage.py — Task 36: Label coverage and completeness report.

Checks which patient-trial label records have all required fields populated
and which are missing or incomplete.

Usage:
    PYTHONPATH=. python eval/report_label_coverage.py
    PYTHONPATH=. python eval/report_label_coverage.py --labels PATH --output PATH
"""

import json
import os
import sys
import argparse
from collections import Counter

DEFAULT_LABELS = "data/processed/labels_llm_reviewed.json"
DEFAULT_OUTPUT = "reports/label_coverage_report.md"

REQUIRED_FIELDS = ["patient_id", "trial_id", "label", "label_status", "rationale", "evidence"]
VALID_LABELS = {"eligible", "not_eligible", "unclear"}


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_json(path: str) -> object:
    """Load and return JSON from path; exit non-zero on error."""
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"ERROR: File not found: {path}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"ERROR: Malformed JSON in {path}: {e}", file=sys.stderr)
        sys.exit(1)


def extract_label_records(data) -> list:
    """Return a flat list of label record dicts from the loaded JSON."""
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("labels", "records", "predictions"):
            if key in data and isinstance(data[key], list):
                return data[key]
    print("ERROR: Unexpected JSON structure; expected a list or dict with 'labels'.", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Value presence checks
# ---------------------------------------------------------------------------

def value_present(value) -> bool:
    """Return True if a scalar/string value is non-None and non-empty."""
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip() != ""
    return bool(value)


def evidence_present(value) -> bool:
    """
    Return True if evidence is meaningfully populated:
    - non-empty string
    - non-empty list with at least one non-empty item
    - non-empty dict with at least one non-empty value
    """
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip() != ""
    if isinstance(value, list):
        return any(
            (v.strip() if isinstance(v, str) else v)
            for v in value
        )
    if isinstance(value, dict):
        return any(
            (v.strip() if isinstance(v, str) else v)
            for v in value.values()
        )
    return False


def record_id(record: dict) -> str:
    """Return a human-readable identifier for a record."""
    pid = record.get("patient_id", "?")
    tid = record.get("trial_id", "?")
    return f"{pid}/{tid}"


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyze_label_coverage(records: list) -> dict:
    """
    Analyse coverage of required fields across all label records.
    Returns a summary dict.
    """
    total = len(records)
    label_counts: Counter = Counter()
    valid_label_count = 0
    invalid_label_count = 0

    # Per-field tracking: present list and missing list (up to 10 examples)
    field_present: dict[str, int] = {f: 0 for f in REQUIRED_FIELDS}
    field_missing: dict[str, int] = {f: 0 for f in REQUIRED_FIELDS}
    field_missing_examples: dict[str, list] = {f: [] for f in REQUIRED_FIELDS}

    complete_records = 0
    incomplete_records = []  # list of {id, missing_fields}

    for rec in records:
        label_val = rec.get("label", "")
        label_counts[label_val] += 1
        if label_val in VALID_LABELS:
            valid_label_count += 1
        else:
            invalid_label_count += 1

        missing_fields = []
        for field in REQUIRED_FIELDS:
            raw = rec.get(field)
            if field == "evidence":
                present = evidence_present(raw)
            else:
                present = value_present(raw)

            if present:
                field_present[field] += 1
            else:
                field_missing[field] += 1
                missing_fields.append(field)
                if len(field_missing_examples[field]) < 10:
                    field_missing_examples[field].append(record_id(rec))

        if not missing_fields:
            complete_records += 1
        else:
            incomplete_records.append({
                "id": record_id(rec),
                "missing": missing_fields,
            })

    # Top 10 incomplete examples sorted by number of missing fields descending
    top_incomplete = sorted(incomplete_records, key=lambda x: -len(x["missing"]))[:10]

    return {
        "total": total,
        "complete_records": complete_records,
        "incomplete_records": len(incomplete_records),
        "valid_label_count": valid_label_count,
        "invalid_label_count": invalid_label_count,
        "label_counts": dict(label_counts),
        "field_present": field_present,
        "field_missing": field_missing,
        "field_missing_examples": field_missing_examples,
        "top_incomplete": top_incomplete,
    }


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

def format_markdown_report(summary: dict) -> str:
    total = summary["total"]
    lines = [
        "# Label Coverage and Completeness Report",
        "",
        f"- **Total label records**: {total}",
        f"- **Complete records** (all required fields present): {summary['complete_records']}",
        f"- **Incomplete records** (at least one field missing): {summary['incomplete_records']}",
        f"- **Valid labels** (eligible / not_eligible / unclear): {summary['valid_label_count']}",
        f"- **Invalid or unexpected labels**: {summary['invalid_label_count']}",
        "",
        "---",
        "",
        "## Label Distribution",
        "",
        "| Label | Count |",
        "|---|---:|",
    ]

    for lbl in ["eligible", "not_eligible", "unclear"]:
        lines.append(f"| {lbl} | {summary['label_counts'].get(lbl, 0)} |")
    other_labels = {
        k: v for k, v in summary["label_counts"].items()
        if k not in VALID_LABELS
    }
    for lbl, cnt in sorted(other_labels.items()):
        lines.append(f"| _{lbl or '(empty)'}_ | {cnt} |")

    lines += ["", "---", "", "## Field Coverage", ""]
    lines.append("| Field | Present | Missing | Coverage |")
    lines.append("|---|---:|---:|---:|")

    for field in REQUIRED_FIELDS:
        present = summary["field_present"][field]
        missing = summary["field_missing"][field]
        coverage = present / total if total > 0 else 0.0
        lines.append(f"| {field} | {present} | {missing} | {coverage:.1%} |")

    lines += ["", "---", "", "## Missing Examples by Field", ""]

    for field in REQUIRED_FIELDS:
        missing_count = summary["field_missing"][field]
        if missing_count == 0:
            lines.append(f"### {field}: no missing records")
            lines.append("")
            continue
        examples = summary["field_missing_examples"][field]
        lines.append(f"### {field}: {missing_count} missing record(s)")
        lines.append("")
        lines.append("First examples (patient_id/trial_id):")
        lines.append("")
        for ex in examples:
            lines.append(f"- `{ex}`")
        lines.append("")

    lines += ["---", "", "## Top Incomplete Records", ""]

    if not summary["top_incomplete"]:
        lines.append("All records are complete.")
    else:
        lines.append("| Record (patient/trial) | Missing Fields |")
        lines.append("|---|---|")
        for rec in summary["top_incomplete"]:
            missing_str = ", ".join(rec["missing"])
            lines.append(f"| `{rec['id']}` | {missing_str} |")

    lines.append("")
    return "\n".join(lines)


def write_text(text: str, path: str) -> None:
    """Write text to a file, creating parent directories if needed."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Label coverage and completeness report.")
    parser.add_argument("--labels", default=DEFAULT_LABELS,
                        help=f"Labels JSON path (default: {DEFAULT_LABELS})")
    parser.add_argument("--output", default=DEFAULT_OUTPUT,
                        help=f"Output Markdown path (default: {DEFAULT_OUTPUT})")
    args = parser.parse_args()

    data = load_json(args.labels)
    records = extract_label_records(data)
    summary = analyze_label_coverage(records)
    report = format_markdown_report(summary)
    write_text(report, args.output)

    print(f"Total records      : {summary['total']}")
    print(f"Complete records   : {summary['complete_records']}")
    print(f"Incomplete records : {summary['incomplete_records']}")
    print(f"Report written     : {args.output}")


if __name__ == "__main__":
    main()
