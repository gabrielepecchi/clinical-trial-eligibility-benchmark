"""
eval/run_label_noise_analysis.py

Label noise / duplicate label consistency analysis for the clinical trial
eligibility benchmark.

Usage:
    PYTHONPATH=. python eval/run_label_noise_analysis.py

Reads available label source files from data/processed/ and reports:
- duplicate patient_id + trial_id pairs
- conflicting duplicate labels within the same file
- invalid labels
- records missing required fields

Writes a Markdown report to reports/label_noise_analysis.md.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_LABELS = {"eligible", "not_eligible", "unclear"}

REQUIRED_FIELDS = ("patient_id", "trial_id", "label")

DEFAULT_LABEL_SOURCES = [
    "data/processed/labels_llm_reviewed.json",
    "data/processed/labels_seed.json",
    "data/processed/labels_sample.json",
    "data/processed/labels_reviewed.json",
]

DEFAULT_REPORT_PATH = "reports/label_noise_analysis.md"


# ---------------------------------------------------------------------------
# Pure helper functions
# ---------------------------------------------------------------------------


def load_json(path: str) -> object:
    """Load and return JSON content from *path*. Raises on malformed JSON."""
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def load_label_records(path: str) -> List[dict]:
    """
    Load a label source file and return a list of record dicts.

    Raises ValueError if the top-level structure is not a JSON array or if
    any element is not an object.
    """
    raw = load_json(path)
    if not isinstance(raw, list):
        raise ValueError(
            f"{path}: expected a JSON array at top level, got {type(raw).__name__}"
        )
    for i, record in enumerate(raw):
        if not isinstance(record, dict):
            raise ValueError(f"{path}: element {i} is not a JSON object")
    return raw


def label_pair_key(record: dict) -> Tuple[str, str]:
    """Return the (patient_id, trial_id) key for a record."""
    return (str(record.get("patient_id", "")), str(record.get("trial_id", "")))


def validate_label_record(record: dict) -> List[str]:
    """
    Validate a single label record.

    Returns a list of issue strings (empty list means the record is valid).
    """
    issues: List[str] = []
    for field in REQUIRED_FIELDS:
        if field not in record or record[field] is None or str(record[field]).strip() == "":
            issues.append(f"missing_field:{field}")
    if "label" in record and record["label"] not in VALID_LABELS:
        issues.append(f"invalid_label:{record['label']!r}")
    return issues


def analyze_label_source(path: str, records: List[dict]) -> dict:
    """
    Analyse one label source file and return a summary dict.

    Keys:
        path, total_records, valid_records, invalid_records,
        missing_field_count, invalid_label_count,
        duplicate_pair_count, conflicting_duplicate_pair_count,
        top_conflicts (list of dicts with patient_id, trial_id, labels_seen)
    """
    total = len(records)
    missing_field_count = 0
    invalid_label_count = 0
    invalid_records = 0

    # Map (patient_id, trial_id) -> list of labels seen
    pair_to_labels: Dict[Tuple[str, str], List[str]] = {}

    for record in records:
        issues = validate_label_record(record)
        has_issue = False
        for issue in issues:
            has_issue = True
            if issue.startswith("missing_field:"):
                missing_field_count += 1
            elif issue.startswith("invalid_label:"):
                invalid_label_count += 1
        if has_issue:
            invalid_records += 1

        key = label_pair_key(record)
        label = record.get("label", "")
        if key not in pair_to_labels:
            pair_to_labels[key] = []
        pair_to_labels[key].append(label)

    # Detect duplicates
    duplicate_pair_count = 0
    conflicting_duplicate_pair_count = 0
    top_conflicts = []

    for key, labels in sorted(pair_to_labels.items()):
        if len(labels) > 1:
            duplicate_pair_count += 1
            unique_labels = set(labels)
            if len(unique_labels) > 1:
                conflicting_duplicate_pair_count += 1
                top_conflicts.append(
                    {
                        "patient_id": key[0],
                        "trial_id": key[1],
                        "labels_seen": sorted(unique_labels),
                    }
                )

    top_conflicts = top_conflicts[:10]

    return {
        "path": path,
        "total_records": total,
        "valid_records": total - invalid_records,
        "invalid_records": invalid_records,
        "missing_field_count": missing_field_count,
        "invalid_label_count": invalid_label_count,
        "duplicate_pair_count": duplicate_pair_count,
        "conflicting_duplicate_pair_count": conflicting_duplicate_pair_count,
        "top_conflicts": top_conflicts,
    }


def find_available_label_sources(paths: List[str]) -> List[Tuple[str, List[dict]]]:
    """
    Return (path, records) for each path that exists on disk.

    Raises ValueError for files that are present but structurally invalid.
    """
    available = []
    for path in paths:
        if os.path.isfile(path):
            records = load_label_records(path)
            available.append((path, records))
    return available


def build_label_noise_summary(sources: List[Tuple[str, List[dict]]]) -> dict:
    """
    Build a full noise summary for all available label sources.

    Returns a dict with key 'source_analyses' (list of per-source dicts).
    """
    analyses = [analyze_label_source(path, records) for path, records in sources]
    return {"source_analyses": analyses}


def format_markdown_report(summary: dict) -> str:
    """Render the label noise summary as a Markdown string."""
    lines = [
        "# Label Noise Analysis",
        "",
    ]

    analyses = summary.get("source_analyses", [])

    if not analyses:
        lines.append("_No label source files found._")
        lines.append("")
        return "\n".join(lines)

    for analysis in analyses:
        name = os.path.basename(analysis["path"])
        lines += [
            f"## {name}",
            "",
            f"| Metric | Count |",
            f"|--------|-------|",
            f"| Total records | {analysis['total_records']} |",
            f"| Valid records | {analysis['valid_records']} |",
            f"| Invalid records | {analysis['invalid_records']} |",
            f"| Missing required field (incidents) | {analysis['missing_field_count']} |",
            f"| Invalid label (incidents) | {analysis['invalid_label_count']} |",
            f"| Duplicate pair count | {analysis['duplicate_pair_count']} |",
            f"| Conflicting duplicate pair count | {analysis['conflicting_duplicate_pair_count']} |",
            "",
        ]

        if analysis["top_conflicts"]:
            lines += [
                "**Top conflicting duplicate examples (up to 10):**",
                "",
                "| patient_id | trial_id | labels_seen |",
                "|------------|----------|-------------|",
            ]
            for ex in analysis["top_conflicts"]:
                labels_str = ", ".join(ex["labels_seen"])
                lines.append(
                    f"| {ex['patient_id']} | {ex['trial_id']} | {labels_str} |"
                )
            lines.append("")
        else:
            lines.append("_No conflicting duplicates found._")
            lines.append("")

    return "\n".join(lines)


def write_text(text: str, path: str) -> None:
    """Write *text* to *path*, creating parent directories as needed."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    try:
        sources = find_available_label_sources(DEFAULT_LABEL_SOURCES)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    if not sources:
        print("No label source files found. Nothing to analyse.")
        print("Exiting successfully.")
        sys.exit(0)

    print(f"Label sources found: {len(sources)}")
    for path, records in sources:
        print(f"  {path}  ({len(records)} records)")

    summary = build_label_noise_summary(sources)
    report_text = format_markdown_report(summary)
    write_text(report_text, DEFAULT_REPORT_PATH)

    print()
    for analysis in summary["source_analyses"]:
        name = os.path.basename(analysis["path"])
        print(
            f"  {name}: "
            f"total={analysis['total_records']}  "
            f"invalid={analysis['invalid_records']}  "
            f"duplicates={analysis['duplicate_pair_count']}  "
            f"conflicts={analysis['conflicting_duplicate_pair_count']}"
        )

    print(f"\nReport written to: {DEFAULT_REPORT_PATH}")


if __name__ == "__main__":
    main()
