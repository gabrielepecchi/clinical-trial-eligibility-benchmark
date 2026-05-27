"""Convert reviewed label CSV rows into labels.json.

This script expects the reviewer to edit data/processed/labels_seed_review.csv.
Rows with label_status set to "reviewed" are exported to labels.json.
"""

import csv
import json
from pathlib import Path

INPUT_FILE = Path("data/processed/labels_seed_review.csv")
OUTPUT_FILE = Path("data/processed/labels.json")

VALID_LABELS = {"eligible", "not_eligible", "unclear"}
REVIEWED_STATUS = "reviewed"


def split_pipe_text(value: str) -> list[str]:
    """Split a pipe-separated CSV cell into a list of strings."""
    if not value.strip():
        return []
    return [item.strip() for item in value.split("|") if item.strip()]


def build_label_record(row: dict) -> dict:
    """Build one labels.json record from one reviewed CSV row."""
    label = row.get("label", "").strip()
    if label not in VALID_LABELS:
        raise ValueError(f"Invalid label: {label!r}")

    return {
        "patient_id": row.get("patient_id", "").strip(),
        "trial_id": row.get("trial_id", "").strip(),
        "label": label,
        "rationale": row.get("rationale", "").strip(),
        "evidence": {
            "patient_facts": split_pipe_text(row.get("patient_facts", "")),
            "trial_criteria": split_pipe_text(row.get("trial_criteria", "")),
        },
    }


def main() -> None:
    with INPUT_FILE.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    reviewed_rows = [
        row for row in rows
        if row.get("label_status", "").strip() == REVIEWED_STATUS
    ]

    labels = [build_label_record(row) for row in reviewed_rows]

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(json.dumps(labels, indent=2), encoding="utf-8")

    print(f"Read {len(rows)} review rows from {INPUT_FILE}")
    print(f"Exported {len(labels)} reviewed labels to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
