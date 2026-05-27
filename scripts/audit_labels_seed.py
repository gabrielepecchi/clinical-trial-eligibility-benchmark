"""Audit seed eligibility labels before manual review."""

import json
from collections import Counter
from pathlib import Path

LABELS_FILE = Path("data/processed/labels_seed.json")
TRIALS_FILE = Path("data/processed/trial_cases.json")


def load_json(path: Path) -> list[dict]:
    """Load a JSON list from disk."""
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    labels = load_json(LABELS_FILE)
    trials = load_json(TRIALS_FILE)
    trial_categories = {trial["trial_id"]: trial.get("category", "unknown") for trial in trials}

    label_counts = Counter(record["label"] for record in labels)
    patient_counts = Counter(record["patient_id"] for record in labels)
    trial_counts = Counter(record["trial_id"] for record in labels)
    status_counts = Counter(record["label_status"] for record in labels)
    category_counts = Counter(
        trial_categories.get(record["trial_id"], "unknown") for record in labels
    )

    print(f"Total seed labels: {len(labels)}")

    print("\nLabels:")
    for label, count in label_counts.most_common():
        print(f"  {label:<15} {count}")

    print("\nLabel statuses:")
    for status, count in status_counts.most_common():
        print(f"  {status:<22} {count}")

    print("\nLabels by trial category:")
    for category, count in category_counts.most_common():
        print(f"  {category:<22} {count}")

    print("\nMost used patients:")
    for patient_id, count in patient_counts.most_common(20):
        print(f"  {patient_id:<6} {count}")

    print("\nTrial label counts:")
    for trial_id, count in trial_counts.most_common():
        print(f"  {trial_id:<6} {count}")

    print("\nFirst 10 seed labels:")
    for record in labels[:10]:
        category = trial_categories.get(record["trial_id"], "unknown")
        print(
            f"  {record['patient_id']} -> {record['trial_id']} "
            f"({category}) = {record['label']} [{record['label_status']}]"
        )


if __name__ == "__main__":
    main()
