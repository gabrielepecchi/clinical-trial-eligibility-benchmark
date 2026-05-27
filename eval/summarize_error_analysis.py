"""Summarize manual benchmark error analysis records from error_analysis_sample.json."""

import json
from collections import Counter
from pathlib import Path

INPUT_FILE = Path("data/processed/error_analysis_sample.json")


def main() -> None:
    records: list[dict] = json.loads(INPUT_FILE.read_text(encoding="utf-8"))

    error_type_counts: Counter = Counter(r["error_type"] for r in records)
    gold_label_counts: Counter = Counter(r["gold_label"] for r in records)
    predicted_label_counts: Counter = Counter(r["predicted_label"] for r in records)

    print(f"=== Error Analysis Summary ({len(records)} records) ===\n")

    print("Errors by type:")
    for error_type, count in error_type_counts.most_common():
        print(f"  {error_type:<45} {count}")

    print("\nErrors by gold label:")
    for label, count in gold_label_counts.most_common():
        print(f"  {label:<20} {count}")

    print("\nErrors by predicted label:")
    for label, count in predicted_label_counts.most_common():
        print(f"  {label:<20} {count}")


if __name__ == "__main__":
    main()
