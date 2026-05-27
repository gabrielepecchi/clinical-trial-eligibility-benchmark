"""Audit the quality and composition of data/processed/trial_cases.json."""

import json
from collections import Counter
from pathlib import Path

DATA_PATH = Path("data/processed/trial_cases.json")


def main() -> None:
    with DATA_PATH.open(encoding="utf-8") as f:
        trials: list[dict] = json.load(f)

    total = len(trials)
    print(f"Total trial cases: {total}")

    categories = Counter(t.get("category", "MISSING") for t in trials)
    print("\nCategory counts:")
    for cat, count in categories.most_common():
        print(f"  {cat}: {count}")

    has_inclusion = sum(1 for t in trials if t.get("inclusion_criteria"))
    has_exclusion = sum(1 for t in trials if t.get("exclusion_criteria"))
    missing_inclusion = total - has_inclusion
    missing_exclusion = total - has_exclusion

    missing_min_age = sum(1 for t in trials if not t.get("minimum_age"))
    missing_max_age = sum(1 for t in trials if not t.get("maximum_age"))
    missing_phase = sum(1 for t in trials if not t.get("phase"))

    print(f"\nWith inclusion criteria:    {has_inclusion}")
    print(f"With exclusion criteria:    {has_exclusion}")
    print(f"Missing inclusion criteria: {missing_inclusion}")
    print(f"Missing exclusion criteria: {missing_exclusion}")
    print(f"Missing minimum_age:        {missing_min_age}")
    print(f"Missing maximum_age:        {missing_max_age}")
    print(f"Missing phase:              {missing_phase}")

    print("\nFirst 10 trial IDs:")
    for t in trials[:10]:
        trial_id = t.get("trial_id") or t.get("nct_id") or "N/A"
        category = t.get("category", "N/A")
        title = t.get("title") or "N/A"
        print(f"  {trial_id} | {category} | {title}")


if __name__ == "__main__":
    main()
