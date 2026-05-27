"""Detailed audit for real trial cases in data/processed/trial_cases.json."""

import json
from collections import Counter
from pathlib import Path

DATA_PATH = Path("data/processed/trial_cases.json")


def _is_missing(value) -> bool:
    """Return True for missing, empty, or whitespace-only values."""
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    if isinstance(value, list):
        return len(value) == 0
    return False


def _criterion_count(trial: dict) -> int:
    """Count inclusion + exclusion criteria for one trial."""
    return len(trial.get("inclusion_criteria") or []) + len(trial.get("exclusion_criteria") or [])


def _print_section(title: str) -> None:
    print(f"\n=== {title} ===")


def main() -> None:
    trials: list[dict] = json.loads(DATA_PATH.read_text(encoding="utf-8"))

    _print_section("Trial Case Audit")
    print(f"Total trial cases: {len(trials)}")

    _print_section("Category Counts")
    for category, count in Counter(t.get("category", "MISSING") for t in trials).most_common():
        print(f"{category:<22} {count}")

    _print_section("Metadata Completeness")
    fields = [
        "minimum_age",
        "maximum_age",
        "phase",
        "overall_status",
        "study_type",
        "sex",
        "healthy_volunteers",
    ]
    for field in fields:
        missing = sum(_is_missing(t.get(field)) for t in trials)
        present = len(trials) - missing
        print(f"{field:<20} present={present:<3} missing={missing:<3}")

    _print_section("Criteria Completeness")
    with_inclusion = sum(bool(t.get("inclusion_criteria")) for t in trials)
    with_exclusion = sum(bool(t.get("exclusion_criteria")) for t in trials)
    print(f"With inclusion criteria:    {with_inclusion}")
    print(f"With exclusion criteria:    {with_exclusion}")
    print(f"Missing inclusion criteria: {len(trials) - with_inclusion}")
    print(f"Missing exclusion criteria: {len(trials) - with_exclusion}")

    _print_section("Lowest Criteria Counts")
    sparse_trials = sorted(trials, key=_criterion_count)[:10]
    for trial in sparse_trials:
        count = _criterion_count(trial)
        print(f"{trial.get('trial_id', 'N/A')} | criteria={count:<2} | {trial.get('category', 'N/A'):<18} | {trial.get('title', 'N/A')}")

    _print_section("Potentially Strong Trial Cases")
    strong_trials = [
        t for t in trials
        if t.get("inclusion_criteria")
        and t.get("exclusion_criteria")
        and not _is_missing(t.get("minimum_age"))
        and _criterion_count(t) >= 5
    ]
    print(f"Strong candidates: {len(strong_trials)}")
    for trial in strong_trials[:15]:
        print(f"{trial.get('trial_id', 'N/A')} | {trial.get('category', 'N/A'):<18} | {trial.get('title', 'N/A')}")

    _print_section("Category Examples")
    by_category: dict[str, list[dict]] = {}
    for trial in trials:
        by_category.setdefault(trial.get("category", "MISSING"), []).append(trial)

    for category in sorted(by_category):
        print(f"\n{category}:")
        for trial in by_category[category][:5]:
            print(f"  {trial.get('trial_id', 'N/A')} | {trial.get('title', 'N/A')}")


if __name__ == "__main__":
    main()
