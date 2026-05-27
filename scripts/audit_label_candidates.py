"""Audit generated label candidates before manual labeling."""

import json
from collections import Counter
from pathlib import Path

CANDIDATES_FILE = Path("data/processed/label_candidates.json")


def main() -> None:
    candidates: list[dict] = json.loads(CANDIDATES_FILE.read_text(encoding="utf-8"))

    print(f"Total label candidates: {len(candidates)}")

    category_counts = Counter(c["trial_category"] for c in candidates)
    patient_counts = Counter(c["patient_id"] for c in candidates)
    trial_counts = Counter(c["trial_id"] for c in candidates)
    status_counts = Counter(c["label_status"] for c in candidates)

    print("\nCandidates by trial category:")
    for category, count in category_counts.most_common():
        print(f"  {category:<22} {count}")

    print("\nCandidates by label status:")
    for status, count in status_counts.most_common():
        print(f"  {status:<22} {count}")

    print("\nMost used patients:")
    for patient_id, count in patient_counts.most_common(20):
        print(f"  {patient_id:<6} {count}")

    print("\nTrials with candidate counts:")
    for trial_id, count in trial_counts.most_common():
        print(f"  {trial_id:<6} {count}")

    print("\nFirst 10 candidates:")
    for candidate in candidates[:10]:
        print(
            f"  {candidate['patient_id']} -> {candidate['trial_id']} "
            f"({candidate['trial_category']}) [{candidate['label_status']}]"
        )


if __name__ == "__main__":
    main()
