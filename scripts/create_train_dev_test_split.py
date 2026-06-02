"""
create_train_dev_test_split.py — Task 38: generate deterministic train/dev/test split metadata.

Usage:
    PYTHONPATH=. python scripts/create_train_dev_test_split.py
"""

import json
import random
from collections import defaultdict
from pathlib import Path

INPUT_PATH  = Path("data/processed/labels_llm_reviewed.json")
OUTPUT_PATH = Path("data/processed/labels_llm_reviewed_with_splits.json")

SEED   = 42
RATIOS = {"train": 0.6, "dev": 0.2, "test": 0.2}
LABEL_VALUES = ["eligible", "not_eligible", "unclear"]


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def write_json(data, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)


def assign_splits(records: list) -> list:
    """
    Assign split field to each record, balanced by label value.
    Records within each label group are shuffled with a fixed seed,
    then allocated train/dev/test in approximate 60/20/20 ratios.
    """
    rng = random.Random(SEED)

    # Group record indices by label
    groups: dict[str, list[int]] = defaultdict(list)
    for i, rec in enumerate(records):
        label = rec.get("label", "unknown") if isinstance(rec, dict) else "unknown"
        groups[label].append(i)

    split_assignment: dict[int, str] = {}

    for label, indices in groups.items():
        shuffled = indices[:]
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_train = round(n * RATIOS["train"])
        n_dev   = round(n * RATIOS["dev"])
        # test gets remainder to ensure total == n
        for j, idx in enumerate(shuffled):
            if j < n_train:
                split_assignment[idx] = "train"
            elif j < n_train + n_dev:
                split_assignment[idx] = "dev"
            else:
                split_assignment[idx] = "test"

    result = []
    for i, rec in enumerate(records):
        updated = dict(rec) if isinstance(rec, dict) else rec
        if isinstance(updated, dict):
            updated["split"] = split_assignment.get(i, "train")
        result.append(updated)

    return result


def count_splits(records: list) -> dict:
    counts: dict[str, int] = defaultdict(int)
    for rec in records:
        if isinstance(rec, dict):
            counts[rec.get("split", "unknown")] += 1
    return dict(counts)


def count_splits_by_label(records: list) -> dict:
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for rec in records:
        if isinstance(rec, dict):
            counts[rec.get("split", "unknown")][rec.get("label", "unknown")] += 1
    return {split: dict(lc) for split, lc in counts.items()}


def main() -> None:
    try:
        labels = load_json(INPUT_PATH)
    except FileNotFoundError:
        print(f"[ERROR] File not found: {INPUT_PATH}")
        return
    except json.JSONDecodeError as exc:
        print(f"[ERROR] Invalid JSON in {INPUT_PATH}: {exc}")
        return

    if not isinstance(labels, list):
        print(f"[ERROR] Expected a JSON array in {INPUT_PATH}.")
        return

    print(f"Loaded {len(labels)} labels")

    labeled = assign_splits(labels)
    write_json(labeled, OUTPUT_PATH)

    split_counts    = count_splits(labeled)
    split_by_label  = count_splits_by_label(labeled)

    print(f"Wrote {len(labeled)} labels with splits")
    for split in ("train", "dev", "test"):
        print(f"  {split}: {split_counts.get(split, 0)}")
    print("Split counts by label:")
    for split in ("train", "dev", "test"):
        lc = split_by_label.get(split, {})
        parts = ", ".join(f"{lbl}={lc.get(lbl, 0)}" for lbl in LABEL_VALUES)
        print(f"  {split}: {parts}")
    print(f"Output written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
