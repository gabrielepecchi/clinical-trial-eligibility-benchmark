"""
run_train_dev_test_split_audit.py — Task 38 (partial): audit train/dev/test split metadata.

Usage:
    PYTHONPATH=. python eval/run_train_dev_test_split_audit.py
"""

import json
from collections import defaultdict
from pathlib import Path

CANDIDATE_PATHS = [
    Path("data/processed/labels_llm_reviewed.json"),
    Path("data/processed/unified_benchmark.json"),
    Path("data/processed/BENCHMARK_VERSION.json"),
    Path("docs/BENCHMARK_VERSION.json"),
]

SPLIT_FIELDS = {"split", "dataset_split", "benchmark_split"}
REQUIRED_SPLITS = {"train", "dev", "test"}

REPORT_PATH = Path("reports/train_dev_test_split_audit.json")


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def write_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)


def extract_records(data) -> list:
    """Return a flat list of dicts from list or dict-with-list structures."""
    if isinstance(data, list):
        return [r for r in data if isinstance(r, dict)]
    if isinstance(data, dict):
        for val in data.values():
            if isinstance(val, list):
                records = [r for r in val if isinstance(r, dict)]
                if records:
                    return records
        return [data]
    return []


def find_split_value(record: dict) -> str | None:
    for field in SPLIT_FIELDS:
        val = record.get(field)
        if val and isinstance(val, str):
            return val.strip().lower()
    return None


def audit_file(path: Path) -> dict:
    try:
        data = load_json(path)
    except FileNotFoundError:
        return {"exists": False, "records": 0, "with_split": 0, "missing_split": 0, "split_counts": {}}
    except json.JSONDecodeError as exc:
        return {"exists": True, "error": str(exc), "records": 0, "with_split": 0, "missing_split": 0, "split_counts": {}}

    records = extract_records(data)
    split_counts: dict[str, int] = defaultdict(int)
    with_split = 0
    missing_split = 0

    for rec in records:
        val = find_split_value(rec)
        if val:
            split_counts[val] += 1
            with_split += 1
        else:
            missing_split += 1

    return {
        "exists": True,
        "records": len(records),
        "with_split": with_split,
        "missing_split": missing_split,
        "split_counts": dict(split_counts),
    }


def build_report() -> dict:
    files_checked: dict[str, dict] = {}
    total_records = 0
    total_with_split = 0
    total_missing_split = 0
    combined_splits: dict[str, int] = defaultdict(int)

    for path in CANDIDATE_PATHS:
        result = audit_file(path)
        files_checked[str(path)] = result
        total_records += result.get("records", 0)
        total_with_split += result.get("with_split", 0)
        total_missing_split += result.get("missing_split", 0)
        for split_val, count in result.get("split_counts", {}).items():
            combined_splits[split_val] += count

    found_splits = set(combined_splits.keys())
    has_split = REQUIRED_SPLITS.issubset(found_splits)

    if has_split:
        recommendation = (
            "Train/dev/test split metadata is present. Task 38 can be completed: "
            "compute per-split metrics (accuracy, macro F1, error rates) and write "
            "a split-stratified evaluation report."
        )
    else:
        missing_splits = sorted(REQUIRED_SPLITS - found_splits)
        recommendation = (
            f"Task 38 cannot be completed yet. "
            f"No records found with split values covering: {', '.join(missing_splits)}. "
            f"Split metadata must be added to the dataset (e.g. a 'split' field in "
            f"labels_llm_reviewed.json or unified_benchmark.json with values "
            f"'train', 'dev', 'test') before per-split evaluation is possible."
        )

    return {
        "files_checked": files_checked,
        "total_records_checked": total_records,
        "records_with_split": total_with_split,
        "records_missing_split": total_missing_split,
        "split_counts": dict(combined_splits),
        "has_train_dev_test_split": has_split,
        "recommendation": recommendation,
    }


def main() -> None:
    report = build_report()
    write_json(report, REPORT_PATH)

    print(f"total_records_checked:    {report['total_records_checked']}")
    print(f"records_with_split:       {report['records_with_split']}")
    print(f"records_missing_split:    {report['records_missing_split']}")
    print(f"split_counts:             {report['split_counts']}")
    print(f"has_train_dev_test_split: {report['has_train_dev_test_split']}")
    print(f"recommendation: {report['recommendation']}")
    print(f"Report written to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
