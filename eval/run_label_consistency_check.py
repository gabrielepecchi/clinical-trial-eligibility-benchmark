"""
eval/run_label_consistency_check.py

Task 63 — Cross-file label consistency check.

Reads available files from data/processed/ and checks that the same
patient_id + trial_id pair never has conflicting gold labels across sources.

Writes reports/label_consistency_check.md

Usage:
    PYTHONPATH=. python eval/run_label_consistency_check.py
"""

from __future__ import annotations

import csv
import json
import os
import sys
from collections import defaultdict
from typing import Any

OUTPUT_PATH = "reports/label_consistency_check.md"

VALID_LABELS = {"eligible", "not_eligible", "unclear"}

def _golden_cases_path() -> str:
    """Return the first golden_cases path that exists, preferring examples/."""
    for p in ("examples/golden_cases.json", "data/processed/golden_cases.json"):
        if os.path.exists(p):
            return p
    return "examples/golden_cases.json"


SOURCES = [
    ("labels_llm_reviewed", "data/processed/labels_llm_reviewed.json"),
    ("results_llm_reviewed", "data/processed/results_llm_reviewed.json"),
    ("unified_benchmark", "data/processed/unified_benchmark.json"),
    ("golden_cases", _golden_cases_path()),
    ("human_review_queue", "data/processed/human_review_queue.csv"),
]

LABEL_FIELDS = ["label", "gold_label", "expected_label", "gold", "correct_label"]


# ---------------------------------------------------------------------------
# Pure functions
# ---------------------------------------------------------------------------

def load_json(path: str, required: bool = False) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        if required:
            print(f"ERROR: File not found: {path}", file=sys.stderr)
            sys.exit(1)
        return None
    except json.JSONDecodeError as exc:
        print(f"ERROR: Malformed JSON in {path}: {exc}", file=sys.stderr)
        sys.exit(1)


def load_csv_rows(path: str) -> list[dict[str, str]] | None:
    try:
        with open(path, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    except FileNotFoundError:
        return None
    except Exception as exc:
        print(f"ERROR: Could not read {path}: {exc}", file=sys.stderr)
        sys.exit(1)


def pair_key(record: dict) -> tuple[str, str] | None:
    pid = str(record.get("patient_id", "")).strip()
    tid = str(record.get("trial_id", "")).strip()
    if pid and tid:
        return (pid, tid)
    return None


def extract_label_from_record(record: dict) -> str | None:
    """Try all known label field names; also handle label as dict."""
    for field in LABEL_FIELDS:
        val = record.get(field)
        if val is None:
            continue
        if isinstance(val, str) and val.strip():
            return val.strip().lower()
        if isinstance(val, dict):
            for sub in LABEL_FIELDS:
                sub_val = val.get(sub)
                if isinstance(sub_val, str) and sub_val.strip():
                    return sub_val.strip().lower()
    return None


def _flatten(data: Any) -> list[dict]:
    """Flatten JSON data into a list of dicts."""
    if isinstance(data, list):
        return [r for r in data if isinstance(r, dict)]
    if isinstance(data, dict):
        for key in ("predictions", "results", "cases", "labels", "records", "examples"):
            if key in data and isinstance(data[key], list):
                return [r for r in data[key] if isinstance(r, dict)]
        # dict keyed by patient/trial id
        flat = []
        for v in data.values():
            if isinstance(v, list):
                flat.extend(r for r in v if isinstance(r, dict))
            elif isinstance(v, dict):
                flat.append(v)
        return flat
    return []


def extract_records_from_source(name: str, data: Any) -> list[dict]:
    """
    Return a list of normalised records: {patient_id, trial_id, label, _raw}.
    Works for both JSON (any shape) and pre-parsed CSV rows (list of dicts).
    """
    if isinstance(data, list) and data and isinstance(data[0], dict) and "patient_id" not in data[0] and set(data[0].keys()) <= {"patient_id", "trial_id", "label", "gold_label", "expected_label", "gold", "correct_label", "prediction", "predicted_label", "notes", "source"}:
        # already flat CSV rows — fall through to _flatten path
        pass
    rows = _flatten(data) if not (isinstance(data, list) and all(isinstance(r, dict) for r in data)) else data
    if isinstance(data, list) and all(isinstance(r, dict) for r in data):
        rows = data

    records = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        pid = str(row.get("patient_id", "")).strip()
        tid = str(row.get("trial_id", "")).strip()
        label = extract_label_from_record(row)
        records.append({
            "patient_id": pid,
            "trial_id": tid,
            "label": label,
            "_raw": row,
        })
    return records


def find_available_sources() -> list[tuple[str, str, str]]:
    """Return list of (name, path, kind) for sources that exist on disk."""
    available = []
    for name, path in SOURCES:
        if os.path.exists(path):
            kind = "csv" if path.endswith(".csv") else "json"
            available.append((name, path, kind))
    return available


def analyze_source_records(source_name: str, records: list[dict]) -> dict[str, Any]:
    total = len(records)
    missing_pid = sum(1 for r in records if not r["patient_id"])
    missing_tid = sum(1 for r in records if not r["trial_id"])
    missing_label = sum(1 for r in records if r["label"] is None)
    invalid_label = sum(
        1 for r in records
        if r["label"] is not None and r["label"] not in VALID_LABELS
    )

    # within-file duplicates
    pair_labels: dict[tuple[str, str], list[str]] = defaultdict(list)
    for r in records:
        key = pair_key(r)
        if key and r["label"] is not None:
            pair_labels[key].append(r["label"])

    duplicate_pairs = {k: v for k, v in pair_labels.items() if len(v) > 1}
    conflicting_duplicates = {k: v for k, v in duplicate_pairs.items() if len(set(v)) > 1}

    # unique pairs with resolved label (first seen wins for cross-file check)
    unique_pairs: dict[tuple[str, str], str] = {}
    for r in records:
        key = pair_key(r)
        if key and r["label"] is not None and key not in unique_pairs:
            unique_pairs[key] = r["label"]

    return {
        "source": source_name,
        "total_records": total,
        "missing_patient_id": missing_pid,
        "missing_trial_id": missing_tid,
        "missing_label": missing_label,
        "invalid_label": invalid_label,
        "unique_pairs": len(unique_pairs),
        "duplicate_pair_count": len(duplicate_pairs),
        "conflicting_duplicate_count": len(conflicting_duplicates),
        "_unique_pairs": unique_pairs,  # internal use
    }


def analyze_cross_file_consistency(
    source_summaries: list[dict],
) -> dict[str, Any]:
    # Merge: pair -> {source: label}
    pair_source_labels: dict[tuple[str, str], dict[str, str]] = defaultdict(dict)
    for s in source_summaries:
        for pair, label in s["_unique_pairs"].items():
            pair_source_labels[pair][s["source"]] = label

    conflicts: list[dict] = []
    for pair, src_labels in pair_source_labels.items():
        unique_labels = set(src_labels.values())
        if len(unique_labels) > 1:
            conflicts.append({
                "patient_id": pair[0],
                "trial_id": pair[1],
                "labels_by_source": src_labels,
            })

    return {
        "total_unique_pairs_across_files": len(pair_source_labels),
        "cross_file_conflict_count": len(conflicts),
        "top_conflicts": conflicts[:20],
    }


def format_markdown_report(
    available: list[tuple[str, str, str]],
    source_summaries: list[dict],
    cross: dict[str, Any],
) -> str:
    lines: list[str] = []
    lines.append("# Label Consistency Check Report\n")

    lines.append("## Sources Checked\n")
    if available:
        lines.append("| Source | Path |")
        lines.append("|--------|------|")
        for name, path, kind in available:
            lines.append(f"| {name} | `{path}` |")
    else:
        lines.append("_No source files found._")
    lines.append("")

    lines.append("## Per-Source Summary\n")
    lines.append("| Source | Records | Unique pairs | Missing pid | Missing tid | Missing label | Invalid label | Dup pairs | Conflict dups |")
    lines.append("|--------|--------:|-------------:|------------:|------------:|--------------:|--------------:|----------:|--------------:|")
    for s in source_summaries:
        lines.append(
            f"| {s['source']} "
            f"| {s['total_records']} "
            f"| {s['unique_pairs']} "
            f"| {s['missing_patient_id']} "
            f"| {s['missing_trial_id']} "
            f"| {s['missing_label']} "
            f"| {s['invalid_label']} "
            f"| {s['duplicate_pair_count']} "
            f"| {s['conflicting_duplicate_count']} |"
        )
    lines.append("")

    lines.append("## Cross-File Consistency\n")
    lines.append(f"- **Total unique pairs across all files:** {cross['total_unique_pairs_across_files']}")
    lines.append(f"- **Cross-file conflicting pairs:** {cross['cross_file_conflict_count']}")
    lines.append("")

    if cross["top_conflicts"]:
        lines.append("### Conflicting Pairs (up to 20)\n")
        lines.append("| patient_id | trial_id | labels by source |")
        lines.append("|------------|----------|------------------|")
        for c in cross["top_conflicts"]:
            label_str = "; ".join(f"{src}={lbl}" for src, lbl in sorted(c["labels_by_source"].items()))
            lines.append(f"| {c['patient_id']} | {c['trial_id']} | {label_str} |")
        lines.append("")
    else:
        lines.append("_No cross-file label conflicts detected._\n")

    if cross["cross_file_conflict_count"] == 0:
        lines.append("**Result: PASS — all shared pairs have consistent gold labels across sources.**\n")
    else:
        lines.append(f"**Result: WARN — {cross['cross_file_conflict_count']} conflicting pair(s) found. Review above.**\n")

    return "\n".join(lines)


def write_text(text: str, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def main() -> None:
    available = find_available_sources()

    if not available:
        print("WARNING: No source files found in data/processed/. Nothing to check.")
        write_text("# Label Consistency Check\n\n_No source files found._\n", OUTPUT_PATH)
        print(f"Report written to: {OUTPUT_PATH}")
        return

    source_summaries: list[dict] = []
    for name, path, kind in available:
        if kind == "csv":
            rows = load_csv_rows(path)
            data: Any = rows if rows is not None else []
        else:
            data = load_json(path, required=False)
            if data is None:
                data = []

        records = extract_records_from_source(name, data)
        summary = analyze_source_records(name, records)
        source_summaries.append(summary)

    cross = analyze_cross_file_consistency(source_summaries)

    # Strip internal key before report
    for s in source_summaries:
        s.pop("_unique_pairs", None)

    report = format_markdown_report(available, source_summaries, cross)
    write_text(report, OUTPUT_PATH)

    total_pairs = cross["total_unique_pairs_across_files"]
    conflicts = cross["cross_file_conflict_count"]
    print(f"Sources checked : {len(available)}")
    print(f"Total pairs     : {total_pairs}")
    print(f"Conflicts found : {conflicts}")
    print(f"Report written  : {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
