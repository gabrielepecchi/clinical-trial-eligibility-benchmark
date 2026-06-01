"""
Task 65: Criterion-level coverage analysis.

Compares trial criteria defined in trial_cases.json against criteria
represented in criterion_level_results.csv and writes a Markdown audit report.

Usage:
    PYTHONPATH=. python eval/run_criterion_coverage_analysis.py
    PYTHONPATH=. python eval/run_criterion_coverage_analysis.py \
        --trials   data/processed/trial_cases.json \
        --results  data/processed/criterion_level_results.csv \
        --output   reports/criterion_coverage_analysis.md
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import defaultdict
from datetime import datetime
from typing import Any

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_TRIALS = "data/processed/trial_cases.json"
DEFAULT_RESULTS = "data/processed/criterion_level_results.csv"
DEFAULT_OUTPUT = "reports/criterion_coverage_analysis.md"

# Fields tried in order when looking for criteria inside a trial record
CRITERIA_FIELDS: list[str] = [
    "criteria_text",
    "eligibility_criteria",
    "inclusion_criteria",
    "exclusion_criteria",
    "criteria",
    "inclusion",
    "exclusion",
    "inclusion_text",
    "exclusion_text",
]

# CSV column candidates for criterion text
CRITERION_TEXT_CANDIDATES: list[str] = [
    "criterion",
    "criterion_text",
    "text",
    "eligibility_criterion",
    "criterion_raw",
]

# Minimum length for conservative substring match
SUBSTRING_MATCH_MIN_LEN = 20

# Bullet / list markers stripped during normalisation
_BULLET_RE = re.compile(r"^\s*[-•*·▪◦]\s*")
_WHITESPACE_RE = re.compile(r"\s+")


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def load_json(path: str) -> Any:
    """Load JSON from *path*. Exits non-zero on missing or malformed file."""
    if not os.path.isfile(path):
        print(f"ERROR: file not found: {path}", file=sys.stderr)
        sys.exit(1)
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except json.JSONDecodeError as exc:
        print(f"ERROR: malformed JSON in {path}: {exc}", file=sys.stderr)
        sys.exit(1)


def load_csv_rows(path: str) -> list[dict[str, str]]:
    """Load a CSV as a list of dicts. Exits non-zero on missing file."""
    if not os.path.isfile(path):
        print(f"ERROR: file not found: {path}", file=sys.stderr)
        sys.exit(1)
    with open(path, encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        return [row for row in reader]


def write_text(text: str, path: str) -> None:
    """Write *text* to *path*, creating parent directories as needed."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)


# ---------------------------------------------------------------------------
# Column detection
# ---------------------------------------------------------------------------


def find_first_existing_column(
    rows: list[dict[str, str]], candidates: list[str]
) -> str | None:
    """Return the first column name from *candidates* present in *rows*."""
    if not rows:
        return None
    headers = set(rows[0].keys())
    for col in candidates:
        if col in headers:
            return col
    return None


# ---------------------------------------------------------------------------
# Text normalisation and splitting
# ---------------------------------------------------------------------------


def normalize_criterion_text(text: str) -> str:
    """
    Return a deterministically normalised version of *text*:
    lowercase, no leading bullet markers, collapsed whitespace.
    """
    text = text.lower()
    text = _BULLET_RE.sub("", text)
    text = _WHITESPACE_RE.sub(" ", text)
    return text.strip()


def split_criteria(value: Any) -> list[str]:
    """
    Split *value* into individual criterion strings.

    - If *value* is a list, each item is returned as-is (non-empty strings only).
    - If *value* is a string, it is split on newlines and semicolons.
    """
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        # Split on newline first, then semicolon within each chunk
        parts: list[str] = []
        for line in value.splitlines():
            for chunk in line.split(";"):
                chunk = chunk.strip()
                if chunk:
                    parts.append(chunk)
        return parts
    return []


# ---------------------------------------------------------------------------
# Trial criteria extraction
# ---------------------------------------------------------------------------


def extract_trial_criteria(trial_record: dict[str, Any]) -> list[str]:
    """
    Return a list of raw criterion strings for one trial record.

    Collects all values from known criteria fields; deduplicates preserving
    order.
    """
    seen: set[str] = set()
    result: list[str] = []
    for field in CRITERIA_FIELDS:
        value = trial_record.get(field)
        if value is None:
            continue
        for item in split_criteria(value):
            if item and item not in seen:
                seen.add(item)
                result.append(item)
    return result


def index_trial_criteria(
    trials: list[dict[str, Any]],
) -> dict[str, list[str]]:
    """
    Return {trial_id: [normalised_criterion, ...]} for every trial in *trials*.

    Accepts a list of trial records or a dict of {trial_id: record}.
    """
    if isinstance(trials, dict):
        trial_list = [
            {**v, "trial_id": k} if "trial_id" not in v else v
            for k, v in trials.items()
        ]
    else:
        trial_list = trials

    index: dict[str, list[str]] = {}
    for record in trial_list:
        tid = str(record.get("trial_id", record.get("id", ""))).strip()
        if not tid:
            continue
        raw = extract_trial_criteria(record)
        index[tid] = [normalize_criterion_text(c) for c in raw if c.strip()]
    return index


# ---------------------------------------------------------------------------
# Result criteria extraction
# ---------------------------------------------------------------------------


def extract_result_criteria(
    rows: list[dict[str, str]],
) -> dict[str, list[str]]:
    """
    Return {trial_id: [normalised_criterion, ...]} from CSV rows.

    Returns an empty dict if required columns are absent.
    """
    tid_col = find_first_existing_column(rows, ["trial_id", "trialid", "trial"])
    crit_col = find_first_existing_column(rows, CRITERION_TEXT_CANDIDATES)

    if not tid_col or not crit_col:
        return {}

    index: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        tid = row.get(tid_col, "").strip()
        crit = row.get(crit_col, "").strip()
        if tid and crit:
            index[tid].append(normalize_criterion_text(crit))
    return dict(index)


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------


def criterion_matches(expected: str, observed_set: set[str]) -> bool:
    """
    Return True if *expected* is covered by any entry in *observed_set*.

    First tries exact match; then tries conservative substring match where
    one string contains the other and both are at least SUBSTRING_MATCH_MIN_LEN
    characters long.
    """
    if expected in observed_set:
        return True
    if len(expected) < SUBSTRING_MATCH_MIN_LEN:
        return False
    for obs in observed_set:
        if len(obs) < SUBSTRING_MATCH_MIN_LEN:
            continue
        if expected in obs or obs in expected:
            return True
    return False


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def analyze_criterion_coverage(
    trial_criteria: dict[str, list[str]],
    result_criteria: dict[str, list[str]],
) -> dict[str, Any]:
    """
    Compare trial_criteria against result_criteria and return a summary dict.
    """
    all_trial_ids = set(trial_criteria.keys())
    result_trial_ids = set(result_criteria.keys())

    full_coverage: list[str] = []
    partial_coverage: list[str] = []
    zero_coverage: list[str] = []
    missing_by_trial: dict[str, list[str]] = {}

    total_extracted = 0
    total_covered = 0
    total_missing = 0

    for tid in sorted(all_trial_ids):
        expected = trial_criteria[tid]
        total_extracted += len(expected)

        if not expected:
            full_coverage.append(tid)
            continue

        observed: set[str] = set(result_criteria.get(tid, []))
        matched = [c for c in expected if criterion_matches(c, observed)]
        missing = [c for c in expected if not criterion_matches(c, observed)]

        total_covered += len(matched)
        total_missing += len(missing)

        if not missing:
            full_coverage.append(tid)
        elif not matched:
            zero_coverage.append(tid)
            missing_by_trial[tid] = missing
        else:
            partial_coverage.append(tid)
            missing_by_trial[tid] = missing

    # Orphan rows: result trial_ids not in trial_cases
    orphan_trial_ids = result_trial_ids - all_trial_ids
    orphan_rows: list[dict[str, str]] = []
    for tid in sorted(orphan_trial_ids):
        for crit in result_criteria[tid]:
            orphan_rows.append({"trial_id": tid, "criterion": crit})

    return {
        "total_trials": len(all_trial_ids),
        "result_trial_ids": sorted(result_trial_ids),
        "total_extracted_criteria": total_extracted,
        "total_covered_criteria": total_covered,
        "total_missing_criteria": total_missing,
        "full_coverage": full_coverage,
        "partial_coverage": partial_coverage,
        "zero_coverage": zero_coverage,
        "missing_by_trial": missing_by_trial,
        "orphan_rows": orphan_rows,
    }


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------


def _trunc(text: str, n: int = 120) -> str:
    text = text.replace("\n", " ").strip()
    return text if len(text) <= n else text[:n].rstrip() + "…"


def format_markdown_report(summary: dict[str, Any]) -> str:
    """Return a Markdown string for *summary*."""
    lines: list[str] = []

    total_trials: int = summary["total_trials"]
    result_ids: list[str] = summary["result_trial_ids"]
    total_extracted: int = summary["total_extracted_criteria"]
    total_covered: int = summary["total_covered_criteria"]
    total_missing: int = summary["total_missing_criteria"]
    full: list[str] = summary["full_coverage"]
    partial: list[str] = summary["partial_coverage"]
    zero: list[str] = summary["zero_coverage"]
    missing_by_trial: dict[str, list[str]] = summary["missing_by_trial"]
    orphan_rows: list[dict[str, str]] = summary["orphan_rows"]

    lines.append("# Criterion Coverage Analysis Report")
    lines.append("")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append(
        "> Compares criteria in trial_cases.json against criterion_level_results.csv.  "
    )
    lines.append("> No labels or source files have been modified.")
    lines.append("")

    lines.append("## Summary")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Trials in trial_cases.json | {total_trials} |")
    lines.append(f"| Trials in criterion_level_results.csv | {len(result_ids)} |")
    lines.append(f"| Total extracted trial criteria | {total_extracted} |")
    lines.append(f"| Covered criteria | {total_covered} |")
    lines.append(f"| Missing criteria | {total_missing} |")
    lines.append(f"| Full coverage trials | {len(full)} |")
    lines.append(f"| Partial coverage trials | {len(partial)} |")
    lines.append(f"| Zero coverage trials | {len(zero)} |")
    lines.append(f"| Orphan criterion rows | {len(orphan_rows)} |")
    lines.append("")

    # Top 20 trials by missing criterion count
    if missing_by_trial:
        lines.append("## Top 20 Trials by Missing Criterion Count")
        lines.append("")
        lines.append("| Trial ID | Missing Count |")
        lines.append("|----------|---------------|")
        ranked = sorted(missing_by_trial.items(), key=lambda x: -len(x[1]))
        for tid, missing in ranked[:20]:
            lines.append(f"| {tid} | {len(missing)} |")
        lines.append("")

        lines.append("## Missing Criteria Examples")
        lines.append("")
        for tid, missing in ranked[:20]:
            lines.append(f"### `{tid}` ({len(missing)} missing)")
            lines.append("")
            for crit in missing[:5]:
                lines.append(f"- {_trunc(crit)}")
            if len(missing) > 5:
                lines.append(f"- *(+{len(missing) - 5} more)*")
            lines.append("")

    # Orphan rows
    if orphan_rows:
        lines.append("## Orphan Criterion Rows")
        lines.append("")
        lines.append(
            "These rows appear in criterion_level_results.csv but do not map "
            "to any trial_id in trial_cases.json."
        )
        lines.append("")
        lines.append("| Trial ID | Criterion (preview) |")
        lines.append("|----------|---------------------|")
        for row in orphan_rows[:30]:
            lines.append(f"| {row['trial_id']} | {_trunc(row['criterion'], 80)} |")
        if len(orphan_rows) > 30:
            lines.append(f"| *(+{len(orphan_rows) - 30} more)* | |")
        lines.append("")

    if not missing_by_trial and not orphan_rows:
        lines.append("**All criteria are fully covered and no orphan rows found.**")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Criterion coverage analysis report (Task 65)."
    )
    parser.add_argument(
        "--trials", default=DEFAULT_TRIALS,
        help=f"Path to trial_cases.json (default: {DEFAULT_TRIALS})",
    )
    parser.add_argument(
        "--results", default=DEFAULT_RESULTS,
        help=f"Path to criterion_level_results.csv (default: {DEFAULT_RESULTS})",
    )
    parser.add_argument(
        "--output", default=DEFAULT_OUTPUT,
        help=f"Output Markdown path (default: {DEFAULT_OUTPUT})",
    )
    args = parser.parse_args()

    trials_raw = load_json(args.trials)
    csv_rows = load_csv_rows(args.results)

    if isinstance(trials_raw, dict) and not any(
        isinstance(v, dict) for v in trials_raw.values()
    ):
        print("ERROR: unrecognised structure in trials file.", file=sys.stderr)
        sys.exit(1)

    trial_list = (
        list(trials_raw.values())
        if isinstance(trials_raw, dict)
        else trials_raw
    )
    # Attach trial_id from dict key when not present in record
    if isinstance(trials_raw, dict):
        enriched: list[dict[str, Any]] = []
        for k, v in trials_raw.items():
            if isinstance(v, dict):
                rec = dict(v)
                if "trial_id" not in rec:
                    rec["trial_id"] = k
                enriched.append(rec)
        trial_list = enriched

    trial_criteria = index_trial_criteria(trial_list)
    result_criteria = extract_result_criteria(csv_rows)

    summary = analyze_criterion_coverage(trial_criteria, result_criteria)
    report = format_markdown_report(summary)
    write_text(report, args.output)

    print(
        f"Criterion coverage report written to: {args.output}\n"
        f"  Trials read            : {summary['total_trials']}\n"
        f"  Criterion rows read    : {len(csv_rows)}\n"
        f"  Full coverage trials   : {len(summary['full_coverage'])}\n"
        f"  Partial coverage trials: {len(summary['partial_coverage'])}\n"
        f"  Zero coverage trials   : {len(summary['zero_coverage'])}"
    )
    sys.exit(0)


if __name__ == "__main__":
    main()
