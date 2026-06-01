"""
Task 33: Label disagreement / ambiguity report.

Reads data/processed/labels_llm_reviewed.json and writes a Markdown report
to reports/label_disagreement_report.md.

Flags records that show ambiguity or disagreement signals using existing
metadata only. Does not invent annotators, modify labels, or touch any
matcher/benchmark logic.

Usage:
    PYTHONPATH=. python eval/label_disagreement_report.py
    PYTHONPATH=. python eval/label_disagreement_report.py \
        --input data/processed/labels_llm_reviewed.json \
        --output reports/label_disagreement_report.md
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_INPUT = "data/processed/labels_llm_reviewed.json"
DEFAULT_OUTPUT = "reports/label_disagreement_report.md"

# label_status words considered strong ambiguity signals.
# Generic workflow terms such as 'needs_spotcheck', 'draft', and 'review'
# are intentionally excluded: they describe routine pipeline state, not
# genuine label uncertainty.
AMBIGUOUS_STATUS_KEYWORDS: list[str] = [
    "uncertain",
    "low_confidence",
    "disagreement",
    "conflict",
    "needs_adjudication",
]

# Words in rationale text that suggest uncertainty
UNCERTAINTY_LANGUAGE: list[str] = [
    "unclear",
    "uncertain",
    "insufficient",
    "missing",
    "not documented",
    "ambiguous",
    "cannot determine",
    "requires",
    "unknown",
]

# Minimum rationale length (chars) below which it is considered very short
SHORT_RATIONALE_THRESHOLD = 30


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def load_json(path: str) -> Any:
    """Load and return JSON from *path*. Raises SystemExit on error."""
    if not os.path.isfile(path):
        print(f"ERROR: input file not found: {path}", file=sys.stderr)
        sys.exit(1)
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except json.JSONDecodeError as exc:
        print(f"ERROR: malformed JSON in {path}: {exc}", file=sys.stderr)
        sys.exit(1)


def extract_label_records(data: Any) -> list[dict[str, Any]]:
    """
    Return a flat list of label records from *data*.

    Accepts:
    - a list of records directly
    - a dict with a 'labels' or 'pairs' key containing a list
    - a dict mapping pair_id -> record
    """
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("labels", "pairs", "records"):
            if key in data and isinstance(data[key], list):
                return data[key]
        # dict of {pair_id: record}
        records: list[dict[str, Any]] = []
        for value in data.values():
            if isinstance(value, dict):
                records.append(value)
        if records:
            return records
    print("ERROR: unrecognised JSON structure in label file.", file=sys.stderr)
    sys.exit(1)


def rationale_preview(text: str, max_chars: int = 160) -> str:
    """Return up to *max_chars* characters of *text*, stripped."""
    if not text:
        return ""
    text = text.strip().replace("\n", " ")
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "…"


def evidence_present(value: Any) -> bool:
    """Return True when *value* contains at least one non-empty piece of evidence."""
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip() != ""
    if isinstance(value, list):
        return any(isinstance(v, str) and v.strip() for v in value)
    if isinstance(value, dict):
        return any(
            (isinstance(v, str) and v.strip()) or (v is not None and not isinstance(v, str))
            for v in value.values()
        )
    return False


def detect_disagreement_flags(record: dict[str, Any]) -> list[str]:
    """
    Return a list of human-readable flag strings for *record*.

    Checks are purely based on existing fields; no labels are changed.
    """
    flags: list[str] = []

    label = str(record.get("label", "")).strip().lower()
    rationale_raw = record.get("rationale") or record.get("explanation") or ""
    rationale = str(rationale_raw).strip()
    evidence_raw = record.get("evidence") or record.get("matched_facts") or ""
    status = str(record.get("label_status", "")).strip().lower()

    # 1. Ambiguous label_status
    for kw in AMBIGUOUS_STATUS_KEYWORDS:
        if kw in status:
            flags.append(f"label_status contains '{kw}'")
            break

    # 2. Label is unclear
    if label == "unclear":
        flags.append("label is unclear")

    # 3. Missing rationale
    if not rationale:
        flags.append("rationale missing")
    elif len(rationale) < SHORT_RATIONALE_THRESHOLD:
        flags.append("rationale very short (<30 chars)")

    # 4. Missing evidence
    if not evidence_present(evidence_raw):
        flags.append("evidence missing or empty")

    # 5. Uncertainty language in rationale
    rationale_lower = rationale.lower()
    found_uncertainty = [
        phrase
        for phrase in UNCERTAINTY_LANGUAGE
        if phrase in rationale_lower
    ]
    if found_uncertainty:
        sample = ", ".join(f"'{p}'" for p in found_uncertainty[:3])
        flags.append(f"rationale contains uncertainty language: {sample}")

    # 6. Weak label/rationale alignment (conservative keyword check only)
    #    eligible label but rationale contains strong negative signals
    #    not_eligible label but rationale lacks any negative signal
    if label == "eligible" and rationale:
        negative_signals = [
            "not eligible", "not_eligible", "excluded", "violates", "fails",
            "does not meet",
        ]
        if any(s in rationale_lower for s in negative_signals):
            flags.append("label 'eligible' but rationale contains negative signal")
    if label == "not_eligible" and rationale:
        positive_signals = [
            "meets all", "satisfies all", "eligible", "no exclusion",
        ]
        if any(s in rationale_lower for s in positive_signals):
            flags.append("label 'not_eligible' but rationale contains positive signal")

    # 7. Unusual / missing optional metadata
    if "confidence" in record:
        conf = str(record["confidence"]).strip().lower()
        if conf in ("low", "0", "none", ""):
            flags.append(f"confidence is '{conf}'")

    return flags


def analyze_label_disagreement(
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Analyse *records* and return a summary dict:
      - total: int
      - flagged: list of {record, flags}
      - counts_by_flag: {flag_type_prefix: count}
      - counts_by_label: {label: count} among flagged
    """
    flagged: list[dict[str, Any]] = []
    counts_by_flag: dict[str, int] = defaultdict(int)

    for rec in records:
        flags = detect_disagreement_flags(rec)
        if flags:
            flagged.append({"record": rec, "flags": flags})
            for flag in flags:
                # Use first ~50 chars as bucket key
                key = flag[:50].rstrip()
                counts_by_flag[key] += 1

    counts_by_label: dict[str, int] = defaultdict(int)
    for item in flagged:
        label = str(item["record"].get("label", "unknown")).strip().lower()
        counts_by_label[label] += 1

    return {
        "total": len(records),
        "flagged": flagged,
        "counts_by_flag": dict(sorted(counts_by_flag.items(), key=lambda x: -x[1])),
        "counts_by_label": dict(sorted(counts_by_label.items(), key=lambda x: -x[1])),
    }


def _record_id(record: dict[str, Any]) -> str:
    """Return a display identifier for *record*."""
    pid = record.get("patient_id", "")
    tid = record.get("trial_id", "")
    pair = record.get("pair_id", "")
    if pid and tid:
        return f"{pid} / {tid}"
    if pair:
        return pair
    if pid:
        return pid
    if tid:
        return tid
    return "(no id)"


def format_markdown_report(summary: dict[str, Any]) -> str:
    """Return a Markdown string for the given *summary* dict."""
    total: int = summary["total"]
    flagged_items: list[dict[str, Any]] = summary["flagged"]
    counts_by_flag: dict[str, int] = summary["counts_by_flag"]
    counts_by_label: dict[str, int] = summary["counts_by_label"]

    n_flagged = len(flagged_items)
    pct = (n_flagged / total * 100) if total else 0.0

    lines: list[str] = []

    lines.append("# Label Disagreement / Ambiguity Report")
    lines.append("")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append(
        "> Flags are based on metadata signals in the existing label file only.  "
    )
    lines.append(
        "> No labels have been modified. No external annotators are implied.  "
    )
    lines.append(
        "> Routine workflow statuses such as `needs_spotcheck`, `draft`, and `review`  "
    )
    lines.append(
        "> are **not** treated as ambiguity signals by themselves."
    )
    lines.append("")

    # --- Summary table ---
    lines.append("## Summary")
    lines.append("")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Total label records | {total} |")
    lines.append(f"| Flagged records | {n_flagged} |")
    lines.append(f"| Flagged percentage | {pct:.1f}% |")
    lines.append("")

    if n_flagged == 0:
        lines.append("**No ambiguity signals detected.**")
        lines.append("")
        return "\n".join(lines)

    # --- Counts by flag type ---
    lines.append("## Flag Type Counts")
    lines.append("")
    lines.append("| Flag (prefix) | Count |")
    lines.append("|---------------|-------|")
    for flag, count in counts_by_flag.items():
        lines.append(f"| {flag} | {count} |")
    lines.append("")

    # --- Counts by label ---
    lines.append("## Flagged Records by Label")
    lines.append("")
    lines.append("| Label | Count |")
    lines.append("|-------|-------|")
    for label, count in counts_by_label.items():
        lines.append(f"| {label} | {count} |")
    lines.append("")

    # --- Per-label examples ---
    lines.append("## Flagged Examples by Label")
    lines.append("")

    by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in flagged_items:
        label = str(item["record"].get("label", "unknown")).strip().lower()
        by_label[label].append(item)

    for label in sorted(by_label.keys()):
        items = by_label[label]
        lines.append(f"### Label: `{label}` ({len(items)} records)")
        lines.append("")
        for item in items:
            rec = item["record"]
            rid = _record_id(rec)
            flags = item["flags"]
            status = rec.get("label_status", "")
            rationale_raw = rec.get("rationale") or rec.get("explanation") or ""
            preview = rationale_preview(str(rationale_raw))

            lines.append(f"**{rid}**")
            if status:
                lines.append(f"- label_status: `{status}`")
            lines.append(f"- flags: {'; '.join(flags)}")
            if preview:
                lines.append(f"- rationale preview: _{preview}_")
            lines.append("")

    return "\n".join(lines)


def write_text(text: str, path: str) -> None:
    """Write *text* to *path*, creating parent directories as needed."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate label disagreement / ambiguity report (Task 33)."
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        help=f"Path to labels JSON (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"Path for Markdown report (default: {DEFAULT_OUTPUT})",
    )
    args = parser.parse_args()

    data = load_json(args.input)
    records = extract_label_records(data)
    summary = analyze_label_disagreement(records)
    report_text = format_markdown_report(summary)
    write_text(report_text, args.output)

    n_flagged = len(summary["flagged"])
    total = summary["total"]
    pct = (n_flagged / total * 100) if total else 0.0
    print(
        f"Label disagreement report written to: {args.output}\n"
        f"  {total} records analysed, {n_flagged} flagged ({pct:.1f}%)"
    )
    sys.exit(0)


if __name__ == "__main__":
    main()
