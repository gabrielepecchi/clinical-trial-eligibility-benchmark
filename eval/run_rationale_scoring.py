"""
Task 97: Rationale / explanation quality scoring.

Analyzes matcher explanations using three heuristic dimensions:
criterion mention, gold-label consistency, and specificity.

This is a heuristic explanation-quality audit, not a proof of
explanation correctness. Results may include false positives and
false negatives.

Usage:
    PYTHONPATH=. python eval/run_rationale_scoring.py
    PYTHONPATH=. python eval/run_rationale_scoring.py \
        --results  data/processed/results_llm_reviewed.json \
        --labels   data/processed/labels_llm_reviewed.json \
        --criteria data/processed/criterion_level_results.csv \
        --output   reports/rationale_scoring_report.md
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from datetime import datetime
from typing import Any

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_RESULTS = "data/processed/results_llm_reviewed.json"
DEFAULT_LABELS = "data/processed/labels_llm_reviewed.json"
DEFAULT_CRITERIA = "data/processed/criterion_level_results.csv"
DEFAULT_OUTPUT = "reports/rationale_scoring_report.md"

# Generic stopwords stripped before token overlap check
STOPWORDS: frozenset[str] = frozenset(
    {
        "the", "and", "that", "this", "with", "for", "not", "but", "from",
        "are", "was", "has", "have", "had", "been", "will", "would", "could",
        "should", "may", "might", "must", "does", "did", "does", "into",
        "upon", "they", "their", "there", "which", "when", "where", "what",
        "also", "than", "then", "thus", "such", "more", "very", "any",
        "all", "each", "only", "both", "about", "over", "after", "before",
        "who", "whom", "its", "it's", "can", "or", "if", "of", "to", "in",
        "is", "a", "an", "at", "as", "be", "by", "on", "no", "do", "so",
        "we", "he", "she", "his", "her", "our", "your", "you", "patient",
        "trial", "criterion", "criteria", "label", "eligible", "eligibility",
    }
)

MIN_TOKEN_LENGTH = 4

# Keywords indicating criterion/patient-specific detail
SPECIFICITY_KEYWORDS: frozenset[str] = frozenset(
    {
        "age", "medication", "diagnosis", "dbs", "moca", "updrs",
        "hoehn", "pacemaker", "lab", "criterion", "exclusion",
        "inclusion", "dose", "score", "history", "stage", "duration",
        "years", "weeks", "months", "cognitive", "cardiac", "renal",
        "hepatic", "pregnancy", "washout", "stable",
    }
)

# Contradiction signal phrases per gold label
CONTRADICTIONS: dict[str, list[str]] = {
    "eligible": [
        "excluded", "exclusion met", "not eligible", "ineligible",
        "fails criterion", "fails criteria", "does not meet", "violates",
        "disqualified",
    ],
    "not_eligible": [
        "meets all", "satisfies all", "no exclusions", "all criteria met",
        "fully eligible", "meets criteria",
    ],
    "unclear": [
        "clearly eligible", "clearly ineligible", "clearly not eligible",
        "definitively eligible", "definitively not eligible",
    ],
}


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def load_json(path: str, required: bool = False) -> Any:
    """Load JSON from *path*. Exits non-zero when *required* and missing or
    malformed; returns None silently when not required."""
    if not os.path.isfile(path):
        if required:
            print(f"ERROR: required file not found: {path}", file=sys.stderr)
            sys.exit(1)
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except json.JSONDecodeError as exc:
        print(f"ERROR: malformed JSON in {path}: {exc}", file=sys.stderr)
        sys.exit(1)


def load_csv_rows(path: str, required: bool = False) -> list[dict[str, str]] | None:
    """Load CSV as list of dicts. Returns None if missing and not required."""
    if not os.path.isfile(path):
        if required:
            print(f"ERROR: required file not found: {path}", file=sys.stderr)
            sys.exit(1)
        return None
    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def write_text(text: str, path: str) -> None:
    """Write *text* to *path*, creating parent directories as needed."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)


# ---------------------------------------------------------------------------
# Data extraction
# ---------------------------------------------------------------------------


def extract_predictions(data: Any) -> list[dict[str, Any]]:
    """Return a flat list of prediction records from *data*."""
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("predictions", "results", "pairs", "evaluations"):
            if key in data and isinstance(data[key], list):
                return data[key]
        candidates = [v for v in data.values() if isinstance(v, dict)]
        if candidates:
            return candidates
    print("ERROR: unrecognised structure in results file.", file=sys.stderr)
    sys.exit(1)


def extract_label_records(data: Any) -> list[dict[str, Any]]:
    """Return a flat list of label records from *data*."""
    if data is None:
        return []
    if isinstance(data, list):
        return [r for r in data if isinstance(r, dict)]
    if isinstance(data, dict):
        for key in ("labels", "records", "pairs", "cases"):
            if key in data and isinstance(data[key], list):
                return [r for r in data[key] if isinstance(r, dict)]
        return [v for v in data.values() if isinstance(v, dict)]
    return []


def pair_key(record: dict[str, Any]) -> str:
    """Return a stable lookup key for *record*."""
    pid = str(record.get("patient_id", "")).strip()
    tid = str(record.get("trial_id", "")).strip()
    pair = str(record.get("pair_id", "")).strip()
    if pair:
        return pair
    if pid and tid:
        return f"{pid}__{tid}"
    return pid or tid


def index_labels(labels_data: Any) -> dict[str, dict[str, Any]]:
    """Return {pair_key: label_record}."""
    return {pair_key(r): r for r in extract_label_records(labels_data)}


def index_criterion_text(rows: list[dict[str, str]] | None) -> dict[str, str]:
    """
    Return {trial_id: concatenated_criterion_text} from CSV rows.

    Returns an empty dict if *rows* is None.
    """
    if not rows:
        return {}
    # Detect criterion text column
    crit_col: str | None = None
    if rows:
        for candidate in ("criterion", "criterion_text", "text", "eligibility_criterion"):
            if candidate in rows[0]:
                crit_col = candidate
                break
    if not crit_col:
        return {}

    by_trial: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        tid = row.get("trial_id", "").strip()
        val = row.get(crit_col, "").strip()
        if tid and val:
            by_trial[tid].append(val)
    return {tid: " ".join(texts).lower() for tid, texts in by_trial.items()}


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------


def normalize_text(value: Any) -> str:
    """Return a normalized lowercase text representation of *value*."""
    if value is None:
        return ""
    if isinstance(value, list):
        text = " ".join(str(i).strip() for i in value if i)
    elif isinstance(value, dict):
        text = " ".join(str(v).strip() for v in value.values() if v is not None)
    else:
        text = str(value)
    return text.lower().strip()


def preview_text(value: Any, max_chars: int = 180) -> str:
    """Return a short plain-text preview of *value*."""
    text = normalize_text(value).replace("\n", " ")
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "…"


def collect_explanation_text(
    record: dict[str, Any],
    label_record: dict[str, Any] | None = None,
) -> str:
    """Collect all explanation-related text from *record* and *label_record*."""
    parts: list[str] = []
    for field in (
        "explanation", "matcher_explanation", "reason", "reasoning",
        "rationale", "reasoning_trace", "blocking_criteria", "uncertain_criteria",
    ):
        val = record.get(field, "")
        if val:
            parts.append(normalize_text(val))
    if label_record:
        for field in ("rationale", "explanation"):
            val = label_record.get(field, "")
            if val:
                parts.append(normalize_text(val))
    return " ".join(parts).strip()


def meaningful_tokens(text: str) -> set[str]:
    """Return lowercase tokens of length >= MIN_TOKEN_LENGTH not in STOPWORDS."""
    raw = text.lower().split()
    return {
        t.strip(".,;:()[]\"'-")
        for t in raw
        if len(t.strip(".,;:()[]\"'-")) >= MIN_TOKEN_LENGTH
        and t.strip(".,;:()[]\"'-") not in STOPWORDS
    }


# ---------------------------------------------------------------------------
# Dimension scorers
# ---------------------------------------------------------------------------


def score_criterion_mention(
    record: dict[str, Any],
    explanation_text: str,
    criterion_index: dict[str, str],
) -> tuple[int | None, str]:
    """
    Return (score, reason) for the criterion-mention dimension.

    score is None if criterion_index is empty (dimension unavailable).
    """
    if not criterion_index:
        return None, "criterion_level_results.csv not available"

    tid = str(record.get("trial_id", "")).strip()
    criterion_text = criterion_index.get(tid, "")

    if not criterion_text:
        return 0, f"no criterion text found for trial_id={tid!r}"

    if not explanation_text:
        return 0, "explanation text is empty"

    exp_tokens = meaningful_tokens(explanation_text)
    crit_tokens = meaningful_tokens(criterion_text)
    overlap = exp_tokens & crit_tokens

    if overlap:
        sample = sorted(overlap)[:3]
        return 1, f"overlap tokens: {sample}"
    return 0, "no meaningful token overlap with criterion text"


def score_gold_label_consistency(
    record: dict[str, Any],
    explanation_text: str,
) -> tuple[int, str]:
    """
    Return (score, reason) for gold-label consistency.

    1 if no obvious contradiction detected, 0 if possible contradiction found.
    """
    gold = str(record.get("gold_label", record.get("gold", ""))).strip().lower()

    if not explanation_text or not gold:
        return 1, "no contradiction check possible (empty explanation or gold label)"

    for phrase in CONTRADICTIONS.get(gold, []):
        if phrase in explanation_text:
            return 0, f"possible contradiction: gold={gold!r} but explanation contains {phrase!r}"

    return 1, "no obvious contradiction detected"


def score_specificity(explanation_text: str) -> tuple[int, str]:
    """
    Return (score, reason) for specificity.

    1 if explanation has >= 12 words OR mentions a specificity keyword.
    """
    if not explanation_text:
        return 0, "explanation text is empty"

    word_count = len(explanation_text.split())
    if word_count >= 12:
        return 1, f"explanation has {word_count} words"

    found = [kw for kw in SPECIFICITY_KEYWORDS if kw in explanation_text]
    if found:
        return 1, f"contains specificity keyword(s): {found[:3]}"

    return 0, f"explanation has only {word_count} words and no specificity keywords"


# ---------------------------------------------------------------------------
# Per-record scoring
# ---------------------------------------------------------------------------


def score_rationale(
    record: dict[str, Any],
    label_record: dict[str, Any] | None = None,
    criterion_index: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Return a full scoring dict for *record*."""
    ci = criterion_index or {}
    explanation_text = collect_explanation_text(record, label_record)

    crit_score, crit_reason = score_criterion_mention(record, explanation_text, ci)
    cons_score, cons_reason = score_gold_label_consistency(record, explanation_text)
    spec_score, spec_reason = score_specificity(explanation_text)

    flags: list[str] = []
    if crit_score == 0:
        flags.append("possible criterion mention issue")
    if cons_score == 0:
        flags.append("possible gold-label contradiction")
    if spec_score == 0:
        flags.append("low specificity")
    if not explanation_text:
        flags.append("no explanation text")

    available_dims = [s for s in (crit_score, cons_score, spec_score) if s is not None]
    total = sum(s for s in (crit_score, cons_score, spec_score) if s is not None)
    available_score = total

    # Total score only counts available dimensions
    total_score = total

    gold = str(record.get("gold_label", record.get("gold", ""))).strip().lower()
    predicted = str(
        record.get("predicted_label", record.get("prediction", ""))
    ).strip().lower()

    rationale_preview = preview_text(
        label_record.get("rationale", label_record.get("explanation", ""))
        if label_record else ""
    )

    return {
        "patient_id": record.get("patient_id", ""),
        "trial_id": record.get("trial_id", ""),
        "gold_label": gold,
        "predicted_label": predicted,
        "confidence": record.get("confidence", record.get("confidence_score")),
        "total_score": total_score,
        "available_score": available_score,
        "available_dims": len(available_dims),
        "criterion_mention_score": crit_score,
        "criterion_mention_reason": crit_reason,
        "gold_consistency_score": cons_score,
        "gold_consistency_reason": cons_reason,
        "specificity_score": spec_score,
        "specificity_reason": spec_reason,
        "rationale_quality_flags": flags,
        "explanation_preview": preview_text(explanation_text),
        "rationale_preview": rationale_preview,
    }


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def analyze_rationales(
    predictions: list[dict[str, Any]],
    label_index: dict[str, dict[str, Any]],
    criterion_index: dict[str, str],
) -> dict[str, Any]:
    """Score all prediction rationales and return summary."""
    scored: list[dict[str, Any]] = []

    for pred in predictions:
        key = pair_key(pred)
        label_rec = label_index.get(key)
        result = score_rationale(pred, label_rec, criterion_index)
        scored.append(result)

    records_with_explanation = sum(1 for s in scored if s["explanation_preview"])

    available_scores = [s["available_score"] for s in scored]
    avg_score = sum(available_scores) / len(available_scores) if available_scores else 0.0

    # Score distribution
    dist: dict[int, int] = defaultdict(int)
    for s in scored:
        dist[s["total_score"]] += 1

    # Flag counts
    flag_counts: dict[str, int] = defaultdict(int)
    for s in scored:
        for flag in s["rationale_quality_flags"]:
            flag_counts[flag] += 1

    # Sort by score ascending (worst first)
    sorted_scored = sorted(
        scored,
        key=lambda x: (x["total_score"], str(x["patient_id"])),
    )

    contradictions = [s for s in scored if s["gold_consistency_score"] == 0]
    low_specificity = [s for s in scored if s["specificity_score"] == 0]
    no_explanation = [s for s in scored if not s["explanation_preview"]]

    return {
        "total": len(scored),
        "records_with_explanation": records_with_explanation,
        "avg_available_score": avg_score,
        "score_distribution": dict(sorted(dist.items())),
        "flag_counts": dict(sorted(flag_counts.items(), key=lambda x: -x[1])),
        "bottom25": sorted_scored[:25],
        "contradictions": contradictions[:15],
        "low_specificity": low_specificity[:15],
        "no_explanation": no_explanation[:15],
        "criterion_available": bool(criterion_index),
    }


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------


def _display_id(item: dict[str, Any]) -> str:
    pid = item.get("patient_id", "")
    tid = item.get("trial_id", "")
    if pid and tid:
        return f"{pid} / {tid}"
    return pid or tid or "(no id)"


def _score_badge(item: dict[str, Any]) -> str:
    crit = item["criterion_mention_score"]
    crit_str = str(crit) if crit is not None else "N/A"
    return (
        f"score={item['total_score']} "
        f"[crit={crit_str} | consist={item['gold_consistency_score']} | "
        f"spec={item['specificity_score']}]"
    )


def format_markdown_report(summary: dict[str, Any]) -> str:
    lines: list[str] = []

    lines.append("# Rationale Quality Scoring Report")
    lines.append("")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append(
        "> This report scores matcher explanations on three heuristic dimensions.  "
    )
    lines.append("> Results are **indicative only** and may include false positives and")
    lines.append("> false negatives. This is not a proof of explanation correctness.")
    if not summary["criterion_available"]:
        lines.append(
            "> **Note:** criterion_level_results.csv was not available; "
            "criterion-mention dimension is marked N/A."
        )
    lines.append("")

    lines.append("## Summary")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Total records analyzed | {summary['total']} |")
    lines.append(f"| Records with explanation text | {summary['records_with_explanation']} |")
    lines.append(f"| Average heuristic available score | {summary['avg_available_score']:.2f} |")
    lines.append(f"| Possible gold-label contradictions | {len(summary['contradictions'])} |")
    lines.append(f"| Low-specificity explanations | {len(summary['low_specificity'])} |")
    lines.append(f"| Records with no explanation | {len(summary['no_explanation'])} |")
    lines.append("")

    lines.append("## Score Distribution")
    lines.append("")
    lines.append("| Heuristic Score | Count |")
    lines.append("|-----------------|-------|")
    for score_val, count in summary["score_distribution"].items():
        lines.append(f"| {score_val} | {count} |")
    lines.append("")

    lines.append("## Rationale Quality Flag Counts")
    lines.append("")
    lines.append("| Flag | Count |")
    lines.append("|------|-------|")
    for flag, count in summary["flag_counts"].items():
        lines.append(f"| {flag} | {count} |")
    lines.append("")

    lines.append("## Bottom 25 Rationale Scores")
    lines.append("")
    for i, item in enumerate(summary["bottom25"], 1):
        rid = _display_id(item)
        lines.append(f"{i}. **{rid}** — {_score_badge(item)}")
        if item["rationale_quality_flags"]:
            flags = "; ".join(item["rationale_quality_flags"])
            lines.append(f"   - flags: {flags}")
        if item["explanation_preview"]:
            lines.append(f"   - _{item['explanation_preview']}_")
    lines.append("")

    lines.append("## Possible Gold-Label Contradiction Examples")
    lines.append("")
    if not summary["contradictions"]:
        lines.append("*None detected.*")
        lines.append("")
    else:
        for item in summary["contradictions"]:
            rid = _display_id(item)
            lines.append(f"- **{rid}**: gold=`{item['gold_label']}`, "
                         f"predicted=`{item['predicted_label']}`")
            lines.append(f"  - reason: _{item['gold_consistency_reason']}_")
            if item["explanation_preview"]:
                lines.append(f"  - _{item['explanation_preview']}_")
        lines.append("")

    lines.append("## Low-Specificity Examples")
    lines.append("")
    if not summary["low_specificity"]:
        lines.append("*None detected.*")
        lines.append("")
    else:
        for item in summary["low_specificity"][:10]:
            rid = _display_id(item)
            lines.append(f"- **{rid}**: _{item['specificity_reason']}_")
            if item["explanation_preview"]:
                lines.append(f"  - _{item['explanation_preview']}_")
        lines.append("")

    if summary["no_explanation"]:
        lines.append("## Records With No Explanation Text")
        lines.append("")
        for item in summary["no_explanation"]:
            rid = _display_id(item)
            lines.append(
                f"- **{rid}**: gold=`{item['gold_label']}`, "
                f"predicted=`{item['predicted_label']}`"
            )
        lines.append("")

    lines.append("## Limitations")
    lines.append("")
    lines.append(
        "- **False positives**: The gold-consistency check uses simple phrase matching "
        "and may flag explanations that use technical language correctly."
    )
    lines.append(
        "- **False negatives**: Subtle contradictions or missing criterion mentions "
        "expressed indirectly will not be caught."
    )
    lines.append(
        "- **Criterion overlap**: Token overlap with criterion_level_results.csv "
        "is a weak proxy; a high overlap score does not guarantee the explanation "
        "is correct or relevant."
    )
    lines.append(
        "- **Specificity**: Word count and keyword presence are rough proxies. "
        "A long generic explanation may score 1 while a short precise one scores 0."
    )
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rationale quality scoring report (Task 97)."
    )
    parser.add_argument(
        "--results", default=DEFAULT_RESULTS,
        help=f"Path to results JSON (default: {DEFAULT_RESULTS})",
    )
    parser.add_argument(
        "--labels", default=DEFAULT_LABELS,
        help=f"Path to labels JSON (default: {DEFAULT_LABELS})",
    )
    parser.add_argument(
        "--criteria", default=DEFAULT_CRITERIA,
        help=f"Path to criterion CSV (default: {DEFAULT_CRITERIA})",
    )
    parser.add_argument(
        "--output", default=DEFAULT_OUTPUT,
        help=f"Output Markdown path (default: {DEFAULT_OUTPUT})",
    )
    args = parser.parse_args()

    results_data = load_json(args.results, required=True)
    labels_data = load_json(args.labels, required=False)
    criterion_rows = load_csv_rows(args.criteria, required=False)

    predictions = extract_predictions(results_data)
    label_index = index_labels(labels_data)
    criterion_index = index_criterion_text(criterion_rows)

    summary = analyze_rationales(predictions, label_index, criterion_index)
    report = format_markdown_report(summary)
    write_text(report, args.output)

    print(
        f"Rationale scoring report written to: {args.output}\n"
        f"  Records analyzed        : {summary['total']}\n"
        f"  Records with explanation: {summary['records_with_explanation']}\n"
        f"  Avg heuristic score     : {summary['avg_available_score']:.2f}"
    )
    sys.exit(0)


if __name__ == "__main__":
    main()
