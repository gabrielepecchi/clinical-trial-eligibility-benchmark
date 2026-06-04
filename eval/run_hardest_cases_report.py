"""
Task 67: Hardest cases report.

Reads prediction results and optional error analysis, scores each record by
deterministic difficulty, and writes a Markdown report of the hardest cases.

Usage:
    PYTHONPATH=. python eval/run_hardest_cases_report.py
    PYTHONPATH=. python eval/run_hardest_cases_report.py \
        --results data/processed/results_llm_reviewed.json \
        --errors  data/processed/error_analysis_llm_reviewed.json \
        --output  reports/hardest_cases_report.md
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
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_RESULTS = "data/processed/results_llm_reviewed.json"
DEFAULT_ERRORS = "data/processed/error_analysis_llm_reviewed.json"
DEFAULT_OUTPUT = "reports/hardest_cases_report.md"

TOP_N = 25


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def load_json(path: str, required: bool = False) -> Any:
    """Load JSON from *path*. Exits non-zero when *required* and file is
    missing or malformed; returns None silently when not required and absent."""
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


def index_error_analysis(error_data: Any) -> dict[str, dict[str, Any]]:
    """Return {pair_key: error_record} from *error_data*; empty dict if None."""
    if error_data is None:
        return {}
    records: list[dict[str, Any]] = []
    if isinstance(error_data, list):
        records = error_data
    elif isinstance(error_data, dict):
        for key in ("errors", "error_analysis", "predictions", "pairs"):
            if key in error_data and isinstance(error_data[key], list):
                records = error_data[key]
                break
        else:
            records = [v for v in error_data.values() if isinstance(v, dict)]
    return {pair_key(r): r for r in records if isinstance(r, dict)}


# ---------------------------------------------------------------------------
# Field helpers
# ---------------------------------------------------------------------------


def get_gold_label(record: dict[str, Any]) -> str:
    return str(record.get("gold_label", record.get("gold", ""))).strip().lower()


def get_predicted_label(record: dict[str, Any]) -> str:
    return str(
        record.get("predicted_label", record.get("prediction", record.get("predicted", "")))
    ).strip().lower()


def parse_confidence(value: Any) -> float | None:
    """Return confidence as a float in [0, 1], or None if absent/malformed."""
    if value is None or value == "":
        return None
    try:
        f = float(value)
        if 0.0 <= f <= 1.0:
            return f
        return None
    except (TypeError, ValueError):
        return None


def preview_text(value: Any, max_chars: int = 180) -> str:
    """Return a short plain-text preview of *value*."""
    if value is None:
        return ""
    if isinstance(value, list):
        text = "; ".join(str(i).strip() for i in value if i)
    elif isinstance(value, dict):
        text = "; ".join(f"{k}: {v}" for k, v in value.items() if v is not None)
    else:
        text = str(value)
    text = text.strip().replace("\n", " ").replace("\r", " ")
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "…"


# ---------------------------------------------------------------------------
# Difficulty scoring
# ---------------------------------------------------------------------------


def compute_difficulty_score(
    record: dict[str, Any],
    error_record: dict[str, Any] | None = None,
) -> tuple[int, list[str]]:
    """
    Return (score, reasons) for *record*.

    All scoring rules are deterministic and based on existing fields only.
    """
    err = error_record or {}
    score = 0
    reasons: list[str] = []

    gold = get_gold_label(record)
    predicted = get_predicted_label(record)
    incorrect = gold != predicted

    conf_value = record.get("confidence", record.get("confidence_score"))
    confidence = parse_confidence(conf_value)

    error_type = str(err.get("error_type", record.get("error_type", ""))).strip().lower()
    severity = str(err.get("severity", record.get("severity", ""))).strip().lower()
    blocking = record.get("blocking_criteria")
    uncertain = record.get("uncertain_criteria")

    # +5 incorrect
    if incorrect:
        score += 5
        reasons.append("incorrect prediction (+5)")

    # +5 high-risk false eligible
    if gold == "not_eligible" and predicted == "eligible":
        score += 5
        reasons.append("high-risk false eligible: gold=not_eligible, predicted=eligible (+5)")

    # +3 predicted unclear when gold is definite
    if predicted == "unclear" and gold not in ("unclear", ""):
        score += 3
        reasons.append("predicted unclear but gold is definite (+3)")

    # +3 gold unclear but predicted definite
    if gold == "unclear" and predicted not in ("unclear", ""):
        score += 3
        reasons.append("gold unclear but predicted definite (+3)")

    # +2 confidence missing or malformed
    if confidence is None:
        score += 2
        reasons.append("confidence missing or malformed (+2)")
    else:
        # +2 confidence < 0.50
        if confidence < 0.50:
            score += 2
            reasons.append(f"low confidence {confidence:.2f} (+2)")
        # +2 incorrect with high confidence
        if incorrect and confidence >= 0.75:
            score += 2
            reasons.append(f"incorrect with high confidence {confidence:.2f} (+2)")

    # +1 error_type present
    if error_type:
        score += 1
        reasons.append(f"error_type present: {error_type} (+1)")

    # +1 severity major or critical
    if severity in ("major", "critical"):
        score += 1
        reasons.append(f"severity={severity} (+1)")

    # +1 blocking_criteria non-empty
    if blocking and preview_text(blocking):
        score += 1
        reasons.append("blocking_criteria present (+1)")

    # +1 uncertain_criteria non-empty
    if uncertain and preview_text(uncertain):
        score += 1
        reasons.append("uncertain_criteria present (+1)")

    return score, reasons


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def analyze_hardest_cases(
    predictions: list[dict[str, Any]],
    error_index: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Score all predictions and return a summary dict."""
    scored: list[dict[str, Any]] = []

    for pred in predictions:
        key = pair_key(pred)
        err = error_index.get(key, {})
        score, reasons = compute_difficulty_score(pred, err)

        gold = get_gold_label(pred)
        predicted = get_predicted_label(pred)
        incorrect = gold != predicted

        conf_value = pred.get("confidence", pred.get("confidence_score"))
        confidence = parse_confidence(conf_value)

        scored.append(
            {
                "patient_id": pred.get("patient_id", ""),
                "trial_id": pred.get("trial_id", ""),
                "pair_id": pred.get("pair_id", ""),
                "gold_label": gold,
                "predicted_label": predicted,
                "correct": not incorrect,
                "confidence": confidence,
                "confidence_raw": conf_value,
                "difficulty_score": score,
                "difficulty_reasons": reasons,
                "error_type": str(err.get("error_type", pred.get("error_type", ""))).strip(),
                "severity": str(err.get("severity", pred.get("severity", ""))).strip(),
                "explanation_preview": preview_text(
                    pred.get("explanation", pred.get("matcher_explanation", ""))
                ),
                "blocking_preview": preview_text(pred.get("blocking_criteria", "")),
                "uncertain_preview": preview_text(pred.get("uncertain_criteria", "")),
                "incorrect": incorrect,
            }
        )

    scored.sort(key=lambda x: (-x["difficulty_score"], str(x["patient_id"]), str(x["trial_id"])))

    total = len(scored)
    total_incorrect = sum(1 for s in scored if s["incorrect"])
    scores = [s["difficulty_score"] for s in scored]
    avg_score = sum(scores) / total if total else 0.0
    max_score = max(scores) if scores else 0

    # Distribution buckets
    dist: dict[int, int] = defaultdict(int)
    for s in scores:
        dist[s] += 1

    # Sub-lists
    top25 = scored[:TOP_N]
    false_eligible = [
        s for s in scored
        if s["gold_label"] == "not_eligible" and s["predicted_label"] == "eligible"
    ]
    unclear_related = [
        s for s in scored
        if (s["predicted_label"] == "unclear" and s["gold_label"] != "unclear")
        or (s["gold_label"] == "unclear" and s["predicted_label"] != "unclear")
    ]
    high_conf_incorrect = [
        s for s in scored
        if s["incorrect"]
        and s["confidence"] is not None
        and s["confidence"] >= 0.75
    ]

    return {
        "total": total,
        "total_incorrect": total_incorrect,
        "avg_score": avg_score,
        "max_score": max_score,
        "score_distribution": dict(sorted(dist.items())),
        "top25": top25,
        "false_eligible": false_eligible[:10],
        "unclear_related": unclear_related[:10],
        "high_conf_incorrect": high_conf_incorrect[:10],
    }


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------


def _display_id(item: dict[str, Any]) -> str:
    pid = item.get("patient_id", "")
    tid = item.get("trial_id", "")
    pair = item.get("pair_id", "")
    if pid and tid:
        return f"{pid} / {tid}"
    return pair or pid or tid or "(no id)"


def _case_block(item: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    lines.append(f"**{_display_id(item)}** — difficulty score: `{item['difficulty_score']}`")
    lines.append(
        f"- gold: `{item['gold_label']}` | predicted: `{item['predicted_label']}` | "
        f"correct: `{item['correct']}`"
    )
    if item["confidence"] is not None:
        lines.append(f"- confidence: `{item['confidence']:.2f}`")
    if item["error_type"]:
        lines.append(f"- error_type: `{item['error_type']}`")
    if item["severity"]:
        lines.append(f"- severity: `{item['severity']}`")
    if item["difficulty_reasons"]:
        lines.append("- reasons: " + "; ".join(item["difficulty_reasons"]))
    if item["explanation_preview"]:
        lines.append(f"- explanation: _{item['explanation_preview']}_")
    if item["blocking_preview"]:
        lines.append(f"- blocking criteria: _{item['blocking_preview']}_")
    if item["uncertain_preview"]:
        lines.append(f"- uncertain criteria: _{item['uncertain_preview']}_")
    lines.append("")
    return lines


def format_markdown_report(summary: dict[str, Any]) -> str:
    lines: list[str] = []

    total: int = summary["total"]
    total_incorrect: int = summary["total_incorrect"]
    avg_score: float = summary["avg_score"]
    max_score: int = summary["max_score"]
    dist: dict[int, int] = summary["score_distribution"]
    top25: list[dict[str, Any]] = summary["top25"]
    false_eligible: list[dict[str, Any]] = summary["false_eligible"]
    unclear_related: list[dict[str, Any]] = summary["unclear_related"]
    high_conf_incorrect: list[dict[str, Any]] = summary["high_conf_incorrect"]

    lines.append("# Hardest Cases Report")
    lines.append("")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append(
        "> Difficulty scores are deterministic and based on existing prediction fields only.  "
    )
    lines.append("> No labels, errors, or clinical facts have been modified or invented.")
    lines.append("")

    lines.append("## Summary")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Total records | {total} |")
    lines.append(f"| Incorrect records | {total_incorrect} |")
    lines.append(f"| Average difficulty score | {avg_score:.2f} |")
    lines.append(f"| Max difficulty score | {max_score} |")
    lines.append(f"| High-risk false eligible cases | {len(summary['false_eligible'])} |")
    lines.append(f"| Unclear/abstention-related cases | {len(summary['unclear_related'])} |")
    lines.append(f"| High-confidence incorrect cases | {len(summary['high_conf_incorrect'])} |")
    lines.append("")

    if total == 0:
        lines.append("**No records found in the results file.**")
        lines.append("")
        return "\n".join(lines)

    # Score distribution
    lines.append("## Difficulty Score Distribution")
    lines.append("")
    lines.append("| Score | Count |")
    lines.append("|-------|-------|")
    for score_val, count in sorted(dist.items()):
        lines.append(f"| {score_val} | {count} |")
    lines.append("")

    # Top 25
    lines.append(f"## Top {len(top25)} Hardest Cases")
    lines.append("")
    if not top25:
        lines.append("*No cases scored above zero.*")
        lines.append("")
    else:
        for item in top25:
            lines.extend(_case_block(item))

    # High-risk false eligible
    lines.append("## High-Risk False Eligible Cases")
    lines.append("")
    lines.append(
        "Records where gold label is `not_eligible` but prediction is `eligible`."
    )
    lines.append("")
    if not false_eligible:
        lines.append("*None found.*")
        lines.append("")
    else:
        for item in false_eligible:
            lines.extend(_case_block(item))

    # Unclear / abstention-related
    lines.append("## Unclear / Abstention-Related Cases")
    lines.append("")
    lines.append(
        "Records where predicted and gold labels disagree on `unclear`."
    )
    lines.append("")
    if not unclear_related:
        lines.append("*None found.*")
        lines.append("")
    else:
        for item in unclear_related:
            lines.extend(_case_block(item))

    # High-confidence incorrect
    lines.append("## High-Confidence Incorrect Cases")
    lines.append("")
    lines.append("Records where the prediction was incorrect with confidence >= 0.75.")
    lines.append("")
    if not high_conf_incorrect:
        lines.append("*None found.*")
        lines.append("")
    else:
        for item in high_conf_incorrect:
            lines.extend(_case_block(item))

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hardest cases report (Task 67)."
    )
    parser.add_argument(
        "--results", default=DEFAULT_RESULTS,
        help=f"Path to results JSON (default: {DEFAULT_RESULTS})",
    )
    parser.add_argument(
        "--errors", default=DEFAULT_ERRORS,
        help=f"Path to error analysis JSON (default: {DEFAULT_ERRORS})",
    )
    parser.add_argument(
        "--output", default=DEFAULT_OUTPUT,
        help=f"Output Markdown path (default: {DEFAULT_OUTPUT})",
    )
    args = parser.parse_args()

    results_data = load_json(args.results, required=True)
    error_data = load_json(args.errors, required=False)

    predictions = extract_predictions(results_data)
    error_index = index_error_analysis(error_data)

    summary = analyze_hardest_cases(predictions, error_index)
    report = format_markdown_report(summary)
    write_text(report, args.output)

    print(
        f"Hardest cases report written to: {args.output}\n"
        f"  Records read     : {summary['total']}\n"
        f"  Incorrect records: {summary['total_incorrect']}\n"
        f"  Max difficulty   : {summary['max_score']}"
    )
    sys.exit(0)


if __name__ == "__main__":
    main()
