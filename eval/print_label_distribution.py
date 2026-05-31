"""Print label distribution report from LLM-reviewed benchmark results.

Usage:
    PYTHONPATH=. python eval/print_label_distribution.py
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

RESULTS_FILE = Path("data/processed/results_llm_reviewed.json")
LABEL_ORDER = ["eligible", "not_eligible", "unclear"]


def load_results(path: Path) -> list[dict]:
    """Load prediction records from results JSON."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        print(f"ERROR: Results file not found: {path}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"ERROR: Malformed JSON in {path}: {e}", file=sys.stderr)
        sys.exit(1)

    if isinstance(data, dict):
        records = data.get("predictions")
        if not isinstance(records, list):
            print(f"ERROR: Expected 'predictions' list in {path}", file=sys.stderr)
            sys.exit(1)
        return records

    if isinstance(data, list):
        return data

    print(f"ERROR: Unexpected format in {path}", file=sys.stderr)
    sys.exit(1)


def count_labels(predictions: list[dict]) -> tuple[dict[str, int], dict[str, int]]:
    """Return (gold_counts, predicted_counts) dicts keyed by label."""
    gold: dict[str, int] = defaultdict(int)
    predicted: dict[str, int] = defaultdict(int)
    for r in predictions:
        gold[r.get("gold_label", "unknown")] += 1
        predicted[r.get("predicted_label", "unknown")] += 1
    return dict(gold), dict(predicted)


def count_error_pairs(predictions: list[dict]) -> dict[tuple[str, str], int]:
    """Return counts of (gold_label, predicted_label) pairs where gold != predicted."""
    pairs: dict[tuple[str, str], int] = defaultdict(int)
    for r in predictions:
        gold = r.get("gold_label", "unknown")
        pred = r.get("predicted_label", "unknown")
        if gold != pred:
            pairs[(gold, pred)] += 1
    return dict(pairs)


def format_label_distribution(
    gold_counts: dict[str, int],
    predicted_counts: dict[str, int],
) -> str:
    """Format gold and predicted label distributions as a terminal table."""
    total_gold = sum(gold_counts.values())
    total_pred = sum(predicted_counts.values())

    all_labels = sorted(
        set(gold_counts) | set(predicted_counts),
        key=lambda l: LABEL_ORDER.index(l) if l in LABEL_ORDER else 99,
    )

    col = 14
    lines = [
        "\n=== Label Distribution ===",
        f"{'label':<{col}}  {'gold':>6}  {'gold %':>7}  {'predicted':>10}  {'pred %':>7}",
        "-" * (col + 38),
    ]
    for label in all_labels:
        g = gold_counts.get(label, 0)
        p = predicted_counts.get(label, 0)
        g_pct = g / total_gold * 100 if total_gold else 0.0
        p_pct = p / total_pred * 100 if total_pred else 0.0
        lines.append(
            f"{label:<{col}}  {g:>6}  {g_pct:>6.1f}%  {p:>10}  {p_pct:>6.1f}%"
        )
    lines.append("-" * (col + 38))
    lines.append(f"{'total':<{col}}  {total_gold:>6}  {'':>7}  {total_pred:>10}")
    return "\n".join(lines)


def format_error_pairs(pair_counts: dict[tuple[str, str], int]) -> str:
    """Format error pair counts as a terminal table."""
    if not pair_counts:
        return "\n=== Error Pairs ===\n  (no errors)"

    total_errors = sum(pair_counts.values())
    sorted_pairs = sorted(pair_counts.items(), key=lambda x: -x[1])

    lines = [
        "\n=== Error Pairs (gold → predicted) ===",
        f"  {'gold':<15}  {'predicted':<15}  {'count':>6}",
        "  " + "-" * 40,
    ]
    for (gold, pred), count in sorted_pairs:
        lines.append(f"  {gold:<15}  {pred:<15}  {count:>6}")
    lines.append("  " + "-" * 40)
    lines.append(f"  {'total errors':<31}  {total_errors:>6}")
    return "\n".join(lines)


def main() -> None:
    predictions = load_results(RESULTS_FILE)
    gold_counts, predicted_counts = count_labels(predictions)
    pair_counts = count_error_pairs(predictions)

    print(f"\nResults file : {RESULTS_FILE}")
    print(f"Total records: {len(predictions)}")
    print(format_label_distribution(gold_counts, predicted_counts))
    print(format_error_pairs(pair_counts))


if __name__ == "__main__":
    main()
