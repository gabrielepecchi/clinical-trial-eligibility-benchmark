"""
run_split_metrics_report.py — Task 38: per-split benchmark metrics report.

Usage:
    PYTHONPATH=. python eval/run_split_metrics_report.py
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

RESULTS_PATH = Path("data/processed/results_llm_reviewed.json")
SPLITS_PATH  = Path("data/processed/labels_llm_reviewed_with_splits.json")
REPORT_PATH  = Path("reports/split_metrics_report.json")

VALID_LABELS = ["eligible", "not_eligible", "unclear"]


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def write_json(data, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)


def build_split_index(split_records: list) -> dict[tuple, str]:
    """Return {(patient_id, trial_id): split} from labels_with_splits."""
    return {
        (r["patient_id"], r["trial_id"]): r["split"]
        for r in split_records
        if isinstance(r, dict) and r.get("patient_id") and r.get("trial_id") and r.get("split")
    }


def compute_metrics(gold: list[str], predicted: list[str]) -> dict:
    n = len(gold)
    if n == 0:
        return {
            "total_pairs": 0, "correct": 0, "error_count": 0,
            "accuracy": None, "macro_f1": None,
            "gold_label_distribution": {}, "predicted_label_distribution": {},
        }

    correct = sum(g == p for g, p in zip(gold, predicted))

    per_class: dict[str, dict] = {}
    for cls in VALID_LABELS:
        tp = sum(g == cls and p == cls for g, p in zip(gold, predicted))
        fp = sum(g != cls and p == cls for g, p in zip(gold, predicted))
        fn = sum(g == cls and p != cls for g, p in zip(gold, predicted))
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall    = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) else 0.0)
        per_class[cls] = f1

    macro_f1 = sum(per_class.values()) / 3

    gold_dist: dict[str, int] = defaultdict(int)
    pred_dist: dict[str, int] = defaultdict(int)
    for g, p in zip(gold, predicted):
        gold_dist[g] += 1
        pred_dist[p] += 1

    return {
        "total_pairs":                n,
        "correct":                    correct,
        "error_count":                n - correct,
        "accuracy":                   correct / n,
        "macro_f1":                   macro_f1,
        "gold_label_distribution":    dict(gold_dist),
        "predicted_label_distribution": dict(pred_dist),
    }


def main() -> None:
    try:
        results_data = load_json(RESULTS_PATH)
    except FileNotFoundError:
        print(f"[ERROR] File not found: {RESULTS_PATH}")
        sys.exit(1)
    except json.JSONDecodeError as exc:
        print(f"[ERROR] Invalid JSON in {RESULTS_PATH}: {exc}")
        sys.exit(1)

    try:
        split_records = load_json(SPLITS_PATH)
    except FileNotFoundError:
        print(f"[ERROR] File not found: {SPLITS_PATH}")
        print("Run scripts/create_train_dev_test_split.py first.")
        sys.exit(1)
    except json.JSONDecodeError as exc:
        print(f"[ERROR] Invalid JSON in {SPLITS_PATH}: {exc}")
        sys.exit(1)

    predictions = results_data.get("predictions", []) if isinstance(results_data, dict) else []
    if not predictions:
        print("[ERROR] No predictions found in results file.")
        sys.exit(1)

    split_index = build_split_index(split_records)

    # Group predictions by split
    split_gold: dict[str, list] = defaultdict(list)
    split_pred: dict[str, list] = defaultdict(list)
    unmatched = 0

    for rec in predictions:
        if not isinstance(rec, dict):
            continue
        pid  = rec.get("patient_id", "")
        tid  = rec.get("trial_id", "")
        gold = rec.get("gold_label", "")
        pred = rec.get("predicted_label", "")
        if not (pid and tid and gold and pred):
            continue
        split = split_index.get((pid, tid))
        if split is None:
            unmatched += 1
            continue
        split_gold[split].append(gold)
        split_pred[split].append(pred)

    # Per-split metrics
    split_metrics: dict[str, dict] = {}
    all_gold: list[str] = []
    all_pred: list[str] = []

    for split in sorted(split_gold.keys()):
        g = split_gold[split]
        p = split_pred[split]
        split_metrics[split] = compute_metrics(g, p)
        all_gold.extend(g)
        all_pred.extend(p)

    overall = compute_metrics(all_gold, all_pred)

    report = {
        "unmatched_predictions": unmatched,
        "overall": overall,
        "splits": split_metrics,
    }

    write_json(report, REPORT_PATH)

    def _fmt(v) -> str:
        return f"{v:.4f}" if isinstance(v, float) else "n/a"

    print(f"{'split':<10}  {'pairs':<7}  {'correct':<8}  {'accuracy':<10}  {'macro_f1'}")
    for split in ("train", "dev", "test"):
        m = split_metrics.get(split, {})
        print(
            f"{split:<10}  {m.get('total_pairs', 0):<7}  {m.get('correct', 0):<8}  "
            f"{_fmt(m.get('accuracy')):<10}  {_fmt(m.get('macro_f1'))}"
        )
    print(
        f"{'overall':<10}  {overall['total_pairs']:<7}  {overall['correct']:<8}  "
        f"{_fmt(overall.get('accuracy')):<10}  {_fmt(overall.get('macro_f1'))}"
    )
    if unmatched:
        print(f"Unmatched predictions (no split label): {unmatched}")
    print(f"Report written to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
