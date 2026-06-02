"""
run_clean_vs_noisy_input_report.py — Task 92: clean vs narrative vs noisy input comparison.

NOTE: This is a synthetic-data robustness comparison, not clinical validation.
All patient inputs are synthetic and do not represent real individuals.

Usage:
    PYTHONPATH=. python eval/run_clean_vs_noisy_input_report.py
"""

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

CLEAN_PATH     = Path("data/processed/patient_cases.json")
NARRATIVE_PATH = Path("data/processed/patient_cases_narrative.json")
NOISY_PATH     = Path("data/processed/patient_cases_noisy.json")
TRIALS_PATH    = Path("data/processed/trial_cases.json")
LABELS_PATH    = Path("data/processed/labels_llm_reviewed.json")
REPORT_PATH    = Path("reports/clean_vs_noisy_input_report.json")

VALID_LABELS = ["eligible", "not_eligible", "unclear"]

REQUIRED_FILES = {
    "clean":     CLEAN_PATH,
    "narrative": NARRATIVE_PATH,
    "noisy":     NOISY_PATH,
    "trials":    TRIALS_PATH,
    "labels":    LABELS_PATH,
}


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def write_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)


# ---------------------------------------------------------------------------
# Index builders
# ---------------------------------------------------------------------------

def index_by_patient_id(records: list) -> dict[str, dict]:
    return {r["patient_id"]: r for r in records if isinstance(r, dict) and r.get("patient_id")}


def index_by_trial_id(records: list) -> dict[str, dict]:
    return {r["trial_id"]: r for r in records if isinstance(r, dict) and r.get("trial_id")}


def build_label_index(labels: list) -> dict[tuple, str]:
    return {
        (r["patient_id"], r["trial_id"]): r["label"]
        for r in labels
        if isinstance(r, dict) and r.get("patient_id") and r.get("trial_id") and r.get("label")
    }


# ---------------------------------------------------------------------------
# Patient adapter — convert narrative/noisy records into matcher-compatible dicts
# ---------------------------------------------------------------------------

def adapt_narrative(record: dict) -> dict:
    """Return a matcher-compatible patient dict from a narrative record."""
    adapted = dict(record)
    # narrative_profile replaces summary for text-based fields
    if record.get("narrative_profile") and not adapted.get("summary"):
        adapted["summary"] = record["narrative_profile"]
    # medication_summary -> medications fallback
    if not adapted.get("medications") and record.get("medication_summary"):
        adapted["medications"] = record["medication_summary"]
    return adapted


def adapt_noisy(record: dict) -> dict:
    """Return a matcher-compatible patient dict from a noisy record."""
    adapted: dict = {
        "patient_id":    record.get("patient_id", ""),
        "summary":       record.get("noisy_profile", ""),
        "key_features":  record.get("key_features", []),
        "labs":          record.get("labs", {}),
        "medications":   [],
        "exclusions":    [],
        "diagnosis":     "",
        "disease_stage": "",
        "age":           None,
        "sex":           "any",
    }
    return adapted


# ---------------------------------------------------------------------------
# Matcher invocation
# ---------------------------------------------------------------------------

def run_matcher(patient: dict, trial: dict) -> str:
    """Call the rule_matcher and return the predicted label string."""
    try:
        from app.eligibility.rule_matcher import match_patient_to_trial
        result = match_patient_to_trial(patient, trial)
        if isinstance(result, dict):
            label = result.get("prediction") or result.get("predicted_label", "unclear")
        else:
            label = str(result)
        return label if label in VALID_LABELS else "unclear"
    except Exception:
        return "unclear"


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(gold: list[str], predicted: list[str]) -> dict:
    n = len(gold)
    if n == 0:
        return {"total_pairs": 0, "correct": 0, "error_count": 0,
                "accuracy": None, "macro_f1": None, "label_distribution": {}}

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
        per_class[cls] = {"precision": precision, "recall": recall, "f1": f1}

    macro_f1 = sum(v["f1"] for v in per_class.values()) / 3
    label_dist = defaultdict(int)
    for lbl in predicted:
        label_dist[lbl] += 1

    return {
        "total_pairs":        n,
        "correct":            correct,
        "error_count":        n - correct,
        "accuracy":           correct / n,
        "macro_f1":           macro_f1,
        "label_distribution": dict(label_dist),
    }


# ---------------------------------------------------------------------------
# Evaluation per input type
# ---------------------------------------------------------------------------

def evaluate_input_type(
    label_pairs: list[tuple],   # [(patient_id, trial_id, gold_label), ...]
    patient_index: dict[str, dict],
    trial_index: dict[str, dict],
    adapter,                    # callable or None (None = use record as-is)
) -> dict:
    gold_list: list[str] = []
    pred_list: list[str] = []
    skipped = 0

    for pid, tid, gold in label_pairs:
        patient_raw = patient_index.get(pid)
        trial = trial_index.get(tid)
        if patient_raw is None or trial is None:
            skipped += 1
            continue
        patient = adapter(patient_raw) if adapter else patient_raw
        predicted = run_matcher(patient, trial)
        gold_list.append(gold)
        pred_list.append(predicted)

    metrics = compute_metrics(gold_list, pred_list)
    metrics["skipped_pairs"] = skipped
    return metrics


# ---------------------------------------------------------------------------
# Readiness audit (fallback when files are missing)
# ---------------------------------------------------------------------------

def build_readiness_report(missing: list[str]) -> dict:
    parts = []
    if str(NOISY_PATH) in missing:
        parts.append(
            "patient_cases_noisy.json is missing — run scripts/generate_noisy_patient_cases.py first."
        )
    if str(NARRATIVE_PATH) in missing:
        parts.append(
            "patient_cases_narrative.json is missing — generate narrative profiles first."
        )
    if str(LABELS_PATH) in missing:
        parts.append("labels_llm_reviewed.json is missing — labels are required.")
    if str(TRIALS_PATH) in missing:
        parts.append("trial_cases.json is missing.")
    return {
        "comparison_ready": False,
        "missing_files": missing,
        "recommendation": "Task 92 cannot be completed yet. " + " ".join(parts),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # Check required files
    missing = [str(p) for name, p in REQUIRED_FILES.items() if not p.exists()]
    if missing:
        report = build_readiness_report(missing)
        write_json(report, REPORT_PATH)
        print("Missing files:")
        for f in missing:
            print(f"  - {f}")
        print(report["recommendation"])
        print(f"Report written to: {REPORT_PATH}")
        sys.exit(0)

    # Load all data
    clean_patients     = load_json(CLEAN_PATH)
    narrative_patients = load_json(NARRATIVE_PATH)
    noisy_patients     = load_json(NOISY_PATH)
    trials             = load_json(TRIALS_PATH)
    labels             = load_json(LABELS_PATH)

    clean_index     = index_by_patient_id(clean_patients)
    narrative_index = index_by_patient_id(narrative_patients)
    noisy_index     = index_by_patient_id(noisy_patients)
    trial_index     = index_by_trial_id(trials)
    label_index     = build_label_index(labels)

    label_pairs = [
        (pid, tid, gold)
        for (pid, tid), gold in sorted(label_index.items())
    ]

    clean_metrics     = evaluate_input_type(label_pairs, clean_index,     trial_index, adapter=None)
    narrative_metrics = evaluate_input_type(label_pairs, narrative_index, trial_index, adapter=adapt_narrative)
    noisy_metrics     = evaluate_input_type(label_pairs, noisy_index,     trial_index, adapter=adapt_noisy)

    report = {
        "_note": (
            "Synthetic-data robustness comparison only. "
            "Not clinical validation. All patients are synthetic."
        ),
        "comparison_ready": True,
        "total_label_pairs": len(label_pairs),
        "clean":     clean_metrics,
        "narrative": narrative_metrics,
        "noisy":     noisy_metrics,
    }

    write_json(report, REPORT_PATH)

    def _fmt(v) -> str:
        return f"{v:.4f}" if isinstance(v, float) else str(v)

    print(f"Label pairs evaluated: {len(label_pairs)}")
    print(f"{'Input type':<12}  {'accuracy':<10}  {'macro_f1':<10}  {'correct':<8}  {'errors':<8}  {'skipped'}")
    for name, m in [("clean", clean_metrics), ("narrative", narrative_metrics), ("noisy", noisy_metrics)]:
        print(
            f"{name:<12}  {_fmt(m['accuracy']):<10}  {_fmt(m['macro_f1']):<10}  "
            f"{m['correct']:<8}  {m['error_count']:<8}  {m['skipped_pairs']}"
        )
    print(f"Report written to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
