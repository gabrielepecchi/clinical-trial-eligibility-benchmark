"""Run the real draft benchmark using LLM-reviewed labels.

This evaluates the rule-based matcher against labels_llm_reviewed.json.
The labels are benchmark draft labels and still need spot-checking.
"""

import csv
import json
from pathlib import Path

from app.eligibility.rule_matcher import match_patient_to_trial, match_patient_to_trial_criteria
from eval.evaluate import compute_metrics

PATIENTS_FILE = Path("data/processed/patient_cases.json")
TRIALS_FILE = Path("data/processed/trial_cases.json")
LABELS_FILE = Path("data/processed/labels_llm_reviewed.json")
RESULTS_FILE = Path("data/processed/results_llm_reviewed.json")
RESULTS_CSV_FILE = Path("data/processed/results_llm_reviewed.csv")
CRITERION_CSV_FILE = Path("data/processed/criterion_level_results.csv")


_CSV_FIELDNAMES = [
    "patient_id", "trial_id", "gold_label", "predicted_label",
    "correct", "label_status", "confidence",
    "matched_facts", "blocking_criteria", "uncertain_criteria",
    "matcher_explanation", "gold_rationale",
]


def build_llm_reviewed_csv_rows(prediction_records: list[dict]) -> list[dict]:
    rows = []
    for r in prediction_records:
        gold = r.get("gold_label", "")
        predicted = r.get("predicted_label", "")
        rows.append({
            "patient_id": r.get("patient_id", ""),
            "trial_id": r.get("trial_id", ""),
            "gold_label": gold,
            "predicted_label": predicted,
            "correct": gold == predicted,
            "label_status": r.get("label_status", ""),
            "confidence": r.get("confidence", ""),
            "matched_facts": "; ".join(r.get("matched_facts") or []),
            "blocking_criteria": "; ".join(r.get("blocking_criteria") or []),
            "uncertain_criteria": "; ".join(r.get("uncertain_criteria") or []),
            "matcher_explanation": r.get("matcher_explanation", ""),
            "gold_rationale": r.get("gold_rationale", ""),
        })
    return rows


def write_llm_reviewed_csv_rows(rows: list[dict], output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


_CRITERION_CSV_FIELDNAMES = [
    "patient_id", "trial_id", "gold_label", "predicted_label",
    "criterion", "criterion_type", "decision", "reason",
]


def build_criterion_level_csv_rows(prediction_records: list[dict]) -> list[dict]:
    rows = []
    for r in prediction_records:
        for cr in r.get("criterion_results") or []:
            rows.append({
                "patient_id": r.get("patient_id", ""),
                "trial_id": r.get("trial_id", ""),
                "gold_label": r.get("gold_label", ""),
                "predicted_label": r.get("predicted_label", ""),
                "criterion": cr.get("criterion_text", ""),
                "criterion_type": cr.get("criterion_type", ""),
                "decision": cr.get("decision", ""),
                "reason": cr.get("reason", ""),
            })
    return rows


def write_criterion_level_csv_rows(rows: list[dict], output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_CRITERION_CSV_FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_safety_uncertainty_summary(prediction_records: list[dict]) -> dict:
    """Compute safety and uncertainty error counts and rates."""
    total = len(prediction_records)
    unsafe = 0
    uncertainty = 0
    conservative = 0
    gold_unclear = 0
    true_unclear = 0
    predicted_unclear = 0
    overcommitted = 0

    for r in prediction_records:
        gold = r.get("gold_label", "")
        pred = r.get("predicted_label", "")
        if gold == "not_eligible" and pred == "eligible":
            unsafe += 1
        if gold == "unclear" and pred in {"eligible", "not_eligible"}:
            uncertainty += 1
        if gold == "eligible" and pred == "not_eligible":
            conservative += 1
        if gold == "unclear":
            gold_unclear += 1
            if pred == "unclear":
                true_unclear += 1
            if pred in {"eligible", "not_eligible"}:
                overcommitted += 1
        if pred == "unclear":
            predicted_unclear += 1

    return {
        "total_predictions": total,
        "unsafe_eligible_errors": unsafe,
        "uncertainty_errors": uncertainty,
        "overly_conservative_errors": conservative,
        "unclear_recall": true_unclear / gold_unclear if gold_unclear else 0,
        "unclear_precision": true_unclear / predicted_unclear if predicted_unclear else 0,
        "overcommitment_rate": overcommitted / gold_unclear if gold_unclear else 0,
    }


def build_benchmark_output(
    metadata: dict,
    metrics: dict,
    safety_uncertainty_summary: dict,
    error_severity_summary: dict,
    prediction_records: list[dict],
) -> dict:
    """Assemble the final benchmark output dict."""
    return {
        "metadata": metadata,
        "metrics": metrics,
        "safety_uncertainty_summary": safety_uncertainty_summary,
        "error_severity_summary": error_severity_summary,
        "predictions": prediction_records,
    }


def format_safety_uncertainty_summary(s: dict) -> str:
    """Format safety and uncertainty summary as a printable string."""
    lines = [
        "\n=== Safety & Uncertainty Summary ===",
        f"Total predictions    : {s['total_predictions']}",
        f"Unsafe eligible errors     : {s['unsafe_eligible_errors']}",
        f"Overly conservative errors : {s['overly_conservative_errors']}",
        f"Uncertainty errors         : {s['uncertainty_errors']}",
        f"Unclear recall             : {s['unclear_recall']:.3f}",
        f"Unclear precision          : {s['unclear_precision']:.3f}",
        f"Overcommitment rate        : {s['overcommitment_rate']:.3f}",
    ]
    return "\n".join(lines)


def format_error_severity_summary(s: dict) -> str:
    """Format error severity summary as a printable string."""
    lines = [
        "\n=== Error Severity Summary ===",
        f"Total errors         : {s['total_errors']}",
        f"Critical errors      : {s['critical_errors']}",
        f"Major errors         : {s['major_errors']}",
        f"Minor errors         : {s['minor_errors']}",
        f"Critical error rate  : {s['critical_error_rate']:.3f}",
        f"Major error rate     : {s['major_error_rate']:.3f}",
        f"Minor error rate     : {s['minor_error_rate']:.3f}",
    ]
    return "\n".join(lines)


def build_error_severity_summary(prediction_records: list[dict]) -> dict:
    """Compute error severity counts and rates."""
    total = len(prediction_records)
    total_errors = 0
    critical = 0
    major = 0
    minor = 0

    for r in prediction_records:
        gold = r.get("gold_label", "")
        pred = r.get("predicted_label", "")
        if gold != pred:
            total_errors += 1
        if gold == "not_eligible" and pred == "eligible":
            critical += 1
        if (gold == "unclear" and pred in {"eligible", "not_eligible"}) or \
           (gold in {"eligible", "not_eligible"} and pred == "unclear"):
            major += 1
        if (gold == "eligible" and pred == "not_eligible") or \
           (gold == "not_eligible" and pred == "unclear"):
            minor += 1

    return {
        "total_predictions": total,
        "total_errors": total_errors,
        "critical_errors": critical,
        "major_errors": major,
        "minor_errors": minor,
        "critical_error_rate": critical / total if total else 0,
        "major_error_rate": major / total if total else 0,
        "minor_error_rate": minor / total if total else 0,
    }


def load_json(path: Path) -> list[dict]:
    """Load a JSON list from disk."""
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    patients = load_json(PATIENTS_FILE)
    trials = load_json(TRIALS_FILE)
    labels = load_json(LABELS_FILE)

    patient_index = {patient["patient_id"]: patient for patient in patients}
    trial_index = {trial["trial_id"]: trial for trial in trials}

    gold_labels: list[str] = []
    predictions: list[str] = []
    prediction_records: list[dict] = []
    skipped = 0

    for record in labels:
        patient_id = record["patient_id"]
        trial_id = record["trial_id"]

        patient = patient_index.get(patient_id)
        trial = trial_index.get(trial_id)

        if patient is None or trial is None:
            skipped += 1
            continue

        result = match_patient_to_trial(patient, trial)
        predicted_label = result["prediction"]
        gold_label = record["label"]

        criterion_results = [
            {
                "criterion_text": cr.criterion_text,
                "criterion_type": cr.criterion_type.value,
                "decision": cr.decision.value,
                "reason": cr.reason,
            }
            for cr in match_patient_to_trial_criteria(patient, trial)
        ]

        gold_labels.append(gold_label)
        predictions.append(predicted_label)

        prediction_records.append(
            {
                "patient_id": patient_id,
                "trial_id": trial_id,
                "gold_label": gold_label,
                "predicted_label": predicted_label,
                "label_status": record.get("label_status", ""),
                "confidence": result["confidence"],
                "matched_facts": result["matched_facts"],
                "blocking_criteria": result["blocking_criteria"],
                "uncertain_criteria": result["uncertain_criteria"],
                "matcher_explanation": result["explanation"],
                "gold_rationale": record.get("rationale", ""),
                "gold_evidence": record.get("evidence", {}),
                "criterion_results": criterion_results,
            }
        )

    metrics = compute_metrics(gold_labels, predictions)

    safety_summary = build_safety_uncertainty_summary(prediction_records)
    error_summary = build_error_severity_summary(prediction_records)

    metadata = {
        "label_source": str(LABELS_FILE),
        "label_status": "llm_reviewed_needs_spotcheck",
        "evaluated_pairs": len(gold_labels),
        "skipped_pairs": skipped,
    }

    output = build_benchmark_output(metadata, metrics, safety_summary, error_summary, prediction_records)

    RESULTS_FILE.write_text(json.dumps(output, indent=2), encoding="utf-8")

    csv_rows = build_llm_reviewed_csv_rows(prediction_records)
    write_llm_reviewed_csv_rows(csv_rows, RESULTS_CSV_FILE)

    criterion_rows = build_criterion_level_csv_rows(prediction_records)
    write_criterion_level_csv_rows(criterion_rows, CRITERION_CSV_FILE)

    print("\n=== LLM-Reviewed Draft Benchmark Results ===")
    print(f"Evaluated pairs : {len(gold_labels)}")
    print(f"Skipped pairs   : {skipped}")
    print(f"Accuracy        : {metrics['accuracy']:.3f}")
    print(f"Macro F1        : {metrics['macro_f1']:.3f}")

    print("\nPer-class F1:")
    for label, values in metrics["per_class"].items():
        print(f"  {label:<15} {values['f1']:.3f}")

    print(f"\nResults saved to {RESULTS_FILE}")
    print(f"Predictions CSV saved to {RESULTS_CSV_FILE}")
    print(f"Criterion-level CSV saved to {CRITERION_CSV_FILE}")

    print(format_safety_uncertainty_summary(safety_summary))
    print(format_error_severity_summary(error_summary))


if __name__ == "__main__":
    main()
