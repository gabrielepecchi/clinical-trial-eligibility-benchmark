"""Run the sample benchmark: match patients to trials and evaluate predictions."""

import json
from pathlib import Path

from app.eligibility.rule_matcher import match_patient_to_trial, match_patient_to_trial_criteria
from eval.evaluate import compute_metrics

# ---------------------------------------------------------------------------
# File paths
# ---------------------------------------------------------------------------

PATIENTS_FILE = Path("data/processed/patient_cases_sample.json")
TRIALS_FILE = Path("data/processed/trial_cases_sample.json")
LABELS_FILE = Path("data/processed/labels_sample.json")
RESULTS_FILE = Path("data/processed/results_sample.json")


def load_json(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_coverage_summary(prediction_records: list[dict]) -> dict:
    return {
        "total_predictions": len(prediction_records),
        "with_missing_information": sum(1 for r in prediction_records if r.get("missing_information")),
        "with_criterion_results": sum(1 for r in prediction_records if r.get("criterion_results")),
    }


def format_coverage_summary(coverage: dict) -> str:
    total = coverage["total_predictions"]
    pct = lambda n: 0.0 if total == 0 else round(n / total * 100, 1)
    return (
        "\nCoverage\n"
        f"  Total predictions       : {total}\n"
        f"  With missing information: {coverage['with_missing_information']} ({pct(coverage['with_missing_information'])}%)\n"
        f"  With criterion results  : {coverage['with_criterion_results']} ({pct(coverage['with_criterion_results'])}%)"
    )


def main() -> None:
    # Load data
    patients = load_json(PATIENTS_FILE)
    trials = load_json(TRIALS_FILE)
    labels = load_json(LABELS_FILE)

    # Index patients and trials by ID for quick lookup
    patient_index = {p["patient_id"]: p for p in patients}
    trial_index = {t["trial_id"]: t for t in trials}

    # Run predictions for each label record
    gold_labels: list[str] = []
    predictions: list[str] = []
    prediction_records: list[dict] = []

    for record in labels:
        patient_id = record["patient_id"]
        trial_id = record["trial_id"]
        gold_label = record["label"]

        patient = patient_index.get(patient_id)
        trial = trial_index.get(trial_id)

        if patient is None:
            print(f"WARNING: patient {patient_id!r} not found — skipping")
            continue
        if trial is None:
            print(f"WARNING: trial {trial_id!r} not found — skipping")
            continue

        result = match_patient_to_trial(patient, trial)
        predicted_label = result["prediction"]

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

        prediction_records.append({
            "patient_id": patient_id,
            "trial_id": trial_id,
            "gold_label": gold_label,
            "predicted_label": predicted_label,
            "confidence": result["confidence"],
            "matched_facts": result["matched_facts"],
            "blocking_criteria": result["blocking_criteria"],
            "uncertain_criteria": result["uncertain_criteria"],
            "explanation": result["explanation"],
            "missing_information": result.get("missing_information", []),
            "criterion_results": criterion_results,
        })

    # Compute metrics
    metrics = compute_metrics(gold_labels, predictions)

    # Compute coverage
    coverage = build_coverage_summary(prediction_records)

    # Print summary
    print("\n=== Sample Benchmark Results ===")
    print(f"Evaluated pairs : {len(gold_labels)}")
    print(f"Accuracy        : {metrics['accuracy']:.3f}")
    print(f"Macro F1        : {metrics['macro_f1']:.3f}")
    print("\nPer-class F1:")
    for label, values in metrics["per_class"].items():
        print(f"  {label:<15} {values['f1']:.3f}")
    print(format_coverage_summary(coverage))

    # Save results
    output = {
        "coverage": coverage,
        "metrics": metrics,
        "predictions": prediction_records,
    }
    RESULTS_FILE.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\nResults saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
