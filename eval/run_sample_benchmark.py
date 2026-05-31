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


def format_label_distribution(label_distribution: dict) -> str:
    labels = ["eligible", "not_eligible", "unclear"]
    gold = label_distribution["gold"]
    predicted = label_distribution["predicted"]
    rows = "".join(
        f"  {l:<15} {gold[l]:>5}  {predicted[l]:>9}\n" for l in labels
    )
    return (
        "\nLabel distribution\n"
        f"  {'':15} {'Gold':>5}  {'Predicted':>9}\n"
        + rows.rstrip()
    )


def format_confusion_matrix(confusion_matrix: dict) -> str:
    labels = ["eligible", "not_eligible", "unclear"]
    header = f"  {'Gold \\ Predicted':<15}" + "".join(f"  {l:>15}" for l in labels)
    rows = "\n".join(
        f"  {g:<15}" + "".join(f"  {confusion_matrix[g][p]:>15}" for p in labels)
        for g in labels
    )
    return "\nConfusion matrix\n" + header + "\n" + rows


def format_benchmark_metadata(metadata: dict) -> str:
    return (
        "\nBenchmark metadata\n"
        f"  Benchmark name : {metadata['benchmark_name']}\n"
        f"  Patients       : {metadata['num_patients']}\n"
        f"  Trials         : {metadata['num_trials']}\n"
        f"  Label records  : {metadata['num_label_records']}\n"
        f"  Evaluated pairs: {metadata['num_evaluated_pairs']}"
    )


def build_confusion_matrix(gold_labels: list[str], predictions: list[str]) -> dict:
    labels = ["eligible", "not_eligible", "unclear"]
    matrix = {g: {p: 0 for p in labels} for g in labels}
    for g, p in zip(gold_labels, predictions):
        if g in matrix and p in matrix[g]:
            matrix[g][p] += 1
    return matrix


def build_label_distribution(gold_labels: list[str], predictions: list[str]) -> dict:
    labels = ["eligible", "not_eligible", "unclear"]
    return {
        "gold": {l: gold_labels.count(l) for l in labels},
        "predicted": {l: predictions.count(l) for l in labels},
    }


def build_benchmark_metadata(
    patients: list[dict],
    trials: list[dict],
    labels: list[dict],
    prediction_records: list[dict],
) -> dict:
    return {
        "benchmark_name": "sample_benchmark",
        "num_patients": len(patients),
        "num_trials": len(trials),
        "num_label_records": len(labels),
        "num_evaluated_pairs": len(prediction_records),
    }


def build_error_cases(prediction_records: list[dict]) -> list[dict]:
    return [r for r in prediction_records if r.get("gold_label") != r.get("predicted_label")]


def format_error_summary(error_cases: list[dict], total_predictions: int) -> str:
    n = len(error_cases)
    pct = 0.0 if total_predictions == 0 else round(n / total_predictions * 100, 1)
    return (
        "\nErrors\n"
        f"  Error cases: {n} / {total_predictions} ({pct}%)"
    )


def build_benchmark_output(
    metadata: dict,
    coverage: dict,
    label_distribution: dict,
    confusion_matrix: dict,
    metrics: dict,
    prediction_records: list[dict],
    error_cases: list[dict],
) -> dict:
    return {
        "metadata": metadata,
        "coverage": coverage,
        "confusion_matrix": confusion_matrix,
        "label_distribution": label_distribution,
        "metrics": metrics,
        "predictions": prediction_records,
        "error_cases": error_cases,
    }


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
    label_distribution = build_label_distribution(gold_labels, predictions)
    confusion_matrix = build_confusion_matrix(gold_labels, predictions)
    metadata = build_benchmark_metadata(patients, trials, labels, prediction_records)
    error_cases = build_error_cases(prediction_records)

    # Print summary
    print("\n=== Sample Benchmark Results ===")
    print(f"Evaluated pairs : {len(gold_labels)}")
    print(f"Accuracy        : {metrics['accuracy']:.3f}")
    print(f"Macro F1        : {metrics['macro_f1']:.3f}")
    print("\nPer-class F1:")
    for label, values in metrics["per_class"].items():
        print(f"  {label:<15} {values['f1']:.3f}")
    print(format_benchmark_metadata(metadata))
    print(format_label_distribution(label_distribution))
    print(format_confusion_matrix(confusion_matrix))
    print(format_error_summary(error_cases, len(prediction_records)))
    print(format_coverage_summary(coverage))

    # Save results
    output = build_benchmark_output(
        metadata,
        coverage,
        label_distribution,
        confusion_matrix,
        metrics,
        prediction_records,
        error_cases,
    )
    RESULTS_FILE.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\nResults saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
