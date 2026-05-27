"""Evaluate predicted eligibility labels against gold benchmark labels."""

LABELS = ["eligible", "not_eligible", "unclear"]


def compute_metrics(gold_labels: list[str], predictions: list[str]) -> dict:
    """Compute evaluation metrics for predicted eligibility labels.

    Args:
        gold_labels:  Ground-truth labels, each one of 'eligible', 'not_eligible', 'unclear'.
        predictions:  Predicted labels in the same order as gold_labels.

    Returns:
        Dictionary with keys:
            accuracy         – fraction of correct predictions (float)
            macro_precision  – unweighted mean precision across classes (float)
            macro_recall     – unweighted mean recall across classes (float)
            macro_f1         – unweighted mean F1 across classes (float)
            per_class        – dict[label] -> {precision, recall, f1}
            confusion_matrix – dict[gold_label][predicted_label] -> int
    """
    if len(gold_labels) != len(predictions):
        raise ValueError(
            f"gold_labels and predictions must have the same length "
            f"({len(gold_labels)} vs {len(predictions)})"
        )

    # Initialise confusion matrix
    confusion_matrix: dict[str, dict[str, int]] = {
        gold: {pred: 0 for pred in LABELS} for gold in LABELS
    }

    if not gold_labels:
        per_class = {
            label: {"precision": 0.0, "recall": 0.0, "f1": 0.0}
            for label in LABELS
        }
        return {
            "accuracy": 0.0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "macro_f1": 0.0,
            "per_class": per_class,
            "confusion_matrix": confusion_matrix,
        }

    for gold, pred in zip(gold_labels, predictions):
        if gold not in confusion_matrix:
            confusion_matrix[gold] = {p: 0 for p in LABELS}
        if pred not in confusion_matrix[gold]:
            confusion_matrix[gold][pred] = 0
        confusion_matrix[gold][pred] += 1

    # Accuracy
    correct = sum(1 for g, p in zip(gold_labels, predictions) if g == p)
    accuracy = correct / len(gold_labels)

    # Per-class metrics
    per_class: dict[str, dict[str, float]] = {}
    for label in LABELS:
        tp = confusion_matrix[label][label]
        fp = sum(confusion_matrix[g][label] for g in LABELS if g != label)
        fn = sum(confusion_matrix[label][p] for p in LABELS if p != label)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        per_class[label] = {"precision": precision, "recall": recall, "f1": f1}

    macro_precision = sum(per_class[l]["precision"] for l in LABELS) / len(LABELS)
    macro_recall = sum(per_class[l]["recall"] for l in LABELS) / len(LABELS)
    macro_f1 = sum(per_class[l]["f1"] for l in LABELS) / len(LABELS)

    return {
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "per_class": per_class,
        "confusion_matrix": confusion_matrix,
    }
