"""Temporal eligibility rule helpers."""

from app.eligibility.clinical_terms import _any_match
from app.eligibility.clinical_units import (
    parse_temporal_exclusion,
    parse_temporal_inclusion,
    get_patient_elapsed_days,
)


def _text(value) -> str:
    """Coerce a value to a lowercase stripped string for pattern matching."""
    if isinstance(value, list):
        return " ".join(str(v) for v in value).lower()
    return str(value).lower()


def _check_temporal_criteria(
    patient: dict, trial: dict
) -> tuple[list[tuple[str, str]], list[tuple[str, str]], list[str]]:
    """Check temporal inclusion/exclusion criteria against patient data.

    Returns:
        (blocks, uncertainties, missing_keys)
        blocks: list of (blocking_criterion, matched_fact)
        uncertainties: list of (uncertain_criterion, matched_fact)
        missing_keys: list of missing_information keys
    """
    blocks: list[tuple[str, str]] = []
    uncertainties: list[tuple[str, str]] = []
    missing_keys: list[str] = []

    patient_all_text = _text(
        patient.get("key_features", [])
        + patient.get("medications", [])
        + patient.get("exclusions", [])
        + [patient.get("summary", "")]
    )
    patient_diag_text = _text(
        [str(patient.get("disease_duration", "") or "")]
        + patient.get("key_features", [])
        + [patient.get("summary", "")]
    )

    # --- Temporal exclusions ---
    for criterion in trial.get("exclusion_criteria", []):
        parsed = parse_temporal_exclusion(criterion)
        if parsed is None:
            continue
        topic, max_days = parsed
        # Pick relevant patient text for the topic
        if topic in ("medication_change",):
            p_text = _text(patient.get("medications", []) + patient.get("key_features", []))
        elif topic in ("dbs_surgery", "surgery"):
            p_text = _text(patient.get("key_features", []) + patient.get("exclusions", []))
        else:
            p_text = patient_all_text

        elapsed = get_patient_elapsed_days(p_text)
        if elapsed is None:
            # Temporal info missing — unclear only if patient has any signal for the topic
            _TOPIC_PATIENT_SIGNALS = {
                "medication_change": [r"medication.*changed", r"adjusted.*dose", r"new.*medication"],
                "investigational_drug": [r"investigational", r"experimental.*drug", r"study.*drug"],
                "trial_participation": [r"clinical.*trial", r"study.*participation", r"enrolled.*trial"],
                "dbs_surgery": [r"dbs", r"deep.*brain.*stimulation", r"dbs.*surgery"],
                "surgery": [r"surgery", r"surgical", r"operation"],
            }
            signals = _TOPIC_PATIENT_SIGNALS.get(topic, [])
            if signals and _any_match(signals, p_text):
                unc_msg = f"temporal exclusion: {criterion.strip()} — patient history present but timing not documented"
                uncertainties.append((unc_msg, f"{topic} noted but timing unknown"))
                if f"{topic}_timing" not in missing_keys:
                    missing_keys.append(f"{topic}_timing")
        elif elapsed <= max_days:
            # Violation — within exclusion window
            blocks.append((
                f"temporal exclusion violated: {criterion.strip()}",
                f"{topic} occurred {elapsed} day(s) ago (must be > {max_days} days ago)",
            ))
        # else: elapsed > max_days → satisfies exclusion, no flag

    # --- Temporal inclusions (disease duration / symptom duration) ---
    for criterion in trial.get("inclusion_criteria", []):
        parsed = parse_temporal_inclusion(criterion)
        if parsed is None:
            continue
        topic, threshold_days, direction = parsed

        elapsed = get_patient_elapsed_days(patient_diag_text)
        if elapsed is None:
            # Check if any duration field is present but unknown
            raw = patient.get("disease_duration")
            if raw is None or str(raw).lower() in ("none", "unknown", "unclear", ""):
                unc_msg = f"temporal inclusion: {criterion.strip()} — {topic} not documented"
                uncertainties.append((unc_msg, f"{topic} not documented"))
                if topic not in missing_keys:
                    missing_keys.append(topic)
        else:
            if direction == "at_least" and elapsed < threshold_days:
                blocks.append((
                    f"temporal inclusion not met: {criterion.strip()}",
                    f"{topic} is {elapsed} day(s); required >= {threshold_days} days",
                ))
            elif direction == "less_than" and elapsed >= threshold_days:
                blocks.append((
                    f"temporal inclusion not met: {criterion.strip()}",
                    f"{topic} is {elapsed} day(s); required < {threshold_days} days",
                ))

    return blocks, uncertainties, missing_keys
