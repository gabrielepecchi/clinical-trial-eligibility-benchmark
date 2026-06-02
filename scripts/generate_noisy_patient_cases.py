"""
generate_noisy_patient_cases.py — Task 12 / Task 92: generate realistic noisy patient inputs.

All patients in the output are fully synthetic. They do not represent real individuals.

Usage:
    PYTHONPATH=. python scripts/generate_noisy_patient_cases.py
"""

import json
import random
from pathlib import Path

STRUCTURED_PATH = Path("data/processed/patient_cases.json")
NARRATIVE_PATH = Path("data/processed/patient_cases_narrative.json")
OUTPUT_PATH = Path("data/processed/patient_cases_noisy.json")

GENERATION_SEED = 42

SYNTHETIC_DISCLAIMER = (
    "All records in this file are fully synthetic and do not represent real patients or clinical data."
)

# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_json(path: Path) -> list:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def write_json(data, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)


# ---------------------------------------------------------------------------
# Noise helpers — deterministic, no patient_id-specific logic
# ---------------------------------------------------------------------------

def _drop_field(rng: random.Random, text: str, field_label: str, probability: float = 0.5) -> str:
    """Replace a field mention with an omission phrase."""
    if rng.random() < probability:
        return text.replace(field_label, "[not documented]")
    return text


def _add_temporal_ambiguity(rng: random.Random, text: str) -> str:
    replacements = [
        ("1 year ago", "some time ago"),
        ("6 months", "several months"),
        ("12 years ago", "many years ago"),
        ("3 years ago", "a few years ago"),
        ("2 weeks ago", "recently"),
        ("6 years ago", "years ago"),
        ("approximately", "roughly"),
    ]
    for original, replacement in replacements:
        if original in text and rng.random() < 0.6:
            text = text.replace(original, replacement, 1)
    return text


def _add_negation_phrasing(rng: random.Random, text: str) -> str:
    if rng.random() < 0.4:
        text = text + " No formal cognitive assessment was documented at this visit."
    if rng.random() < 0.3:
        text = text + " Patient denied recent falls, though family reported otherwise."
    return text


def _add_medication_uncertainty(rng: random.Random, text: str) -> str:
    uncertain_phrases = [
        " Medication adherence uncertain per self-report.",
        " Exact dose and frequency could not be verified.",
        " Medication list based on patient recall only.",
        " Pharmacy records unavailable at time of assessment.",
    ]
    if rng.random() < 0.5:
        text = text + rng.choice(uncertain_phrases)
    return text


def _add_dbs_uncertainty(rng: random.Random, text: str) -> str:
    if "DBS" in text or "deep brain stimulation" in text.lower():
        if rng.random() < 0.5:
            text = text + " DBS implant status not confirmed by device card."
    return text


def _add_ambiguous_value(rng: random.Random, text: str) -> str:
    replacements = [
        ("Hoehn and Yahr stage 1", "Hoehn and Yahr stage approximately 1–2"),
        ("Hoehn and Yahr stage 2", "Hoehn and Yahr stage 2 or possibly 3"),
        ("Hoehn and Yahr stage 3", "Hoehn and Yahr stage 3 (borderline)"),
        ("MoCA score 19", "MoCA score approximately 19–21"),
        ("MMSE score 22", "MMSE score in the low-normal range"),
        ("MoCA score 28", "MoCA score 28 (unconfirmed)"),
    ]
    for original, replacement in replacements:
        if original in text and rng.random() < 0.5:
            text = text.replace(original, replacement, 1)
    return text


def _truncate_features(rng: random.Random, features: list) -> list:
    """Randomly drop 0–2 key features."""
    if len(features) <= 1:
        return features
    n_drop = rng.randint(0, min(2, len(features) - 1))
    indices = sorted(rng.sample(range(len(features)), n_drop))
    return [f for i, f in enumerate(features) if i not in indices]


def _drop_labs(rng: random.Random, labs: dict) -> dict:
    if labs and rng.random() < 0.4:
        return {}
    return labs


# ---------------------------------------------------------------------------
# Noise type detection
# ---------------------------------------------------------------------------

def _detect_noise_types(original_text: str, noisy_text: str, dropped_features: bool,
                         labs_dropped: bool) -> list[str]:
    types: list[str] = []
    if "[not documented]" in noisy_text:
        types.append("missing_fields")
    if any(w in noisy_text for w in ["approximately", "borderline", "roughly", "unconfirmed",
                                      "possibly", "low-normal"]):
        types.append("ambiguous_values")
    if any(w in noisy_text for w in ["adherence uncertain", "could not be verified",
                                      "patient recall", "unavailable"]):
        types.append("incomplete_medication_history")
    if "DBS implant status not confirmed" in noisy_text:
        types.append("uncertain_dbs_device_history")
    if any(w in noisy_text for w in ["some time ago", "several months", "many years ago",
                                      "a few years ago", "recently", "roughly"]):
        types.append("temporal_ambiguity")
    if any(w in noisy_text for w in ["denied", "No formal", "family reported"]):
        types.append("negation_heavy_narrative_phrasing")
    if dropped_features:
        types.append("missing_fields")
    if labs_dropped:
        types.append("missing_fields")
    return sorted(set(types))


# ---------------------------------------------------------------------------
# Core generator
# ---------------------------------------------------------------------------

def generate_noisy_record(
    structured: dict,
    narrative: dict | None,
    rng: random.Random,
) -> dict:
    pid = structured["patient_id"]

    # Prefer narrative_profile as base text; fall back to structured summary
    if narrative and narrative.get("narrative_profile"):
        base_text: str = narrative["narrative_profile"]
        source_format = "narrative"
    else:
        base_text = structured.get("summary", "")
        source_format = "structured_fallback"

    original_text = base_text

    # Apply noise transforms
    noisy_text = _add_temporal_ambiguity(rng, base_text)
    noisy_text = _add_ambiguous_value(rng, noisy_text)
    noisy_text = _add_medication_uncertainty(rng, noisy_text)
    noisy_text = _add_dbs_uncertainty(rng, noisy_text)
    noisy_text = _add_negation_phrasing(rng, noisy_text)
    noisy_text = _drop_field(rng, noisy_text, "disease duration", probability=0.3)

    # Drop some key features
    original_features = list(structured.get("key_features", []))
    noisy_features = _truncate_features(rng, original_features)
    features_dropped = len(noisy_features) < len(original_features)

    # Maybe drop labs
    original_labs = dict(structured.get("labs", {}))
    noisy_labs = _drop_labs(rng, original_labs)
    labs_dropped = bool(original_labs) and not noisy_labs

    noise_types = _detect_noise_types(original_text, noisy_text, features_dropped, labs_dropped)
    if not noise_types:
        noise_types = ["minimal_noise"]

    return {
        "patient_id": pid,
        "source_patient_id": pid,
        "noisy_profile": noisy_text,
        "key_features": noisy_features,
        "labs": noisy_labs,
        "noise_types": noise_types,
        "source_format": source_format,
        "generation_seed": GENERATION_SEED,
        "_note": SYNTHETIC_DISCLAIMER,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    try:
        structured_patients = load_json(STRUCTURED_PATH)
    except FileNotFoundError:
        print(f"[ERROR] File not found: {STRUCTURED_PATH}")
        return

    try:
        narrative_patients = load_json(NARRATIVE_PATH)
        narrative_index = {p["patient_id"]: p for p in narrative_patients if isinstance(p, dict)}
    except FileNotFoundError:
        print(f"[WARN] Narrative file not found: {NARRATIVE_PATH} — using structured fallback.")
        narrative_index = {}

    rng = random.Random(GENERATION_SEED)

    noisy_records = []
    for rec in structured_patients:
        if not isinstance(rec, dict):
            continue
        pid = rec.get("patient_id", "")
        narrative_rec = narrative_index.get(pid)
        noisy = generate_noisy_record(rec, narrative_rec, rng)
        noisy_records.append(noisy)

    write_json(noisy_records, OUTPUT_PATH)

    print(f"Loaded {len(structured_patients)} structured patients")
    print(f"Loaded {len(narrative_index)} narrative patients")
    print(f"Generated {len(noisy_records)} noisy patients")
    print(f"Output written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
