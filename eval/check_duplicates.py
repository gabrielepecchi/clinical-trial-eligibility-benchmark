"""Check for duplicate and near-duplicate trial cases.

Usage:
    PYTHONPATH=. python eval/check_duplicates.py
"""

import json
import sys
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

TRIALS_FILE = Path("data/processed/trial_cases.json")

NEAR_DUPLICATE_THRESHOLD = 0.90  # similarity ratio for near-duplicate criteria warning


def load_json_list(path: Path) -> list[dict]:
    """Load a JSON list from disk, exiting on error."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        print(f"ERROR: File not found: {path}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"ERROR: Malformed JSON in {path}: {e}", file=sys.stderr)
        sys.exit(1)
    if not isinstance(data, list):
        print(f"ERROR: Expected a JSON list in {path}", file=sys.stderr)
        sys.exit(1)
    return data


def normalize_text(text: str) -> str:
    """Lowercase, strip, and collapse whitespace for comparison."""
    return " ".join(text.lower().split())


def extract_trial_id(record: dict) -> str | None:
    """Return the first non-empty identifier found in the record."""
    for field in ("trial_id", "nct_id", "id"):
        val = record.get(field)
        if val and isinstance(val, str) and val.strip():
            return val.strip()
    return None


def extract_criteria_text(record: dict) -> str:
    """Return normalized combined criteria text from the record."""
    parts: list[str] = []
    for field in (
        "criteria_text", "eligibility_criteria", "criteria",
        "inclusion_criteria", "exclusion_criteria",
        "inclusion", "exclusion",
        "inclusion_text", "exclusion_text", "eligibility",
    ):
        val = record.get(field)
        if val and isinstance(val, str):
            parts.append(val)
        elif val and isinstance(val, list):
            parts.extend(str(v) for v in val if v)
    return normalize_text(" ".join(parts))


def find_duplicate_ids(records: list[dict]) -> dict[str, list[int]]:
    """Return a dict mapping each duplicated ID to the list of record indices."""
    seen: dict[str, list[int]] = defaultdict(list)
    for i, record in enumerate(records):
        trial_id = extract_trial_id(record)
        if trial_id:
            seen[trial_id].append(i)
    return {tid: indices for tid, indices in seen.items() if len(indices) > 1}


def find_duplicate_criteria(
    records: list[dict],
) -> list[tuple[int, int, float]]:
    """Return list of (index_a, index_b, similarity) for near-duplicate criteria pairs.

    Uses SequenceMatcher for a simple deterministic comparison.
    Skips records with empty criteria text.
    """
    texts = [(i, extract_criteria_text(r)) for i, r in enumerate(records)]
    texts = [(i, t) for i, t in texts if t]

    near_duplicates: list[tuple[int, int, float]] = []
    for a in range(len(texts)):
        for b in range(a + 1, len(texts)):
            idx_a, text_a = texts[a]
            idx_b, text_b = texts[b]
            ratio = SequenceMatcher(None, text_a, text_b).ratio()
            if ratio >= NEAR_DUPLICATE_THRESHOLD:
                near_duplicates.append((idx_a, idx_b, ratio))
    return near_duplicates


def main() -> None:
    records = load_json_list(TRIALS_FILE)
    total = len(records)

    duplicate_ids = find_duplicate_ids(records)
    near_duplicate_criteria = find_duplicate_criteria(records)

    print(f"\n=== Duplicate Check Report ===")
    print(f"Input file    : {TRIALS_FILE}")
    print(f"Total records : {total}")

    print(f"\n--- Duplicate IDs ---")
    if duplicate_ids:
        for tid, indices in sorted(duplicate_ids.items()):
            print(f"  DUPLICATE ID '{tid}' at record indices: {indices}")
    else:
        print("  No duplicate IDs found.")

    print(f"\n--- Near-Duplicate Criteria (threshold: {NEAR_DUPLICATE_THRESHOLD:.0%}) ---")
    if near_duplicate_criteria:
        for idx_a, idx_b, ratio in sorted(near_duplicate_criteria, key=lambda x: -x[2]):
            id_a = extract_trial_id(records[idx_a]) or f"index {idx_a}"
            id_b = extract_trial_id(records[idx_b]) or f"index {idx_b}"
            print(f"  WARNING: {id_a!r} and {id_b!r} — similarity {ratio:.3f}")
    else:
        print("  No near-duplicate criteria found.")

    print()
    if duplicate_ids:
        print(f"FAIL: {len(duplicate_ids)} duplicate ID(s) found.", file=sys.stderr)
        sys.exit(1)
    else:
        print("OK: No exact duplicate IDs.")


if __name__ == "__main__":
    main()
