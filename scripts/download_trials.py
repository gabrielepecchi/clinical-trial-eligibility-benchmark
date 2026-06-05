"""Download a bounded raw Parkinson disease trial snapshot from ClinicalTrials.gov v2 API.

This script fetches a fixed maximum number of studies and saves them as a local JSON file.
It does NOT download all trials from ClinicalTrials.gov — only a reproducible small snapshot.
"""

import argparse
import json
import sys
from pathlib import Path

import requests

# Constants
API_BASE_URL = "https://clinicaltrials.gov/api/v2/studies"
SEARCH_CONDITION = "Parkinson disease"
OUTPUT_FILE = Path("data/raw/parkinson_trials_raw.json")
MAX_TRIALS = 60
PAGE_SIZE = 10


def build_params(page_token: str | None, page_size: int) -> dict:
    """Build query parameters for a single API request."""
    params: dict = {
        "query.cond": SEARCH_CONDITION,
        "pageSize": page_size,
        "format": "json",
    }
    if page_token:
        params["pageToken"] = page_token
    return params


def fetch_trials(max_trials: int = MAX_TRIALS) -> list[dict]:
    """Fetch up to max_trials studies from the ClinicalTrials.gov v2 API."""
    trials: list[dict] = []
    next_page_token: str | None = None

    while len(trials) < max_trials:
        page_size = min(PAGE_SIZE, max_trials - len(trials))
        params = build_params(next_page_token, page_size)

        response = requests.get(API_BASE_URL, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        studies = data.get("studies", [])
        if not studies:
            break

        trials.extend(studies)
        next_page_token = data.get("nextPageToken")
        if not next_page_token:
            break

    return trials


def save_trials(trials: list[dict], output_file: Path) -> None:
    """Save trials list as JSON to output_file."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(trials, f, indent=2)
    print(f"Saved {len(trials)} trials to {output_file}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download a Parkinson disease trial snapshot from ClinicalTrials.gov."
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Skip network requests and reuse an existing output file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_FILE,
        help=f"Path to the output JSON file (default: {OUTPUT_FILE}).",
    )
    parser.add_argument(
        "--max-trials",
        type=int,
        default=MAX_TRIALS,
        help=f"Maximum number of trials to fetch (default: {MAX_TRIALS}).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    output_path: Path = args.output

    if args.offline:
        if output_path.exists():
            print(f"Offline mode: reusing existing raw trials file at {output_path}")
        else:
            print(
                f"Error: offline mode requested but output file does not exist: {output_path}",
                file=sys.stderr,
            )
            sys.exit(1)
    else:
        print(f"Fetching up to {args.max_trials} trials for: {SEARCH_CONDITION}")
        trials = fetch_trials(args.max_trials)
        save_trials(trials, output_path)
