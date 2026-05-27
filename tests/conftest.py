"""Pytest path setup for local project modules."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

for relative_path in ["app", "app/eligibility", "scripts", "eval"]:
    path = ROOT / relative_path
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
