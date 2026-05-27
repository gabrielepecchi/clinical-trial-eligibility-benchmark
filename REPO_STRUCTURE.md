# Repository Structure

```
clinical-trial-eligibility-benchmark/
│
├── data/
│   ├── raw/
│   │   └── parkinson_trials_raw.json        # Raw JSON from ClinicalTrials.gov API
│   └── processed/
│       ├── eligibility_criteria.json        # Extracted eligibility criteria
│       ├── trial_cases.json                 # Selected trial cases
│       ├── patient_cases.json               # Synthetic patient profiles
│       ├── label_candidates.json            # Candidate labels
│       ├── labels_seed.json                 # Seed labels
│       ├── labels_seed_review.csv           # Labels exported for manual review
│       ├── labels_llm_reviewed.json         # LLM-reviewed draft benchmark labels *
│       ├── labels.json                      # Merged/final labels
│       ├── results_sample.json              # Sample benchmark results
│       ├── results_llm_reviewed.json        # Real benchmark results
│       ├── error_analysis_sample.json       # Sample benchmark error analysis
│       └── error_analysis_llm_reviewed.json # Real benchmark error analysis
│
├── app/
│   ├── models.py                            # Core Pydantic models
│   └── eligibility/
│       ├── criteria_parser.py               # Free-text criteria parser
│       └── rule_matcher.py                  # Deterministic rule-based matcher
│
├── eval/
│   ├── evaluate.py                          # Scoring and metrics
│   ├── run_sample_benchmark.py              # Sample benchmark runner
│   ├── summarize_error_analysis.py          # Sample error analysis
│   ├── run_llm_reviewed_benchmark.py        # Real benchmark runner
│   └── summarize_llm_reviewed_errors.py     # Real benchmark error analysis
│
├── scripts/
│   ├── download_trials.py                   # Fetch raw trials from ClinicalTrials.gov
│   ├── extract_eligibility.py               # Extract eligibility criteria from raw trials
│   ├── select_trial_cases.py                # Select trial cases for benchmark
│   ├── audit_trial_cases.py                 # Audit selected trial cases
│   ├── generate_label_candidates.py         # Generate candidate labels
│   ├── audit_label_candidates.py            # Audit candidate labels
│   ├── generate_labels_seed.py              # Generate seed labels
│   ├── audit_labels_seed.py                 # Audit seed labels
│   ├── export_labels_seed_review.py         # Export seed labels to CSV for review
│   └── import_reviewed_labels.py            # Import reviewed labels back
│
├── tests/                                   # pytest unit tests (267 passing)
│   ├── parser, model, matcher tests
│   ├── data validation tests
│   ├── pipeline script tests
│   ├── LLM-reviewed labels tests
│   └── unclear matcher logic tests
│
├── README.md
├── PROJECT_SPEC.md
├── REPO_STRUCTURE.md
├── IMPLEMENTATION_PLAN.md
├── requirements.txt
└── .gitignore
```

## Notes

- `data/` is gitignored except for small example fixtures used in tests.
- `app/eligibility/` contains all importable Python modules.
- `eval/` contains benchmark scoring logic, separate from app code.
- `scripts/` contains standalone pipeline scripts not meant to be imported.
- No UI code.
- `__pycache__/` and `.pytest_cache/` are not committed.

## Benchmark labels

> \* `labels_llm_reviewed.json` contains LLM-reviewed draft benchmark labels used by `run_llm_reviewed_benchmark.py` to produce `results_llm_reviewed.json`. These are not clinical gold truth and have not been clinically validated.
