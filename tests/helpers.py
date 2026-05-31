"""Shared test helpers for rule_matcher test suite."""


def make_patient(**kwargs) -> dict:
    base = {
        "patient_id": "P_TEST",
        "age": 60,
        "sex": "male",
        "diagnosis": ["Parkinson disease"],
        "key_features": ["Hoehn and Yahr stage 2"],
        "exclusions": [],
        "medications": ["levodopa/carbidopa 100/25 mg three times daily"],
    }
    base.update(kwargs)
    return base


def incl_trial(criterion: str) -> dict:
    return {"trial_id": "T_FMT", "inclusion_criteria": [criterion], "exclusion_criteria": []}


def excl_trial(criterion: str) -> dict:
    return {"trial_id": "T_EXCL", "inclusion_criteria": [], "exclusion_criteria": [criterion]}
