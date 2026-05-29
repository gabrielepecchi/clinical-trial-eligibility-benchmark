"""Core Pydantic data models for the clinical trial eligibility benchmark."""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class EligibilityLabelValue(str, Enum):
    eligible = "eligible"
    not_eligible = "not_eligible"
    unclear = "unclear"


class Trial(BaseModel):
    """A clinical trial from ClinicalTrials.gov."""

    nct_id: str = Field(..., description="ClinicalTrials.gov identifier, e.g. NCT12345678")
    title: str
    eligibility_text: Optional[str] = None
    inclusion_criteria: list[str] = Field(default_factory=list)
    exclusion_criteria: list[str] = Field(default_factory=list)


class Patient(BaseModel):
    """A synthetic patient profile."""

    patient_id: str
    age: int
    sex: str  # "male" | "female" | "other"
    diagnosis: list[str] = Field(default_factory=list)
    comorbidities: list[str] = Field(default_factory=list)
    medications: list[str] = Field(default_factory=list)
    labs: dict[str, float] = Field(default_factory=dict)


class EligibilityLabel(BaseModel):
    """Expected eligibility outcome for a patient-trial pair."""

    trial_id: str
    patient_id: str
    label: EligibilityLabelValue
    notes: Optional[str] = None


class CriterionType(str, Enum):
    inclusion = "inclusion"
    exclusion = "exclusion"
    unknown = "unknown"


class CriterionDecision(str, Enum):
    met = "met"
    not_met = "not_met"
    unknown = "unknown"


class CriterionMatchResult(BaseModel):
    """Result of evaluating a single criterion against a patient."""

    criterion_text: str
    criterion_type: CriterionType
    decision: CriterionDecision
    reason: str = ""
