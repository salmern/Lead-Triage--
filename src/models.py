"""Core data models shared across the pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

# Canonical recommendation labels (keep in sync with the UI)
CONTACT_NOW = "Contact Now"
NURTURE = "Nurture"
DISQUALIFY = "Disqualify"


@dataclass
class SignalSet:
    """Structured, explainable signals extracted from unstructured notes.

    Every list holds the *verbatim matched phrases* so each signal can be
    shown to a reviewer as evidence.
    """

    intent_level: str = "none"  # strong | moderate | weak | none
    intent_phrases: List[str] = field(default_factory=list)

    pain_strong: bool = False  # "X is eating our week"
    pain_phrases: List[str] = field(default_factory=list)  # concrete processes
    pain_vague: bool = False

    urgency_level: str = "none"  # immediate | this_month | one_month | none
    urgency_phrases: List[str] = field(default_factory=list)

    authority_level: str = "none"  # confirmed | team_needed | none
    authority_phrases: List[str] = field(default_factory=list)

    budget_signal: str = "none"  # approved | have_some | sensitive | wont_share | not_locked | none
    budget_phrases: List[str] = field(default_factory=list)

    disqualifiers: List[str] = field(default_factory=list)  # category names
    negatives: List[str] = field(default_factory=list)  # "can't pay" etc.
    fit_tags: List[str] = field(default_factory=list)  # agency, small_local, saas...


@dataclass
class ScoringResult:
    """Explainable output for one lead."""

    lead_id: str
    total: int
    recommendation: str
    factors: Dict[str, int]  # factor name -> points awarded
    factor_details: Dict[str, List[str]]  # factor name -> human reasons
    disqualifiers: List[str]
    key_reason: str
    is_disqualifier_override: bool = False
    used_early_stage_floor: bool = False
