"""Scoring tests: the 10 required scenarios, determinism, and a real-data regression."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.cleaning import clean_dataframe
from src.models import CONTACT_NOW, DISQUALIFY, NURTURE
from src.scoring import SCORING_CONFIG, score_dataframe, score_row

REPO_ROOT = Path(__file__).resolve().parents[1]


def row(**kw) -> pd.Series:
    base = {
        "lead_id": "T-1",
        "name": "Test",
        "email_norm": "t@x.co",
        "company": "TestCo",
        "title": "",
        "website": "",
        "notes": "",
        "monthly_budget": None,
        "budget_monthly": np.nan,
        "budget_status": "missing",
        "employees": "",
        "employee_count": np.nan,
        "employees_status": "missing",
        "created_parsed": None,
        "date_status": "missing",
        "source": "test",
    }
    base.update(kw)
    return pd.Series(base)


# ---------------------------------------------------------------------------
# The 10 required scenarios
# ---------------------------------------------------------------------------
def test_1_excellent_lead_contact_now():
    s = score_row(
        row(
            company="BigAgency",
            title="Head of Ops",
            notes=(
                "We're a media buying agency, 58 people. Enriching and scoring leads one by one is "
                "eating our week. Want it automated end to end. Budget approved, wants to start ASAP. "
                "This is my priority to solve."
            ),
            budget_status="numeric",
            budget_monthly=8000.0,
            employee_count=58.0,
        )
    )
    assert s.recommendation == CONTACT_NOW
    assert s.total >= 80


def test_2_moderate_lead_nurture():
    s = score_row(
        row(
            company="MidAgency",
            title="Head of RevOps",
            notes="Curious about automating pacing ad budgets across dozens of accounts manually. Comparing a few options.",
            budget_status="numeric",
            budget_monthly=6000.0,
            employee_count=20.0,
        )
    )
    assert s.recommendation == NURTURE
    assert 35 <= s.total < 65


def test_3_weak_lead_disqualified():
    s = score_row(
        row(
            company="SmallCo",
            title="Freelancer",
            notes="would love to learn but can't really pay right now. maybe later.",
            budget_status="zero",
            budget_monthly=0.0,
            employee_count=1.0,
        )
    )
    assert s.recommendation == DISQUALIFY


def test_4_job_seeker_disqualified():
    s = score_row(
        row(
            company="LuxAuto",
            title="Developer",
            notes="Not looking to buy — I'm a developer looking for a role. Attaching my CV.",
            budget_status="numeric",
            budget_monthly=20000.0,
            employee_count=50.0,
        )
    )
    assert s.recommendation == DISQUALIFY
    assert s.is_disqualifier_override


def test_5_student_disqualified():
    s = score_row(
        row(title="Student", notes="hi! CS student, i love what you do. could you send a free template or resources?")
    )
    assert s.recommendation == DISQUALIFY
    assert "student / learner" in s.disqualifiers


def test_6_spam_disqualified():
    s = score_row(row(notes="You have WON $1,000,000!!! Click here to claim."))
    assert s.recommendation == DISQUALIFY
    assert "spam" in s.disqualifiers


def test_7_high_budget_but_non_buyer_disqualified():
    """The exact trap from the brief: CEO + $20k budget, but notes say job-seeker."""
    s = score_row(
        row(
            title="CEO",
            notes="Not looking to buy — I'm a developer looking for a role.",
            budget_status="numeric",
            budget_monthly=20000.0,
            employee_count=80.0,
        )
    )
    assert s.recommendation == DISQUALIFY
    assert s.is_disqualifier_override


def test_8_low_budget_but_urgent_buyer_contact_now():
    s = score_row(
        row(
            company="LeanAgency",
            title="Founder",
            notes=(
                "We're a cold email agency. Manual lead routing is eating our week. "
                "Want it automated end to end. Budget approved, want to start this month."
            ),
            budget_status="numeric",
            budget_monthly=1500.0,
            employee_count=12.0,
        )
    )
    assert s.recommendation == CONTACT_NOW
    assert s.factors["budget"] < 10  # low budget must not sink an otherwise strong buyer


def test_9_missing_budget_not_auto_disqualified():
    s = score_row(
        row(
            company="BigAgency",
            title="Head of Ops",
            notes=(
                "We're a social media agency, 28 people. Triaging a flooded shared inbox is eating our week. "
                "Want it automated end to end. Budget approved, want to start this month. I make the call here."
            ),
            budget_status="missing",
            budget_monthly=np.nan,
            employee_count=28.0,
        )
    )
    assert s.recommendation != DISQUALIFY
    assert s.recommendation == CONTACT_NOW


def test_10_missing_title_handled_safely():
    s = score_row(
        row(
            company="AgencyCo",
            title="",
            notes="We're a full-service marketing agency. Want to automate lead routing.",
            budget_status="numeric",
            budget_monthly=6000.0,
            employee_count=30.0,
        )
    )
    assert s.recommendation in (CONTACT_NOW, NURTURE, DISQUALIFY)
    assert s.factors["authority"] == SCORING_CONFIG["authority"]["none"]


# ---------------------------------------------------------------------------
# Behavioural guarantees
# ---------------------------------------------------------------------------
def test_buying_signal_gate_high_score_alone_not_contact_now():
    """Score >= 65 but no hard buying signal -> Nurture, not Contact Now."""
    s = score_row(
        row(
            company="BigAgency",
            title="VP Growth",
            notes=(
                "Exploring automating pacing ad budgets across dozens of accounts manually. "
                "Have some budget. Decision in about a month."
            ),
            budget_status="numeric",
            budget_monthly=15000.0,
            employee_count=60.0,
        )
    )
    assert s.total >= 65
    assert s.recommendation == NURTURE


def test_score_dataframe_empty_valid_frame_does_not_crash():
    empty = pd.DataFrame(
        columns=["lead_id", "name", "email_norm", "company", "title", "website", "notes",
                 "monthly_budget", "budget_monthly", "budget_status", "employees",
                 "employee_count", "employees_status", "created_parsed", "date_status", "source"]
    )
    out = score_dataframe(empty)
    assert len(out) == 0
    assert "score" in out.columns


def test_early_stage_floor_keeps_legit_leads_warm():
    s = score_row(
        row(
            company="EarlyCo",
            title="Owner",
            notes="very early startup, 3 people, no real budget yet but sharp and might grow.",
            budget_status="missing",
            budget_monthly=np.nan,
            employee_count=3.0,
        )
    )
    assert s.recommendation == NURTURE
    assert s.used_early_stage_floor


def test_factor_points_sum_to_total_and_in_range():
    s = score_row(
        row(
            company="A",
            title="CEO",
            notes="Want it automated end to end. Budget approved. Start ASAP.",
            budget_status="numeric",
            budget_monthly=12000.0,
            employee_count=40.0,
        )
    )
    assert sum(s.factors.values()) == s.total
    assert 0 <= s.total <= 100


def test_deterministic_same_input_same_output():
    kw = dict(
        company="AgencyX",
        title="Head of Ops",
        notes=(
            "We're a B2B marketing agency. Moving leads between apollo and the crm by hand is eating our week. "
            "Want it automated end to end. Budget approved, decision this month. I make the call here."
        ),
        budget_status="numeric",
        budget_monthly=10000.0,
        employee_count=45.0,
    )
    a, b = score_row(row(**kw)), score_row(row(**kw))
    assert (a.total, a.recommendation, a.factors) == (b.total, b.recommendation, b.factors)


# ---------------------------------------------------------------------------
# Regression on the real 520-row dataset (invariants, not hardcoded counts)
# ---------------------------------------------------------------------------
def test_regression_real_dataset():
    path = REPO_ROOT / "data" / "sample" / "sample_leads.xlsx"
    df = pd.read_excel(path)
    cleaned, report = clean_dataframe(df)
    scored = score_dataframe(cleaned[cleaned["valid_lead"]].copy())

    dist = scored["recommendation"].value_counts().to_dict()
    print(
        "\nREGRESSION ->", report["valid_leads"], "valid leads |",
        dist, "| avg score", round(scored["score"].mean(), 1),
    )

    assert report["valid_leads"] > 0
    assert sum(dist.values()) == len(scored)
    assert set(dist) <= {CONTACT_NOW, NURTURE, DISQUALIFY}
    assert scored["score"].between(0, 100).all() and scored["score"].notna().all()
    assert scored["recommendation"].notna().all()

    # Determinism across passes
    scored2 = score_dataframe(cleaned[cleaned["valid_lead"]].copy())
    assert dist == scored2["recommendation"].value_counts().to_dict()
