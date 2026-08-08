"""Data-cleaning tests: malformed files, missing values, duplicates, parsing."""
import numpy as np
import pandas as pd
import pytest

from src.cleaning import (
    clean_dataframe,
    clean_email,
    org_size_from_notes,
    parse_budget,
    parse_created,
    parse_employees,
)


def _df(rows):
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Emails
# ---------------------------------------------------------------------------
def test_clean_email_normalises_at_style():
    assert clean_email("Ola[at]GrowthMedia.agency") == ("ola@growthmedia.agency", True)


@pytest.mark.parametrize(
    "raw",
    [None, "", "weird-email-no-domain", "ivan@", "deji m.@scaleforge", "a@b"],
)
def test_clean_email_rejects_missing_and_malformed(raw):
    _, valid = clean_email(raw)
    assert valid is False


# ---------------------------------------------------------------------------
# Budgets
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "raw,expected",
    [
        ("5,000/mo", 5000.0),
        ("$6k/mo", 6000.0),
        ("$6-8k", 7000.0),      # range -> midpoint
        ("5k-7k", 6000.0),
        ("18k", 18000.0),
        ("10,000", 10000.0),
        ("4000", 4000.0),
        ("$8,000/mo", 8000.0),
        ("15k/mo", 15000.0),
        (0, 0.0),
        (4000, 4000.0),
    ],
)
def test_parse_budget_numeric(raw, expected):
    val, status = parse_budget(raw)
    assert val == expected
    assert status in ("numeric", "zero")


@pytest.mark.parametrize("raw", ["TBD", "tbd", "depends", "", None, "?"])
def test_parse_budget_unknown_is_not_fatal(raw):
    val, status = parse_budget(raw)
    assert val is None
    assert status in ("missing", "tbd")


def test_parse_budget_unparseable():
    val, status = parse_budget("asdf")
    assert val is None
    assert status == "unparseable"


# ---------------------------------------------------------------------------
# Employees
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "raw,expected",
    [("35-55", 45.0), ("19+", 19.0), ("~43", 43.0), ("70+", 70.0), ("76-96", 86.0), ("20", 20.0)],
)
def test_parse_employees_ranges(raw, expected):
    val, status = parse_employees(raw)
    assert val == expected
    assert status == "numeric"


def test_parse_employees_missing_and_unparseable():
    assert parse_employees(None)[1] == "missing"
    assert parse_employees("asdf")[1] == "unparseable"


def test_org_size_from_notes():
    assert org_size_from_notes("We're a agency, 26 people. Want automation.") == 26.0
    assert org_size_from_notes("no size here") is None


# ---------------------------------------------------------------------------
# Dates
# ---------------------------------------------------------------------------
def test_parse_created_day_first_for_dashes():
    val, status = parse_created("28-06-2024")
    assert status == "parsed"
    assert val == pd.Timestamp("2024-06-28")


def test_parse_created_month_first_for_slashes():
    val, status = parse_created("06/28/2024")
    assert status == "parsed"
    assert val == pd.Timestamp("2024-06-28")


def test_parse_created_iso():
    val, status = parse_created("2024-06-08")
    assert status == "parsed"
    assert val == pd.Timestamp("2024-06-08")


def test_parse_created_invalid():
    assert parse_created("not-a-date")[1] == "unparseable"
    assert parse_created(None)[1] == "missing"


# ---------------------------------------------------------------------------
# Pipeline behaviour
# ---------------------------------------------------------------------------
def test_pipeline_flags_junk_rows():
    df = _df(
        [
            {"lead_id": "header", "created": "lead_id", "name": "name", "email": "email", "notes": ""},
            {"lead_id": "asdf", "email": "asdf@asdf.com", "notes": "test test ignore this"},
            {"lead_id": "L-9001", "email": "", "notes": ""},
            {"lead_id": "L-1", "email": "a@b.co", "notes": "Interested in automating X."},
        ]
    )
    cleaned, report = clean_dataframe(df)
    assert report["rows_junk"] == 3
    assert bool(cleaned.loc[cleaned["lead_id"] == "L-1", "valid_lead"].iloc[0])


def test_pipeline_flags_duplicates_prefers_non_dup_suffix():
    df = _df(
        [
            {"lead_id": "L-1205-dup", "email": "x@y.co", "notes": "(duplicate submission) A"},
            {"lead_id": "L-1205", "email": "x@y.co", "notes": "B"},
        ]
    )
    cleaned, report = clean_dataframe(df)
    assert report["rows_duplicate"] == 1
    assert cleaned.loc[cleaned["is_duplicate"], "lead_id"].iloc[0] == "L-1205-dup"


def test_pipeline_flags_exact_duplicate_ids():
    df = _df(
        [
            {"lead_id": "L-1032", "email": "a@b.co", "notes": "one"},
            {"lead_id": "L-1032", "email": "a@b.co", "notes": "two"},
        ]
    )
    cleaned, report = clean_dataframe(df)
    assert report["rows_duplicate"] == 1
    assert cleaned["valid_lead"].sum() == 1


def test_pipeline_blank_rows_removed():
    df = _df(
        [
            {"lead_id": "L-1", "email": "a@b.co", "notes": "ok"},
            {"lead_id": None, "email": None, "notes": None},
        ]
    )
    cleaned, report = clean_dataframe(df)
    assert report["blank_rows_removed"] == 1


def test_pipeline_missing_columns_safe():
    df = pd.DataFrame({"lead_id": ["L-1"], "notes": ["interested"]})
    cleaned, report = clean_dataframe(df)
    assert bool(cleaned["valid_lead"].iloc[0])
    assert any("missing column" in w for w in report["warnings"])


def test_pipeline_empty_input():
    cleaned, report = clean_dataframe(pd.DataFrame())
    assert report["rows_total"] == 0
    assert len(cleaned) == 0


def test_pipeline_invalid_budget_and_dates_recorded_not_fatal():
    df = _df(
        [
            {
                "lead_id": "L-1",
                "email": "a@b.co",
                "monthly_budget": "asdf",
                "created": "not-a-date",
                "notes": "Interested in automating X.",
            }
        ]
    )
    cleaned, report = clean_dataframe(df)
    assert (cleaned["budget_status"] == "unparseable").any()
    assert (cleaned["date_status"] == "unparseable").any()
    assert bool(cleaned["valid_lead"].iloc[0])


def test_pipeline_duplicate_emails_flagged_not_removed():
    df = _df(
        [
            {"lead_id": "L-1", "email": "same@x.co", "notes": "a"},
            {"lead_id": "L-2", "email": "same@x.co", "notes": "b"},
        ]
    )
    cleaned, report = clean_dataframe(df)
    assert report["email_duplicates"] == 2  # both rows share the duplicate address
    assert cleaned["valid_lead"].sum() == 2


def test_pipeline_normalises_column_aliases():
    df = _df([{"lead": "L-1", "budget": "$6k/mo", "notes": "Interested in automating X."}])
    cleaned, _ = clean_dataframe(df)
    assert "lead_id" in cleaned.columns
    assert "monthly_budget" in cleaned.columns
    assert cleaned["budget_monthly"].iloc[0] == 6000.0
