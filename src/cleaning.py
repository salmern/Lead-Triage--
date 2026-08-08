"""Reusable, documented data-cleaning pipeline for lead exports.

The pipeline is designed to accept any future export (xlsx/csv) that has the
same *general* structure as the assessment workbook.

Principles
----------
* **Normalise, don't destroy** — original values are preserved (trimmed) and
  new typed/derived columns are added (``email_norm``, ``budget_monthly``,
  ``employee_count``, ``created_parsed`` ...).
* **Flag, don't silently drop** — junk and duplicate rows are marked with a
  reason (``is_junk`` / ``is_duplicate``) and reported in the cleaning report;
  scoring only runs on rows where ``valid_lead`` is True.
* **Tolerant parsing** — handles the messy real-world formats seen in the
  dataset (``$6-8k``, ``5,000/mo``, ``TBD``, ``35-55`` employees, day-first
  dates, ``[at]`` emails ...).
"""
from __future__ import annotations

import re
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

# Column aliases so future exports can name things slightly differently
COLUMN_ALIASES = {
    "budget": "monthly_budget",
    "created_at": "created",
    "createddate": "created",
    "lead": "lead_id",
    "company_name": "company",
    "company_size": "employees",
    "contact": "name",
    "fullname": "name",
    "monthlybudget": "monthly_budget",
}

EMPTY_TOKENS = {"", "nan", "none", "n/a", "na", "-", "?", "null", "unknown"}

JUNK_LEAD_IDS = {"header", "asdf", "testrow"}
JUNK_NOTES_RE = re.compile(r"\btest test\b|\btest entry\b|\bqa test\b|\bignore this\b", re.I)
QA_ID_PREFIX = "l-900"  # dataset author's QA-marker rows (L-9001 ... L-9004)

ORG_SIZE_RE = re.compile(r"\b(\d{1,3})\s*\+?\s*people\b", re.I)


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase/underscore columns and apply the alias map."""
    df = df.copy()
    rename = {}
    for c in df.columns:
        norm = str(c).strip().lower().replace(" ", "_").replace("-", "_")
        norm = COLUMN_ALIASES.get(norm, norm)
        rename[c] = norm
    return df.rename(columns=rename)


def _col(df: pd.DataFrame, name: str) -> pd.Series:
    """Return a column or an all-NaN placeholder (missing column is safe)."""
    if name in df.columns:
        return df[name]
    return pd.Series(np.nan, index=df.index, dtype=object)


def _clean_str(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def clean_email(value: Any) -> Tuple[str, bool]:
    """Normalise an email; return (normalised, is_valid). [at]/(at) -> @."""
    em = _clean_str(value).lower().replace("[at]", "@").replace("(at)", "@")
    em = re.sub(r"\s+", "", em)
    valid = bool(re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]{2,}", em))
    return em, valid


def parse_created(value: Any) -> Tuple[Optional[pd.Timestamp], str]:
    """Parse a created date. Dash-separated dates are read day-first
    (the dataset uses DD-MM-YYYY for dashes, MM/DD/YYYY for slashes)."""
    if value is None or pd.isna(value):
        return None, "missing"
    s = str(value).strip()
    if not s or s.lower() in EMPTY_TOKENS:
        return None, "missing"
    try:
        if re.fullmatch(r"\d{1,2}-\d{1,2}-\d{4}", s):
            return pd.to_datetime(s, format="%d-%m-%Y"), "parsed"
        if re.fullmatch(r"\d{4}-\d{1,2}-\d{1,2}", s):  # ISO
            return pd.to_datetime(s), "parsed"
        return pd.to_datetime(s, format="mixed"), "parsed"
    except (ValueError, TypeError):
        return None, "unparseable"


def parse_budget(value: Any) -> Tuple[Optional[float], str]:
    """Parse a budget string to a monthly midpoint value.

    Handles: 5,000/mo | $6k/mo | $6-8k | 15k/mo | 10,000 | 4000 | 0 | TBD | depends
    Ranges collapse to their midpoint (documented assumption).
    """
    if value is None or pd.isna(value):
        return None, "missing"
    if isinstance(value, (int, float)):
        v = float(value)
        return (v, "zero") if v == 0 else (v, "numeric")
    t = (
        str(value)
        .lower()
        .replace(",", "")
        .replace("$", "")
        .replace("/mo", "")
        .replace("/month", "")
        .strip()
    )
    if t in ("", "nan", "none"):
        return None, "missing"
    if t in ("tbd", "depends", "-", "n/a", "?", "unknown", "undisclosed"):
        return None, "tbd"
    # range where the k suffix applies to the whole span: "6-8k" -> 6k-8k
    m_range = re.fullmatch(r"(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*(k)?", t)
    if m_range:
        lo, hi = float(m_range.group(1)), float(m_range.group(2))
        if m_range.group(3):
            lo, hi = lo * 1000, hi * 1000
        return (lo + hi) / 2.0, "numeric"
    nums = re.findall(r"(\d+(?:\.\d+)?)\s*(k)?", t)
    vals = [float(a) * (1000 if b == "k" else 1) for a, b in nums]
    if not vals:
        return None, "unparseable"
    mid = (min(vals) + max(vals)) / 2.0
    return (mid, "zero") if mid == 0 else (mid, "numeric")


def parse_employees(value: Any) -> Tuple[Optional[float], str]:
    """Parse an employee-count string to a midpoint value.

    Handles: 35-55 | 19+ | ~43 | 76-96 | 50+ | 20 | blank
    """
    if value is None or pd.isna(value):
        return None, "missing"
    if isinstance(value, (int, float)):
        return float(value), "numeric"
    t = (
        str(value)
        .lower()
        .replace(",", "")
        .replace("~", "")
        .replace("approx", "")
        .strip()
    )
    t = re.sub(r"\+$", "", t).strip()
    nums = re.findall(r"(\d+(?:\.\d+)?)", t)
    if not nums:
        return None, "unparseable"
    vals = [float(a) for a in nums]
    return (min(vals) + max(vals)) / 2.0, "numeric"


def org_size_from_notes(notes: str) -> Optional[float]:
    """Fallback organisation size from notes ('...agency, 26 people...')."""
    m = ORG_SIZE_RE.search(notes or "")
    return float(m.group(1)) if m else None


def _base_id(lead_id: str) -> str:
    """Normalise a lead id to compare duplicates: L-1032 == l-1032 == 1032."""
    b = lead_id.replace("-dup", "").strip().lower()
    if b.startswith("l-"):
        b = b[2:]
    return b.strip()


def clean_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Run the full cleaning pipeline.

    Returns (cleaned_frame, report). ``cleaned_frame`` contains *all* rows
    with flags; scoring should use rows where ``valid_lead`` is True.
    """
    df = normalize_column_names(df).copy()
    report: Dict[str, Any] = {"columns": list(df.columns), "warnings": []}
    n0 = len(df)

    # ------------------------------------------------------------------
    # 1. Drop fully blank rows (nothing to salvage; recorded in report)
    # ------------------------------------------------------------------
    blank = df.isna().all(axis=1)
    df = df[~blank].copy()
    report["blank_rows_removed"] = int(blank.sum())

    # ------------------------------------------------------------------
    # 2. Flag junk rows (embedded header, test rows, QA markers)
    # ------------------------------------------------------------------
    ids = _col(df, "lead_id").fillna("").astype(str).str.strip().str.lower()
    notes = _col(df, "notes").fillna("").astype(str).str.strip()
    is_junk = (
        ids.isin(JUNK_LEAD_IDS)
        | ids.str.startswith(QA_ID_PREFIX)
        | notes.str.contains(JUNK_NOTES_RE, regex=True)
        | (_col(df, "email").fillna("").astype(str).str.strip().str.lower() == "email")
        | (_col(df, "title").fillna("").astype(str).str.strip().str.lower() == "title")
    )
    df["is_junk"] = is_junk
    df["junk_reason"] = np.where(is_junk, "test/QA/header marker row", "")

    # ------------------------------------------------------------------
    # 3. Trim whitespace on free-text fields (preserve original case)
    # ------------------------------------------------------------------
    for c in ("lead_id", "name", "email", "company", "website", "title", "source", "notes"):
        if c in df.columns:
            df[c] = df[c].map(lambda v: _clean_str(v))

    # ------------------------------------------------------------------
    # 4. Emails — normalise + validity flag
    # ------------------------------------------------------------------
    if "email" in df.columns:
        norm_emails = df["email"].map(lambda v: clean_email(v))
        df["email_norm"] = [e for e, _ in norm_emails]
        df["email_valid"] = [v for _, v in norm_emails]
        # informational: duplicate email addresses (not auto-removed)
        valid_em = df["email_norm"].where(df["email_valid"], "")
        df["email_duplicate"] = valid_em.ne("") & valid_em.duplicated(keep=False)
    else:
        df["email_norm"] = ""
        df["email_valid"] = False
        df["email_duplicate"] = False
        report["warnings"].append("missing column: email")

    # ------------------------------------------------------------------
    # 5. Dates
    # ------------------------------------------------------------------
    if "created" in df.columns:
        parsed_dates = df["created"].map(parse_created)
        df["created_parsed"] = [d for d, _ in parsed_dates]
        df["date_status"] = [s for _, s in parsed_dates]
    else:
        df["created_parsed"] = pd.NaT
        df["date_status"] = "missing"
        report["warnings"].append("missing column: created")

    # ------------------------------------------------------------------
    # 6. Budget
    # ------------------------------------------------------------------
    if "monthly_budget" in df.columns:
        parsed_budget = df["monthly_budget"].map(parse_budget)
        df["budget_monthly"] = [v for v, _ in parsed_budget]
        df["budget_status"] = [s for _, s in parsed_budget]
    else:
        df["budget_monthly"] = np.nan
        df["budget_status"] = "missing"
        report["warnings"].append("missing column: monthly_budget")

    # ------------------------------------------------------------------
    # 7. Employees (field value, falling back to notes "N people")
    # ------------------------------------------------------------------
    if "employees" in df.columns:
        parsed_emp = df["employees"].map(parse_employees)
        emp_val = [v for v, _ in parsed_emp]
        emp_status = [s for _, s in parsed_emp]
    else:
        emp_val = [None] * len(df)
        emp_status = ["missing"] * len(df)
    df["employee_count"] = [
        v if v is not None else org_size_from_notes(n)
        for v, n in zip(emp_val, df.get("notes", pd.Series("", index=df.index)).map(lambda x: str(x)))
    ]
    df["employees_status"] = [
        s if v is not None else ("missing" if n is None else "numeric(notes)")
        for v, n, s in zip(emp_val, df["employee_count"], emp_status)
    ]

    # ------------------------------------------------------------------
    # 8. Duplicates — keep the *first* occurrence per normalised lead id,
    #    preferring the non-"-dup" row. Flagged, not silently dropped.
    # ------------------------------------------------------------------
    base = ids.map(_base_id)
    # Rows with no usable lead id all map to "no-id"; only the first is kept
    # (blank ids carry no identity, so a second blank-id row is treated as a
    # duplicate). This is a documented policy, not a silent drop.
    base = base.where(base.ne(""), "no-id")
    df["_base"] = base
    df["_pref"] = (~ids.str.contains("-dup", na=False)).astype(int)
    df["_seq"] = np.arange(len(df))
    df.sort_values("_pref", ascending=False, kind="stable", inplace=True)
    keep_first = ~df.duplicated(subset=["_base"], keep="first")
    df["is_duplicate"] = ~keep_first
    df["duplicate_group"] = np.where(df["is_duplicate"], base, "")
    df.sort_values("_seq", inplace=True)
    df.drop(columns=["_pref", "_seq", "_base"], inplace=True)

    df["valid_lead"] = ~df["is_junk"] & ~df["is_duplicate"]

    # ------------------------------------------------------------------
    # 9. Report
    # ------------------------------------------------------------------
    report["rows_total"] = n0
    display_ids = df["lead_id"] if "lead_id" in df.columns else ids
    report["rows_junk"] = int(df["is_junk"].sum())
    report["junk_ids"] = sorted(str(x) for x in display_ids[df["is_junk"]].tolist() if str(x))
    report["rows_duplicate"] = int(df["is_duplicate"].sum())
    report["duplicate_ids"] = sorted(
        str(x) for x in display_ids[df["is_duplicate"]].tolist() if str(x)
    )
    report["valid_leads"] = int(df["valid_lead"].sum())
    report["email_invalid"] = int((~df["email_valid"]).sum())
    report["email_duplicates"] = int(df["email_duplicate"].sum())
    report["dates_unparseable"] = int((df["date_status"] == "unparseable").sum())
    report["budget_missing_or_tbd"] = int(
        df["budget_status"].isin(["missing", "tbd"]).sum()
    )
    report["employees_missing"] = int((df["employees_status"] == "missing").sum())
    return df, report
