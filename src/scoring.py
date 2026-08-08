"""Deterministic, explainable lead scoring.

Every factor is a transparent rule that records *why* points were awarded.
Weights and thresholds are centralised in ``SCORING_CONFIG`` so the model can
be tuned without touching any logic.

Scoring dimensions (max 100 points)
-----------------------------------
Intent 25 | Pain/Need 20 | Budget 20 | Urgency 15 | Authority 10 | Fit 10

Recommendation logic
--------------------
1. Any disqualifier signal  -> ``Disqualify`` (override, even at 100 points)
2. score >= contact_now     -> ``Contact Now``
3. score >= nurture         -> ``Nurture``
4. early-stage floor        -> ``Nurture`` (legit business, no negatives,
                              no weak-fit tags — "very early startup, might grow")
5. otherwise                -> ``Disqualify`` (consistently weak signals)
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

import pandas as pd

from .models import CONTACT_NOW, DISQUALIFY, NURTURE, ScoringResult, SignalSet
from .notes import extract_signals

# ---------------------------------------------------------------------------
# THE ONE PLACE TO TUNE THE MODEL
# ---------------------------------------------------------------------------
SCORING_CONFIG: Dict[str, Any] = {
    "intent": {"max": 25, "strong": 25, "moderate": 15, "weak": 8, "none": 0},
    "pain": {"max": 20, "strong": 20, "moderate": 12, "weak": 6, "none": 0},
    "budget": {
        "max": 20,
        "bins": [  # (monthly_usd, points)
            (15000, 20),
            (10000, 18),
            (5000, 16),
            (2500, 12),
            (1000, 8),
            (1, 4),
            (0, 0),
        ],
        "missing": 6,          # unknown budget is neutral, never fatal
        "missing_approved": 16,  # unknown but notes say "budget approved"
        "missing_have_some": 10,  # "have some budget" / "real budget" / "budgeted"
        "missing_sensitive": 4,   # "price sensitive"
    },
    "urgency": {"max": 15, "immediate": 15, "this_month": 12, "one_month": 6, "none": 0},
    "authority": {
        "max": 10,
        "confirmed": 10,   # DM title + "decision is mine" / "I make the call"
        "title_only": 7,   # decision-maker title, no extra confirmation
        "team_needed": 4,  # "would need to loop in the team" / "who signs off?"
        "none": 3,
    },
    "fit": {"max": 10, "min": 2},
    "thresholds": {"contact_now": 65, "nurture": 35},
    "early_stage_floor": True,
}

DM_TITLE_RE = re.compile(
    r"founder|coo|cto|ceo|owner|director|head of|managing|partner|vp|president|principal",
    re.I,
)
PLACEHOLDER_COMPANY = {"", "nan", "none", "n/a", "-", "asdf", "test", "mystery"}


def _fit_points(
    employee_count: float | None,
    company: str,
    tags: List[str],
) -> Tuple[int, List[str]]:
    """Score business fit from org size + company context + note tags."""
    reasons: List[str] = []

    if employee_count is not None:
        emp = float(employee_count)
        if emp >= 50:
            base = 9
        elif emp >= 10:
            base = 8
        elif emp >= 6:
            base = 6
        else:
            base = 4
        reasons.append(f"~{emp:.0f} employees")
    else:
        base = 6
        reasons.append("company size unknown")

    if not company or company.strip().lower() in PLACEHOLDER_COMPANY:
        base -= 3
        reasons.append("no company info")

    if "agency" in tags or "saas" in tags or "ecom_brand" in tags or "car_dealership" in tags:
        base = min(base + 1, SCORING_CONFIG["fit"]["max"])
        reasons.append(f"context: {', '.join(t for t in tags if t in ('agency','saas','ecom_brand','car_dealership'))}")
    if "small_local" in tags or "cheap_chatbot" in tags:
        base = max(base - 3, SCORING_CONFIG["fit"]["min"])
        reasons.append("off-profile: small local business / cheap chatbot")
    if "solo" in tags:
        base = max(base - 2, SCORING_CONFIG["fit"]["min"])
        reasons.append("solo consultant / one-man shop")
    if "early_startup" in tags:
        base = max(base - 1, 4)
        reasons.append("very early stage")
    if "not_agency" in tags:
        base = max(base - 1, SCORING_CONFIG["fit"]["min"])

    return max(min(base, SCORING_CONFIG["fit"]["max"]), SCORING_CONFIG["fit"]["min"]), reasons


def _recommend(
    total: int,
    sig: SignalSet,
    genuine_business: bool,
    weak_fit: bool,
    config: Dict[str, Any],
) -> Tuple[str, bool]:
    """Map a score to a recommendation. Returns (label, used_floor)."""
    if sig.disqualifiers:
        return DISQUALIFY, False
    if total >= config["thresholds"]["contact_now"]:
        # A high score alone is not enough — Contact Now is reserved for leads
        # with a hard buying signal: strong intent, OR clear pain + approved
        # budget. Otherwise a great-fit but uncommitted lead stays Nurture.
        if sig.intent_level == "strong" or (sig.pain_strong and sig.budget_signal == "approved"):
            return CONTACT_NOW, False
        return NURTURE, False
    if total >= config["thresholds"]["nurture"]:
        return NURTURE, False
    # Early-stage floor: genuine business context, no negative signals,
    # and no weak-fit tags -> keep warm rather than discard.
    if (
        config["early_stage_floor"]
        and genuine_business
        and not sig.negatives
        and not weak_fit
    ):
        return NURTURE, True
    return DISQUALIFY, False


def score_row(row: pd.Series, sig: SignalSet | None = None) -> ScoringResult:
    """Score a single cleaned lead row (a Series from the cleaned frame)."""
    sig = sig or extract_signals(
        str(row.get("notes", "")), str(row.get("title", "")), str(row.get("company", ""))
    )
    cfg = SCORING_CONFIG
    factors: Dict[str, int] = {}
    details: Dict[str, List[str]] = {}

    # ---- Intent (25) ----------------------------------------------------
    factors["intent"] = cfg["intent"][sig.intent_level]
    details["intent"] = list(sig.intent_phrases) or [f"no intent signal ({sig.intent_level})"]

    # ---- Pain / need (20) ----------------------------------------------
    if sig.pain_strong:
        factors["pain"] = cfg["pain"]["strong"]
        details["pain"] = ["'eating our week' — concrete, painful process"] + list(sig.pain_phrases)
    elif sig.pain_phrases:
        factors["pain"] = cfg["pain"]["moderate"]
        details["pain"] = ["concrete process described"] + list(sig.pain_phrases)
    elif sig.pain_vague:
        factors["pain"] = cfg["pain"]["weak"]
        details["pain"] = ["vague mention of inefficiency"]
    else:
        factors["pain"] = cfg["pain"]["none"]
        details["pain"] = ["no concrete pain identified"]

    # ---- Budget (20) ----------------------------------------------------
    status = str(row.get("budget_status", "missing"))
    value = row.get("budget_monthly")
    b_cfg = cfg["budget"]
    if status == "zero":
        factors["budget"] = b_cfg["bins"][-1][1]
        details["budget"] = ["budget is 0"]
    elif status == "numeric" and pd.notna(value):
        pts = next(p for floor, p in b_cfg["bins"] if float(value) >= floor)
        factors["budget"] = pts
        details["budget"] = [f"monthly budget ~${float(value):,.0f}"]
    else:  # missing / tbd / unparseable -> neutral, informed by notes
        if sig.budget_signal == "approved":
            factors["budget"] = b_cfg["missing_approved"]
            details["budget"] = ["budget not disclosed but notes say 'budget approved'"]
        elif sig.budget_signal == "have_some":
            factors["budget"] = b_cfg["missing_have_some"]
            details["budget"] = ["budget not disclosed but notes indicate funding exists"]
        elif sig.budget_signal == "sensitive":
            factors["budget"] = b_cfg["missing_sensitive"]
            details["budget"] = ["budget unknown and notes signal price sensitivity"]
        else:
            factors["budget"] = b_cfg["missing"]
            details["budget"] = [f"budget unknown ({status}) — treated neutrally"]

    # ---- Urgency (15) ---------------------------------------------------
    factors["urgency"] = cfg["urgency"][sig.urgency_level]
    details["urgency"] = list(sig.urgency_phrases) or ["no timeline given"]

    # ---- Authority (10) -------------------------------------------------
    title = str(row.get("title", ""))
    title_dm = bool(DM_TITLE_RE.search(title))
    if sig.authority_level == "confirmed":
        factors["authority"] = cfg["authority"]["confirmed"]
        details["authority"] = [f"{title or 'unknown'} title"] + list(sig.authority_phrases)
    elif sig.authority_level == "team_needed":
        factors["authority"] = cfg["authority"]["team_needed"]
        details["authority"] = [f"{title or 'unknown'} title"] + list(sig.authority_phrases)
    elif title_dm:
        factors["authority"] = cfg["authority"]["title_only"]
        details["authority"] = [f"{title} (decision-maker title)"]
    else:
        factors["authority"] = cfg["authority"]["none"]
        details["authority"] = [f"{title or 'no title'} — no authority signal"]

    # ---- Fit (10) -------------------------------------------------------
    emp = row.get("employee_count")
    factors["fit"], details["fit"] = _fit_points(
        None if pd.isna(emp) else float(emp),
        str(row.get("company", "")),
        sig.fit_tags,
    )

    total = int(sum(factors.values()))
    weak_fit = bool({"small_local", "cheap_chatbot"} & set(sig.fit_tags))
    genuine_business = bool(
        (company := str(row.get("company", "")).strip().lower()) not in PLACEHOLDER_COMPANY
        or pd.notna(row.get("employee_count"))
        or str(row.get("website", "")).strip()
    )

    rec, used_floor = _recommend(total, sig, genuine_business, weak_fit, cfg)

    # ---- Key reason (one line for the results table) ----------------------
    key_reason = _key_reason(rec, total, sig, factors, details, used_floor)

    return ScoringResult(
        lead_id=str(row.get("lead_id", "")),
        total=total,
        recommendation=rec,
        factors=factors,
        factor_details=details,
        disqualifiers=list(sig.disqualifiers),
        key_reason=key_reason,
        is_disqualifier_override=bool(sig.disqualifiers),
        used_early_stage_floor=used_floor,
    )


def _key_reason(
    rec: str,
    total: int,
    sig: SignalSet,
    factors: Dict[str, int],
    details: Dict[str, List[str]],
    used_floor: bool = False,
) -> str:
    if rec == DISQUALIFY and sig.disqualifiers:
        return f"Disqualified: {', '.join(sig.disqualifiers[:3])}."
    if rec == CONTACT_NOW:
        parts = [f"Strong intent ({factors['intent']})"]
        if sig.pain_strong:
            parts.append("clear automation pain")
        if sig.budget_signal == "approved":
            parts.append("budget approved")
        if sig.urgency_level == "immediate":
            parts.append("wants to start soon")
        if sig.authority_level == "confirmed":
            parts.append("decision-maker")
        return " + ".join(parts) + "."
    if rec == NURTURE:
        if used_floor:
            return "Early-stage / underdeveloped lead — kept warm for later follow-up."
        gaps = []
        if sig.budget_signal in ("not_locked", "wont_share") or factors["budget"] < 10:
            gaps.append("budget not established")
        if sig.urgency_level == "none":
            gaps.append("no timeline")
        if sig.authority_level == "team_needed":
            gaps.append("needs internal sign-off")
        if sig.intent_level == "weak":
            gaps.append("still researching")
        if not gaps:
            gaps.append("good fit, timing/budget still open")
        return f"Real interest ({total} pts) but {', '.join(gaps)}."
    return f"Weak signals overall ({total} pts): low intent, no budget, no urgency."


def score_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Score every *valid* lead row in the cleaned frame; add result columns."""
    out = df.copy()
    results = [score_row(row) for _, row in df.iterrows()]
    out["score"] = [r.total for r in results]
    out["recommendation"] = [r.recommendation for r in results]
    out["key_reason"] = [r.key_reason for r in results]
    out["disqualifiers"] = [", ".join(r.disqualifiers) for r in results]
    out["factors"] = [r.factors for r in results]
    out["factor_details"] = [r.factor_details for r in results]
    out["is_disqualifier_override"] = [r.is_disqualifier_override for r in results]
    out["used_early_stage_floor"] = [r.used_early_stage_floor for r in results]
    # factor columns for easy filtering/export
    for f in ("intent", "pain", "budget", "urgency", "authority", "fit"):
        out[f"pts_{f}"] = [r.factors.get(f, 0) for r in results]
    return out
