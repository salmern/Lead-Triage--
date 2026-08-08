"""Deterministic signal extraction from unstructured lead notes.

Design
------
A rule-based phrase/pattern matcher (case-insensitive, word-boundary regex).
It is intentionally *not* an LLM so that results are:

* reproducible  — same input, same output, always
* explainable   — every signal records the exact matched phrase
* free to run   — no API keys, no network
* easy to test  — plain functions over plain strings

An LLM layer could be added later as an *optional* enrichment, never as a
dependency of the core score.
"""
from __future__ import annotations

import re
from typing import List

from .models import SignalSet

# --------------------------------------------------------------------------
# Phrase tables (single source of truth for signals)
# --------------------------------------------------------------------------

INTENT_STRONG = [
    r"\bwant it automated\b",            # "Want it automated end to end."
    r"\bwants? to start\b",              # "wants to start ASAP"
    r"\bready to pilot\b",               # "ready to pilot in the next 2 weeks"
    r"\bkeen to move fast\b",
    r"\bbudget approved\b",
    r"\bthis is my priority to solve\b",
    r"\bwant to automate\b",             # "Want to automate lead routing."
    r"\bwant an embedded dev\b|\bwant to embed\b|\bwant a dev\b",
    r"\bbudgeted,? serious\b",           # "Budgeted, serious."
    r"\bwant(?:s|ing)? (?:an|a) lead chatbot\b",  # "wants a lead chatbot" / "wanting a lead chatbot"
]

INTENT_MODERATE = [
    r"\blooking into automating\b",
    r"\bexploring automating\b",
    r"\bcurious about automating\b",
    r"\binterested in automating\b",
    r"\binteres?ted in automating\b",
    r"\bcomparing a few options\b",
    r"\bautomate my own\b",
    r"\bai ops help\b",
    r"\bwants? ai ops help\b",
    r"\bwould like to automate\b",
    r"\bwant to add an ai service line\b",
    r"\bwant to automate lead routing\b",
]

INTENT_WEAK = [
    r"\bresearching the market\b",
    r"\bmaybe later\b",
    r"\bmight grow\b",
    r"\bmostly researching\b",
    r"\bjust learning\b|\blearning :\)\b",
    r"\bwould love to learn\b",
]

INTENT_VAGUE = [r"\bvague on scope\b", r"\bnot totally sure what we need yet\b"]

# Concrete, automatable processes the agency's services can address
PROCESS_PATTERNS = [
    r"\bpacing ad budgets\b",
    r"\benriching and scoring leads\b",
    r"\bmoving leads between apollo and the crm\b",
    r"\bcopy-pasting data between spreadsheets and hubspot\b",
    r"\bstitching together reporting\b",
    r"\bbuilding client reports\b",
    r"\bsummarizing call recordings into crm notes\b",
    r"\btriaging a flooded shared inbox\b",
    r"\bqualifying inbound leads before they hit the reps\b",
    r"\bchasing follow-ups across email and whatsapp\b",
    r"\bmanual lead routing\b",
    r"\bresearching prospects and drafting first-touch messages\b",
    r"\blead routing\b",
    r"\blead chatbot\b",
    r"\binternal tools\b|\binternal ai tools\b",
    r"\bmanual processes\b",
    r"\brepetitive workflows\b",
    r"\boperational\b",
]

PAIN_PHRASE = re.compile(r"\beating our week\b", re.I)
PAIN_VAGUE = [r"\bwasting\b", r"\btoo much time\b", r"\bpain point\b"]

URGENCY = {
    "immediate": [
        r"\bstart asap\b",
        r"\bwants? to start asap\b",
        r"\bkeen to move fast\b",
        r"\bnext 2 weeks\b|\bnext two weeks\b",
        r"\bwithin 2 weeks\b|\bwithin two weeks\b",
        r"\bmove in 2 weeks\b",
        r"\bmove within 2 weeks\b",
        r"\basap\b",
    ],
    "this_month": [
        r"\bstart this month\b",
        r"\bwant to start this month\b",
        r"\bdecision this month\b",
        r"\bthis month\b",
        r"\bpriority for the quarter\b",
        r"\bthis is a priority for the quarter\b",
    ],
    "one_month": [
        r"\bdecision in about a month\b",
        r"\bin about a month\b",
        r"\bwithin a month\b",
        r"\bnext month\b",
    ],
}

AUTHORITY_CONFIRMED = [
    r"\bdecision is mine\b",
    r"\bi make the call\b",
    r"\bi make the call here\b",
    r"\bthis is my priority to solve\b",
    r"\bmy decision\b",
]
AUTHORITY_TEAM = [
    r"\bwould need to loop in the team\b",
    r"\bneed to loop in\b",
    r"\bloop in the team\b",
    r"\bnot sure who signs off\b",
    r"\bwho signs off internally\b",
    r"\bsigns off internally\b",
]

BUDGET_SIGNALS = {
    "approved": [r"\bbudget approved\b"],
    "have_some": [
        r"\bhave some budget\b",
        r"\breal budget\b",
        r"\bmoney to spend\b",
        r"\bbudgeted\b",
    ],
    "sensitive": [r"\bprice sensitive\b", r"\bbudget way below range\b"],
    "wont_share": [r"\bwont share budget\b", r"\bwon't share budget\b", r"\bdepends what you can do\b"],
    "not_locked": [r"\bbudget not locked\b", r"\bnot locked yet\b"],
}

# Disqualifier categories. "not a buyer/not a client" are collected under the
# explicit category and also enrich category-specific ones.
DISQUALIFIERS = {
    "job seeker": [
        r"\bnot looking to buy\b",
        r"\blooking for a role\b",
        r"\blooking for employment\b",
        r"\blooking for a job\b",
        r"\bjoin your team\b",
        r"\bhiring developers\b|\bhiring devs\b",
        r"\battaching my cv\b",
    ],
    "student / learner": [
        r"\bstudent\b",
        r"\bbootcamp\b",
        r"\buniversity project\b",
        r"\binterview your founder\b",
        r"\bfree material\b",
        r"\bfree template\b",
        r"\bfinal year\b",
        r"\bdoing a project on\b",
    ],
    "vendor / seller": [
        r"\bdevs on our bench\b",
        r"\bplace candidates\b",
        r"\boffshore dev team\b",
        r"\bcheap smm panel\b",
        r"\bbuy followers\b",
        r"\bhigh-da backlinks\b",
        r"\bbulk email blasting\b",
        r"\bselling a bulk\b",
    ],
    "spam": [
        r"\bwon \$1,000,000\b",
        r"\bclick here to claim\b",
        r"\breply stop to opt out\b",
    ],
    "competitor": [
        r"\bcompeting automation agency\b|\bcompeting agency\b",
        r"\bwe do similar work\b",
        r"\bbenchmarking\b",
    ],
    "vc intro (not a direct buyer)": [
        r"\bvc here\b",
        r"\bportfolio companies\b",
    ],
    "press / journalist": [
        r"\bjournalist\b",
        r"\blooking for a quote\b",
    ],
    "newsletter signup by mistake": [
        r"\bnewsletter signup\b",
    ],
    "test / qa": [
        r"\btest test\b",
        r"\btest entry\b",
        r"\bqa test\b",
        r"\bignore this\b",
    ],
    "explicitly not a buyer/client": [
        r"\bnot a buyer\b",
        r"\bnot a client\b",
        r"\bnot a direct buyer\b",
    ],
}

# Non-fatal negatives that kill the "early-stage floor" (below)
NEGATIVES = [
    r"\bcan't really pay\b|\bcant really pay\b|\bcan't pay\b|\bcant pay\b",
    r"\bwould love to learn\b",
    r"\bjust learning\b|\blearning :\)\b",
    r"\bmaybe later\b",
    r"\bnot looking to buy\b",
]

FIT_TAGS = {
    "agency": [r"\bagency\b"],
    "not_agency": [r"\bnot an agency\b"],
    "small_local": [r"\bsmall local business\b"],
    "cheap_chatbot": [r"\bcheap chatbot\b"],
    "car_dealership": [r"\bcar dealership\b"],
    "ecom_brand": [r"\becom brand\b"],
    "saas": [r"\bsaas company\b"],
    "solo": [r"\bsolo consultant\b|\bone-man shop\b|\bone-person shop\b"],
    "early_startup": [r"\bvery early startup\b|\bno real budget yet\b|\bearly stage\b"],
}


def _match(patterns: List[str], text: str) -> List[str]:
    """Return verbatim matches of `patterns` inside `text` (deduped)."""
    found: List[str] = []
    seen = set()
    for pat in patterns:
        for m in re.finditer(pat, text, flags=re.IGNORECASE):
            phrase = m.group(0).strip()
            if phrase.lower() not in seen:
                seen.add(phrase.lower())
                found.append(phrase)
    return found


def _has_any(patterns: List[str], text: str) -> bool:
    return any(re.search(p, text, flags=re.IGNORECASE) for p in patterns)


def extract_signals(notes: str, title: str = "", company: str = "") -> SignalSet:
    """Extract structured signals from a lead's notes (plus title/company context)."""
    text = notes or ""
    ctx = " ".join(x for x in (notes, title, company) if x)

    sig = SignalSet()

    # --- Intent (priority order: strong > moderate > weak > none) ---
    strong = _match(INTENT_STRONG, text)
    moderate = _match(INTENT_MODERATE, text)
    weak = _match(INTENT_WEAK, text)
    if strong:
        sig.intent_level = "strong"
        sig.intent_phrases = strong
    elif moderate:
        # "vague on scope" downgrades an otherwise-moderate interest
        if _has_any(INTENT_VAGUE, text):
            sig.intent_level = "weak"
            sig.intent_phrases = moderate + _match(INTENT_VAGUE, text)
        else:
            sig.intent_level = "moderate"
            sig.intent_phrases = moderate
    elif weak:
        sig.intent_level = "weak"
        sig.intent_phrases = weak
    else:
        sig.intent_level = "none"

    # --- Pain / need ---
    if PAIN_PHRASE.search(text):
        sig.pain_strong = True
        sig.pain_phrases = [PAIN_PHRASE.search(text).group(0)]
    sig.pain_phrases += _match(PROCESS_PATTERNS, text)
    if _has_any(PAIN_VAGUE, text):
        sig.pain_vague = True

    # --- Urgency ---
    for level in ("immediate", "this_month", "one_month"):
        found = _match(URGENCY[level], text)
        if found:
            sig.urgency_level = level
            sig.urgency_phrases = found
            break

    # --- Authority (notes only; title handled in scoring) ---
    confirmed = _match(AUTHORITY_CONFIRMED, text)
    team = _match(AUTHORITY_TEAM, text)
    if confirmed:
        sig.authority_level = "confirmed"
        sig.authority_phrases = confirmed
    elif team:
        sig.authority_level = "team_needed"
        sig.authority_phrases = team

    # --- Budget signals from notes ---
    for label, pats in BUDGET_SIGNALS.items():
        found = _match(pats, text)
        if found:
            sig.budget_signal = label
            sig.budget_phrases = found
            break

    # --- Disqualifiers ---
    for category, pats in DISQUALIFIERS.items():
        if _has_any(pats, ctx):
            sig.disqualifiers.append(category)

    # --- Negatives ---
    sig.negatives = _match(NEGATIVES, text)

    # --- Fit tags ---
    for tag, pats in FIT_TAGS.items():
        if _has_any(pats, ctx):
            sig.fit_tags.append(tag)

    return sig
