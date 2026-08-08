"""Lead Triage — Streamlit application.

Run with:  streamlit run app.py

The app accepts any future lead export (.xlsx / .csv) with the same general
structure, cleans it, scores it with a fully explainable rule-based model and
shows a prioritised, filterable pipeline. No API keys required.

This module is presentation-only. All cleaning, scoring and qualification
logic lives in ``src/`` and is intentionally left untouched by the UI.
"""
from __future__ import annotations

import html
import io
import os
import sys

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.cleaning import clean_dataframe
from src.models import CONTACT_NOW, DISQUALIFY, NURTURE
from src.scoring import SCORING_CONFIG, score_dataframe

SAMPLE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data", "sample", "sample_leads.xlsx"
)
SAMPLE_NAME = "bundled sample dataset"

REC_ORDER = [CONTACT_NOW, NURTURE, DISQUALIFY]
FACTOR_ORDER = ["intent", "pain", "budget", "urgency", "authority", "fit"]
FACTOR_MAX = {f: SCORING_CONFIG[f]["max"] for f in FACTOR_ORDER}
FACTOR_LABEL = {
    "intent": "Intent",
    "pain": "Pain / Need",
    "budget": "Budget",
    "urgency": "Urgency",
    "authority": "Authority",
    "fit": "Company Fit",
}

# Presentation-only status styling. Recommendation values are unchanged.
PILL_CLASS = {
    CONTACT_NOW: "pill-now",
    NURTURE: "pill-nurture",
    DISQUALIFY: "pill-disq",
}
DOT_CLASS = {
    CONTACT_NOW: "dot-now",
    NURTURE: "dot-nurture",
    DISQUALIFY: "dot-disq",
}

LOG_LABELS = {
    "rows_total": "Raw rows",
    "valid_leads": "Scored leads",
    "blank_rows_removed": "Blank rows removed",
    "rows_junk": "Junk / QA rows excluded",
    "rows_duplicate": "Duplicate rows excluded",
    "email_invalid": "Invalid email addresses",
    "email_duplicates": "Duplicate email addresses",
    "dates_unparseable": "Unparseable dates",
    "budget_missing_or_tbd": "Budget missing or TBD",
    "employees_missing": "Employee count missing",
}

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;650;700&display=swap');

:root {
  --bg: #FFFFFF;
  --surface: #FFFFFF;
  --surface-2: #FAFAFA;
  --border: #E4E4E7;
  --border-soft: #F1F1F3;
  --text: #18181B;
  --text-2: #3F3F46;
  --muted: #71717A;
  --faint: #A1A1AA;
  --green: #15803D; --green-bg: #F0FDF4; --green-border: #BBF7D0;
  --amber: #B45309; --amber-bg: #FFFBEB; --amber-border: #FDE68A;
  --red: #B91C1C; --red-bg: #FEF2F2; --red-border: #FECACA;
}

html, body, .stApp, [data-testid="stAppViewContainer"] {
  background: var(--bg);
  color: var(--text);
  -webkit-font-smoothing: antialiased;
}
/* Safety net: never let the page itself scroll horizontally */
html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
  overflow-x: hidden;
}
.stMainBlockContainer, [data-testid="stMainBlockContainer"] { max-width: 100%; }
[data-testid="stColumn"], .db-row > div, .rec-block { min-width: 0; }
html, body, .stApp, .stMarkdown, [data-testid="stSidebar"], [data-testid="stSidebarContent"] {
  font-family: 'Inter', ui-sans-serif, system-ui, -apple-system, 'Segoe UI', Roboto, Arial, sans-serif;
}

#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"], [data-testid="stStatusWidget"] {
  display: none;
}
[data-testid="stHeader"] { display: none; }
.main .block-container { padding-top: 1.8rem; padding-bottom: 3rem; max-width: 1360px; }

/* ---- typography ---- */
.main h1 { font-size: 24px; font-weight: 650; letter-spacing: -0.02em; color: var(--text); margin: 0 0 2px; }
.main h2 { font-size: 16px; font-weight: 600; letter-spacing: -0.01em; color: var(--text); margin: 1.6rem 0 0.55rem; }
.main h3 { font-size: 15px; font-weight: 600; letter-spacing: -0.01em; color: var(--text); margin: 1.3rem 0 0.5rem; }
.main [data-testid="stCaptionContainer"] { color: var(--muted); font-size: 13px; }

hr { border: 0; border-top: 1px solid var(--border); margin: 1.2rem 0; }

code {
  background: var(--surface-2);
  border: 1px solid var(--border);
  border-radius: 4px;
  padding: 1px 5px;
  font-size: 12px;
  color: var(--text-2);
}

/* ---- filters panel (custom sidebar) ---- */
[data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:has(#lt-panel-mark) {
  background: var(--surface-2);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 1.05rem 0.9rem 1.35rem;
  align-self: flex-start;
  position: sticky;
  top: 0.6rem;
  max-height: calc(100vh - 1.2rem);
  overflow-y: auto;
}
[data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:has(#lt-panel-mark) hr { margin: 1rem 0; }

/* ---- sidebar toggle button ---- */
[data-testid="stBaseButton-primary"] {
  width: 34px;
  min-width: 34px;
  height: 34px;
  padding: 0;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  font-size: 15px;
  line-height: 1;
  color: var(--text-2) !important;
  background: #fff !important;
  border: 1px solid var(--border) !important;
  box-shadow: none !important;
}
[data-testid="stBaseButton-primary"]:hover {
  background: var(--surface-2) !important;
  border-color: var(--faint) !important;
  color: var(--text) !important;
}

.side-brand {
  display: flex; align-items: center; gap: 8px;
  font-size: 12.5px; font-weight: 650; letter-spacing: 0.02em;
  color: var(--text); margin: 0 0 1.1rem;
}
.brand-mark {
  width: 22px; height: 22px; border-radius: 6px; background: var(--text); color: #fff;
  font-size: 10.5px; font-weight: 700; display: inline-flex; align-items: center; justify-content: center;
  flex: 0 0 auto;
}
.side-section {
  font-size: 11px; font-weight: 600; letter-spacing: 0.09em; text-transform: uppercase;
  color: var(--muted); margin: 0 0 0.55rem;
}
.side-hr { height: 1px; background: var(--border); margin: 1.05rem 0; }

/* ---- controls ---- */
.stButton > button, [data-testid="stDownloadButton"] button {
  border: 1px solid var(--border);
  background: #fff;
  color: var(--text);
  border-radius: 6px;
  font-size: 13px;
  font-weight: 500;
  min-height: 34px;
  transition: background .12s ease, border-color .12s ease;
}
.stButton > button:hover, [data-testid="stDownloadButton"] button:hover {
  background: var(--surface-2);
  border-color: var(--faint);
  color: var(--text);
}

div[data-baseweb="select"] > div {
  border: 1px solid var(--border) !important;
  border-radius: 6px !important;
  background: #fff !important;
  box-shadow: none !important;
  transition: border-color .12s ease;
}
div[data-baseweb="select"] > div:hover { border-color: var(--faint) !important; }
div[data-baseweb="select"]:focus-within > div { border-color: var(--text) !important; box-shadow: 0 0 0 1px var(--text) !important; }
[data-baseweb="popover"] > div { border-radius: 8px; border: 1px solid var(--border); box-shadow: 0 8px 24px rgba(24,24,27,0.08); }

[data-testid="stFileUploader"] section {
  border: 1px dashed var(--border);
  border-radius: 8px;
  background: #fff;
  transition: border-color .12s ease;
}
[data-testid="stFileUploader"] section:hover { border-color: var(--faint); }
[data-testid="stFileUploader"] section button { border-radius: 6px; }

[data-testid="stCheckbox"] label { font-size: 13px; color: var(--text-2); }

/* ---- dataset banner ---- */
.dbanner {
  border: 1px solid var(--border);
  border-radius: 8px;
  background: var(--surface);
  padding: 12px 16px;
  margin: 4px 0 18px;
}
.db-row { display: flex; gap: 12px; align-items: flex-start; }
.db-dot { width: 8px; height: 8px; border-radius: 50%; background: #16A34A; margin-top: 6px; flex: 0 0 auto; }
.db-title { font-size: 13.5px; font-weight: 600; color: var(--text); }
.db-src { font-size: 12px; color: var(--muted); margin-top: 1px; }
.db-stats { font-size: 13px; color: var(--text-2); margin-top: 4px; }
.db-sub { font-size: 12px; color: var(--muted); margin-top: 1px; }
.db-hint { font-size: 12px; color: var(--faint); margin-top: 6px; }

/* ---- KPI row ---- */
.kpi-row { display: grid; grid-template-columns: repeat(5, 1fr); gap: 12px; margin: 2px 0 6px; }
.kpi-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 12px 14px;
}
.kpi-label { font-size: 13px; font-weight: 500; color: var(--muted); display: flex; align-items: center; gap: 7px; }
.kpi-dot { width: 7px; height: 7px; border-radius: 50%; flex: 0 0 auto; }
.kpi-value {
  font-size: 23px; font-weight: 650; letter-spacing: -0.02em;
  color: var(--text); margin-top: 3px; font-variant-numeric: tabular-nums;
}
.dot-now { background: #16A34A; }
.dot-nurture { background: #F59E0B; }
.dot-disq { background: #DC2626; }
@media (max-width: 900px) { .kpi-row { grid-template-columns: repeat(2, 1fr); } }

/* ---- status pills ---- */
.pill {
  display: inline-flex; align-items: center;
  padding: 3px 10px; border-radius: 999px;
  font-size: 12px; font-weight: 600; border: 1px solid transparent;
}
.pill-now { color: var(--green); background: var(--green-bg); border-color: var(--green-border); }
.pill-nurture { color: var(--amber); background: var(--amber-bg); border-color: var(--amber-border); }
.pill-disq { color: var(--red); background: var(--red-bg); border-color: var(--red-border); }

/* ---- data grid ---- */
[data-testid="stDataFrame"] { border: 1px solid var(--border); border-radius: 8px; overflow: hidden; }
[data-testid="stDataFrame"] .ag-root-wrapper { font-size: 13px; color: var(--text-2); }
[data-testid="stDataFrame"] .ag-header { background: var(--surface-2); border-bottom: 1px solid var(--border); }
[data-testid="stDataFrame"] .ag-header-cell {
  font-size: 12px; font-weight: 600; color: var(--text-2);
  font-variant-numeric: tabular-nums;
}
[data-testid="stDataFrame"] .ag-row { border-bottom: 1px solid var(--border-soft); background: #fff; }
[data-testid="stDataFrame"] .ag-row-hover { background: #F6F6F7; }
[data-testid="stDataFrame"] .ag-cell { font-size: 13px; }

/* ---- lead detail ---- */
.rec-block { display: flex; align-items: center; gap: 14px; margin-bottom: 6px; }
.score-big { font-size: 27px; font-weight: 650; letter-spacing: -0.02em; color: var(--text); font-variant-numeric: tabular-nums; }
.score-max { font-size: 14px; font-weight: 500; color: var(--faint); }
.key-reason { font-size: 13.5px; color: var(--text-2); margin: 4px 0 8px; line-height: 1.5; }
.flag-note { font-size: 12px; color: var(--muted); margin: 2px 0; }

.sec-title {
  font-size: 11px; font-weight: 600; letter-spacing: 0.07em; text-transform: uppercase;
  color: var(--muted); margin: 1.1rem 0 0.35rem;
}
.field { display: flex; padding: 5px 0; border-bottom: 1px solid var(--border-soft); }
.field:last-child { border-bottom: none; }
.field-k { flex: 0 0 106px; font-size: 12.5px; color: var(--muted); padding-top: 1px; }
.field-v { flex: 1; font-size: 13.5px; color: var(--text-2); word-break: break-word; }
.tag-invalid {
  font-size: 10.5px; font-weight: 600; color: var(--red);
  background: var(--red-bg); border: 1px solid var(--red-border);
  border-radius: 4px; padding: 1px 5px; margin-left: 6px; vertical-align: 1px;
}
.notes-box {
  background: var(--surface-2); border: 1px solid var(--border); border-radius: 6px;
  padding: 10px 12px; font-size: 13px; color: var(--text-2);
  line-height: 1.55; white-space: pre-wrap;
}

/* ---- scoring factor bars ---- */
.fbar { margin-bottom: 12px; }
.fbar-head { display: flex; justify-content: space-between; align-items: baseline; font-size: 12.5px; margin-bottom: 4px; }
.fbar-name { font-weight: 600; color: var(--text); }
.fbar-val { color: var(--muted); font-weight: 500; font-variant-numeric: tabular-nums; }
.fbar-track { height: 6px; background: var(--border-soft); border-radius: 3px; overflow: hidden; }
.fbar-fill { height: 100%; background: var(--text); border-radius: 3px; }
.fbar-reason { margin-top: 4px; font-size: 12px; color: var(--muted); line-height: 1.45; }

.notice {
  display: flex; gap: 8px; align-items: baseline;
  background: var(--amber-bg); border: 1px solid var(--amber-border);
  border-radius: 6px; padding: 8px 10px; margin-top: 12px;
}
.notice-label { font-size: 12px; font-weight: 600; color: var(--amber); white-space: nowrap; }
.notice-text { font-size: 12.5px; color: var(--text-2); }

/* ---- expanders (methodology / cleaning log) ---- */
details[data-testid="stExpander"] {
  border: 1px solid var(--border);
  border-radius: 8px;
  background: var(--surface);
  overflow: hidden;
}
details[data-testid="stExpander"] > summary {
  padding: 10px 14px;
  font-size: 13.5px; font-weight: 600; color: var(--text);
}
details[data-testid="stExpander"][open] > summary { border-bottom: 1px solid var(--border-soft); }
[data-testid="stExpanderDetails"] { padding: 12px 14px 14px; }

/* ---- markdown tables (methodology) ---- */
.stMarkdown table { border-collapse: collapse; font-size: 13px; width: 100%; }
.stMarkdown th {
  background: var(--surface-2); font-weight: 600; color: var(--text-2);
  text-align: left; padding: 8px 12px; border-bottom: 1px solid var(--border);
}
.stMarkdown td { padding: 8px 12px; border-bottom: 1px solid var(--border-soft); color: var(--text-2); }

/* ---- cleaning log ---- */
.log-table { width: 100%; border-collapse: collapse; font-size: 13px; }
.log-table th { text-align: left; background: var(--surface-2); color: var(--text-2); font-weight: 600; padding: 7px 12px; border-bottom: 1px solid var(--border); }
.log-table td { padding: 7px 12px; border-bottom: 1px solid var(--border-soft); color: var(--text-2); }
.log-table tr:last-child td { border-bottom: none; }
.log-warns { font-size: 12.5px; color: var(--text-2); margin: 10px 0 0; padding-left: 18px; }
.log-note { font-size: 12px; color: var(--muted); margin: 10px 0 0; }

.db-src { word-break: break-all; }

/* =====================================================================
   Responsive layout
   ===================================================================== */

/* ---- tablet ---- */
@media (max-width: 1100px) {
  .main .block-container { padding-left: 1.4rem; padding-right: 1.4rem; }
  .kpi-row { grid-template-columns: repeat(3, 1fr); }
}

/* ---- mobile ---- */
@media (max-width: 768px) {
  .main .block-container { padding: 1rem 0.9rem 2.2rem; max-width: 100%; }
  .main h1 { font-size: 20px; }
  .main h2 { font-size: 15px; }
  .main h3 { font-size: 14px; }
  .kpi-row { grid-template-columns: repeat(2, 1fr); gap: 8px; margin-bottom: 10px; }
  .kpi-card { padding: 10px 12px; }
  .kpi-value { font-size: 20px; }
  .kpi-label { font-size: 12px; }
  /* stack every st.columns block (lead detail, header row) on mobile */
  [data-testid="stHorizontalBlock"] { flex-wrap: wrap; gap: 0.75rem !important; }
  [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
    flex: 1 1 100% !important;
    min-width: 0;
  }

  /* larger tap targets */
  .stButton > button, [data-testid="stDownloadButton"] button { min-height: 38px; }
  [data-testid="stFileUploader"] section { min-height: 44px; }
  /* detail record: narrower label column, notes full width */
  .field-k { flex: 0 0 92px; font-size: 12px; }
  .rec-block { flex-wrap: wrap; gap: 10px; }
  .score-big { font-size: 24px; }
  .dbanner { padding: 10px 12px; }
  /* long tables scroll inside their own container, never the page */
  .stMarkdown table, .log-table { display: block; overflow-x: auto; }
  [data-testid="stDataFrame"] .ag-cell, [data-testid="stDataFrame"] .ag-header-cell { font-size: 12px; }
}

/* ---- small mobile ---- */
@media (max-width: 400px) {
  .main .block-container { padding-left: 0.7rem; padding-right: 0.7rem; }
  .kpi-value { font-size: 18px; }
  .field-k { flex: 0 0 84px; }
}
"""

st.set_page_config(page_title="Lead Triage", layout="wide")
st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Sidebar toggle — custom panel, fully Streamlit-native
#
# The "sidebar" is a normal column in the main area (see render_filters_panel
# below), so no Streamlit sidebar chrome is involved and there is nothing to
# "collapse" in the native sense. The open/closed state lives in the URL
# (?sb=open|closed), which survives refreshes and new sessions, and the toggle
# is an ordinary st.button. No JavaScript, no DOM poking, no reloads.
#
# On mobile the panel becomes a fixed overlay drawer via injected CSS, and the
# toggle floats to the top-right above it, so there is always a visible way to
# open the panel and to close the drawer.
#
# On the very first mobile load there is no ?sb= state yet; a tiny hidden
# same-origin iframe (rendered once, before any state exists) sets ?sb=closed
# and reloads so the drawer does not auto-open on phones. It is a no-op on
# desktop and whenever a state is already present.
# ---------------------------------------------------------------------------
MOBILE_INIT_IFRAME = (
    "<iframe srcdoc=\"<script>try{var p=parent;if(p.innerWidth<769&&p.location.href.indexOf('sb=')<0){var u=new URL(p.location.href);u.searchParams.set('sb','closed');p.location.replace(u.href)}}catch(e){}</script>\" "
    'style="width:0;height:0;border:0;position:absolute;visibility:hidden" '
    'aria-hidden="true" tabindex="-1"></iframe>'
)

# Applied only while the panel is open. On mobile it turns the panel into an
# overlay drawer and floats the toggle above it so both stay reachable.
MOBILE_DRAWER_CSS = """
@media (max-width: 768px) {
  [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:has(#lt-panel-mark) {
    position: fixed !important;
    top: 0; left: 0;
    width: min(88vw, 340px) !important;
    height: 100dvh;
    max-height: none !important;
    background: var(--surface-2);
    border: 1px solid var(--border);
    border-radius: 0;
    z-index: 1000;
    overflow-y: auto;
    padding: 1rem 1rem 1.8rem;
    box-shadow: 0 0 40px rgba(24, 24, 27, 0.18);
  }
  [data-testid="stBaseButton-primary"] {
    position: fixed !important;
    top: 10px; right: 10px;
    z-index: 9999;
  }
}
"""


def sidebar_is_open() -> bool:
    """The panel is open unless the URL says otherwise (?sb=closed)."""
    return st.query_params.get("sb", "open") != "closed"


def set_sidebar(open_: bool) -> None:
    """Persist the panel state in the URL (survives refreshes + new sessions)."""
    st.query_params["sb"] = "open" if open_ else "closed"
    st.rerun()




# ---------------------------------------------------------------------------
# Presentation helpers (pure HTML rendering — no data logic here)
# ---------------------------------------------------------------------------
def esc(value: object) -> str:
    """HTML-escape a value for safe interpolation; NaN/None become ''."""
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return html.escape(str(value))


def pill(rec: str) -> str:
    return f'<span class="pill {PILL_CLASS[rec]}">{esc(rec)}</span>'


def kpi_card(label: str, value: str, dot: str | None = None) -> str:
    d = f'<span class="kpi-dot {dot}"></span>' if dot else ""
    return (
        '<div class="kpi-card">'
        f'<div class="kpi-label">{d}{esc(label)}</div>'
        f'<div class="kpi-value">{esc(value)}</div>'
        "</div>"
    )


def field(k: str, v: str) -> str:
    return f'<div class="field"><span class="field-k">{esc(k)}</span><span class="field-v">{v}</span></div>'


def dataset_banner(source_name: str, report: dict) -> str:
    is_sample = source_name == SAMPLE_NAME
    title = "Task 1 dataset loaded" if is_sample else "Custom dataset loaded"
    src_line = "" if is_sample else f'<div class="db-src">{esc(source_name)}</div>'
    n_raw = report.get("rows_total", 0)
    n_valid = report.get("valid_leads", 0)
    n_blank = report.get("blank_rows_removed", 0)
    n_junk = report.get("rows_junk", 0)
    n_dup = report.get("rows_duplicate", 0)
    stats = f"{n_raw:,} raw rows → {n_valid:,} scored leads"
    excluded = f"{n_blank} blank · {n_junk} junk/QA · {n_dup} duplicates excluded"
    return (
        '<div class="dbanner"><div class="db-row">'
        '<span class="db-dot"></span><div>'
        f'<div class="db-title">{esc(title)}</div>{src_line}'
        f'<div class="db-stats">{esc(stats)}</div>'
        f'<div class="db-sub">{esc(excluded)}</div>'
        '<div class="db-hint">Want to test another export? Upload a CSV or XLSX from the sidebar.</div>'
        "</div></div></div>"
    )


def cleaning_log_html(report: dict) -> str:
    rows = []
    for k, label in LOG_LABELS.items():
        if k in report:
            rows.append(f"<tr><td>{esc(label)}</td><td>{report[k]:,}</td></tr>")
    for key in ("junk_ids", "duplicate_ids"):
        ids = report.get(key) or []
        if not ids:
            continue
        shown = ", ".join(str(x) for x in ids[:8])
        if len(ids) > 8:
            shown += " …"
        label = "Junk row IDs" if key == "junk_ids" else "Duplicate row IDs"
        rows.append(f"<tr><td>{esc(label)}</td><td>{esc(shown)}</td></tr>")
    warnings = report.get("warnings") or []
    warn_html = ""
    if warnings:
        warn_html = '<ul class="log-warns">' + "".join(f"<li>{esc(w)}</li>" for w in warnings) + "</ul>"
    return (
        f'<table class="log-table"><tbody>{"".join(rows)}</tbody></table>'
        + warn_html
        + '<p class="log-note">Rows are flagged and counted — never silently dropped. Scoring runs only on valid leads.</p>'
    )


def lead_detail_html(row: pd.Series) -> tuple[str, str]:
    """Render the left (record) and right (score breakdown) HTML panels."""
    rec = row["recommendation"]

    left = [
        f'<div class="rec-block">{pill(rec)}'
        f'<span class="score-big">{int(row["score"])}<span class="score-max"> / 100</span></span></div>',
        f'<p class="key-reason">{esc(row.get("key_reason"))}</p>',
    ]
    if row.get("is_disqualifier_override"):
        left.append('<p class="flag-note">Recommendation forced by a disqualifier signal — overrides the score.</p>')
    if row.get("used_early_stage_floor"):
        left.append('<p class="flag-note">Kept warm via the early-stage floor — genuine business, no negative signals.</p>')

    contact = []
    contact.append(field("Name", esc(row.get("name")) or "—"))
    email_v = esc(row.get("email_norm")) or "—"
    if not row.get("email_valid", True):
        email_v += '<span class="tag-invalid">invalid</span>'
    contact.append(field("Email", email_v))
    contact.append(field("Title", esc(row.get("title")) or "—"))
    contact.append(field("Source", esc(row.get("source")) or "—"))

    company = []
    company.append(field("Company", esc(row.get("company")) or "—"))
    company.append(field("Website", esc(row.get("website")) or "—"))
    emp_raw = esc(row.get("employees")) or "—"
    emp_parsed = f"~{int(row['employee_count'])}" if pd.notna(row.get("employee_count")) else "unknown"
    company.append(field("Employees", f"{emp_raw} → {emp_parsed}"))
    bud_raw = esc(row.get("monthly_budget")) or "—"
    company.append(field("Budget", f"<code>{bud_raw}</code> → {fmt_money(row.get('budget_monthly'))}/mo"))

    notes = esc(row.get("notes")) or "<em>No notes recorded.</em>"

    left.append('<div class="sec-title">Contact</div>' + "".join(contact))
    left.append('<div class="sec-title">Company</div>' + "".join(company))
    left.append('<div class="sec-title">Original notes</div>' + f'<div class="notes-box">{notes}</div>')

    bars = []
    factors = row["factors"]
    details = row["factor_details"]
    for f in FACTOR_ORDER:
        pts = int(factors[f])
        mx = FACTOR_MAX[f]
        pct = round(pts / mx * 100)
        reasons = details.get(f) or []
        reason_text = "; ".join(esc(r) for r in reasons[:3]) or "no signal detected"
        bars.append(
            '<div class="fbar">'
            f'<div class="fbar-head"><span class="fbar-name">{FACTOR_LABEL[f]}</span>'
            f'<span class="fbar-val">{pts} / {mx}</span></div>'
            f'<div class="fbar-track"><div class="fbar-fill" style="width:{pct}%"></div></div>'
            f'<div class="fbar-reason">{reason_text}</div>'
            "</div>"
        )
    right = ['<div class="sec-title">Scoring breakdown</div>', "".join(bars)]
    if row.get("disqualifiers"):
        dq = esc(", ".join(str(d) for d in row["disqualifiers"]))
        right.append(f'<div class="notice"><span class="notice-label">Disqualifiers detected</span>'
                     f'<span class="notice-text">{dq}</span></div>')
    return "".join(left), "".join(right)


def fmt_money(v) -> str:
    if pd.isna(v):
        return "—"
    return f"${float(v):,.0f}"


# ---------------------------------------------------------------------------
# Pipeline helpers
# ---------------------------------------------------------------------------
def process_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Clean + score a raw export. Returns (scored_valid, cleaned_all, report)."""
    cleaned, report = clean_dataframe(df)
    valid = cleaned[cleaned["valid_lead"]].copy()
    scored = score_dataframe(valid)
    scored = scored.sort_values(["score", "lead_id"], ascending=[False, True]).reset_index(drop=True)
    scored.insert(0, "rank", range(1, len(scored) + 1))
    return scored, cleaned, report


def read_upload(file_bytes: bytes, filename: str) -> pd.DataFrame:
    if filename.lower().endswith(".csv"):
        return pd.read_csv(io.BytesIO(file_bytes))
    return pd.read_excel(io.BytesIO(file_bytes), sheet_name=0)


def load_sample() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    df = pd.read_excel(SAMPLE_PATH, sheet_name=0)
    return process_frame(df)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
if "source_name" not in st.session_state:
    try:
        scored, cleaned, report = load_sample()
        st.session_state["scored"] = scored
        st.session_state["cleaned"] = cleaned
        st.session_state["report"] = report
        st.session_state["source_name"] = SAMPLE_NAME
    except Exception as exc:  # pragma: no cover - defensive
        st.session_state["source_name"] = None
        st.session_state["load_error"] = str(exc)


# ---------------------------------------------------------------------------
# Filters panel (the app's custom "sidebar") — upload, filters, exports.
# Functionality is unchanged from the former st.sidebar block; it is rendered
# as a main-area column (or a drawer on mobile). All filter widgets carry
# explicit keys so their state survives collapsing and reopening the panel.
# ---------------------------------------------------------------------------
def filter_bounds(base: pd.DataFrame) -> tuple[int, int]:
    emp_max = max(50, int(base["employee_count"].dropna().max() // 50 * 50 + 50))
    bud_max = max(20000, int(base["budget_monthly"].dropna().max() // 1000 * 1000 + 1000))
    return emp_max, bud_max


def apply_filters(
    df: pd.DataFrame,
    rec_sel, score_lo, score_hi, src_sel,
    emp_lo, emp_hi, inc_emp_unknown,
    b_lo, b_hi, inc_budget_unknown,
) -> pd.DataFrame:
    out = df[df["recommendation"].isin(rec_sel)]
    out = out[(out["score"] >= score_lo) & (out["score"] <= score_hi)]
    if "(any)" not in src_sel:
        out = out[out["source"].astype(str).isin(src_sel)]
    emp = out["employee_count"]
    if inc_emp_unknown:
        out = out[(emp.isna()) | ((emp >= emp_lo) & (emp <= emp_hi))]
    else:
        out = out[(emp.notna()) & (emp >= emp_lo) & (emp <= emp_hi)]
    bud = out["budget_monthly"]
    if inc_budget_unknown:
        out = out[(bud.isna()) | ((bud >= b_lo) & (bud <= b_hi))]
    else:
        out = out[(bud.notna()) & (bud >= b_lo) & (bud <= b_hi)]
    return out


def render_filters_panel() -> None:
    """Upload, filters and exports for the lead pipeline."""
    # CSS marker: identifies this column so it can be styled / turned into a drawer.
    st.markdown('<span id="lt-panel-mark"></span>', unsafe_allow_html=True)
    st.markdown(
        '<div class="side-brand"><span class="brand-mark">LT</span>Lead Triage</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="side-section">Data</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Upload a lead export",
        type=["xlsx", "csv"],
        key="lead_upload",
        help="Any export with the same general structure as the sample (lead_id, created, name, email, company, employees, website, title, source, monthly_budget, notes).",
    )
    if uploaded is not None:
        try:
            df = read_upload(uploaded.getvalue(), uploaded.name)
            scored, cleaned, report = process_frame(df)
            st.session_state["scored"] = scored
            st.session_state["cleaned"] = cleaned
            st.session_state["report"] = report
            st.session_state["source_name"] = uploaded.name
        except Exception as exc:
            st.error(f"Could not parse `{uploaded.name}`. Expected .xlsx or .csv with a header row.\n\n{exc}")
    if st.button("Reset to sample dataset", type="secondary", width="stretch"):
        # Clear the uploader widget state first, otherwise Streamlit restores the
        # uploaded file on rerun and silently re-loads it over the sample.
        st.session_state.pop("lead_upload", None)
        scored, cleaned, report = load_sample()
        st.session_state["scored"] = scored
        st.session_state["cleaned"] = cleaned
        st.session_state["report"] = report
        st.session_state["source_name"] = SAMPLE_NAME
        st.rerun()

    st.markdown('<div class="side-hr"></div>', unsafe_allow_html=True)
    st.markdown('<div class="side-section">Filters</div>', unsafe_allow_html=True)
    base = st.session_state.get("scored")
    if base is not None:
        rec_sel = st.multiselect("Recommendation", REC_ORDER, default=REC_ORDER, key="f_rec")
        score_lo, score_hi = st.slider("Score range", 0, 100, (0, 100), key="f_score")
        srcs = sorted(str(s) for s in base["source"].dropna().unique() if str(s).strip())
        src_sel = st.multiselect("Source", ["(any)"] + srcs, default=["(any)"], key="f_src")
        emp_max, bud_max = filter_bounds(base)
        emp_lo, emp_hi = st.slider("Company size (employees)", 0, emp_max, (0, emp_max), step=5, key="f_emp")
        inc_emp_unknown = st.checkbox("Include unknown company size", value=True, key="f_inc_emp")
        b_lo, b_hi = st.slider("Monthly budget (USD)", 0, bud_max, (0, bud_max), step=1000, key="f_budget")
        inc_budget_unknown = st.checkbox("Include unknown budget", value=True, key="f_inc_bud")

        filtered = apply_filters(base, rec_sel, score_lo, score_hi, src_sel,
                                 emp_lo, emp_hi, inc_emp_unknown, b_lo, b_hi, inc_budget_unknown)
        st.session_state["filtered"] = filtered
    else:
        st.warning("No data loaded.")

    st.markdown('<div class="side-hr"></div>', unsafe_allow_html=True)
    st.markdown('<div class="side-section">Export</div>', unsafe_allow_html=True)
    if base is not None:
        export_cols = [
            "rank", "lead_id", "name", "email_norm", "company", "title", "source",
            "monthly_budget", "budget_monthly", "employees", "employee_count",
            "score", "recommendation", "key_reason", "disqualifiers",
            "pts_intent", "pts_pain", "pts_budget", "pts_urgency", "pts_authority", "pts_fit",
            "notes", "is_disqualifier_override", "used_early_stage_floor",
        ]
        export_cols = [c for c in export_cols if c in base.columns]

        def to_csv_bytes(df: pd.DataFrame) -> bytes:
            return df[export_cols].to_csv(index=False).encode("utf-8")

        st.download_button(
            "Full scored results (CSV)",
            data=to_csv_bytes(base),
            file_name="triage_all.csv",
            mime="text/csv",
            width="stretch",
        )
        for rec in REC_ORDER:
            sub = base[base["recommendation"] == rec]
            st.download_button(
                f"{rec} ({len(sub)})",
                data=to_csv_bytes(sub),
                file_name=f"triage_{rec.lower().replace(' ', '_')}.csv",
                mime="text/csv",
                width="stretch",
            )


def apply_stashed_filters() -> None:
    """Re-apply the filter widgets' persisted values when the panel is hidden,
    so collapsing the panel never changes the results."""
    base = st.session_state.get("scored")
    if base is None:
        st.session_state["filtered"] = None
        return
    emp_max, bud_max = filter_bounds(base)
    params = (
        st.session_state.get("f_rec", REC_ORDER),
        *st.session_state.get("f_score", (0, 100)),
        st.session_state.get("f_src", ["(any)"]),
        *st.session_state.get("f_emp", (0, emp_max)),
        st.session_state.get("f_inc_emp", True),
        *st.session_state.get("f_budget", (0, bud_max)),
        st.session_state.get("f_inc_bud", True),
    )
    st.session_state["filtered"] = apply_filters(base, *params)


# ---------------------------------------------------------------------------
# Main panel
# ---------------------------------------------------------------------------
def render_main() -> None:
    """Header (with the sidebar toggle), dataset banner, KPIs, ranked leads,
    lead inspection, methodology and cleaning log."""
    header = st.columns([0.07, 0.93], vertical_alignment="center")
    with header[0]:
        if st.button("☰", key="sb_toggle", type="primary", help="Show or hide the filters panel"):
            set_sidebar(not sidebar_is_open())
    with header[1]:
        st.title("Lead Triage")
        st.caption("Automatically clean, score, qualify and prioritise inbound leads — with every score explained.")

    scored = st.session_state.get("scored")
    filtered = st.session_state.get("filtered")

    if scored is None or scored.empty:
        st.error("No valid leads to display. Upload a lead export to get started.")
        st.stop()

    if filtered is None or filtered.empty:
        st.warning("No leads match the current filters — adjust or clear them in the sidebar.")
        st.stop()

    report = st.session_state.get("report", {})

    # -- dataset status ---------------------------------------------------------
    st.markdown(
        dataset_banner(st.session_state.get("source_name", SAMPLE_NAME), report),
        unsafe_allow_html=True,
    )

    # -- summary metrics --------------------------------------------------------
    st.subheader("Pipeline")
    kpis = [
        kpi_card("Total Leads", f"{len(filtered):,}"),
        kpi_card("Contact Now", f"{int((filtered['recommendation'] == CONTACT_NOW).sum()):,}", DOT_CLASS[CONTACT_NOW]),
        kpi_card("Nurture", f"{int((filtered['recommendation'] == NURTURE).sum()):,}", DOT_CLASS[NURTURE]),
        kpi_card("Disqualified", f"{int((filtered['recommendation'] == DISQUALIFY).sum()):,}", DOT_CLASS[DISQUALIFY]),
        kpi_card("Average Score", f"{filtered['score'].mean():.0f}" if len(filtered) else "—"),
    ]
    st.markdown(f'<div class="kpi-row">{"".join(kpis)}</div>', unsafe_allow_html=True)
    if len(filtered) < len(scored):
        st.caption(f"Metrics reflect current filters ({len(scored):,} leads in the full dataset).")

    # -- results table ----------------------------------------------------------
    st.subheader("Ranked leads")
    display_cols = {
        "rank": st.column_config.NumberColumn("Rank", format="%d"),
        "lead_id": st.column_config.TextColumn("Lead ID"),
        "company": st.column_config.TextColumn("Company"),
        "title": st.column_config.TextColumn("Title"),
        "budget_monthly": st.column_config.NumberColumn("Budget", format="$%d"),
        "employee_count": st.column_config.NumberColumn("Staff", format="%.0f"),
        "score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100, format="%d"),
        "recommendation": st.column_config.TextColumn("Recommendation"),
        "key_reason": st.column_config.TextColumn("Key reason"),
    }
    table = filtered[
        ["rank", "lead_id", "company", "title", "budget_monthly", "employee_count",
         "score", "recommendation", "key_reason"]
    ].copy()
    st.dataframe(
        table,
        column_config=display_cols,
        hide_index=True,
        width="stretch",
        height=min(560, 40 + 35 * max(len(table), 1)),
    )

    # -- lead detail ------------------------------------------------------------
    st.subheader("Lead inspection")
    if len(filtered) == 0:
        st.warning("No leads match the current filters.")
        st.stop()

    options = filtered[["lead_id", "company", "name", "score", "recommendation", "rank"]].copy()
    options["label"] = (
        "#" + options["rank"].astype(str) + " · "
        + options["company"].fillna("?").astype(str)
        + " — " + options["name"].fillna("?").astype(str)
        + " (" + options["score"].astype(str) + " pts)"
    )
    labels = options["label"].tolist()
    selected_label = st.selectbox("Select a lead to see its full explanation", labels, index=0, key="lead_select")
    # option order == filtered row order, so positional lookup is exact even when filters leave gaps in the index
    row = filtered.iloc[labels.index(selected_label)]

    st.divider()
    left_html, right_html = lead_detail_html(row)
    left, right = st.columns([1, 1.15], gap="large")
    with left:
        st.markdown(left_html, unsafe_allow_html=True)
    with right:
        st.markdown(right_html, unsafe_allow_html=True)

    # -- methodology ------------------------------------------------------------
    with st.expander("Methodology"):
        st.markdown(
            """
**Deterministic, explainable model — max 100 pts.** All weights live in
`src/scoring.py` → `SCORING_CONFIG` and can be tuned without touching any logic.

| Factor | Max | What it measures |
|---|---|---|
| Intent | 25 | "want it automated", "ready to pilot", "budget approved" → 25 · "exploring / comparing options" → 15 · researching → 8 |
| Pain / Need | 20 | concrete process + "eating our week" → 20 · process named → 12 |
| Budget | 20 | ≥ $15k → 20 · $5–10k → 16 · missing is neutral (6), raised to 16 when notes say "budget approved" |
| Urgency | 15 | ASAP / ≤ 2 weeks → 15 · this month → 12 · ~1 month → 6 |
| Authority | 10 | decision-maker + "decision is mine / I make the call" → 10 · title only → 7 · "need to loop in team" → 4 |
| Fit | 10 | agency / SaaS with 10+ staff → 8–10 · small or early → 4–6 |

**Recommendation rules**

- **Contact Now** — score ≥ 65 **and** a hard buying signal (strong intent, or clear pain + approved budget)
- **Nurture** — score 35–64, plus an early-stage floor (genuine business, no negative signals → kept warm)
- **Disqualify** — score < 35, or **any** disqualifier (job seeker, student / learner, vendor / seller, spam, competitor, VC intro, press, newsletter error, explicit non-buyer) overrides the score.
"""
        )

    # -- cleaning log -----------------------------------------------------------
    with st.expander("Cleaning log"):
        st.markdown(cleaning_log_html(report), unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Layout: the filters panel is a column on the left; the rest is main content.
# ---------------------------------------------------------------------------
# First mobile load: start with the panel closed so the drawer never auto-opens
# on phones (a hidden iframe sets ?sb=closed once and reloads — see above).
if "sb" not in st.query_params:
    st.markdown(MOBILE_INIT_IFRAME, unsafe_allow_html=True)

if sidebar_is_open():
    st.markdown(f"<style>{MOBILE_DRAWER_CSS}</style>", unsafe_allow_html=True)
    left, right = st.columns([0.26, 0.74], gap="large")
    with left:
        render_filters_panel()
    with right:
        render_main()
else:
    apply_stashed_filters()
    render_main()
