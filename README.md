# Lead Triage

Automatically **clean, score, qualify and prioritise inbound leads** for a
marketing agency. Upload any lead export (`.xlsx` / `.csv`), and the app
returns every lead ranked with an **explainable score** and one of three
recommendations — **Contact Now**, **Nurture** or **Disqualify** — with the
reasoning behind each decision shown.

Built for the Koya AI Automation Developer Assessment (Task 1), but designed
to be **reusable against future lead exports with zero code changes**.

---

## What it does

| Capability | How |
|---|---|
| Data cleaning | Normalises column names, emails (`[at]`→`@`), dates (day-first), budgets (`$6-8k`→7,000), employee counts (`35-55`→45); flags junk/QA rows and duplicates **with reasons** — never silently deletes |
| Notes analysis | Deterministic phrase/signal engine extracts intent, pain, urgency, authority, budget signals and disqualifiers — reproducible, free, no API keys |
| Scoring | 6-factor, weighted, transparent model (max 100) — every point explained |
| Recommendation | Contact Now / Nurture / Disqualify, with disqualifier overrides and an early-stage floor |
| UI | Streamlit app: upload → metrics → sortable table → filters → per-lead explainable drill-down → CSV exports |

## Results on the supplied 520-row dataset

Produced by running the pipeline (not hardcoded):

```
Raw rows            : 520
Excluded            : 20   (1 blank + 7 junk/QA + 12 duplicates, all logged)
Valid leads scored  : 500
Contact Now         : 109
Nurture             : 225
Disqualify          : 166
Average score       : 47.9
```

## Run locally

Requires **Python 3.10+**.

```bash
cd lead-triage
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

streamlit run app.py             # opens the web app
```

Run the pipeline headlessly (prints the distribution + writes a scored CSV):

```bash
python run_triage.py                          # uses the bundled sample
python run_triage.py path/to/leads.xlsx -o out.csv
```

Run the tests:

```bash
pip install pytest
python -m pytest tests/ -q
```

## Architecture

```
app.py               Streamlit UI (upload, filters, metrics, drill-down, exports)
run_triage.py        CLI runner (headless processing + CSV export)
src/
  cleaning.py        Reusable cleaning pipeline: normalise → parse → flag (report-first)
  notes.py           Deterministic signal extraction from unstructured notes
  scoring.py         SCORING_CONFIG + 6-factor explainable scoring + recommendations
  models.py          Data models (SignalSet, ScoringResult)
data/
  sample/            Bundled sample export (same as the assessment workbook)
  output/            Generated scored CSVs (git-ignored)
tests/
  test_cleaning.py   20+ data-quality tests (malformed files, missing values, duplicates…)
  test_scoring.py    10 required scenarios + determinism + real-data regression
```

## How scoring works (high level)

| Factor | Max | What it measures |
|---|---|---|
| Intent | 25 | "want it automated", "ready to pilot", "budget approved" → 25 · "exploring / comparing options" → 15 · researching → 8 |
| Pain / Need | 20 | concrete process + "eating our week" → 20 · process named → 12 |
| Budget | 20 | ≥$15k→20, $5–10k→16, $2.5–5k→12 … unknown is **neutral (6)** — raised to 16 if notes say "budget approved"; missing budget never auto-disqualifies |
| Urgency | 15 | ASAP / ≤2 weeks → 15 · this month → 12 · ~1 month → 6 |
| Authority | 10 | decision-maker + "decision is mine / I make the call" → 10 · title only → 7 · "need to loop in team" → 4 |
| Fit | 10 | agency/SaaS with 10+ staff → 8–10 · small / early-stage → 4–6 |

**Recommendations**
- `Contact Now ≥ 65` **and** a hard buying signal (strong intent, or clear pain + approved budget)
- `Nurture 35–64` — plus an **early-stage floor**: genuine businesses with no negative signals (e.g. "very early startup, no budget yet") are kept warm instead of discarded
- `Disqualify < 35` — or **any disqualifier** overrides the score: job seeker, student/learner, vendor/seller, spam, competitor, VC intro (not a direct buyer), press, newsletter signup error, explicit non-buyer

All weights and thresholds live in `SCORING_CONFIG` in `src/scoring.py` — tune the model without touching any logic.

## Deploy on Streamlit Community Cloud (free)

1. Push this folder to a GitHub repository (make sure `requirements.txt` is at the repo root; no secrets or local paths are used).
2. Go to **https://streamlit.io/cloud** → *Create app* → connect GitHub.
3. Set: Repository = yours, Branch = `main`, **Main file path = `app.py`**.
4. Click **Deploy**. The app auto-starts with the bundled sample; reviewers can upload their own exports in the sidebar.

Notes:
- No paid services or API keys are used anywhere in the core pipeline.
- The app is fully self-contained — `requirements.txt` installs `pandas`, `openpyxl` and `streamlit`.
