#!/usr/bin/env python3
"""Command-line runner for the lead-triage pipeline.

Usage:
    python run_triage.py [path/to/leads.xlsx|.csv] [-o out.csv] [--top N]

With no path it looks for the bundled sample in ``data/sample/``.
Writes a fully-scored CSV to ``data/output/triage_results.csv`` by default.
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import Counter

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.cleaning import clean_dataframe
from src.scoring import score_dataframe


DEFAULT_INPUT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data", "sample", "sample_leads.xlsx"
)
DEFAULT_OUTPUT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data", "output", "triage_results.csv"
)


def load_input(path: str) -> pd.DataFrame:
    if path.lower().endswith(".csv"):
        return pd.read_csv(path)
    return pd.read_excel(path, sheet_name=0)


def main() -> int:
    parser = argparse.ArgumentParser(description="Lead triage pipeline")
    parser.add_argument("input", nargs="?", default=DEFAULT_INPUT, help="xlsx/csv export")
    parser.add_argument("-o", "--output", default=DEFAULT_OUTPUT, help="output csv path")
    parser.add_argument("--top", type=int, default=10, help="rows to preview per bucket")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"ERROR: input file not found: {args.input}", file=sys.stderr)
        return 2

    df = load_input(args.input)
    print(f"Loaded {len(df):,} rows x {df.shape[1]} cols from {os.path.basename(args.input)}")

    # ---- clean ----
    cleaned, report = clean_dataframe(df)
    print("\n=== CLEANING REPORT ===")
    for k, v in report.items():
        if k in ("junk_ids", "duplicate_ids"):
            print(f"  {k:<22} {v}")
        else:
            print(f"  {k:<22} {v}")

    valid = cleaned[cleaned["valid_lead"]].copy()
    if valid.empty:
        print("ERROR: no valid leads after cleaning.", file=sys.stderr)
        return 1

    # ---- score ----
    scored = score_dataframe(valid)
    scored = scored.sort_values(["score", "lead_id"], ascending=[False, True]).reset_index(drop=True)
    scored.insert(0, "rank", range(1, len(scored) + 1))

    # ---- distribution ----
    dist = Counter(scored["recommendation"])
    print("\n=== RESULTS ===")
    print(f"  Total processed        : {len(scored)}")
    print(f"  Contact Now            : {dist.get('Contact Now', 0)}")
    print(f"  Nurture                : {dist.get('Nurture', 0)}")
    print(f"  Disqualify             : {dist.get('Disqualify', 0)}")
    print(f"  Average score          : {scored['score'].mean():.1f}")
    print(f"  Disqualifier overrides : {int(scored['is_disqualifier_override'].sum())}")
    print(f"  Early-stage floor used : {int(scored['used_early_stage_floor'].sum())}")

    # ---- previews ----
    def preview(mask, label):
        sub = scored[mask].head(args.top)
        print(f"\n--- {label} (showing {len(sub)}) ---")
        for _, r in sub.iterrows():
            print(
                f"  #{int(r['rank']):>3} [{r['lead_id']}] {str(r['company'])[:20]:<20} "
                f"| {str(r['title'])[:18]:<18} | ${r['budget_monthly'] if pd.notna(r['budget_monthly']) else '?'} "
                f"| {int(r['score']):>3} | {r['key_reason'][:70]}"
            )

    preview(scored["recommendation"] == "Contact Now", "CONTACT NOW")
    preview(scored["recommendation"] == "Nurture", "NURTURE")
    preview(scored["recommendation"] == "Disqualify", "DISQUALIFY")

    # ---- export ----
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    export_cols = [
        "rank", "lead_id", "name", "email_norm", "company", "title", "source",
        "monthly_budget", "budget_monthly", "employees", "employee_count",
        "score", "recommendation", "key_reason", "disqualifiers",
        "pts_intent", "pts_pain", "pts_budget", "pts_urgency", "pts_authority", "pts_fit",
        "notes", "is_disqualifier_override", "used_early_stage_floor",
    ]
    export_cols = [c for c in export_cols if c in scored.columns]
    scored[export_cols].to_csv(args.output, index=False)
    print(f"\nWrote scored results to {args.output}")

    # ---- errors guard ----
    if any("unparseable" in str(r["budget_status"]) for _, r in cleaned.iterrows()):
        print("note: some budget values were unparseable (recorded, treated neutrally)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
