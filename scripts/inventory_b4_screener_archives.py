#!/usr/bin/env python3
"""Inventory point-in-time screener / plan archives that bound B4 production windows.

Writes a JSON summary of earliest archived ``etf_screened_today.csv`` and
``proposed_trades.csv`` dates, calendar gaps, and the implied archive floor
for production replay (currently documented as 2026-02-27 when archives begin).

Usage:
  python scripts/inventory_b4_screener_archives.py
  python scripts/inventory_b4_screener_archives.py --out path/to/report.json
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date, timedelta
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import pandas as pd  # noqa: E402

from scripts.production_actual_backtest import (  # noqa: E402
    archive_coverage_summary,
    list_archived_plan_dates,
    list_archived_screened_dates,
)


def _gap_dates(dates: list[pd.Timestamp]) -> list[str]:
    if len(dates) < 2:
        return []
    gaps: list[str] = []
    for prev, cur in zip(dates, dates[1:]):
        delta = int((cur - prev).days)
        if delta > 5:  # allow weekends/holidays; flag larger holes
            gaps.append(f"{prev.date()}->{cur.date()} ({delta}d)")
    return gaps


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", default="2025-05-01", help="Coverage summary start date")
    ap.add_argument(
        "--out",
        type=Path,
        default=REPO / "risk_dashboard" / "data" / "b4_archive_inventory.json",
    )
    args = ap.parse_args(argv)

    screened = list_archived_screened_dates()
    plans = list_archived_plan_dates()
    coverage = archive_coverage_summary(args.start)
    floor = str(screened[0].date()) if screened else None
    golden_doc_floor = "2026-02-27"
    report = {
        "generated_at": date.today().isoformat(),
        "documented_golden_window_start": golden_doc_floor,
        "archive_floor_screened": floor,
        "archive_floor_plans": str(plans[0].date()) if plans else None,
        "screened": {
            "n_dates": len(screened),
            "first": floor,
            "last": str(screened[-1].date()) if screened else None,
            "large_gaps": _gap_dates(screened)[:40],
        },
        "plans": {
            "n_dates": len(plans),
            "first": str(plans[0].date()) if plans else None,
            "last": str(plans[-1].date()) if plans else None,
            "large_gaps": _gap_dates(plans)[:40],
        },
        "coverage_vs_start": coverage.to_dict(orient="records"),
        "ops_next_steps": [
            "Backfill missing daily etf_screened_today.csv archives before the floor where recoverable from git history.",
            "Hash every new archive day into the B4 export manifest input_hashes.",
            "Re-run export_b4_dashboard.py --run-production --start <new_floor> and import fail-closed in etf-dashboard.",
            "Keep inception_research for listing → first honest plan_entry_date gap.",
            "Do not forward-fill latest screener into missing archive days.",
        ],
        "note": (
            "Production plan_entry_date cannot precede the earliest archived screener/plan "
            "day used by generate_trade_plan replay. Research inception paths cover pre-floor history."
        ),
    }
    if floor and floor > golden_doc_floor:
        report["warning"] = (
            f"Live archive floor {floor} is later than documented golden start {golden_doc_floor}."
        )
    elif floor and floor < golden_doc_floor:
        report["opportunity"] = (
            f"Archives begin {floor}, earlier than documented golden start {golden_doc_floor}; "
            "consider extending the production replay window after validation."
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "ok": True,
        "out": str(args.out.resolve()),
        "archive_floor_screened": floor,
        "n_screened": len(screened),
        "n_plans": len(plans),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
