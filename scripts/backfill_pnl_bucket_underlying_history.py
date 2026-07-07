#!/usr/bin/env python3
"""Rebuild data/ledger/pnl_bucket_underlying_history.csv from data/runs/."""

from __future__ import annotations

import argparse
from pathlib import Path

from risk_dashboard.pnl_ledger import rebuild_bucket_underlying_history_from_runs

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "data" / "ledger" / "pnl_bucket_underlying_history.csv"
DEFAULT_RUNS = ROOT / "data" / "runs"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--start-date", default="2026-02-27")
    args = parser.parse_args()
    hist = rebuild_bucket_underlying_history_from_runs(
        args.runs_root, args.out, start_date=args.start_date
    )
    print(f"Wrote {len(hist)} rows to {args.out}")


if __name__ == "__main__":
    main()
