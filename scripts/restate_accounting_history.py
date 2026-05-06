#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = PROJECT_ROOT / "data" / "runs"
ACCOUNTING_STATE = PROJECT_ROOT / "data" / "accounting" / "underlying_bucket_state.csv"
PNL_HISTORY_CSV = PROJECT_ROOT / "data" / "ledger" / "pnl_history.csv"
START_DATE = "2026-02-27"
REQUIRED_FLEX_FILES = ("flex_trades.xml", "flex_cash.xml", "flex_positions.xml")


def discover_run_dates() -> list[str]:
    dates: list[str] = []
    for child in RUNS_ROOT.iterdir() if RUNS_ROOT.exists() else []:
        if not child.is_dir():
            continue
        try:
            datetime.strptime(child.name, "%Y-%m-%d")
        except ValueError:
            continue
        flex_dir = child / "ibkr_flex"
        if all((flex_dir / name).exists() for name in REQUIRED_FLEX_FILES):
            dates.append(child.name)
    return sorted(dates)


def backup_and_reset_state() -> Path | None:
    ACCOUNTING_STATE.parent.mkdir(parents=True, exist_ok=True)
    if not ACCOUNTING_STATE.exists():
        return None
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = ACCOUNTING_STATE.with_name(f"{ACCOUNTING_STATE.stem}.backup_{stamp}{ACCOUNTING_STATE.suffix}")
    shutil.copy2(ACCOUNTING_STATE, backup)
    ACCOUNTING_STATE.unlink()
    return backup


def run_accounting(run_date: str) -> None:
    cmd = [sys.executable, str(PROJECT_ROOT / "ibkr_accounting.py"), run_date]
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def _skip_pnl_history_run_date(run_date: str) -> bool:
    """Saturday Flex snapshots are omitted from pnl_history (use next weekday row)."""
    return datetime.strptime(run_date, "%Y-%m-%d").weekday() == 5


def rebuild_pnl_history(run_dates: list[str]) -> None:
    rows: list[dict] = []
    for run_date in run_dates:
        if run_date < START_DATE:
            continue
        if _skip_pnl_history_run_date(run_date):
            continue
        totals_path = RUNS_ROOT / run_date / "accounting" / "totals.json"
        if not totals_path.exists():
            continue
        obj = json.loads(totals_path.read_text(encoding="utf-8"))
        bp = obj.get("bucket_pnl") or {}
        b1 = float(bp.get("bucket_1", 0.0))
        b2 = float(bp.get("bucket_2", 0.0))
        b3 = float(bp.get("bucket_3", 0.0))
        b4 = float(bp.get("bucket_4", 0.0))
        rows.append(
            {
                "date": run_date,
                "pnl_bucket_1": b1,
                "pnl_bucket_2": b2,
                "pnl_bucket_3": b3,
                "pnl_bucket_4": b4,
                "total_pnl": b1 + b2 + b3 + b4,
            }
        )

    if not rows:
        return
    PNL_HISTORY_CSV.parent.mkdir(parents=True, exist_ok=True)
    hist = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    hist.to_csv(PNL_HISTORY_CSV, index=False)


def main() -> int:
    parser = argparse.ArgumentParser(description="Restate historical IBKR accounting outputs from raw Flex files.")
    parser.add_argument("--from-date", default=None, help="First run date to restate, inclusive.")
    parser.add_argument("--to-date", default=None, help="Last run date to restate, inclusive.")
    parser.add_argument("--resume", action="store_true", help="Continue from the existing bucket state instead of resetting it.")
    parser.add_argument("--history-only", action="store_true", help="Only rebuild pnl_history.csv from existing totals.json files.")
    parser.add_argument("--dry-run", action="store_true", help="Print run dates without changing outputs.")
    args = parser.parse_args()

    run_dates = discover_run_dates()
    if args.from_date:
        run_dates = [d for d in run_dates if d >= args.from_date]
    if args.to_date:
        run_dates = [d for d in run_dates if d <= args.to_date]

    if not run_dates:
        print("[RESTATE] No run dates with required Flex files found.")
        return 0

    print(f"[RESTATE] {len(run_dates)} run(s): {run_dates[0]} -> {run_dates[-1]}")
    if args.history_only:
        rebuild_pnl_history(discover_run_dates())
        print(f"[RESTATE] wrote {PNL_HISTORY_CSV}")
        return 0
    if args.dry_run:
        for d in run_dates:
            print(f"[RESTATE] would run {d}")
        return 0

    if args.resume:
        print("[RESTATE] resume mode: keeping existing bucket state")
    else:
        backup = backup_and_reset_state()
        if backup is not None:
            print(f"[RESTATE] backed up bucket state to {backup}")
        else:
            print("[RESTATE] no prior bucket state found")

    for i, run_date in enumerate(run_dates, start=1):
        print(f"[RESTATE] ({i}/{len(run_dates)}) accounting {run_date}")
        run_accounting(run_date)

    rebuild_pnl_history(discover_run_dates())
    print(f"[RESTATE] wrote {PNL_HISTORY_CSV}")
    print("[RESTATE] complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
