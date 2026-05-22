#!/usr/bin/env python3
"""Audit plan B4 structural targets vs FIFO ledger / IBKR spot (Phase 0)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ibkr_accounting import build_b4_plan_ledger_reconciliation  # noqa: E402


def audit_run(run_date: str, *, fail_on_drift: bool) -> int:
    accounting = PROJECT_ROOT / "data" / "runs" / run_date / "accounting"
    recon_path = accounting / "b4_plan_ledger_reconciliation.csv"
    if not recon_path.exists():
        print(f"FAIL: missing {recon_path.relative_to(PROJECT_ROOT)}")
        print("Run ibkr_accounting_pnl for this date first.")
        return 1

    df = pd.read_csv(recon_path)
    if df.empty:
        print(f"OK: {run_date} — empty B4 reconciliation (no registry names)")
        return 0

    missing = df[df["ledger_missing_b4"] == True]  # noqa: E712
    plan_active = df[df["plan_exposure_active"] == True]  # noqa: E712
    print(f"{run_date}: {len(df)} B4 names, {len(missing)} ledger_missing_b4, "
          f"{len(plan_active)} plan_exposure_active")

    if not missing.empty:
        cols = ["underlying", "plan_b4_usd", "ledger_qty_b4", "ibkr_qty", "orphan_frac"]
        print("\nPlan B4 but ledger qty_b4 ~= 0:")
        print(missing[cols].head(20).to_string(index=False))

    if fail_on_drift and not missing.empty:
        n_unfixed = missing[~missing["plan_exposure_active"]].shape[0]
        if n_unfixed:
            print(f"\nFAIL: {n_unfixed} name(s) missing plan exposure override")
            return 1

    print(f"\nOK: audit complete ({recon_path.relative_to(PROJECT_ROOT)})")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_date", nargs="?", default="2026-05-21")
    ap.add_argument(
        "--fail-on-drift",
        action="store_true",
        help="Exit 1 when plan B4 exists but plan_exposure_active is false",
    )
    args = ap.parse_args()
    return audit_run(args.run_date, fail_on_drift=args.fail_on_drift)


if __name__ == "__main__":
    sys.exit(main())
