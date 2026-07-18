#!/usr/bin/env python3
"""Backfill the hedged-vs-unhedged PnL ledger from existing accounting runs.

Replays every ``data/runs/<date>/accounting`` folder in chronological order
through :func:`hedged_pnl.compute_hedged_split`, seeding
``data/ledger/hedged_pnl_history.csv`` and the per-run split artifacts.

Idempotent: by default the ledger is rebuilt from scratch (--keep-ledger to
append/refresh only missing dates). The earliest date defaults to the PnL
history start (2026-02-27) so the lens lines up with pnl_history.csv.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from hedged_pnl import (  # noqa: E402
    HEDGED_PNL_HISTORY_CSV,
    RUNS_ROOT,
    compute_hedged_split,
    list_accounting_run_dates,
)

DEFAULT_START = "2026-02-27"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs-root", type=Path, default=RUNS_ROOT)
    ap.add_argument("--ledger", type=Path, default=HEDGED_PNL_HISTORY_CSV)
    ap.add_argument("--start", default=DEFAULT_START, help="Earliest run date to include")
    ap.add_argument("--end", default=None, help="Latest run date to include (default: all)")
    ap.add_argument(
        "--keep-ledger",
        action="store_true",
        help="Do not delete the existing ledger before replaying (rows are still recomputed per date)",
    )
    args = ap.parse_args()

    dates = [d for d in list_accounting_run_dates(args.runs_root) if d >= args.start]
    if args.end:
        dates = [d for d in dates if d <= args.end]
    if not dates:
        print(f"[backfill-hedged-pnl] no accounting runs found under {args.runs_root}")
        return 1

    if not args.keep_ledger and args.ledger.is_file():
        args.ledger.unlink()
        print(f"[backfill-hedged-pnl] removed existing ledger {args.ledger}")

    # Chain start: continue from the last ledger row before the window (if any),
    # else from the last accounting run before the window. Only a truly empty
    # history seeds with full-YTD values.
    prev: str | None = None
    if args.ledger.is_file():
        try:
            import pandas as pd

            led = pd.read_csv(args.ledger)
            earlier = [d for d in led["date"].astype(str) if d < dates[0]]
            if earlier:
                prev = max(earlier)
        except Exception:
            prev = None
    if prev is None:
        candidates = [d for d in list_accounting_run_dates(args.runs_root) if d < dates[0]]
        if candidates and args.keep_ledger:
            prev = max(candidates)
    for ds in dates:
        try:
            res = compute_hedged_split(
                ds,
                runs_root=args.runs_root,
                ledger_path=args.ledger,
                prev_run_date=prev,
            )
        except Exception as exc:  # keep replaying; a single bad run should not stop the seed
            print(f"[backfill-hedged-pnl] {ds}: FAILED ({exc})")
            continue
        print(
            f"[backfill-hedged-pnl] {ds}: hedged_ytd={res['hedged_pnl_ytd']:,.2f} "
            f"unhedged_ytd={res['unhedged_pnl_ytd']:,.2f} "
            f"(daily {res['hedged_daily']:+,.2f} / {res['unhedged_daily']:+,.2f})"
            + (" [seed]" if res["seeded"] else "")
        )
        prev = ds

    print(f"[backfill-hedged-pnl] done -> {args.ledger}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
