#!/usr/bin/env python3
"""
Export daily strategy PnL (from run_eod_pnl_email ledger history) plus net market
capital from each day's IBKR Flex positions snapshot.

Daily PnL: day-over-day change in cumulative ``total_pnl`` from ``data/ledger/pnl_history.csv``
(the same series ``run_eod_pnl_email.update_pnl_history`` maintains).

Net market capital (per user spec): sum of long ``positionValue_base`` minus the sum of
absolute short ``positionValue_base`` (base currency), from
``data/runs/<date>/ibkr_flex/flex_positions.xml``, excluding ``EXCLUDE_SYMBOLS`` from
``ibkr_accounting`` (same as accounting noise filter).

Usage:
  python scripts/export_daily_pnl_net_capital.py
  python scripts/export_daily_pnl_net_capital.py --output data/ledger/my_export.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from ibkr_accounting import EXCLUDE_SYMBOLS, canonical_symbol, parse_open_positions

LEDGER_DIR = _PROJECT_ROOT / "data" / "ledger"
RUNS_ROOT = _PROJECT_ROOT / "data" / "runs"
PNL_HISTORY_CSV = LEDGER_DIR / "pnl_history.csv"

BUCKET_COLS = ("pnl_bucket_1", "pnl_bucket_2", "pnl_bucket_3", "pnl_bucket_4")


def net_market_capital_usd(pos: pd.DataFrame) -> float | None:
    """Long MV sum minus sum of |short MV| in base currency."""
    if pos is None or pos.empty or "position" not in pos.columns:
        return None
    p = pos.copy()
    p["position"] = pd.to_numeric(p["position"], errors="coerce").fillna(0.0)
    if "positionValue_base" not in p.columns:
        return None
    p["positionValue_base"] = pd.to_numeric(p["positionValue_base"], errors="coerce").fillna(0.0)
    longs = p[p["position"] > 0.0]
    shorts = p[p["position"] < 0.0]
    long_sum = float(longs["positionValue_base"].sum())
    short_abs = float(shorts["positionValue_base"].abs().sum())
    return long_sum - short_abs


def load_positions_for_run(run_date: str) -> pd.DataFrame | None:
    xml = RUNS_ROOT / run_date / "ibkr_flex" / "flex_positions.xml"
    if not xml.is_file():
        return None
    try:
        pos = parse_open_positions(xml)
    except (OSError, ValueError):
        return None
    if pos.empty or "symbol" not in pos.columns:
        return pos
    sym = pos["symbol"].map(lambda s: canonical_symbol(str(s)))
    mask = ~sym.isin(EXCLUDE_SYMBOLS)
    return pos.loc[mask].copy()


def build_export(
    history_path: Path,
    *,
    runs_root: Path,
) -> pd.DataFrame:
    if not history_path.is_file():
        raise FileNotFoundError(f"Missing PnL history: {history_path}")

    hist = pd.read_csv(history_path)
    if "date" not in hist.columns or "total_pnl" not in hist.columns:
        raise ValueError(f"{history_path} must contain at least date, total_pnl")

    hist["date"] = pd.to_datetime(hist["date"], errors="coerce")
    hist = hist.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    hist["total_pnl"] = pd.to_numeric(hist["total_pnl"], errors="coerce")

    out = pd.DataFrame({"date": hist["date"]})
    out["cumulative_total_pnl"] = hist["total_pnl"]
    out["daily_total_pnl"] = hist["total_pnl"].diff()

    for c in BUCKET_COLS:
        if c in hist.columns:
            hist[c] = pd.to_numeric(hist[c], errors="coerce")
            out[f"daily_{c}"] = hist[c].diff()

    capitals: list[float | None] = []
    for d in out["date"]:
        ds = d.strftime("%Y-%m-%d")
        pos = load_positions_for_run(ds)
        capitals.append(net_market_capital_usd(pos) if pos is not None else None)

    out["net_market_capital_usd"] = capitals
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--history",
        type=Path,
        default=PNL_HISTORY_CSV,
        help="pnl_history.csv path (default: data/ledger/pnl_history.csv)",
    )
    ap.add_argument(
        "--runs-root",
        type=Path,
        default=RUNS_ROOT,
        help="data/runs root (default: project data/runs)",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=LEDGER_DIR / "daily_pnl_and_net_capital.csv",
        help="output CSV path",
    )
    args = ap.parse_args()

    df = build_export(args.history, runs_root=args.runs_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Wrote {len(df)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
