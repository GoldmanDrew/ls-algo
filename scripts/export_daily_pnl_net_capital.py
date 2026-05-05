#!/usr/bin/env python3
"""
Export daily strategy PnL (from run_eod_pnl_email ledger history) plus net market
capital for the **same symbol set as PnL attribution** (``pnl_by_symbol.csv`` per run).

Cumulative PnL columns are **rebased to zero on 2026-02-27** (same anchor as
``run_eod_pnl_email.START_DATE``): each value has that day's cumulative subtracted.
``daily_*`` columns are day-over-day **differences of the rebased cumulatives** (so the
first calendar row has blank daily fields; 2026-02-27 shows cumulative 0).

Net market capital: sum(long ``positionValue_base``) − sum(|short ``positionValue_base``|)
only for tickers listed in ``data/runs/<date>/accounting/pnl_by_symbol.csv`` (the file
used with Flex positions to build ``pnl_attribution_history`` / long-short split).

Usage:
  python scripts/export_daily_pnl_net_capital.py
  python scripts/export_daily_pnl_net_capital.py --anchor-date 2026-02-27 --output data/ledger/out.csv
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

# Must match run_eod_pnl_email.START_DATE (PnL history / plots anchor).
DEFAULT_ANCHOR_DATE = "2026-02-27"


def load_attribution_symbols(run_date: str, runs_root: Path) -> set[str]:
    """Symbols included in accounting PnL-by-symbol (same scope as attribution split)."""
    p = runs_root / run_date / "accounting" / "pnl_by_symbol.csv"
    if not p.is_file():
        return set()
    df = pd.read_csv(p)
    if "symbol" not in df.columns:
        return set()
    out: set[str] = set()
    for s in df["symbol"].dropna():
        cs = canonical_symbol(str(s).strip())
        if cs:
            out.add(cs)
    return out


def net_market_capital_usd(pos: pd.DataFrame) -> float | None:
    """Long MV sum minus sum of |short MV| in base currency (already filtered rows)."""
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


def load_positions_for_run(run_date: str, runs_root: Path) -> pd.DataFrame | None:
    xml = runs_root / run_date / "ibkr_flex" / "flex_positions.xml"
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


def net_capital_attribution_scope(run_date: str, runs_root: Path) -> float | None:
    """Net market capital for symbols that appear in pnl_by_symbol for that run."""
    sym_set = load_attribution_symbols(run_date, runs_root)
    pos = load_positions_for_run(run_date, runs_root)
    if pos is None:
        return None
    if not sym_set:
        # No symbol-level PnL file → cannot align to attribution scope.
        return None
    pos = pos.copy()
    pos["symbol"] = pos["symbol"].map(lambda s: canonical_symbol(str(s)))
    pos = pos[pos["symbol"].isin(sym_set)]
    return net_market_capital_usd(pos)


def build_export(
    history_path: Path,
    *,
    runs_root: Path,
    anchor_date: str,
) -> pd.DataFrame:
    if not history_path.is_file():
        raise FileNotFoundError(f"Missing PnL history: {history_path}")

    hist = pd.read_csv(history_path)
    if "date" not in hist.columns or "total_pnl" not in hist.columns:
        raise ValueError(f"{history_path} must contain at least date, total_pnl")

    hist["date"] = pd.to_datetime(hist["date"], errors="coerce")
    hist = hist.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    hist["total_pnl"] = pd.to_numeric(hist["total_pnl"], errors="coerce")

    anchor = pd.to_datetime(anchor_date).normalize()
    anchor_row = hist[hist["date"].dt.normalize() == anchor]
    if anchor_row.empty:
        raise ValueError(
            f"Anchor date {anchor_date!r} not found in {history_path}; "
            "cannot rebase PnL to zero on that day."
        )
    anchor_idx = int(anchor_row.index[0])
    base_total = float(hist.loc[anchor_idx, "total_pnl"] or 0.0)

    out = pd.DataFrame({"date": hist["date"]})
    out["cumulative_total_pnl"] = hist["total_pnl"] - base_total

    bucket_bases: dict[str, float] = {}
    for c in BUCKET_COLS:
        if c not in hist.columns:
            continue
        hist[c] = pd.to_numeric(hist[c], errors="coerce")
        bucket_bases[c] = float(hist.loc[anchor_idx, c] or 0.0)
        out[f"cumulative_{c}"] = hist[c] - bucket_bases[c]

    out["daily_total_pnl"] = out["cumulative_total_pnl"].diff()
    for c in BUCKET_COLS:
        col = f"cumulative_{c}"
        if col in out.columns:
            out[f"daily_{c}"] = out[col].diff()

    capitals: list[float | None] = []
    for d in out["date"]:
        ds = d.strftime("%Y-%m-%d")
        capitals.append(net_capital_attribution_scope(ds, runs_root))

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
        "--anchor-date",
        default=DEFAULT_ANCHOR_DATE,
        help="rebase cumulatives to zero on this date (default: 2026-02-27)",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=LEDGER_DIR / "daily_pnl_and_net_capital.csv",
        help="output CSV path",
    )
    args = ap.parse_args()

    df = build_export(args.history, runs_root=args.runs_root, anchor_date=args.anchor_date)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Wrote {len(df)} rows to {args.output} (anchor={args.anchor_date!r})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
