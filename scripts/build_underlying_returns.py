#!/usr/bin/env python3
"""
Populate ``paths.underlying_returns_csv`` (wide daily **adjusted** close prices) so
``generate_trade_plan.apply_covariance_balance`` can run.

Reads unique ``Underlying`` symbols from the screened universe CSV in config, downloads
history from Yahoo via yfinance, writes one column per ticker.

Usage (from repo root):
    python scripts/build_underlying_returns.py
    python scripts/build_underlying_returns.py --period 3y
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from strategy_config import load_config  # noqa: E402
from underlying_returns_builder import (  # noqa: E402
    build_wide_adj_close,
    write_underlying_returns_csv,
    yahoo_ticker,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--period",
        default="5y",
        help="yfinance period string (default 5y)",
    )
    ap.add_argument(
        "--config",
        default=str(_REPO / "config" / "strategy_config.yml"),
        help="Path to strategy_config.yml",
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    paths = cfg.get("paths") or {}
    screened = Path(paths.get("screened_csv") or "")
    if not screened.is_file():
        raise SystemExit(f"screened CSV not found: {screened}")

    out_path = Path(paths.get("underlying_returns_csv") or "").expanduser()
    if not out_path:
        raise SystemExit("paths.underlying_returns_csv not set in config")
    if not out_path.is_absolute():
        out_path = (_REPO / out_path).resolve()

    df = pd.read_csv(screened, usecols=["Underlying"], dtype=str)
    syms = sorted(
        {yahoo_ticker(x) for x in df["Underlying"].dropna().unique() if str(x).strip()}
    )
    print(f"[INFO] {len(syms)} symbols -> {out_path} (period={args.period})")

    wide, ok_cols = build_wide_adj_close(syms, period=args.period)
    write_underlying_returns_csv(wide, out_path)
    head = ", ".join(ok_cols[:5])
    more = " ..." if len(ok_cols) > 5 else ""
    print(f"[OK] wrote {wide.shape[0]} rows x {wide.shape[1]} columns ({head}{more})")


if __name__ == "__main__":
    main()
