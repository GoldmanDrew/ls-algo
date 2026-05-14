"""
Build wide daily **adjusted close** history for strategy underlyings (Yahoo / yfinance).

Used by ``daily_screener`` and ``scripts/build_underlying_returns`` so
``generate_trade_plan.apply_covariance_balance`` can load ``paths.underlying_returns_csv``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


def yahoo_ticker(sym: str) -> str:
    """Map screened symbol to yfinance ticker (minimal BRK-style handling)."""
    s = str(sym).strip().upper().replace(".", "-")
    if s == "BRK-B":
        return "BRK-B"
    return s


def build_wide_adj_close(symbols: Iterable[str], *, period: str = "5y") -> tuple[pd.DataFrame, list[str]]:
    """
    Download adjusted closes for unique symbols; return wide frame (index=dates) and
    the list of columns successfully fetched. Raises RuntimeError if fewer than 2 series.
    """
    try:
        import yfinance as yf
    except ImportError as e:
        raise RuntimeError("yfinance is required (pip install yfinance)") from e

    seen: set[str] = set()
    ordered: list[str] = []
    for raw in symbols:
        s = yahoo_ticker(raw)
        if not s or s in seen:
            continue
        seen.add(s)
        ordered.append(s)

    if len(ordered) < 2:
        raise RuntimeError("need at least 2 distinct underlying symbols")

    series: list[pd.Series] = []
    ok_cols: list[str] = []
    for sym in ordered:
        t = yahoo_ticker(sym)
        try:
            ydf = yf.Ticker(t).history(period=period, auto_adjust=True)
        except Exception as ex:
            print(f"[WARN] underlying_returns {t}: download failed ({ex})")
            continue
        if ydf.empty or "Close" not in ydf.columns:
            print(f"[WARN] underlying_returns {t}: no data")
            continue
        s = ydf["Close"].copy()
        s.name = sym
        series.append(s)
        ok_cols.append(sym)

    if len(series) < 2:
        raise RuntimeError(
            f"fewer than 2 symbols returned data (got {len(series)}); check network / tickers"
        )

    wide = pd.concat(series, axis=1)
    wide = wide.sort_index()
    wide.index = pd.to_datetime(wide.index).tz_localize(None)
    wide = wide.sort_index()
    return wide, ok_cols


def write_underlying_returns_csv(
    wide: pd.DataFrame, out_path: Path
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    wide.to_csv(out_path)
