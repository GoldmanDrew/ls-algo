"""
Notebook-aligned price / borrow / beta helpers for Bucket 4 research.

Vendored from ``notebooks/Bucket_4_Backtest.ipynb`` cells 2–3 so
``bucket4_weekly_opt2`` can run without executing the notebook.
"""

from __future__ import annotations

import io
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:  # pragma: no cover
    yf = None  # type: ignore

# Default relative to cwd (notebook convention); override via configure_price_cache_dirs().
_PRICE_CACHE_DIRS: list[Path] = [
    Path("data/notebook_price_cache"),
    Path("data/price_cache_v6"),
]


def configure_price_cache_dirs(dirs: Sequence[str | Path] | None) -> None:
    global _PRICE_CACHE_DIRS
    if dirs is None:
        return
    _PRICE_CACHE_DIRS = [Path(d) for d in dirs]


def _norm_sym(x: str) -> str:
    return str(x).strip().upper()


def _extract_close_series(raw: pd.DataFrame, ticker: str) -> pd.Series:
    if raw is None or len(raw) == 0:
        return pd.Series(dtype=float, name=ticker)
    close = None
    if isinstance(raw.columns, pd.MultiIndex):
        lvl0 = [str(x).lower() for x in raw.columns.get_level_values(0)]
        lvl1 = [str(x).lower() for x in raw.columns.get_level_values(1)]
        if "close" in lvl0:
            close = raw.xs("Close", axis=1, level=0)
        elif "close" in lvl1:
            close = raw.xs("Close", axis=1, level=1)
    else:
        if "Close" in raw.columns:
            close = raw["Close"]
    if close is None:
        raise RuntimeError(f"Missing Close column for {ticker}")
    if isinstance(close, pd.DataFrame):
        if close.shape[1] == 0:
            raise RuntimeError(f"Empty Close data for {ticker}")
        if ticker in close.columns:
            s = close[ticker]
        elif str(ticker).upper() in [str(c).upper() for c in close.columns]:
            pick = next(c for c in close.columns if str(c).upper() == str(ticker).upper())
            s = close[pick]
        else:
            s = close.iloc[:, 0]
    else:
        s = close
    return pd.to_numeric(s, errors="coerce").rename(ticker)


def _read_close_from_csv_cache(ticker: str, start: str, end: str | None) -> pd.Series | None:
    for base in _PRICE_CACHE_DIRS:
        p = base / f"{ticker}.csv"
        if not p.is_file():
            continue
        df = pd.read_csv(p, index_col=0, parse_dates=True)
        cols = {str(c).lower(): c for c in df.columns}
        col = cols.get("close", df.columns[0] if len(df.columns) == 1 else None)
        if col is None:
            continue
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        s.index = pd.to_datetime(s.index).tz_localize(None)
        s = s.sort_index().loc[s.index >= pd.Timestamp(start)]
        if end:
            s = s.loc[: pd.Timestamp(end)]
        if len(s) > 0:
            return s.rename(ticker)
    return None


def _download_close_series(ticker: str, start: str, end: str | None, *, retries: int = 3) -> pd.Series:
    if yf is None:
        raise RuntimeError("yfinance is not installed")
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            tkr = yf.Ticker(ticker)
            df = tkr.history(start=start, end=end, auto_adjust=True, repair=True)
            if df is not None and len(df) > 0 and "Close" in df.columns:
                s = df["Close"].dropna()
                s.index = pd.to_datetime(s.index).tz_localize(None)
                return s.rename(ticker)
        except Exception as e:
            last_err = e
        try:
            kwargs = dict(start=start, end=end, auto_adjust=True, progress=False)
            try:
                raw = yf.download(ticker, **kwargs, timeout=90)
            except TypeError:
                raw = yf.download(ticker, **kwargs)
            s = _extract_close_series(raw, ticker)
            if len(s.dropna()) > 0:
                return s
        except Exception as e:
            last_err = e
        time.sleep(0.5 * (attempt + 1))
    hint = ", ".join(str(d) for d in _PRICE_CACHE_DIRS)
    err = RuntimeError(
        f"No Yahoo/cache price data for {ticker} (last error: {last_err}). "
        f"Add {ticker}.csv under one of: {hint}"
    )
    if last_err:
        raise err from last_err
    raise err


def _load_leg_close(ticker: str, start: str, end: str | None) -> pd.Series:
    c = _read_close_from_csv_cache(ticker, start, end)
    if c is not None:
        return c
    return _download_close_series(ticker, start, end)


def load_single_close(ticker: str, start: str, end: str | None = None) -> pd.Series:
    """One symbol, adjusted close series (same source order as ``load_prices``)."""
    return _load_leg_close(ticker, start, end)


def load_prices(leg_a_ticker: str, leg_b_ticker: str, start: str, end: str | None = None) -> pd.DataFrame:
    a = _load_leg_close(leg_a_ticker, start, end).rename("a_px")
    b = _load_leg_close(leg_b_ticker, start, end).rename("b_px")
    first_a = a.dropna().index.min()
    first_b = b.dropna().index.min()
    if pd.isna(first_a) or pd.isna(first_b):
        raise RuntimeError(f"No valid price history for selected pair: {leg_a_ticker}/{leg_b_ticker}")
    aligned_start = max(first_a, first_b)
    px = pd.concat([a, b], axis=1)
    px = px.loc[px.index >= aligned_start].dropna()
    if px.empty:
        raise RuntimeError("No overlapping price data for selected pair after start alignment.")
    return px


def _pick_borrow_fee_only(row: pd.Series) -> float | None:
    for key in ("borrow_current", "borrow_fee_annual", "borrow_net_annual"):
        if key not in row.index:
            continue
        v = row.get(key)
        if pd.isna(v):
            continue
        try:
            return float(v)
        except Exception:
            continue
    return None


def _collect_screened_csv_paths(glob_patterns: list[str] | str) -> list[str]:
    patterns = [glob_patterns] if isinstance(glob_patterns, str) else list(glob_patterns)
    out: list[str] = []
    seen: set[str] = set()
    for pattern in patterns:
        if not pattern:
            continue
        for p in Path().glob(pattern):
            if not p.is_file():
                continue
            rp = str(p.resolve())
            if rp in seen:
                continue
            seen.add(rp)
            out.append(rp)
    out.sort()
    return out


def load_pair_borrow_rates(
    ticker_a: str,
    ticker_b: str,
    screened_csv_globs: list[str] | str,
    use_screened: bool,
    fallback_annual: float,
    underlying_ibkr_map: dict[str, float] | None = None,
    manual_override: dict[str, float] | None = None,
) -> tuple[float, float]:
    ticker_a = _norm_sym(ticker_a)
    ticker_b = _norm_sym(ticker_b)
    rates = {ticker_a: float(fallback_annual), ticker_b: float(fallback_annual)}
    if use_screened:
        vals_a: list[float] = []
        for path in _collect_screened_csv_paths(screened_csv_globs):
            try:
                df = pd.read_csv(path)
            except Exception:
                continue
            if "ETF" not in df.columns:
                continue
            df["ETF"] = df["ETF"].astype(str).str.upper().str.strip()
            for _, row in df.iterrows():
                if str(row.get("ETF", "")).upper().strip() != ticker_a:
                    continue
                borrow = _pick_borrow_fee_only(row)
                if borrow is not None and np.isfinite(borrow):
                    vals_a.append(float(borrow))
        if vals_a:
            rates[ticker_a] = float(np.mean(vals_a))
    if underlying_ibkr_map is not None and ticker_b in underlying_ibkr_map and np.isfinite(
        underlying_ibkr_map[ticker_b]
    ):
        rates[ticker_b] = float(underlying_ibkr_map[ticker_b])
    if manual_override:
        for k, v in manual_override.items():
            rates[_norm_sym(k)] = float(v)
    return rates[ticker_a], rates[ticker_b]


def load_beta_values(
    ticker_a: str,
    ticker_b: str,
    screened_csv: str,
    use_screened: bool,
    fallback_a: float,
    fallback_b: float,
) -> tuple[float, float]:
    delta_map = {_norm_sym(ticker_a): float(fallback_a), _norm_sym(ticker_b): float(fallback_b)}
    if use_screened:
        try:
            df = pd.read_csv(screened_csv)
            cols_lower = {c.lower(): c for c in df.columns}
            etf_col = cols_lower.get("etf")
            delta_col = cols_lower.get("delta") or cols_lower.get("beta")
            if etf_col and delta_col:
                df[etf_col] = df[etf_col].astype(str).str.upper().str.strip()
                tmp = df[[etf_col, delta_col]].dropna()
                for _, r in tmp.iterrows():
                    etf = str(r[etf_col])
                    if etf in delta_map:
                        delta_map[etf] = float(r[delta_col])
        except Exception as e:
            print(f"[WARN] Could not load betas from {screened_csv}: {e}")
    return delta_map[_norm_sym(ticker_a)], delta_map[_norm_sym(ticker_b)]


def perf_stats(bt: pd.DataFrame) -> pd.Series:
    n = len(bt)
    if n < 2:
        return pd.Series(dtype=float)
    total_return = bt["equity"].iloc[-1] / bt["equity"].iloc[0] - 1.0
    ann_factor = 252 / max(1, n - 1)
    cagr = (1 + total_return) ** ann_factor - 1 if total_return > -1 else np.nan
    vol = bt["ret"].std() * np.sqrt(252)
    sharpe = (bt["ret"].mean() * 252 / vol) if vol > 0 else np.nan
    max_dd = bt["drawdown"].min()
    reb_friction = float(bt["rebalance_fee"].sum()) if "rebalance_fee" in bt.columns else 0.0
    slip_only = float(bt["slippage_cost"].sum()) if "slippage_cost" in bt.columns else 0.0
    comm_only = (
        float(bt["rebalance_commission"].sum())
        if "rebalance_commission" in bt.columns
        else max(0.0, reb_friction - slip_only)
    )
    return pd.Series(
        {
            "Total Return": total_return,
            "CAGR": cagr,
            "Annual Vol": vol,
            "Sharpe": sharpe,
            "Max Drawdown": max_dd,
            "Avg |Beta Notional|": bt["beta_notional"].abs().mean(),
            "Total Borrow Cost": bt["borrow_cost"].sum(),
            "Total Short Proceeds Credit": bt.get("short_proceeds_credit", pd.Series(dtype=float)).sum(),
            "Net Financing PnL": bt.get("financing_pnl", pd.Series(dtype=float)).sum(),
            "Commission (fee_bps)": comm_only,
            "Total rebalance friction (fee+slip)": reb_friction,
            "Slippage component ($)": slip_only,
            "Rebalance Count": int(bt["rebalance"].sum()),
        }
    )
