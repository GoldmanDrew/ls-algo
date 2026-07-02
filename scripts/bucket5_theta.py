"""
ThetaData SPX option helpers for Bucket 5 put overlays.

Uses the official ``thetadata`` Python client (gRPC / cloud MDDS) when
``THETADATA_API_KEY`` is set in the environment. Caches EOD put quotes under
``data/cache/spx_options/theta/``.

Without credentials the module returns ``None`` and callers fall back to the
Black-Scholes skew model in ``bucket5_put_overlay``.
"""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

THETA_CACHE = Path("data/cache/spx_options/theta")
DEFAULT_SYMBOL = "SPX"
# Theta EOD history is only fetched for the live SPX options era; pre-2022 uses BS.
LIVE_ERA = pd.Timestamp("2022-03-30")


def _spx_monthly_expiries(from_date: date, months: int = 24) -> list[date]:
    """Standard SPX AM-settled monthly expiries (3rd Friday)."""
    out: list[date] = []
    base = pd.Timestamp(from_date).normalize()
    for i in range(months):
        m = base + pd.DateOffset(months=i)
        month_start = m.replace(day=1)
        month_end = month_start + pd.offsets.MonthEnd(0)
        fridays = pd.date_range(month_start, month_end, freq="W-FRI")
        if len(fridays) >= 3:
            out.append(fridays[2].date())
        elif len(fridays):
            out.append(fridays[-1].date())
    return out


def _nearest_listed_expiry(asof: pd.Timestamp, dte_target: int, *, min_dte: int = 63) -> date:
    target = asof + pd.offsets.BDay(dte_target)
    cands = [d for d in _spx_monthly_expiries(asof.date()) if pd.Timestamp(d) > asof + pd.Timedelta(days=min_dte)]
    if not cands:
        cands = _spx_monthly_expiries(asof.date())
    return min(cands, key=lambda d: abs((pd.Timestamp(d) - target).days))


def _client():
    if not os.environ.get("THETADATA_API_KEY"):
        return None
    try:
        from thetadata import ThetaClient
        return ThetaClient(dataframe_type="pandas")
    except Exception:
        return None


def theta_available() -> bool:
    return _client() is not None


def _cache_path(symbol: str, exp: date, strike: float, right: str) -> Path:
    return THETA_CACHE / f"{symbol}_{exp:%Y%m%d}_{int(round(strike*1000))}_{right}.parquet"


def _naive_index(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    idx = pd.to_datetime(idx)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    return idx.normalize()


def fetch_put_eod(
    exp: date,
    strike: float,
    start: date,
    end: date,
    *,
    symbol: str = DEFAULT_SYMBOL,
    right: str = "P",
    refresh: bool = False,
) -> pd.DataFrame | None:
    """EOD put series for one contract; ``None`` if Theta unavailable."""
    cache = _cache_path(symbol, exp, strike, right)
    if cache.is_file() and not refresh:
        df = pd.read_parquet(cache)
        df.index = _naive_index(df.index)
        m = (df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))
        return df.loc[m]

    # No cache and pre-live-era: skip API (extended backtests use Black-Scholes).
    if pd.Timestamp(end) < LIVE_ERA:
        return None

    client = _client()
    if client is None:
        return None
    try:
        raw = client.option_history_eod(
            start_date=start,
            end_date=end,
            symbol=symbol,
            expiration=exp,
            strike=f"{strike:.3f}",
            right=right,
        )
    except Exception:
        return None
    if raw is None or len(raw) == 0:
        return None

    df = _normalize_eod(raw)
    if not df.empty:
        df.index = _naive_index(df.index)
        THETA_CACHE.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache)
    m = (df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))
    return df.loc[m]


def _normalize_eod(raw: pd.DataFrame) -> pd.DataFrame:
    """Map ThetaData EOD frame -> date index with bid/ask/mid."""
    if raw.empty:
        return pd.DataFrame()
    df = raw.copy()
    date_col = next(
        (c for c in ("date", "Date", "trade_date", "created", "last_trade") if c in df.columns),
        None,
    )
    if date_col:
        df["date"] = pd.to_datetime(df[date_col], errors="coerce")
        if getattr(df["date"].dt, "tz", None) is not None:
            df["date"] = df["date"].dt.tz_convert("UTC").dt.tz_localize(None)
        df["date"] = df["date"].dt.normalize()
        df = df.dropna(subset=["date"]).set_index("date").sort_index()
        df = df[~df.index.duplicated(keep="last")]
    bid = pd.to_numeric(df.get("bid"), errors="coerce")
    ask = pd.to_numeric(df.get("ask"), errors="coerce")
    close = pd.to_numeric(df.get("close"), errors="coerce")
    df["mid"] = np.where((bid > 0) & (ask > 0), (bid + ask) / 2.0, close)
    cols = [c for c in ("bid", "ask", "mid", "close", "volume") if c in df.columns]
    return df[cols]


def put_mid_on_date(
    asof: pd.Timestamp,
    spot: float,
    otm_pct: float,
    dte_target: int,
    *,
    symbol: str = DEFAULT_SYMBOL,
) -> tuple[float, dict] | None:
    """Mid price for a put with ~``dte_target`` DTE and ``otm_pct`` OTM on ``asof``.

    Returns (mid_per_share, meta) or None. Uses cached Theta EOD only (no live).
    """
    if pd.Timestamp(asof).normalize() < LIVE_ERA:
        return None
    strike_target = round(spot * (1.0 - otm_pct) / 5.0) * 5.0
    exp = _nearest_listed_expiry(asof, dte_target)
    start = (asof - pd.Timedelta(days=5)).date()
    end = asof.date()
    for bump in (0, -5, 5, -10, 10, -25, 25):
        strike = strike_target + bump
        if strike <= 0:
            continue
        df = fetch_put_eod(exp, strike, start, end, symbol=symbol)
        if df is None or df.empty:
            continue
        row = df.loc[df.index <= asof.normalize()].tail(1)
        if row.empty:
            continue
        mid = float(row["mid"].iloc[0])
        if np.isfinite(mid) and mid > 0:
            return mid, {"strike": strike, "exp": exp, "source": "theta"}
    return None


def put_mtm_on_date(
    asof: pd.Timestamp,
    strike: float,
    exp: date,
    *,
    symbol: str = DEFAULT_SYMBOL,
) -> float | None:
    """Cached Theta EOD mid for an open put on ``asof`` (cache-only, no API)."""
    cache = _cache_path(symbol, exp, strike, "P")
    if not cache.is_file():
        return None
    df = pd.read_parquet(cache)
    df.index = _naive_index(df.index)
    row = df.loc[df.index <= asof.normalize()].tail(1)
    if row.empty:
        return None
    mid = float(row["mid"].iloc[0])
    return mid if np.isfinite(mid) and mid > 0 else None


def prefetch_roll_puts(
    panel_index: pd.DatetimeIndex,
    spot: pd.Series,
    otm_pcts: tuple[float, ...] = (0.20, 0.25, 0.30, 0.35),
    buy_dte: int = 126,
) -> dict:
    """Download Theta EOD puts for approximate roll dates in ``panel_index``."""
    client = _client()
    if client is None:
        return {"status": "no_credentials", "fetched": 0}

    # ~quarterly rolls
    roll_dates = panel_index[::63]
    fetched = 0
    errors = 0
    total = len(roll_dates) * len(otm_pcts)
    for i, dt in enumerate(roll_dates):
        s = float(spot.reindex([dt]).ffill().iloc[0])
        if s <= 0:
            continue
        exp = _nearest_listed_expiry(pd.Timestamp(dt), buy_dte)
        start = (dt - pd.Timedelta(days=5)).date()
        end = min((pd.Timestamp(exp) + pd.Timedelta(days=5)).date(), panel_index.max().date())
        for otm in otm_pcts:
            strike = round(s * (1.0 - otm) / 5.0) * 5.0
            df = fetch_put_eod(exp, strike, start, end, refresh=False)
            if df is not None and not df.empty:
                fetched += 1
            else:
                errors += 1
        if (i + 1) % 3 == 0 or i + 1 == len(roll_dates):
            print(f"  theta prefetch {i + 1}/{len(roll_dates)} rolls  ok={fetched} err={errors}", flush=True)
    return {"status": "ok", "fetched": fetched, "errors": errors, "rolls": len(roll_dates), "attempts": total}


if __name__ == "__main__":
    print("theta_available:", theta_available())
    if theta_available():
        r = prefetch_roll_puts(
            pd.date_range("2024-01-01", "2024-06-01", freq="B"),
            pd.Series(5000.0, index=pd.date_range("2024-01-01", "2024-06-01", freq="B")),
            otm_pcts=(0.25,),
        )
        print("prefetch:", r)
    else:
        print("Set THETADATA_API_KEY to enable ThetaData pricing.")
