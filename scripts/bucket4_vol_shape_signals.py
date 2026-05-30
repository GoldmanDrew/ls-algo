"""
Bucket 4 adaptive hedge-frequency signals & rebalance-cadence policies.

Turns **Trend Ratio (TR)** and **Variance Contribution Ratio (VCR)** time series into a
set of rebalance dates, so the hedge *cadence* (how often a pair re-hedges) becomes a
function of the underlying's vol-shape regime. The per-pair Bucket-4 engine
(``scripts.bucket4_dynamic_bt.run_bucket4_backtest_dynamic_h``) already accepts an explicit
``rebal_dates`` index, so cadence is expressed entirely through the date set produced here
-- no engine change required.

Signal source (primary): ``etf-dashboard/data/vol_shape_history.json`` (60d window, <=252
points, keyed per ETF ticker). Fallback / longer history / other windows: recompute via
``vol_shape.build_underlying_vol_shape_history`` from the pair's underlying price series.

Definitions (see ``vol_shape.py``): ``TR = rv_weekly / rv_daily`` (iid ~ 1, perfect drift ~ sqrt(5)),
``VCR = max(daily r^2) / sum(daily r^2)`` over the window.

No-lookahead: signals are ``shift(lookahead_shift)`` (default 1) so the decision to trade on
day *d*'s close only uses TR/VCR known as of *d-1*.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd


def _default_norm(x: str) -> str:
    return str(x).strip().upper().replace(".", "-")


# --------------------------------------------------------------------------------------
# Signal loading
# --------------------------------------------------------------------------------------
def load_vol_shape_history(
    path: str | Path,
    *,
    norm_sym: Callable[[str], str] = _default_norm,
) -> dict[str, pd.DataFrame]:
    """Parse ``vol_shape_history.json`` into ``{ETF: DataFrame[tr, vcr, vcr_med, rv_daily, rv_weekly]}``.

    Each frame is indexed by (tz-naive, normalized) date. Returns an empty dict if the file
    is missing or unparseable.
    """
    p = Path(path)
    if not p.is_file():
        return {}
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}
    symbols = raw.get("symbols") if isinstance(raw, dict) else None
    if not isinstance(symbols, dict):
        return {}

    out: dict[str, pd.DataFrame] = {}
    for sym, payload in symbols.items():
        series = (payload or {}).get("series") if isinstance(payload, dict) else None
        if not series:
            continue
        df = pd.DataFrame(series)
        if "date" not in df.columns or df.empty:
            continue
        ix = pd.to_datetime(df["date"], errors="coerce")
        if getattr(ix, "tz", None) is not None:
            ix = ix.tz_convert("UTC").tz_localize(None)
        df.index = pd.DatetimeIndex(ix).normalize()
        df = df.loc[df.index.notna()].sort_index()
        df = df[~df.index.duplicated(keep="last")]
        keep = {}
        keep["tr"] = pd.to_numeric(df.get("trend_ratio"), errors="coerce")
        keep["vcr"] = pd.to_numeric(df.get("vcr"), errors="coerce")
        keep["vcr_med"] = pd.to_numeric(df.get("vcr_median"), errors="coerce")
        keep["rv_daily"] = pd.to_numeric(df.get("rv_daily"), errors="coerce")
        keep["rv_weekly"] = pd.to_numeric(df.get("rv_weekly"), errors="coerce")
        out[norm_sym(sym)] = pd.DataFrame(keep, index=df.index)
    return out


def _recompute_signal_from_prices(
    prices: pd.Series,
    *,
    window: int = 60,
) -> pd.DataFrame:
    """Fallback: rolling TR/VCR from a price series via ``vol_shape.build_underlying_vol_shape_history``."""
    try:
        from vol_shape import build_underlying_vol_shape_history
    except Exception:
        return pd.DataFrame(columns=["tr", "vcr", "vcr_med", "rv_daily", "rv_weekly"])
    hist = build_underlying_vol_shape_history(pd.Series(prices).astype(float), window=int(window), max_points=0)
    series = hist.get("series") or []
    if not series:
        return pd.DataFrame(columns=["tr", "vcr", "vcr_med", "rv_daily", "rv_weekly"])
    df = pd.DataFrame(series)
    ix = pd.to_datetime(df["date"], errors="coerce")
    if getattr(ix, "tz", None) is not None:
        ix = ix.tz_convert("UTC").tz_localize(None)
    df.index = pd.DatetimeIndex(ix).normalize()
    df = df.loc[df.index.notna()].sort_index()
    return pd.DataFrame(
        {
            "tr": pd.to_numeric(df.get("trend_ratio"), errors="coerce"),
            "vcr": pd.to_numeric(df.get("vcr"), errors="coerce"),
            "vcr_med": pd.to_numeric(df.get("vcr_median"), errors="coerce"),
            "rv_daily": pd.to_numeric(df.get("rv_daily"), errors="coerce"),
            "rv_weekly": pd.to_numeric(df.get("rv_weekly"), errors="coerce"),
        },
        index=df.index,
    )


def get_pair_signal(
    etf: str,
    und: str,
    calendar: pd.DatetimeIndex,
    *,
    history: dict[str, pd.DataFrame],
    underlying_prices: pd.Series | None = None,
    window: int = 60,
    lookahead_shift: int = 1,
    norm_sym: Callable[[str], str] = _default_norm,
) -> pd.DataFrame:
    """Per-pair TR/VCR aligned to *calendar* (ffill), shifted to avoid lookahead.

    Source priority: ``history[ETF]`` first; if absent/empty and *underlying_prices* is
    given, recompute from prices. ``vcr_med`` is filled with the expanding median of VCR
    when missing so signal-relative policies still work.
    """
    cal = pd.DatetimeIndex(calendar)
    if getattr(cal, "tz", None) is not None:
        cal = cal.tz_convert("UTC").tz_localize(None)
    cal = cal.normalize()

    df = history.get(norm_sym(etf))
    source = "history"
    if (df is None or df.empty) and underlying_prices is not None:
        df = _recompute_signal_from_prices(underlying_prices, window=window)
        source = "recompute"
    if df is None or df.empty:
        empty = pd.DataFrame(
            {"tr": np.nan, "vcr": np.nan, "vcr_med": np.nan},
            index=cal,
        )
        empty.attrs["signal_source"] = "missing"
        return empty

    aligned = df.reindex(df.index.union(cal)).sort_index().ffill().reindex(cal)
    if "vcr_med" not in aligned or aligned["vcr_med"].isna().all():
        aligned["vcr_med"] = aligned["vcr"].expanding(min_periods=1).median()
    aligned["vcr_med"] = aligned["vcr_med"].fillna(aligned["vcr"].expanding(min_periods=1).median())

    if lookahead_shift:
        aligned = aligned.shift(int(lookahead_shift))
    aligned.attrs["signal_source"] = source
    return aligned


# --------------------------------------------------------------------------------------
# Cadence policies  ->  (rebal_dates, diag_frame)
# --------------------------------------------------------------------------------------
def _calendar_after_warmup(calendar: pd.DatetimeIndex, warmup: int) -> pd.DatetimeIndex:
    cal = pd.DatetimeIndex(calendar).sort_values().unique()
    if getattr(cal, "tz", None) is not None:
        cal = cal.tz_convert("UTC").tz_localize(None)
    if warmup > 0:
        cal = cal[int(warmup) :]
    return cal


def policy_continuous_interval(
    calendar: pd.DatetimeIndex,
    signal: pd.DataFrame,
    *,
    base_days: float = 10.0,
    k_tr: float = 1.5,
    m_vcr: float = 2.0,
    min_interval: int = 2,
    max_interval: int = 42,
    warmup_bdays: int = 0,
) -> tuple[pd.DatetimeIndex, pd.DataFrame]:
    """Cadence as a smooth function of TR & VCR.

    ``interval = clip(base_days / (1 + k_tr*(TR-1) + m_vcr*(VCR-VCR_med)), min, max)`` trading days.
    Higher TR (trend) and elevated VCR (concentration) shorten the interval -> hedge more often.
    """
    cal = _calendar_after_warmup(calendar, warmup_bdays)
    if len(cal) == 0:
        return pd.DatetimeIndex([]), pd.DataFrame()
    tr = signal.get("tr") if signal is not None else None
    vcr = signal.get("vcr") if signal is not None else None
    vcr_med = signal.get("vcr_med") if signal is not None else None

    dates: list[pd.Timestamp] = []
    diag: list[dict[str, Any]] = []
    i, n = 0, len(cal)
    while i < n:
        d = pd.Timestamp(cal[i])
        dates.append(d)
        t = float(tr.get(d, np.nan)) if tr is not None else np.nan
        v = float(vcr.get(d, np.nan)) if vcr is not None else np.nan
        vm = float(vcr_med.get(d, np.nan)) if vcr_med is not None else np.nan
        denom = 1.0
        if np.isfinite(t):
            denom += k_tr * (t - 1.0)
        if np.isfinite(v) and np.isfinite(vm):
            denom += m_vcr * (v - vm)
        interval = base_days / denom if denom > 1e-9 else float(max_interval)
        if not np.isfinite(interval):
            interval = float(max_interval)
        interval = int(np.clip(round(interval), int(min_interval), int(max_interval)))
        diag.append({"date": d, "tr": t, "vcr": v, "vcr_med": vm, "interval_days": interval})
        i += max(1, interval)
    return pd.DatetimeIndex(dates), pd.DataFrame(diag)


def policy_signal_change(
    calendar: pd.DatetimeIndex,
    signal: pd.DataFrame,
    *,
    d_tr: float = 0.10,
    d_vcr: float = 0.08,
    min_interval: int = 2,
    max_interval: int = 42,
    warmup_bdays: int = 0,
) -> tuple[pd.DatetimeIndex, pd.DataFrame]:
    """Event-driven cadence: rebalance when |dTR| or |dVCR| since the last rebalance exceeds a
    threshold, clamped by ``min_interval`` (suppress churn) and ``max_interval`` (force a trade).
    """
    cal = _calendar_after_warmup(calendar, warmup_bdays)
    if len(cal) == 0:
        return pd.DatetimeIndex([]), pd.DataFrame()
    tr = signal.get("tr") if signal is not None else None
    vcr = signal.get("vcr") if signal is not None else None

    dates: list[pd.Timestamp] = []
    diag: list[dict[str, Any]] = []
    last_idx = 0
    last_tr = float(tr.get(cal[0], np.nan)) if tr is not None else np.nan
    last_vcr = float(vcr.get(cal[0], np.nan)) if vcr is not None else np.nan
    for i in range(len(cal)):
        d = pd.Timestamp(cal[i])
        t = float(tr.get(d, np.nan)) if tr is not None else np.nan
        v = float(vcr.get(d, np.nan)) if vcr is not None else np.nan
        if i == 0:
            dates.append(d)
            diag.append({"date": d, "tr": t, "vcr": v, "reason": "initial"})
            last_idx, last_tr, last_vcr = 0, t, v
            continue
        since = i - last_idx
        if since < int(min_interval):
            continue
        d_tr_now = abs(t - last_tr) if np.isfinite(t) and np.isfinite(last_tr) else 0.0
        d_vcr_now = abs(v - last_vcr) if np.isfinite(v) and np.isfinite(last_vcr) else 0.0
        trig = (d_tr_now >= float(d_tr)) or (d_vcr_now >= float(d_vcr))
        force = since >= int(max_interval)
        if trig or force:
            dates.append(d)
            reason = "max_interval" if force and not trig else ("tr_change" if d_tr_now >= float(d_tr) else "vcr_change")
            diag.append(
                {"date": d, "tr": t, "vcr": v, "d_tr": d_tr_now, "d_vcr": d_vcr_now, "reason": reason}
            )
            last_idx, last_tr, last_vcr = i, t, v
    return pd.DatetimeIndex(sorted(set(dates))), pd.DataFrame(diag)


def policy_fixed(
    calendar: pd.DatetimeIndex,
    *,
    freq: str = "W-FRI",
    warmup_bdays: int = 0,
) -> tuple[pd.DatetimeIndex, pd.DataFrame]:
    """Fixed calendar baseline (e.g. weekly ``W-FRI``) via ``bucket4_weekly_opt2.weekly_rebalance_dates``."""
    from scripts.bucket4_weekly_opt2 import weekly_rebalance_dates

    ix = weekly_rebalance_dates(pd.DatetimeIndex(calendar), freq, warmup_bdays=int(warmup_bdays))
    return ix, pd.DataFrame({"date": ix})


def policy_every_n_days(
    calendar: pd.DatetimeIndex,
    *,
    n_days: int = 5,
    warmup_bdays: int = 0,
) -> tuple[pd.DatetimeIndex, pd.DataFrame]:
    """Fixed interval on the trading calendar: rebalance every *n_days* sessions (step on *calendar*)."""
    cal = _calendar_after_warmup(calendar, warmup_bdays)
    n = max(1, int(n_days))
    if len(cal) == 0:
        return pd.DatetimeIndex([]), pd.DataFrame()
    dates: list[pd.Timestamp] = []
    i, m = 0, len(cal)
    while i < m:
        dates.append(pd.Timestamp(cal[i]))
        i += n
    ix = pd.DatetimeIndex(dates)
    return ix, pd.DataFrame({"date": ix, "n_days": n})


def rebalance_cadence_stats(rebal_dates: pd.DatetimeIndex) -> dict[str, float]:
    """Summary cadence metrics for a rebalance schedule."""
    ix = pd.DatetimeIndex(rebal_dates).sort_values().unique()
    n = int(len(ix))
    if n <= 1:
        return {"n_rebalances": float(n), "mean_interval_days": float("nan"), "median_interval_days": float("nan")}
    gaps = np.diff(ix.values).astype("timedelta64[D]").astype(float)
    return {
        "n_rebalances": float(n),
        "mean_interval_days": float(np.mean(gaps)),
        "median_interval_days": float(np.median(gaps)),
    }


__all__ = [
    "load_vol_shape_history",
    "get_pair_signal",
    "policy_continuous_interval",
    "policy_signal_change",
    "policy_fixed",
    "policy_every_n_days",
    "rebalance_cadence_stats",
]
