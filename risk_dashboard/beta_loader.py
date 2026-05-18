"""Compute underlying-to-index betas + realized vol per underlying.

Phase I durable factor model. Replaces the hardcoded ``BETA_TO_SPY`` map
with live betas computed from a rolling window of daily log returns.

Indices covered:
    * SPY (SPX proxy)
    * QQQ (NDX proxy)
    * IWM (RUT proxy)

For each underlying in the book this module returns a ``BetaResult``
carrying ``beta_to_spy``, ``beta_to_ndx``, ``beta_to_rut`` plus standard
error, observation count, R^2, annualized realized vol, and provenance
flag (``computed`` / ``curated_fallback`` / ``default_fallback``).

Caching:
    * yfinance is hit once per (symbol, business day). Results cached on
      disk under ``data/cache/beta_history/<symbol>.csv`` (raw closes).
    * A circuit breaker aborts further yfinance batches if one entire
      batch returns zero symbols (network blocked / rate-limited) so the
      build never stalls on a cold cache.
"""

from __future__ import annotations

import json
import math
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .factor_map import (
    BETA_TO_SPY,
    DEFAULT_BROAD_INDEX_BETA,
    DEFAULT_SINGLE_NAME_BETA,
    SECTOR_MAP,
)


INDEX_TICKERS: dict[str, str] = {
    "spy": "SPY",
    "ndx": "QQQ",
    "rut": "IWM",
}

DEFAULT_WINDOW_DAYS = 60
MIN_OBS_FOR_TRUST = 30
SHRINKAGE_PRIOR = 1.0
CACHE_DIR_DEFAULT = Path("data/cache/beta_history")
SUMMARY_CACHE_PATH_DEFAULT = Path("data/cache/beta_summary.json")
CACHE_MAX_AGE_HOURS = 24.0
TRADING_DAYS_PER_YEAR = 252


@dataclass
class BetaResult:
    underlying: str
    beta_to_spy: float | None = None
    beta_to_ndx: float | None = None
    beta_to_rut: float | None = None
    beta_se: float | None = None
    n_obs: int = 0
    r2: float | None = None
    regime_vol_pct: float | None = None
    provenance: str = "default_fallback"
    shrinkage_applied: bool = False
    cache_age_hours: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "underlying": self.underlying,
            "beta_to_spy": self.beta_to_spy,
            "beta_to_ndx": self.beta_to_ndx,
            "beta_to_rut": self.beta_to_rut,
            "beta_se": self.beta_se,
            "n_obs": self.n_obs,
            "r2": self.r2,
            "regime_vol_pct": self.regime_vol_pct,
            "provenance": self.provenance,
            "shrinkage_applied": self.shrinkage_applied,
            "cache_age_hours": self.cache_age_hours,
        }


def _csv_cache_path(cache_dir: Path, symbol: str) -> Path:
    safe = symbol.replace("/", "_").replace("\\", "_").upper()
    return cache_dir / f"{safe}.csv"


def _load_cached_closes(cache_dir: Path, symbol: str) -> pd.Series | None:
    path = _csv_cache_path(cache_dir, symbol)
    if not path.is_file():
        return None
    try:
        df = pd.read_csv(path, parse_dates=["date"])
    except Exception:
        return None
    if df.empty or "close" not in df.columns:
        return None
    s = df.set_index("date")["close"].astype(float).sort_index()
    s.name = symbol
    return s


def _save_cached_closes(cache_dir: Path, symbol: str, series: pd.Series) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    df = series.rename("close").reset_index().rename(columns={"index": "date"})
    if "date" not in df.columns:
        df = df.rename(columns={df.columns[0]: "date"})
    df.to_csv(_csv_cache_path(cache_dir, symbol), index=False)


def _cache_age_hours(cache_dir: Path, symbol: str) -> float | None:
    path = _csv_cache_path(cache_dir, symbol)
    if not path.is_file():
        return None
    mtime = path.stat().st_mtime
    return max(0.0, (time.time() - mtime) / 3600.0)


def _fetch_yfinance_closes(
    symbols: Iterable[str],
    *,
    window_days: int,
    cache_dir: Path,
    refresh_max_age_hours: float = CACHE_MAX_AGE_HOURS,
    yf_module: Any | None = None,
    chunk_size: int = 25,
    max_failed_chunks_before_abort: int = 1,
) -> dict[str, pd.Series]:
    """Cache-first daily closes. Circuit-breaks after one empty batch so
    the build never stalls on a dead network."""
    needed: list[str] = []
    out: dict[str, pd.Series] = {}
    for sym in symbols:
        age = _cache_age_hours(cache_dir, sym)
        cached = _load_cached_closes(cache_dir, sym)
        if cached is not None and age is not None and age <= refresh_max_age_hours:
            out[sym] = cached
            continue
        needed.append(sym)
        if cached is not None:
            out[sym] = cached  # fallback if refresh fails

    if not needed:
        return out

    yf = yf_module
    if yf is None:
        try:
            import yfinance as yf  # type: ignore
        except Exception:
            return out

    period_days = max(window_days * 2 + 10, 90)
    failed_streak = 0
    for i in range(0, len(needed), chunk_size):
        batch = needed[i : i + chunk_size]
        new_in_batch = 0
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                data = yf.download(
                    tickers=" ".join(batch),
                    period=f"{period_days}d",
                    interval="1d",
                    auto_adjust=True,
                    actions=False,
                    progress=False,
                    group_by="ticker",
                    threads=True,
                )
        except Exception:
            data = None
        if data is not None and len(data) > 0:
            for sym in batch:
                try:
                    if isinstance(data.columns, pd.MultiIndex):
                        if sym not in data.columns.get_level_values(0):
                            continue
                        close = data[sym]["Close"].dropna().astype(float)
                    else:
                        close = data["Close"].dropna().astype(float)
                except Exception:
                    continue
                if close.empty:
                    continue
                close.name = sym
                close.index = pd.to_datetime(close.index).normalize()
                out[sym] = close
                _save_cached_closes(cache_dir, sym, close)
                new_in_batch += 1
        if new_in_batch == 0:
            failed_streak += 1
            if failed_streak >= max_failed_chunks_before_abort:
                break
        else:
            failed_streak = 0

    return out


def _ols_beta(
    y: pd.Series,
    x: pd.Series,
) -> tuple[float | None, float | None, int, float | None]:
    """OLS log-return beta. Returns (beta, beta_se, n_obs, r2)."""
    paired = pd.concat([y, x], axis=1, join="inner").dropna()
    if len(paired) < 5:
        return None, None, len(paired), None
    yv = paired.iloc[:, 0].to_numpy(dtype=float)
    xv = paired.iloc[:, 1].to_numpy(dtype=float)
    xv_mean = xv.mean()
    yv_mean = yv.mean()
    sxx = float(((xv - xv_mean) ** 2).sum())
    if sxx <= 1e-12:
        return None, None, len(paired), None
    sxy = float(((xv - xv_mean) * (yv - yv_mean)).sum())
    beta = sxy / sxx
    alpha = yv_mean - beta * xv_mean
    resid = yv - (alpha + beta * xv)
    sse = float((resid ** 2).sum())
    sst = float(((yv - yv_mean) ** 2).sum())
    r2 = (1.0 - sse / sst) if sst > 1e-18 else None
    dof = max(len(paired) - 2, 1)
    sigma2 = sse / dof
    beta_se = math.sqrt(sigma2 / sxx) if sxx > 0 else None
    return beta, beta_se, len(paired), r2


def _log_returns(close: pd.Series | None, window_days: int) -> pd.Series:
    if close is None or close.empty:
        return pd.Series(dtype=float)
    s = close.sort_index().tail(window_days + 1)
    return np.log(s).diff().dropna()


def _shrink_beta(
    beta: float | None, n_obs: int, prior: float = SHRINKAGE_PRIOR
) -> tuple[float | None, bool]:
    if beta is None:
        return None, False
    if n_obs >= MIN_OBS_FOR_TRUST:
        return beta, False
    if n_obs <= 0:
        return prior, True
    w = n_obs / MIN_OBS_FOR_TRUST
    return prior * (1.0 - w) + beta * w, True


def compute_betas(
    underlyings: Iterable[str],
    *,
    window_days: int = DEFAULT_WINDOW_DAYS,
    cache_dir: Path | None = None,
    yf_module: Any | None = None,
    refresh_max_age_hours: float = CACHE_MAX_AGE_HOURS,
) -> dict[str, BetaResult]:
    """Compute beta-to-{SPY,QQQ,IWM} for each ``underlying``.

    Falls back to curated ``BETA_TO_SPY`` map (SPY only) when yfinance
    returns no data. NDX/RUT betas approximate via a coarse scaling of
    the SPY beta when paired returns are unavailable so the UI slide
    strips stay populated end-to-end.
    """
    cache_dir = cache_dir or CACHE_DIR_DEFAULT
    symbols = sorted({str(u).strip().upper() for u in underlyings if str(u).strip()})
    if not symbols:
        return {}

    universe = symbols + list(INDEX_TICKERS.values())
    closes = _fetch_yfinance_closes(
        universe,
        window_days=window_days,
        cache_dir=cache_dir,
        refresh_max_age_hours=refresh_max_age_hours,
        yf_module=yf_module,
    )

    spy_ret = _log_returns(closes.get("SPY"), window_days)
    qqq_ret = _log_returns(closes.get("QQQ"), window_days)
    iwm_ret = _log_returns(closes.get("IWM"), window_days)

    results: dict[str, BetaResult] = {}
    for sym in symbols:
        res = BetaResult(underlying=sym, cache_age_hours=_cache_age_hours(cache_dir, sym))
        y = _log_returns(closes.get(sym), window_days)

        if not y.empty and not spy_ret.empty:
            b_spy, se_spy, n_spy, r2_spy = _ols_beta(y, spy_ret)
            b_spy, shrunk_spy = _shrink_beta(b_spy, n_spy)
            res.beta_to_spy = b_spy
            res.beta_se = se_spy
            res.n_obs = n_spy
            res.r2 = r2_spy
            res.shrinkage_applied = shrunk_spy
            std_d = float(y.std(ddof=1)) if len(y) > 1 else 0.0
            res.regime_vol_pct = std_d * math.sqrt(TRADING_DAYS_PER_YEAR) * 100.0
            res.provenance = "computed"
        else:
            curated = BETA_TO_SPY.get(sym)
            if curated is not None:
                res.beta_to_spy = float(curated)
                res.provenance = "curated_fallback"
            else:
                sector = SECTOR_MAP.get(sym)
                res.beta_to_spy = (
                    DEFAULT_BROAD_INDEX_BETA if sector == "broad" else DEFAULT_SINGLE_NAME_BETA
                )
                res.provenance = "default_fallback"

        if not y.empty and not qqq_ret.empty:
            b, _, _, _ = _ols_beta(y, qqq_ret)
            b, _ = _shrink_beta(b, len(y))
            res.beta_to_ndx = b
        if not y.empty and not iwm_ret.empty:
            b, _, _, _ = _ols_beta(y, iwm_ret)
            b, _ = _shrink_beta(b, len(y))
            res.beta_to_rut = b

        # Coarse fallback: scale SPY beta when paired returns missing.
        if res.beta_to_ndx is None and res.beta_to_spy is not None:
            res.beta_to_ndx = res.beta_to_spy * 0.90
        if res.beta_to_rut is None and res.beta_to_spy is not None:
            res.beta_to_rut = res.beta_to_spy * 0.85

        results[sym] = res

    return results


def write_summary_cache(
    results: dict[str, BetaResult],
    *,
    snapshot_date: str,
    path: Path | None = None,
) -> None:
    path = path or SUMMARY_CACHE_PATH_DEFAULT
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "snapshot_date": snapshot_date,
        "generated_at_utc": datetime.utcnow().isoformat(),
        "rows": [r.to_dict() for r in results.values()],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def read_summary_cache(path: Path | None = None) -> dict[str, Any] | None:
    path = path or SUMMARY_CACHE_PATH_DEFAULT
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
