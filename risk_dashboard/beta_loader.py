"""Compute underlying-to-index betas + realized vol per underlying.

Phase II durable factor model. Computes 252-day OLS betas to SPY/QQQ/
IWM with two-pass shrinkage toward sector-mean priors plus AR(1)-
adjusted effective sample sizes (same shape as
:func:`daily_screener.compute_beta_shrunk`).

Data sources, in fail-over order:

    1. Yahoo Finance v8 chart API (adjclose; cached on disk).
    2. Stooq CSV (``https://stooq.com/q/d/l/?s={sym}.us&i=d``) when enabled.
    3. Stale local cache (``data/cache/beta_history/<SYM>.csv``).
    4. Skip -- mark provenance and fall back to curated / default.

Each :class:`BetaResult` carries ``beta_to_spy``, ``beta_to_ndx``,
``beta_to_rut`` plus standard error, observation count, R^2, annualized
realized vol (60d), AR(1)-adjusted effective sample size, and
provenance flag (``computed`` / ``shrunk`` / ``curated_fallback`` /
``default_fallback``).
"""

from __future__ import annotations

import csv
import io
import json
import math
import random
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import requests

import numpy as np
import pandas as pd

from .factor_map import (
    BETA_TO_SPY,
    DEFAULT_BROAD_INDEX_BETA,
    DEFAULT_SINGLE_NAME_BETA,
    OVERRIDE_SECTOR_MAP,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

INDEX_TICKERS: dict[str, str] = {
    "spy": "SPY",
    "ndx": "QQQ",
    "rut": "IWM",
}

DEFAULT_WINDOW_DAYS = 252
REGIME_VOL_WINDOW_DAYS = 60
MIN_OBS_FOR_TRUST = 60
BETA_K_BASE = 60                  # matches daily_screener.BETA_SHRINK_K_BASE
BETA_AR1_FLOOR_RATIO = 0.10       # n_eff floor as fraction of n
SECTOR_MIN_COMPUTED_NAMES = 5     # min names to trust a sector-mean prior
SHRINKAGE_PRIOR = 1.0             # legacy fallback prior when no sector
CACHE_DIR_DEFAULT = Path("data/cache/beta_history")
SUMMARY_CACHE_PATH_DEFAULT = Path("data/cache/beta_summary.json")
CACHE_MAX_AGE_HOURS = 24.0
STALE_FALLBACK_MAX_AGE_DAYS = 14.0
TRADING_DAYS_PER_YEAR = 252

STOOQ_URL_TEMPLATE = "https://stooq.com/q/d/l/?s={sym}.us&i=d"
STOOQ_TIMEOUT_SECONDS = 8.0
YAHOO_V8_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{sym}"
YAHOO_V8_TIMEOUT_SECONDS = 15.0
YAHOO_V8_MAX_RETRIES = 3
YAHOO_V8_USER_AGENT = "Mozilla/5.0"
YAHOO_V8_MAX_WORKERS = 8


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class BetaResult:
    underlying: str
    beta_to_spy: float | None = None
    beta_to_ndx: float | None = None
    beta_to_rut: float | None = None
    beta_to_spy_raw: float | None = None   # pre-shrinkage OLS beta
    beta_to_ndx_raw: float | None = None
    beta_to_rut_raw: float | None = None
    beta_se: float | None = None
    n_obs: int = 0
    n_eff: int = 0
    r2: float | None = None
    regime_vol_pct: float | None = None
    sector: str | None = None
    prior_used_spy: float | None = None
    prior_source: str | None = None      # "sector_mean" / "curated" / "default"
    shrinkage_weight: float | None = None  # w (OLS weight) for SPY
    provenance: str = "default_fallback"
    shrinkage_applied: bool = False
    cache_age_hours: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "underlying": self.underlying,
            "beta_to_spy": self.beta_to_spy,
            "beta_to_ndx": self.beta_to_ndx,
            "beta_to_rut": self.beta_to_rut,
            "beta_to_spy_raw": self.beta_to_spy_raw,
            "beta_to_ndx_raw": self.beta_to_ndx_raw,
            "beta_to_rut_raw": self.beta_to_rut_raw,
            "beta_se": self.beta_se,
            "n_obs": self.n_obs,
            "n_eff": self.n_eff,
            "r2": self.r2,
            "regime_vol_pct": self.regime_vol_pct,
            "sector": self.sector,
            "prior_used_spy": self.prior_used_spy,
            "prior_source": self.prior_source,
            "shrinkage_weight": self.shrinkage_weight,
            "provenance": self.provenance,
            "shrinkage_applied": self.shrinkage_applied,
            "cache_age_hours": self.cache_age_hours,
        }


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Data sources
# ---------------------------------------------------------------------------


def _yahoo_v8_range(period_days: int) -> str:
    """Pick a Yahoo ``range`` param wide enough for ``period_days`` OLS."""
    if period_days >= 400:
        return "5y"
    if period_days >= 200:
        return "2y"
    if period_days >= 90:
        return "1y"
    return "6mo"


def _parse_yahoo_v8_chart(payload: dict[str, Any], symbol: str) -> pd.Series | None:
    """Parse adjclose series from a Yahoo v8 chart JSON body."""
    result = ((payload.get("chart") or {}).get("result") or [None])[0]
    if not result:
        return None
    timestamps = result.get("timestamp") or []
    adjclose = ((result.get("indicators") or {}).get("adjclose") or [{}])[0].get(
        "adjclose"
    )
    if not timestamps or not adjclose:
        return None
    idx = (
        pd.to_datetime(timestamps, unit="s", utc=True)
        .tz_convert("America/New_York")
        .normalize()
    )
    s = pd.Series(adjclose, index=idx, name=symbol, dtype=float).dropna()
    if s.empty:
        return None
    return s


def _fetch_one_yahoo_v8_series(
    symbol: str,
    *,
    range_param: str,
) -> pd.Series | None:
    """Fetch one symbol's adjclose history via Yahoo v8 chart API."""
    sym = str(symbol or "").strip().upper()
    if not sym:
        return None
    yahoo_sym = sym.replace(".", "-")
    url = YAHOO_V8_CHART_URL.format(sym=yahoo_sym)
    params = {"range": range_param, "interval": "1d", "events": "div,splits"}
    headers = {"User-Agent": YAHOO_V8_USER_AGENT}
    last_exc: Exception | None = None
    for attempt in range(1, YAHOO_V8_MAX_RETRIES + 1):
        try:
            resp = requests.get(
                url,
                params=params,
                headers=headers,
                timeout=YAHOO_V8_TIMEOUT_SECONDS,
            )
            if resp.status_code == 429:
                raise requests.HTTPError("429 Too Many Requests")
            resp.raise_for_status()
            series = _parse_yahoo_v8_chart(resp.json(), sym)
            if series is not None and not series.empty:
                return series
            raise ValueError("empty adjclose series")
        except Exception as exc:
            last_exc = exc
            if attempt < YAHOO_V8_MAX_RETRIES:
                sleep_s = (0.5 * (2 ** (attempt - 1))) + random.uniform(0.0, 0.25)
                time.sleep(sleep_s)
    _ = last_exc
    return None


def _fetch_yfinance_chunk(
    batch: list[str],
    *,
    period_days: int,
    yf_module: Any | None = None,
) -> dict[str, pd.Series]:
    """Fetch a batch of symbols via Yahoo v8 chart API.

    The name is kept for back-compat with callers/tests. ``yf_module`` is
    ignored -- the v8 API replaces broken ``yfinance.download`` batches.
    """
    _ = yf_module
    if not batch:
        return {}
    range_param = _yahoo_v8_range(max(period_days * 2 + 10, 90))
    out: dict[str, pd.Series] = {}
    workers = min(YAHOO_V8_MAX_WORKERS, max(1, len(batch)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                _fetch_one_yahoo_v8_series,
                sym,
                range_param=range_param,
            ): sym
            for sym in batch
        }
        for fut in as_completed(futures):
            sym = futures[fut]
            try:
                series = fut.result()
            except Exception:
                continue
            if series is not None and not series.empty:
                out[sym] = series
    return out


def _fetch_stooq_closes(
    symbols: Iterable[str],
    *,
    timeout: float = STOOQ_TIMEOUT_SECONDS,
) -> dict[str, pd.Series]:
    """Stooq CSV adapter. Free, no auth, US tickers only.

    Returns ``{symbol: close_series}`` for every symbol Stooq returns
    daily bars for. Symbols Stooq does not recognise (or any network /
    parse failure) are silently skipped -- the caller is expected to
    chain another data source after this one.
    """
    out: dict[str, pd.Series] = {}
    for raw in symbols:
        sym = str(raw or "").strip().upper()
        if not sym:
            continue
        url = STOOQ_URL_TEMPLATE.format(sym=sym.lower())
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "ls-algo-risk-dashboard/1.0"},
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = resp.read().decode("utf-8", errors="ignore")
        except (urllib.error.URLError, TimeoutError, OSError):
            continue
        except Exception:
            continue
        series = _parse_stooq_csv(body, sym)
        if series is not None and not series.empty:
            out[sym] = series
    return out


def _parse_stooq_csv(body: str, symbol: str) -> pd.Series | None:
    """Parse the Stooq daily-bars CSV body. None on bad input."""
    if not body or body.lstrip().lower().startswith("no data"):
        return None
    reader = csv.DictReader(io.StringIO(body))
    rows: list[tuple[pd.Timestamp, float]] = []
    for row in reader:
        d = (row.get("Date") or row.get("date") or "").strip()
        c = (row.get("Close") or row.get("close") or "").strip()
        if not d or not c:
            continue
        try:
            ts = pd.Timestamp(d).normalize()
            val = float(c)
        except (ValueError, TypeError):
            continue
        if math.isfinite(val):
            rows.append((ts, val))
    if not rows:
        return None
    idx, vals = zip(*sorted(rows))
    s = pd.Series(vals, index=pd.DatetimeIndex(idx), name=symbol, dtype=float)
    return s


def _fetch_closes(
    symbols: Iterable[str],
    *,
    window_days: int,
    cache_dir: Path,
    refresh_max_age_hours: float = CACHE_MAX_AGE_HOURS,
    yf_module: Any | None = None,
    chunk_size: int = 25,
    max_failed_chunks_before_abort: int = 1,
    enable_stooq_fallback: bool = True,
    stale_fallback_max_age_days: float = STALE_FALLBACK_MAX_AGE_DAYS,
) -> dict[str, pd.Series]:
    """Cache-first daily closes with Yahoo v8 -> Stooq -> stale-cache chain.

    Order of operations per symbol:

        1. Fresh on-disk cache (age <= ``refresh_max_age_hours``) wins.
        2. Otherwise request Yahoo v8 chart API in chunks. A chunk with
           zero successful symbols counts toward an abort streak so the
           build never stalls on a dead network.
        3. Any symbol still missing after Yahoo is retried via Stooq
           (when ``enable_stooq_fallback``).
        4. Any symbol still missing falls back to its (possibly stale)
           on-disk cache provided the cache is at most
           ``stale_fallback_max_age_days`` old.
    """
    needed: list[str] = []
    out: dict[str, pd.Series] = {}
    stale_cache: dict[str, pd.Series] = {}

    syms_list = [str(s).strip().upper() for s in symbols if str(s).strip()]
    for sym in syms_list:
        age = _cache_age_hours(cache_dir, sym)
        cached = _load_cached_closes(cache_dir, sym)
        if cached is not None and age is not None and age <= refresh_max_age_hours:
            out[sym] = cached
            continue
        needed.append(sym)
        if cached is not None:
            stale_cache[sym] = cached

    if not needed:
        return out

    # ----- Tier 1: yfinance -----
    failed_streak = 0
    for i in range(0, len(needed), chunk_size):
        batch = needed[i : i + chunk_size]
        batch_out = _fetch_yfinance_chunk(
            batch, period_days=max(window_days * 2 + 10, 90), yf_module=yf_module
        )
        for sym, close in batch_out.items():
            out[sym] = close
            _save_cached_closes(cache_dir, sym, close)
        if not batch_out:
            failed_streak += 1
            if failed_streak >= max_failed_chunks_before_abort:
                break
        else:
            failed_streak = 0

    # ----- Tier 2: Stooq fallback for anything still missing -----
    still_missing = [s for s in needed if s not in out]
    if still_missing and enable_stooq_fallback:
        stooq_out = _fetch_stooq_closes(still_missing)
        for sym, close in stooq_out.items():
            out[sym] = close
            _save_cached_closes(cache_dir, sym, close)

    # ----- Tier 3: stale on-disk cache as last resort -----
    stale_cutoff_hours = stale_fallback_max_age_days * 24.0
    for sym in needed:
        if sym in out:
            continue
        cached = stale_cache.get(sym)
        age = _cache_age_hours(cache_dir, sym)
        if cached is not None and age is not None and age <= stale_cutoff_hours:
            out[sym] = cached

    return out


# Back-compat alias. Older callers (vol_vix_beta, tests) imported this name.
_fetch_yfinance_closes = _fetch_closes


# ---------------------------------------------------------------------------
# Estimator
# ---------------------------------------------------------------------------


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


def _ar1_n_eff(returns: np.ndarray | pd.Series) -> tuple[float, int]:
    """Lag-1 autocorrelation and AR(1)-adjusted effective sample size.

    Mirrors ``daily_screener._ar1_n_eff`` so the two estimators agree
    on what counts as an "independent observation"::

        n_eff = n * (1 - rho) / (1 + rho),
        clipped to [BETA_AR1_FLOOR_RATIO * n, n].
    """
    arr = np.asarray(returns, dtype=float)
    n = arr.size
    if n < 5:
        return 0.0, n
    r = arr - arr.mean()
    denom = float((r * r).sum())
    if denom <= 0:
        return 0.0, n
    rho = float((r[1:] * r[:-1]).sum() / denom)
    rho = max(-0.95, min(0.95, rho))
    factor = (1.0 - rho) / (1.0 + rho)
    n_eff = int(round(max(BETA_AR1_FLOOR_RATIO * n, min(n, n * factor))))
    return rho, max(1, n_eff)


def _shrink_beta_to_sector(
    beta_ols: float | None,
    n_eff: int,
    *,
    prior: float,
    k_base: float = BETA_K_BASE,
) -> tuple[float | None, float, bool]:
    """Bayesian shrinkage toward an explicit ``prior``.

    Shape mirrors ``daily_screener.compute_beta_shrunk``::

        k = k_base * max(1, prior ** 2)
        w = n_eff / (n_eff + k)
        beta_final = w * beta_ols + (1 - w) * prior

    Returns ``(beta_shrunk, w, shrinkage_applied)`` where ``w`` is the
    OLS weight (closer to 1 = trust the OLS more; closer to 0 = trust
    the prior more) and ``shrinkage_applied`` is True whenever the
    prior contributed more than 5% to the blended estimate
    (``w < 0.95``).
    """
    if beta_ols is None:
        return prior, 0.0, True
    k = float(k_base) * (max(1.0, float(prior) * float(prior)))
    w = float(n_eff) / (float(n_eff) + k) if n_eff > 0 else 0.0
    blended = w * float(beta_ols) + (1.0 - w) * float(prior)
    return blended, w, (w < 0.95)


# Back-compat wrapper for callers that don't have a sector prior
# (e.g. vol_vix_beta shrinks toward a fixed DEFAULT_VOL_VIX_BETA).
def _shrink_beta(
    beta: float | None,
    n_obs: int,
    *,
    prior: float = SHRINKAGE_PRIOR,
) -> tuple[float | None, bool]:
    """Legacy shrinkage helper: blend ``beta`` toward ``prior``.

    Kept for ``vol_vix_beta`` and external callers; new factor-panel
    code uses :func:`_shrink_beta_to_sector` with AR(1)-adjusted
    ``n_eff``.
    """
    if beta is None:
        return None, False
    if n_obs >= MIN_OBS_FOR_TRUST:
        return beta, False
    if n_obs <= 0:
        return prior, True
    w = n_obs / MIN_OBS_FOR_TRUST
    return prior * (1.0 - w) + beta * w, True


# ---------------------------------------------------------------------------
# Sector-mean prior
# ---------------------------------------------------------------------------


def _resolve_sector(symbol: str, sectors_override: dict[str, str] | None) -> str:
    """Sector lookup that prefers caller-supplied map, falls back to override."""
    sym = symbol.upper()
    if sectors_override and sym in sectors_override:
        sec = (sectors_override[sym] or "").strip().lower()
        if sec:
            return sec
    sec = OVERRIDE_SECTOR_MAP.get(sym)
    return sec if sec else "other"


def _curated_or_default_prior(symbol: str, sector: str) -> float:
    """Tier-3 prior when a sector has fewer than ``SECTOR_MIN_COMPUTED_NAMES``."""
    curated = BETA_TO_SPY.get(symbol.upper())
    if curated is not None:
        return float(curated)
    return DEFAULT_BROAD_INDEX_BETA if sector == "broad" else DEFAULT_SINGLE_NAME_BETA


def _compute_sector_means(
    pass1: dict[str, "BetaResult"],
) -> dict[str, dict[str, float]]:
    """Build per-sector mean betas from pass-1 OLS results.

    Only names with ``provenance == "computed"`` (i.e. n_obs >=
    MIN_OBS_FOR_TRUST) contribute. Sectors with fewer than
    ``SECTOR_MIN_COMPUTED_NAMES`` reliable names are omitted so the
    caller falls back to curated / default priors instead of using a
    noisy sector mean.
    """
    bucket: dict[str, dict[str, list[float]]] = {}
    for res in pass1.values():
        if res.provenance != "computed":
            continue
        sec = (res.sector or "other").lower()
        slot = bucket.setdefault(sec, {"spy": [], "ndx": [], "rut": []})
        if res.beta_to_spy is not None:
            slot["spy"].append(float(res.beta_to_spy))
        if res.beta_to_ndx is not None:
            slot["ndx"].append(float(res.beta_to_ndx))
        if res.beta_to_rut is not None:
            slot["rut"].append(float(res.beta_to_rut))

    out: dict[str, dict[str, float]] = {}
    for sec, slot in bucket.items():
        spy_vals = slot["spy"]
        if len(spy_vals) < SECTOR_MIN_COMPUTED_NAMES:
            continue
        means: dict[str, float] = {"spy": float(np.median(spy_vals))}
        if len(slot["ndx"]) >= SECTOR_MIN_COMPUTED_NAMES:
            means["ndx"] = float(np.median(slot["ndx"]))
        if len(slot["rut"]) >= SECTOR_MIN_COMPUTED_NAMES:
            means["rut"] = float(np.median(slot["rut"]))
        out[sec] = means
    return out


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def compute_betas(
    underlyings: Iterable[str],
    *,
    window_days: int = DEFAULT_WINDOW_DAYS,
    cache_dir: Path | None = None,
    yf_module: Any | None = None,
    refresh_max_age_hours: float = CACHE_MAX_AGE_HOURS,
    sectors: dict[str, str] | None = None,
    fetch_fn: Any | None = None,
) -> dict[str, BetaResult]:
    """Compute beta-to-{SPY,QQQ,IWM} for each ``underlying``.

    Two-pass shrinkage:

        1. OLS over the trailing ``window_days`` log returns for each
           underlying against each index. Names with at least
           ``MIN_OBS_FOR_TRUST`` paired observations are marked
           ``computed``.
        2. Per-sector median of pass-1 ``computed`` betas becomes the
           prior for pass-2 shrinkage of every name in that sector. A
           sector needs at least ``SECTOR_MIN_COMPUTED_NAMES`` reliable
           names to be used as a prior; otherwise the caller falls
           back to the curated ``BETA_TO_SPY`` value (or
           DEFAULT_SINGLE_NAME_BETA / DEFAULT_BROAD_INDEX_BETA).

    Falls back to curated map / default constants when no price data
    is available.

    Parameters
    ----------
    underlyings:
        Iterable of underlying tickers.
    sectors:
        Optional ``{symbol: sector}`` map (e.g. from
        ``risk_dashboard.sector_loader.batch_resolve``). When omitted
        we fall back to ``OVERRIDE_SECTOR_MAP``.
    fetch_fn:
        Override the data-fetch callable for tests. Must match the
        signature of :func:`_fetch_closes`.
    """
    cache_dir = cache_dir or CACHE_DIR_DEFAULT
    symbols = sorted({str(u).strip().upper() for u in underlyings if str(u).strip()})
    if not symbols:
        return {}

    universe = symbols + list(INDEX_TICKERS.values())
    fetcher = fetch_fn or _fetch_closes
    closes = fetcher(
        universe,
        window_days=window_days,
        cache_dir=cache_dir,
        refresh_max_age_hours=refresh_max_age_hours,
        yf_module=yf_module,
    )

    spy_ret = _log_returns(closes.get("SPY"), window_days)
    qqq_ret = _log_returns(closes.get("QQQ"), window_days)
    iwm_ret = _log_returns(closes.get("IWM"), window_days)

    # ----- Pass 1: raw OLS -----
    pass1: dict[str, BetaResult] = {}
    for sym in symbols:
        res = BetaResult(
            underlying=sym,
            cache_age_hours=_cache_age_hours(cache_dir, sym),
            sector=_resolve_sector(sym, sectors),
        )
        y = _log_returns(closes.get(sym), window_days)

        if not y.empty and not spy_ret.empty:
            b_spy, se_spy, n_spy, r2_spy = _ols_beta(y, spy_ret)
            res.beta_to_spy = b_spy
            res.beta_to_spy_raw = b_spy
            res.beta_se = se_spy
            res.n_obs = n_spy
            res.r2 = r2_spy
            _, n_eff = _ar1_n_eff(y.tail(n_spy).to_numpy(dtype=float))
            res.n_eff = n_eff
            if n_spy >= MIN_OBS_FOR_TRUST:
                res.provenance = "computed"
            elif n_spy > 0:
                res.provenance = "shrunk"

        if not y.empty and not qqq_ret.empty:
            b, _, _, _ = _ols_beta(y, qqq_ret)
            res.beta_to_ndx = b
            res.beta_to_ndx_raw = b
        if not y.empty and not iwm_ret.empty:
            b, _, _, _ = _ols_beta(y, iwm_ret)
            res.beta_to_rut = b
            res.beta_to_rut_raw = b

        # 60d regime vol -- captured independent of the 252d window so
        # the slide panel's regime overlay can use a current vol point
        # while OLS uses a wider sample for stable beta.
        regime_returns = _log_returns(closes.get(sym), REGIME_VOL_WINDOW_DAYS)
        if not regime_returns.empty and len(regime_returns) > 1:
            std_d = float(regime_returns.std(ddof=1))
            res.regime_vol_pct = std_d * math.sqrt(TRADING_DAYS_PER_YEAR) * 100.0

        pass1[sym] = res

    # ----- Build sector-mean priors from reliable pass-1 results -----
    sector_means = _compute_sector_means(pass1)

    # ----- Pass 2: shrink + fill fallbacks -----
    results: dict[str, BetaResult] = {}
    for sym, res in pass1.items():
        sec = res.sector or "other"
        prior_pack = sector_means.get(sec)

        if res.beta_to_spy is not None and res.n_eff > 0:
            # Pick the prior: sector mean first, then curated map, then default.
            if prior_pack and "spy" in prior_pack:
                prior_spy = prior_pack["spy"]
                res.prior_source = "sector_mean"
            else:
                prior_spy = _curated_or_default_prior(sym, sec)
                res.prior_source = (
                    "curated" if sym in BETA_TO_SPY else "default"
                )
            res.prior_used_spy = prior_spy
            shrunk, w, applied = _shrink_beta_to_sector(
                res.beta_to_spy, res.n_eff, prior=prior_spy
            )
            res.beta_to_spy = shrunk
            res.shrinkage_weight = w
            res.shrinkage_applied = applied
            if applied and res.provenance == "computed":
                res.provenance = "shrunk"

            if res.beta_to_ndx is not None:
                ndx_prior = (
                    prior_pack["ndx"]
                    if prior_pack and "ndx" in prior_pack
                    else prior_spy * 0.90
                )
                shrunk_ndx, _, _ = _shrink_beta_to_sector(
                    res.beta_to_ndx, res.n_eff, prior=ndx_prior
                )
                res.beta_to_ndx = shrunk_ndx
            if res.beta_to_rut is not None:
                rut_prior = (
                    prior_pack["rut"]
                    if prior_pack and "rut" in prior_pack
                    else prior_spy * 0.85
                )
                shrunk_rut, _, _ = _shrink_beta_to_sector(
                    res.beta_to_rut, res.n_eff, prior=rut_prior
                )
                res.beta_to_rut = shrunk_rut
        else:
            # No price data -- pure fallback.
            curated = BETA_TO_SPY.get(sym)
            if curated is not None:
                res.beta_to_spy = float(curated)
                res.provenance = "curated_fallback"
                res.prior_source = "curated"
            else:
                res.beta_to_spy = (
                    DEFAULT_BROAD_INDEX_BETA if sec == "broad" else DEFAULT_SINGLE_NAME_BETA
                )
                res.provenance = "default_fallback"
                res.prior_source = "default"
            res.prior_used_spy = res.beta_to_spy

        # Coarse multi-index fillers when paired returns missing.
        if res.beta_to_ndx is None and res.beta_to_spy is not None:
            res.beta_to_ndx = res.beta_to_spy * 0.90
        if res.beta_to_rut is None and res.beta_to_spy is not None:
            res.beta_to_rut = res.beta_to_spy * 0.85

        results[sym] = res

    return results


# ---------------------------------------------------------------------------
# Summary cache persistence
# ---------------------------------------------------------------------------


def write_summary_cache(
    results: dict[str, BetaResult],
    *,
    snapshot_date: str,
    path: Path | None = None,
    sector_means: dict[str, dict[str, float]] | None = None,
) -> None:
    """Persist BetaResults + sector means to ``beta_summary.json``."""
    path = path or SUMMARY_CACHE_PATH_DEFAULT
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "snapshot_date": snapshot_date,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "window_days": DEFAULT_WINDOW_DAYS,
        "regime_vol_window_days": REGIME_VOL_WINDOW_DAYS,
        "min_obs_for_trust": MIN_OBS_FOR_TRUST,
        "k_base": BETA_K_BASE,
        "sector_means": sector_means or {},
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
