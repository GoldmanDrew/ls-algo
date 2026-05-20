"""Historical beta of underlying realized vol to VIX (level regression).

Used by slide-risk VIX shocks: map a VIX point change to a revised
forecast σ per name, then re-run the scenario decay engine at SPX 0%.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .beta_loader import (
    CACHE_DIR_DEFAULT,
    MIN_OBS_FOR_TRUST,
    TRADING_DAYS_PER_YEAR,
    _fetch_yfinance_closes,
    _ols_beta,
    _shrink_beta,
)

VIX_TICKER = "^VIX"
VOL_ROLLING_DAYS = 20
VOL_VIX_HISTORY_DAYS = 252
DEFAULT_VOL_VIX_BETA = 0.75
MIN_SIGMA = 0.05


@dataclass
class VolVixBetaResult:
    underlying: str
    beta_vol_vix: float | None = None
    alpha: float | None = None
    n_obs: int = 0
    r2: float | None = None
    provenance: str = "default"
    shrinkage_applied: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "underlying": self.underlying,
            "beta_vol_vix": self.beta_vol_vix,
            "alpha": self.alpha,
            "n_obs": self.n_obs,
            "r2": self.r2,
            "provenance": self.provenance,
            "shrinkage_applied": self.shrinkage_applied,
        }


def vix_to_decimal(vix_level: float) -> float:
    """Convert VIX index level (e.g. 20.5) to decimal vol (0.205)."""
    return float(vix_level) / 100.0


def rolling_annualized_vol(close: pd.Series, window: int = VOL_ROLLING_DAYS) -> pd.Series:
    s = pd.to_numeric(close, errors="coerce").dropna().sort_index()
    if len(s) < window + 2:
        return pd.Series(dtype=float)
    log_r = np.log(s / s.shift(1)).replace([np.inf, -np.inf], np.nan).dropna()
    return log_r.rolling(window).std() * math.sqrt(TRADING_DAYS_PER_YEAR)


def shocked_sigma_annual(
    sigma_base: float,
    *,
    beta_vol_vix: float,
    vix_current_decimal: float,
    vix_shock_pts: float,
) -> float:
    """Map a VIX point shock to a revised annualized σ."""
    base = float(sigma_base)
    if not math.isfinite(base) or base <= 0:
        return MIN_SIGMA
    delta_vix = float(vix_shock_pts) / 100.0
    sigma = base + float(beta_vol_vix) * delta_vix
    return max(MIN_SIGMA, sigma)


def _level_ols_with_alpha(
    y: pd.Series,
    x: pd.Series,
) -> tuple[float | None, float | None, int, float | None]:
    beta, _se, n_obs, r2 = _ols_beta(y, x)
    if beta is None or n_obs < 5:
        return None, None, n_obs, r2
    paired = pd.concat([y, x], axis=1, join="inner").dropna()
    if paired.empty:
        return None, None, n_obs, r2
    yv = paired.iloc[:, 0].to_numpy(dtype=float)
    xv = paired.iloc[:, 1].to_numpy(dtype=float)
    alpha = float(yv.mean() - beta * xv.mean())
    return float(beta), alpha, n_obs, r2


def compute_vol_vix_betas(
    underlyings: list[str],
    *,
    cache_dir: Path | None = None,
    history_days: int = VOL_VIX_HISTORY_DAYS,
    yf_module: Any | None = None,
) -> dict[str, Any]:
    """Return per-underlying vol→VIX betas plus current VIX level."""
    symbols = sorted({str(u).strip().upper() for u in underlyings if str(u).strip()})
    cache_dir = cache_dir or CACHE_DIR_DEFAULT
    universe = symbols + [VIX_TICKER]
    closes = _fetch_yfinance_closes(
        universe,
        window_days=history_days,
        cache_dir=cache_dir,
        yf_module=yf_module,
    )
    vix_close = closes.get(VIX_TICKER)
    vix_current_decimal: float | None = None
    vix_series_decimal = pd.Series(dtype=float)
    if vix_close is not None and not vix_close.empty:
        vix_current_decimal = vix_to_decimal(float(vix_close.iloc[-1]))
        vix_series_decimal = vix_close.astype(float) / 100.0
    if vix_current_decimal is None:
        vix_current_decimal = 0.20

    out: dict[str, VolVixBetaResult] = {}
    for sym in symbols:
        res = VolVixBetaResult(underlying=sym)
        close = closes.get(sym)
        if close is None or close.empty or vix_series_decimal.empty:
            res.beta_vol_vix = DEFAULT_VOL_VIX_BETA
            res.provenance = "default"
            out[sym] = res
            continue
        vol_s = rolling_annualized_vol(close)
        paired = pd.concat([vol_s, vix_series_decimal], axis=1, join="inner").dropna()
        paired = paired.tail(history_days)
        if len(paired) < 30:
            res.beta_vol_vix = DEFAULT_VOL_VIX_BETA
            res.provenance = "default_insufficient_history"
            out[sym] = res
            continue
        beta, alpha, n_obs, r2 = _level_ols_with_alpha(paired.iloc[:, 0], paired.iloc[:, 1])
        beta_shrunk, shrunk = _shrink_beta(beta, n_obs, prior=DEFAULT_VOL_VIX_BETA)
        res.beta_vol_vix = beta_shrunk if beta_shrunk is not None else DEFAULT_VOL_VIX_BETA
        res.alpha = alpha
        res.n_obs = n_obs
        res.r2 = r2
        res.shrinkage_applied = shrunk
        res.provenance = "computed" if n_obs >= MIN_OBS_FOR_TRUST else "computed_shrunk"
        out[sym] = res

    return {
        "vix_current": vix_current_decimal,
        "vix_current_pts": vix_current_decimal * 100.0,
        "betas": out,
        "n_computed": sum(1 for r in out.values() if r.provenance.startswith("computed")),
        "n_total": len(out),
        "vix_provenance": "live" if vix_series_decimal.size else "fallback_20",
    }


def leg_sigma_for_vix_shock(
    leg: dict[str, Any],
    *,
    underlying: str,
    underlying_sigma: float | None,
    vol_vix_pack: dict[str, Any],
    vix_shock_pts: float,
) -> float | None:
    """Resolve scenario σ for a leg under a VIX shock."""
    sigma_base, _ = _resolve_sigma_for_leg(leg, underlying_sigma=underlying_sigma)
    if sigma_base is None:
        return None
    betas: dict[str, VolVixBetaResult] = vol_vix_pack.get("betas") or {}
    vix_dec = vol_vix_pack.get("vix_current")
    if vix_dec is None:
        vix_dec = 0.20
    beta_res = betas.get(underlying.upper())
    beta = (
        float(beta_res.beta_vol_vix)
        if beta_res is not None and beta_res.beta_vol_vix is not None
        else DEFAULT_VOL_VIX_BETA
    )
    return shocked_sigma_annual(
        sigma_base,
        beta_vol_vix=beta,
        vix_current_decimal=float(vix_dec),
        vix_shock_pts=float(vix_shock_pts),
    )


def _resolve_sigma_for_leg(
    leg: dict[str, Any],
    *,
    underlying_sigma: float | None,
) -> tuple[float | None, str]:
    from .scenario_engine import resolve_sigma_annual

    return resolve_sigma_annual(leg, underlying_sigma=underlying_sigma)
