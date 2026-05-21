"""Historical beta of underlying realized vol to VIX (diff OLS, v2).

Maps a VIX point shock to a revised forecast σ per name for slide-risk
VIX strips (SPX 0%, 12M decay matrix).

Estimator (v2): first differences of 20d realized vol vs first differences
of VIX level (decimal), OLS over up to 252 trading days, shrunk toward a
product-class / sector-mean prior using AR(1)-adjusted effective sample size
(same shape as ``daily_screener.compute_beta_shrunk``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from daily_screener import BETA_AR1_FLOOR_RATIO, BETA_SHRINK_K_BASE, _ar1_n_eff

from .beta_loader import (
    CACHE_DIR_DEFAULT,
    MIN_OBS_FOR_TRUST,
    TRADING_DAYS_PER_YEAR,
    _fetch_yfinance_closes,
    _ols_beta,
)
from .factor_map import OVERRIDE_SECTOR_MAP

VIX_TICKER = "^VIX"
VOL_ROLLING_DAYS = 20
VOL_VIX_HISTORY_DAYS = 252
MIN_HISTORY_FOR_COMPUTE = 120
DEFAULT_VOL_VIX_BETA = 0.75
MIN_SIGMA = 0.05
ESTIMATOR_VERSION = "v2_diff_ols"
BETA_VOL_VIX_MIN = 0.0
BETA_VOL_VIX_MAX = 2.0
NEGATIVE_BETA_OLS_THRESHOLD = 0.2

PRIOR_VOLATILITY_ETP = 1.0
PRIOR_BROAD = 0.5
PRIOR_DEFAULT = DEFAULT_VOL_VIX_BETA
SECTOR_MEAN_MIN_NAMES = 5


@dataclass
class VolVixBetaResult:
    underlying: str
    beta_vol_vix: float | None = None
    beta_se: float | None = None
    alpha: float | None = None
    n_obs: int = 0
    n_eff: int = 0
    r2: float | None = None
    residual_std: float | None = None
    rho_ar1: float | None = None
    beta_prior: float | None = None
    provenance: str = "default"
    shrinkage_applied: bool = False
    estimator_version: str = ESTIMATOR_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "underlying": self.underlying,
            "beta_vol_vix": self.beta_vol_vix,
            "beta_se": self.beta_se,
            "alpha": self.alpha,
            "n_obs": self.n_obs,
            "n_eff": self.n_eff,
            "r2": self.r2,
            "residual_std": self.residual_std,
            "rho_ar1": self.rho_ar1,
            "beta_prior": self.beta_prior,
            "provenance": self.provenance,
            "shrinkage_applied": self.shrinkage_applied,
            "estimator_version": self.estimator_version,
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
    """Map a VIX point shock to a revised annualized σ (additive)."""
    del vix_current_decimal  # reserved for future multiplicative mode
    base = float(sigma_base)
    if not math.isfinite(base) or base <= 0:
        return MIN_SIGMA
    delta_vix = float(vix_shock_pts) / 100.0
    sigma = base + float(beta_vol_vix) * delta_vix
    return max(MIN_SIGMA, sigma)


def _product_class_prior(product_class: str) -> float:
    cls = (product_class or "").strip().lower()
    if cls == "volatility_etp":
        return PRIOR_VOLATILITY_ETP
    if cls == "broad":
        return PRIOR_BROAD
    return PRIOR_DEFAULT


def _resolve_beta_prior(
    sym: str,
    *,
    product_class: str,
    sector: str,
    sector_means: dict[str, float],
) -> tuple[float, str]:
    sec = (sector or OVERRIDE_SECTOR_MAP.get(sym.upper()) or "other").strip().lower()
    if sec in sector_means:
        return sector_means[sec], "sector_mean"
    pc = (product_class or "").strip().lower()
    if pc:
        return _product_class_prior(pc), "product_class"
    if sec == "broad":
        return PRIOR_BROAD, "sector_broad"
    return PRIOR_DEFAULT, "default_prior"


def _shrink_vol_vix_beta(
    beta_ols: float | None,
    n_obs: int,
    delta_sigma: np.ndarray,
    *,
    beta_prior: float,
) -> tuple[float | None, bool, int, float]:
    """AR(1) shrinkage toward ``beta_prior`` (``compute_beta_shrunk`` shape)."""
    if beta_ols is None or n_obs < 2:
        return beta_prior, True, n_obs, 0.0
    rho, n_eff = _ar1_n_eff(delta_sigma)
    k = float(BETA_SHRINK_K_BASE) * max(1.0, float(beta_prior) ** 2)
    w = float(n_eff) / float(n_eff + k) if (n_eff + k) > 0 else 0.0
    beta = w * float(beta_ols) + (1.0 - w) * float(beta_prior)
    shrunk = w < 0.999
    return beta, shrunk, n_eff, rho


def _clip_beta(beta: float) -> float:
    return float(max(BETA_VOL_VIX_MIN, min(BETA_VOL_VIX_MAX, beta)))


def _diff_ols_with_stats(
    vol_s: pd.Series,
    vix_decimal: pd.Series,
    *,
    history_days: int,
) -> tuple[float | None, float | None, float | None, int, float | None, float | None, pd.Series]:
    paired = pd.concat([vol_s, vix_decimal], axis=1, join="inner").dropna()
    paired = paired.tail(history_days)
    if len(paired) < 5:
        empty = pd.Series(dtype=float)
        return None, None, None, len(paired), None, None, empty
    d_sigma = paired.iloc[:, 0].diff().dropna()
    d_vix = paired.iloc[:, 1].diff().dropna()
    aligned = pd.concat([d_sigma, d_vix], axis=1, join="inner").dropna()
    if len(aligned) < 5:
        return None, None, None, len(aligned), None, None, aligned.iloc[:, 0]

    y = aligned.iloc[:, 0]
    x = aligned.iloc[:, 1]
    beta, beta_se, n_obs, r2 = _ols_beta(y, x)
    if beta is None:
        return None, beta_se, None, n_obs, r2, None, y.to_numpy(dtype=float)

    yv = y.to_numpy(dtype=float)
    xv = x.to_numpy(dtype=float)
    alpha = float(yv.mean() - beta * xv.mean())
    resid = yv - (alpha + beta * xv)
    resid_std = float(np.std(resid, ddof=1)) if len(resid) > 1 else None
    return float(beta), beta_se, alpha, n_obs, r2, resid_std, y.to_numpy(dtype=float)


def _sector_mean_priors(
    ols_by_sym: dict[str, tuple[float, int]],
) -> dict[str, float]:
    buckets: dict[str, list[float]] = {}
    for sym, (beta, n_obs) in ols_by_sym.items():
        if beta is None or n_obs < MIN_OBS_FOR_TRUST:
            continue
        sec = OVERRIDE_SECTOR_MAP.get(sym.upper(), "other")
        buckets.setdefault(sec, []).append(float(beta))
    out: dict[str, float] = {}
    for sec, vals in buckets.items():
        if len(vals) >= SECTOR_MEAN_MIN_NAMES:
            out[sec] = float(np.mean(vals))
    return out


def compute_vol_vix_betas(
    underlyings: list[str],
    *,
    cache_dir: Path | None = None,
    history_days: int = VOL_VIX_HISTORY_DAYS,
    yf_module: Any | None = None,
    underlying_meta: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Return per-underlying vol→VIX betas plus current VIX level."""
    symbols = sorted({str(u).strip().upper() for u in underlyings if str(u).strip()})
    cache_dir = cache_dir or CACHE_DIR_DEFAULT
    meta = underlying_meta or {}
    universe = symbols + [VIX_TICKER]
    closes = _fetch_yfinance_closes(
        universe,
        window_days=max(history_days, MIN_HISTORY_FOR_COMPUTE) + VOL_ROLLING_DAYS + 10,
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

    # Pass 1: raw diff-OLS for sector-mean priors
    ols_pass: dict[str, tuple[float | None, int]] = {}
    raw: dict[str, dict[str, Any]] = {}
    for sym in symbols:
        close = closes.get(sym)
        if close is None or close.empty or vix_series_decimal.empty:
            ols_pass[sym] = (None, 0)
            continue
        vol_s = rolling_annualized_vol(close)
        beta, beta_se, alpha, n_obs, r2, resid_std, d_arr = _diff_ols_with_stats(
            vol_s, vix_series_decimal, history_days=history_days
        )
        ols_pass[sym] = (beta, n_obs)
        raw[sym] = {
            "beta": beta,
            "beta_se": beta_se,
            "alpha": alpha,
            "n_obs": n_obs,
            "r2": r2,
            "residual_std": resid_std,
            "d_arr": d_arr,
        }

    sector_means = _sector_mean_priors(
        {s: (b, n) for s, (b, n) in ols_pass.items() if b is not None}
    )

    out: dict[str, VolVixBetaResult] = {}
    n_computed = 0
    n_shrunk = 0
    for sym in symbols:
        res = VolVixBetaResult(underlying=sym)
        m = meta.get(sym) or {}
        product_class = str(m.get("product_class") or "")
        sector = str(m.get("sector") or "")

        if sym not in raw:
            prior, _ = _resolve_beta_prior(
                sym, product_class=product_class, sector=sector, sector_means=sector_means
            )
            res.beta_vol_vix = _clip_beta(prior)
            res.beta_prior = prior
            res.provenance = "default"
            out[sym] = res
            continue

        r = raw[sym]
        beta_ols = r["beta"]
        n_obs = int(r["n_obs"])
        d_arr = r["d_arr"]

        prior, prior_source = _resolve_beta_prior(
            sym, product_class=product_class, sector=sector, sector_means=sector_means
        )
        res.beta_prior = prior
        res.beta_se = r.get("beta_se")
        res.alpha = r.get("alpha")
        res.n_obs = n_obs
        res.r2 = r.get("r2")
        res.residual_std = r.get("residual_std")

        if n_obs < MIN_OBS_FOR_TRUST or beta_ols is None:
            res.beta_vol_vix = _clip_beta(prior)
            res.provenance = "default_insufficient_history"
            out[sym] = res
            continue

        if (
            beta_ols < 0
            and abs(beta_ols) > NEGATIVE_BETA_OLS_THRESHOLD
        ):
            beta_ols = prior
            res.provenance = "computed_negative_snapped"

        beta_final, shrunk, n_eff, rho = _shrink_vol_vix_beta(
            beta_ols,
            n_obs,
            d_arr if isinstance(d_arr, np.ndarray) else np.asarray(d_arr, dtype=float),
            beta_prior=prior,
        )
        res.n_eff = n_eff
        res.rho_ar1 = rho
        res.shrinkage_applied = shrunk
        res.beta_vol_vix = _clip_beta(beta_final if beta_final is not None else prior)

        if res.provenance == "computed_negative_snapped":
            n_computed += 1
        elif shrunk or prior_source == "sector_mean":
            res.provenance = "computed_shrunk"
            n_shrunk += 1
            n_computed += 1
        else:
            res.provenance = "computed"
            n_computed += 1

        out[sym] = res

    return {
        "estimator_version": ESTIMATOR_VERSION,
        "vix_current": vix_current_decimal,
        "vix_current_pts": vix_current_decimal * 100.0,
        "betas": out,
        "n_computed": n_computed,
        "n_shrunk": n_shrunk,
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
