"""Vol→VIX elasticity estimator (v3): EWMA log-log OLS.

Regresses log EWMA realized vol on log VIX level with product-class
priors and multiplicative σ shock mapping.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .beta_loader import (
    CACHE_DIR_DEFAULT,
    MIN_OBS_FOR_TRUST,
    TRADING_DAYS_PER_YEAR,
    _fetch_yfinance_closes,
)
from .factor_map import OVERRIDE_SECTOR_MAP
from .vol_vix_beta import VIX_TICKER, MIN_SIGMA, vix_to_decimal

ESTIMATOR_VERSION = "v3_log_elasticity"
EWMA_LAMBDA_DEFAULT = 0.94
HISTORY_DAYS_DEFAULT = 504
MIN_HISTORY_FOR_COMPUTE = 252
SHRINK_K_DEFAULT = 30.0
BETA_VOL_VIX_MIN = 0.1
BETA_VOL_VIX_MAX = 2.5
VIX_REGIME_SPLIT = 20.0

PRODUCT_CLASS_PRIORS: dict[str, float] = {
    "volatility_etp": 1.50,
    "single_stock_yield_boost": 1.20,
    "income_yieldboost": 1.20,
    "covered_call_1x": 1.10,
    "scraped_income": 1.10,
    "letf_long": 1.10,
    "letf_inverse": 1.10,
    "broad": 1.00,
}

SECTOR_PRIORS: dict[str, float] = {
    "semis": 1.10,
    "software": 1.10,
    "tech": 1.10,
    "growth": 1.10,
    "consumer": 1.00,
    "healthcare": 0.90,
    "financials": 0.90,
    "industrials": 0.90,
    "energy": 1.00,
    "defensives": 0.60,
    "staples": 0.60,
    "utilities": 0.60,
    "gold": 0.40,
    "commodity": 0.40,
    "bonds": 0.30,
    "crypto": 1.30,
    "broad": 1.00,
    "other": 0.85,
}

DEFAULT_VOL_VIX_BETA = 0.85


@dataclass
class VolVixBetaResultV3:
    underlying: str
    beta_vol_vix: float | None = None
    beta_vol_vix_low: float | None = None
    beta_vol_vix_high: float | None = None
    beta_se: float | None = None
    alpha: float | None = None
    n_obs: int = 0
    r2: float | None = None
    beta_prior: float | None = None
    prior_source: str = "default"
    provenance: str = "default"
    shrinkage_applied: bool = False
    estimator_version: str = ESTIMATOR_VERSION
    sigma_ewma: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "underlying": self.underlying,
            "beta_vol_vix": self.beta_vol_vix,
            "beta_vol_vix_low": self.beta_vol_vix_low,
            "beta_vol_vix_high": self.beta_vol_vix_high,
            "beta_se": self.beta_se,
            "alpha": self.alpha,
            "n_obs": self.n_obs,
            "r2": self.r2,
            "beta_prior": self.beta_prior,
            "prior_source": self.prior_source,
            "provenance": self.provenance,
            "shrinkage_applied": self.shrinkage_applied,
            "estimator_version": self.estimator_version,
            "sigma_ewma": self.sigma_ewma,
        }


def ewma_realized_vol(close: pd.Series, *, lam: float = EWMA_LAMBDA_DEFAULT) -> pd.Series:
    """EWMA annualized vol from daily log returns."""
    s = pd.to_numeric(close, errors="coerce").dropna().sort_index()
    if len(s) < 5:
        return pd.Series(dtype=float)
    log_r = np.log(s / s.shift(1)).replace([np.inf, -np.inf], np.nan).dropna()
    var = log_r.pow(2).ewm(alpha=1.0 - lam, adjust=False).mean()
    return np.sqrt(var * TRADING_DAYS_PER_YEAR)


def _resolve_product_class_prior(product_class: str, sector: str, sym: str) -> tuple[float, str]:
    pc = (product_class or "").strip().lower()
    if pc in PRODUCT_CLASS_PRIORS:
        return PRODUCT_CLASS_PRIORS[pc], "product_class"
    sec = (sector or OVERRIDE_SECTOR_MAP.get(sym.upper()) or "other").strip().lower()
    if sec in SECTOR_PRIORS:
        return SECTOR_PRIORS[sec], "sector"
    return DEFAULT_VOL_VIX_BETA, "default_prior"


def _clip_beta(beta: float) -> float:
    return float(max(BETA_VOL_VIX_MIN, min(BETA_VOL_VIX_MAX, beta)))


def _ewma_weights(n: int, lam: float) -> np.ndarray:
    ages = np.arange(n - 1, -1, -1, dtype=float)
    w = np.power(lam, ages)
    return w / w.sum()


def _weighted_log_log_ols(
    log_sigma: np.ndarray,
    log_vix: np.ndarray,
    weights: np.ndarray,
) -> tuple[float | None, float | None, float | None, int, float | None]:
    mask = np.isfinite(log_sigma) & np.isfinite(log_vix) & (log_sigma > -20) & (log_vix > -20)
    if mask.sum() < 5:
        return None, None, None, int(mask.sum()), None
    y = log_sigma[mask]
    x = log_vix[mask]
    w = weights[mask]
    w = w / w.sum()
    x_mean = float(np.sum(w * x))
    y_mean = float(np.sum(w * y))
    x_c = x - x_mean
    y_c = y - y_mean
    var_x = float(np.sum(w * x_c * x_c))
    if var_x <= 1e-12:
        return None, None, None, len(y), None
    beta = float(np.sum(w * x_c * y_c) / var_x)
    alpha = y_mean - beta * x_mean
    y_hat = alpha + beta * x
    ss_res = float(np.sum(w * (y - y_hat) ** 2))
    ss_tot = float(np.sum(w * (y - y_mean) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else None
    n_eff = float(1.0 / np.sum(w * w)) if len(w) > 0 else float(len(w))
    resid_var = ss_res / max(1.0, n_eff - 2)
    beta_se = math.sqrt(resid_var / var_x) if var_x > 0 else None
    return beta, beta_se, alpha, len(y), r2


def _regime_split_ols(
    vol_s: pd.Series,
    vix_s: pd.Series,
    *,
    lam: float,
) -> tuple[float | None, float | None]:
    paired = pd.concat([vol_s, vix_s], axis=1, join="inner").dropna()
    paired.columns = ["sigma", "vix"]
    if len(paired) < MIN_OBS_FOR_TRUST:
        return None, None
    low = paired[paired["vix"] <= VIX_REGIME_SPLIT / 100.0]
    high = paired[paired["vix"] > VIX_REGIME_SPLIT / 100.0]
    out: list[float | None] = []
    for sub in (low, high):
        if len(sub) < 30:
            out.append(None)
            continue
        ls = np.log(sub["sigma"].to_numpy(dtype=float))
        lv = np.log(sub["vix"].to_numpy(dtype=float))
        w = _ewma_weights(len(ls), lam)
        b, _, _, _, _ = _weighted_log_log_ols(ls, lv, w)
        out.append(b)
    return out[0], out[1]


def shocked_sigma_multiplicative(
    sigma_base: float,
    *,
    beta_vol_vix: float,
    vix_current_pts: float,
    vix_new_pts: float,
) -> float:
    """σ_new = σ_base × (VIX_new / VIX_now)^β."""
    base = float(sigma_base)
    if not math.isfinite(base) or base <= 0:
        return MIN_SIGMA
    v_now = max(float(vix_current_pts), 1.0)
    v_new = max(float(vix_new_pts), 1.0)
    ratio = v_new / v_now
    if ratio <= 0:
        return MIN_SIGMA
    return max(MIN_SIGMA, base * (ratio ** float(beta_vol_vix)))


def compute_vol_vix_betas_v3(
    underlyings: list[str],
    *,
    cache_dir: Path | None = None,
    history_days: int = HISTORY_DAYS_DEFAULT,
    ewma_lambda: float = EWMA_LAMBDA_DEFAULT,
    shrink_k: float = SHRINK_K_DEFAULT,
    yf_module: Any | None = None,
    underlying_meta: Mapping[str, Mapping[str, Any]] | None = None,
    extra_tickers: list[str] | None = None,
) -> dict[str, Any]:
    """Return per-underlying vol elasticity betas plus VIX term structure."""
    symbols = sorted({str(u).strip().upper() for u in underlyings if str(u).strip()})
    cache_dir = cache_dir or CACHE_DIR_DEFAULT
    meta = underlying_meta or {}
    term_tickers = ["^VIX9D", "^VIX3M", "^VVIX"]
    universe = sorted(set(symbols + [VIX_TICKER] + term_tickers + (extra_tickers or [])))
    closes = _fetch_yfinance_closes(
        universe,
        window_days=max(history_days, MIN_HISTORY_FOR_COMPUTE) + 30,
        cache_dir=cache_dir,
        yf_module=yf_module,
    )

    vix_close = closes.get(VIX_TICKER)
    vix_current_pts: float | None = None
    vix_series = pd.Series(dtype=float)
    if vix_close is not None and not vix_close.empty:
        vix_current_pts = float(vix_close.iloc[-1])
        vix_series = vix_close.astype(float) / 100.0
    if vix_current_pts is None:
        vix_current_pts = 20.0
        vix_series = pd.Series([0.20], index=pd.DatetimeIndex([pd.Timestamp.today()]))

    term_structure: dict[str, Any] = {"vix_spot_pts": vix_current_pts}
    for tk, key in (("^VIX9D", "vix9d_pts"), ("^VIX3M", "vix3m_pts"), ("^VVIX", "vvix_pts")):
        s = closes.get(tk)
        if s is not None and not s.empty:
            term_structure[key] = float(s.iloc[-1])
    v9 = term_structure.get("vix9d_pts")
    v3 = term_structure.get("vix3m_pts")
    if v9 and vix_current_pts:
        term_structure["vix9d_over_spot"] = v9 / vix_current_pts
    if v3 and vix_current_pts:
        term_structure["vix3m_over_spot"] = v3 / vix_current_pts
        term_structure["term_structure"] = (
            "backwardation" if v3 > vix_current_pts else "contango"
        )

    out: dict[str, VolVixBetaResultV3] = {}
    n_computed = 0
    n_shrunk = 0
    for sym in symbols:
        res = VolVixBetaResultV3(underlying=sym)
        m = meta.get(sym) or {}
        product_class = str(m.get("product_class") or m.get("instrument_class") or "")
        sector = str(m.get("sector") or "")
        prior, prior_source = _resolve_product_class_prior(product_class, sector, sym)
        res.beta_prior = prior
        res.prior_source = prior_source

        close = closes.get(sym)
        if close is None or close.empty or vix_series.empty:
            res.beta_vol_vix = _clip_beta(prior)
            res.provenance = "default"
            out[sym] = res
            continue

        vol_s = ewma_realized_vol(close, lam=ewma_lambda).tail(history_days)
        vix_aligned = vix_series.reindex(vol_s.index).ffill()
        paired = pd.concat([vol_s, vix_aligned], axis=1, join="inner").dropna()
        if len(paired) < MIN_OBS_FOR_TRUST:
            res.beta_vol_vix = _clip_beta(prior)
            res.provenance = "default_insufficient_history"
            out[sym] = res
            continue

        res.sigma_ewma = float(paired.iloc[-1, 0])
        log_sigma = np.log(paired.iloc[:, 0].to_numpy(dtype=float))
        log_vix = np.log(paired.iloc[:, 1].to_numpy(dtype=float))
        weights = _ewma_weights(len(log_sigma), ewma_lambda)
        beta_ols, beta_se, alpha, n_obs, r2 = _weighted_log_log_ols(log_sigma, log_vix, weights)
        res.n_obs = n_obs
        res.beta_se = beta_se
        res.alpha = alpha
        res.r2 = r2

        beta_low, beta_high = _regime_split_ols(vol_s, vix_aligned, lam=ewma_lambda)
        res.beta_vol_vix_low = _clip_beta(beta_low) if beta_low is not None else None
        res.beta_vol_vix_high = _clip_beta(beta_high) if beta_high is not None else None

        if beta_ols is None or n_obs < MIN_OBS_FOR_TRUST:
            res.beta_vol_vix = _clip_beta(prior)
            res.provenance = "default_insufficient_history"
            out[sym] = res
            continue

        if beta_ols < 0:
            beta_ols = prior
            res.provenance = "computed_negative_snapped"

        n_eff = float(1.0 / np.sum(weights * weights))
        w = n_eff / (n_eff + shrink_k) if (n_eff + shrink_k) > 0 else 0.0
        beta_final = w * float(beta_ols) + (1.0 - w) * prior
        res.shrinkage_applied = w < 0.999
        res.beta_vol_vix = _clip_beta(beta_final)

        if res.provenance != "computed_negative_snapped":
            res.provenance = "computed_shrunk" if res.shrinkage_applied else "computed"
        n_computed += 1
        if res.shrinkage_applied:
            n_shrunk += 1
        out[sym] = res

    return {
        "estimator_version": ESTIMATOR_VERSION,
        "vix_current": vix_to_decimal(vix_current_pts),
        "vix_current_pts": vix_current_pts,
        "betas": out,
        "n_computed": n_computed,
        "n_shrunk": n_shrunk,
        "n_total": len(out),
        "vix_provenance": "live" if vix_series.size > 1 else "fallback_20",
        "term_structure": term_structure,
        "ewma_lambda": ewma_lambda,
        "history_days": history_days,
    }
