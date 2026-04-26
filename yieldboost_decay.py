"""Distributional forecast of YieldBOOST income-strategy gross decay.

Why this module exists
----------------------
The dashboard's other product classes (LETF, inverse, volatility ETP) get an
"expected gross decay" forecast from ``decay_distribution.py`` — a HARQ-Log
anchored lognormal of the underlying's 1y integrated variance, mapped through
the Avellaneda-Zhang ``(β² − β)/2 · IV_T`` identity to per-ETF decay quantiles.

YieldBOOST income ETFs (AMYY, AZYY, BBYY, COYY, …) **break that identity**.
Their realized β is roughly 0.4–0.6 because the 2× LETF NAV is sleeved with a
weekly 95/88 SPX-style put-spread written on the underlying. The
Avellaneda-Zhang term collapses (cb ≈ −0.125 at β ≈ 0.5), so the LETF
distribution emits ~2–3% as the YB "expected decay", which dramatically
under-states the true NAV bleed driven by the put-spread premium.

This module replaces that mapping for YB rows with a put-spread Monte Carlo:

    1. Take the same HARQ-Log lognormal moments ``(μ_logIV_T, σ_logIV_T)`` we
       fit on the *underlying*. (Single-name underlyings: TSLA, NVDA, MSTR,
       SPX-by-proxy SPY, etc.)
    2. Sample ``log_iv_i ~ N(μ_logIV_T, σ²_logIV_T)`` for ``N_DRAWS`` draws.
       At horizon T=1y, σ_annual_i = sqrt(exp(log_iv_i)).
    3. For each σ_annual_i, compute the weekly put-spread loss ``L_w(σ_i)``
       under the same Black-Scholes-style closed form the dashboard uses on
       the Scenarios tab (``expectedPutSpreadLossWeekly`` in index.html), with
       underlying shock = 0 (pure decay forecast), horizon = 1y.
    4. Compound 52 weeks net of expense ratio:
            d_i = 1 − (1 − L_w(σ_i) − f_w)^52,   f_w = expense_ratio_annual / 52
    5. Take p10/p50/p90/mean of {d_i} → emits to the same
       ``expected_gross_decay_p10/p50/p90_annual`` columns LETF rows use, so
       downstream consumers (net-edge bootstrap, dashboard headline column)
       can route uniformly.

Fallback: if the underlying has too thin a panel for HARQ-Log, fall back to
a point estimate at a caller-supplied σ (typically the trailing 1y realized
vol of the underlying). Returns p10=p50=p90=mean in that case.

The model is intentionally i.i.d. across the 52 weekly steps — we do *not*
draw a fresh σ each week. The lognormal of σ at horizon T already encodes
the volatility-regime uncertainty. Stochastic-vol (multi-period σ) refinements
are left for a follow-up.

Constants are duplicated from the front-end's ``expectedPutSpreadLossWeekly``
to keep realized vs. expected in the same units. If you change strikes /
expense ratio there, change them here too.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Optional

import numpy as np
import pandas as pd

# Mirrors the front-end's expectedPutSpreadLossWeekly + estimateIncomeStyleScenarioReturn
# (see etf-dashboard/index.html). Keep these in sync.
_PUT_SPREAD_SHORT_STRIKE = 0.95
_PUT_SPREAD_LONG_STRIKE = 0.88
_PUT_SPREAD_LEVERAGE = 2.0
_DEFAULT_EXPENSE_RATIO_ANNUAL = 0.0099
_WEEKS_PER_YEAR = 52
_WEEKLY_TAU_YEARS = 1.0 / _WEEKS_PER_YEAR

# Monte Carlo size. 4096 σ-draws gives p10/p90 standard error well under 0.5%
# absolute decay for plausible σ_logIV ranges (sanity-checked in tests).
_DEFAULT_N_DRAWS = 4096
_DEFAULT_SEED = 17

# Numerical clamps.
_SIGMA_FLOOR = 1e-4                  # avoid σ=0 in the BS formula
_SIGMA_CEIL = 5.0                    # 500% σ — well past any sane regime
_LOSS_CEIL = (
    _PUT_SPREAD_SHORT_STRIKE - _PUT_SPREAD_LONG_STRIKE
)                                    # 0.07; max possible put-spread loss


# ─── normal CDF ──────────────────────────────────────────────────────────

def _normal_cdf(x: np.ndarray) -> np.ndarray:
    """Vectorised standard-normal CDF using ``erf``."""
    return 0.5 * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))


# ─── put-spread weekly loss ──────────────────────────────────────────────

def _weekly_put_spread_loss(
    sigma_annual: np.ndarray,
    *,
    underlying_return: float = 0.0,
    horizon_years: float = 1.0,
) -> np.ndarray:
    """Weekly put-spread loss L_w(σ) under a 2× lognormal underlying.

    Mirrors ``expectedPutSpreadLossWeekly`` in etf-dashboard/index.html.
    Returns an ``np.ndarray`` of weekly losses in [0, 0.07].
    """
    sigma = np.clip(np.asarray(sigma_annual, dtype=float), _SIGMA_FLOOR, _SIGMA_CEIL)
    u = float(underlying_return)
    t = float(horizon_years)
    if not (math.isfinite(u) and u > -0.9999 and math.isfinite(t) and t > 0):
        return np.zeros_like(sigma)

    tau = _WEEKLY_TAU_YEARS
    mu_annual = math.log1p(u) / t
    lev = _PUT_SPREAD_LEVERAGE
    m = (lev * mu_annual - lev * sigma * sigma) * tau
    s = lev * sigma * math.sqrt(tau)
    s = np.where(s > _SIGMA_FLOOR, s, _SIGMA_FLOOR)

    def _spread_put(k: float) -> np.ndarray:
        log_k = math.log(k)
        alpha = (log_k - m) / s
        beta = alpha - s
        forward = np.exp(m + 0.5 * s * s)
        return k * _normal_cdf(alpha) - forward * _normal_cdf(beta)

    loss = _spread_put(_PUT_SPREAD_SHORT_STRIKE) - _spread_put(_PUT_SPREAD_LONG_STRIKE)
    loss = np.where(np.isfinite(loss), loss, 0.0)
    return np.clip(loss, 0.0, _LOSS_CEIL)


def _annual_intrinsic_decay(
    sigma_annual: np.ndarray,
    *,
    expense_ratio_annual: float = _DEFAULT_EXPENSE_RATIO_ANNUAL,
    underlying_return: float = 0.0,
    horizon_years: float = 1.0,
) -> np.ndarray:
    """1y compounded NAV decay = 1 − (1 − L_w − f_w)^52, vectorised over σ."""
    weekly_loss = _weekly_put_spread_loss(
        sigma_annual,
        underlying_return=underlying_return,
        horizon_years=horizon_years,
    )
    f_w = max(0.0, float(expense_ratio_annual)) / _WEEKS_PER_YEAR
    q = np.clip(1.0 - weekly_loss - f_w, 1e-4, 1.5)
    return 1.0 - np.power(q, _WEEKS_PER_YEAR)


# ─── headline entry point ────────────────────────────────────────────────

def yieldboost_decay_distribution(
    *,
    mu_log_iv_annual: float,
    sigma_log_iv_annual: float,
    expense_ratio_annual: float = _DEFAULT_EXPENSE_RATIO_ANNUAL,
    n_draws: int = _DEFAULT_N_DRAWS,
    seed: int = _DEFAULT_SEED,
) -> Optional[Dict[str, float]]:
    """Sample annual intrinsic decay under HARQ-Log uncertainty in σ.

    ``mu_log_iv_annual`` and ``sigma_log_iv_annual`` are the moments of
    ``log(IV_T)`` at horizon T = 1y, as produced by
    ``decay_distribution._empirical_log_iv_moments`` /
    ``_harq_log_conditional_shift``. At T=1y, ``IV_T = σ²_annual``, so we
    convert with ``σ_annual = sqrt(exp(log_iv))``.

    Returns p10/p50/p90/mean of the simulated annual decay distribution.
    Returns ``None`` if inputs are not finite.
    """
    mu = float(mu_log_iv_annual)
    sigma = float(sigma_log_iv_annual)
    if not (math.isfinite(mu) and math.isfinite(sigma) and sigma >= 0.0):
        return None
    rng = np.random.default_rng(int(seed))
    z = rng.standard_normal(int(n_draws))
    log_iv = mu + max(0.0, sigma) * z
    sigma_annual = np.sqrt(np.exp(log_iv))
    decay = _annual_intrinsic_decay(
        sigma_annual,
        expense_ratio_annual=expense_ratio_annual,
        underlying_return=0.0,
        horizon_years=1.0,
    )
    return {
        "p10": float(round(np.percentile(decay, 10), 6)),
        "p50": float(round(np.percentile(decay, 50), 6)),
        "p90": float(round(np.percentile(decay, 90), 6)),
        "mean": float(round(np.mean(decay), 6)),
        "mu_log_iv": round(float(mu), 6),
        "sigma_log_iv": round(float(max(0.0, sigma)), 6),
        "n_obs": float(n_draws),
        "model": "yieldboost_put_spread",
    }


def yieldboost_decay_point_estimate(
    *,
    sigma_annual: float,
    expense_ratio_annual: float = _DEFAULT_EXPENSE_RATIO_ANNUAL,
) -> Optional[Dict[str, float]]:
    """Fallback: point estimate at a single σ (no HARQ-Log fit available).

    Used when the underlying has too thin a panel for the empirical
    lognormal anchor. Caller passes the trailing realised σ as a best
    guess; we emit p10=p50=p90=mean and tag the model accordingly.
    """
    sigma = float(sigma_annual) if sigma_annual is not None else float("nan")
    if not (math.isfinite(sigma) and sigma > 0):
        return None
    decay = float(
        _annual_intrinsic_decay(
            np.array([sigma]),
            expense_ratio_annual=expense_ratio_annual,
        )[0]
    )
    decay = round(decay, 6)
    mu_log_iv = math.log(max(sigma * sigma, 1e-12))
    return {
        "p10": decay,
        "p50": decay,
        "p90": decay,
        "mean": decay,
        "mu_log_iv": round(mu_log_iv, 6),
        "sigma_log_iv": 0.0,
        "n_obs": 1.0,
        "model": "yieldboost_put_spread_point",
    }


__all__ = (
    "yieldboost_decay_distribution",
    "yieldboost_decay_point_estimate",
)
