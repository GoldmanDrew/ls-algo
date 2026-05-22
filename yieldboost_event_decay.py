"""Event-aware YieldBOOST put-spread decay Monte Carlo.

Extends yieldboost_decay with jump-diffusion weeks when earnings/events fall
inside the horizon. See PLAN_event_devol_and_forward_straddle_decomposition.md.
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

from yieldboost_decay import (
    _DEFAULT_EXPENSE_RATIO_ANNUAL,
    _DEFAULT_N_DRAWS,
    _DEFAULT_SEED,
    _LOSS_CEIL,
    _PUT_SPREAD_LONG_STRIKE,
    _PUT_SPREAD_SHORT_STRIKE,
    _SIGMA_CEIL,
    _SIGMA_FLOOR,
    _WEEKS_PER_YEAR,
    _normal_cdf,
    _weekly_put_spread_loss,
)

_LEVERAGE = 2.0


def _put_spread_payoff(ratio: np.ndarray, k_long: float, k_short: float) -> np.ndarray:
    """Payoff of long lower / short higher put spread on 2x spot ratio."""
    return np.maximum(k_short - ratio, 0.0) - np.maximum(k_long - ratio, 0.0)


def event_aware_weekly_loss(
    sigma_base_annual: float,
    *,
    event_jump_underlying: float = 0.0,
    underlying_return: float = 0.0,
    horizon_years: float = 1.0 / _WEEKS_PER_YEAR,
    strike_long: float = _PUT_SPREAD_LONG_STRIKE,
    strike_short: float = _PUT_SPREAD_SHORT_STRIKE,
    n_draws: int = 2048,
    seed: int = _DEFAULT_SEED,
) -> float:
    """
    Expected weekly put-spread loss with optional one-day event jump on underlying.

    ``event_jump_underlying`` is a signed log-return shock (e.g. -0.06 for -6%).
    """
    sigma = float(np.clip(sigma_base_annual, _SIGMA_FLOOR, _SIGMA_CEIL))
    if event_jump_underlying != 0.0:
        rng = np.random.default_rng(seed)
        tau = 1.0 / _WEEKS_PER_YEAR
        mu_annual = math.log1p(float(underlying_return)) / float(horizon_years)
        m = (_LEVERAGE * mu_annual - _LEVERAGE * sigma * sigma) * tau
        s = _LEVERAGE * sigma * math.sqrt(tau)
        s = max(s, _SIGMA_FLOOR)
        z = rng.standard_normal(n_draws)
        ratio = np.exp(m + s * z)
        ratio *= np.exp(_LEVERAGE * float(event_jump_underlying))
        loss = _put_spread_payoff(ratio, strike_long, strike_short)
        return float(np.clip(np.mean(loss), 0.0, _LOSS_CEIL))

    arr = _weekly_put_spread_loss(
        np.asarray([sigma]),
        underlying_return=underlying_return,
        horizon_years=horizon_years,
    )
    return float(arr[0]) if arr.size else 0.0


def event_aware_decay_distribution(
    mu_log_iv_annual: float,
    sigma_log_iv_annual: float,
    *,
    week_has_event: Sequence[bool] | None = None,
    event_jump_pool: Sequence[float] | None = None,
    expense_ratio_annual: float = _DEFAULT_EXPENSE_RATIO_ANNUAL,
    strike_long: float = _PUT_SPREAD_LONG_STRIKE,
    strike_short: float = _PUT_SPREAD_SHORT_STRIKE,
    n_draws: int = _DEFAULT_N_DRAWS,
    seed: int = _DEFAULT_SEED,
) -> dict[str, Any] | None:
    """Annual gross decay quantiles with optional event jumps in selected weeks."""
    if not (math.isfinite(mu_log_iv_annual) and math.isfinite(sigma_log_iv_annual)):
        return None
    if sigma_log_iv_annual < 0:
        return None

    rng = np.random.default_rng(seed)
    log_iv = rng.normal(mu_log_iv_annual, sigma_log_iv_annual, size=n_draws)
    sigma_draws = np.sqrt(np.exp(log_iv))
    sigma_draws = np.clip(sigma_draws, _SIGMA_FLOOR, _SIGMA_CEIL)

    weeks = list(week_has_event or [False] * _WEEKS_PER_YEAR)
    if len(weeks) < _WEEKS_PER_YEAR:
        weeks = weeks + [False] * (_WEEKS_PER_YEAR - len(weeks))
    jumps = list(event_jump_pool or [])

    weekly_expense = max(0.0, expense_ratio_annual) / _WEEKS_PER_YEAR
    decays = np.empty(n_draws, dtype=float)
    for i, sigma in enumerate(sigma_draws):
        q = 1.0
        for w in range(_WEEKS_PER_YEAR):
            jump = 0.0
            if weeks[w] and jumps:
                jump = float(rng.choice(jumps))
            loss = event_aware_weekly_loss(
                float(sigma),
                event_jump_underlying=jump,
                strike_long=strike_long,
                strike_short=strike_short,
                n_draws=1,
                seed=seed + i * 17 + w,
            )
            q *= max(0.0001, 1.0 - loss - weekly_expense)
        decays[i] = 1.0 - q

    p10, p50, p90 = np.quantile(decays, [0.10, 0.50, 0.90])
    return {
        "p10": float(p10),
        "p50": float(p50),
        "p90": float(p90),
        "mean": float(np.mean(decays)),
        "mu_log_iv": float(mu_log_iv_annual),
        "sigma_log_iv": float(sigma_log_iv_annual),
        "event_aware": any(weeks),
    }


def fair_spread_mid_from_sigma(
    sigma_base_annual: float,
    *,
    event_jump_underlying: float = 0.0,
    strike_long: float = _PUT_SPREAD_LONG_STRIKE,
    strike_short: float = _PUT_SPREAD_SHORT_STRIKE,
) -> float | None:
    """Fair put-spread mid (fraction of spot) for one week."""
    if not math.isfinite(sigma_base_annual) or sigma_base_annual <= 0:
        return None
    loss = event_aware_weekly_loss(
        sigma_base_annual,
        event_jump_underlying=event_jump_underlying,
        strike_long=strike_long,
        strike_short=strike_short,
        n_draws=4096,
    )
    return float(loss) if math.isfinite(loss) else None


__all__ = [
    "event_aware_decay_distribution",
    "event_aware_weekly_loss",
    "fair_spread_mid_from_sigma",
]
