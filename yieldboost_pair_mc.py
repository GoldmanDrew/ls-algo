"""Weekly-rebalanced compound pair-P&L Monte Carlo for YieldBOOST rows.

Mirrors ``etf-dashboard/scripts/income_schedule.py::simulate_weekly_compound_pair_pnl``.
Outputs **log_continuous_annual** pair P&L quantiles (short-favorable positive) for
``expected_gross_decay_p{10,50,90}_annual`` and net-edge forward anchoring.

Keep put-spread constants in sync with ``yieldboost_decay.py`` / dashboard
``expectedPutSpreadLossWeekly``.
"""

from __future__ import annotations

import math
import zlib
from typing import Any, Dict, Optional

import numpy as np

PUT_SPREAD_SHORT_STRIKE = 0.95
PUT_SPREAD_LONG_STRIKE = 0.88
PUT_SPREAD_LEVERAGE = 2.0
DEFAULT_EXPENSE_RATIO_ANNUAL = 0.0099
DEFAULT_CROSS_FUND_RATIO = 0.65
WEEKS_PER_YEAR = 52

_PAIR_WEEK_FLOOR = -0.99
_PAIR_WEEK_CEIL = 5.0
_DEFAULT_N_PATHS = 4096
_DEFAULT_SEED = 17


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def expected_put_spread_loss_weekly(
    underlying_return: float,
    sigma_annual: float,
    horizon_years: float = 1.0,
) -> float:
    u = float(underlying_return)
    sigma = float(sigma_annual)
    t = float(horizon_years)
    if not math.isfinite(u) or u <= -0.9999:
        return float("nan")
    if not math.isfinite(sigma) or sigma <= 0:
        return float("nan")
    if not math.isfinite(t) or t <= 0:
        return float("nan")
    tau = 1.0 / WEEKS_PER_YEAR
    mu_annual = math.log1p(u) / t
    m = (PUT_SPREAD_LEVERAGE * mu_annual - PUT_SPREAD_LEVERAGE * sigma * sigma) * tau
    s = PUT_SPREAD_LEVERAGE * sigma * math.sqrt(tau)
    if not math.isfinite(m) or not math.isfinite(s) or s <= 0:
        return float("nan")

    def _spread_put(k: float) -> float:
        alpha = (math.log(k) - m) / s
        beta = alpha - s
        forward = math.exp(m + 0.5 * s * s)
        return k * _norm_cdf(alpha) - forward * _norm_cdf(beta)

    loss = _spread_put(PUT_SPREAD_SHORT_STRIKE) - _spread_put(PUT_SPREAD_LONG_STRIKE)
    if not math.isfinite(loss):
        return float("nan")
    max_loss = PUT_SPREAD_SHORT_STRIKE - PUT_SPREAD_LONG_STRIKE
    return max(0.0, min(max_loss, loss))


def stable_seed_from_symbol(symbol: str, *, salt: int = 0) -> int:
    sym = (symbol or "").strip().upper()
    base = zlib.crc32(sym.encode("utf-8")) & 0xFFFFFFFF
    return int((base ^ (int(salt) & 0xFFFFFFFF)) & 0x7FFFFFFF)


def _put_spread_payoff_vec(sleeve_ret: np.ndarray) -> np.ndarray:
    end = 1.0 + sleeve_ret
    short_put = np.maximum(0.0, PUT_SPREAD_SHORT_STRIKE - end)
    long_put = np.maximum(0.0, PUT_SPREAD_LONG_STRIKE - end)
    spread = short_put - long_put
    return np.clip(spread, 0.0, PUT_SPREAD_SHORT_STRIKE - PUT_SPREAD_LONG_STRIKE)


def simulate_weekly_compound_pair_pnl(
    sigma_annual: float | None,
    mu_annual: float = 0.0,
    beta: float = 0.0,
    capture_ratio: float = DEFAULT_CROSS_FUND_RATIO,
    *,
    expense_ratio_annual: float = DEFAULT_EXPENSE_RATIO_ANNUAL,
    borrow_annual: float = 0.0,
    weeks: int = WEEKS_PER_YEAR,
    n_paths: int = _DEFAULT_N_PATHS,
    seed: int = _DEFAULT_SEED,
) -> dict | None:
    if sigma_annual is None:
        return None
    try:
        sigma = float(sigma_annual)
        mu = float(mu_annual)
        b = float(beta)
        er = float(expense_ratio_annual)
        borrow = float(borrow_annual)
        cap_r = float(capture_ratio) if capture_ratio is not None else 0.0
    except (TypeError, ValueError):
        return None
    if not math.isfinite(sigma) or sigma <= 0:
        return None
    if not (math.isfinite(mu) and math.isfinite(b) and math.isfinite(er) and math.isfinite(borrow)):
        return None
    weeks_i = int(max(1, weeks))
    n_paths_i = int(max(1, n_paths))

    sigma_w = sigma / math.sqrt(WEEKS_PER_YEAR)
    mu_w = mu / WEEKS_PER_YEAR - 0.5 * sigma_w * sigma_w
    weekly_er = max(0.0, er) / WEEKS_PER_YEAR
    weekly_borrow = max(0.0, borrow) / WEEKS_PER_YEAR
    bs_weekly = expected_put_spread_loss_weekly(0.0, sigma, 1.0)
    if not math.isfinite(bs_weekly):
        bs_weekly = 0.0
    weekly_dist = max(0.0, cap_r * bs_weekly) if cap_r > 0 else 0.0

    rng = np.random.default_rng(int(seed) & 0x7FFFFFFF)
    z = rng.standard_normal(size=(n_paths_i, weeks_i))
    log_und = mu_w + sigma_w * z
    r_und = np.expm1(log_und)
    sleeve_ret = np.expm1(2.0 * log_und)
    L = _put_spread_payoff_vec(sleeve_ret)
    pair_w = L + weekly_er - weekly_borrow - weekly_dist + b * r_und
    pair_w = np.clip(pair_w, _PAIR_WEEK_FLOOR, _PAIR_WEEK_CEIL)
    log_pair_week = np.log1p(pair_w)
    log_pair_total = log_pair_week.sum(axis=1)
    annualization = WEEKS_PER_YEAR / float(weeks_i)
    log_pair_annual = log_pair_total * annualization

    quantiles = np.quantile(log_pair_annual, [0.10, 0.25, 0.50, 0.75, 0.90])
    return {
        "p10_log": float(quantiles[0]),
        "p25_log": float(quantiles[1]),
        "p50_log": float(quantiles[2]),
        "p75_log": float(quantiles[3]),
        "p90_log": float(quantiles[4]),
        "mean_log": float(log_pair_annual.mean()),
        "std_log": float(log_pair_annual.std(ddof=1)) if n_paths_i > 1 else 0.0,
        "n_paths": int(n_paths_i),
        "sigma_used": float(sigma),
        "beta_used": float(b),
        "capture_used": float(cap_r),
        "borrow_annual": float(borrow),
        "axis": "log_continuous_annual",
    }


def yieldboost_pair_decay_distribution(
    *,
    sigma_annual: float,
    beta: float,
    capture_ratio: float = DEFAULT_CROSS_FUND_RATIO,
    borrow_annual: float = 0.0,
    expense_ratio_annual: float = DEFAULT_EXPENSE_RATIO_ANNUAL,
    n_paths: int = _DEFAULT_N_PATHS,
    seed: int = _DEFAULT_SEED,
    mu_log_iv_annual: float | None = None,
) -> Optional[Dict[str, Any]]:
    """Headline forward pair P&L distribution for screener ``expected_gross_decay_*``.

    Returns log-continuous-annual quantiles compatible with LETF net-edge blending.
    """
    mc = simulate_weekly_compound_pair_pnl(
        sigma_annual=sigma_annual,
        mu_annual=0.0,
        beta=beta,
        capture_ratio=capture_ratio,
        expense_ratio_annual=expense_ratio_annual,
        borrow_annual=borrow_annual,
        n_paths=n_paths,
        seed=seed,
    )
    if mc is None:
        return None
    mu_log_iv = (
        float(mu_log_iv_annual)
        if mu_log_iv_annual is not None and math.isfinite(float(mu_log_iv_annual))
        else math.log(max(float(sigma_annual) ** 2, 1e-12))
    )
    return {
        "p10": float(round(mc["p10_log"], 6)),
        "p25": float(round(mc["p25_log"], 6)),
        "p50": float(round(mc["p50_log"], 6)),
        "p75": float(round(mc["p75_log"], 6)),
        "p90": float(round(mc["p90_log"], 6)),
        "mean": float(round(mc["mean_log"], 6)),
        "std_log": float(round(mc["std_log"], 6)),
        "mu_log_iv": round(mu_log_iv, 6),
        "sigma_log_iv": 0.0,
        "n_obs": float(n_paths),
        "model": "yieldboost_weekly_compound_mc",
        "axis": "log_continuous_annual",
    }


def _borrow_from_row(row) -> float:
    for key in ("borrow_for_net_annual", "borrow_current", "borrow_fee_annual"):
        try:
            v = row.get(key)
            if v is None:
                continue
            f = float(v)
            if math.isfinite(f) and f >= 0:
                return f
        except (TypeError, ValueError):
            continue
    return 0.0


__all__ = [
    "DEFAULT_CROSS_FUND_RATIO",
    "expected_put_spread_loss_weekly",
    "simulate_weekly_compound_pair_pnl",
    "stable_seed_from_symbol",
    "yieldboost_pair_decay_distribution",
    "_borrow_from_row",
]
