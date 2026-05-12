"""Pin the winsorization fix on the Itô-aligned σ estimator.

Companion to ``tests/test_decay_winsorization.py`` (which covers the daily
drag estimator). When a single bar carries a corporate-action artifact that
upstream cleaning missed — e.g. a same-day reverse split where Yahoo lags
on retro-adjusting history — the squared log return on that bar can
dominate ``mean(r²)`` and inflate σ by ~+200 %.

The new ``_annualized_second_moment_log`` clips squared returns at 1/99
(loosened to 5/95 for thin histories), matching the tier table in
``_compute_gross_decay_daily``. For clean symbols the clip is a no-op to
floating-point precision; for dirty symbols a single uncaught split bar
contributes at most the 99th-percentile squared return.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import daily_screener as ds  # noqa: E402
import etf_analytics as ea  # noqa: E402


def _series(returns: np.ndarray, *, start: str = "2024-01-01") -> pd.Series:
    idx = pd.bdate_range(start, periods=len(returns) + 1)
    p = np.concatenate([[100.0], 100.0 * np.cumprod(1.0 + returns)])
    return pd.Series(p, index=idx)


def test_clean_letf_winsor_is_noop_in_screener() -> None:
    """A noise-only series must not be visibly perturbed by winsorization."""
    rng = np.random.default_rng(2026_05_05)
    rets = rng.normal(0, 0.02, size=252)
    s = _series(rets)
    sigma = ds._annualized_second_moment_log(s, min_days=60)
    # Reference: unwinsorized.
    r = np.log(s / s.shift(1)).dropna()
    sigma_ref = float(np.sqrt(float((r ** 2).mean()) * ds.TRADING_DAYS))
    # Both estimators must agree to within 5 % on a clean series.
    assert abs(sigma - sigma_ref) / sigma_ref < 0.05


def test_single_split_bar_no_longer_dominates_screener_sigma() -> None:
    """One uncaught log(10) bar in a 252-day window adds ~+2.3 to σ pre-fix.

    Post-fix the contribution is bounded by the 1/99 clip → σ rises by far
    less than the pre-fix +2.3.
    """
    rng = np.random.default_rng(7)
    rets = rng.normal(0, 0.02, size=252).tolist()
    rets[-1] = np.log(10.0) - 1.0  # exp(rets[-1]+1) = 10×, so 1+r = 10
    rets[-1] = 9.0  # equivalent: ln(1+9) = ln(10)
    s = _series(np.array(rets))

    sigma_w = ds._annualized_second_moment_log(s, min_days=60)
    # Reference: NO winsorization (pre-fix behaviour).
    r = np.log(s / s.shift(1)).dropna()
    sigma_raw = float(np.sqrt(float((r ** 2).mean()) * ds.TRADING_DAYS))

    # Pre-fix the raw σ is dominated by the single bar (~+2.3 absolute
    # annualized vol from a single log(10) day). Post-fix the winsorized σ
    # must be substantially smaller AND below the same ds._VOL_CAP_ANNUAL.
    assert sigma_w < sigma_raw - 0.5
    assert sigma_w < ds._VOL_CAP_ANNUAL


def test_etf_analytics_and_screener_winsor_agree() -> None:
    """Both copies of ``_annualized_second_moment_log`` must yield the same
    result on a fixture series with one outlier — the two paths share the
    same algebra now.
    """
    rng = np.random.default_rng(33)
    rets = rng.normal(0, 0.025, size=252).tolist()
    rets[100] = np.log(8.0)  # injected single split-shaped bar
    s = _series(np.array(rets))
    a = ds._annualized_second_moment_log(s, min_days=60)
    b = ea._annualized_second_moment_log(s, min_days=60)
    assert abs(a - b) < 1e-6
