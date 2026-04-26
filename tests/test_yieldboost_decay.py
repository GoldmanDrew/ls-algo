"""Tests for yieldboost_decay put-spread Monte-Carlo distributional model.

Sanity properties:
    1. p10 ≤ p50 ≤ p90 by construction (percentiles of finite sample).
    2. Annual decay is monotonically increasing in σ_underlying — more
       underlying volatility → higher expected weekly put-spread loss
       → higher annual NAV decay.
    3. Median decay for plausible single-name vol ranges (35-70% σ) lands
       in the 15-50% bracket the dashboard heuristics expect for YB ETFs
       (COYY, MTYY, NVYY, …). This is the "regression test" guarding
       against silent breakage of the BS-style closed-form.
    4. Point-estimate fallback returns p10 == p50 == p90 (no dispersion).
    5. Higher σ_logIV (lognormal dispersion) widens the (p90 − p10) band.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from yieldboost_decay import (
    yieldboost_decay_distribution,
    yieldboost_decay_point_estimate,
    _annual_intrinsic_decay,
    _weekly_put_spread_loss,
)


def _mu_sigma_from_annual(sigma_annual: float, sigma_logIV: float = 0.0):
    """Translate (σ_annual, σ_logIV) → HARQ-Log moments at T=1y."""
    mu_logIV = math.log(max(sigma_annual * sigma_annual, 1e-12))
    return mu_logIV, sigma_logIV


# ─── core ordering ────────────────────────────────────────────────────────


def test_quantile_ordering():
    mu, sigma = _mu_sigma_from_annual(0.50, sigma_logIV=0.45)
    out = yieldboost_decay_distribution(
        mu_log_iv_annual=mu, sigma_log_iv_annual=sigma, n_draws=4096
    )
    assert out is not None
    assert out["p10"] <= out["p50"] <= out["p90"]
    assert 0.0 <= out["p10"] <= 1.0
    assert 0.0 <= out["p90"] <= 1.0


def test_monotonicity_in_sigma_underlying():
    """Higher underlying σ → strictly higher expected annual decay (for fixed σ_logIV=0)."""
    decays = []
    for sigma_a in (0.20, 0.30, 0.40, 0.50, 0.60, 0.80):
        mu, sigma_lv = _mu_sigma_from_annual(sigma_a, sigma_logIV=0.0)
        out = yieldboost_decay_distribution(
            mu_log_iv_annual=mu, sigma_log_iv_annual=sigma_lv, n_draws=2048
        )
        assert out is not None
        decays.append(out["p50"])
    for a, b in zip(decays, decays[1:]):
        assert b > a, f"decay is not monotonic in σ: {decays}"


def test_dispersion_grows_with_sigma_logIV():
    """Higher σ_logIV → wider (p90 − p10) band."""
    mu, _ = _mu_sigma_from_annual(0.50, sigma_logIV=0.0)
    bands = []
    for s_lv in (0.05, 0.30, 0.60):
        out = yieldboost_decay_distribution(
            mu_log_iv_annual=mu, sigma_log_iv_annual=s_lv, n_draws=4096
        )
        assert out is not None
        bands.append(out["p90"] - out["p10"])
    for a, b in zip(bands, bands[1:]):
        assert b > a, f"band did not widen with σ_logIV: {bands}"


# ─── plausible-range regression ────────────────────────────────────────────


@pytest.mark.parametrize(
    "sigma_annual,expected,abs_tol",
    [
        # Output of the put-spread BS-style closed form (single-σ deterministic
        # case at u=0, T=1y, expense=0.99%/yr, strikes 95/88, leverage=2×).
        # These pin the model to the same numbers the dashboard's front-end
        # ``yieldBoostIntrinsicAnnualDecay`` computes for the same σ_und. A
        # silent bug (dropping the 2× factor, mis-clamping, etc.) would shift
        # all three substantially outside the 0.01 absolute tolerance.
        (0.18, 0.186, 0.01),  # SPX-like underlying
        (0.30, 0.449, 0.01),  # NVDA-ish mid-vol
        (0.45, 0.620, 0.01),  # high-vol single name
        (0.65, 0.722, 0.01),  # very-high-vol single name (MSTR / SBET tier)
    ],
)
def test_decay_matches_closed_form_per_sigma(sigma_annual, expected, abs_tol):
    """Deterministic closed-form values guard the BS-style put-spread math.

    Anchoring tests against numerical fixtures (rather than loose bands)
    catches both directional drift and the regression case where someone
    introduces a multiplicative factor of 2× / ½ / etc.
    """
    mu, sigma_lv = _mu_sigma_from_annual(sigma_annual, sigma_logIV=0.0)
    out = yieldboost_decay_distribution(
        mu_log_iv_annual=mu, sigma_log_iv_annual=sigma_lv, n_draws=4096
    )
    assert out is not None
    assert abs(out["p50"] - expected) <= abs_tol, (
        f"σ={sigma_annual}: p50={out['p50']:.3f} not within "
        f"{abs_tol} of expected {expected:.3f}"
    )


# ─── point-estimate fallback ─────────────────────────────────────────────


def test_point_estimate_zero_dispersion():
    out = yieldboost_decay_point_estimate(sigma_annual=0.45)
    assert out is not None
    assert out["p10"] == out["p50"] == out["p90"] == out["mean"]
    assert out["sigma_log_iv"] == 0.0
    assert out["model"] == "yieldboost_put_spread_point"


def test_point_estimate_invalid_sigma_returns_none():
    assert yieldboost_decay_point_estimate(sigma_annual=0.0) is None
    assert yieldboost_decay_point_estimate(sigma_annual=float("nan")) is None
    assert yieldboost_decay_point_estimate(sigma_annual=-0.1) is None


# ─── numerical edges ─────────────────────────────────────────────────────


def test_zero_sigma_invalid_inputs_return_none():
    out = yieldboost_decay_distribution(
        mu_log_iv_annual=float("nan"), sigma_log_iv_annual=0.5
    )
    assert out is None


def test_compounding_matches_closed_form_at_zero_dispersion():
    """At σ_logIV=0 the MC degenerates to a deterministic σ_annual.

    The model output should equal 1 − (1 − L_w(σ) − f_w)^52, which we recompute
    directly from the helper.
    """
    sigma_annual = 0.50
    mu, _ = _mu_sigma_from_annual(sigma_annual, sigma_logIV=0.0)
    out = yieldboost_decay_distribution(
        mu_log_iv_annual=mu, sigma_log_iv_annual=0.0, n_draws=512
    )
    expected = float(_annual_intrinsic_decay(np.array([sigma_annual]))[0])
    assert out is not None
    assert math.isclose(out["p50"], round(expected, 6), abs_tol=1e-4)


def test_weekly_loss_in_unit_interval():
    """L_w stays in [0, 0.07] (max 95/88 spread payoff)."""
    sigmas = np.array([0.05, 0.10, 0.30, 0.60, 1.00, 2.00])
    loss = _weekly_put_spread_loss(sigmas)
    assert np.all(loss >= 0.0)
    assert np.all(loss <= 0.07 + 1e-12)
