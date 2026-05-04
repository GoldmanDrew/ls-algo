"""Unit tests for the AR(1)-adjusted shrinkage hedge-ratio estimator.

The estimator lives in ``daily_screener``:

    β̂ = w · β̂_OLS + (1 − w) · L
    w  = n_eff / (n_eff + k(L)),  k(L) = K_BASE · max(1, L²)
    n_eff = n · (1 − ρ_AR1) / (1 + ρ_AR1)

These tests pin down four properties that the production code relies on:

  1. With a long, clean sample the estimator essentially recovers OLS.
  2. With a short sample it leans heavily on the listed-leverage prior.
  3. Sign mismatch is hard-snapped to L (data-corruption guard).
  4. AR(1) > 0 (positively autocorrelated underlying) shrinks more than
     the same-length white-noise series.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from daily_screener import (
    BETA_SHRINK_K_BASE,
    _ar1_n_eff,
    compute_beta_shrunk,
)


def _series(returns: np.ndarray) -> pd.Series:
    """Convert a returns array into a price series with daily timestamps."""
    idx = pd.date_range("2024-01-01", periods=len(returns) + 1, freq="B")
    prices = np.concatenate([[100.0], 100.0 * np.cumprod(1.0 + returns)])
    return pd.Series(prices, index=idx)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(20260504)


def test_long_sample_recovers_ols(rng: np.random.Generator) -> None:
    """500 daily obs of a clean 2× LETF: shrinkage barely moves OLS."""
    n = 500
    r_und = rng.normal(0, 0.02, size=n)
    r_etf = 1.85 * r_und + rng.normal(0, 0.001, size=n)  # tight realised slope
    beta, n_obs, src = compute_beta_shrunk(
        _series(r_etf), _series(r_und), exp_leverage=2.0
    )
    assert src == "shrunk_to_L"
    assert n_obs == n
    # Pulled toward 2.0 but not by more than 0.10 (low-noise, long sample).
    assert 1.83 < beta < 1.95


def test_short_sample_leans_on_prior(rng: np.random.Generator) -> None:
    """80 daily obs with a noisy underlying: shrinkage pulls firmly to L."""
    n = 80
    r_und = rng.normal(0, 0.02, size=n)
    # Realised slope happens to come in compressed at ~1.30.
    r_etf = 1.30 * r_und + rng.normal(0, 0.01, size=n)
    beta, n_obs, src = compute_beta_shrunk(
        _series(r_etf), _series(r_und), exp_leverage=2.0
    )
    assert src == "shrunk_to_L"
    assert n_obs == n
    # With k(L=2) = 4·K_BASE = 240 and n_eff ≤ 80, prior weight is > 70 %,
    # so β must land much closer to 2.0 than to 1.30.
    assert beta > 1.55, f"expected strong pull toward L, got {beta:.3f}"


def test_sign_mismatch_snaps_to_leverage(rng: np.random.Generator) -> None:
    """An OLS β with the wrong sign is treated as data corruption."""
    n = 200
    r_und = rng.normal(0, 0.02, size=n)
    r_etf = -1.5 * r_und + rng.normal(0, 0.001, size=n)  # wrong sign
    beta, _, src = compute_beta_shrunk(
        _series(r_etf), _series(r_und), exp_leverage=2.0
    )
    assert src == "imputed_sign_mismatch"
    assert beta == 2.0


def test_no_overlap_returns_prior(rng: np.random.Generator) -> None:
    """Below the minimum sample, return L with source ``imputed_no_overlap``."""
    r = rng.normal(0, 0.02, size=10)
    beta, n_obs, src = compute_beta_shrunk(
        _series(r), _series(r), exp_leverage=-2.0, min_days=60
    )
    assert src == "imputed_no_overlap"
    assert n_obs == 0
    assert beta == -2.0


def test_ar1_adjustment_increases_shrinkage(rng: np.random.Generator) -> None:
    """Positively autocorrelated returns deflate n_eff and pull β toward L."""
    n = 300
    eps = rng.normal(0, 0.02, size=n)
    # White-noise underlying.
    r_und_iid = eps.copy()
    # AR(1) underlying with ρ ≈ 0.4 (real LETF underlyings are usually
    # closer to 0.05, but a stronger ρ makes the test deterministic).
    rho = 0.4
    r_und_ar = np.empty(n)
    r_und_ar[0] = eps[0]
    for t in range(1, n):
        r_und_ar[t] = rho * r_und_ar[t - 1] + np.sqrt(1 - rho * rho) * eps[t]

    # Common compressed realised slope (β_OLS ≈ 1.40).
    noise = rng.normal(0, 0.005, size=n)
    r_etf_iid = 1.40 * r_und_iid + noise
    r_etf_ar = 1.40 * r_und_ar + noise

    beta_iid, _, _ = compute_beta_shrunk(
        _series(r_etf_iid), _series(r_und_iid), exp_leverage=2.0
    )
    beta_ar, _, _ = compute_beta_shrunk(
        _series(r_etf_ar), _series(r_und_ar), exp_leverage=2.0
    )

    # AR(1) path has fewer effective independent observations, so it
    # must end up *closer to L = 2* than the IID path.
    assert beta_ar > beta_iid, (beta_iid, beta_ar)


def test_n_eff_helper_handles_edges() -> None:
    """``_ar1_n_eff`` returns sensible values for tiny / degenerate inputs."""
    rho, n_eff = _ar1_n_eff(np.zeros(2))
    assert rho == 0.0
    assert n_eff == 2

    rho, n_eff = _ar1_n_eff(np.zeros(50))  # zero variance
    assert rho == 0.0
    assert n_eff == 50

    # Strongly positive ρ: n_eff must be far below n.
    n = 400
    eps = np.random.default_rng(0).normal(size=n)
    r = np.empty(n)
    r[0] = eps[0]
    for t in range(1, n):
        r[t] = 0.8 * r[t - 1] + 0.6 * eps[t]
    rho, n_eff = _ar1_n_eff(r)
    assert rho > 0.7
    assert n_eff < n / 2


def test_k_base_and_min_days_exposed_for_callers() -> None:
    """Module-level constants are part of the public interface."""
    assert BETA_SHRINK_K_BASE == 60
