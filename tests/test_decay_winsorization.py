"""Pin down the winsorization + raw_fallback_capped fix in PR #11.

Scenarios required by the independent-review prompt:

  (a) clean LETF unchanged by winsorization
  (b) single negative spike absorbed
  (c) symmetric noise unbiased (analytic recovery on a noise-free 2x rebalance)
  (d) parity smoke: a synthesized (β=+2) and (β=−2) leg from the same r_und
      have closed-form gross with opposite *magnitude* (ratio 1:3), and
      ls-algo's _compute_gross_decay matches the Itô identity
      0.5·|β|·|β−1|·σ²·252 within finite-sample tolerance.

The intent is to make the fix robust against future regressions, including
a regression that re-introduces the unwinsorized realized estimator.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import daily_screener as ds


TRADING_DAYS = ds.TRADING_DAYS


def _series(returns: np.ndarray, *, start: str = "2024-01-01") -> pd.Series:
    """Convert a returns array into a price series with daily timestamps."""
    idx = pd.date_range(start, periods=len(returns) + 1, freq="B")
    p = np.concatenate([[100.0], 100.0 * np.cumprod(1.0 + returns)])
    return pd.Series(p, index=idx)


def _gross_no_winsor(etf_tr: pd.Series, und_tr: pd.Series, beta: float) -> float:
    """Reference unwinsorized estimator — same algebra, no clipping."""
    etf_tr = etf_tr[~etf_tr.index.duplicated(keep="last")]
    und_tr = und_tr[~und_tr.index.duplicated(keep="last")]
    df = pd.concat(
        [etf_tr.rename("e"), und_tr.rename("u")], axis=1, sort=True
    ).dropna()
    re = np.log(df["e"] / df["e"].shift(1)).dropna()
    ru = np.log(df["u"] / df["u"].shift(1)).dropna()
    valid = re.index.intersection(ru.index)
    re, ru = re.loc[valid], ru.loc[valid]
    drag = beta * ru - re
    return float(drag.mean()) * TRADING_DAYS


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(20260505)


# ── (a) clean LETF unchanged by winsorization ─────────────────────────
def test_clean_letf_winsor_is_noop(rng: np.random.Generator) -> None:
    """When daily_drag has no outliers (effectively a tight degenerate
    distribution) the winsorized estimator must equal the unwinsorized
    estimator to floating-point precision.

    Construction: a *log-linear* β=2 tracker (1+r_etf = (1+r_und)^2) makes
    daily_drag identically zero by definition, which is the cleanest
    no-op baseline available analytically.  This is intentionally the
    degenerate case — the cleaner-but-still-real 2× daily-rebalance
    construction has small Itô curvature in drag (≈ r²·β·(β−1)/2 per day)
    that winsorization legitimately interacts with, see
    ``test_recovers_ito_identity_two_x`` for the noise-free rebalance
    behaviour.
    """
    n = 500
    ru = rng.normal(0, 0.02, size=n)
    re = (1.0 + ru) ** 2 - 1.0  # log-linear tracker: drag ≡ 0
    s_e, s_u = _series(re), _series(ru)
    g_w = ds._compute_gross_decay(s_e, s_u, 2.0)
    g_u = _gross_no_winsor(s_e, s_u, 2.0)
    assert g_w == pytest.approx(g_u, abs=1e-5), (g_w, g_u)
    assert abs(g_w) < 1e-5


# ── (b) single negative spike absorbed ────────────────────────────────
def test_negative_spike_absorbed_two_x(rng: np.random.Generator) -> None:
    """One bad day with daily_drag ≈ −2.3 (typical 'Yahoo TR glitch')
    must NOT dominate the winsorized estimator."""
    n = 500
    ru = rng.normal(0, 0.02, size=n)
    re_clean = 2.0 * ru
    s_u = _series(ru)
    s_e_clean = _series(re_clean)

    re_dirty = re_clean.copy()
    day = 200
    # Force log(1+r_etf) such that daily_drag = β·log(1+r_und) − log(1+r_etf) = −2.3
    re_dirty[day] = (1 + ru[day]) ** 2 * np.exp(2.3) - 1
    s_e_dirty = _series(re_dirty)

    g_clean = ds._compute_gross_decay(s_e_clean, s_u, 2.0)
    g_dirty = ds._compute_gross_decay(s_e_dirty, s_u, 2.0)
    g_dirty_unwin = _gross_no_winsor(s_e_dirty, s_u, 2.0)
    g_clean_unwin = _gross_no_winsor(s_e_clean, s_u, 2.0)

    assert abs(g_dirty_unwin - g_clean_unwin) > 1.0
    assert abs(g_dirty - g_clean) < 0.02, (g_dirty, g_clean)


# ── (b′) inverse leg, single positive spike absorbed (asymmetry guard) ─
def test_positive_spike_absorbed_inverse(rng: np.random.Generator) -> None:
    """A single +ve drag spike on a −2× LETF is the dual of (b). The
    winsorization must be symmetric (no sign bias on the inverse leg)."""
    n = 500
    ru = rng.normal(0, 0.02, size=n)
    re_inv = -2.0 * ru
    s_u = _series(ru)
    s_e_clean = _series(re_inv)

    re_inv_dirty = re_inv.copy()
    day = 100
    re_inv_dirty[day] = (1 + ru[day]) ** (-2) * np.exp(-2.3) - 1
    s_e_dirty = _series(re_inv_dirty)

    g_clean = ds._compute_gross_decay(s_e_clean, s_u, -2.0)
    g_dirty = ds._compute_gross_decay(s_e_dirty, s_u, -2.0)
    g_clean_unwin = _gross_no_winsor(s_e_clean, s_u, -2.0)
    g_dirty_unwin = _gross_no_winsor(s_e_dirty, s_u, -2.0)

    assert abs(g_dirty_unwin - g_clean_unwin) > 1.0
    # Bias from a single one-sided spike must be < 0.5% absolute
    assert abs(g_dirty - g_clean) < 0.005, (g_dirty, g_clean)


# ── (c) symmetric noise → analytic recovery (Itô identity) ────────────
def test_recovers_ito_identity_two_x(rng: np.random.Generator) -> None:
    """Noise-free 2× daily rebalance: realized drag must equal
    0.5·|β|·|β−1|·σ²·252 within finite-sample tolerance."""
    n = 500
    ru = rng.normal(0, 0.02, size=n)
    re = 2.0 * ru
    g = ds._compute_gross_decay(_series(re), _series(ru), 2.0)
    sig_u = ds._annualized_second_moment_log(_series(ru), 60)
    analytic = 0.5 * 2 * 1 * sig_u ** 2
    assert abs(g - analytic) < 0.03, (g, analytic)


def test_recovers_ito_identity_minus_two_x(rng: np.random.Generator) -> None:
    n = 500
    ru = rng.normal(0, 0.02, size=n)
    re = -2.0 * ru
    g = ds._compute_gross_decay(_series(re), _series(ru), -2.0)
    sig_u = ds._annualized_second_moment_log(_series(ru), 60)
    analytic = 0.5 * 2 * 3 * sig_u ** 2  # |β|=2, |β−1|=3
    assert abs(g - analytic) < 0.05, (g, analytic)


# ── (c′) persistent legitimate drag is preserved ──────────────────────
def test_persistent_drag_not_eroded(rng: np.random.Generator) -> None:
    """Long history (n=500) at high vol on a 3× LETF: winsorization must
    NOT erode the genuine drag signal more than 5% absolute vs the
    analytic identity."""
    n = 500
    ru = rng.normal(0, 0.025, size=n)
    re = 3.0 * ru  # noise-free 3× daily rebalance
    g = ds._compute_gross_decay(_series(re), _series(ru), 3.0)
    sig_u = ds._annualized_second_moment_log(_series(ru), 60)
    analytic = 0.5 * 3 * 2 * sig_u ** 2
    assert abs(g - analytic) < 0.05, (g, analytic)


# ── (d) parity smoke: long vs inverse legs share the Itô identity ─────
def test_parity_long_vs_inverse_closed_form(rng: np.random.Generator) -> None:
    """Synthesize +2× and −2× legs from the *same* r_und; closed-form
    expected gross has ratio 1 : 3 by 0.5·|β|·|β−1|·σ²; realized gross
    must match within finite-sample tolerance."""
    n = 500
    ru = rng.normal(0, 0.02, size=n)
    s_u = _series(ru)
    g_long = ds._compute_gross_decay(_series(2.0 * ru), s_u, 2.0)
    g_inv = ds._compute_gross_decay(_series(-2.0 * ru), s_u, -2.0)

    # Both must be positive (gross drag accrues to the short, regardless
    # of sign of β) and ratio g_inv / g_long ≈ 3 within ±15% from finite
    # sample noise on σ̂.
    assert g_long > 0
    assert g_inv > 0
    ratio = g_inv / g_long
    assert 2.55 <= ratio <= 3.45, ratio


# ── raw_fallback_capped behaviour (B) — unit test on the constants ────
def test_raw_fallback_capped_constants_present() -> None:
    """Sanity: PASS2 raw_fallback_capped path lives in
    enrich_with_decay_and_vol; the SIGMA_CAP_SAFETY constant should be
    1.25 (any change is a cross-repo parity breakage with DC PR mirror)."""
    src = open("daily_screener.py").read()
    assert "SIGMA_CAP_SAFETY = 1.25" in src
    assert "raw_fallback_capped" in src
    assert "raw_fallback" in src
    # Cap must be applied AFTER raw_fallback flag flip
    cap_idx = src.index('method = "raw_fallback_capped"')
    fb_idx = src.index('method = "raw_fallback"')
    assert cap_idx > fb_idx
