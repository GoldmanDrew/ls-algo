"""Regression tests for ``decay_distribution.py`` (Phase 1).

These tests are deliberately self-contained: they synthesise GBM price
paths and check that

  * the inverse-normal helper matches well-known Φ⁻¹ values,
  * lognormal decay quantiles satisfy ``p10 < p50 < p90`` and bracket the
    Itô plug-in with realistic empirical sigma,
  * ``enrich_with_decay_distribution`` populates the four new columns
    on a multi-row DataFrame and routes correctly to the HARQ-Log path
    when given enough history (and to the simple-Itô fallback otherwise).

We do **not** hit the network. ``daily_screener.py`` integration is
covered indirectly by ``test_volatility_etp_expected_decay.py`` (which
exercises ``apply_volatility_etp_expected_decay_adjustment``) — running
``main()`` here would require Yahoo + IBKR access.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from decay_distribution import (
    TRADING_DAYS,
    _HORIZON_PANEL_RATIO_MIN,
    _LOG_IV_SIGMA_ANNUAL_CAP,
    _R2_WINSOR_CAP,
    _acklam_inv_normal,
    _c_beta,
    _cap_mu_log_iv,
    _empirical_log_iv_moments,
    _fit_harq_log,
    _harq_log_conditional_shift,
    _lognormal_decay_from_logiv,
    _realized_variance_panel,
    enrich_with_decay_distribution,
    forecast_decay_distribution,
)


# ─── helpers ────────────────────────────────────────────────────────────────

def _gbm_total_return(
    *,
    n_days: int,
    sigma_annual: float,
    mu_annual: float = 0.0,
    seed: int = 7,
) -> pd.Series:
    """Geometric Brownian motion daily TR series with known annualised vol."""
    rng = np.random.default_rng(seed)
    dt = 1.0 / 252.0
    drift = (mu_annual - 0.5 * sigma_annual ** 2) * dt
    shocks = rng.normal(loc=drift, scale=sigma_annual * math.sqrt(dt), size=n_days)
    log_levels = np.cumsum(shocks)
    idx = pd.bdate_range(end=pd.Timestamp("2024-12-31"), periods=n_days)
    return pd.Series(np.exp(log_levels), index=idx, name="GBM")


# ─── inverse normal sanity ──────────────────────────────────────────────────

def test_acklam_inverse_normal_matches_known_quantiles():
    cases = {
        0.5: 0.0,
        0.84134474606: 1.0,
        0.97724986805: 2.0,
        0.99865010196: 3.0,
        0.025: -1.95996398454,
        0.10: -1.28155156554,
        0.90: 1.28155156554,
    }
    for p, expected in cases.items():
        got = _acklam_inv_normal(p)
        assert abs(got - expected) < 1e-6, f"Φ⁻¹({p}) = {got}, expected {expected}"


# ─── decay-quantile mapping math ────────────────────────────────────────────

def test_lognormal_decay_quantiles_are_ordered_and_bracket_simple_ito():
    # mu chosen so exp(mu) = 0.04 (≈ 20 % annualised vol → IV_T = 0.04)
    mu = math.log(0.04)
    sigma = 0.30
    beta = 2.0
    out = _lognormal_decay_from_logiv(mu, sigma, beta, (0.10, 0.50, 0.90))
    assert out["p10"] < out["p50"] < out["p90"]
    cb = _c_beta(beta)
    simple_ito = cb * 0.04
    # p50 should equal cb * exp(mu) = cb * 0.04 exactly.
    assert abs(out["p50"] - simple_ito) < 1e-6
    # Mean of a lognormal is exp(mu + 0.5 σ²) > median, so cb·E[IV] > p50.
    assert out["mean"] > out["p50"]


def test_c_beta_matches_avellaneda_zhang():
    # The vol-drag coefficient (β² − β)/2 reduces to:
    #   β=+2 → 1.0 ;  β=+3 → 3.0 ;  β=−1 → 1.0 ;  β=−2 → 3.0
    assert abs(_c_beta(2.0) - 1.0) < 1e-12
    assert abs(_c_beta(3.0) - 3.0) < 1e-12
    assert abs(_c_beta(-1.0) - 1.0) < 1e-12
    assert abs(_c_beta(-2.0) - 3.0) < 1e-12
    # Unleveraged or zero exposure → no drag.
    assert _c_beta(1.0) == 0.0
    assert _c_beta(0.0) == 0.0


def test_zero_drag_for_unit_or_zero_beta():
    tr = _gbm_total_return(n_days=600, sigma_annual=0.30)
    out = forecast_decay_distribution(tr, beta=1.0)
    assert out is not None
    assert out["p10"] == 0.0 and out["p50"] == 0.0 and out["p90"] == 0.0


# ─── HARQ-Log fit against a known-σ GBM ────────────────────────────────────

def test_harq_log_fit_is_stationary_on_gbm():
    tr = _gbm_total_return(n_days=1500, sigma_annual=0.25, seed=11)
    panel = _realized_variance_panel(tr)
    assert panel is not None and len(panel) > 1000
    fit = _fit_harq_log(panel)
    assert fit is not None
    # Persistence on GBM-derived log-RV should be close to but below 1.
    assert 0.0 < fit["persistence"] < 1.05
    # The conditional shift over a 1y horizon must be small (we're far
    # from any shock so the latest state is near the long-run mean).
    shift = _harq_log_conditional_shift(fit, panel, horizon_days=252)
    assert abs(shift) < 0.5


def test_empirical_log_iv_moments_recover_known_gbm_iv():
    sigma = 0.30
    tr = _gbm_total_return(n_days=1500, sigma_annual=sigma, seed=3)
    panel = _realized_variance_panel(tr)
    moments = _empirical_log_iv_moments(panel, horizon_days=252)
    assert moments is not None
    mu, sigma_log_iv = moments
    # On a GBM with sigma_annual=0.30, the 1y integrated variance is
    # essentially σ²·1 = 0.09 with very small dispersion (variance of
    # the average of 252 χ²(1) draws is tiny). So mu should be near
    # log(0.09) = -2.41.
    assert abs(mu - math.log(sigma ** 2)) < 0.2, f"mu={mu}, expected≈{math.log(sigma**2)}"
    # And sigma_log_iv should be small (< 0.4) for a single GBM regime.
    assert 0.01 < sigma_log_iv < 0.5


# ─── full forecast end-to-end ──────────────────────────────────────────────

def test_forecast_decay_distribution_uses_harq_log_when_enough_history():
    sigma = 0.40
    tr = _gbm_total_return(n_days=1500, sigma_annual=sigma, seed=99)
    out = forecast_decay_distribution(tr, beta=2.0, horizon_days=252)
    assert out is not None
    assert out["model"] == "harq_log_anchored"
    assert out["p10"] < out["p50"] < out["p90"]
    # Theoretical drag for β=2 at σ=0.40: cb·σ² = 1·0.16 = 0.16. Allow
    # ±50 % of the target to account for single-path GBM sampling noise.
    target = 1.0 * sigma ** 2
    assert 0.5 * target < out["p50"] < 1.6 * target


def test_forecast_decay_distribution_falls_back_to_simple_ito_with_short_history():
    # Only ~80 days of data — too short for HARQ-Log or empirical sigma.
    tr = _gbm_total_return(n_days=80, sigma_annual=0.30, seed=5)
    out = forecast_decay_distribution(
        tr, beta=2.0, horizon_days=252,
        fallback_expected_decay=0.09,  # = cb·σ² for σ=0.30, β=2
    )
    assert out is not None
    assert out["model"] == "simple_ito_fallback"
    # With sigma_log_iv = 0 the three quantiles collapse to the median.
    assert abs(out["p10"] - out["p50"]) < 1e-6
    assert abs(out["p90"] - out["p50"]) < 1e-6
    assert abs(out["p50"] - 0.09) < 1e-3


def test_forecast_returns_none_when_no_history_and_no_fallback():
    tr = _gbm_total_return(n_days=20, sigma_annual=0.30, seed=4)
    out = forecast_decay_distribution(tr, beta=2.0, fallback_expected_decay=None)
    assert out is None


# ─── DataFrame enrichment ──────────────────────────────────────────────────

def test_enrich_with_decay_distribution_populates_new_columns_and_caches_per_und():
    tr_qqq = _gbm_total_return(n_days=1300, sigma_annual=0.18, seed=1)
    tr_tsla = _gbm_total_return(n_days=1300, sigma_annual=0.55, seed=2)
    tr_map = {"QQQ": tr_qqq, "TSLA": tr_tsla}

    df = pd.DataFrame([
        {"ETF": "TQQQ", "Underlying": "QQQ", "Beta": 3.0,
         "expected_gross_decay_annual": 0.10},
        {"ETF": "QLD",  "Underlying": "QQQ", "Beta": 2.0,
         "expected_gross_decay_annual": 0.04},
        {"ETF": "TSLL", "Underlying": "TSLA", "Beta": 2.0,
         "expected_gross_decay_annual": 0.30},
    ])

    out = enrich_with_decay_distribution(df, tr_map, horizon_days=252)

    new_cols = [
        "expected_gross_decay_p10_annual",
        "expected_gross_decay_p50_annual",
        "expected_gross_decay_p90_annual",
        "expected_gross_decay_mean_annual",
        "expected_gross_decay_dist_model",
        "expected_logIV_mu_annual",
        "expected_logIV_sigma_annual",
    ]
    for col in new_cols:
        assert col in out.columns
        assert out[col].notna().all(), f"{col} has NaN: {out[col].tolist()}"

    # Original column is untouched.
    assert (out["expected_gross_decay_annual"] == df["expected_gross_decay_annual"]).all()

    # Quantiles are ordered.
    assert (
        out["expected_gross_decay_p10_annual"]
        < out["expected_gross_decay_p50_annual"]
    ).all()
    assert (
        out["expected_gross_decay_p50_annual"]
        < out["expected_gross_decay_p90_annual"]
    ).all()

    # Higher-vol underlying → larger median decay for same |β|.
    qld_p50 = out.loc[out["ETF"] == "QLD", "expected_gross_decay_p50_annual"].iloc[0]
    tsll_p50 = out.loc[out["ETF"] == "TSLL", "expected_gross_decay_p50_annual"].iloc[0]
    assert tsll_p50 > qld_p50

    # Higher |β| on the same underlying → larger median decay.
    tqqq_p50 = out.loc[out["ETF"] == "TQQQ", "expected_gross_decay_p50_annual"].iloc[0]
    assert tqqq_p50 > qld_p50

    # Both QQQ-backed ETFs should share the SAME log-IV moments (cached).
    qqq_rows = out[out["Underlying"] == "QQQ"]
    assert qqq_rows["expected_logIV_mu_annual"].nunique() == 1
    assert qqq_rows["expected_logIV_sigma_annual"].nunique() == 1


def test_enrich_falls_back_to_simple_ito_when_underlying_missing():
    df = pd.DataFrame([
        {"ETF": "OBSCURE", "Underlying": "NEW", "Beta": 2.0,
         "expected_gross_decay_annual": 0.08},
    ])
    # tr_map intentionally empty — no history for NEW.
    out = enrich_with_decay_distribution(df, {}, horizon_days=252)
    assert out["expected_gross_decay_dist_model"].iloc[0] == "simple_ito_fallback"
    assert abs(out["expected_gross_decay_p50_annual"].iloc[0] - 0.08) < 1e-3
    # Width is zero by design in this fallback.
    assert abs(
        out["expected_gross_decay_p90_annual"].iloc[0]
        - out["expected_gross_decay_p10_annual"].iloc[0]
    ) < 1e-6


# ─── plausibility / sample-size guard rails (Phase 1.1) ────────────────────

def test_short_panel_with_spike_falls_back_to_simple_ito():
    """Pre-fix this would (mis-)report a ~280% σ HARQ-Log forecast.

    Build an underlying with only ~1.5× horizon of history (well below
    K_min · horizon = 2.5 · 252 = 630 days) plus a single ±60 % spike day
    that, before winsorisation + threshold, would dominate the empirical
    rolling-1y-IV. The new threshold should recognise the panel is too
    short and fall back to the simple-Itô point estimate.
    """
    # ~1.5 years of moderate-vol GBM (well below 2.5y threshold)
    panel_len = int(1.5 * TRADING_DAYS)
    rng = np.random.default_rng(123)
    sigma = 0.40
    dt = 1.0 / TRADING_DAYS
    shocks = rng.normal(0.0, sigma * math.sqrt(dt), size=panel_len)
    shocks[panel_len // 2] = 0.6  # +60% single-day spike
    levels = np.exp(np.cumsum(shocks))
    idx = pd.bdate_range(end=pd.Timestamp("2024-12-31"), periods=panel_len)
    tr = pd.Series(levels, index=idx, name="SPIKEY")

    out = forecast_decay_distribution(
        tr, beta=2.0, horizon_days=252,
        fallback_expected_decay=0.16,  # plausible 1·0.40² simple-Itô
    )
    assert out is not None
    assert out["model"] == "simple_ito_fallback", (
        f"expected fallback, got {out['model']} "
        f"(panel_len={panel_len}, threshold={int(_HORIZON_PANEL_RATIO_MIN * 252)})"
    )
    # Quantiles collapse to the simple-Itô point.
    assert abs(out["p10"] - out["p50"]) < 1e-6
    assert abs(out["p90"] - out["p50"]) < 1e-6
    assert abs(out["p50"] - 0.16) < 1e-3


def test_cap_mu_log_iv_clamps_extreme_centres_at_sigma_max():
    horizon = 252
    # Centre that implies σ_T = 3.0 (300% annualised) — past the cap.
    mu_extreme = math.log(3.0 ** 2)
    capped = _cap_mu_log_iv(mu_extreme, horizon)
    expected_cap = math.log(_LOG_IV_SIGMA_ANNUAL_CAP ** 2)
    assert abs(capped - expected_cap) < 1e-9

    # Centre below the cap is left alone.
    mu_quiet = math.log(0.20 ** 2)
    assert _cap_mu_log_iv(mu_quiet, horizon) == mu_quiet


def test_lognormal_decay_p50_is_bounded_when_centre_hits_cap():
    """Even when the implied σ would otherwise be 300 %, p50 must stay
    within ``c(β) · σ_max² · horizon_years`` after the cap.

    This is the SBTU-style case: the legacy model produced p50 = 8.27
    (≈ 283 % implied σ) on a thin-history single-name. With the cap, p50
    is bounded at c(β=2) · 1.5² = 2.25 (still extreme, but plausible).
    """
    out = forecast_decay_distribution.__globals__["_lognormal_decay_from_logiv"](
        _cap_mu_log_iv(math.log(8.0), 252),    # would imply σ = 283 %
        sigma_log_iv=0.07,
        beta=2.0,
        quantiles=(0.10, 0.50, 0.90),
    )
    bound = 1.0 * (_LOG_IV_SIGMA_ANNUAL_CAP ** 2) * 1.0
    assert out["p50"] <= bound + 1e-6, (
        f"p50={out['p50']} exceeded bound={bound}"
    )


def test_realized_variance_panel_winsorises_extreme_daily_returns():
    """A single 100 % up-day must not push RV above the winsor cap.

    Otherwise that one observation rolls through 252 consecutive 1y
    rolling sums and dominates the empirical anchor on thin-history
    single-name underlyings.
    """
    rng = np.random.default_rng(42)
    n = 800
    dt = 1.0 / TRADING_DAYS
    shocks = rng.normal(0.0, 0.20 * math.sqrt(dt), size=n)
    shocks[400] = 1.0  # +172 % single-day move (log-return = 1.0)
    levels = np.exp(np.cumsum(shocks))
    idx = pd.bdate_range(end=pd.Timestamp("2024-12-31"), periods=n)
    tr = pd.Series(levels, index=idx, name="SPIKE")

    panel = _realized_variance_panel(tr)
    assert panel is not None
    # No single RV observation in the panel may exceed the winsor cap.
    assert panel["RV"].max() <= _R2_WINSOR_CAP + 1e-12


def test_threshold_blocks_panel_just_under_K_min_horizon():
    """A panel with ``len < K_min · horizon`` must not produce empirical
    moments. We test directly so the threshold change is locked in.
    """
    # Just below the 2.5 · 252 = 630 threshold.
    just_below = int(_HORIZON_PANEL_RATIO_MIN * 252) - 1
    tr = _gbm_total_return(n_days=just_below + 30, sigma_annual=0.25, seed=8)
    panel = _realized_variance_panel(tr)
    if panel is None or len(panel) >= int(_HORIZON_PANEL_RATIO_MIN * 252):
        # Synthesised path was trimmed differently than expected; build
        # one explicitly at the boundary.
        rng = np.random.default_rng(9)
        n = just_below
        dt = 1.0 / TRADING_DAYS
        shocks = rng.normal(0.0, 0.25 * math.sqrt(dt), size=n)
        levels = np.exp(np.cumsum(shocks))
        idx = pd.bdate_range(end=pd.Timestamp("2024-12-31"), periods=n)
        tr2 = pd.Series(levels, index=idx)
        panel = _realized_variance_panel(tr2)
        assert panel is not None and len(panel) < int(_HORIZON_PANEL_RATIO_MIN * 252)
    moments = _empirical_log_iv_moments(panel, horizon_days=252)
    assert moments is None, (
        f"empirical moments should be blocked under threshold "
        f"(panel_len={len(panel)}, threshold={int(_HORIZON_PANEL_RATIO_MIN * 252)})"
    )


def test_threshold_passes_panel_above_K_min_horizon():
    tr = _gbm_total_return(n_days=int(_HORIZON_PANEL_RATIO_MIN * TRADING_DAYS) + 100,
                            sigma_annual=0.25, seed=13)
    panel = _realized_variance_panel(tr)
    assert panel is not None
    moments = _empirical_log_iv_moments(panel, horizon_days=252)
    assert moments is not None


def test_simple_ito_fallback_records_panel_length_in_n_obs():
    """When we fall back due to short history, ``n_obs`` should report
    the panel length we *did* see (not 0), so the dashboard can tell
    'no history at all' apart from 'history below threshold'.
    """
    panel_len = int(1.5 * TRADING_DAYS)
    rng = np.random.default_rng(77)
    dt = 1.0 / TRADING_DAYS
    shocks = rng.normal(0.0, 0.30 * math.sqrt(dt), size=panel_len)
    levels = np.exp(np.cumsum(shocks))
    idx = pd.bdate_range(end=pd.Timestamp("2024-12-31"), periods=panel_len)
    tr = pd.Series(levels, index=idx, name="SHORT")

    df = pd.DataFrame([
        {"ETF": "TSHORT", "Underlying": "SHORT", "Beta": 2.0,
         "expected_gross_decay_annual": 0.09},
    ])
    out = enrich_with_decay_distribution(df, {"SHORT": tr}, horizon_days=252)
    assert out["expected_gross_decay_dist_model"].iloc[0] == "simple_ito_fallback"
    n_obs = float(out["expected_gross_decay_dist_n_obs"].iloc[0])
    assert n_obs > 0, "panel length should be reported on fallback"
    # Allow some slack for the panel trimming inside _realized_variance_panel.
    assert n_obs < int(_HORIZON_PANEL_RATIO_MIN * 252), (
        f"reported n_obs={n_obs} should still be below threshold "
        f"{int(_HORIZON_PANEL_RATIO_MIN * 252)}"
    )
