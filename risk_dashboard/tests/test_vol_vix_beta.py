"""Tests for vol→VIX beta (v2 diff-OLS) and shocked σ mapping."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from daily_screener import BETA_SHRINK_K_BASE, _ar1_n_eff
from risk_dashboard.vol_vix_beta import (
    BETA_VOL_VIX_MAX,
    DEFAULT_VOL_VIX_BETA,
    ESTIMATOR_VERSION,
    PRIOR_VOLATILITY_ETP,
    _clip_beta,
    _diff_ols_with_stats,
    _product_class_prior,
    _resolve_beta_prior,
    _shrink_vol_vix_beta,
    compute_vol_vix_betas,
    rolling_annualized_vol,
    shocked_sigma_annual,
    vix_to_decimal,
)


def test_vix_to_decimal():
    assert vix_to_decimal(20.5) == pytest.approx(0.205)


def test_shocked_sigma_moves_with_beta():
    base = 0.60
    out = shocked_sigma_annual(
        base,
        beta_vol_vix=0.8,
        vix_current_decimal=0.20,
        vix_shock_pts=10.0,
    )
    assert out == pytest.approx(base + 0.8 * 0.10, abs=1e-9)
    assert shocked_sigma_annual(base, beta_vol_vix=0.8, vix_current_decimal=0.20, vix_shock_pts=0.0) == pytest.approx(base)


def test_rolling_annualized_vol_positive():
    idx = pd.date_range("2024-01-01", periods=80, freq="B")
    prices = pd.Series(np.exp(np.cumsum(np.random.default_rng(1).normal(0, 0.02, 80))), index=idx)
    vol = rolling_annualized_vol(prices)
    assert vol.dropna().iloc[-1] > 0


def test_product_class_priors():
    assert _product_class_prior("volatility_etp") == PRIOR_VOLATILITY_ETP
    assert _product_class_prior("broad") == 0.5
    assert _product_class_prior("") == DEFAULT_VOL_VIX_BETA


def test_clip_beta_bounds():
    assert _clip_beta(-0.5) == 0.0
    assert _clip_beta(3.0) == BETA_VOL_VIX_MAX


def test_diff_ols_recovers_known_beta():
    rng = np.random.default_rng(42)
    n = 280
    idx = pd.date_range("2023-01-01", periods=n, freq="B")
    true_b = 0.85
    vix = pd.Series(0.18 + np.cumsum(rng.normal(0, 0.002, n)), index=idx)
    vol = pd.Series(0.40 + true_b * (vix - vix.iloc[0]) + rng.normal(0, 0.001, n), index=idx)
    beta, _, _, n_obs, r2, _, _ = _diff_ols_with_stats(vol, vix, history_days=252)
    assert n_obs >= 30
    assert beta is not None
    assert beta == pytest.approx(true_b, abs=0.15)
    assert r2 is not None and r2 > 0.5


def test_shrinkage_weight_matches_ar1_shape():
    rng = np.random.default_rng(7)
    n = 200
    d = rng.normal(0, 0.01, n)
    beta_ols = 0.9
    prior = 0.75
    beta, shrunk, n_eff, rho = _shrink_vol_vix_beta(beta_ols, n, d, beta_prior=prior)
    _, n_eff_ref = _ar1_n_eff(d)
    k = BETA_SHRINK_K_BASE * max(1.0, prior**2)
    w = n_eff_ref / (n_eff_ref + k)
    expected = w * beta_ols + (1.0 - w) * prior
    assert n_eff == n_eff_ref
    assert beta == pytest.approx(expected, rel=1e-9)
    assert shrunk is True or w < 1.0


def test_volatility_etp_prior_when_no_data(monkeypatch):
    def _empty_fetch(symbols, **kwargs):
        return {}

    monkeypatch.setattr(
        "risk_dashboard.vol_vix_beta._fetch_yfinance_closes",
        _empty_fetch,
    )
    pack = compute_vol_vix_betas(
        ["SVIX"],
        underlying_meta={"SVIX": {"product_class": "volatility_etp", "sector": "other"}},
    )
    res = pack["betas"]["SVIX"]
    assert res.beta_vol_vix == pytest.approx(PRIOR_VOLATILITY_ETP)
    assert res.provenance == "default"


def test_compute_vol_vix_betas_computed_with_synthetic_history(monkeypatch):
    def _fake_fetch(symbols, **kwargs):
        idx = pd.date_range("2023-01-01", periods=320, freq="B")
        rng = np.random.default_rng(2)
        vix_px = 18 + np.cumsum(rng.normal(0, 0.3, 320))
        vix = pd.Series(vix_px, index=idx, name="^VIX")
        spy_px = np.exp(np.cumsum(rng.normal(0, 0.012, 320)))
        spy = pd.Series(spy_px, index=idx, name="SPY")
        out = {"^VIX": vix}
        for s in symbols:
            if s != "^VIX":
                out[s] = spy.rename(s)
        return out

    monkeypatch.setattr(
        "risk_dashboard.vol_vix_beta._fetch_yfinance_closes",
        _fake_fetch,
    )
    pack = compute_vol_vix_betas(
        ["SPY"],
        underlying_meta={"SPY": {"product_class": "broad", "sector": "broad"}},
    )
    assert pack["estimator_version"] == ESTIMATOR_VERSION
    assert pack["vix_current_pts"] == pytest.approx(float(_fake_fetch([])["^VIX"].iloc[-1]), abs=0.05)
    res = pack["betas"]["SPY"]
    assert res.beta_vol_vix is not None
    assert res.provenance.startswith("computed")
    assert pack["n_computed"] >= 1


def test_resolve_beta_prior_sector_mean():
    prior, src = _resolve_beta_prior(
        "NVDA",
        product_class="",
        sector="semis",
        sector_means={"semis": 0.82},
    )
    assert prior == 0.82
    assert src == "sector_mean"
