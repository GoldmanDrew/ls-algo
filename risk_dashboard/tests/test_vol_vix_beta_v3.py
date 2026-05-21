"""Tests for v3 log-log vol→VIX elasticity estimator."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from risk_dashboard.vol_vix_beta_v3 import (
    DEFAULT_VOL_VIX_BETA,
    PRODUCT_CLASS_PRIORS,
    _resolve_product_class_prior,
    _weighted_log_log_ols,
    compute_vol_vix_betas_v3,
    ewma_realized_vol,
    shocked_sigma_multiplicative,
)


def test_weighted_log_log_ols_recovers_beta():
    rng = np.random.default_rng(1)
    n = 300
    true_b = 0.9
    log_vix = 0.18 + np.cumsum(rng.normal(0, 0.01, n))
    log_sigma = 0.5 + true_b * (log_vix - log_vix.mean()) + rng.normal(0, 0.02, n)
    w = np.ones(n) / n
    beta, _, _, n_obs, r2 = _weighted_log_log_ols(log_sigma, log_vix, w)
    assert n_obs == n
    assert beta is not None
    assert beta == pytest.approx(true_b, abs=0.15)
    assert r2 is not None and r2 > 0.3


def test_multiplicative_shock_convexity():
    base = 0.60
    beta = 1.0
    low = shocked_sigma_multiplicative(base, beta_vol_vix=beta, vix_current_pts=15.0, vix_new_pts=45.0)
    high = shocked_sigma_multiplicative(base, beta_vol_vix=beta, vix_current_pts=25.0, vix_new_pts=55.0)
    assert low > high  # 15→45 is 3× ratio vs 25→55 is 2.2×


def test_product_class_prior_vol_etp():
    prior, src = _resolve_product_class_prior("volatility_etp", "other", "SVIX")
    assert prior == PRODUCT_CLASS_PRIORS["volatility_etp"]
    assert src == "product_class"


def test_priors_anchor_low_data_names(monkeypatch):
    def _empty_fetch(symbols, **kwargs):
        idx = pd.date_range("2024-01-01", periods=320, freq="B")
        vix = pd.Series(18.0 + np.zeros(320), index=idx, name="^VIX")
        return {"^VIX": vix}

    monkeypatch.setattr(
        "risk_dashboard.vol_vix_beta_v3._fetch_yfinance_closes",
        _empty_fetch,
    )
    pack = compute_vol_vix_betas_v3(
        ["UNKNOWN"],
        underlying_meta={"UNKNOWN": {"product_class": "broad", "sector": "other"}},
    )
    res = pack["betas"]["UNKNOWN"]
    assert res.beta_vol_vix == pytest.approx(PRODUCT_CLASS_PRIORS["broad"])
    assert res.provenance == "default"


def test_compute_v3_synthetic_history(monkeypatch):
    def _fake_fetch(symbols, **kwargs):
        idx = pd.date_range("2023-01-01", periods=400, freq="B")
        rng = np.random.default_rng(3)
        vix_px = 18 + np.cumsum(rng.normal(0, 0.25, 400))
        vix = pd.Series(vix_px, index=idx, name="^VIX")
        spy_px = np.exp(np.cumsum(rng.normal(0, 0.012, 400)))
        spy = pd.Series(spy_px, index=idx, name="SPY")
        out = {"^VIX": vix, "^VIX9D": vix * 0.95, "^VIX3M": vix * 1.05, "^VVIX": vix * 0.8}
        for s in symbols:
            if s not in out:
                out[s] = spy.rename(s)
        return out

    monkeypatch.setattr(
        "risk_dashboard.vol_vix_beta_v3._fetch_yfinance_closes",
        _fake_fetch,
    )
    pack = compute_vol_vix_betas_v3(
        ["SPY"],
        underlying_meta={"SPY": {"product_class": "broad", "sector": "broad"}},
    )
    assert pack["estimator_version"] == "v3_log_elasticity"
    res = pack["betas"]["SPY"]
    assert res.beta_vol_vix is not None
    assert res.beta_vol_vix >= 0.1
    assert pack["n_computed"] >= 1
    assert "term_structure" in pack


def test_ewma_vol_positive():
    idx = pd.date_range("2024-01-01", periods=100, freq="B")
    prices = pd.Series(np.exp(np.cumsum(np.random.default_rng(2).normal(0, 0.02, 100))), index=idx)
    vol = ewma_realized_vol(prices)
    assert vol.dropna().iloc[-1] > 0
