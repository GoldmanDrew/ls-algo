"""Tests for vol→VIX beta and shocked σ mapping."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from risk_dashboard.vol_vix_beta import (
    DEFAULT_VOL_VIX_BETA,
    VolVixBetaResult,
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


def test_compute_vol_vix_betas_uses_default_without_network(monkeypatch):
    def _fake_fetch(symbols, **kwargs):
        idx = pd.date_range("2024-01-01", periods=80, freq="B")
        rng = np.random.default_rng(2)
        vix = pd.Series(18 + rng.normal(0, 1, 80), index=idx, name="^VIX")
        spy = pd.Series(np.exp(np.cumsum(rng.normal(0, 0.01, 80))), index=idx, name="SPY")
        return {"^VIX": vix, "SPY": spy}

    monkeypatch.setattr(
        "risk_dashboard.vol_vix_beta._fetch_yfinance_closes",
        _fake_fetch,
    )
    pack = compute_vol_vix_betas(["SPY"])
    assert pack["vix_current_pts"] == pytest.approx(float(_fake_fetch([])["^VIX"].iloc[-1]), abs=0.01)
    assert "SPY" in pack["betas"]
    assert pack["betas"]["SPY"].beta_vol_vix is not None
