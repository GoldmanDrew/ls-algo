"""Tests for variance decomposition layer."""

from __future__ import annotations

import pytest

from risk_dashboard.variance_decomp import (
    compute_idio_sigma,
    corr_lift,
    shocked_sigma_variance_decomp,
    vix_to_realized_spx_sigma,
)


def test_vix_to_realized_spx():
    assert vix_to_realized_spx_sigma(20.0, vrp_factor=0.80) == pytest.approx(0.16)


def test_idio_decomp():
    sys, idio, total = compute_idio_sigma(0.50, beta_spy=1.2, sigma_spx=0.16)
    assert total == pytest.approx(0.50)
    assert sys == pytest.approx(0.192)
    assert idio > 0


def test_shocked_sigma_decomp_vs_elasticity():
    base = 0.55
    decomp = shocked_sigma_variance_decomp(
        base,
        beta_spy=1.5,
        beta_vol_vix=0.5,
        vix_current_pts=18.0,
        vix_new_pts=45.0,
        use_decomp=True,
    )
    elastic = shocked_sigma_variance_decomp(
        base,
        beta_spy=None,
        beta_vol_vix=1.0,
        vix_current_pts=18.0,
        vix_new_pts=45.0,
        use_decomp=False,
    )
    assert decomp > base
    assert elastic > base


def test_corr_lift_increases_idio():
    s_calm = shocked_sigma_variance_decomp(
        0.50,
        beta_spy=1.0,
        beta_vol_vix=1.0,
        vix_current_pts=18.0,
        vix_new_pts=20.0,
        corr_lift_override=1.0,
        use_decomp=True,
    )
    s_stress = shocked_sigma_variance_decomp(
        0.50,
        beta_spy=1.0,
        beta_vol_vix=1.0,
        vix_current_pts=18.0,
        vix_new_pts=35.0,
        corr_lift_override=1.85,
        use_decomp=True,
    )
    assert s_stress > s_calm
