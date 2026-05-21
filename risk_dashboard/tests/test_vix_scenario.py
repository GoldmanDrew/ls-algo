"""Tests for VIX scenario engine (path integral, historical)."""

from __future__ import annotations

import pytest

from risk_dashboard.vix_scenario import (
    borrow_rate_vix_stress,
    corr_lift,
    effective_sigma_from_path_integral,
    historical_scenario_specs,
    resolve_shocked_sigma,
    sigma_sq_path_integral,
)


def test_corr_lift_piecewise():
    assert corr_lift(15) == pytest.approx(1.0)
    assert corr_lift(25) == pytest.approx(1.35)
    assert corr_lift(40) == pytest.approx(1.85)


def test_sigma_path_integral_sustained_equals_terminal_sq():
    s = 0.6
    t = 1.0
    integral = sigma_sq_path_integral(s, s, kappa=5.0, horizon_years=t)
    assert integral == pytest.approx(s * s * t)


def test_spike_revert_integral_smaller_than_sustained():
    s0 = 0.9
    sinf = 0.5
    t = 1.0
    spike = sigma_sq_path_integral(s0, sinf, kappa=5.0, horizon_years=t)
    sustained = s0 * s0 * t
    assert spike < sustained


def test_effective_sigma_from_integral():
    integral = 0.36  # 0.6^2 * 1y
    eff = effective_sigma_from_path_integral(integral, 1.0)
    assert eff == pytest.approx(0.6)


def test_borrow_vix_stress_widens():
    base = 0.05
    calm = borrow_rate_vix_stress(base, vix_pts=15.0)
    stress = borrow_rate_vix_stress(base, vix_pts=35.0, is_htb=True)
    assert stress > calm


def test_historical_scenarios_catalog():
    specs = historical_scenario_specs()
    assert len(specs) == 5
    keys = {s.key for s in specs}
    assert "mar_2020_covid" in keys


def test_resolve_shocked_sigma_multiplicative_fallback():
    pack = {
        "estimator_version": "v3_log_elasticity",
        "vix_current_pts": 20.0,
        "betas": {
            "SPY": {
                "beta_vol_vix": 1.0,
                "beta_vol_vix_low": 0.9,
                "beta_vol_vix_high": 1.1,
            }
        },
        "config": {},
    }
    s = resolve_shocked_sigma(
        sigma_base=0.5,
        underlying="SPY",
        vol_vix_pack=pack,
        variance_decomp=None,
        vix_current_pts=20.0,
        vix_new_pts=40.0,
        mode="sustained",
    )
    assert s == pytest.approx(0.5 * (40 / 20) ** 1.1, rel=0.02)
