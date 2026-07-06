"""Tests for borrow stress path integration and carry validation."""

from __future__ import annotations

import pytest

from risk_dashboard.borrow_stress import (
    build_vix_cumulative_path,
    borrow_rate_vix_stress,
    effective_annual_borrow_from_vix_path,
    event_borrow_lift_at_day,
)
from risk_dashboard.carry_validation import compute_carry_validation
from risk_dashboard.spx_scenario import (
    PATH_STEPS_DEFAULT,
    build_spx_cumulative_path,
    rolling_realized_vol_from_spx_path,
)


def test_build_spx_daily_path_has_many_steps():
    path = build_spx_cumulative_path(
        spx_start=0.0,
        spx_peak=-0.34,
        spx_end=0.15,
        peak_days=21,
        horizon_days=252,
        n_steps=PATH_STEPS_DEFAULT,
    )
    assert len(path) == PATH_STEPS_DEFAULT + 1
    deltas = [path[i] - path[i - 1] for i in range(1, len(path))]
    assert min(deltas) > -0.10


def test_vix_path_integrated_borrow_below_peak_flat_rate():
    path = build_vix_cumulative_path(
        vix_start=14.0,
        vix_peak=82.0,
        vix_end=35.0,
        peak_days=45,
        horizon_days=252,
        n_steps=252,
    )
    base = 0.30
    integrated = effective_annual_borrow_from_vix_path(
        base, path, tier="htb", borrow_lift=2.0, peak_days=45
    )
    peak_flat = borrow_rate_vix_stress(base, vix_pts=82.0, tier="htb", borrow_lift=2.0)
    assert integrated < peak_flat


def test_event_lift_only_near_peak():
    assert event_borrow_lift_at_day(45, borrow_lift=2.0, peak_days=45, window_days=45) == 2.0
    assert event_borrow_lift_at_day(200, borrow_lift=2.0, peak_days=45, window_days=45) == 1.0


def test_rolling_realized_vol_rises_on_crash_path():
    path = build_spx_cumulative_path(
        spx_start=0.0,
        spx_peak=-0.34,
        spx_end=0.15,
        peak_days=21,
        horizon_days=252,
        n_steps=252,
    )
    calm = rolling_realized_vol_from_spx_path(path, 5, base_sigma=0.25)
    after = rolling_realized_vol_from_spx_path(path, 25, base_sigma=0.25)
    assert after >= calm


def test_carry_validation_runs_on_repo_history():
    root = pytest.importorskip("pathlib").Path(__file__).resolve().parents[1]
    val = compute_carry_validation(predicted_carry_pct_nav=0.30, repo_root=root)
    assert "available" in val
