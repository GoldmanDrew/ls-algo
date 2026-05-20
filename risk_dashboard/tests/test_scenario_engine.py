"""Tests for etf-dashboard-aligned scenario engine."""

from __future__ import annotations

import math

import pytest

from risk_dashboard.scenario_engine import (
    SLIDE_SCENARIO_HORIZONS,
    aggregate_leg_scenario_pnl,
    build_shock_rows,
    estimate_etf_return,
    estimate_income_style_return,
    horizon_to_years,
    model_leg_return,
)


def test_horizon_to_years_matches_etf_dashboard():
    assert horizon_to_years("1M") == pytest.approx(1 / 12)
    assert horizon_to_years("3M") == pytest.approx(3 / 12)
    assert horizon_to_years("6M") == pytest.approx(6 / 12)
    assert horizon_to_years("12M") == pytest.approx(1.0)
    assert horizon_to_years("1Y") == pytest.approx(1.0)


def test_estimate_etf_return_zero_shock_decay_inverse_letf():
    """Inverse LETF vol drag at 1M matches etf-dashboard log-return formula."""
    sigma = 1.0282
    leverage = -2.802
    t = horizon_to_years("1M")
    result = estimate_etf_return(
        leverage=leverage,
        underlying_return=0.0,
        sigma_annual=sigma,
        horizon_years=t,
    )
    assert result.ok is True
    drag_log = 0.5 * leverage * (leverage - 1.0) * (sigma**2) * t
    expected = math.exp(-drag_log) - 1.0
    assert result.total_return == pytest.approx(expected, abs=1e-9)
    assert result.total_return == pytest.approx(result.price_return + result.decay_return, abs=1e-9)


def test_estimate_etf_return_includes_borrow_on_short():
    t = horizon_to_years("1M")
    base = estimate_etf_return(
        leverage=2.0,
        underlying_return=0.0,
        sigma_annual=0.5,
        horizon_years=t,
        is_short=False,
    )
    short = estimate_etf_return(
        leverage=2.0,
        underlying_return=0.0,
        sigma_annual=0.5,
        horizon_years=t,
        is_short=True,
        borrow_annual=0.48,
    )
    assert short.borrow_return == pytest.approx(0.48 * t, abs=1e-9)
    assert short.total_return == pytest.approx(base.total_return + short.borrow_return, abs=1e-9)
    assert -500_000 * short.borrow_return < 0


def test_build_shock_rows_log_space():
    rows = build_shock_rows(1.0, horizon_to_years("1M"))
    zero = next(r for r in rows if r["sigma_multiple"] == 0.0)
    assert zero["underlying_return"] == pytest.approx(0.0, abs=1e-12)


def test_aggregate_leg_scenario_decomposition_sums():
    legs = [
        {
            "symbol": "APLX",
            "net_notional_usd": -1_000_000.0,
            "product_class": "letf_long",
            "leverage_k": 2.0,
            "vol_underlying_annual": 0.7,
            "borrow_fee_annual": 0.05,
        }
    ]
    agg = aggregate_leg_scenario_pnl(
        legs,
        underlying_return=-0.10,
        horizon_key="1M",
    )
    assert agg["total_pnl_usd"] == pytest.approx(
        agg["beta_pnl_usd"]
        + agg["decay_pnl_usd"]
        + agg["borrow_pnl_usd"]
        + agg["distribution_pnl_usd"],
        rel=1e-6,
        abs=1.0,
    )


def test_model_leg_yieldboost_when_income_present():
    leg = {
        "symbol": "FBYY",
        "net_notional_usd": -500_000.0,
        "product_class": "income_yieldboost",
        "leverage_k": 0.5,
        "vol_underlying_annual": 1.0,
        "income_distributions_annual": 0.35,
        "borrow_fee_annual": 0.50,
        "expense_ratio_annual": 0.0099,
    }
    result = model_leg_return(
        leg=leg,
        underlying_return=0.0,
        horizon_key="1M",
    )
    assert result.ok is True
    assert result.model == "yieldboost"
    assert result.borrow_return < 0


def test_income_style_short_includes_borrow_drag():
    yb = estimate_income_style_return(
        underlying_return=0.0,
        sigma_annual=0.8,
        annual_income_yield=0.30,
        horizon_years=horizon_to_years("3M"),
        annual_borrow_cost=0.40,
        is_short=True,
    )
    assert yb is not None
    assert yb.borrow_return == pytest.approx(-horizon_to_years("3M") * 0.40, abs=1e-9)
    assert yb.total_return == pytest.approx(
        yb.price_return + yb.decay_return + yb.borrow_return + yb.distribution_return,
        abs=1e-9,
    )


def test_slide_scenario_horizons_order():
    assert list(SLIDE_SCENARIO_HORIZONS) == ["1M", "3M", "6M", "12M"]
