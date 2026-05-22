"""SPX slide-risk phases 1–5: scaling, stress β, paths, panel integration."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from risk_dashboard.metrics import (
    _build_historical_spx_scenarios,
    _effective_spx_shock,
    _slide_horizon_scenario_totals,
    compute_slide_risk_panel,
)
from risk_dashboard.scenario_engine import aggregate_leg_scenario_pnl, scale_spx_shock_for_horizon
from risk_dashboard.spx_scenario import (
    build_spx_cumulative_path,
    historical_scenario_specs as historical_spx_scenario_specs,
)
from risk_dashboard.spx_shock_config import DEFAULT_SPX_SHOCK_CONFIG, load_spx_shock_config
from risk_dashboard.spx_stress_beta import stress_beta_to_spy, underlying_return_for_leg


def test_scale_spx_shock_rms_smaller_than_terminal_at_1m():
    shock = -0.20
    rms = scale_spx_shock_for_horizon(shock, "1M", mode="rms")
    term = scale_spx_shock_for_horizon(shock, "1M", mode="terminal")
    assert abs(rms) < abs(term)
    assert rms == pytest.approx(shock * math.sqrt(1.0 / 12.0))


def test_scale_spx_shock_t0_unscaled():
    assert scale_spx_shock_for_horizon(-0.10, "T+0", mode="rms") == pytest.approx(-0.10)


def test_stress_beta_widens_on_large_down_move():
    base = 1.5
    calm = stress_beta_to_spy(base, -0.02, stress_cfg=DEFAULT_SPX_SHOCK_CONFIG["stress_beta"])
    stressed = stress_beta_to_spy(base, -0.20, stress_cfg=DEFAULT_SPX_SHOCK_CONFIG["stress_beta"])
    assert abs(stressed) > abs(calm)


def test_underlying_return_per_leg_uses_effective_shock():
    row = {"beta_to_spy": 2.0, "underlying": "NVDA"}
    leg = {"symbol": "NVDL", "product_class": "letf_long", "beta_to_underlying": 2.0}
    u = underlying_return_for_leg(
        row,
        leg,
        -0.20,
        -0.20 * math.sqrt(1.0 / 12.0),
        stress_cfg={"enabled": False},
    )
    assert u == pytest.approx(2.0 * (-0.20 * math.sqrt(1.0 / 12.0)))


def test_historical_spx_path_endpoints():
    path = build_spx_cumulative_path(
        spx_start=0.0,
        spx_peak=-0.34,
        spx_end=0.15,
        peak_days=21,
        horizon_days=252,
        n_steps=12,
    )
    assert path[0] == pytest.approx(0.0)
    assert min(path) <= -0.30
    assert path[-1] == pytest.approx(0.15, abs=0.02)


def test_slide_horizon_rms_reduces_1m_beta_vs_terminal(tmp_path: Path):
    screener = tmp_path / "screener.csv"
    import pandas as pd

    pd.DataFrame(
        [
            {
                "ETF": "NVDL",
                "Underlying": "NVDA",
                "Delta": 2.0,
                "Delta_product_class": "letf_long",
                "vol_underlying_annual": 0.8,
                "borrow_fee_annual": 0.05,
            }
        ]
    ).to_csv(screener, index=False)
    flex = tmp_path / "flex.xml"
    flex.write_text(
        '<FlexQueryResponse><OpenPosition symbol="NVDL" position="-100000" '
        'markPrice="50" positionValue="-5000000" underlyingSymbol="NVDA" '
        'fxRateToBase="1" multiplier="1" /></FlexQueryResponse>',
        encoding="utf-8",
    )
    enriched = [
        {
            "underlying": "NVDA",
            "symbols": "NVDL",
            "net_notional_usd": -5_000_000.0,
            "beta_to_spy": 2.0,
            "sigma": 0.8,
            "legs": [
                {
                    "symbol": "NVDL",
                    "underlying": "NVDA",
                    "net_notional_usd": -5_000_000.0,
                    "product_class": "letf_long",
                    "leverage_k": 2.0,
                    "vol_underlying_annual": 0.8,
                    "borrow_fee_annual": 0.05,
                    "beta_to_underlying": 2.0,
                    "is_letf": True,
                }
            ],
            "is_letf": True,
        }
    ]
    cfg = dict(DEFAULT_SPX_SHOCK_CONFIG)
    rms = _slide_horizon_scenario_totals(
        enriched,
        etf_meta={},
        shock_pct=0.20,
        horizon_key="1M",
        spx_shock_cfg=cfg,
        horizon_shock_mode="rms",
        per_leg_beta=True,
    )
    term = _slide_horizon_scenario_totals(
        enriched,
        etf_meta={},
        shock_pct=0.20,
        horizon_key="1M",
        spx_shock_cfg=cfg,
        horizon_shock_mode="terminal",
        per_leg_beta=True,
    )
    assert abs(rms["beta_pnl_usd"]) < abs(term["beta_pnl_usd"])


def test_compute_slide_risk_panel_has_spx_phases_metadata():
    panel = compute_slide_risk_panel(
        factor_panel={
            "available": True,
            "rows": [
                {
                    "underlying": "NVDA",
                    "symbols": "NVDA",
                    "net_notional_usd": 100_000.0,
                    "gross_notional_usd": 100_000.0,
                    "beta_to_spy": 1.7,
                    "regime_vol_pct": 50.0,
                }
            ],
            "totals": {"net_beta_to_spy": 0.17},
        },
        nav_usd=1_000_000.0,
        flex_positions_xml=None,
        screener_csv=None,
    )
    spx = next(i for i in panel["indices"] if i["index"] == "SPX")
    assert spx.get("horizon_shock_mode") == "rms"
    assert spx.get("net_beta_to_spy") == pytest.approx(0.17)
    assert spx.get("historical_spx_scenarios")
    assert len(spx["historical_spx_scenarios"]) == len(historical_spx_scenario_specs())
    row = spx["shock_rows"][0]
    keys = [h["horizon_key"] for h in row["horizons"]]
    assert "12M-terminal" in keys
    t0 = next(h for h in row["horizons"] if h["horizon_key"] == "T+0")
    m1 = next(h for h in row["horizons"] if h["horizon_key"] == "1M")
    assert t0["decay_pnl_usd"] == 0.0
    assert m1.get("spx_shock_effective_pct") is not None


def test_build_historical_spx_scenarios_distinct_totals():
    enriched = [
        {
            "underlying": "SPY",
            "symbols": "SPY",
            "net_notional_usd": 50_000.0,
            "beta_to_spy": 1.0,
            "sigma": 0.15,
            "legs": [
                {
                    "symbol": "SPY",
                    "underlying": "SPY",
                    "net_notional_usd": 50_000.0,
                    "product_class": "passive_low_delta",
                    "leverage_k": 1.0,
                    "vol_underlying_annual": 0.15,
                    "borrow_fee_annual": 0.0,
                    "is_letf": False,
                }
            ],
            "is_letf": False,
        }
    ]
    cfg = dict(DEFAULT_SPX_SHOCK_CONFIG)
    hist = _build_historical_spx_scenarios(
        enriched,
        etf_meta={},
        nav_usd=1_000_000.0,
        spx_shock_cfg=cfg,
        variance_decomp=None,
    )
    totals = [h["total_pnl_usd"] for h in hist]
    assert len(set(round(t, 2) for t in totals)) > 1


def test_load_spx_shock_config_from_repo():
    root = Path(__file__).resolve().parents[2]
    cfg = load_spx_shock_config(root)
    assert cfg["horizon_shock_mode"] in ("terminal", "rms", "drift")
    assert cfg["stress_beta"]["enabled"] is True


def test_aggregate_zero_borrow_still_works_with_per_leg_path():
    legs = [
        {
            "symbol": "XY",
            "net_notional_usd": -100_000.0,
            "product_class": "letf_long",
            "leverage_k": 2.0,
            "vol_underlying_annual": 0.6,
            "borrow_fee_annual": 0.40,
        }
    ]
    agg = aggregate_leg_scenario_pnl(
        legs, underlying_return=-0.05, horizon_key="1M", zero_borrow=True
    )
    assert agg["borrow_pnl_usd"] == 0.0


def test_effective_spx_shock_override():
    assert _effective_spx_shock(0.10, "12M", horizon_shock_mode="rms", shock_scale_override=1.0) == pytest.approx(
        0.10
    )
