from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.calibrate_hedge_safe_rebalancing import (
    hedge_drift_diagnostics,
    load_cached_plan_timeline,
    select_sensitivity_defaults,
)


def test_load_cached_plan_timeline_normalizes_date_named_files(tmp_path):
    pd.DataFrame(
        [{
            "ETF": "AAA", "Underlying": "BBB", "sleeve": "core_leveraged",
            "Delta": 2.0, "long_usd": 200.0, "short_usd": -100.0,
            "gross_target_usd": 300.0,
        }]
    ).to_csv(tmp_path / "2026-03-02.csv", index=False)
    pd.DataFrame({"x": [1]}).to_csv(tmp_path / "not-a-date.csv", index=False)
    timeline = load_cached_plan_timeline(
        tmp_path, start=pd.Timestamp("2026-02-27"), end=pd.Timestamp("2026-03-03")
    )
    assert list(timeline) == [pd.Timestamp("2026-03-02")]
    assert timeline[pd.Timestamp("2026-03-02")].iloc[0]["ETF"] == "AAA"


def test_hedge_drift_diagnostics_counts_breaches_and_orphans():
    day = pd.Timestamp("2026-03-02")
    plan = pd.DataFrame(
        [{"ETF": "AAA", "Underlying": "BBB", "Delta": 2.0}]
    )
    pair_daily = pd.DataFrame(
        [
            {
                "date": day, "ETF": "AAA", "Underlying": "BBB",
                "active_plan_date": "2026-03-02", "etf_usd": -100.0,
                "underlying_usd": 250.0,
            },
            {
                "date": day + pd.offsets.BDay(1), "ETF": "AAA", "Underlying": "BBB",
                "active_plan_date": "2026-03-02", "etf_usd": 0.0,
                "underlying_usd": 50.0,
            },
        ]
    )
    drift, metrics = hedge_drift_diagnostics(
        pair_daily, {day: plan}, long_trigger=0.04, short_trigger=0.01
    )
    assert len(drift) == 2
    assert metrics["hedge_breach_group_days"] == 2
    assert metrics["hedge_too_long_group_days"] == 2
    assert metrics["orphan_pair_days"] == 1


def test_hedge_drift_uses_stock_residual_and_keeps_raw_all_sleeve_info():
    day = pd.Timestamp("2026-03-02")
    plan = pd.DataFrame([
        {"ETF": "CORE", "Underlying": "SHARED", "Delta": 2.0},
        {"ETF": "B4", "Underlying": "SHARED", "Delta": -2.0},
    ])
    pair_daily = pd.DataFrame([
        {
            "date": day, "ETF": "CORE", "Underlying": "SHARED",
            "sleeve": "core_leveraged", "active_plan_date": "2026-03-02",
            "etf_usd": -100.0, "underlying_usd": 200.0,
        },
        {
            "date": day, "ETF": "B4", "Underlying": "SHARED",
            "sleeve": "inverse_decay_bucket4", "active_plan_date": "2026-03-02",
            "etf_usd": -100.0, "underlying_usd": 0.0,
        },
    ])
    drift, metrics = hedge_drift_diagnostics(
        pair_daily, {day: plan}, long_trigger=0.04, short_trigger=0.01
    )
    assert metrics["hedge_breach_group_days"] == 0
    assert metrics["raw_all_sleeve_hedge_breach_group_days"] == 1
    assert drift.iloc[0]["breach"] == ""
    assert drift.iloc[0]["raw_all_sleeve_breach"] == "too_long"

    b4_only = pair_daily[pair_daily["ETF"] == "B4"]
    _, b4_metrics = hedge_drift_diagnostics(
        b4_only, {day: plan}, long_trigger=0.04, short_trigger=0.01
    )
    assert b4_metrics["hedge_breach_group_days"] == 0
    assert b4_metrics["raw_all_sleeve_hedge_breach_group_days"] == 1


def test_hedge_drift_uses_position_delta_for_decaying_pair_absent_from_plan():
    day = pd.Timestamp("2026-05-13")
    active_plan_day = pd.Timestamp("2026-05-12")
    active_plan = pd.DataFrame([
        {"ETF": "LIVE", "Underlying": "MSFT", "Delta": 2.0}
    ])
    pair_daily = pd.DataFrame([
        {
            "date": day, "ETF": "LIVE", "Underlying": "MSFT",
            "sleeve": "core_leveraged",
            "active_plan_date": str(active_plan_day.date()),
            "etf_usd": -100.0, "underlying_usd": 250.0, "Delta": 2.0,
        },
        {
            "date": day, "ETF": "MSFL", "Underlying": "MSFT",
            "sleeve": "core_leveraged",
            "active_plan_date": str(active_plan_day.date()),
            "etf_usd": -25.0, "underlying_usd": 0.0, "Delta": 2.0,
        },
    ])
    drift, metrics = hedge_drift_diagnostics(
        pair_daily, {active_plan_day: active_plan},
        long_trigger=0.04, short_trigger=0.01,
    )
    assert metrics["missing_delta_rows"] == 0
    assert metrics["missing_delta_group_days"] == 0
    assert metrics["hedge_breach_group_days"] == 0
    assert float(drift.iloc[0]["hedge_net_usd"]) == pytest.approx(0.0)

    missing = pair_daily.copy()
    missing.loc[missing["ETF"] == "MSFL", "Delta"] = np.nan
    _, missing_metrics = hedge_drift_diagnostics(
        missing, {active_plan_day: active_plan},
        long_trigger=0.04, short_trigger=0.01,
    )
    assert missing_metrics["missing_delta_rows"] == 1
    assert missing_metrics["missing_delta_group_days"] == 1


def test_sensitivity_selection_applies_safety_blocker_then_turnover():
    grid = pd.DataFrame([
        {
            "max_daily_turnover_pct": 0.08, "remaining_gap_rate": 0.15,
            "turnover_usd": 8_000_000, "txn_cost_usd": 10_000,
            "hedge_breach_group_days": 100, "orphan_pair_days": 0,
            "median_deployed_desired_gross_ratio": 0.40,
            "p10_deployed_desired_gross_ratio": 0.20,
            "ending_deployed_desired_gross_ratio": 0.35,
            "n_b4_cadence_rebals": 12,
        },
        {
            "max_daily_turnover_pct": 0.10, "remaining_gap_rate": 0.20,
            "turnover_usd": 9_000_000, "txn_cost_usd": 12_000,
            "hedge_breach_group_days": 200, "orphan_pair_days": 0,
            "median_deployed_desired_gross_ratio": 0.90,
            "p10_deployed_desired_gross_ratio": 0.60,
            "ending_deployed_desired_gross_ratio": 0.85,
            "n_b4_cadence_rebals": 12,
        },
        {
            "max_daily_turnover_pct": 0.15, "remaining_gap_rate": 0.25,
            "turnover_usd": 12_000_000, "txn_cost_usd": 15_000,
            "hedge_breach_group_days": 100, "orphan_pair_days": 0,
            "median_deployed_desired_gross_ratio": 0.95,
            "p10_deployed_desired_gross_ratio": 0.65,
            "ending_deployed_desired_gross_ratio": 0.90,
            "n_b4_cadence_rebals": 12,
        },
        {
            "max_daily_turnover_pct": 0.06, "remaining_gap_rate": 0.20,
            "turnover_usd": 8_500_000, "txn_cost_usd": 11_000,
            "hedge_breach_group_days": 150, "orphan_pair_days": 0,
            "median_deployed_desired_gross_ratio": 0.90,
            "p10_deployed_desired_gross_ratio": 0.60,
            "ending_deployed_desired_gross_ratio": 0.85,
            "n_b4_cadence_rebals": 11,
        },
    ])
    selected, ranked = select_sensitivity_defaults(
        grid, legacy_breach_group_days=1000
    )
    assert selected["max_daily_turnover_pct"] == 0.10
    assert bool(selected["hedge_safety_pass"])
    assert bool(selected["deployment_pass"])
    assert bool(selected["b4_cadence_pass"])
    assert bool(selected["turnover_target_pass"])
    assert ranked["selected"].sum() == 1
    assert not bool(ranked.loc[3, "b4_cadence_pass"])


def test_sensitivity_selection_rejects_missing_position_delta_groups():
    common = {
        "remaining_gap_rate": 0.20,
        "txn_cost_usd": 10_000,
        "hedge_breach_group_days": 10,
        "orphan_pair_days": 0,
        "median_deployed_desired_gross_ratio": 0.90,
        "p10_deployed_desired_gross_ratio": 0.60,
        "ending_deployed_desired_gross_ratio": 0.85,
        "n_b4_cadence_rebals": 29,
    }
    grid = pd.DataFrame([
        {
            **common, "max_daily_turnover_pct": 0.10,
            "turnover_usd": 5_000_000, "missing_delta_group_days": 1,
        },
        {
            **common, "max_daily_turnover_pct": 0.15,
            "turnover_usd": 6_000_000, "missing_delta_group_days": 0,
        },
    ])
    selected, ranked = select_sensitivity_defaults(
        grid, legacy_breach_group_days=1000
    )
    assert selected["max_daily_turnover_pct"] == 0.15
    assert not bool(ranked.loc[0, "hedge_safety_pass"])


def test_sensitivity_selection_minimizes_hard_band_breaches_before_turnover():
    common = {
        "max_daily_turnover_pct": 0.12,
        "remaining_gap_rate": 0.20,
        "txn_cost_usd": 20_000,
        "orphan_pair_days": 0,
        "missing_delta_rows": 0,
        "missing_delta_group_days": 0,
        "median_deployed_desired_gross_ratio": 0.85,
        "p10_deployed_desired_gross_ratio": 0.57,
        "ending_deployed_desired_gross_ratio": 0.82,
        "n_b4_cadence_rebals": 29,
    }
    grid = pd.DataFrame([
        {
            **common, "target_blend_alpha": 0.15,
            "turnover_usd": 6_592_000,
            "hedge_breach_group_days": 33,
            "max_abs_hedge_net_pct": 0.0945,
        },
        {
            **common, "target_blend_alpha": 0.25,
            "turnover_usd": 9_177_000,
            "hedge_breach_group_days": 9,
            "max_abs_hedge_net_pct": 0.0657,
        },
        {
            **common, "target_blend_alpha": 0.20,
            "turnover_usd": 8_000_000,
            "hedge_breach_group_days": 9,
            "max_abs_hedge_net_pct": 0.0800,
        },
    ])
    selected, _ = select_sensitivity_defaults(
        grid, legacy_breach_group_days=1309
    )
    assert selected["target_blend_alpha"] == 0.25
    assert selected["hedge_breach_group_days"] == 9
