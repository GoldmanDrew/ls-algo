"""Regression tests for production actual backtest audit fixes."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.production_actual_backtest import (
    B5_SLEEVE,
    _b5_sleeve_nav,
    _pair_stats_from_navs,
    _stock_sleeve_nav,
    _targets_from_plan,
    normalize_plan,
    prepare_screened_for_gtp_approx,
    simulate_book_from_plan_timeline,
)
from scripts.sizing_tilt_cadence_bt import load_price_panel, pair_daily_returns


def test_prepare_screened_uses_borrow_avg_when_finite():
    df = pd.DataFrame(
        {
            "ETF": ["AAA", "BBB", "CCC"],
            "borrow_current": [0.10, 0.20, 0.30],
            "borrow_avg_annual": [0.05, np.nan, 0.40],
            "net_edge_p50_annual": [0.5, 0.6, 0.7],
        }
    )
    out = prepare_screened_for_gtp_approx(df)
    assert float(out.loc[0, "borrow_current"]) == pytest.approx(0.05)
    assert float(out.loc[1, "borrow_current"]) == pytest.approx(0.20)
    assert float(out.loc[2, "borrow_current"]) == pytest.approx(0.40)
    assert list(out["borrow_used_for_sizing"]) == [
        "borrow_avg_annual",
        "borrow_current",
        "borrow_avg_annual",
    ]
    assert float(out.loc[0, "net_edge_p50_annual"]) == pytest.approx(0.5)


def test_pair_eq_not_wiped_on_nan_friday():
    """Inactive names keep equity across NaN Fridays (SMYY/COYY bug)."""
    cal = pd.bdate_range("2025-05-01", periods=80)
    # Pair A live entire window; pair B starts halfway through
    a_px = pd.Series(100.0 * (1.001 ** np.arange(len(cal))), index=cal)
    b_px = pd.Series(50.0 * (1.0005 ** np.arange(len(cal))), index=cal)
    late = cal[40:]
    c_px = pd.Series(np.nan, index=cal)
    c_px.loc[late] = 20.0 * (1.002 ** np.arange(len(late)))
    d_px = pd.Series(np.nan, index=cal)
    d_px.loc[late] = 10.0 * (1.001 ** np.arange(len(late)))

    panel = {
        "AAA": pd.DataFrame({"a_px": a_px, "b_px": b_px}),
        "BBB": pd.DataFrame({"a_px": c_px, "b_px": d_px}),
    }
    uni = pd.DataFrame(
        [
            {
                "ETF": "AAA",
                "Underlying": "X",
                "sleeve": "yieldboost",
                "long_usd": 40_000.0,
                "short_usd": -60_000.0,
                "gross_target_usd": 100_000.0,
                "borrow_current": 0.10,
            },
            {
                "ETF": "BBB",
                "Underlying": "Y",
                "sleeve": "yieldboost",
                "long_usd": 40_000.0,
                "short_usd": -60_000.0,
                "gross_target_usd": 100_000.0,
                "borrow_current": 0.10,
            },
        ]
    )
    nav, meta, stats = _stock_sleeve_nav(
        uni,
        panel,
        sleeve="yieldboost",
        start=cal[0],
        budget_usd=200_000.0,
        enter_band_pct=0.12,
        slippage_bps=0.0,
    )
    assert len(nav) > 40
    assert meta["n_pairs"] == 2
    bbb = stats[stats["ETF"] == "BBB"]
    assert len(bbb) == 1
    # Must not be a fake -100% wipe
    assert float(bbb["ret"].iloc[0]) > -0.99
    assert float(bbb["end_usd"].iloc[0]) > 0


def test_pair_stats_drops_leading_zeros_not_corrupt():
    idx = pd.bdate_range("2025-05-01", periods=30)
    s = pd.Series(0.0, index=idx)
    s.iloc[10:] = np.linspace(1000, 1100, 20)
    stats = _pair_stats_from_navs(
        {"XYZ": s},
        sleeve="inverse_decay_bucket4",
        und_by_etf={"XYZ": "UND"},
        start_usd_by_etf={"XYZ": 1000.0},
    )
    assert len(stats) == 1
    assert float(stats["start_usd"].iloc[0]) == 1000.0
    assert float(stats["ret"].iloc[0]) == pytest.approx(0.1, rel=1e-3)
    assert not bool(stats["stats_corrupt"].iloc[0])


def test_pair_daily_returns_borrow_on_short_etf():
    idx = pd.bdate_range("2025-01-01", periods=5)
    px = pd.DataFrame({"a_px": [10, 10, 10, 10, 10], "b_px": [100, 100, 100, 100, 100]}, index=idx)
    row = pd.Series(
        {
            "long_usd": 25_000.0,
            "short_usd": -75_000.0,
            "borrow_current": 0.252,  # 25.2%/yr → 0.001/day on short leg
        }
    )
    r0 = pair_daily_returns(row, px, borrow_on_etf=False, borrow_on_underlying=False)
    r1 = pair_daily_returns(row, px, borrow_on_etf=True, borrow_on_underlying=False)
    # Flat prices → only borrow drag differs
    assert float(r0.dropna().iloc[-1]) == pytest.approx(0.0, abs=1e-12)
    assert float(r1.dropna().iloc[-1]) < 0


def test_coyy_split_adjusted_no_500pct_day():
    panel = load_price_panel("2026-07-10")
    assert "COYY" in panel
    r = panel["COYY"]["a_px"].pct_change().loc["2026-05-28":"2026-06-12"]
    assert float(r.abs().max()) < 0.5


def test_b5_uses_carry_engine_not_b4():
    """Smoke: B5 path returns carry-engine meta (skips if vol panel unavailable)."""
    uni = pd.DataFrame(
        [
            {
                "ETF": "UVIX",
                "Underlying": "SVIX",
                "sleeve": B5_SLEEVE,
                "long_usd": -7000.0,
                "short_usd": -3500.0,
                "gross_target_usd": 10500.0,
                "borrow_current": 0.03,
            }
        ]
    )
    nav, meta, stats = _b5_sleeve_nav(
        uni,
        start=pd.Timestamp("2025-05-01"),
        budget_usd=10500.0,
        slippage_bps=20.0,
    )
    if meta.get("skipped"):
        pytest.skip(f"vol panel unavailable: {meta.get('reason')}")
    assert "bucket5_carry" in str(meta.get("engine", ""))
    assert "rho" in meta
    assert float(meta["rho"]) == pytest.approx(2.0)
    # Must not be the old B4 dynamic-h label
    assert "bucket4_dynamic" not in str(meta.get("engine", ""))
    assert len(nav) > 40


def _simple_plan(date: pd.Timestamp, *, long_usd: float = 200.0, short_usd: float = -200.0):
    return normalize_plan(
        pd.DataFrame(
            [
                {
                    "ETF": "AAA",
                    "Underlying": "BBB",
                    "sleeve": "core_leveraged",
                    "long_usd": long_usd,
                    "short_usd": short_usd,
                    "gross_target_usd": abs(long_usd) + abs(short_usd),
                    "borrow_current": 0.0,
                }
            ]
        ),
        source_date=str(date.date()),
    )


def test_replay_preserves_four_x_gross_and_next_close_timing():
    cal = pd.bdate_range("2025-01-01", periods=30)
    panel = {
        "AAA": pd.DataFrame(
            {
                "a_px": 100.0 * (0.99 ** np.arange(len(cal))),
                "b_px": 100.0,
            },
            index=cal,
        )
    }
    nav, audit, meta, _, daily = simulate_book_from_plan_timeline(
        {cal[0]: _simple_plan(cal[0])},
        panel,
        budgets={"core_leveraged": 400.0},
        capital_usd=100.0,
        start=cal[0],
        slippage_bps=0.0,
        commission_per_share=0.0,
        margin_rate_annual=0.0,
        min_trade_usd=0.0,
    )
    # Plan stamped at the first close executes at the second close.  It cannot
    # earn either first-day or second-day close-to-close return.
    assert nav.iloc[0] == pytest.approx(100.0)
    assert nav.iloc[1] == pytest.approx(100.0)
    assert nav.iloc[2] == pytest.approx(102.0)  # short 50% ETF leg * -1% * 4x gross
    assert float(daily.loc[daily["date"] == cal[1], "gross_leverage"].iloc[0]) == pytest.approx(4.0)
    assert pd.Timestamp(audit.iloc[0]["date"]) == cal[1]
    assert meta["execution_lag_sessions"] == 1


def test_replay_does_not_use_latest_plan_leg_mix_in_history():
    cal = pd.bdate_range("2025-01-01", periods=35)
    panel = {
        "AAA": pd.DataFrame(
            {
                "a_px": 100.0 * (1.01 ** np.arange(len(cal))),
                "b_px": 100.0 * (0.99 ** np.arange(len(cal))),
            },
            index=cal,
        )
    }
    # First plan is long underlying / short ETF (negative on this tape).  The
    # later plan reverses it.  Earlier P&L must retain the first plan's signs.
    p0 = _simple_plan(cal[0], long_usd=200.0, short_usd=-200.0)
    p1 = _simple_plan(cal[15], long_usd=-200.0, short_usd=200.0)
    _, _, _, _, daily = simulate_book_from_plan_timeline(
        {cal[0]: p0, cal[15]: p1},
        panel,
        budgets={"core_leveraged": 400.0},
        capital_usd=100.0,
        start=cal[0],
        slippage_bps=0.0,
        commission_per_share=0.0,
        margin_rate_annual=0.0,
        min_trade_usd=0.0,
    )
    early = daily[(daily["date"] >= cal[2]) & (daily["date"] < cal[15])]
    assert (early["daily_price_pnl"] < 0).all()


def test_replay_charges_opening_cost_and_reconciles_daily_pnl():
    cal = pd.bdate_range("2025-01-01", periods=25)
    panel = {"AAA": pd.DataFrame({"a_px": 100.0, "b_px": 50.0}, index=cal)}
    nav, _, _, pair_stats, daily = simulate_book_from_plan_timeline(
        {cal[0]: _simple_plan(cal[0])},
        panel,
        budgets={"core_leveraged": 400.0},
        capital_usd=100.0,
        start=cal[0],
        slippage_bps=10.0,
        commission_per_share=0.0035,
        margin_rate_annual=0.0,
        min_trade_usd=0.0,
    )
    open_day = daily[daily["daily_txn_cost"] > 0].iloc[0]
    assert open_day["daily_txn_cost"] > 0
    assert nav.loc[open_day["date"]] < 100.0
    assert float(daily["pnl_recon_residual"].abs().max()) < 1e-9
    assert float(pair_stats["txn_cost_usd"].sum()) == pytest.approx(float(daily["daily_txn_cost"].sum()))


def test_plan_schema_maps_short_usd_to_etf_and_long_usd_to_underlying():
    cal = pd.bdate_range("2025-01-01", periods=3)
    raw = pd.DataFrame(
        [
            {
                "ETF": "AAA",
                "Underlying": "BBB",
                "sleeve": "core_leveraged",
                "long_usd": 300.0,
                "short_usd": -100.0,
                "underlying_target_usd": 300.0,
                "etf_target_usd": -100.0,
                "gross_target_usd": 400.0,
                "borrow_current": 0.10,
                "underlying_borrow_annual": 0.02,
            }
        ]
    )
    plan = normalize_plan(raw, source_date="2025-01-01")
    panel = {"AAA": pd.DataFrame({"a_px": 100.0, "b_px": 50.0}, index=cal)}
    target = _targets_from_plan(
        plan,
        budgets={"core_leveraged": 400.0},
        panel=panel,
        equity=100.0,
        capital_usd=100.0,
        target_notional_mode="fixed_plan_usd",
    )
    assert float(target.at["AAA", "etf_usd"]) == pytest.approx(-100.0)
    assert float(target.at["AAA", "underlying_usd"]) == pytest.approx(300.0)
    assert float(target.at["AAA", "borrow_underlying"]) == pytest.approx(0.02)


def test_replay_phase2b_band_skips_small_existing_resize():
    cal = pd.bdate_range("2025-01-01", periods=30)
    panel = {"AAA": pd.DataFrame({"a_px": 100.0, "b_px": 50.0}, index=cal)}
    p0 = normalize_plan(
        pd.DataFrame([{
            "ETF": "AAA", "Underlying": "BBB", "sleeve": "core_leveraged",
            "long_usd": 3000.0, "short_usd": -1000.0, "gross_target_usd": 4000.0,
        }]), source_date=str(cal[0].date()),
    )
    # Five-percent resize is inside the 12% enter band on both legs.
    p1 = normalize_plan(
        pd.DataFrame([{
            "ETF": "AAA", "Underlying": "BBB", "sleeve": "core_leveraged",
            "long_usd": 3150.0, "short_usd": -1050.0, "gross_target_usd": 4200.0,
        }]), source_date=str(cal[10].date()),
    )
    _, audit, _, _, _ = simulate_book_from_plan_timeline(
        {cal[0]: p0, cal[10]: p1}, panel,
        budgets={"core_leveraged": 5000.0}, capital_usd=1000.0, start=cal[0],
        slippage_bps=0.0, commission_per_share=0.0, margin_rate_annual=0.0,
        enter_band_pct=0.12, exit_band_pct=0.04, min_trade_usd=250.0,
    )
    first = audit.iloc[0]
    changed = audit[audit["plan_date"] == str(cal[10].date())].iloc[0]
    assert float(first["turnover_usd"]) == pytest.approx(4000.0)
    assert float(changed["turnover_usd"]) == pytest.approx(0.0)
