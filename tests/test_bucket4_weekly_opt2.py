"""Tests for ``scripts.bucket4_weekly_opt2``."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.bucket4_dynamic_bt import run_bucket4_backtest_dynamic_h
from scripts.bucket4_weekly_opt2 import (
    Bucket4WeeklyConfig,
    load_bucket4_pairs_from_screened,
    run_bucket4_backtest,
    run_bucket4_pair_backtest_threshold,
    weekly_rebalance_dates,
)
from scripts.v6_b4_pf_weights import V6PfParams


def _synth_prices(n: int = 40) -> pd.DataFrame:
    idx = pd.bdate_range("2024-01-02", periods=n, freq="C")
    return pd.DataFrame(
        {
            "a_px": 100.0 * np.exp(0.0002 * np.arange(n)),
            "b_px": 50.0 * np.exp(-0.0001 * np.arange(n)),
        },
        index=idx,
    )


def test_weekly_rebalance_schedule_w_fri():
    idx = pd.bdate_range("2024-01-02", periods=30, freq="C")
    w = weekly_rebalance_dates(idx, "W-FRI", warmup_bdays=0)
    assert len(w) >= 4
    assert list(w) == sorted(w)
    assert max(w) <= idx.max()


def test_weekly_default_matches_resample():
    idx = pd.bdate_range("2024-03-01", periods=25, freq="C")
    w = weekly_rebalance_dates(idx, "W-FRI", warmup_bdays=0)
    s = pd.Series(1.0, index=idx)
    ref = s.resample("W-FRI", label="right", closed="right").last().index
    assert len(w) == len(ref)


def test_threshold_rebalance_fires_and_reason():
    prices = _synth_prices(30)
    # Force a large one-day move so leg notionals drift far enough to trigger
    # the daily threshold path between scheduled weekly rebalances.
    prices.iloc[2:, prices.columns.get_loc("a_px")] *= 1.20
    prices.iloc[2:, prices.columns.get_loc("b_px")] *= 0.80
    idx = prices.index
    h = pd.Series(0.75, index=idx)
    sched = pd.DatetimeIndex([idx[0], idx[-1]])
    bt, log = run_bucket4_pair_backtest_threshold(
        prices,
        h,
        sched,
        drift_threshold_long=0.02,
        drift_threshold_short=0.02,
        initial_capital=100_000.0,
        gross_multiplier=1.0,
        beta_a=-2.0,
        beta_b=1.0,
        borrow_a_annual=0.0,
        borrow_b_annual=0.0,
        fee_bps=0.0,
        slippage_bps=0.0,
    )
    thr = log.loc[log["is_threshold"] == True]
    assert len(thr) > 0
    assert set(thr["reason"].unique()).issubset({"long_drift", "short_drift", "both_drift"})


def test_no_threshold_when_below_threshold():
    prices = _synth_prices(15)
    idx = prices.index
    h = pd.Series(0.75, index=idx)
    sched = weekly_rebalance_dates(idx, "W-FRI", warmup_bdays=0)
    bt, log = run_bucket4_pair_backtest_threshold(
        prices,
        h,
        sched,
        drift_threshold_long=10.0,
        drift_threshold_short=10.0,
        initial_capital=100_000.0,
        gross_multiplier=1.0,
        beta_a=-2.0,
        beta_b=1.0,
        fee_bps=0.0,
        slippage_bps=0.0,
    )
    assert log["is_threshold"].sum() == 0


def test_parity_with_dynamic_h_when_no_threshold():
    prices = _synth_prices(35)
    idx = prices.index
    h = pd.Series(0.72, index=idx)
    sched = pd.DatetimeIndex([idx[0], idx[12], idx[24]])
    bt_old = run_bucket4_backtest_dynamic_h(
        prices,
        h,
        sched,
        initial_capital=100_000.0,
        gross_multiplier=1.0,
        beta_a=-2.0,
        beta_b=1.0,
        borrow_a_annual=0.12,
        borrow_b_annual=0.04,
        fee_bps=2.0,
        slippage_bps=3.0,
        opt2_h_base=0.75,
    )
    bt_new, _ = run_bucket4_pair_backtest_threshold(
        prices,
        h,
        sched,
        drift_threshold_long=float("inf"),
        drift_threshold_short=float("inf"),
        initial_capital=100_000.0,
        gross_multiplier=1.0,
        beta_a=-2.0,
        beta_b=1.0,
        borrow_a_annual=0.12,
        borrow_b_annual=0.04,
        fee_bps=2.0,
        slippage_bps=3.0,
        opt2_h_base=0.75,
    )
    np.testing.assert_allclose(bt_new["equity"].values, bt_old["equity"].values, rtol=0, atol=1e-6)


def test_sco_excluded_from_pairs(tmp_path):
    csv = tmp_path / "s.csv"
    rows = [
        "ETF,Underlying,Beta,bucket,vol_underlying_annual,bucket4_net_edge_annual",
        "SCO,USO,-1.5,bucket_4,0.8,0.3",
        "XYZ,QQQ,-1.2,bucket_4,0.8,0.3",
    ]
    csv.write_text("\n".join(rows), encoding="utf-8")
    pairs, _ = load_bucket4_pairs_from_screened(str(csv), min_underlying_vol=0.5, min_net_decay=0.1)
    etfs = {p[0] for p in pairs}
    assert "SCO" not in etfs
    assert "XYZ" in etfs


def test_targets_short_legs_and_weights(tmp_path):
    csv = tmp_path / "s.csv"
    csv.write_text(
        "\n".join(
            [
                "ETF,Underlying,Beta,bucket,vol_underlying_annual,bucket4_net_edge_annual,borrow_current",
                "E1,U1,-2.0,bucket_4,0.8,0.3,0.05",
                "E2,U2,-1.5,bucket_4,0.8,0.3,0.04",
            ]
        ),
        encoding="utf-8",
    )
    idx = pd.bdate_range("2024-01-02", periods=80, freq="C")
    px1 = pd.DataFrame({"a_px": 100.0 + np.arange(80) * 0.01, "b_px": 50.0 + np.arange(80) * 0.01}, index=idx)
    px2 = pd.DataFrame({"a_px": 20.0 + np.arange(80) * 0.01, "b_px": 200.0 - np.arange(80) * 0.01}, index=idx)
    cache = {
        ("E1", "U1"): {
            "prices": px1,
            "kw": dict(
                initial_capital=100_000.0,
                gross_multiplier=1.0,
                beta_a=-2.0,
                beta_b=1.0,
                borrow_a_annual=0.05,
                borrow_b_annual=0.0,
                short_proceeds_annual=0.0,
                fee_bps=1.0,
                slippage_bps=2.0,
            ),
        },
        ("E2", "U2"): {
            "prices": px2,
            "kw": dict(
                initial_capital=100_000.0,
                gross_multiplier=1.0,
                beta_a=-1.5,
                beta_b=1.0,
                borrow_a_annual=0.04,
                borrow_b_annual=0.0,
                short_proceeds_annual=0.0,
                fee_bps=1.0,
                slippage_bps=2.0,
            ),
        },
    }
    und_cols = {"U1": px1["b_px"].rename("U1"), "U2": px2["b_px"].rename("U2"), "E1": px1["a_px"].rename("E1"), "E2": px2["a_px"].rename("E2")}
    closes = pd.DataFrame(und_cols).sort_index()
    from scripts.bucket4_weekly_opt2 import (
        Bucket4State,
        build_hedge_panel_opt2,
        compute_bucket4_targets,
        panel_to_hedge_by_underlying,
    )

    panel, rebal, _ = build_hedge_panel_opt2(
        closes,
        [("E1", "U1"), ("E2", "U2")],
        weekly_rebalance_freq="W-FRI",
        warmup_bdays=5,
        overlay={"regime_overlay_enable": False},
    )
    hmap = panel_to_hedge_by_underlying(panel, closes.index.sort_values(), ["U1", "U2"], 0.75)
    cfg = Bucket4WeeklyConfig(screened_csv=str(csv), start="2024-01-01", end=None, pf_params=V6PfParams(min_pairs=1))
    st = Bucket4State(
        pair_cache=cache,
        hedge_by_underlying=hmap,
        hedge_panel=panel,
        rebalance_dates=rebal,
        hedge_base=0.75,
        screened_subset=pd.DataFrame(),
        pair_metadata=[],
        diagnostics={},
        config=cfg,
        closes_broad=closes,
        bucket4_pairs=[("E1", "U1"), ("E2", "U2")],
    )
    w = {("E1", "U1"): 0.5, ("E2", "U2"): 0.5}
    tgt, meta = compute_bucket4_targets(st, w, "2024-04-01", 100_000.0, partial_hedge_ratio=1.0, beta_floor=0.1)
    assert len(tgt) == 2
    assert float(tgt["pair_weight"].sum()) == pytest.approx(1.0)
    assert all(tgt["inverse_etf_short_usd"] > 0)
    assert all(tgt["underlying_short_usd"] > 0)
    assert float(tgt["gross_target_usd"].sum()) == pytest.approx(100_000.0, rel=1e-5)


def test_targets_threshold_diagnostics_when_current_legs_supplied(tmp_path):
    csv = tmp_path / "s.csv"
    csv.write_text(
        "\n".join(
            [
                "ETF,Underlying,Beta,bucket,vol_underlying_annual,bucket4_net_edge_annual,borrow_current",
                "E1,U1,-2.0,bucket_4,0.8,0.3,0.05",
            ]
        ),
        encoding="utf-8",
    )
    idx = pd.bdate_range("2024-01-02", periods=30, freq="C")
    px = pd.DataFrame({"a_px": 100.0 + np.arange(30) * 0.01, "b_px": 50.0 + np.arange(30) * 0.01}, index=idx)
    cfg = Bucket4WeeklyConfig(
        screened_csv=str(csv),
        start="2024-01-01",
        drift_threshold_long=0.05,
        drift_threshold_short=0.05,
        pf_params=V6PfParams(min_pairs=1),
    )
    from scripts.bucket4_weekly_opt2 import Bucket4State, compute_bucket4_targets

    st = Bucket4State(
        pair_cache={
            ("E1", "U1"): {
                "prices": px,
                "kw": dict(
                    initial_capital=100_000.0,
                    gross_multiplier=1.0,
                    beta_a=-2.0,
                    beta_b=1.0,
                    borrow_a_annual=0.05,
                    borrow_b_annual=0.0,
                    fee_bps=1.0,
                    slippage_bps=2.0,
                ),
            }
        },
        hedge_by_underlying={"U1": pd.Series(0.75, index=idx)},
        hedge_panel=pd.DataFrame(),
        rebalance_dates=pd.DatetimeIndex([idx[0]]),
        hedge_base=0.75,
        screened_subset=pd.DataFrame(),
        pair_metadata=[],
        diagnostics={},
        config=cfg,
        closes_broad=pd.DataFrame({"U1": px["b_px"], "E1": px["a_px"]}),
        bucket4_pairs=[("E1", "U1")],
    )
    tgt, meta = compute_bucket4_targets(
        st,
        {("E1", "U1"): 1.0},
        idx[10],
        100_000.0,
        current_leg_notional_by_pair={
            ("E1", "U1"): {"inverse_etf_short_usd": 1.0, "underlying_short_usd": 1.0}
        },
    )
    assert bool(meta["thresholds_evaluated"])
    assert bool(tgt.loc[0, "is_threshold_rebalance"])
    assert tgt.loc[0, "rebalance_reason"] in {"long_drift", "short_drift", "both_drift"}


def test_uvix_included_from_screened_when_eligible(tmp_path):
    csv = tmp_path / "s.csv"
    csv.write_text(
        "\n".join(
            [
                "ETF,Underlying,Beta,bucket,vol_underlying_annual,bucket4_net_edge_annual,borrow_current,inverse_shortable",
                "UVIX,SVIX,-1.0,bucket_4,0.8,0.3,0.15,True",
                "SCO,USO,-1.5,bucket_4,0.8,0.3,0.10,True",
                "BAD,QQQ,-1.5,bucket_4,0.8,0.3,0.10,False",
            ]
        ),
        encoding="utf-8",
    )
    pairs, _ = load_bucket4_pairs_from_screened(str(csv), min_underlying_vol=0.5, min_net_decay=0.1)
    assert ("UVIX", "SVIX") in pairs
    assert ("SCO", "USO") not in pairs
    assert ("BAD", "QQQ") not in pairs


def test_portfolio_nav_equals_sum_pairs():
    idx = pd.bdate_range("2024-01-02", periods=50, freq="C")
    px1 = pd.DataFrame({"a_px": 100.0 + np.arange(50) * 0.02, "b_px": 50.0 + np.arange(50) * 0.01}, index=idx)
    px2 = pd.DataFrame({"a_px": 20.0 + np.arange(50) * 0.01, "b_px": 200.0 - np.arange(50) * 0.02}, index=idx)
    cache = {
        ("E1", "U1"): {
            "prices": px1,
            "kw": dict(
                initial_capital=50_000.0,
                gross_multiplier=1.0,
                beta_a=-2.0,
                beta_b=1.0,
                borrow_a_annual=0.05,
                borrow_b_annual=0.02,
                short_proceeds_annual=0.0,
                fee_bps=1.0,
                slippage_bps=2.0,
            ),
        },
        ("E2", "U2"): {
            "prices": px2,
            "kw": dict(
                initial_capital=50_000.0,
                gross_multiplier=1.0,
                beta_a=-2.0,
                beta_b=1.0,
                borrow_a_annual=0.05,
                borrow_b_annual=0.02,
                short_proceeds_annual=0.0,
                fee_bps=1.0,
                slippage_bps=2.0,
            ),
        },
    }
    closes = pd.DataFrame({"U1": px1["b_px"], "U2": px2["b_px"], "E1": px1["a_px"], "E2": px2["a_px"]}).sort_index()
    from scripts.bucket4_weekly_opt2 import Bucket4State, build_hedge_panel_opt2, panel_to_hedge_by_underlying

    panel, rebal, _ = build_hedge_panel_opt2(
        closes,
        [("E1", "U1"), ("E2", "U2")],
        weekly_rebalance_freq="W-FRI",
        warmup_bdays=3,
        overlay={"regime_overlay_enable": False},
    )
    hmap = panel_to_hedge_by_underlying(panel, closes.index.sort_values(), ["U1", "U2"], 0.75)
    cfg = Bucket4WeeklyConfig(screened_csv="dummy.csv", start="2024-01-01", fee_bps=1.0, slippage_bps=2.0, drift_threshold_long=float("inf"), drift_threshold_short=float("inf"))
    st = Bucket4State(
        pair_cache=cache,
        hedge_by_underlying=hmap,
        hedge_panel=panel,
        rebalance_dates=rebal,
        hedge_base=0.75,
        screened_subset=pd.DataFrame(),
        pair_metadata=[],
        diagnostics={},
        config=cfg,
        closes_broad=closes,
        bucket4_pairs=[("E1", "U1"), ("E2", "U2")],
    )
    out = run_bucket4_backtest(st, {("E1", "U1"): 0.4, ("E2", "U2"): 0.6}, initial_capital=100_000.0, use_thresholds=False)
    nav = out["portfolio_curve"]["nav"]
    s_last = sum(float(bt["equity"].iloc[-1]) for bt in out["pair_backtests"].values())
    assert abs(float(nav.iloc[-1]) - s_last) < 1.0


def test_fee_slippage_affect_friction():
    idx = pd.bdate_range("2024-01-02", periods=20, freq="C")
    prices = pd.DataFrame({"a_px": np.linspace(100, 101, 20), "b_px": np.linspace(50, 49, 20)}, index=idx)
    h = pd.Series(0.75, index=idx)
    sched = pd.DatetimeIndex([idx[0], idx[10]])
    bt0, _ = run_bucket4_pair_backtest_threshold(
        prices,
        h,
        sched,
        drift_threshold_long=float("inf"),
        drift_threshold_short=float("inf"),
        fee_bps=0.0,
        slippage_bps=0.0,
        initial_capital=100_000.0,
        beta_a=-2.0,
        beta_b=1.0,
    )
    bt1, _ = run_bucket4_pair_backtest_threshold(
        prices,
        h,
        sched,
        drift_threshold_long=float("inf"),
        drift_threshold_short=float("inf"),
        fee_bps=10.0,
        slippage_bps=10.0,
        initial_capital=100_000.0,
        beta_a=-2.0,
        beta_b=1.0,
    )
    assert float(bt1["equity"].iloc[-1]) < float(bt0["equity"].iloc[-1])
    assert float(bt1["rebalance_fee"].sum()) > float(bt0["rebalance_fee"].sum())
