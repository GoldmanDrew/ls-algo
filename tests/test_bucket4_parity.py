"""Regression tests for ``scripts.bucket4_dynamic_bt`` / ``scripts.bucket4_tail_portfolio``."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from scripts.bucket4_dynamic_bt import run_bucket4_backtest_dynamic_h
from scripts.bucket4_pair_diagnostics import (
    b4_trading_nav_series,
    build_b4_attribution_from_pair_bts,
    diagnose_b4_attribution_vs_nav,
)
from scripts.bucket4_tail_portfolio import aggregate_tail_risk_weighted_portfolio


def test_run_bucket4_backtest_dynamic_h_golden_equity():
    """Fixed synthetic path; golden last equity (tolerance 1e-6 on NAV)."""
    idx = pd.bdate_range("2024-03-01", periods=35, freq="C")
    prices = pd.DataFrame(
        {"a_px": 100.0 * np.exp(0.0003 * np.arange(len(idx))), "b_px": 55.0 * np.exp(-0.0001 * np.arange(len(idx)))},
        index=idx,
    )
    h = pd.Series(0.72, index=idx)
    rebal = pd.DatetimeIndex([idx[0], idx[12], idx[24]])
    bt = run_bucket4_backtest_dynamic_h(
        prices,
        h,
        rebal,
        initial_capital=100_000.0,
        gross_multiplier=1.0,
        beta_a=-2.0,
        beta_b=1.0,
        borrow_a_annual=0.12,
        borrow_b_annual=0.04,
        short_proceeds_annual=0.0,
        fee_bps=2.0,
        slippage_bps=3.0,
        opt2_h_base=0.75,
    )
    expected_last = 98_753.35873098962
    np.testing.assert_allclose(float(bt["equity"].iloc[-1]), expected_last, rtol=0.0, atol=1e-5)


def test_aggregate_tail_risk_weighted_portfolio_two_pairs_golden():
    """Two synthetic pairs, fixed weights; golden portfolio last equity."""
    idx = pd.bdate_range("2024-01-02", periods=45, freq="C")
    pr1 = pd.DataFrame(
        {"a_px": 100.0 + np.arange(len(idx)) * 0.02, "b_px": 50.0 + np.arange(len(idx)) * 0.01},
        index=idx,
    )
    pr2 = pd.DataFrame(
        {"a_px": 20.0 + np.arange(len(idx)) * 0.01, "b_px": 200.0 - np.arange(len(idx)) * 0.02},
        index=idx,
    )
    h1 = pd.Series(0.8, index=idx)
    h2 = pd.Series(0.75, index=idx)
    base_kw = dict(
        initial_capital=50_000.0,
        gross_multiplier=1.0,
        beta_a=-2.0,
        beta_b=1.0,
        borrow_a_annual=0.05,
        borrow_b_annual=0.02,
        short_proceeds_annual=0.0,
        fee_bps=1.0,
        slippage_bps=2.0,
    )
    cache = {
        ("E1", "U1"): {"prices": pr1, "kw": dict(base_kw)},
        ("E2", "U2"): {"prices": pr2, "kw": dict(base_kw)},
    }
    hmap = {"U1": h1, "U2": h2}
    rebal = pd.DatetimeIndex([idx[0], idx[10], idx[20], idx[35]])
    w = {("E1", "U1"): 0.4, ("E2", "U2"): 0.6}
    bt_pf, by = aggregate_tail_risk_weighted_portfolio(
        cache,
        hmap,
        rebal,
        w,
        start_sim=idx[0],
        pf_initial=100_000.0,
        opt2_h_base=0.75,
    )
    assert len(by) == 2
    expected_last = 98_703.51942222994
    np.testing.assert_allclose(float(bt_pf["equity"].iloc[-1]), expected_last, rtol=0.0, atol=1e-5)


def _make_drift_pair():
    idx = pd.bdate_range("2024-01-02", periods=80, freq="C")
    a = 100.0 + np.cumsum(np.linspace(0.05, 0.05, len(idx)))
    b = 60.0 - np.cumsum(np.linspace(0.04, 0.04, len(idx)))
    prices = pd.DataFrame({"a_px": a, "b_px": b}, index=idx)
    h = pd.Series(0.75, index=idx)
    weekly = pd.DatetimeIndex(pd.Series(1.0, index=idx).resample("W-FRI", label="right", closed="right").last().dropna().index)
    rebal = pd.DatetimeIndex([t for t in weekly if t in idx])
    if rebal[0] != idx[0]:
        rebal = pd.DatetimeIndex([idx[0], *rebal])
    return prices, h, rebal


def test_drift_threshold_none_equals_legacy_path():
    """Default ``drift_threshold_share_of_gross=None`` keeps legacy behaviour identical."""
    prices, h, rebal = _make_drift_pair()
    base_kw = dict(initial_capital=100_000.0, beta_a=-2.0, beta_b=1.0, opt2_h_base=0.75, fee_bps=1.0, slippage_bps=2.0)
    bt_legacy = run_bucket4_backtest_dynamic_h(prices, h, rebal, **base_kw)
    bt_thr = run_bucket4_backtest_dynamic_h(prices, h, rebal, drift_threshold_share_of_gross=None, **base_kw)
    np.testing.assert_allclose(bt_legacy["equity"].to_numpy(), bt_thr["equity"].to_numpy(), atol=1e-10)
    assert bt_thr["rebalance"].sum() == bt_thr["rebalance_scheduled"].sum()
    assert bt_thr["rebalance_skipped_below_drift"].sum() == 0


def test_drift_threshold_huge_disables_after_first():
    """A 100% threshold means nothing ever triggers after the initial entry."""
    prices, h, rebal = _make_drift_pair()
    bt = run_bucket4_backtest_dynamic_h(
        prices,
        h,
        rebal,
        initial_capital=100_000.0,
        beta_a=-2.0,
        beta_b=1.0,
        opt2_h_base=0.75,
        drift_threshold_share_of_gross=1.0,
    )
    assert bt["rebalance"].sum() == 1
    assert bt["rebalance_skipped_below_drift"].sum() == int(bt["rebalance_scheduled"].sum() - 1)


def test_drift_threshold_5pct_skips_some_and_traces_drift_share():
    """5% threshold leaves at least one initial rebal; drift_share_of_gross is populated on schedule days."""
    prices, h, rebal = _make_drift_pair()
    bt = run_bucket4_backtest_dynamic_h(
        prices,
        h,
        rebal,
        initial_capital=100_000.0,
        beta_a=-2.0,
        beta_b=1.0,
        opt2_h_base=0.75,
        drift_threshold_share_of_gross=0.05,
    )
    sched_after_first = bt.loc[bt["rebalance_scheduled"] & (bt.index > bt.index[0])]
    assert (sched_after_first["drift_share_of_gross"].notna() & np.isfinite(sched_after_first["drift_share_of_gross"])).all()
    assert int(bt["rebalance"].sum()) >= 1
    assert int(bt["rebalance_skipped_below_drift"].sum()) >= 0


def test_drift_threshold_propagates_through_aggregator():
    """``aggregate_tail_risk_weighted_portfolio`` forwards drift kwarg to per-pair engine."""
    idx = pd.bdate_range("2024-01-02", periods=60, freq="C")
    pr1 = pd.DataFrame({"a_px": 100.0 + 0.1 * np.arange(len(idx)), "b_px": 60.0 - 0.05 * np.arange(len(idx))}, index=idx)
    pr2 = pd.DataFrame({"a_px": 25.0 + 0.05 * np.arange(len(idx)), "b_px": 200.0 - 0.1 * np.arange(len(idx))}, index=idx)
    cache = {
        ("E1", "U1"): {"prices": pr1, "kw": dict(initial_capital=50_000.0, beta_a=-2.0, beta_b=1.0)},
        ("E2", "U2"): {"prices": pr2, "kw": dict(initial_capital=50_000.0, beta_a=-2.0, beta_b=1.0)},
    }
    hmap = {"U1": pd.Series(0.75, index=idx), "U2": pd.Series(0.75, index=idx)}
    weekly = pd.DatetimeIndex(pd.Series(1.0, index=idx).resample("W-FRI", label="right", closed="right").last().dropna().index)
    weights = {("E1", "U1"): 0.5, ("E2", "U2"): 0.5}
    _, by_legacy = aggregate_tail_risk_weighted_portfolio(
        cache, hmap, weekly, weights, start_sim=idx[0], pf_initial=100_000.0, opt2_h_base=0.75
    )
    _, by_thr = aggregate_tail_risk_weighted_portfolio(
        cache, hmap, weekly, weights, start_sim=idx[0], pf_initial=100_000.0, opt2_h_base=0.75, drift_threshold_share_of_gross=1.0
    )
    for k in by_legacy:
        legacy = by_legacy[k]
        thr = by_thr[k]
        assert legacy["rebalance"].sum() >= thr["rebalance"].sum()
        assert int(thr["rebalance"].sum()) == 1


def test_b4_trading_nav_series_strips_late_pair_inception():
    idx = pd.bdate_range("2024-01-02", periods=6, freq="C")
    sub = pd.DataFrame(
        {
            "equity": [10_000.0, 10_000.0],
            "a_shares": [-1.0, -1.0],
            "b_shares": [-1.0, -1.0],
            "a_px": [100.0, 100.0],
            "b_px": [50.0, 50.0],
        },
        index=idx[2:4],
    )
    nav = pd.Series([0.0, 0.0, 10_000.0, 10_000.0, 10_000.0, 10_000.0], index=idx)
    adj = b4_trading_nav_series(nav, {("E", "U"): sub})
    assert abs(float(adj.loc[idx[2]])) < 1e-6
    assert abs(float(adj.loc[idx[3]]) - float(adj.loc[idx[2]])) < 1e-6


def test_attribution_does_not_count_pair_inception_as_trading_pnl():
    """Late-starting pair: 0→book step on union calendar must not appear as attributed PnL."""
    idx = pd.bdate_range("2024-01-02", periods=8, freq="C")
    fp = idx[3]
    sub = pd.DataFrame(
        {
            "equity": [50_000.0, 50_000.0, 50_000.0, 50_000.0],
            "a_shares": [-100.0] * 4,
            "b_shares": [-50.0] * 4,
            "a_px": [100.0] * 4,
            "b_px": [100.0] * 4,
        },
        index=idx[3:7],
    )
    attr = build_b4_attribution_from_pair_bts({("E", "U"): sub}, index=idx)
    pnl = attr["cum_pnl_by_etf"]
    daily = pnl.diff()
    if len(pnl) > 0:
        daily.iloc[0] = pnl.iloc[0]
    flow = daily.fillna(0.0).sum(axis=1)
    assert abs(float(flow.loc[fp])) < 1e-3


def test_diagnose_b4_attribution_vs_nav_two_pairs():
    """Ticker-level daily sums match nav.diff() minus pair inception flows (trading-only)."""
    idx = pd.bdate_range("2024-01-02", periods=45, freq="C")
    pr1 = pd.DataFrame(
        {"a_px": 100.0 + np.arange(len(idx)) * 0.02, "b_px": 50.0 + np.arange(len(idx)) * 0.01},
        index=idx,
    )
    pr2 = pd.DataFrame(
        {"a_px": 20.0 + np.arange(len(idx)) * 0.01, "b_px": 200.0 - np.arange(len(idx)) * 0.02},
        index=idx,
    )
    h1 = pd.Series(0.8, index=idx)
    h2 = pd.Series(0.75, index=idx)
    base_kw = dict(
        initial_capital=50_000.0,
        gross_multiplier=1.0,
        beta_a=-2.0,
        beta_b=1.0,
        borrow_a_annual=0.05,
        borrow_b_annual=0.02,
        short_proceeds_annual=0.0,
        fee_bps=1.0,
        slippage_bps=2.0,
    )
    cache = {
        ("E1", "U1"): {"prices": pr1, "kw": dict(base_kw)},
        ("E2", "U2"): {"prices": pr2, "kw": dict(base_kw)},
    }
    hmap = {"U1": h1, "U2": h2}
    rebal = pd.DatetimeIndex([idx[0], idx[10], idx[20], idx[35]])
    w = {("E1", "U1"): 0.4, ("E2", "U2"): 0.6}
    bt_pf, by = aggregate_tail_risk_weighted_portfolio(
        cache,
        hmap,
        rebal,
        w,
        start_sim=idx[0],
        pf_initial=100_000.0,
        opt2_h_base=0.75,
    )
    diag = diagnose_b4_attribution_vs_nav(bt_pf, by, tol_usd=1.0)
    assert diag.get("ok") is True, diag


@pytest.mark.skipif(os.environ.get("RUN_B4_BOOTSTRAP_ALIAS_TEST") != "1", reason="set RUN_B4_BOOTSTRAP_ALIAS_TEST=1 to exec Bucket_4 notebook (slow)")
def test_bootstrap_aliases_engine_function():
    """``v6_bucket4_bootstrap_from_nb`` overwrites notebook exec with shared module."""
    from pathlib import Path

    from scripts import v6_bucket4_bootstrap_from_nb as boot

    g: dict = {}
    screened = Path(__file__).resolve().parent.parent / "data" / "etf_screened_today.csv"
    if not screened.is_file():
        pytest.skip("screener CSV not present")
    boot.ensure_v6_bucket4_globals_from_notebook(g, repo_root=screened.parent.parent, screened_csv=screened)
    fn = g.get("run_bucket4_backtest_dynamic_h")
    assert callable(fn)
    import scripts.bucket4_dynamic_bt as mod

    assert fn is mod.run_bucket4_backtest_dynamic_h
