"""Tests for the grow-only inverse-ETF ratchet in compute_bucket4_targets.

The inverse-ETF short leg must never be proposed smaller than what we already hold
(or a persisted floor): we never 'cover' hard-to-relocate inverse inventory. All
delta reduction flows through the bidirectional underlying leg instead.
"""
from __future__ import annotations

import math

import pandas as pd
import pytest

from scripts.bucket4_weekly_opt2 import (
    Bucket4State,
    Bucket4WeeklyConfig,
    compute_bucket4_targets,
)


def _make_state() -> Bucket4State:
    dates = pd.date_range("2026-01-02", periods=10, freq="B")
    etf, und = "TSTZ", "TSTU"
    pair_cache = {(etf, und): {"kw": {"beta_a": -2.0, "beta_b": 1.0, "borrow_a_annual": 0.05}}}
    hedge = {und: pd.Series(0.5, index=dates)}  # constant h = 0.5
    cfg = Bucket4WeeklyConfig(
        screened_csv="unused.csv", start="2024-01-01",
        drift_threshold_long=math.inf, drift_threshold_short=math.inf,
        fee_bps=1.0, slippage_bps=20.0, hedge_base=0.5,
    )
    return Bucket4State(
        pair_cache=pair_cache,
        hedge_by_underlying=hedge,
        hedge_panel=pd.DataFrame(),
        rebalance_dates=pd.DatetimeIndex([dates[0]]),
        hedge_base=0.5,
        screened_subset=pd.DataFrame(),
        pair_metadata=[],
        diagnostics={},
        config=cfg,
        closes_broad=pd.DataFrame(index=dates),
        bucket4_pairs=[(etf, und)],
    )


# With h=0.5, beta_used=2.0: denom=2.0 -> n_inv=50% of gross, n_und=50% of gross.
PAIR = ("TSTZ", "TSTU")
GROSS = 100_000.0


def test_no_ratchet_baseline_unchanged():
    st = _make_state()
    df, _ = compute_bucket4_targets(st, {PAIR: 1.0}, "2026-01-02", GROSS, partial_hedge_ratio=1.0)
    r = df.iloc[0]
    assert r["inverse_etf_short_usd"] == pytest.approx(50_000.0, rel=1e-6)
    assert r["underlying_short_usd"] == pytest.approx(50_000.0, rel=1e-6)
    assert bool(r["ratchet_binding"]) is False
    assert r["ratchet_source"] == "solve"


def test_ratchet_floors_to_held_position_and_resolves_underlying():
    st = _make_state()
    # already hold a larger inverse short than the solve wants -> never cover
    cur = {PAIR: {"inverse_etf_short_usd": 60_000.0, "underlying_short_usd": 55_000.0}}
    df, _ = compute_bucket4_targets(
        st, {PAIR: 1.0}, "2026-01-02", GROSS, partial_hedge_ratio=1.0,
        ratchet_enabled=True, current_leg_notional_by_pair=cur,
    )
    r = df.iloc[0]
    assert r["inverse_short_solved_usd"] == pytest.approx(50_000.0, rel=1e-6)
    assert r["inverse_etf_short_usd"] == pytest.approx(60_000.0, rel=1e-6)  # floored up to held
    # underlying re-solved = h(0.5)*beta(2.0)*inv(60k)*phr(1.0) = 60k
    assert r["underlying_short_usd"] == pytest.approx(60_000.0, rel=1e-6)
    assert bool(r["ratchet_binding"]) is True
    assert r["ratchet_source"] == "held_position"
    assert "floored" in r["ratchet_explain"]


def test_ratchet_floors_to_persisted_state_when_higher():
    st = _make_state()
    cur = {PAIR: {"inverse_etf_short_usd": 55_000.0}}
    df, _ = compute_bucket4_targets(
        st, {PAIR: 1.0}, "2026-01-02", GROSS, partial_hedge_ratio=1.0,
        ratchet_enabled=True, current_leg_notional_by_pair=cur,
        ratchet_floor_by_pair={PAIR: 70_000.0},
    )
    r = df.iloc[0]
    assert r["inverse_etf_short_usd"] == pytest.approx(70_000.0, rel=1e-6)  # state floor wins
    assert r["underlying_short_usd"] == pytest.approx(70_000.0, rel=1e-6)
    assert r["ratchet_source"] == "ratchet_state"


def test_ratchet_does_not_bind_when_solve_exceeds_floor():
    st = _make_state()
    cur = {PAIR: {"inverse_etf_short_usd": 40_000.0}}  # we hold less than solve wants
    df, _ = compute_bucket4_targets(
        st, {PAIR: 1.0}, "2026-01-02", GROSS, partial_hedge_ratio=1.0,
        ratchet_enabled=True, current_leg_notional_by_pair=cur,
    )
    r = df.iloc[0]
    # solve (50k) >= held (40k): grow toward 50k, ratchet not binding
    assert r["inverse_etf_short_usd"] == pytest.approx(50_000.0, rel=1e-6)
    assert bool(r["ratchet_binding"]) is False


def test_ratchet_partial_hedge_ratio_applies_to_underlying_only():
    st = _make_state()
    cur = {PAIR: {"inverse_etf_short_usd": 60_000.0}}
    df, _ = compute_bucket4_targets(
        st, {PAIR: 1.0}, "2026-01-02", GROSS, partial_hedge_ratio=0.75,
        ratchet_enabled=True, current_leg_notional_by_pair=cur,
    )
    r = df.iloc[0]
    assert r["inverse_etf_short_usd"] == pytest.approx(60_000.0, rel=1e-6)
    # underlying = 0.5*2.0*60k*0.75 = 45k
    assert r["underlying_short_usd"] == pytest.approx(45_000.0, rel=1e-6)


def test_continuous_trim_releases_and_reduces_inverse():
    st = _make_state()
    cur = {PAIR: {"inverse_etf_short_usd": 80_000.0}}
    df, _ = compute_bucket4_targets(
        st, {PAIR: 1.0}, "2026-01-02", GROSS, partial_hedge_ratio=1.0,
        ratchet_enabled=True,
        ratchet_trim_enabled=True,
        ratchet_trim_max=0.5,
        ratchet_trim_creep_full=1.0,
        current_leg_notional_by_pair=cur,
        forward_edge_by_pair={PAIR: 0.10},
    )
    r = df.iloc[0]
    assert r["inverse_short_solved_usd"] == pytest.approx(50_000.0, rel=1e-6)
    assert 50_000.0 < r["inverse_etf_short_usd"] < 80_000.0
    assert bool(r["ratchet_released"]) is True
    assert float(r["ratchet_trim_lambda"]) > 0.0
    assert float(r["ratchet_trim_usd"]) > 0.0
    assert r["ratchet_source"] == "edge_trim"
