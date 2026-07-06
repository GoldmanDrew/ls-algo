"""Bucket-4 harvest logic: candidate lanes, live hedge-ratio resolver, underlying netting."""
from __future__ import annotations

import pandas as pd
import pytest

from harvest_underexposed_shorts import (
    build_harvest_candidates,
    compute_underlying_delta_usd,
    load_plan_pair_attrs,
    net_underlying_orders,
    resolve_b4_hedge_ratio,
)


# --------------------------------------------------------------------------
# Candidate lanes: sign of delta drives lane + underlying direction
# --------------------------------------------------------------------------

def _disc(symbols_gap: dict[str, float]) -> pd.DataFrame:
    rows = []
    for sym, gap in symbols_gap.items():
        rows.append(
            {
                "symbol": sym,
                "abs_discrepancy_usd": abs(gap),
                "gross_gap_usd": gap,  # negative == under-exposed
                "under_exposed": gap < 0,
            }
        )
    return pd.DataFrame(rows)


def test_build_candidates_splits_lanes_by_delta_sign():
    disc = _disc({"COYY": -50_000.0, "QBTZ": -8_000.0})
    cands = build_harvest_candidates(
        disc,
        etf_to_under={"COYY": "COIN", "QBTZ": "QBTS"},
        etf_to_delta={"COYY": 0.37, "QBTZ": -2.0},
        etf_to_sleeve={"COYY": "yieldboost", "QBTZ": "inverse_decay_bucket4"},
        plan_ratio_map={"COYY": 0.37, "QBTZ": 2.0},
        blocked_symbols=set(),
        top_n=0,
    )
    by = cands.set_index("symbol")
    assert by.loc["COYY", "lane"] == "long_hedge"
    assert by.loc["COYY", "under_hedge_dir"] == "BUY"
    assert by.loc["QBTZ", "lane"] == "b4_pair"
    assert by.loc["QBTZ", "under_hedge_dir"] == "SELL"


def test_build_candidates_drops_zero_delta_and_blocked_and_only_underexposed():
    disc = _disc({"COYY": -50_000.0, "ZERO": -10_000.0, "OVER": +9_000.0, "BLK": -1_000.0})
    cands = build_harvest_candidates(
        disc,
        etf_to_under={"COYY": "COIN", "ZERO": "ZZZ", "OVER": "OVR", "BLK": "BKU"},
        etf_to_delta={"COYY": 0.37, "ZERO": 0.0, "OVER": 2.0, "BLK": -2.0},
        etf_to_sleeve={},
        plan_ratio_map={},
        blocked_symbols={"BLK"},
        top_n=0,
    )
    syms = set(cands["symbol"])
    assert syms == {"COYY"}  # ZERO (delta 0), OVER (not under-exposed), BLK (blocked) all gone


# --------------------------------------------------------------------------
# Underlying delta: BUY for long_hedge (+|delta|), SELL for b4 (-r_live)
# --------------------------------------------------------------------------

def test_underlying_delta_long_hedge_is_buy_delta_neutral():
    d = compute_underlying_delta_usd(
        lane="long_hedge", filled_inverse_notional=10_000.0, delta=0.5, r_live=0.0, buffer_pct=0.0
    )
    assert d == pytest.approx(5_000.0)  # +|delta| x notional, BUY


def test_underlying_delta_b4_is_sell_at_r_live():
    d = compute_underlying_delta_usd(
        lane="b4_pair", filled_inverse_notional=10_000.0, delta=-2.0, r_live=2.0, buffer_pct=0.0
    )
    assert d == pytest.approx(-20_000.0)  # -r_live x notional, SELL


def test_underlying_delta_applies_buffer_both_lanes():
    long_d = compute_underlying_delta_usd(
        lane="long_hedge", filled_inverse_notional=10_000.0, delta=1.0, r_live=0.0, buffer_pct=0.01
    )
    b4_d = compute_underlying_delta_usd(
        lane="b4_pair", filled_inverse_notional=10_000.0, delta=-2.0, r_live=1.5, buffer_pct=0.01
    )
    assert long_d == pytest.approx(9_900.0)
    assert b4_d == pytest.approx(-14_850.0)


# --------------------------------------------------------------------------
# Live hedge-ratio resolver
# --------------------------------------------------------------------------

def test_resolve_hedge_ratio_prefers_b4_detail_even_on_shared_names():
    # COIN net long at book level, but B4 detail isolates the structural short (12,316) vs
    # a raw inverse short of 6,158 -> ratio ~2.0.
    r, src = resolve_b4_hedge_ratio(
        underlying="COIN",
        etf="CONI",
        inverse_raw_notional=6_158.0,
        beta=2.0,
        b4_detail_etf_ratio={"CONI": 2.0},
        b4_detail_map={"COIN": 12_316.0},
        live_net_under_notional=+80_000.0,  # net long, unusable directly
        plan_ratio=2.0,
        mode="live",
    )
    assert src == "b4_detail_pair"
    assert r == pytest.approx(2.0, abs=0.01)


def test_resolve_hedge_ratio_b4_detail_underlying_fallback():
    r, src = resolve_b4_hedge_ratio(
        underlying="COIN",
        etf="CONI",
        inverse_raw_notional=6_158.0,
        beta=2.0,
        b4_detail_etf_ratio={},
        b4_detail_map={"COIN": 12_316.0},
        live_net_under_notional=+80_000.0,
        plan_ratio=2.0,
        mode="live",
    )
    assert src == "b4_detail"
    assert r == pytest.approx(2.0, abs=0.01)


def test_resolve_hedge_ratio_uses_live_net_when_underlying_net_short():
    r, src = resolve_b4_hedge_ratio(
        underlying="QBTS",
        inverse_raw_notional=10_000.0,
        beta=2.0,
        b4_detail_map={},
        live_net_under_notional=-15_000.0,  # net short -> true live ratio 1.5
        plan_ratio=2.0,
        mode="live",
    )
    assert src == "live_net"
    assert r == pytest.approx(1.5, abs=0.01)


def test_resolve_hedge_ratio_falls_back_to_plan_ratio():
    r, src = resolve_b4_hedge_ratio(
        underlying="NEW",
        inverse_raw_notional=10_000.0,
        beta=2.0,
        b4_detail_map={},
        live_net_under_notional=+5_000.0,  # net long, no B4 detail -> plan ratio
        plan_ratio=1.8,
        mode="live",
    )
    assert src == "plan_ratio"
    assert r == pytest.approx(1.8, abs=0.01)


def test_resolve_hedge_ratio_zero_inverse_uses_plan_ratio():
    r, src = resolve_b4_hedge_ratio(
        underlying="NEW",
        inverse_raw_notional=0.0,
        beta=2.0,
        b4_detail_map={"NEW": 5_000.0},
        live_net_under_notional=-5_000.0,
        plan_ratio=1.7,
        mode="live",
    )
    assert src == "plan_ratio_no_inverse"
    assert r == pytest.approx(1.7, abs=0.01)


def test_resolve_hedge_ratio_clips_to_beta_multiple():
    # Detail short is huge relative to a tiny inverse -> ratio would be 100; clip to 2.0 * 1.5.
    r, src = resolve_b4_hedge_ratio(
        underlying="X",
        inverse_raw_notional=100.0,
        beta=2.0,
        b4_detail_map={"X": 10_000.0},
        live_net_under_notional=0.0,
        plan_ratio=None,
        mode="live",
    )
    assert src == "b4_detail"
    assert r == pytest.approx(3.0)  # 2.0 * 1.5 clip


def test_resolve_hedge_ratio_mode_delta_neutral_forces_beta():
    r, src = resolve_b4_hedge_ratio(
        underlying="QBTS",
        inverse_raw_notional=10_000.0,
        beta=2.0,
        b4_detail_map={"QBTS": 30_000.0},
        live_net_under_notional=-30_000.0,
        plan_ratio=0.5,
        mode="delta_neutral",
    )
    assert src == "delta_neutral"
    assert r == pytest.approx(2.0)


def test_resolve_hedge_ratio_mode_plan_ratio_ignores_live():
    r, src = resolve_b4_hedge_ratio(
        underlying="QBTS",
        inverse_raw_notional=10_000.0,
        beta=2.0,
        b4_detail_map={"QBTS": 30_000.0},
        live_net_under_notional=-30_000.0,
        plan_ratio=1.2,
        mode="plan_ratio",
    )
    assert src == "plan_ratio"
    assert r == pytest.approx(1.2)


# --------------------------------------------------------------------------
# Underlying netting: B1/B2 buy vs B4 sell cancel on shared underlying
# --------------------------------------------------------------------------

def test_netting_cancels_buy_and_sell_on_shared_underlying():
    rows = [
        {"underlying": "MSTR", "etf": "MTYY", "under_delta_usd": +10_000.0},  # yieldboost BUY
        {"underlying": "MSTR", "etf": "MSTZ", "under_delta_usd": -4_000.0},   # B4 SELL
    ]
    orders = net_underlying_orders(rows, {"MSTR": 100.0}, min_trade_usd=200.0)
    assert len(orders) == 1
    o = orders[0]
    assert o["action"] == "BUY"
    assert o["net_usd"] == pytest.approx(6_000.0)
    assert o["qty"] == 60
    assert o["gross_buy_usd"] == pytest.approx(10_000.0)
    assert o["gross_sell_usd"] == pytest.approx(4_000.0)
    assert set(o["etfs"].split(",")) == {"MTYY", "MSTZ"}


def test_netting_near_total_cancellation_skips():
    rows = [
        {"underlying": "COIN", "etf": "COYY", "under_delta_usd": +5_000.0},
        {"underlying": "COIN", "etf": "CONI", "under_delta_usd": -4_950.0},
    ]
    orders = net_underlying_orders(rows, {"COIN": 150.0}, min_trade_usd=200.0)
    assert len(orders) == 1
    assert orders[0]["decision"] == "skip"
    assert orders[0]["action"] is None


def test_netting_pure_b4_is_sell():
    rows = [{"underlying": "QBTS", "etf": "QBTZ", "under_delta_usd": -9_000.0}]
    orders = net_underlying_orders(rows, {"QBTS": 30.0}, min_trade_usd=200.0)
    assert orders[0]["action"] == "SELL"
    assert orders[0]["qty"] == 300


def test_netting_skips_when_no_price():
    rows = [{"underlying": "QBTS", "etf": "QBTZ", "under_delta_usd": -9_000.0}]
    orders = net_underlying_orders(rows, {}, min_trade_usd=200.0)
    assert orders[0]["decision"] == "skip"
    assert orders[0]["reason"] == "no_price_for_underlying"


# --------------------------------------------------------------------------
# Plan pair attrs
# --------------------------------------------------------------------------

def test_load_plan_pair_attrs_ratio_and_sleeve(tmp_path):
    p = tmp_path / "proposed_trades.csv"
    pd.DataFrame(
        [
            {"ETF": "QBTZ", "Underlying": "QBTS", "sleeve": "inverse_decay_bucket4",
             "long_usd": -46_000.0, "short_usd": -23_000.0},
            {"ETF": "COYY", "Underlying": "COIN", "sleeve": "yieldboost",
             "long_usd": 3_700.0, "short_usd": -10_000.0},
        ]
    ).to_csv(p, index=False)
    sleeves, ratios = load_plan_pair_attrs(p)
    assert sleeves["QBTZ"] == "inverse_decay_bucket4"
    assert ratios["QBTZ"] == pytest.approx(2.0)  # |−46000/−23000|
    assert ratios["COYY"] == pytest.approx(0.37)
