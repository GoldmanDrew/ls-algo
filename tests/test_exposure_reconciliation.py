from __future__ import annotations

import pandas as pd
import pytest

from ibkr_accounting import (
    SpotBucketRatios,
    _normalize_exposure_bucket_ratios,
    _spot_ledger_bucket_ratios,
    apply_b4_structural_short_to_exposure_detail,
    build_net_exposure_unbucketed,
    complete_etf_maps_from_positions,
)


def test_complete_etf_maps_adds_one_x_benchmark_holdings():
    etf_to_under = {"SPXL": "SPY", "TQQQ": "QQQ"}
    etf_to_delta = {"SPXL": 3.0, "TQQQ": 3.0}
    pos = pd.DataFrame([{"symbol": "SPY", "position": 1.0}])

    out_under, out_delta = complete_etf_maps_from_positions(pos, etf_to_under, etf_to_delta)

    assert out_under["SPY"] == "SPY"
    assert out_delta["SPY"] == pytest.approx(1.0)


def test_spot_ledger_bucket_ratios_caps_stale_ledger_and_assigns_orphan():
    r1, r2, r4 = _spot_ledger_bucket_ratios(
        100.0,
        {"bucket_1": 250.0, "bucket_2": 750.0, "bucket_4": 0.0},
    )
    assert r1 == pytest.approx(0.25)
    assert r2 == pytest.approx(0.75)
    assert r4 == pytest.approx(0.0)

    r1, r2, r4 = _spot_ledger_bucket_ratios(
        1000.0,
        {"bucket_1": 100.0, "bucket_2": 0.0, "bucket_4": 0.0},
    )
    assert r1 == pytest.approx(1.0)
    assert r2 == pytest.approx(0.0)
    assert r4 == pytest.approx(0.0)


def test_normalize_exposure_bucket_ratios_fills_zero_sum_etf_row():
    detail = pd.DataFrame(
        [
            {
                "symbol": "SPY",
                "underlying": "SPY",
                "gross_notional_usd": 1000.0,
                "_is_etf": True,
                "_ratio_b1": 0.0,
                "_ratio_b2": 0.0,
                "_ratio_b4": 0.0,
            }
        ]
    )
    out = _normalize_exposure_bucket_ratios(
        detail,
        etf_to_delta_map={"SPY": 1.0},
        flow_short_set=set(),
        b4_etf_syms=set(),
    )
    assert out["_ratio_b2"].iloc[0] == pytest.approx(1.0)


def test_normalize_b4_registry_spot_preserves_fifo_b12_ratios():
    """B4 registry membership alone must not zero spot bucket ratios."""
    detail = pd.DataFrame(
        [
            {
                "symbol": "IONQ",
                "underlying": "IONQ",
                "gross_notional_usd": 60_000.0,
                "_is_etf": False,
                "_ratio_b1": 1.0,
                "_ratio_b2": 0.0,
                "_ratio_b4": 0.0,
            }
        ]
    )
    out = _normalize_exposure_bucket_ratios(
        detail,
        etf_to_delta_map={"IONQ": 1.0},
        flow_short_set=set(),
        b4_etf_syms=set(),
        b4_spot_b12_only_underlyings={"IONQ"},
    )
    assert out["_ratio_b1"].iloc[0] == pytest.approx(1.0)
    assert out["_ratio_b2"].iloc[0] == pytest.approx(0.0)
    assert out["_ratio_b4"].iloc[0] == pytest.approx(0.0)


def test_normalize_structural_b4_spot_clears_b4_and_renormalizes_b12():
    detail = pd.DataFrame(
        [
            {
                "symbol": "COIN",
                "underlying": "COIN",
                "gross_notional_usd": 100_000.0,
                "_is_etf": False,
                "_ratio_b1": 0.5,
                "_ratio_b2": 0.2,
                "_ratio_b4": 0.3,
            }
        ]
    )
    out = _normalize_exposure_bucket_ratios(
        detail,
        etf_to_delta_map={"COIN": 1.0},
        flow_short_set=set(),
        b4_etf_syms=set(),
        b4_spot_b12_only_underlyings={"COIN"},
    )
    assert out["_ratio_b4"].iloc[0] == pytest.approx(0.0)
    assert out["_ratio_b1"].iloc[0] + out["_ratio_b2"].iloc[0] == pytest.approx(1.0)
    assert out["_ratio_b1"].iloc[0] == pytest.approx(0.5 / 0.7)


def test_normalize_sleeve_balance_preserves_orphan_frac_on_spot():
    detail = pd.DataFrame(
        [
            {
                "symbol": "IONQ",
                "underlying": "IONQ",
                "gross_notional_usd": 60_000.0,
                "_is_etf": False,
                "_ratio_b1": 0.323,
                "_ratio_b2": 0.602,
                "_ratio_b4": 0.0,
            }
        ]
    )
    out = _normalize_exposure_bucket_ratios(
        detail,
        etf_to_delta_map={"IONQ": 1.0},
        flow_short_set=set(),
        b4_etf_syms=set(),
        preserve_partial_spot_ratios=True,
    )
    assert out["_ratio_b1"].iloc[0] + out["_ratio_b2"].iloc[0] == pytest.approx(0.925, rel=1e-3)


def test_build_net_exposure_unbucketed_from_partial_ratios():
    detail = pd.DataFrame(
        [
            {
                "symbol": "IONQ",
                "underlying": "IONQ",
                "net_notional_usd": 60_000.0,
                "gross_notional_usd": 60_000.0,
                "_is_etf": False,
                "_ratio_b1": 0.323,
                "_ratio_b2": 0.602,
                "_ratio_b4": 0.0,
            }
        ]
    )
    out = build_net_exposure_unbucketed(detail, {"IONQ": "IONQ"})
    assert len(out) == 1
    assert out["net_notional_usd"].iloc[0] == pytest.approx(60_000.0 * 0.075, rel=1e-2)


def test_apply_b4_structural_short_carves_spot_line_and_preserves_book():
    detail = pd.DataFrame(
        [
            {
                "symbol": "COIN",
                "underlying": "COIN",
                "net_notional_usd": 60_000.0,
                "gross_notional_usd": 60_000.0,
                "_is_etf": False,
                "_ratio_b1": 0.6,
                "_ratio_b2": 0.4,
                "_ratio_b4": 0.0,
                "leg_class": "spot",
            }
        ]
    )
    out = apply_b4_structural_short_to_exposure_detail(
        detail,
        {"COIN": -15_000.0},
        hedge_spot_ratio_map={
            "COIN": SpotBucketRatios(0.6, 0.4, 0.0, "hedge_ratio"),
        },
    )
    assert out["_ratio_b4"].iloc[0] == pytest.approx(-15_000.0 / 60_000.0)
    assert out["_ratio_b1"].iloc[0] + out["_ratio_b2"].iloc[0] + out["_ratio_b4"].iloc[0] == pytest.approx(1.0)
    b4_net = float(out["net_notional_usd"].iloc[0] * out["_ratio_b4"].iloc[0])
    assert b4_net == pytest.approx(-15_000.0)
