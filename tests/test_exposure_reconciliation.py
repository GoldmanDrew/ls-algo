from __future__ import annotations

import pandas as pd
import pytest

from ibkr_accounting import (
    _normalize_exposure_bucket_ratios,
    _spot_ledger_bucket_ratios,
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
        flow_low_delta_bucket3_syms=set(),
        flow_short_set=set(),
        b4_etf_syms=set(),
    )
    assert out["_ratio_b2"].iloc[0] == pytest.approx(1.0)
