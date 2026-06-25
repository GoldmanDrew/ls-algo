"""Bucket 5 (volatility_etp_bucket5) wiring in plot_proposed_trades."""

import pandas as pd

from plot_proposed_trades import B4_CORE_SLEEVE, B5_SLEEVE, split_buckets


def test_split_buckets_routes_b5_by_sleeve():
    df = pd.DataFrame(
        [
            {"ETF": "TQQQ", "Underlying": "QQQ", "Delta": 3.0, "sleeve": "core_leveraged",
             "long_usd": 1000, "short_usd": -1000, "purgatory": False},
            {"ETF": "YMAX", "Underlying": "TSLA", "Delta": 1.2, "sleeve": "yieldboost",
             "long_usd": 500, "short_usd": -500, "purgatory": False},
            {"ETF": "SQQQ", "Underlying": "QQQ", "Delta": -3.0, "sleeve": B4_CORE_SLEEVE,
             "long_usd": 800, "short_usd": -800, "purgatory": False},
            {"ETF": "UVIX", "Underlying": "VIX", "Delta": -1.98, "sleeve": B5_SLEEVE,
             "long_usd": 400, "short_usd": -400, "purgatory": False},
        ]
    )
    b1, b2, b4, b5 = split_buckets(df)
    assert set(b1["ETF"]) == {"TQQQ"}
    assert set(b2["ETF"]) == {"YMAX"}
    assert set(b4["ETF"]) == {"SQQQ"}
    assert set(b5["ETF"]) == {"UVIX"}


def test_split_buckets_does_not_drop_b5_into_stock_fallback():
    df = pd.DataFrame(
        [
            {"ETF": "UVIX", "Underlying": "VIX", "Delta": -1.98, "sleeve": B5_SLEEVE,
             "long_usd": 400, "short_usd": -400, "purgatory": False},
        ]
    )
    b1, b2, b4, b5 = split_buckets(df)
    assert b1.empty and b2.empty and b4.empty
    assert len(b5) == 1
