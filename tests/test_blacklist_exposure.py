from __future__ import annotations

import pandas as pd

from ibkr_accounting import (
    _filter_exposure_df,
    _filter_positions_blacklist,
    expand_blacklist,
)


def test_expand_blacklist_maps_apld_complex() -> None:
    etf_to_under = {
        "APLD": "APLD",
        "APLX": "APLD",
        "APLZ": "APLD",
        "KEEX": "KEEL",
    }
    blocked_symbols, blocked_underlyings = expand_blacklist({"APLD"}, etf_to_under)
    assert blocked_underlyings == {"APLD"}
    assert blocked_symbols == {"APLD", "APLX", "APLZ"}


def test_filter_exposure_df_drops_blacklisted_underlying() -> None:
    df = pd.DataFrame(
        [
            {"underlying": "APLD", "net_notional_usd": 100.0, "gross_notional_usd": 100.0},
            {"underlying": "KEEL", "net_notional_usd": 50.0, "gross_notional_usd": 50.0},
        ]
    )
    out = _filter_exposure_df(df, {"APLD", "APLX", "APLZ"})
    assert out["underlying"].tolist() == ["KEEL"]


def test_filter_positions_blacklist_drops_etf_legs() -> None:
    pos = pd.DataFrame(
        [
            {"symbol": "APLX", "underlyingSymbol": "APLD", "position": 100},
            {"symbol": "KEEX", "underlyingSymbol": "KEEL", "position": 200},
        ]
    )
    etf_to_under = {"APLX": "APLD", "APLZ": "APLD", "KEEX": "KEEL"}
    out = _filter_positions_blacklist(pos, {"APLD", "APLX", "APLZ"}, {"APLD"}, etf_to_under)
    assert out["symbol"].tolist() == ["KEEX"]
