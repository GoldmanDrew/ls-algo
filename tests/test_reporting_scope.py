from __future__ import annotations

import pandas as pd

from reporting_scope import (
    load_blocked_exposure_keys,
    screened_etf_and_underlying_sets,
    screened_universe_symbols,
)


def test_screened_universe_symbols_from_etf_and_underlying_columns():
    screened = pd.DataFrame(
        {
            "ETF": ["SPXL", "TQQQ"],
            "Underlying": ["SPY", "QQQ"],
        }
    )
    etfs, unders = screened_etf_and_underlying_sets(screened)
    assert etfs == {"SPXL", "TQQQ"}
    assert unders == {"SPY", "QQQ"}
    assert screened_universe_symbols(screened) == {"SPXL", "TQQQ", "SPY", "QQQ"}


def test_load_blocked_exposure_keys_empty_without_config(tmp_path):
    keys = load_blocked_exposure_keys(
        config_yml=tmp_path / "missing.yml",
        project_root=tmp_path,
    )
    assert keys == set()
