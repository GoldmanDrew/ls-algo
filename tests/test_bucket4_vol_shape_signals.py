"""Tests for TR/VCR signal alignment (warmup / underlying recompute)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.bucket4_vol_shape_signals import get_pair_signal


def test_recompute_uses_pre_start_warmup():
    """Underlying series must include dates before calendar start for rolling window."""
    idx = pd.bdate_range("2025-08-01", periods=120)
    px = pd.Series(100.0 * np.cumprod(1.0 + 0.001 * np.random.default_rng(0).standard_normal(len(idx))), index=idx)
    cal = pd.bdate_range("2025-10-07", periods=40)

    sig_trunc = get_pair_signal(
        "ETF",
        "UND",
        cal,
        history={},
        underlying_prices=px.loc[cal],
        window=20,
        lookahead_shift=0,
        prefer_underlying_recompute=True,
    )
    sig_full = get_pair_signal(
        "ETF",
        "UND",
        cal,
        history={},
        underlying_prices=px,
        window=20,
        lookahead_shift=0,
        prefer_underlying_recompute=True,
    )

    assert sig_trunc["tr"].notna().sum() < sig_full["tr"].notna().sum()
    assert sig_full["tr"].notna().iloc[0]


def test_prefers_underlying_over_short_etf_history():
    short_hist = pd.DataFrame(
        {
            "tr": [1.1],
            "vcr": [0.2],
            "vcr_med": [0.2],
        },
        index=pd.DatetimeIndex(["2025-12-01"]),
    )
    idx = pd.bdate_range("2025-09-01", periods=80)
    px = pd.Series(np.linspace(100, 120, len(idx)), index=idx)
    cal = pd.bdate_range("2025-10-07", periods=30)

    sig = get_pair_signal(
        "ETF",
        "UND",
        cal,
        history={"ETF": short_hist},
        underlying_prices=px,
        window=20,
        lookahead_shift=0,
        prefer_underlying_recompute=True,
    )
    assert sig.attrs["signal_source"] == "recompute_underlying"
    assert sig["tr"].notna().sum() >= 10
