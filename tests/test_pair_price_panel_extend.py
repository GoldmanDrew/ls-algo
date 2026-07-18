"""Unit tests for Yahoo/metrics price-panel extension helpers."""
from __future__ import annotations

import pandas as pd

from scripts.pair_price_panel import (
    _align_extend,
    _beta_etf_vs_und,
    _synthesize_etf_from_und,
)


def test_align_extend_level_matches_at_join():
    idx = pd.bdate_range("2026-06-01", periods=5)
    base = pd.Series([10.0, 10.5, 11.0, 11.5, 12.0], index=idx)
    # Yahoo on a different scale overlapping last two days, then continues.
    ext_idx = pd.bdate_range("2026-06-04", periods=5)
    ext = pd.Series([100.0, 110.0, 120.0, 130.0, 140.0], index=ext_idx)
    out = _align_extend(base, ext)
    assert out.index.max() == ext_idx.max()
    # First new day after base should continue from last base level (scaled).
    assert abs(float(out.loc[idx[-1]]) - 12.0) < 1e-9
    # Tail after join should move with Yahoo returns on the extended bars.
    tail = out.loc[out.index > idx[-1]]
    assert len(tail) >= 2
    r0 = float(tail.iloc[1] / tail.iloc[0] - 1.0)
    assert abs(r0 - (130.0 / 120.0 - 1.0)) < 1e-9


def test_synthesize_etf_from_und_rolls_with_beta():
    idx = pd.bdate_range("2026-06-01", periods=10)
    # Inverse ~ -2x und.
    und = pd.Series([100.0 * (1.01**i) for i in range(10)], index=idx)
    etf = pd.Series([50.0], index=idx[:1])
    for i in range(1, 7):
        r = float(und.iloc[i] / und.iloc[i - 1] - 1.0)
        etf.loc[idx[i]] = float(etf.iloc[-1]) * (1.0 - 2.0 * r)
    # Und continues 3 more days without ETF prints.
    a = etf.iloc[:7]
    b = und
    out = _synthesize_etf_from_und(a, b)
    assert out.index.max() == idx.max()
    assert len(out) == 10
    beta = _beta_etf_vs_und(a, b.loc[: a.index.max()])
    assert beta < -1.0  # inverse
    # Last synthetic day should not equal flat last ETF print.
    assert abs(float(out.iloc[-1]) - float(a.iloc[-1])) > 1e-6
