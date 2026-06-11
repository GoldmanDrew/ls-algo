"""Tests for Bucket 4 WS4/WS5 sizing overlays (concentration + cluster caps)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.bucket4_sizing import (  # noqa: E402
    apply_cluster_caps_to_b4,
    apply_concentration_to_b4,
    concentration_scores,
)


def _frame():
    return pd.DataFrame({
        "ETF": ["AAA", "BBB", "CCC", "DDD", "EEE"],
        "Underlying": ["UA", "MSTR", "UC", "COIN", "UE"],
        "bucket4_net_edge_annual": [3.0, 2.0, 1.0, 0.5, np.nan],
        "borrow_current": [0.10, 0.50, 0.05, 0.05, 0.05],
        "vol_underlying_annual": [1.0, 1.0, 0.5, 0.5, 0.8],
        "purgatory": [False] * 5,
    })


def test_scores_rank_edge_over_borrow_per_vol():
    s = concentration_scores(_frame())
    # AAA: (3-0.1)/1=2.9 | BBB: 1.5 | CCC: (0.95)/0.5=1.9 | DDD: 0.9 | EEE: -inf
    assert s.idxmax() == 0
    assert s.iloc[4] == -np.inf


def test_concentration_drops_unheld_keeps_held_open():
    df = _frame()
    w = np.full(5, 0.2)
    out, w2, info = apply_concentration_to_b4(df, w, top_n=2, held={"CCC"})
    # top-2 = AAA, CCC(1.9) ... wait CCC is in top2; held CCC irrelevant then
    assert set(out["ETF"]) == {"AAA", "CCC"}
    assert info["n_dropped"] == 3 and info["n_keep_open"] == 0
    assert w2.sum() == pytest.approx(1.0)


def test_concentration_held_leftover_goes_keep_open():
    df = _frame()
    w = np.full(5, 0.2)
    out, w2, info = apply_concentration_to_b4(df, w, top_n=1, held={"DDD"})
    # top-1 = AAA; DDD held -> keep-open row with weight 0; BBB/CCC/EEE dropped
    assert set(out["ETF"]) == {"AAA", "DDD"}
    assert info == {"n_dropped": 3, "n_keep_open": 1}
    d_row = out[out["ETF"] == "DDD"].iloc[0]
    assert bool(d_row["purgatory"]) is True
    assert w2[list(out["ETF"]).index("DDD")] == 0.0
    assert w2.sum() == pytest.approx(1.0)


def test_concentration_noop_when_disabled_or_small():
    df = _frame()
    w = np.full(5, 0.2)
    out, w2, info = apply_concentration_to_b4(df, w, top_n=0, held=set())
    assert len(out) == 5 and info["n_dropped"] == 0
    out, w2, info = apply_concentration_to_b4(df, w, top_n=10, held=set())
    assert len(out) == 5


def test_cluster_cap_scales_and_redistributes():
    df = _frame()
    w = np.array([0.1, 0.4, 0.1, 0.3, 0.1])  # crypto (MSTR+COIN) = 0.7
    caps = {"crypto": {"cap": 0.35, "members": ["MSTR", "COIN"]}}
    w2, info = apply_cluster_caps_to_b4(df, w, caps)
    crypto = w2[[1, 3]].sum()
    assert crypto == pytest.approx(0.35)
    assert w2.sum() == pytest.approx(1.0)
    assert info["crypto"]["capped"] is True


def test_cluster_cap_noop_below_cap():
    df = _frame()
    w = np.array([0.4, 0.1, 0.3, 0.1, 0.1])  # crypto = 0.2 < 0.35
    caps = {"crypto": {"cap": 0.35, "members": ["MSTR", "COIN"]}}
    w2, info = apply_cluster_caps_to_b4(df, w, caps)
    assert np.allclose(w, w2)
    assert info["crypto"]["capped"] is False
