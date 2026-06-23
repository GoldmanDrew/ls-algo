"""Tests for TR/VCR signal alignment (warmup / underlying recompute)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.bucket4_vol_shape_signals import get_pair_signal, load_vol_shape_history, policy_continuous_interval


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


def test_history_loader_preserves_forward_and_cadence_metrics(tmp_path):
    path = tmp_path / "vol_shape_history.json"
    path.write_text(
        """
        {
          "symbols": {
            "ETF": {
              "series": [
                {
                  "date": "2026-01-05",
                  "trend_ratio": 1.1,
                  "trend_ratio_fwd": 1.07,
                  "rebalance_cadence_score": 1.12,
                  "vcr": 0.2,
                  "vcr_median": 0.18,
                  "rv_daily": 0.3,
                  "rv_weekly": 0.33
                }
              ]
            }
          }
        }
        """,
        encoding="utf-8",
    )
    hist = load_vol_shape_history(path)
    frame = hist["ETF"]
    assert frame.loc[pd.Timestamp("2026-01-05"), "tr_est"] == 1.07
    assert frame.loc[pd.Timestamp("2026-01-05"), "cadence_score"] == 1.12


def test_policy_can_use_cadence_score_signal_column():
    cal = pd.bdate_range("2026-01-01", periods=8)
    sig = pd.DataFrame(
        {
            "tr": [1.0] * len(cal),
            "cadence_score": [1.2] * len(cal),
            "vcr": [0.2] * len(cal),
            "vcr_med": [0.2] * len(cal),
        },
        index=cal,
    )
    _, diag = policy_continuous_interval(
        cal,
        sig,
        signal_col="cadence_score",
        base_days=5,
        k_tr=2.0,
        m_vcr=0.0,
        min_interval=1,
        max_interval=10,
    )
    assert diag["signal_col"].iloc[0] == "cadence_score"
    assert diag["signal"].iloc[0] == 1.2
