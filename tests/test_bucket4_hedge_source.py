"""Tests for the tr_vcr engine source branch in build_bucket4_state helpers.

_build_hedge_cadence_engine must produce a per-underlying hedge series, a union
rebalance schedule, and reverse-engineerable cadence diagnostics.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.bucket4_weekly_opt2 import Bucket4WeeklyConfig, _build_hedge_cadence_engine


def _synthetic_closes(seed: int = 3, n: int = 300) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    rng = np.random.default_rng(seed)

    def gp(mu: float, sig: float) -> pd.Series:
        r = rng.normal(mu, sig, len(idx))
        return pd.Series(100 * np.exp(np.cumsum(r)), index=idx)

    return pd.DataFrame({
        "APLD": gp(0.0, 0.04), "NBIS": gp(0.0, 0.02),
        "APLZ": gp(0.0, 0.05), "NBIZ": gp(0.0, 0.03),
    })


def _cfg() -> Bucket4WeeklyConfig:
    return Bucket4WeeklyConfig(
        screened_csv="unused.csv", start="2024-01-01",
        hedge_source="tr_vcr", warmup_bdays=65, hedge_base=0.5,
        hedge_cadence_policy={
            "h_mid": 0.55, "k_vcr": 1.0, "h_min": 0.30, "h_max": 0.80,
            "m_vcr": 2.5, "base_days": 4.0, "max_interval": 10,
        },
    )


def test_engine_builds_hedge_and_cadence():
    closes = _synthetic_closes()
    pairs = [("APLZ", "APLD"), ("NBIZ", "NBIS")]
    master = closes.index.sort_values()
    hm, rebal, panel, cad = _build_hedge_cadence_engine(
        closes, pairs, ["APLD", "NBIS"], master, _cfg()
    )
    # hedge series cover the whole master calendar, within guardrails
    assert set(hm) == {"APLD", "NBIS"}
    for u, ser in hm.items():
        assert len(ser) == len(master)
        assert ser.dropna().between(0.30, 0.80).all()
    # union rebalance schedule is non-empty
    assert len(rebal) > 0
    # cadence diagnostics are reverse-engineerable
    for u in ("APLD", "NBIS"):
        c = cad[u]
        assert isinstance(c["interval_days"], int)
        assert 1 <= c["interval_days"] <= 10
        assert "denom=" in c["interval_explain"]
        assert "h=" in c["h_explain"]


def test_engine_falls_back_on_short_history():
    closes = _synthetic_closes(n=40)  # < warmup + 5
    pairs = [("APLZ", "APLD")]
    master = closes.index.sort_values()
    hm, rebal, panel, cad = _build_hedge_cadence_engine(
        closes, pairs, ["APLD"], master, _cfg()
    )
    assert (hm["APLD"] == 0.5).all()  # falls back to hedge_base
    assert cad["APLD"]["reason"] == "insufficient_history"
