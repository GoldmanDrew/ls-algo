"""Tests for the Bucket 4 hedge-ratio + cadence engine (scripts/bucket4_hedge_cadence.py).

Each test asserts a value AND that the human-readable explanation lets you
reconstruct it (the 'reverse-engineerable' requirement).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from scripts.bucket4_hedge_cadence import (
    HedgeCadenceKnobs,
    NameTilt,
    build_h_series,
    build_rebal_dates,
    compute_pair_policy,
    load_name_tilts,
    load_policy_from_config,
)


def test_hedge_ratio_closed_form():
    k = HedgeCadenceKnobs(h_mid=0.55, k_vcr=1.0, h_min=0.30, h_max=0.80, alpha=0.0)
    p = compute_pair_policy(1.0, 0.061, 0.040, knobs=k)
    # h_raw = 0.55 + 1.0*(0.061-0.040) = 0.571 ; no EMA, no tilt
    assert p.h == pytest.approx(0.571, abs=1e-9)
    assert p.h_raw == pytest.approx(0.571, abs=1e-9)
    assert "0.571" in p.h_explain


def test_hedge_ratio_clip_upper():
    k = HedgeCadenceKnobs(h_mid=0.55, k_vcr=1.0, h_max=0.80, alpha=0.0)
    p = compute_pair_policy(1.0, 0.500, 0.040, knobs=k)  # huge VCR -> way above cap
    assert p.h == pytest.approx(0.80, abs=1e-9)
    assert "clip[0.30,0.80]=0.8000" in p.h_explain


def test_hedge_ratio_ema_smoothing():
    k = HedgeCadenceKnobs(h_mid=0.55, k_vcr=1.0, alpha=0.25)
    # h_clipped = 0.571 ; EMA from prev 0.50 = 0.75*0.50 + 0.25*0.571 = 0.51775
    p = compute_pair_policy(1.0, 0.061, 0.040, knobs=k, prev_h=0.50)
    assert p.h == pytest.approx(0.75 * 0.50 + 0.25 * 0.571, abs=1e-9)


def test_cadence_trending_is_faster():
    k = HedgeCadenceKnobs(base_days=4.0, k_tr=2.25, m_vcr=2.5, min_interval=1, max_interval=10)
    # TR=1.42, VCR=0.061, med=0.040: denom = 1 + 2.25*0.42 + 2.5*0.021 = 1.9975
    p = compute_pair_policy(1.42, 0.061, 0.040, knobs=k)
    assert p.denom == pytest.approx(1.9975, abs=1e-6)
    assert p.interval_raw == pytest.approx(4.0 / 1.9975, abs=1e-6)
    assert p.interval_days == 2


def test_cadence_choppy_is_slower():
    k = HedgeCadenceKnobs(base_days=4.0, k_tr=2.25, m_vcr=2.5, min_interval=1, max_interval=10)
    # TR=0.90, VCR=0.030, med=0.045: denom = 1 - 0.225 - 0.0375 = 0.7375 -> 4/0.7375=5.42 -> 5
    p = compute_pair_policy(0.90, 0.030, 0.045, knobs=k)
    assert p.denom == pytest.approx(0.7375, abs=1e-6)
    assert p.interval_days == 5


def test_cadence_clipped_to_max():
    k = HedgeCadenceKnobs(base_days=4.0, k_tr=2.25, m_vcr=2.5, min_interval=1, max_interval=10)
    # very choppy -> tiny denom -> huge raw interval -> clip to max
    p = compute_pair_policy(0.60, 0.010, 0.060, knobs=k)
    assert p.interval_days == 10


def test_missing_signal_is_neutral():
    k = HedgeCadenceKnobs(h_mid=0.55, base_days=4.0, max_interval=10)
    p = compute_pair_policy(np.nan, np.nan, np.nan, knobs=k)
    assert not p.signal_ok
    assert p.h == pytest.approx(0.55, abs=1e-9)        # neutral hedge
    assert p.interval_days == 4                         # base_days, denom=1
    assert "signal missing" in p.h_explain


def test_name_tilt_applies_and_stays_bounded():
    k = HedgeCadenceKnobs(h_mid=0.55, k_vcr=1.0, h_min=0.30, h_max=0.80)
    tilt = NameTilt(h_mult=0.9, h_shift=0.0, interval_mult=1.5, note="tail-cap")
    p = compute_pair_policy(1.0, 0.061, 0.040, knobs=k, name_tilt=tilt)
    # h_raw=0.571 -> *0.9 = 0.5139 (within bounds)
    assert p.h == pytest.approx(0.571 * 0.9, abs=1e-9)
    assert k.h_min <= p.h <= k.h_max
    # a fat-finger tilt cannot escape the envelope
    bad = NameTilt(h_mult=5.0)
    p2 = compute_pair_policy(1.0, 0.061, 0.040, knobs=k, name_tilt=bad)
    assert p2.h == pytest.approx(k.h_max, abs=1e-9)


def test_tilt_interval_mult():
    k = HedgeCadenceKnobs(base_days=4.0, k_tr=2.25, m_vcr=2.5, min_interval=1, max_interval=10)
    base = compute_pair_policy(1.42, 0.061, 0.040, knobs=k)         # 2d raw 2.003
    tilted = compute_pair_policy(1.42, 0.061, 0.040, knobs=k,
                                 name_tilt=NameTilt(interval_mult=2.0))
    assert tilted.interval_days >= base.interval_days
    assert tilted.interval_days == 4  # 2.003 * 2 = 4.006 -> round 4


def test_build_h_series_ema_path():
    k = HedgeCadenceKnobs(h_mid=0.55, k_vcr=1.0, alpha=0.25)
    idx = pd.date_range("2026-01-01", periods=5, freq="B")
    sig = pd.DataFrame({"tr": 1.0, "vcr": 0.061, "vcr_med": 0.040}, index=idx)
    h = build_h_series(sig, idx, knobs=k)
    assert len(h) == 5
    # monotone approach toward 0.571 from first value
    assert h.iloc[0] == pytest.approx(0.571, abs=1e-9)  # prev_h None on first -> clipped value
    assert h.iloc[-1] == pytest.approx(0.571, abs=1e-3)


def test_build_rebal_dates_steps_by_interval():
    k = HedgeCadenceKnobs(base_days=4.0, k_tr=2.25, m_vcr=2.5, max_interval=10)
    idx = pd.date_range("2026-01-01", periods=40, freq="B")
    # steady trending signal -> ~2 day spacing
    sig = pd.DataFrame({"tr": 1.42, "vcr": 0.061, "vcr_med": 0.040}, index=idx)
    dates, diag = build_rebal_dates(sig, idx, knobs=k)
    assert len(dates) > 0
    assert "interval_explain" in diag.columns
    gaps = np.diff(dates.values).astype("timedelta64[D]").astype(int)
    # business-day spacing of 2 sessions ~ 2-4 calendar days
    assert gaps.min() >= 1


def test_load_name_tilts_and_config():
    block = {
        "source": "tr_vcr",
        "h_mid": 0.50,
        "name_tilt": {"UVIX": {"h_mult": 0.9, "note": "tail-cap"}},
    }
    cfg = {"portfolio": {"sleeves": {"inverse_decay_bucket4": {"rules": {"hedge_cadence_policy": block}}}}}
    knobs, tilts, source = load_policy_from_config(cfg)
    assert source == "tr_vcr"
    assert knobs.h_mid == 0.50
    assert "UVIX" in tilts and tilts["UVIX"].h_mult == 0.9
