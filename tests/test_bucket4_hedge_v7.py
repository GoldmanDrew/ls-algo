"""Tests for bucket4_hedge_v7."""
from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.bucket4_hedge_v7 import (
    V7_GLOBAL_H_MAX,
    V7_GLOBAL_H_MIN,
    build_h_series_v7,
    resolve_pair_h_bounds,
    vcr_to_h_star,
)


def test_vcr_monotonic_h_star():
    mid = 0.55
    lo = vcr_to_h_star(0.10, 0.12, h_mid=mid, k_vcr=1.0)
    hi = vcr_to_h_star(0.35, 0.12, h_mid=mid, k_vcr=1.0)
    assert hi > lo


def test_pair_bounds_respect_global():
    lo, hi = resolve_pair_h_bounds("X", "Y", pair_bounds={("X", "Y"): (0.2, 0.9)})
    assert lo >= V7_GLOBAL_H_MIN
    assert hi <= V7_GLOBAL_H_MAX


def test_build_h_series_in_envelope():
    idx = pd.bdate_range("2025-10-07", periods=30)
    vcr = pd.Series(np.linspace(0.08, 0.30, len(idx)), index=idx)
    med = pd.Series(0.12, index=idx)
    sig = pd.DataFrame({"vcr": vcr, "vcr_med": med})
    h = build_h_series_v7(sig, idx, h_min=0.3, h_max=0.8, h_mid=0.55, k_vcr=1.0, smooth_alpha=0.0)
    assert h.min() >= V7_GLOBAL_H_MIN - 1e-9
    assert h.max() <= V7_GLOBAL_H_MAX + 1e-9
    assert float(np.corrcoef(vcr.values, h.values)[0, 1]) > 0.5
