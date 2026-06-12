"""Tests for the flat vol-ETP sizing haircut in the B4 weight engine."""
from __future__ import annotations

import pandas as pd
import pytest

from scripts.v6_b4_pf_weights import (
    V6PfParams,
    apply_vol_etp_weight_penalty,
    is_vol_etp_pair,
)


def test_is_vol_etp_pair_detects_vix_complex():
    assert is_vol_etp_pair("UVIX", "SVIX")
    assert is_vol_etp_pair("UVXY", "VIX")
    assert is_vol_etp_pair("uvix", "svix")  # case-insensitive
    assert not is_vol_etp_pair("CLSZ", "CLSK")
    assert not is_vol_etp_pair("MSTZ", "MSTR")


def test_penalty_haircuts_vol_pair_and_redistributes():
    w = pd.Series([0.25, 0.45, 0.30], index=["UVIX/SVIX", "LITZ/LITE", "CLSZ/CLSK"])
    is_vol = pd.Series([True, False, False], index=w.index)
    out = apply_vol_etp_weight_penalty(w, is_vol, penalty=1.0 / 3.0)

    assert out.sum() == pytest.approx(1.0)
    # vol pair: 0.25 * (2/3) = 0.1667 pre-renorm; renorm factor 1/0.9167
    assert out["UVIX/SVIX"] == pytest.approx(0.25 * (2 / 3) / (1 - 0.25 / 3))
    assert out["UVIX/SVIX"] < 0.25 * 0.75  # materially below original
    # non-vol pairs keep their relative ratio and absorb the freed weight
    assert out["LITZ/LITE"] / out["CLSZ/CLSK"] == pytest.approx(0.45 / 0.30)
    assert out["LITZ/LITE"] > 0.45


def test_penalty_zero_or_no_vol_pairs_is_identity():
    w = pd.Series([0.5, 0.5], index=["A/B", "C/D"])
    no_vol = pd.Series([False, False], index=w.index)
    pd.testing.assert_series_equal(apply_vol_etp_weight_penalty(w, no_vol, penalty=0.333), w)
    some_vol = pd.Series([True, False], index=w.index)
    pd.testing.assert_series_equal(apply_vol_etp_weight_penalty(w, some_vol, penalty=0.0), w)


def test_params_default_off():
    assert V6PfParams().vol_etp_weight_penalty == 0.0
