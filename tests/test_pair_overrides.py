"""Tests for operator B4/B5 pair overrides (config/pair_overrides.yml)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from strategy_config import load_pair_overrides  # noqa: E402
from generate_trade_plan import apply_pair_overrides_to_sized  # noqa: E402


def _sized_row(sleeve="inverse_decay_bucket4", etf="BITX", und="IBIT", gross=10000.0, h=0.45, beta=1.0):
    # Engine identity: inv = gross/(1+h*beta), und = (gross-inv)*phr (phr=1 here).
    denom = 1.0 + h * beta
    inv = gross / denom
    und_usd = gross - inv
    return pd.DataFrame(
        [{
            "ETF": etf,
            "Underlying": und,
            "sleeve": sleeve,
            "gross_target_usd": gross,
            "delta_abs": beta,
            "b4_opt2_inverse_etf_short_usd": inv,
            "b4_opt2_underlying_short_usd": und_usd,
            "b4_opt2_hedge_ratio": h,
        }]
    )


def _apply(frame, overrides, h_min=0.30, h_max=0.80, phr=1.0, delta_floor=0.1):
    return apply_pair_overrides_to_sized(
        frame, overrides, h_min=h_min, h_max=h_max, partial_hedge_ratio=phr, delta_floor=delta_floor
    )


# --------------------------- load_pair_overrides ---------------------------
def test_load_normalizes_keys_and_defaults():
    cfg = {"pair_overrides": {"bitx/ibit": {"hedge_ratio_add": 0.05, "gross_mult": 1.2, "note": "x"}}}
    out = load_pair_overrides(cfg)
    assert ("BITX", "IBIT") in out
    assert out[("BITX", "IBIT")]["hedge_ratio_add"] == 0.05
    assert out[("BITX", "IBIT")]["gross_mult"] == 1.2
    assert out[("BITX", "IBIT")]["note"] == "x"


def test_load_skips_malformed_and_bad_gross_mult():
    cfg = {"pair_overrides": {
        "NOSLASH": {"gross_mult": 2.0},
        "A/B": "not-a-dict",
        "C/D": {"gross_mult": -1.0},   # non-positive -> reset to 1.0
        "E/F": {"gross_mult": "bad"},  # unparseable -> 1.0
    }}
    out = load_pair_overrides(cfg)
    assert ("NOSLASH",) not in out and len(out) == 2
    assert out[("C", "D")]["gross_mult"] == 1.0
    assert out[("E", "F")]["gross_mult"] == 1.0


def test_load_empty_or_missing():
    assert load_pair_overrides({}) == {}
    assert load_pair_overrides({"pair_overrides": None}) == {}


# ------------------------- apply: gross multiplier -------------------------
def test_gross_mult_scales_gross_and_legs_preserves_hedge():
    df = _sized_row(gross=10000.0, h=0.45)
    inv0 = df.at[0, "b4_opt2_inverse_etf_short_usd"]
    und0 = df.at[0, "b4_opt2_underlying_short_usd"]
    out = _apply(df, {("BITX", "IBIT"): {"gross_mult": 1.2, "hedge_ratio_add": 0.0, "note": ""}})
    assert out.at[0, "gross_target_usd"] == pytest.approx(12000.0)
    assert out.at[0, "b4_opt2_inverse_etf_short_usd"] == pytest.approx(inv0 * 1.2)
    assert out.at[0, "b4_opt2_underlying_short_usd"] == pytest.approx(und0 * 1.2)
    assert out.at[0, "b4_opt2_hedge_ratio"] == pytest.approx(0.45)  # unchanged
    assert out.at[0, "pair_override_gross_mult"] == pytest.approx(1.2)
    assert out.at[0, "gross_target_usd_pre_override"] == pytest.approx(10000.0)


# --------------------------- apply: hedge add ------------------------------
def test_hedge_add_resolves_legs_from_engine_identity():
    df = _sized_row(gross=10000.0, h=0.45, beta=1.0)
    out = _apply(df, {("BITX", "IBIT"): {"hedge_ratio_add": 0.05, "gross_mult": 1.0}})
    assert out.at[0, "b4_opt2_hedge_ratio"] == pytest.approx(0.50)
    # h'=0.5, beta=1, gross=10000 -> inv=10000/1.5, und=gross-inv
    assert out.at[0, "b4_opt2_inverse_etf_short_usd"] == pytest.approx(10000.0 / 1.5)
    assert out.at[0, "b4_opt2_underlying_short_usd"] == pytest.approx(10000.0 - 10000.0 / 1.5)
    assert out.at[0, "pair_override_hedge_add"] == pytest.approx(0.05)
    assert out.at[0, "b4_opt2_hedge_ratio_pre_override"] == pytest.approx(0.45)


def test_hedge_add_clipped_to_guardrails():
    df = _sized_row(h=0.78)
    out = _apply(df, {("BITX", "IBIT"): {"hedge_ratio_add": 0.10}}, h_max=0.80)
    assert out.at[0, "b4_opt2_hedge_ratio"] == pytest.approx(0.80)


def test_gross_then_hedge_combine():
    df = _sized_row(gross=10000.0, h=0.45, beta=1.0)
    out = _apply(df, {("BITX", "IBIT"): {"gross_mult": 1.2, "hedge_ratio_add": 0.05}})
    # gross first -> 12000, then legs re-solved at h'=0.5
    assert out.at[0, "gross_target_usd"] == pytest.approx(12000.0)
    assert out.at[0, "b4_opt2_inverse_etf_short_usd"] == pytest.approx(12000.0 / 1.5)
    assert out.at[0, "b4_opt2_underlying_short_usd"] == pytest.approx(12000.0 - 12000.0 / 1.5)


# ------------------------------ scope rules --------------------------------
def test_bucket5_in_scope():
    df = _sized_row(sleeve="volatility_etp_bucket5", etf="UVIX", und="SVIX")
    out = _apply(df, {("UVIX", "SVIX"): {"gross_mult": 0.5}})
    assert out.at[0, "gross_target_usd"] == pytest.approx(5000.0)


def test_non_b4_sleeve_ignored():
    df = _sized_row(sleeve="core_leveraged", etf="TQQQ", und="QQQ", gross=10000.0)
    out = _apply(df, {("TQQQ", "QQQ"): {"gross_mult": 2.0}})
    assert out.at[0, "gross_target_usd"] == pytest.approx(10000.0)  # untouched


def test_unknown_pair_ignored():
    df = _sized_row(etf="BITX", und="IBIT")
    out = _apply(df, {("FOO", "BAR"): {"gross_mult": 2.0}})
    assert out.at[0, "gross_target_usd"] == pytest.approx(10000.0)


def test_empty_overrides_noop():
    df = _sized_row()
    out = _apply(df, {})
    pd.testing.assert_frame_equal(out, df)


def test_hedge_add_without_opt2_columns_applies_gross_only(capsys):
    df = pd.DataFrame([{
        "ETF": "BITX", "Underlying": "IBIT", "sleeve": "inverse_decay_bucket4",
        "gross_target_usd": 10000.0, "delta_abs": 1.0,
    }])
    out = _apply(df, {("BITX", "IBIT"): {"gross_mult": 1.5, "hedge_ratio_add": 0.1}})
    assert out.at[0, "gross_target_usd"] == pytest.approx(15000.0)
    assert "ignored" in capsys.readouterr().out
