"""Low-N (short-history) uncertainty size haircut for admitted inverse names."""

import pandas as pd

from generate_trade_plan import apply_low_n_size_haircut


def _b4_frame():
    return pd.DataFrame([
        {"ETF": "SPCG", "Underlying": "SPCX", "sleeve": "inverse_decay_bucket4",
         "gross_target_usd": 600.0},
        {"ETF": "SMZ", "Underlying": "SMR", "sleeve": "inverse_decay_bucket4",
         "gross_target_usd": 22000.0},
        {"ETF": "TQQQ", "Underlying": "QQQ", "sleeve": "core_leveraged",
         "gross_target_usd": 50000.0},
    ])


def test_haircut_scales_only_low_n_keys():
    frame = _b4_frame()
    out = apply_low_n_size_haircut(frame, {("SPCG", "SPCX")}, 0.33)
    spcg = out.loc[out["ETF"] == "SPCG"].iloc[0]
    smz = out.loc[out["ETF"] == "SMZ"].iloc[0]
    tqqq = out.loc[out["ETF"] == "TQQQ"].iloc[0]
    assert abs(float(spcg["gross_target_usd"]) - 600.0 * 0.33) < 1e-6
    assert bool(spcg["low_n_included"]) is True
    assert abs(float(spcg["low_n_size_mult"]) - 0.33) < 1e-9
    # Non low-N B4 row and the core row untouched.
    assert abs(float(smz["gross_target_usd"]) - 22000.0) < 1e-6
    assert abs(float(tqqq["gross_target_usd"]) - 50000.0) < 1e-6


def test_haircut_noop_when_mult_is_one():
    frame = _b4_frame()
    out = apply_low_n_size_haircut(frame, {("SPCG", "SPCX")}, 1.0)
    assert abs(float(out.loc[out["ETF"] == "SPCG", "gross_target_usd"].iloc[0]) - 600.0) < 1e-6


def test_haircut_noop_when_no_keys():
    frame = _b4_frame()
    out = apply_low_n_size_haircut(frame, set(), 0.33)
    assert abs(float(out.loc[out["ETF"] == "SPCG", "gross_target_usd"].iloc[0]) - 600.0) < 1e-6


def test_haircut_scales_opt2_legs_when_present():
    frame = pd.DataFrame([{
        "ETF": "SPCG", "Underlying": "SPCX", "sleeve": "inverse_decay_bucket4",
        "gross_target_usd": 600.0,
        "b4_opt2_inverse_etf_short_usd": 400.0,
        "b4_opt2_underlying_short_usd": 200.0,
    }])
    out = apply_low_n_size_haircut(frame, {("SPCG", "SPCX")}, 0.5)
    r = out.iloc[0]
    assert abs(float(r["b4_opt2_inverse_etf_short_usd"]) - 200.0) < 1e-6
    assert abs(float(r["b4_opt2_underlying_short_usd"]) - 100.0) < 1e-6
