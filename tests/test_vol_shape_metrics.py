import math

import numpy as np
import pandas as pd

from screener_v2_fields import (
    _VOL_SHAPE_PRIMARY_WINDOW,
    _VOL_SHAPE_WINDOWS,
    _underlying_vol_shape,
    _underlying_vol_shape_20d,
    _underlying_vol_shape_panel,
)


def _tr_from_log_returns(returns):
    idx = pd.bdate_range("2026-01-01", periods=len(returns) + 1)
    levels = [100.0]
    for r in returns:
        levels.append(levels[-1] * math.exp(float(r)))
    return pd.Series(levels, index=idx)


def test_vol_shape_constant_grind_has_high_trend_ratio_low_vcr():
    """Perfect daily drift maxes the trend ratio at sqrt(5) at any window."""
    tr = _tr_from_log_returns([0.01] * 20)

    out = _underlying_vol_shape_20d(tr)

    assert out["und_vol_shape_20d"] == "quiet_trend"
    assert np.isclose(out["und_trend_ratio_20d"], math.sqrt(5.0), atol=1e-6)
    assert np.isclose(out["und_vcr_20d"], 0.05, atol=1e-6)
    assert np.isclose(out["und_return_20d"], 0.20, atol=1e-6)


def test_vol_shape_single_jump_has_high_vcr():
    tr = _tr_from_log_returns(([0.006] * 19) + [0.036277])

    out = _underlying_vol_shape_20d(tr)

    assert out["und_vcr_20d"] > 0.65


def test_vol_shape_alternating_path_is_mean_reverting():
    tr = _tr_from_log_returns([0.01 if i % 2 == 0 else -0.01 for i in range(20)])

    out = _underlying_vol_shape_20d(tr)

    assert out["und_vol_shape_20d"] == "quiet_chop"
    assert out["und_trend_ratio_20d"] < 0.5


def test_vol_shape_60d_constant_grind_matches_analytic_form():
    """Trend ratio is window-independent under the fixed 5-day annualization.

    For a constant log return r, both RV_daily and RV_weekly are independent
    of W: RV_d = |r|·sqrt(252), RV_w = 5|r|·sqrt(252/5), so TR = sqrt(5).
    """
    tr = _tr_from_log_returns([0.01] * 60)

    out = _underlying_vol_shape(tr, 60)

    assert np.isclose(out["und_trend_ratio_60d"], math.sqrt(5.0), atol=1e-6)
    # 60 days, each contributing 1/60 of variance
    assert np.isclose(out["und_vcr_60d"], 1.0 / 60.0, atol=1e-6)
    assert np.isclose(out["und_return_60d"], 0.60, atol=1e-6)


def test_vol_shape_iid_returns_have_trend_ratio_near_one():
    """A white-noise-ish path with sign flips lands TR near the iid baseline of 1."""
    rng = np.random.default_rng(7)
    returns = rng.normal(0.0, 0.01, size=600)
    tr = _tr_from_log_returns(returns.tolist())

    out = _underlying_vol_shape(tr, 60)

    # 600 samples is plenty: TR should be in a tight band around 1.
    assert 0.7 < out["und_trend_ratio_60d"] < 1.3


def test_vol_shape_panel_emits_both_windows():
    panel = _underlying_vol_shape_panel(_tr_from_log_returns([0.005] * 65))

    for w in _VOL_SHAPE_WINDOWS:
        assert f"und_trend_ratio_{w}d" in panel
        assert f"und_vcr_{w}d" in panel
        assert f"und_vcr_{w}d_median" in panel
        assert f"und_vol_shape_{w}d" in panel


def test_vol_shape_primary_window_is_60_to_align_with_risk_dashboard():
    """Display window must match risk_dashboard.beta_loader.DEFAULT_WINDOW_DAYS."""
    from risk_dashboard.beta_loader import DEFAULT_WINDOW_DAYS

    assert _VOL_SHAPE_PRIMARY_WINDOW == DEFAULT_WINDOW_DAYS == 60


def test_vol_shape_short_series_returns_empty_panel_for_long_window():
    """A 30-bar series cannot fill a 60d window; that slot stays empty
    while the 20d slot still produces values."""
    tr = _tr_from_log_returns([0.004] * 30)

    out_20 = _underlying_vol_shape(tr, 20)
    out_60 = _underlying_vol_shape(tr, 60)

    assert np.isfinite(out_20["und_trend_ratio_20d"])
    assert math.isnan(out_60["und_trend_ratio_60d"]) and out_60["und_vol_shape_60d"] == ""
