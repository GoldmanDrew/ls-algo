import math

import numpy as np
import pandas as pd

from screener_v2_fields import _underlying_vol_shape_20d


def _tr_from_log_returns(returns):
    idx = pd.bdate_range("2026-01-01", periods=len(returns) + 1)
    levels = [100.0]
    for r in returns:
        levels.append(levels[-1] * math.exp(float(r)))
    return pd.Series(levels, index=idx)


def test_vol_shape_constant_grind_has_high_trend_ratio_low_vcr():
    tr = _tr_from_log_returns([0.01] * 20)

    out = _underlying_vol_shape_20d(tr)

    assert out["und_vol_shape_20d"] == "quiet_trend"
    assert np.isclose(out["und_trend_ratio_20d"], math.sqrt(1.25), atol=1e-6)
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
