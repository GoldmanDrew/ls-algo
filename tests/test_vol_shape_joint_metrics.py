"""Vol-shape on joint etf_metrics_daily matches dashboard product truth."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from vol_shape import (
    VOL_SHAPE_PRIMARY_WINDOW,
    load_joint_vol_shape_panels_by_etf,
    resolve_etf_metrics_daily_path,
    underlying_vol_shape_panel,
)


def _metrics_path() -> Path | None:
    return resolve_etf_metrics_daily_path()


@pytest.mark.skipif(_metrics_path() is None, reason="etf_metrics_daily.csv not available")
def test_aplx_joint_metrics_differs_from_full_underlying_tr():
    """APLX TR on joint metrics is lower than naive full-series export (regression guard)."""
    path = _metrics_path()
    assert path is not None
    joint = load_joint_vol_shape_panels_by_etf(path, {"APLX"})
    assert "APLX" in joint
    tr_joint = joint["APLX"][f"und_trend_ratio_{VOL_SHAPE_PRIMARY_WINDOW}d"]
    assert tr_joint is not None
    assert float(tr_joint) < 1.0
    assert joint["APLX"]["und_vol_shape_price_basis"] == "joint_etf_metrics"


@pytest.mark.skipif(_metrics_path() is None, reason="etf_metrics_daily.csv not available")
def test_joint_panel_matches_recompute_from_csv_rows():
    path = _metrics_path()
    assert path is not None
    import pandas as pd

    from vol_shape import joint_metrics_price_series

    df = pd.read_csv(path)
    grp = df[df["ticker"].astype(str).str.upper() == "APLX"]
    px = joint_metrics_price_series(grp)
    assert px is not None
    direct = underlying_vol_shape_panel(px)
    cached = load_joint_vol_shape_panels_by_etf(path, {"APLX"})["APLX"]
    for key in (f"und_trend_ratio_{VOL_SHAPE_PRIMARY_WINDOW}d", f"und_vcr_{VOL_SHAPE_PRIMARY_WINDOW}d"):
        assert np.isclose(float(direct[key]), float(cached[key]), rtol=0, atol=1e-6)
