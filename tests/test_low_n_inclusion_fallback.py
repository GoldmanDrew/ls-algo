"""Low-N edge/vol fallback helpers (CBRZ-style rows with no bootstrap edge)."""

import pandas as pd

from generate_trade_plan import _low_n_mechanical_net_edge, _resolve_low_n_fallback_edge


def _cbrz_row() -> pd.Series:
    return pd.Series({
        "ETF": "CBRZ",
        "Underlying": "CBRS",
        "Delta": -2.0,
        "Delta_quality": "low_n",
        "Delta_n_obs": 17,
        "primary_edge_annual": float("nan"),
        "net_decay_annual": float("nan"),
        "blended_gross_decay": float("nan"),
        "expected_gross_decay_annual": float("nan"),
        "vol_underlying_annual": float("nan"),
        "und_rv_20d_daily_annual": 1.388139,
        "expense_ratio_annual": 0.0149,
        "borrow_current": 0.107197,
    })


def test_mechanical_net_edge_positive_for_cbrz_like_row():
    row = _cbrz_row()
    edge = _low_n_mechanical_net_edge(row, vol=1.388139)
    assert edge > 0.5


def test_resolve_fallback_uses_mechanical_when_screener_edge_missing():
    row = _cbrz_row()
    edge = _resolve_low_n_fallback_edge(
        row,
        edge_cols=["primary_edge_annual", "net_decay_annual"],
        vol_cols=["und_rv_60d_daily_annual", "und_rv_20d_daily_annual"],
        use_mechanical=True,
    )
    assert edge > 0.5


def test_resolve_fallback_returns_nan_when_mechanical_disabled_and_no_columns():
    row = _cbrz_row()
    edge = _resolve_low_n_fallback_edge(
        row,
        edge_cols=["primary_edge_annual"],
        vol_cols=["und_rv_20d_daily_annual"],
        use_mechanical=False,
    )
    assert pd.isna(edge)
