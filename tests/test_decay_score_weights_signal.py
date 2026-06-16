import numpy as np
import pandas as pd

from generate_trade_plan import (
    _decay_score_weights,
    _sizing_signal_column,
    _trend_percentile_signal,
    _with_b1_trend_audit_columns,
)


def test_sizing_signal_column_defaults():
    assert _sizing_signal_column({}, sleeve_name="core_leveraged") == ("blended_decay", "net_edge_p50_annual")
    cfg = {"sizing_signal": "net_edge"}
    assert _sizing_signal_column(cfg, sleeve_name="core_leveraged")[1] == "net_edge_p50_annual"
    assert _sizing_signal_column(cfg, sleeve_name="inverse_decay_bucket4")[1] == "net_edge_p50_annual"
    cfg2 = {"sizing_signal": "net_edge", "sizing_edge_column": "primary_edge_annual"}
    assert _sizing_signal_column(cfg2, sleeve_name="inverse_decay_bucket4")[1] == "primary_edge_annual"


def test_default_decay_score_uses_net_edge_p50_then_explicit_blended_column():
    df = pd.DataFrame(
        {
            "delta_abs": [2.0, 2.0],
            "blended_gross_decay": [0.20, 0.10],
            "net_edge_p50_annual": [0.12, 0.18],
            "borrow_current": [0.02, 0.02],
        }
    )
    w_p50 = _decay_score_weights(
        df,
        {"sizing_signal": "blended_decay", "borrow_aversion": 3.0, "eq_blend": 0.0, "margin_efficiency_power": 0.0},
        sleeve_name="core_leveraged",
    )
    # p50 raw: 0.12-0.06 vs 0.18-0.06 -> more mass on row 1
    assert w_p50[1] > w_p50[0]
    w_blend = _decay_score_weights(
        df,
        {
            "sizing_signal": "blended_decay",
            "sizing_edge_column": "blended_gross_decay",
            "borrow_aversion": 3.0,
            "eq_blend": 0.0,
            "margin_efficiency_power": 0.0,
        },
        sleeve_name="core_leveraged",
    )
    # blended raw: 0.14 vs 0.04 -> more mass on row 0
    assert w_blend[0] > w_blend[1]
    np.testing.assert_allclose(w_p50.sum(), 1.0)
    np.testing.assert_allclose(w_blend.sum(), 1.0)


def test_net_edge_mode_matches_default_blended_when_both_resolve_to_p50():
    df = pd.DataFrame(
        {
            "delta_abs": [2.0, 2.0],
            "blended_gross_decay": [0.20, 0.10],
            "net_edge_p50_annual": [0.12, 0.18],
            "borrow_current": [0.02, 0.02],
        }
    )
    w_blended = _decay_score_weights(
        df,
        {"sizing_signal": "blended_decay", "borrow_aversion": 3.0, "eq_blend": 0.0, "margin_efficiency_power": 0.0},
        sleeve_name="core_leveraged",
    )
    w_edge = _decay_score_weights(
        df,
        {"sizing_signal": "net_edge", "borrow_aversion": 3.0, "eq_blend": 0.0, "margin_efficiency_power": 0.0},
        sleeve_name="core_leveraged",
    )
    np.testing.assert_allclose(w_blended, w_edge)


def test_net_edge_falls_back_when_column_missing():
    df = pd.DataFrame(
        {
            "delta_abs": [2.0],
            "blended_gross_decay": [0.15],
            "borrow_current": [0.01],
        }
    )
    w = _decay_score_weights(
        df,
        {"sizing_signal": "net_edge", "borrow_aversion": 2.0, "eq_blend": 0.0, "margin_efficiency_power": 0.0},
        sleeve_name="core_leveraged",
    )
    np.testing.assert_allclose(w, [1.0])


def _trend_cfg() -> dict:
    return {
        "trend_percentile_multiplier": {
            "enabled": True,
            "column": "und_trend_ratio_60d",
            "percentile_mode": "cross_sectional",
            "alpha": 1.0,
            "neutral_pctile": 0.5,
            "floor": 0.5,
            "ceiling": 1.5,
            "missing": "neutral",
        }
    }


def test_b1_trend_multiplier_uses_raw_cross_sectional_rank():
    df = pd.DataFrame(
        {
            "und_trend_ratio_60d": [0.70, 1.00, 1.30],
            "und_trend_ratio_60d_pctile": [1.0, 0.5, 0.0],
        }
    )
    pctile, mult = _trend_percentile_signal(df, _trend_cfg())
    np.testing.assert_allclose(pctile, [1.0 / 3.0, 2.0 / 3.0, 1.0])
    assert mult[0] > mult[1] > mult[2]


def test_b1_trend_multiplier_missing_raw_trend_is_neutral():
    df = pd.DataFrame({"und_trend_ratio_60d": [0.70, np.nan, 1.30]})
    pctile, mult = _trend_percentile_signal(df, _trend_cfg())
    assert pctile[1] == 0.5
    assert mult[1] == 1.0


def test_b1_trend_sizing_ignores_historical_pctile_column():
    df = pd.DataFrame(
        {
            "delta_abs": [2.0, 2.0],
            "net_edge_p50_annual": [0.20, 0.20],
            "borrow_current": [0.0, 0.0],
            "und_trend_ratio_60d": [0.75, 1.25],
            # Deliberately inverted: this historical diagnostic must not drive B1 sizing.
            "und_trend_ratio_60d_pctile": [1.0, 0.0],
        }
    )
    w = _decay_score_weights(
        df,
        {
            **_trend_cfg(),
            "sizing_signal": "net_edge",
            "borrow_aversion": 0.0,
            "eq_blend": 0.0,
            "margin_efficiency_power": 0.0,
            "max_name_weight": 1.0,
        },
        sleeve_name="core_leveraged",
    )
    assert w[0] > w[1]


def test_b1_trend_audit_columns_are_attached_for_core_frame():
    df = pd.DataFrame({"und_trend_ratio_60d": [0.75, 1.25]})
    out = _with_b1_trend_audit_columns(df, _trend_cfg())
    assert {"b1_trend_xsec_pctile", "b1_trend_multiplier"}.issubset(out.columns)
    assert out.loc[0, "b1_trend_multiplier"] > out.loc[1, "b1_trend_multiplier"]
