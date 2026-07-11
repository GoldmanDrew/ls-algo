"""Purgatory candidates surface as distinct proxy bars in the bucket plots."""

import pandas as pd

from plot_proposed_trades import append_purgatory_screened_rows


def _screened_b4_purgatory_row(**over):
    base = {
        "ETF": "IREZ",
        "Underlying": "IREN",
        "Delta": -2.0,
        "is_yieldboost": False,
        "inverse_shortable": True,
        "purgatory": True,
        "shares_available": 55000,  # has inventory -> NOT a no-ibkr proxy
        "borrow_current": 0.80,  # in keep band (0.7-0.9): the purgatory reason
        "net_edge_p50_annual": 2.50,
        "bucket4_net_edge_annual": 2.66,
        "vol_underlying_annual": 1.05,
    }
    base.update(over)
    return base


def test_purgatory_b4_candidate_added_with_flag():
    screened = pd.DataFrame([_screened_b4_purgatory_row()])
    # Inject a counterfactual size so the test does not call the live opt2 engine.
    size_map = {
        ("IREZ", "IREN"): {"gross": 566.0, "inv": 317.0, "und": 249.0, "h": 0.39, "scale": 0.5},
    }
    out = append_purgatory_screened_rows(
        pd.DataFrame(), screened, bucket="b4", b4_size_map=size_map
    )
    assert not out.empty
    row = out.loc[out["ETF"] == "IREZ"].iloc[0]
    assert bool(row["_purgatory_proxy"]) is True
    assert bool(row["_b4_counterfactual_size"]) is True
    assert row["sleeve"] == "inverse_decay_bucket4"
    assert abs(float(row["optimal_gross_target_usd"]) - 566.0) < 1e-6
    assert abs(float(row["optimal_short_usd"]) + 317.0) < 1e-6
    assert abs(float(row["optimal_long_usd"]) + 249.0) < 1e-6


def test_non_purgatory_row_not_added():
    screened = pd.DataFrame([_screened_b4_purgatory_row(purgatory=False)])
    out = append_purgatory_screened_rows(pd.DataFrame(), screened, bucket="b4")
    assert out.empty


def test_purgatory_borrow_relaxed_but_edge_vol_still_gated():
    # Edge below the B4 floor -> excluded even though purgatory + high borrow.
    screened = pd.DataFrame(
        [_screened_b4_purgatory_row(net_edge_p50_annual=-0.10, bucket4_net_edge_annual=-0.10)]
    )
    out = append_purgatory_screened_rows(pd.DataFrame(), screened, bucket="b4")
    assert out.empty


def test_purgatory_dedupes_against_existing_rows():
    existing = pd.DataFrame(
        [
            {
                "ETF": "IREZ",
                "Underlying": "IREN",
                "Delta": -2.0,
                "sleeve": "inverse_decay_bucket4",
                "long_usd": -100.0,
                "short_usd": -100.0,
                "optimal_long_usd": -100.0,
                "optimal_short_usd": -100.0,
                "optimal_gross_target_usd": 200.0,
            }
        ]
    )
    screened = pd.DataFrame([_screened_b4_purgatory_row()])
    out = append_purgatory_screened_rows(existing, screened, bucket="b4")
    assert (out["ETF"] == "IREZ").sum() == 1
