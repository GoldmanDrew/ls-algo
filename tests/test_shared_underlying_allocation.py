from __future__ import annotations

import numpy as np
import pandas as pd

from generate_trade_plan import (
    _hierarchical_shared_underlying_weights,
    apply_gross_sizing_book_caps,
)


def _weighting(**overrides) -> dict:
    return {
        "sizing_signal": "net_edge",
        "sizing_edge_column": "net_edge_p50_annual",
        "borrow_aversion": 0.25,
        "margin_efficiency_power": 0.0,
        "score_concavity_p": 1.0,
        "eq_blend": 0.0,
        "max_name_weight": 0.60,
        **overrides,
    }


def _alloc(**overrides) -> dict:
    return {
        "enabled": True,
        "utility_signal_column": "net_edge_p50_annual",
        "avoid_double_counting": True,
        "adverse_current_borrow_penalty": 1.0,
        "max_wrapper_weight": 0.60,
        **overrides,
    }


def _frame(rows: list[dict]) -> pd.DataFrame:
    defaults = {
        "bucket": "bucket_1",
        "product_class": "letf_long",
        "Delta": 2.0,
        "delta_abs": 2.0,
        "borrow_current": 0.10,
        "borrow_posterior_annual": 0.10,
        "shares_available": 1_000_000,
        "purgatory": False,
        "purgatory_no_locate": False,
        "exclude_no_shares": False,
        "borrow_missing_from_ftp": False,
        "strategy_blacklisted": False,
    }
    return pd.DataFrame([{**defaults, **row} for row in rows])


def _run(df: pd.DataFrame, *, pair_cap: float = 0.60, und_cap: float = 1.0):
    cfg = _weighting(max_name_weight=pair_cap)
    acfg = _alloc(max_wrapper_weight=pair_cap)
    return _hierarchical_shared_underlying_weights(
        df,
        np.ones(len(df)) / len(df),
        cfg,
        acfg,
        sleeve_name="core_leveraged",
        max_underlying_weight=und_cap,
    )


def test_duplicate_wrappers_do_not_inflate_underlying_weight():
    df = _frame(
        [
            {"ETF": "U1A", "Underlying": "U1", "net_edge_p50_annual": 0.40},
            {"ETF": "U1B", "Underlying": "U1", "net_edge_p50_annual": 0.40},
            {"ETF": "U2A", "Underlying": "U2", "net_edge_p50_annual": 0.40},
        ]
    )
    w, _ = _run(df)
    np.testing.assert_allclose(w[:2].sum(), 0.50, atol=1e-10)
    np.testing.assert_allclose(w[2], 0.50, atol=1e-10)


def test_best_wrapper_fills_first_then_pair_cap_overflows():
    df = _frame(
        [
            {"ETF": "CHEAP", "Underlying": "U1", "net_edge_p50_annual": 0.60},
            {"ETF": "RICH", "Underlying": "U1", "net_edge_p50_annual": 0.30},
            {"ETF": "OTHER", "Underlying": "U2", "net_edge_p50_annual": 0.10},
        ]
    )
    w, audit = _run(df, pair_cap=0.40, und_cap=0.70)
    assert w[0] == 0.40
    assert 0 < w[1] <= 0.30 + 1e-12
    assert audit.iloc[0]["shared_underlying_allocation_reason"] == "primary"
    assert audit.iloc[1]["shared_underlying_allocation_reason"] == "pair_cap_overflow"


def test_net_edge_is_not_charged_full_borrow_twice():
    df = _frame(
        [
            {
                "ETF": "LOW",
                "Underlying": "U1",
                "net_edge_p50_annual": 0.40,
                "borrow_current": 0.10,
                "borrow_posterior_annual": 0.10,
            },
            {
                "ETF": "HIGH",
                "Underlying": "U1",
                "net_edge_p50_annual": 0.42,
                "borrow_current": 0.30,
                "borrow_posterior_annual": 0.30,
            },
            {"ETF": "OTHER", "Underlying": "U2", "net_edge_p50_annual": 0.20},
        ]
    )
    _, audit = _run(df)
    # Both current rates equal their posterior expectations, so authoritative net edge wins.
    assert audit.iloc[1]["wrapper_utility"] > audit.iloc[0]["wrapper_utility"]


def test_adverse_current_borrow_shock_penalizes_wrapper():
    df = _frame(
        [
            {
                "ETF": "STABLE",
                "Underlying": "U1",
                "net_edge_p50_annual": 0.40,
                "borrow_current": 0.10,
                "borrow_posterior_annual": 0.10,
            },
            {
                "ETF": "SPIKE",
                "Underlying": "U1",
                "net_edge_p50_annual": 0.42,
                "borrow_current": 0.30,
                "borrow_posterior_annual": 0.10,
            },
            {"ETF": "OTHER", "Underlying": "U2", "net_edge_p50_annual": 0.20},
        ]
    )
    _, audit = _run(df)
    assert audit.iloc[0]["wrapper_utility"] > audit.iloc[1]["wrapper_utility"]


def test_no_locate_wrapper_cannot_receive_new_allocation():
    df = _frame(
        [
            {
                "ETF": "NOLOC",
                "Underlying": "U1",
                "net_edge_p50_annual": 1.00,
                "purgatory_no_locate": True,
                "exclude_no_shares": True,
                "shares_available": 0,
            },
            {"ETF": "OK", "Underlying": "U1", "net_edge_p50_annual": 0.30},
            {"ETF": "OTHER", "Underlying": "U2", "net_edge_p50_annual": 0.20},
        ]
    )
    w, audit = _run(df)
    assert w[0] == 0.0
    assert np.isneginf(audit.iloc[0]["wrapper_utility"])
    assert w[1] > 0


def test_bucket_direction_and_product_class_are_separate_groups():
    df = _frame(
        [
            {"ETF": "B1", "Underlying": "U1", "net_edge_p50_annual": 0.30},
            {
                "ETF": "B4",
                "Underlying": "U1",
                "bucket": "bucket_4",
                "product_class": "inverse",
                "Delta": -2.0,
                "net_edge_p50_annual": 0.30,
            },
            {
                "ETF": "YB",
                "Underlying": "U1",
                "bucket": "bucket_2",
                "product_class": "yieldboost",
                "net_edge_p50_annual": 0.30,
            },
        ]
    )
    _, audit = _run(df)
    assert set(audit["shared_underlying_group_size"]) == {1}


def test_allocation_is_deterministic_under_row_reordering():
    df = _frame(
        [
            {"ETF": "B", "Underlying": "U1", "net_edge_p50_annual": 0.40},
            {"ETF": "A", "Underlying": "U1", "net_edge_p50_annual": 0.40},
            {"ETF": "C", "Underlying": "U2", "net_edge_p50_annual": 0.20},
        ]
    )
    w1, _ = _run(df)
    shuffled = df.iloc[[2, 0, 1]].reset_index(drop=True)
    w2, _ = _run(shuffled)
    got1 = dict(zip(df["ETF"], w1))
    got2 = dict(zip(shuffled["ETF"], w2))
    assert got1 == got2
    assert got1["A"] > got1["B"]  # alphabetic final tie-break


def test_incumbent_hysteresis_blocks_small_challenger_advantage():
    df = _frame(
        [
            {"ETF": "OLD", "Underlying": "U1", "net_edge_p50_annual": 0.40},
            {"ETF": "NEW", "Underlying": "U1", "net_edge_p50_annual": 0.42},
            {"ETF": "OTHER", "Underlying": "U2", "net_edge_p50_annual": 0.20},
        ]
    )
    weighting = _weighting(max_name_weight=1.0)
    alloc = _alloc(
        max_wrapper_weight=1.0,
        switching={"min_utility_advantage_annual": 0.03},
    )
    w, audit = _hierarchical_shared_underlying_weights(
        df,
        np.ones(len(df)) / len(df),
        weighting,
        alloc,
        sleeve_name="core_leveraged",
        max_underlying_weight=1.0,
        incumbent_by_underlying={"U1": "OLD"},
    )
    assert w[0] > 0
    assert w[1] == 0
    assert bool(audit.iloc[0]["shared_underlying_switch_blocked"])


def test_invalid_incumbent_is_bypassed_immediately():
    df = _frame(
        [
            {
                "ETF": "OLD",
                "Underlying": "U1",
                "net_edge_p50_annual": 1.00,
                "purgatory_no_locate": True,
            },
            {"ETF": "NEW", "Underlying": "U1", "net_edge_p50_annual": 0.30},
            {"ETF": "OTHER", "Underlying": "U2", "net_edge_p50_annual": 0.20},
        ]
    )
    weighting = _weighting(max_name_weight=1.0)
    alloc = _alloc(
        max_wrapper_weight=1.0,
        switching={"min_utility_advantage_annual": 1.00},
    )
    w, _ = _hierarchical_shared_underlying_weights(
        df,
        np.ones(len(df)) / len(df),
        weighting,
        alloc,
        sleeve_name="core_leveraged",
        max_underlying_weight=1.0,
        incumbent_by_underlying={"U1": "OLD"},
    )
    assert w[0] == 0
    assert w[1] > 0


def test_missing_borrow_column_fails_closed_without_crashing():
    df = _frame(
        [{"ETF": "A", "Underlying": "U1", "net_edge_p50_annual": 0.40}]
    ).drop(columns=["borrow_current", "borrow_posterior_annual"])
    w, audit = _run(df)
    assert w.sum() == 0
    assert np.isneginf(audit.iloc[0]["wrapper_utility"])


def _cap_strategy() -> dict:
    return {
        "gross_sizing_caps": {
            "enabled": True,
            "liquidity_book_reference": "deployed_book",
            "max_pair_weight_cap": 0.99,
            "max_underlying_weight_cap": 1.0,
            "aum_use_pct": 0.0,
            "short_avail_use_pct": 1.0,
            "missing_shares_cap": 0.99,
            "shares_outstanding_use_frac": 0.0,
            "median_daily_volume_use_pct": 0.0,
            "per_sleeve": {
                "core_leveraged": {
                    "max_pair_weight": 0.80,
                    "max_underlying_weight": 1.0,
                    "pre_cap_score_haircut_multiplier": 1.0,
                    "shared_underlying_allocation": {
                        "enabled": True,
                        "max_wrapper_weight": 0.80,
                    },
                }
            },
        }
    }


def _sized_for_caps() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ETF": ["BEST", "BACKUP"],
            "Underlying": ["U1", "U1"],
            "bucket": ["bucket_1", "bucket_1"],
            "product_class": ["letf_long", "letf_long"],
            "Delta": [2.0, 2.0],
            "delta_abs": [2.0, 2.0],
            "sleeve": ["core_leveraged", "core_leveraged"],
            "gross_target_usd": [500.0, 500.0],
            "_pre_cap_score_weight": [0.5, 0.5],
            "wrapper_utility": [0.50, 0.30],
            "borrow_current": [0.05, 0.10],
            "borrow_price_ref": [100.0, 100.0],
            "shares_available": [1.0, 1_000_000.0],
        }
    )


def test_executable_waterfill_uses_day_locate_then_overflows():
    out, diag = apply_gross_sizing_book_caps(
        _sized_for_caps(),
        target_gross_usd=1000.0,
        delta_floor=0.1,
        strategy=_cap_strategy(),
        shares_out_map={},
        cap_mode="structural_plus_day_liquidity",
    )
    # A 2x stock pair has one-third ETF short. One share at $100 supports $300 pair gross.
    np.testing.assert_allclose(out["gross_target_usd"], [300.0, 700.0], atol=1e-6)
    assert diag["shared_underlying_allocation"]["applied"] is True


def test_structural_target_ignores_day_locate_and_fills_primary_to_pair_cap():
    out, _ = apply_gross_sizing_book_caps(
        _sized_for_caps(),
        target_gross_usd=1000.0,
        delta_floor=0.1,
        strategy=_cap_strategy(),
        shares_out_map={},
        cap_mode="structural_only",
    )
    np.testing.assert_allclose(out["gross_target_usd"], [800.0, 200.0], atol=1e-6)
