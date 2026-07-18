"""Regression tests for production actual backtest audit fixes."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from scripts import production_actual_backtest
from scripts.gtp_prod_sizing import held_from_plan, remap_cfg_state_paths
from scripts.production_actual_backtest import (
    B5_SLEEVE,
    _b5_sleeve_nav,
    _pair_stats_from_navs,
    _stock_sleeve_nav,
    _targets_from_plan,
    normalize_plan,
    prepare_screened_for_gtp_approx,
    prod_replay_plan_timeline,
    simulate_book_from_plan_timeline,
)
from scripts.sizing_tilt_cadence_bt import load_price_panel, pair_daily_returns
from strategy_config import load_config


def test_same_run_churn_rollback_flows_into_backtest_knobs():
    cfg = {
        "portfolio": {
            "rebalance": {
                "same_run_churn": {"enabled": False},
            }
        }
    }
    assert (
        production_actual_backtest.rebalance_knobs(cfg)[
            "same_run_churn_enabled"
        ]
        is False
    )


def test_prepare_screened_for_prod_replay_shims_net_decay():
    from scripts.production_actual_backtest import prepare_screened_for_prod_replay

    df = pd.DataFrame(
        {
            "ETF": ["AAA", "BBB"],
            "Underlying": ["X", "Y"],
            "Beta": [1.0, -1.0],
            "net_decay_annual": [0.10, 0.20],
            "purgatory": [False, False],
        }
    )
    out, diag = prepare_screened_for_prod_replay(df)
    assert "Delta" in out.columns
    assert diag["edge_source"] == "net_decay_annual"
    assert diag["n_edge_fallback"] == 2
    assert float(out.loc[0, "net_edge_p50_annual"]) == pytest.approx(0.10)
    assert float(out.loc[1, "net_edge_p50_annual"]) == pytest.approx(0.20)


def test_v6_thin_book_skips_without_raise(tmp_path):
    """Historical thin B4 books must not abort the whole trade plan."""
    from scripts.v6_b4_pf_weights import V6PfParams, compute_v6_b4_pf_weight_dict

    idx = pd.bdate_range("2024-01-01", periods=10)
    prices = pd.DataFrame({"a": 10.0, "b": 100.0}, index=idx)
    pair_cache = {
        ("AAA", "BBB"): {
            "prices": prices,
            "kw": {"borrow_a_annual": 0.05},
        }
    }
    screened = tmp_path / "screened.csv"
    pd.DataFrame(
        {"ETF": ["AAA"], "Underlying": ["BBB"], "net_decay_annual": [0.5]}
    ).to_csv(screened, index=False)
    w, _df, meta = compute_v6_b4_pf_weight_dict(
        pair_cache=pair_cache,
        v6_opt2_h_daily_map={"BBB": pd.Series(1.0, index=idx)},
        screened_csv=str(screened),
        closes_broad=None,
        norm_sym=lambda x: str(x).strip().upper(),
        get_ibkr_borrow_map=lambda _syms: {},
        opt2_h_base=1.0,
        params=V6PfParams(min_pairs=2),
        use_ibkr_uvix_borrow=False,
    )
    assert w == {}
    assert meta.get("skipped_thin_book") is True
    assert int(meta.get("n_pairs_live", -1)) == 1


def test_normalize_plan_marks_purgatory_reduce_only():
    raw = pd.DataFrame(
        [
            {
                "ETF": "AAA",
                "Underlying": "BBB",
                "sleeve": "core_leveraged",
                "long_usd": 300.0,
                "short_usd": -100.0,
                "gross_target_usd": 400.0,
                "purgatory": False,
            },
            {
                "ETF": "CCC",
                "Underlying": "DDD",
                "sleeve": "core_leveraged",
                "long_usd": 0.0,
                "short_usd": 0.0,
                "gross_target_usd": 0.0,
                "purgatory": True,
            },
        ]
    )
    plan = normalize_plan(raw, source_date="2026-02-27")
    assert set(plan["ETF"]) == {"AAA", "CCC"}
    ccc = plan[plan["ETF"] == "CCC"].iloc[0]
    assert bool(ccc["reduce_only"]) is True
    assert bool(ccc["keep_open"]) is False
    assert float(ccc["gross_target_usd"]) == pytest.approx(0.0)


def test_normalize_plan_infers_yieldboost_from_bucket_when_sleeve_blank():
    """Purgatory keep rows often have sleeve=NaN but bucket=bucket_2 (AMYY/MUYY)."""
    raw = pd.DataFrame(
        [
            {
                "ETF": "AMYY",
                "Underlying": "AMD",
                "sleeve": np.nan,
                "bucket": "bucket_2",
                "product_class": "income_yieldboost",
                "Delta": 0.85,
                "long_usd": 0.0,
                "short_usd": 0.0,
                "gross_target_usd": 0.0,
                "purgatory": True,
            },
            {
                "ETF": "MUYY",
                "Underlying": "MU",
                "sleeve": "",
                "bucket": "bucket_2",
                "Delta": 0.9,
                "long_usd": 0.0,
                "short_usd": 0.0,
                "gross_target_usd": 0.0,
                "purgatory": True,
            },
            {
                "ETF": "NVDL",
                "Underlying": "NVDA",
                "sleeve": np.nan,
                "bucket": "bucket_1",
                "Delta": 1.5,
                "long_usd": 100.0,
                "short_usd": -50.0,
                "gross_target_usd": 150.0,
            },
        ]
    )
    plan = normalize_plan(raw, source_date="2026-07-13")
    by_etf = plan.set_index("ETF")["sleeve"].to_dict()
    assert by_etf["AMYY"] == "yieldboost"
    assert by_etf["MUYY"] == "yieldboost"
    assert by_etf["NVDL"] == "core_leveraged"


def test_normalize_plan_overrides_wrong_core_stamp_for_yieldboost():
    """Cached plans may already say core_leveraged while bucket=bucket_2."""
    raw = pd.DataFrame(
        [
            {
                "ETF": "AMYY",
                "Underlying": "AMD",
                "sleeve": "core_leveraged",
                "bucket": "bucket_2",
                "product_class": "income_yieldboost",
                "is_yieldboost": True,
                "Delta": 0.24,
                "long_usd": 0.0,
                "short_usd": 0.0,
                "gross_target_usd": 0.0,
                "purgatory": True,
            },
            {
                "ETF": "SQQQ",
                "Underlying": "QQQ",
                "sleeve": "core_leveraged",  # wrong, but delta<0 + no yb evidence → B4 via delta
                "bucket": "bucket_4",
                "Delta": -1.0,
                "long_usd": 50.0,
                "short_usd": -50.0,
                "gross_target_usd": 100.0,
            },
        ]
    )
    plan = normalize_plan(raw, source_date="2026-07-13")
    by_etf = plan.set_index("ETF")["sleeve"].to_dict()
    assert by_etf["AMYY"] == "yieldboost"
    assert by_etf["SQQQ"] == "inverse_decay_bucket4"


def test_normalize_plan_honors_hold_rollback(monkeypatch):
    monkeypatch.setattr(
        production_actual_backtest,
        "load_config",
        lambda: {"portfolio": {"rebalance": {"purgatory_execution": "hold"}}},
    )
    raw = pd.DataFrame([
        {
            "ETF": "CCC",
            "Underlying": "C",
            "sleeve": "core_leveraged",
            "purgatory": True,
            "long_usd": 0.0,
            "short_usd": 0.0,
        }
    ])
    plan = normalize_plan(raw, source_date="2026-02-27")
    assert bool(plan.iloc[0]["keep_open"]) is True
    assert bool(plan.iloc[0]["reduce_only"]) is False


def test_purgatory_zero_model_target_share_holds():
    """Executable 0 + missing/zero model must not flatten (live trim/hold contract)."""
    cal = pd.bdate_range("2025-01-01", periods=30)
    panel = {"AAA": pd.DataFrame({"a_px": 100.0, "b_px": 50.0}, index=cal)}
    p0 = normalize_plan(
        pd.DataFrame(
            [
                {
                    "ETF": "AAA",
                    "Underlying": "BBB",
                    "sleeve": "core_leveraged",
                    "long_usd": 200.0,
                    "short_usd": -200.0,
                    "gross_target_usd": 400.0,
                    "purgatory": False,
                }
            ]
        ),
        source_date=str(cal[0].date()),
    )
    p1 = normalize_plan(
        pd.DataFrame(
            [
                {
                    "ETF": "AAA",
                    "Underlying": "BBB",
                    "sleeve": "core_leveraged",
                    "long_usd": 0.0,
                    "short_usd": 0.0,
                    "gross_target_usd": 0.0,
                    "purgatory": True,
                }
            ]
        ),
        source_date=str(cal[10].date()),
    )
    _, _, meta, _, daily = simulate_book_from_plan_timeline(
        {cal[0]: p0, cal[10]: p1},
        panel,
        budgets={"core_leveraged": 400.0},
        capital_usd=100.0,
        start=cal[0],
        slippage_bps=0.0,
        commission_per_share=0.0,
        margin_rate_annual=0.0,
        min_trade_usd=0.0,
        scale_sleeves_to_budget=False,
        retarget_on_plan_change=True,
        purgatory_model_zero_policy="hold",
    )
    after = daily[daily["date"] > cal[10]]
    assert len(after) > 0
    assert float(after["n_positions"].iloc[-1]) == 1
    assert meta.get("purgatory_model_zero_policy") == "hold"


def test_purgatory_lower_model_trims_not_adds():
    cal = pd.bdate_range("2025-01-01", periods=30)
    panel = {"AAA": pd.DataFrame({"a_px": 100.0, "b_px": 50.0}, index=cal)}
    p0 = normalize_plan(
        pd.DataFrame(
            [
                {
                    "ETF": "AAA",
                    "Underlying": "BBB",
                    "sleeve": "core_leveraged",
                    "long_usd": 400.0,
                    "short_usd": -400.0,
                    "gross_target_usd": 800.0,
                    "purgatory": False,
                }
            ]
        ),
        source_date=str(cal[0].date()),
    )
    p1 = normalize_plan(
        pd.DataFrame(
            [
                {
                    "ETF": "AAA",
                    "Underlying": "BBB",
                    "sleeve": "core_leveraged",
                    "long_usd": 0.0,
                    "short_usd": 0.0,
                    "gross_target_usd": 0.0,
                    "purgatory": True,
                    "execution_policy": "reduce_only",
                    "model_long_usd": 100.0,
                    "model_short_usd": -100.0,
                    "model_gross_target_usd": 200.0,
                }
            ]
        ),
        source_date=str(cal[10].date()),
    )
    _, audit, _, _, daily = simulate_book_from_plan_timeline(
        {cal[0]: p0, cal[10]: p1},
        panel,
        budgets={"core_leveraged": 800.0},
        capital_usd=200.0,
        start=cal[0],
        slippage_bps=0.0,
        commission_per_share=0.0,
        margin_rate_annual=0.0,
        min_trade_usd=0.0,
        scale_sleeves_to_budget=False,
        retarget_on_plan_change=True,
        use_resize_bands=False,
    )
    after = daily[daily["date"] > cal[10]]
    assert float(after["n_positions"].iloc[-1]) == 1
    # Gross should have been reduced (pair still open).
    changed = audit[audit["plan_date"] == str(cal[10].date())]
    assert len(changed) > 0
    assert float(changed.iloc[0]["turnover_usd"]) > 0.0


def test_purgatory_zero_model_exit_policy_legacy():
    """A/B: purgatory_model_zero_policy=exit restores flatten-on-zero-model."""
    cal = pd.bdate_range("2025-01-01", periods=30)
    panel = {"AAA": pd.DataFrame({"a_px": 100.0, "b_px": 50.0}, index=cal)}
    p0 = normalize_plan(
        pd.DataFrame(
            [
                {
                    "ETF": "AAA",
                    "Underlying": "BBB",
                    "sleeve": "core_leveraged",
                    "long_usd": 200.0,
                    "short_usd": -200.0,
                    "gross_target_usd": 400.0,
                    "purgatory": False,
                }
            ]
        ),
        source_date=str(cal[0].date()),
    )
    p1 = normalize_plan(
        pd.DataFrame(
            [
                {
                    "ETF": "AAA",
                    "Underlying": "BBB",
                    "sleeve": "core_leveraged",
                    "long_usd": 0.0,
                    "short_usd": 0.0,
                    "gross_target_usd": 0.0,
                    "purgatory": True,
                    "model_gross_target_usd": 0.0,
                    "model_long_usd": 0.0,
                    "model_short_usd": 0.0,
                }
            ]
        ),
        source_date=str(cal[10].date()),
    )
    _, _, _, _, daily = simulate_book_from_plan_timeline(
        {cal[0]: p0, cal[10]: p1},
        panel,
        budgets={"core_leveraged": 400.0},
        capital_usd=100.0,
        start=cal[0],
        slippage_bps=0.0,
        commission_per_share=0.0,
        margin_rate_annual=0.0,
        min_trade_usd=0.0,
        scale_sleeves_to_budget=False,
        retarget_on_plan_change=True,
        purgatory_model_zero_policy="exit",
    )
    after = daily[daily["date"] > cal[10]]
    assert float(after["n_positions"].iloc[-1]) == 0


def test_normalize_plan_backfills_model_from_optimal():
    raw = pd.DataFrame(
        [
            {
                "ETF": "AAA",
                "Underlying": "BBB",
                "sleeve": "core_leveraged",
                "long_usd": 0.0,
                "short_usd": 0.0,
                "gross_target_usd": 0.0,
                "purgatory": True,
                "optimal_long_usd": -150.0,
                "optimal_short_usd": -150.0,
                "optimal_gross_target_usd": 300.0,
            }
        ]
    )
    plan = normalize_plan(raw, source_date="2026-02-27")
    assert float(plan.iloc[0]["model_gross_target_usd"]) == pytest.approx(300.0)
    assert float(plan.iloc[0]["model_long_usd"]) == pytest.approx(-150.0)
    assert bool(plan.iloc[0]["ratchet_released"]) is False


def test_membership_day_set_operator_5d():
    cal = pd.bdate_range("2026-01-02", periods=12)
    days = production_actual_backtest._membership_day_set(
        cal, mode="operator_5d", check_days=5
    )
    assert cal[0] in days
    assert cal[5] in days
    assert cal[1] not in days


def test_b4_ratchet_cover_guard_pins_when_not_released():
    etf, und, rsn = production_actual_backtest._apply_b4_ratchet_cover_guard(
        -1000.0,
        -400.0,
        -500.0,
        plan_row={"ratchet_released": False, "ratchet_trim_usd": 0.0},
        allow_inverse_cover=True,
        h=0.5,
        beta_abs=2.0,
    )
    assert rsn == "pin"
    assert float(etf) == pytest.approx(-1000.0)


def test_b4_ratchet_cover_guard_caps_trim():
    etf, und, rsn = production_actual_backtest._apply_b4_ratchet_cover_guard(
        -1000.0,
        -400.0,
        -200.0,
        plan_row={"ratchet_released": True, "ratchet_trim_usd": 200.0},
        allow_inverse_cover=True,
        h=0.5,
        beta_abs=2.0,
    )
    assert rsn == "trim_cap"
    assert float(etf) == pytest.approx(-800.0)


def test_notebook_b4_borrow_overrides_shift_ramp():
    from scripts.production_actual_backtest import apply_notebook_b4_borrow_overrides
    import copy
    from strategy_config import load_config

    cfg = copy.deepcopy(load_config())
    audit = apply_notebook_b4_borrow_overrides(
        cfg,
        entry_borrow_cap=0.60,
        keep_borrow_cap=0.80,
        shift_ramp_with_band=True,
    )
    b4 = cfg["screener"]["per_bucket"]["bucket_4"]
    assert float(b4["entry_borrow_cap"]) == pytest.approx(0.60)
    assert float(b4["keep_borrow_cap"]) == pytest.approx(0.80)
    opt2 = (
        cfg["portfolio"]["sleeves"]["inverse_decay_bucket4"]["rules"]["bucket4_weekly_opt2"]
    )
    assert float(opt2["borrow_ramp_lo"]) == pytest.approx(0.70)
    assert float(opt2["borrow_ramp_hi"]) == pytest.approx(1.10)
    assert audit["borrow_ramp_lo"]["old"] == pytest.approx(0.80)


def test_b4_empty_plan_holds_instead_of_wipe():
    """Zero-size B4 plan must not liquidate held pairs on an operator day."""
    cal = pd.bdate_range("2025-01-01", periods=40)
    panel = {
        "QBTZ": pd.DataFrame({"a_px": 10.0, "b_px": 20.0}, index=cal),
        "DRIP": pd.DataFrame({"a_px": 10.0, "b_px": 20.0}, index=cal),
    }
    p0 = normalize_plan(
        pd.DataFrame(
            [
                {
                    "ETF": "QBTZ",
                    "Underlying": "QBTS",
                    "sleeve": "inverse_decay_bucket4",
                    "Delta": -2.0,
                    "long_usd": -500.0,
                    "short_usd": -500.0,
                    "gross_target_usd": 1000.0,
                    "purgatory": False,
                }
            ]
        ),
        source_date=str(cal[0].date()),
    )
    # After the book is live, replace with a zero-size B4 plan (archive-gap style).
    p_empty = normalize_plan(
        pd.DataFrame(
            [
                {
                    "ETF": "DRIP",
                    "Underlying": "XOP",
                    "sleeve": "inverse_decay_bucket4",
                    "Delta": -2.0,
                    "long_usd": 0.0,
                    "short_usd": 0.0,
                    "gross_target_usd": 0.0,
                    "purgatory": True,
                }
            ]
        ),
        source_date=str(cal[6].date()),
    )
    _, _, meta, pair_stats, _ = simulate_book_from_plan_timeline(
        {cal[0]: p0, cal[6]: p_empty},
        panel,
        budgets={"inverse_decay_bucket4": 1000.0},
        capital_usd=500.0,
        start=cal[0],
        slippage_bps=0.0,
        commission_per_share=0.0,
        margin_rate_annual=0.0,
        min_trade_usd=0.0,
        scale_sleeves_to_budget=False,
        retarget_on_plan_change=True,
        b4_execution="cadence",
        b4_membership_clock="operator_5d",
        operator_check_days=5,
        b4_empty_plan_policy="hold",
        use_resize_bands=False,
    )
    daily = meta.get("pair_daily")
    assert isinstance(daily, pd.DataFrame) and not daily.empty
    q = daily[daily["ETF"].astype(str).str.upper() == "QBTZ"].copy()
    q["gross"] = q["etf_usd"].abs() + q["underlying_usd"].abs()
    # Must remain open after the empty plan arrives (incl. later operator days).
    after = q[q["date"] >= cal[8]]
    assert len(after) > 0
    assert float(after["gross"].min()) > 1.0
    assert int(meta.get("n_b4_empty_plan_holds", 0)) >= 1
    assert not pair_stats.empty


def test_b4_membership_clock_defers_true_drop():
    """Pair dropped from plan mid-window stays open until operator day."""
    cal = pd.bdate_range("2025-01-01", periods=40)
    # Constant prices so marks do not create phantom turnover.
    panel = {
        "QBTZ": pd.DataFrame({"a_px": 10.0, "b_px": 20.0}, index=cal),
    }
    p0 = normalize_plan(
        pd.DataFrame(
            [
                {
                    "ETF": "QBTZ",
                    "Underlying": "QBTS",
                    "sleeve": "inverse_decay_bucket4",
                    "Delta": -2.0,
                    "long_usd": -500.0,
                    "short_usd": -500.0,
                    "gross_target_usd": 1000.0,
                    "purgatory": False,
                }
            ]
        ),
        source_date=str(cal[0].date()),
    )
    # Day 2 plan drops the pair entirely.
    p_drop = normalize_plan(
        pd.DataFrame(
            [
                {
                    "ETF": "OTHER",
                    "Underlying": "X",
                    "sleeve": "core_leveraged",
                    "Delta": 2.0,
                    "long_usd": 100.0,
                    "short_usd": -100.0,
                    "gross_target_usd": 200.0,
                    "purgatory": False,
                }
            ]
        ),
        source_date=str(cal[2].date()),
    )
    # Need OTHER in panel too so book can trade it on Fridays.
    panel["OTHER"] = pd.DataFrame({"a_px": 10.0, "b_px": 20.0}, index=cal)
    _, _, meta, _, daily = simulate_book_from_plan_timeline(
        {cal[0]: p0, cal[2]: p_drop},
        panel,
        budgets={"inverse_decay_bucket4": 1000.0, "core_leveraged": 200.0},
        capital_usd=500.0,
        start=cal[0],
        slippage_bps=0.0,
        commission_per_share=0.0,
        margin_rate_annual=0.0,
        min_trade_usd=0.0,
        scale_sleeves_to_budget=False,
        retarget_on_plan_change=True,
        b4_execution="cadence",
        b4_membership_clock="operator_5d",
        operator_check_days=5,
        b4_apply_resize_bands=True,
        b4_ratchet_execution_guard=True,
        use_resize_bands=False,
    )
    # Immediately after drop plan becomes effective (lag 1 → cal[3]), QBTZ still held.
    mid = daily[(daily["date"] > cal[3]) & (daily["date"] < cal[5])]
    assert len(mid) > 0
    # Book still has at least the deferred B4 name until next operator day.
    assert int(meta.get("n_b4_membership_deferred", 0)) >= 1


def test_prepare_screened_uses_borrow_avg_when_finite():
    df = pd.DataFrame(
        {
            "ETF": ["AAA", "BBB", "CCC"],
            "borrow_current": [0.10, 0.20, 0.30],
            "borrow_avg_annual": [0.05, np.nan, 0.40],
            "net_edge_p50_annual": [0.5, 0.6, 0.7],
        }
    )
    out = prepare_screened_for_gtp_approx(df)
    assert float(out.loc[0, "borrow_current"]) == pytest.approx(0.05)
    assert float(out.loc[1, "borrow_current"]) == pytest.approx(0.20)
    assert float(out.loc[2, "borrow_current"]) == pytest.approx(0.40)
    assert list(out["borrow_used_for_sizing"]) == [
        "borrow_avg_annual",
        "borrow_current",
        "borrow_avg_annual",
    ]
    assert float(out.loc[0, "net_edge_p50_annual"]) == pytest.approx(0.5)


def test_remap_cfg_state_paths_isolates_under_root(tmp_path: Path):
    cfg = load_config()
    remapped = remap_cfg_state_paths(cfg, tmp_path)
    paths = remapped["paths"]
    assert Path(paths["core_leveraged_decay_state_json"]).parent == tmp_path
    assert Path(paths["proposed_trades_csv"]).parent == tmp_path
    assert "data/core_leveraged_decay_state.json" not in paths["core_leveraged_decay_state_json"].replace(
        "\\", "/"
    )
    rules = remapped["portfolio"]["sleeves"]["inverse_decay_bucket4"]["rules"]
    opt2 = rules["bucket4_weekly_opt2"]
    assert Path(opt2["crash_budget"]["l_state_json"]).parent == tmp_path
    assert Path(opt2["weight_smoothing"]["state_json"]).parent == tmp_path
    assert Path(rules["ratchet"]["state_json"]).parent == tmp_path


def test_held_from_plan_b4_shorts():
    plan = pd.DataFrame(
        [
            {
                "ETF": "SOXS",
                "Underlying": "SOXX",
                "sleeve": "inverse_decay_bucket4",
                "etf_target_usd": -5000.0,
                "underlying_target_usd": 2000.0,
                "gross_target_usd": 7000.0,
            },
            {
                "ETF": "TQQQ",
                "Underlying": "QQQ",
                "sleeve": "core_leveraged",
                "etf_target_usd": -1000.0,
                "underlying_target_usd": 2000.0,
                "gross_target_usd": 3000.0,
            },
        ]
    )
    held = held_from_plan(plan)
    assert ("SOXS", "SOXX") in held
    assert ("TQQQ", "QQQ") not in held
    assert held[("SOXS", "SOXX")]["inverse_etf_short_usd"] == pytest.approx(5000.0)
    assert held[("SOXS", "SOXX")]["underlying_short_usd"] == pytest.approx(2000.0)


def test_prod_replay_timeline_carries_held_and_keeps_state(tmp_path: Path):
    """Mocked two-day loop: held from day1 fed into day2; state_root untouched live."""
    cfg = load_config()
    d0 = pd.Timestamp("2026-01-05")
    d1 = pd.Timestamp("2026-01-06")
    live_crash = Path("data/b4_crash_l_state.json")
    mtime0 = live_crash.stat().st_mtime if live_crash.exists() else None

    plan1 = pd.DataFrame(
        [
            {
                "ETF": "SOXS",
                "Underlying": "SOXX",
                "sleeve": "inverse_decay_bucket4",
                "long_usd": 2000.0,
                "short_usd": -5000.0,
                "etf_target_usd": -5000.0,
                "underlying_target_usd": 2000.0,
                "gross_target_usd": 7000.0,
                "borrow_current": 0.05,
            }
        ]
    )
    plan2 = plan1.copy()
    plan2["gross_target_usd"] = 7200.0
    calls: list[dict] = []

    def _fake_size(
        screened,
        run_date,
        cfg_in,
        *,
        state_root,
        held_inverse_short_by_pair=None,
        quiet=True,
    ):
        calls.append(
            {
                "run_date": run_date,
                "state_root": Path(state_root),
                "n_held": len(held_inverse_short_by_pair or {}),
                "held": dict(held_inverse_short_by_pair or {}),
            }
        )
        root = Path(state_root)
        (root / "b4_crash_l_state.json").write_text(
            json.dumps({"stage": "crash_l", "weight_by_pair": {f"day_{run_date}": 1.0}}),
            encoding="utf-8",
        )
        (root / "b4_weight_ema_state.json").write_text(
            json.dumps({"stage": "post_cap_scale", "weight_by_pair": {run_date: 0.5}}),
            encoding="utf-8",
        )
        (root / "b4_inverse_ratchet_state.json").write_text(
            json.dumps({"inverse_short_usd_by_pair": {"SOXS|SOXX": 5000.0 + len(calls)}}),
            encoding="utf-8",
        )
        return (plan1 if run_date.endswith("05") else plan2), {
            "n_held_pairs": len(held_inverse_short_by_pair or {})
        }

    import scripts.gtp_prod_sizing as gps

    with patch(
        "scripts.production_actual_backtest.list_archived_screened_dates",
        return_value=[d0, d1],
    ), patch(
        "pandas.read_csv",
        return_value=pd.DataFrame({"ETF": ["SOXS"], "Underlying": ["SOXX"], "Beta": [1.0]}),
    ), patch.object(gps, "size_book_from_screened", side_effect=_fake_size):
        timeline, diag = prod_replay_plan_timeline(
            cfg=cfg,
            start=d0,
            end=d1,
            state_root=tmp_path,
            keep_state=True,
        )

    assert len(timeline) == 2
    assert len(calls) == 2
    assert calls[0]["n_held"] == 0
    assert calls[1]["n_held"] == 1
    assert ("SOXS", "SOXX") in calls[1]["held"]
    assert calls[0]["state_root"] == tmp_path
    assert (tmp_path / "b4_crash_l_state.json").exists()
    assert (tmp_path / "b4_weight_ema_state.json").exists()
    assert (tmp_path / "b4_inverse_ratchet_state.json").exists()
    crash = json.loads((tmp_path / "b4_crash_l_state.json").read_text(encoding="utf-8"))
    assert "day_2026-01-06" in crash["weight_by_pair"]
    if mtime0 is not None:
        assert live_crash.stat().st_mtime == mtime0
    assert float(diag.loc[diag["ok"], "gross_b4"].max()) < 50_000.0


def test_size_book_isolated_no_live_write_and_finite_b4(tmp_path: Path):
    """Integration: one archived day through full GTP into isolated state."""
    screened_path = Path("data/runs/2026-07-10/etf_screened_today.csv")
    if not screened_path.is_file():
        pytest.skip("archived screened CSV missing")
    from scripts.gtp_prod_sizing import size_book_from_screened

    cfg = load_config()
    live_files = [
        Path("data/b4_crash_l_state.json"),
        Path("data/b4_weight_ema_state.json"),
        Path("data/b4_inverse_ratchet_state.json"),
        Path("data/core_leveraged_decay_state.json"),
    ]
    mtimes = {p: p.stat().st_mtime for p in live_files if p.exists()}
    screened = pd.read_csv(screened_path)
    plan, diag = size_book_from_screened(
        screened,
        "2026-07-10",
        cfg,
        state_root=tmp_path,
        quiet=True,
    )
    assert diag["n_plan_rows"] > 0
    b4 = plan[plan["sleeve"].astype(str).str.contains("inverse", na=False)]
    b4_gross = float(b4["gross_target_usd"].sum()) if len(b4) else 0.0
    # Sleeve budget scale: B4 should be order of sleeve allocation, not millions.
    assert b4_gross < 2_000_000.0
    assert np.isfinite(b4_gross)
    for p, mt in mtimes.items():
        assert p.stat().st_mtime == mt, f"live state mutated: {p}"
    assert (tmp_path / "b4_crash_l_state.json").exists()
    assert (tmp_path / "runs" / "2026-07-10" / "proposed_trades.csv").exists()


def test_pair_eq_not_wiped_on_nan_friday():
    """Inactive names keep equity across NaN Fridays (SMYY/COYY bug)."""
    cal = pd.bdate_range("2025-05-01", periods=80)
    # Pair A live entire window; pair B starts halfway through
    a_px = pd.Series(100.0 * (1.001 ** np.arange(len(cal))), index=cal)
    b_px = pd.Series(50.0 * (1.0005 ** np.arange(len(cal))), index=cal)
    late = cal[40:]
    c_px = pd.Series(np.nan, index=cal)
    c_px.loc[late] = 20.0 * (1.002 ** np.arange(len(late)))
    d_px = pd.Series(np.nan, index=cal)
    d_px.loc[late] = 10.0 * (1.001 ** np.arange(len(late)))

    panel = {
        "AAA": pd.DataFrame({"a_px": a_px, "b_px": b_px}),
        "BBB": pd.DataFrame({"a_px": c_px, "b_px": d_px}),
    }
    uni = pd.DataFrame(
        [
            {
                "ETF": "AAA",
                "Underlying": "X",
                "sleeve": "yieldboost",
                "long_usd": 40_000.0,
                "short_usd": -60_000.0,
                "gross_target_usd": 100_000.0,
                "borrow_current": 0.10,
            },
            {
                "ETF": "BBB",
                "Underlying": "Y",
                "sleeve": "yieldboost",
                "long_usd": 40_000.0,
                "short_usd": -60_000.0,
                "gross_target_usd": 100_000.0,
                "borrow_current": 0.10,
            },
        ]
    )
    nav, meta, stats = _stock_sleeve_nav(
        uni,
        panel,
        sleeve="yieldboost",
        start=cal[0],
        budget_usd=200_000.0,
        enter_band_pct=0.12,
        slippage_bps=0.0,
    )
    assert len(nav) > 40
    assert meta["n_pairs"] == 2
    bbb = stats[stats["ETF"] == "BBB"]
    assert len(bbb) == 1
    # Must not be a fake -100% wipe
    assert float(bbb["ret"].iloc[0]) > -0.99
    assert float(bbb["end_usd"].iloc[0]) > 0


def test_pair_stats_drops_leading_zeros_not_corrupt():
    idx = pd.bdate_range("2025-05-01", periods=30)
    s = pd.Series(0.0, index=idx)
    s.iloc[10:] = np.linspace(1000, 1100, 20)
    stats = _pair_stats_from_navs(
        {"XYZ": s},
        sleeve="inverse_decay_bucket4",
        und_by_etf={"XYZ": "UND"},
        start_usd_by_etf={"XYZ": 1000.0},
    )
    assert len(stats) == 1
    assert float(stats["start_usd"].iloc[0]) == 1000.0
    assert float(stats["ret"].iloc[0]) == pytest.approx(0.1, rel=1e-3)
    assert not bool(stats["stats_corrupt"].iloc[0])


def test_pair_daily_returns_borrow_on_short_etf():
    idx = pd.bdate_range("2025-01-01", periods=5)
    px = pd.DataFrame({"a_px": [10, 10, 10, 10, 10], "b_px": [100, 100, 100, 100, 100]}, index=idx)
    row = pd.Series(
        {
            "long_usd": 25_000.0,
            "short_usd": -75_000.0,
            "borrow_current": 0.252,  # 25.2%/yr → 0.001/day on short leg
        }
    )
    r0 = pair_daily_returns(row, px, borrow_on_etf=False, borrow_on_underlying=False)
    r1 = pair_daily_returns(row, px, borrow_on_etf=True, borrow_on_underlying=False)
    # Flat prices → only borrow drag differs
    assert float(r0.dropna().iloc[-1]) == pytest.approx(0.0, abs=1e-12)
    assert float(r1.dropna().iloc[-1]) < 0


def test_coyy_split_adjusted_no_500pct_day():
    panel = load_price_panel("2026-07-10")
    assert "COYY" in panel
    r = panel["COYY"]["a_px"].pct_change().loc["2026-05-28":"2026-06-12"]
    assert float(r.abs().max()) < 0.5


def test_b5_uses_carry_engine_not_b4():
    """Smoke: B5 path returns carry-engine meta (skips if vol panel unavailable)."""
    uni = pd.DataFrame(
        [
            {
                "ETF": "UVIX",
                "Underlying": "SVIX",
                "sleeve": B5_SLEEVE,
                "long_usd": -7000.0,
                "short_usd": -3500.0,
                "gross_target_usd": 10500.0,
                "borrow_current": 0.03,
            }
        ]
    )
    nav, meta, stats = _b5_sleeve_nav(
        uni,
        start=pd.Timestamp("2025-05-01"),
        budget_usd=10500.0,
        slippage_bps=20.0,
    )
    if meta.get("skipped"):
        pytest.skip(f"vol panel unavailable: {meta.get('reason')}")
    assert "bucket5_carry" in str(meta.get("engine", ""))
    assert "rho" in meta
    assert float(meta["rho"]) == pytest.approx(2.0)
    # Must not be the old B4 dynamic-h label
    assert "bucket4_dynamic" not in str(meta.get("engine", ""))
    assert len(nav) > 40


def _simple_plan(date: pd.Timestamp, *, long_usd: float = 200.0, short_usd: float = -200.0):
    return normalize_plan(
        pd.DataFrame(
            [
                {
                    "ETF": "AAA",
                    "Underlying": "BBB",
                    "sleeve": "core_leveraged",
                    "long_usd": long_usd,
                    "short_usd": short_usd,
                    "gross_target_usd": abs(long_usd) + abs(short_usd),
                    "borrow_current": 0.0,
                }
            ]
        ),
        source_date=str(date.date()),
    )


def test_replay_preserves_four_x_gross_and_next_close_timing():
    cal = pd.bdate_range("2025-01-01", periods=30)
    panel = {
        "AAA": pd.DataFrame(
            {
                "a_px": 100.0 * (0.99 ** np.arange(len(cal))),
                "b_px": 100.0,
            },
            index=cal,
        )
    }
    nav, audit, meta, _, daily = simulate_book_from_plan_timeline(
        {cal[0]: _simple_plan(cal[0])},
        panel,
        budgets={"core_leveraged": 400.0},
        capital_usd=100.0,
        start=cal[0],
        slippage_bps=0.0,
        commission_per_share=0.0,
        margin_rate_annual=0.0,
        short_proceeds_credit_annual=0.0,
        min_trade_usd=0.0,
    )
    # Plan stamped at the first close executes at the second close.  It cannot
    # earn either first-day or second-day close-to-close return.
    assert nav.iloc[0] == pytest.approx(100.0)
    assert nav.iloc[1] == pytest.approx(100.0)
    assert nav.iloc[2] == pytest.approx(102.0)  # short 50% ETF leg * -1% * 4x gross
    assert float(daily.loc[daily["date"] == cal[1], "gross_leverage"].iloc[0]) == pytest.approx(4.0)
    assert pd.Timestamp(audit.iloc[0]["date"]) == cal[1]
    assert meta["execution_lag_sessions"] == 1


def test_replay_does_not_use_latest_plan_leg_mix_in_history():
    cal = pd.bdate_range("2025-01-01", periods=35)
    panel = {
        "AAA": pd.DataFrame(
            {
                "a_px": 100.0 * (1.01 ** np.arange(len(cal))),
                "b_px": 100.0 * (0.99 ** np.arange(len(cal))),
            },
            index=cal,
        )
    }
    # First plan is long underlying / short ETF (negative on this tape).  The
    # later plan reverses it.  Earlier P&L must retain the first plan's signs.
    p0 = _simple_plan(cal[0], long_usd=200.0, short_usd=-200.0)
    p1 = _simple_plan(cal[15], long_usd=-200.0, short_usd=200.0)
    _, _, _, _, daily = simulate_book_from_plan_timeline(
        {cal[0]: p0, cal[15]: p1},
        panel,
        budgets={"core_leveraged": 400.0},
        capital_usd=100.0,
        start=cal[0],
        slippage_bps=0.0,
        commission_per_share=0.0,
        margin_rate_annual=0.0,
        min_trade_usd=0.0,
    )
    early = daily[(daily["date"] >= cal[2]) & (daily["date"] < cal[15])]
    assert (early["daily_price_pnl"] < 0).all()


def test_short_proceeds_credit_at_3p8_reconciles():
    """IBKR-style 3.8% credit on short sale proceeds (Actual/360)."""
    cal = pd.bdate_range("2025-01-01", periods=25)
    panel = {"AAA": pd.DataFrame({"a_px": 100.0, "b_px": 50.0}, index=cal)}
    nav, _, meta, _, daily = simulate_book_from_plan_timeline(
        {cal[0]: _simple_plan(cal[0], long_usd=200.0, short_usd=-200.0)},
        panel,
        budgets={"core_leveraged": 400.0},
        capital_usd=100.0,
        start=cal[0],
        slippage_bps=0.0,
        commission_per_share=0.0,
        margin_rate_annual=0.0,
        short_proceeds_credit_annual=0.038,
        financing_daycount=360.0,
        min_trade_usd=0.0,
        retarget_on_plan_change=True,
    )
    assert not daily.empty
    assert float(daily["pnl_recon_residual"].abs().max()) < 1e-9
    # Flat prices → credit only: 200 * 0.038 / 360 per day after open
    after = daily[daily["date"] > cal[1]]
    assert (after["daily_short_credit"] > 0).all()
    assert float(after["daily_short_credit"].iloc[0]) == pytest.approx(200.0 * 0.038 / 360.0, rel=1e-6)
    assert float(meta["short_proceeds_credit_annual"]) == pytest.approx(0.038)
    assert meta["same_run_churn_enabled"] is True
    assert meta["one_terminal_target_per_symbol"] is True
    assert meta["avoided_round_trip_usd"] == 0.0
    assert meta["risk_override_turnover_usd"] == 0.0


def test_replay_charges_opening_cost_and_reconciles_daily_pnl():
    cal = pd.bdate_range("2025-01-01", periods=25)
    panel = {"AAA": pd.DataFrame({"a_px": 100.0, "b_px": 50.0}, index=cal)}
    nav, _, meta, pair_stats, daily = simulate_book_from_plan_timeline(
        {cal[0]: _simple_plan(cal[0])},
        panel,
        budgets={"core_leveraged": 400.0},
        capital_usd=100.0,
        start=cal[0],
        slippage_bps=10.0,
        commission_per_share=0.0035,
        margin_rate_annual=0.0,
        min_trade_usd=0.0,
    )
    open_day = daily[daily["daily_txn_cost"] > 0].iloc[0]
    assert open_day["daily_txn_cost"] > 0
    assert nav.loc[open_day["date"]] < 100.0
    assert float(daily["pnl_recon_residual"].abs().max()) < 1e-9
    assert float(pair_stats["txn_cost_usd"].sum()) == pytest.approx(float(daily["daily_txn_cost"].sum()))
    pair_daily = meta["pair_daily"]
    assert not pair_daily.empty
    assert "cum_pnl" in pair_daily.columns
    end_pnl = float(pair_daily.loc[pair_daily["ETF"] == "AAA", "cum_pnl"].iloc[-1])
    assert end_pnl == pytest.approx(float(pair_stats.loc[pair_stats["ETF"] == "AAA", "pnl_usd"].iloc[0]), rel=1e-6)


def test_plan_schema_maps_short_usd_to_etf_and_long_usd_to_underlying():
    cal = pd.bdate_range("2025-01-01", periods=3)
    raw = pd.DataFrame(
        [
            {
                "ETF": "AAA",
                "Underlying": "BBB",
                "sleeve": "core_leveraged",
                "long_usd": 300.0,
                "short_usd": -100.0,
                "underlying_target_usd": 300.0,
                "etf_target_usd": -100.0,
                "gross_target_usd": 400.0,
                "borrow_current": 0.10,
                "underlying_borrow_annual": 0.02,
            }
        ]
    )
    plan = normalize_plan(raw, source_date="2025-01-01")
    panel = {"AAA": pd.DataFrame({"a_px": 100.0, "b_px": 50.0}, index=cal)}
    target = _targets_from_plan(
        plan,
        budgets={"core_leveraged": 400.0},
        panel=panel,
        equity=100.0,
        capital_usd=100.0,
        target_notional_mode="fixed_plan_usd",
        scale_sleeves_to_budget=False,
    )
    assert float(target.at["AAA", "etf_usd"]) == pytest.approx(-100.0)
    assert float(target.at["AAA", "underlying_usd"]) == pytest.approx(300.0)
    assert float(target.at["AAA", "borrow_underlying"]) == pytest.approx(0.02)


def test_scale_sleeves_to_budget_upsizes_undersized_plan():
    """Plan sleeve gross $50k with budget $100k → targets sum to $100k."""
    cal = pd.bdate_range("2025-01-01", periods=3)
    plan = normalize_plan(
        pd.DataFrame(
            [
                {
                    "ETF": "AAA",
                    "Underlying": "BBB",
                    "sleeve": "core_leveraged",
                    "long_usd": 30_000.0,
                    "short_usd": -20_000.0,
                    "gross_target_usd": 50_000.0,
                    "borrow_current": 0.0,
                }
            ]
        ),
        source_date="2025-01-01",
    )
    panel = {"AAA": pd.DataFrame({"a_px": 100.0, "b_px": 50.0}, index=cal)}
    scaled = _targets_from_plan(
        plan,
        budgets={"core_leveraged": 100_000.0},
        panel=panel,
        equity=100.0,
        capital_usd=100.0,
        target_notional_mode="fixed_plan_usd",
        scale_sleeves_to_budget=True,
    )
    unscaled = _targets_from_plan(
        plan,
        budgets={"core_leveraged": 100_000.0},
        panel=panel,
        equity=100.0,
        capital_usd=100.0,
        target_notional_mode="fixed_plan_usd",
        scale_sleeves_to_budget=False,
    )
    assert float(scaled["gross_usd"].sum()) == pytest.approx(100_000.0)
    assert float(unscaled["gross_usd"].sum()) == pytest.approx(50_000.0)
    assert float(scaled.attrs["planned_gross_usd"]) == pytest.approx(100_000.0)
    # Leg mix preserved (60/40 long/short of gross)
    assert float(scaled.at["AAA", "underlying_usd"]) == pytest.approx(60_000.0)
    assert float(scaled.at["AAA", "etf_usd"]) == pytest.approx(-40_000.0)


def test_replay_phase2b_band_skips_small_existing_resize():
    cal = pd.bdate_range("2025-01-01", periods=30)
    panel = {"AAA": pd.DataFrame({"a_px": 100.0, "b_px": 50.0}, index=cal)}
    p0 = normalize_plan(
        pd.DataFrame(
            [
                {
                    "ETF": "AAA",
                    "Underlying": "BBB",
                    "sleeve": "core_leveraged",
                    "long_usd": 3000.0,
                    "short_usd": -1000.0,
                    "gross_target_usd": 4000.0,
                }
            ]
        ),
        source_date=str(cal[0].date()),
    )
    # Five-percent resize is inside the 12% enter band on both legs.
    p1 = normalize_plan(
        pd.DataFrame(
            [
                {
                    "ETF": "AAA",
                    "Underlying": "BBB",
                    "sleeve": "core_leveraged",
                    "long_usd": 3150.0,
                    "short_usd": -1050.0,
                    "gross_target_usd": 4200.0,
                }
            ]
        ),
        source_date=str(cal[10].date()),
    )
    _, audit, _, _, _ = simulate_book_from_plan_timeline(
        {cal[0]: p0, cal[10]: p1},
        panel,
        budgets={"core_leveraged": 5000.0},
        capital_usd=1000.0,
        start=cal[0],
        slippage_bps=0.0,
        commission_per_share=0.0,
        margin_rate_annual=0.0,
        enter_band_pct=0.12,
        exit_band_pct=0.04,
        min_trade_usd=250.0,
        scale_sleeves_to_budget=False,
    )
    first = audit.iloc[0]
    changed = audit[audit["plan_date"] == str(cal[10].date())].iloc[0]
    assert float(first["turnover_usd"]) == pytest.approx(4000.0)
    assert float(changed["turnover_usd"]) == pytest.approx(0.0)


def test_net_shared_underlyings_financing_and_price_pnl():
    """B1 long + B4 short on same und: borrow only on residual; price PnL still marks both."""
    from scripts.production_actual_backtest import (
        _netted_book_notionals,
        _underlying_net_by_symbol,
        compute_sleeve_return_metrics,
    )

    cur = pd.DataFrame(
        [
            {
                "etf_usd": -1000.0,
                "underlying_usd": 800.0,
                "Underlying": "AAA",
                "sleeve": "core_leveraged",
                "borrow_current": 0.10,
                "borrow_underlying": 0.20,
            },
            {
                "etf_usd": -2000.0,
                "underlying_usd": -500.0,
                "Underlying": "AAA",
                "sleeve": "inverse_decay_bucket4",
                "borrow_current": 0.05,
                "borrow_underlying": 0.20,
            },
        ],
        index=["ETF1", "ETF2"],
    )
    nets = _underlying_net_by_symbol(cur)
    assert nets["AAA"] == pytest.approx(300.0)
    long_n, short_n, gross, net = _netted_book_notionals(cur)
    # ETF shorts 1000+2000; und net long 300 → long=300, short=3000
    assert long_n == pytest.approx(300.0)
    assert short_n == pytest.approx(3000.0)
    assert gross == pytest.approx(3300.0)
    assert net == pytest.approx(-2700.0)

    cal = pd.bdate_range("2024-01-02", periods=40)
    panel = {
        "ETF1": pd.DataFrame({"a_px": 10.0, "b_px": 20.0}, index=cal),
        "ETF2": pd.DataFrame({"a_px": 10.0, "b_px": 20.0}, index=cal),
    }

    plan = normalize_plan(
        pd.DataFrame(
            [
                {
                    "ETF": "ETF1",
                    "Underlying": "AAA",
                    "sleeve": "core_leveraged",
                    "long_usd": 800.0,
                    "short_usd": -1000.0,
                    "gross_target_usd": 1800.0,
                    "borrow_current": 0.10,
                    "borrow_underlying": 0.40,
                },
                {
                    "ETF": "ETF2",
                    "Underlying": "AAA",
                    "sleeve": "inverse_decay_bucket4",
                    "long_usd": -500.0,
                    "short_usd": -2000.0,
                    "gross_target_usd": 2500.0,
                    "Delta": -1.0,
                    "borrow_current": 0.05,
                    "borrow_underlying": 0.40,
                },
            ]
        ),
        source_date=str(cal[0].date()),
    )
    budgets = {
        "core_leveraged": 5000.0,
        "inverse_decay_bucket4": 5000.0,
    }
    common = dict(
        budgets=budgets,
        capital_usd=5000.0,
        start=cal[0],
        slippage_bps=0.0,
        commission_per_share=0.0,
        margin_rate_annual=0.0,
        short_proceeds_credit_annual=0.0,
        scale_sleeves_to_budget=False,
        retarget_on_plan_change=True,
        use_borrow_history=False,
        min_trade_usd=0.0,
        b4_execution="plan",
    )
    _, _, meta_net, _, daily_net = simulate_book_from_plan_timeline(
        {cal[0]: plan},
        panel,
        net_shared_underlyings=True,
        **common,
    )
    _, _, meta_gross, _, daily_gross = simulate_book_from_plan_timeline(
        {cal[0]: plan},
        panel,
        net_shared_underlyings=False,
        **common,
    )
    assert bool(meta_net.get("net_shared_underlyings")) is True
    assert bool(meta_gross.get("net_shared_underlyings")) is False
    # Flat marks → price PnL ~0 both ways; netting should cut und borrow.
    assert float(daily_net["daily_price_pnl"].sum()) == pytest.approx(0.0, abs=1e-6)
    assert float(daily_gross["daily_price_pnl"].sum()) == pytest.approx(0.0, abs=1e-6)
    assert float(daily_net["daily_borrow_cost"].sum()) < float(
        daily_gross["daily_borrow_cost"].sum()
    ) - 1e-9
    # Residual und is net long → no und borrow under netting; gross still pays on 500 short.
    # ETF borrow remains in both.
    assert "core_leveraged__net_cap" in daily_net.columns
    assert "core_leveraged__gross_cap" in daily_net.columns
    metrics = compute_sleeve_return_metrics(daily_net)
    assert "BOOK" in set(metrics["sleeve"])
    assert "avg_gross_cap" in metrics.columns
    b1 = metrics.loc[metrics["sleeve"] == "core_leveraged"].iloc[0]
    assert float(b1["avg_gross_cap"]) > 0


def test_compute_sleeve_return_metrics_roc_rog():
    from scripts.production_actual_backtest import compute_sleeve_return_metrics

    daily = pd.DataFrame(
        {
            "daily_net_pnl": [10.0, 20.0],
            "net_notional": [100.0, 100.0],
            "gross_notional": [200.0, 200.0],
            "core_leveraged": [5.0, 5.0],
            "core_leveraged__net_cap": [50.0, 50.0],
            "core_leveraged__gross_cap": [100.0, 100.0],
            "yieldboost": [0.0, 0.0],
            "yieldboost__net_cap": [0.0, 0.0],
            "yieldboost__gross_cap": [0.0, 0.0],
            "inverse_decay_bucket4": [10.0, 10.0],
            "inverse_decay_bucket4__net_cap": [-40.0, -40.0],
            "inverse_decay_bucket4__gross_cap": [80.0, 80.0],
            "volatility_etp_bucket5": [0.0, 0.0],
            "volatility_etp_bucket5__net_cap": [0.0, 0.0],
            "volatility_etp_bucket5__gross_cap": [0.0, 0.0],
        }
    )
    m = compute_sleeve_return_metrics(daily)
    book = m.loc[m["sleeve"] == "BOOK"].iloc[0]
    assert float(book["pnl_usd"]) == pytest.approx(30.0)
    assert float(book["roc"]) == pytest.approx(0.30)  # 30/100
    assert float(book["rog"]) == pytest.approx(0.15)  # 30/200
    assert float(book["rog_deployed"]) == pytest.approx(0.15)
    b1 = m.loc[m["sleeve"] == "core_leveraged"].iloc[0]
    assert float(b1["roc"]) == pytest.approx(0.20)  # 10/50
    assert float(b1["rog"]) == pytest.approx(0.10)  # 10/100
    assert float(b1["rog_deployed"]) == pytest.approx(0.10)
    assert float(b1["roc_deployed"]) == pytest.approx(0.20)  # 10/50
    assert float(b1["deployed_day_frac"]) == pytest.approx(1.0)
    b4 = m.loc[m["sleeve"] == "inverse_decay_bucket4"].iloc[0]
    assert pd.isna(b4["roc"])  # avg net negative → n/a
    assert float(b4["rog"]) == pytest.approx(0.25)  # 20/80
    assert float(b4["rog_deployed"]) == pytest.approx(0.25)
    assert float(b4["roc_deployed"]) == pytest.approx(0.50)  # 20 / mean(|net|)=40


def test_stock_rebal_only_operator_5d_skips_off_clock_days():
    """B1 should not trade between operator_5d sessions under rebal_only."""
    from scripts.production_actual_backtest import _membership_day_set

    cal = pd.bdate_range("2025-01-02", periods=20)
    panel = {
        "AAA": pd.DataFrame(
            {
                "a_px": np.linspace(100, 110, len(cal)),
                "b_px": np.linspace(50, 55, len(cal)),
            },
            index=cal,
        )
    }
    plan = _simple_plan(cal[0], long_usd=200.0, short_usd=-100.0)
    check = {
        pd.Timestamp(d).normalize()
        for d in _membership_day_set(cal, mode="operator_5d", check_days=5)
    }
    assert len(check) >= 3
    _, audit, meta, _, _ = simulate_book_from_plan_timeline(
        {cal[0]: plan},
        panel,
        budgets={"core_leveraged": 400.0},
        capital_usd=100.0,
        start=cal[0],
        slippage_bps=0.0,
        commission_per_share=0.0,
        margin_rate_annual=0.0,
        scale_sleeves_to_budget=False,
        min_trade_usd=0.0,
        use_resize_bands=False,
        turnover_pace_mode="hedge_safe_v1",
        stock_rebalance_clock="operator_5d",
        operator_check_days=5,
        stock_midweek_mode="rebal_only",
        midweek_hedge_repair=False,
        confirmation_count=1,
        remaining_gap_rate=1.0,
        max_daily_turnover_pct=10.0,
        hedge_reserve_frac=0.0,
    )
    assert meta["stock_rebalance_clock"] == "operator_5d"
    assert meta["stock_midweek_mode"] == "rebal_only"
    # Off-clock sessions should not print turnover (share-hold; hedge repair off).
    adates = pd.to_datetime(audit["date"]).dt.normalize()
    off = audit.loc[~adates.isin(check)]
    if not off.empty and "turnover_usd" in off.columns:
        assert float(off["turnover_usd"].fillna(0).sum()) == pytest.approx(0.0)


def test_compute_sleeve_return_metrics_excludes_flat_days():
    from scripts.production_actual_backtest import compute_sleeve_return_metrics

    daily = pd.DataFrame(
        {
            "daily_net_pnl": [0.0, 10.0],
            "net_notional": [0.0, 100.0],
            "gross_notional": [0.0, 200.0],
            "core_leveraged": [0.0, 10.0],
            "core_leveraged__net_cap": [0.0, 50.0],
            "core_leveraged__gross_cap": [0.0, 100.0],
            "yieldboost": [0.0, 0.0],
            "yieldboost__net_cap": [0.0, 0.0],
            "yieldboost__gross_cap": [0.0, 0.0],
            "inverse_decay_bucket4": [0.0, 0.0],
            "inverse_decay_bucket4__net_cap": [0.0, 0.0],
            "inverse_decay_bucket4__gross_cap": [0.0, 0.0],
            "volatility_etp_bucket5": [0.0, 0.0],
            "volatility_etp_bucket5__net_cap": [0.0, 0.0],
            "volatility_etp_bucket5__gross_cap": [0.0, 0.0],
        }
    )
    m = compute_sleeve_return_metrics(daily)
    b1 = m.loc[m["sleeve"] == "core_leveraged"].iloc[0]
    # Calendar ROG dilutes with the flat day (avg gross 50); deployed does not.
    assert float(b1["rog"]) == pytest.approx(0.20)  # 10/50
    assert float(b1["rog_deployed"]) == pytest.approx(0.10)  # 10/100
    assert float(b1["deployed_day_frac"]) == pytest.approx(0.5)
    assert int(b1["n_deployed_days"]) == 1
    assert float(b1["rog_deployed_ann"]) == pytest.approx(0.10 * 252 / 2)


def test_pace_leg_caps_step():
    from scripts.production_actual_backtest import _pace_leg

    # 100% jump toward 200 → at most 25% of max(|100|,|200|)=200 → 50 step
    out = _pace_leg(100.0, 200.0, max_leg_step_pct=0.25, min_trade_usd=1.0)
    assert out == pytest.approx(150.0)
    # Within step → full move
    assert _pace_leg(100.0, 120.0, max_leg_step_pct=0.25, min_trade_usd=1.0) == pytest.approx(
        120.0
    )


def test_sleeve_gross_ema_smooths_cliff():
    from scripts.production_actual_backtest import _apply_sleeve_gross_ema

    ema: dict[str, float] = {}
    t1 = pd.DataFrame(
        {
            "ETF": ["A", "B"],
            "sleeve": ["core_leveraged", "core_leveraged"],
            "etf_usd": [-50.0, -50.0],
            "underlying_usd": [50.0, 50.0],
            "gross_usd": [100.0, 100.0],
        }
    ).set_index("ETF")
    t1 = _apply_sleeve_gross_ema(t1, ema, alpha=0.35, sleeves=("core_leveraged",))
    assert ema["core_leveraged"] == pytest.approx(200.0)
    # Cliff: plan halves to 100
    t2 = pd.DataFrame(
        {
            "ETF": ["A", "B"],
            "sleeve": ["core_leveraged", "core_leveraged"],
            "etf_usd": [-25.0, -25.0],
            "underlying_usd": [25.0, 25.0],
            "gross_usd": [50.0, 50.0],
        }
    ).set_index("ETF")
    t2 = _apply_sleeve_gross_ema(t2, ema, alpha=0.35, sleeves=("core_leveraged",))
    # ema = 0.35*100 + 0.65*200 = 165
    assert ema["core_leveraged"] == pytest.approx(165.0)
    assert float(t2[["etf_usd", "underlying_usd"]].abs().sum().sum()) == pytest.approx(165.0)


def test_allocate_turnover_budget_defers_resizes():
    from scripts.production_actual_backtest import _allocate_turnover_budget

    fills = [
        {"etf": "X", "old_a": -100.0, "old_b": 100.0, "new_a": 0.0, "new_b": 0.0, "priority": 0},
        {"etf": "Y", "old_a": 0.0, "old_b": 0.0, "new_a": -50.0, "new_b": 50.0, "priority": 1},
        {"etf": "Z", "old_a": -100.0, "old_b": 100.0, "new_a": -200.0, "new_b": 200.0, "priority": 2},
    ]
    # Exit costs 200; remaining budget 50 with establish_frac=0.5 → est_cap=25, resize gets rest
    accepted, deferred = _allocate_turnover_budget(
        fills, budget_usd=250.0, establish_budget_frac=0.5
    )
    assert any(f["etf"] == "X" for f in accepted)
    # Exit uses 200; remaining 50. Establish cap 25; resize can use remainder.
    etfs = {f["etf"] for f in accepted}
    assert "X" in etfs
    assert deferred >= 0
    # Tight budget: only exit
    accepted2, deferred2 = _allocate_turnover_budget(
        fills, budget_usd=10.0, establish_budget_frac=0.5
    )
    assert [f["etf"] for f in accepted2] == ["X"]
    assert deferred2 == 2


def test_turnover_pace_enabled_false_matches_full_chase():
    """With pacing off, a 50% Friday resize executes in full (bands off)."""
    cal = pd.bdate_range("2025-01-03", periods=40)  # includes Fridays
    panel = {
        "AAA": pd.DataFrame({"a_px": 100.0, "b_px": 50.0}, index=cal),
    }
    # First Friday in range
    fridays = [d for d in cal if d.weekday() == 4]
    d0, d1 = fridays[0], fridays[1]
    p0 = normalize_plan(
        pd.DataFrame(
            [
                {
                    "ETF": "AAA",
                    "Underlying": "BBB",
                    "sleeve": "core_leveraged",
                    "long_usd": 1000.0,
                    "short_usd": -1000.0,
                    "gross_target_usd": 2000.0,
                }
            ]
        ),
        source_date=str(d0.date()),
    )
    p1 = normalize_plan(
        pd.DataFrame(
            [
                {
                    "ETF": "AAA",
                    "Underlying": "BBB",
                    "sleeve": "core_leveraged",
                    "long_usd": 2000.0,
                    "short_usd": -2000.0,
                    "gross_target_usd": 4000.0,
                }
            ]
        ),
        source_date=str(d1.date()),
    )
    common = dict(
        budgets={"core_leveraged": 10000.0},
        capital_usd=2000.0,
        start=cal[0],
        slippage_bps=0.0,
        commission_per_share=0.0,
        margin_rate_annual=0.0,
        short_proceeds_credit_annual=0.0,
        scale_sleeves_to_budget=False,
        use_resize_bands=False,
        min_trade_usd=0.0,
        retarget_on_plan_change=False,
        use_borrow_history=False,
        b4_execution="plan",
    )
    _, audit_off, meta_off, _, _ = simulate_book_from_plan_timeline(
        {d0: p0, d1: p1},
        panel,
        turnover_pace_enabled=False,
        **common,
    )
    _, audit_on, meta_on, _, _ = simulate_book_from_plan_timeline(
        {d0: p0, d1: p1},
        panel,
        turnover_pace_enabled=True,
        max_leg_step_pct=0.25,
        max_daily_turnover_pct=0.15,
        sleeve_gross_ema_alpha=0.35,
        **common,
    )
    assert meta_off.get("turnover_pace_enabled") is False
    assert meta_on.get("turnover_pace_enabled") is True
    # Second Friday turnover should be lower with pacing on.
    def _fri_turn(audit, day):
        rows = audit[audit["date"] == day]
        return float(rows["turnover_usd"].sum()) if len(rows) else 0.0

    # Execution lag: plan on Friday executes next session; find rebal after d1
    turn_off = float(audit_off["turnover_usd"].sum())
    turn_on = float(audit_on["turnover_usd"].sum())
    assert turn_on < turn_off - 1.0


def test_hedge_safe_pair_ramp_scales_both_leg_changes_atomically():
    from scripts.production_actual_backtest import _pace_pair_atomic

    a, b, frac = _pace_pair_atomic(
        -100.0,
        200.0,
        -300.0,
        500.0,
        pair_gross_ramp_pct=0.25,
        min_trade_usd=0.0,
    )
    assert 0.0 < frac < 1.0
    assert (a - (-100.0)) / (-300.0 - (-100.0)) == pytest.approx(frac)
    assert (b - 200.0) / (500.0 - 200.0) == pytest.approx(frac)


def test_hedge_safe_asymmetric_short_repair_targets_flat():
    from scripts.production_actual_backtest import _hedge_correction_usd

    correction = _hedge_correction_usd(
        net_notional=-20.0,
        reference_gross=1000.0,
        long_trigger_net_pct=0.04,
        long_target_net_pct=0.01,
        short_trigger_net_pct=0.01,
        short_target_net_pct=0.00,
    )
    assert correction == pytest.approx(20.0)


def test_hedge_safe_allocator_risk_order_age_and_atomic_partial():
    from scripts.production_actual_backtest import _allocate_hedge_safe_budget

    def fill(etf, cls, age, old, new):
        return {
            "etf": etf,
            "risk_class": cls,
            "age": age,
            "old_a": -old,
            "old_b": old,
            "new_a": -new,
            "new_b": new,
        }

    fills = [
        fill("GROW", "growth", 20, 0.0, 100.0),
        fill("YOUNG", "resize", 1, 100.0, 150.0),
        fill("OLD", "resize", 8, 100.0, 200.0),
        fill("REDUCE", "gross_reduction", 0, 200.0, 100.0),
        fill("EXIT", "hard_exit", 0, 100.0, 0.0),
        fill("HEDGE", "hedge", 0, 100.0, 125.0),
    ]
    accepted, deferred = _allocate_hedge_safe_budget(
        fills, budget_usd=250.0, establish_budget_frac=0.5
    )
    names = [f["etf"] for f in accepted]
    assert names[:3] == ["EXIT", "HEDGE", "REDUCE"]
    assert "OLD" in names
    assert "YOUNG" not in names
    assert "GROW" not in names
    partial = next(f for f in accepted if f["etf"] == "OLD")
    fa = (partial["new_a"] - partial["old_a"]) / (
        -200.0 - partial["old_a"]
    )
    fb = (partial["new_b"] - partial["old_b"]) / (
        200.0 - partial["old_b"]
    )
    assert fa == pytest.approx(fb)
    assert {f["etf"] for f in deferred} >= {"OLD", "YOUNG", "GROW"}


def test_hedge_safe_controller_repairs_long_to_one_percent_outside_budget():
    cal = pd.bdate_range("2025-01-01", periods=25)
    panel = {"AAA": pd.DataFrame({"a_px": 100.0, "b_px": 50.0}, index=cal)}
    plan = normalize_plan(
        pd.DataFrame(
            [
                {
                    "ETF": "AAA",
                    "Underlying": "BBB",
                    "sleeve": "core_leveraged",
                    "Delta": 2.0,
                    "long_usd": 250.0,
                    "short_usd": -100.0,
                    "gross_target_usd": 350.0,
                }
            ]
        ),
        source_date=str(cal[0].date()),
    )
    _, audit, meta, _, _ = simulate_book_from_plan_timeline(
        {cal[0]: plan},
        panel,
        budgets={"core_leveraged": 350.0},
        capital_usd=100.0,
        start=cal[0],
        slippage_bps=0.0,
        commission_per_share=0.0,
        margin_rate_annual=0.0,
        scale_sleeves_to_budget=False,
        min_trade_usd=0.0,
        turnover_pace_mode="hedge_safe_v1",
        confirmation_count=1,
        target_blend_alpha=1.0,
        max_daily_turnover_pct=0.0,
        use_resize_bands=False,
    )
    first = audit[audit["n_hedge_repairs"] > 0].iloc[0]
    assert float(first["hedge_repair_turnover_usd"]) == pytest.approx(4.55)
    assert float(first["turnover_usd"]) > float(first["turnover_budget_usd"])
    assert meta["turnover_pace_mode"] == "hedge_safe_v1"
    assert meta["turnover_pace_version"] == "1"
    assert meta["n_hedge_repairs"] >= 1
    assert meta["risk_override_turnover_usd"] >= 4.55
    repaired = meta["pending_target_audit"]
    assert "etf" in set(repaired["hedge_repair_leg"].dropna())


def test_hedge_safe_blocks_growth_when_delta_is_unavailable():
    cal = pd.bdate_range("2025-01-01", periods=25)
    panel = {"AAA": pd.DataFrame({"a_px": 100.0, "b_px": 50.0}, index=cal)}
    plan = _simple_plan(cal[0])
    _, _, meta, _, daily = simulate_book_from_plan_timeline(
        {cal[0]: plan},
        panel,
        budgets={"core_leveraged": 400.0},
        capital_usd=100.0,
        start=cal[0],
        slippage_bps=0.0,
        commission_per_share=0.0,
        margin_rate_annual=0.0,
        scale_sleeves_to_budget=False,
        min_trade_usd=0.0,
        turnover_pace_mode="hedge_safe_v1",
        confirmation_count=1,
        use_resize_bands=False,
    )
    assert int(daily["n_positions"].max()) == 0
    assert meta["n_growth_blocked_hedge_infeasible"] > 0


def test_config_defaults_to_versioned_hedge_safe_controller():
    knobs = production_actual_backtest.rebalance_knobs(load_config())
    assert knobs["turnover_pace_mode"] == "hedge_safe_v1"
    assert knobs["stock_rebalance_clock"] == "operator_5d"
    assert knobs["confirmation_count"] == 1
    assert knobs["entry_ramp_sessions"] == 1
    assert knobs["reduction_ramp_sessions"] == 1
    assert knobs["remaining_gap_rate"] == pytest.approx(0.25)
    assert knobs["target_blend_alpha"] == pytest.approx(0.25)
    assert knobs["stock_midweek_mode"] == "rebal_only"
    assert knobs["midweek_hedge_repair"] is False
    assert knobs["hedge_reserve_frac"] == pytest.approx(0.15)
    assert knobs["max_daily_turnover_pct"] == pytest.approx(0.10)
    assert knobs["legacy_max_daily_turnover_pct"] == pytest.approx(0.15)
    assert knobs["adv_participation_pct"] == pytest.approx(0.10)
    assert knobs["hedge_long_trigger_net_pct"] == pytest.approx(0.04)
    assert knobs["hedge_long_target_net_pct"] == pytest.approx(0.01)
    assert knobs["hedge_short_trigger_net_pct"] == pytest.approx(0.01)
    assert knobs["hedge_short_target_net_pct"] == pytest.approx(0.00)


def test_rebal_only_operator_day_closes_remaining_gap_fraction():
    """On operator_5d, rebal_only should step gross by remaining_gap_rate, not 100%."""
    from scripts.production_actual_backtest import _membership_day_set

    cal = pd.bdate_range("2025-01-02", periods=30)
    panel = {
        "AAA": pd.DataFrame(
            {"a_px": 100.0, "b_px": 50.0},
            index=cal,
        )
    }
    plan = normalize_plan(
        pd.DataFrame(
            [
                {
                    "ETF": "AAA",
                    "Underlying": "BBB",
                    "sleeve": "core_leveraged",
                    "Delta": 2.0,
                    "long_usd": 200.0,
                    "short_usd": -100.0,
                    "gross_target_usd": 300.0,
                    "borrow_current": 0.0,
                }
            ]
        ),
        source_date=str(cal[0].date()),
    )
    check = sorted(
        pd.Timestamp(d).normalize()
        for d in _membership_day_set(cal, mode="operator_5d", check_days=5)
    )
    assert len(check) >= 2
    _, audit, meta, _, daily = simulate_book_from_plan_timeline(
        {cal[0]: plan},
        panel,
        budgets={"core_leveraged": 400.0},
        capital_usd=100.0,
        start=cal[0],
        slippage_bps=0.0,
        commission_per_share=0.0,
        margin_rate_annual=0.0,
        scale_sleeves_to_budget=False,
        min_trade_usd=0.0,
        use_resize_bands=False,
        turnover_pace_mode="hedge_safe_v1",
        stock_rebalance_clock="operator_5d",
        operator_check_days=5,
        stock_midweek_mode="rebal_only",
        midweek_hedge_repair=False,
        confirmation_count=1,
        remaining_gap_rate=0.25,
        target_blend_alpha=1.0,
        max_daily_turnover_pct=10.0,
        hedge_reserve_frac=0.0,
    )
    assert meta.get("error") is None
    assert meta["remaining_gap_rate"] == pytest.approx(0.25)
    pdaily = meta.get("pair_daily")
    assert pdaily is not None and not pdaily.empty
    gseries = (
        pd.to_numeric(pdaily["etf_usd"], errors="coerce").fillna(0.0).abs()
        + pd.to_numeric(pdaily["underlying_usd"], errors="coerce").fillna(0.0).abs()
    )
    # First step: 25% of destination gross 300 → 75; both legs stay at Δ=2 hedge.
    first = pdaily.loc[gseries > 1e-9].iloc[0]
    g0 = float(abs(first["etf_usd"]) + abs(first["underlying_usd"]))
    assert g0 == pytest.approx(75.0, rel=0.08, abs=8.0)
    assert g0 < 200.0
    assert abs(float(first["underlying_usd"]) / float(first["etf_usd"])) == pytest.approx(
        2.0, rel=1e-6
    )


def test_hedge_safe_persistent_target_ramps_after_confirm_day():
    cal = pd.bdate_range("2025-01-03", periods=35)
    panel = {"AAA": pd.DataFrame({"a_px": 100.0, "b_px": 50.0}, index=cal)}
    d0, d1 = [d for d in cal if d.weekday() == 4][:2]
    p0 = normalize_plan(
        pd.DataFrame([{
            "ETF": "AAA", "Underlying": "BBB", "sleeve": "core_leveraged",
            "Delta": 2.0, "long_usd": 200.0, "short_usd": -100.0,
            "gross_target_usd": 300.0,
        }]),
        source_date=str(d0.date()),
    )
    p1 = normalize_plan(
        pd.DataFrame([{
            "ETF": "AAA", "Underlying": "BBB", "sleeve": "core_leveraged",
            "Delta": 2.0, "long_usd": 600.0, "short_usd": -300.0,
            "gross_target_usd": 900.0,
        }]),
        source_date=str(d1.date()),
    )
    _, audit, meta, _, _ = simulate_book_from_plan_timeline(
        {d0: p0, d1: p1},
        panel,
        budgets={"core_leveraged": 900.0},
        capital_usd=300.0,
        start=cal[0],
        slippage_bps=0.0,
        commission_per_share=0.0,
        margin_rate_annual=0.0,
        scale_sleeves_to_budget=False,
        min_trade_usd=0.0,
        retarget_on_plan_change=False,
        use_resize_bands=False,
        turnover_pace_mode="hedge_safe_v1",
        pair_gross_ramp_pct=0.20,
        max_daily_turnover_pct=0.20,
        establish_budget_frac=1.0,
    )
    post = audit[pd.to_datetime(audit["date"]) > d1]
    assert len(post) >= 2
    assert (post["persistent_target_pairs"] == 1).all()
    assert (~post["target_confirmed_today"].astype(bool)).any()
    assert (post["turnover_usd"] > 0).sum() >= 2
    assert meta["turnover_pace_mode"] == "hedge_safe_v1"


def test_hedge_safe_freezes_stock_gross_between_weekly_decisions_and_refreshes_delta():
    from scripts.production_actual_backtest import _refresh_stock_target_metadata

    frozen = pd.Series({
        "ETF": "AAA", "Underlying": "BBB", "sleeve": "core_leveraged",
        "etf_usd": -100.0, "underlying_usd": 200.0, "gross_usd": 300.0,
        "Delta": 2.0, "etf_adv_usd": 1000.0,
    })
    fresh = frozen.copy()
    fresh.update({
        "etf_usd": -100.0, "underlying_usd": 500.0, "gross_usd": 600.0,
        "Delta": 4.0, "etf_adv_usd": 2500.0,
    })
    refreshed = _refresh_stock_target_metadata(frozen, fresh)
    assert abs(refreshed["etf_usd"]) + abs(refreshed["underlying_usd"]) == pytest.approx(300.0)
    assert refreshed["Delta"] == pytest.approx(4.0)
    assert refreshed["etf_adv_usd"] == pytest.approx(2500.0)
    assert abs(refreshed["underlying_usd"] / refreshed["etf_usd"]) == pytest.approx(5.0)

    cal = pd.bdate_range("2025-01-06", periods=25)  # Monday start
    panel = {"AAA": pd.DataFrame({"a_px": 100.0, "b_px": 50.0}, index=cal)}
    first = _simple_plan(cal[0], long_usd=200.0, short_usd=-100.0)
    first["Delta"] = 2.0
    resized = _simple_plan(cal[1], long_usd=500.0, short_usd=-100.0)
    resized["Delta"] = 4.0
    _, _, meta, _, _ = simulate_book_from_plan_timeline(
        {cal[0]: first, cal[1]: resized},
        panel,
        budgets={"core_leveraged": 600.0},
        capital_usd=100.0,
        start=cal[0],
        slippage_bps=0.0,
        commission_per_share=0.0,
        margin_rate_annual=0.0,
        scale_sleeves_to_budget=False,
        min_trade_usd=0.0,
        retarget_on_plan_change=False,
        use_resize_bands=False,
        turnover_pace_mode="hedge_safe_v1",
        confirmation_count=1,
        entry_ramp_sessions=1,
        remaining_gap_rate=1.0,
        target_blend_alpha=1.0,
        max_daily_turnover_pct=10.0,
    )
    ledger = meta["pending_target_audit"]
    between = ledger[
        (ledger["ETF"] == "AAA")
        & (pd.to_datetime(ledger["date"]) == cal[2])
        & (ledger["priority"] != "hedge_repair")
    ]
    assert not between.empty
    assert float(between.iloc[-1]["desired_gross_usd"]) == pytest.approx(300.0)


def test_hedge_safe_turnover_budget_uses_desired_gross_when_underdeployed():
    from scripts.production_actual_backtest import _turnover_budget_reference_gross

    assert _turnover_budget_reference_gross(
        60.0, 300.0, hedge_safe=True
    ) == pytest.approx(300.0)
    assert _turnover_budget_reference_gross(
        60.0, 300.0, hedge_safe=False
    ) == pytest.approx(60.0)

    cal = pd.bdate_range("2025-01-06", periods=25)
    panel = {"AAA": pd.DataFrame({"a_px": 100.0, "b_px": 50.0}, index=cal)}
    plan = _simple_plan(cal[0], long_usd=200.0, short_usd=-100.0)
    plan["Delta"] = 2.0
    _, audit, _, _, _ = simulate_book_from_plan_timeline(
        {cal[0]: plan},
        panel,
        budgets={"core_leveraged": 300.0},
        capital_usd=100.0,
        start=cal[0],
        slippage_bps=0.0,
        commission_per_share=0.0,
        margin_rate_annual=0.0,
        scale_sleeves_to_budget=False,
        min_trade_usd=0.0,
        use_resize_bands=False,
        turnover_pace_mode="hedge_safe_v1",
        confirmation_count=1,
        entry_ramp_sessions=5,
        target_blend_alpha=1.0,
        max_daily_turnover_pct=0.08,
    )
    paced = audit[pd.to_numeric(audit["turnover_budget_usd"], errors="coerce").notna()]
    second = paced.iloc[1]
    assert float(second["deployed_gross_usd"]) < float(
        second["confirmed_desired_gross_usd"]
    )
    assert float(second["turnover_reference_gross_usd"]) == pytest.approx(300.0)
    assert float(second["turnover_budget_usd"]) == pytest.approx(24.0)


def test_weekly_stock_target_blend_is_convex_and_preserves_hedge_ratios():
    from scripts.production_actual_backtest import _blend_stock_structural_targets

    prior = pd.DataFrame(
        [
            {
                "Underlying": "UA", "sleeve": "core_leveraged",
                "etf_usd": -100.0, "underlying_usd": 200.0, "gross_usd": 300.0,
                "Delta": 2.0,
            },
            {
                "Underlying": "UB", "sleeve": "core_leveraged",
                "etf_usd": -100.0, "underlying_usd": 200.0, "gross_usd": 300.0,
                "Delta": 2.0,
            },
        ],
        index=["A", "B"],
    )
    raw = pd.DataFrame(
        [
            {
                "Underlying": "UA", "sleeve": "core_leveraged",
                "etf_usd": -100.0, "underlying_usd": 300.0, "gross_usd": 400.0,
                "Delta": 3.0,
            },
            {
                "Underlying": "UC", "sleeve": "core_leveraged",
                "etf_usd": -50.0, "underlying_usd": 150.0, "gross_usd": 200.0,
                "Delta": 3.0,
            },
        ],
        index=["A", "C"],
    )
    blended, audit = _blend_stock_structural_targets(
        prior, raw, confirmed_members={"A", "C"}, alpha=0.25
    )
    gross = blended[["etf_usd", "underlying_usd"]].abs().sum(axis=1)
    assert gross["A"] == pytest.approx(325.0)  # retained convex transition
    assert gross["B"] == pytest.approx(225.0)  # confirmed drop decays by alpha
    assert gross["C"] == pytest.approx(50.0)   # new pair enters through alpha
    assert gross.sum() == pytest.approx(600.0)  # equal sleeve totals stay equal
    assert abs(blended.at["A", "underlying_usd"] / blended.at["A", "etf_usd"]) == pytest.approx(3.0)
    assert abs(blended.at["B", "underlying_usd"] / blended.at["B", "etf_usd"]) == pytest.approx(2.0)
    assert {row["blend_status"] for row in audit} == {"retained", "drop_decay", "new"}

    larger = raw.copy()
    larger.loc["A", ["etf_usd", "underlying_usd", "gross_usd"]] = [-150.0, 450.0, 600.0]
    larger.loc["C", ["etf_usd", "underlying_usd", "gross_usd"]] = [-100.0, 300.0, 400.0]
    blended_larger, _ = _blend_stock_structural_targets(
        prior, larger, confirmed_members={"A", "C"}, alpha=0.25
    )
    larger_total = blended_larger[["etf_usd", "underlying_usd"]].abs().sum().sum()
    assert larger_total == pytest.approx(700.0)  # 75% * 600 + 25% * 1000


def test_weekly_stock_target_blend_audits_raw_and_blended_gross():
    cal = pd.bdate_range("2025-01-06", periods=25)
    panel = {"AAA": pd.DataFrame({"a_px": 100.0, "b_px": 50.0}, index=cal)}
    plan = _simple_plan(cal[0], long_usd=200.0, short_usd=-100.0)
    plan["Delta"] = 2.0
    _, audit, meta, _, _ = simulate_book_from_plan_timeline(
        {cal[0]: plan}, panel,
        budgets={"core_leveraged": 300.0}, capital_usd=100.0, start=cal[0],
        slippage_bps=0.0, commission_per_share=0.0, margin_rate_annual=0.0,
        scale_sleeves_to_budget=False, min_trade_usd=0.0,
        turnover_pace_mode="hedge_safe_v1", confirmation_count=1,
        target_blend_alpha=0.25, entry_ramp_sessions=1,
        max_daily_turnover_pct=10.0, use_resize_bands=False,
    )
    formation = meta["pending_target_audit"].query(
        "ETF == 'AAA' and priority == 'target_formation'"
    ).iloc[0]
    assert float(formation["raw_plan_gross_usd"]) == pytest.approx(300.0)
    assert float(formation["blended_structural_gross_usd"]) == pytest.approx(75.0)
    first = audit.iloc[0]
    assert float(first["raw_plan_stock_gross_usd"]) == pytest.approx(300.0)
    assert float(first["blended_stock_structural_gross_usd"]) == pytest.approx(75.0)
    pair_daily = meta["pair_daily"].query("ETF == 'AAA'")
    assert "Delta" in pair_daily.columns
    assert pair_daily["Delta"].dropna().eq(2.0).all()


def test_hedge_safe_adv_cap_is_atomic_and_missing_is_noop():
    from scripts.production_actual_backtest import _apply_adv_participation_cap

    base = {
        "etf": "AAA", "old_a": 0.0, "old_b": 0.0,
        "new_a": -1000.0, "new_b": 2000.0,
        "new": pd.Series({"etf_adv_usd": 2000.0, "underlying_adv_usd": 10000.0}),
        "old": None,
    }
    capped, reason = _apply_adv_participation_cap(
        base, etf_price=10.0, underlying_price=20.0, adv_participation_pct=0.10
    )
    assert reason == "adv_cap"
    assert capped["new_a"] == pytest.approx(-200.0)
    assert capped["new_b"] == pytest.approx(400.0)
    no_adv = dict(base)
    no_adv["new"] = pd.Series(dtype=object)
    uncapped, reason2 = _apply_adv_participation_cap(
        no_adv, etf_price=10.0, underlying_price=20.0, adv_participation_pct=0.10
    )
    assert reason2 is None
    assert uncapped["new_a"] == pytest.approx(-1000.0)


def test_hedge_safe_simulator_applies_plan_adv_fields():
    cal = pd.bdate_range("2025-01-01", periods=25)
    panel = {"AAA": pd.DataFrame({"a_px": 100.0, "b_px": 50.0}, index=cal)}
    raw = pd.DataFrame([{
        "ETF": "AAA", "Underlying": "BBB", "sleeve": "core_leveraged",
        "Delta": 2.0, "long_usd": 200.0, "short_usd": -100.0,
        "gross_target_usd": 300.0, "etf_adv_usd": 500.0,
        "underlying_adv_usd": 5000.0,
    }])
    plan = normalize_plan(raw, source_date=str(cal[0].date()))
    _, _, meta, _, _ = simulate_book_from_plan_timeline(
        {cal[0]: plan}, panel,
        budgets={"core_leveraged": 300.0}, capital_usd=100.0, start=cal[0],
        slippage_bps=0.0, commission_per_share=0.0, margin_rate_annual=0.0,
        scale_sleeves_to_budget=False, min_trade_usd=0.0,
        turnover_pace_mode="hedge_safe_v1", confirmation_count=1,
        entry_ramp_sessions=1, target_blend_alpha=1.0,
        adv_participation_pct=0.10,
        max_daily_turnover_pct=10.0, use_resize_bands=False,
    )
    first = meta["pending_target_audit"].query(
        "ETF == 'AAA' and block_reason == 'adv_cap'"
    ).iloc[0]
    assert first["block_reason"] == "adv_cap"
    assert float(first["next_gross_usd"]) == pytest.approx(150.0)


def test_hedge_safe_live_leg_selection_and_fallback_avoids_b4():
    from scripts.production_actual_backtest import _select_live_semantic_hedge_repair

    group = pd.DataFrame(
        [
            {
                "sleeve": "inverse_decay_bucket4", "Delta": -2.0,
                "etf_usd": -500.0, "underlying_usd": -400.0,
                "shares_available": 10000,
            },
            {
                "sleeve": "core_leveraged", "Delta": 2.0,
                "etf_usd": -100.0, "underlying_usd": 250.0,
                "shares_available": 1000,
            },
        ],
        index=["B4ETF", "CORE"],
    )
    etf, leg, change, reason = _select_live_semantic_hedge_repair(
        group, correction_usd=-40.0, etf_prices={"CORE": 10.0}
    )
    assert (etf, leg, reason) == ("CORE", "etf", None)
    assert change == pytest.approx(-20.0)

    blocked = group.copy()
    blocked.loc["CORE", "shares_available"] = 0
    etf2, leg2, change2, reason2 = _select_live_semantic_hedge_repair(
        blocked, correction_usd=-40.0, etf_prices={"CORE": 10.0}
    )
    assert (etf2, leg2, reason2) == ("CORE", "underlying", "fallback_reduce_long")
    assert change2 == pytest.approx(-40.0)

    etf3, leg3, change3, reason3 = _select_live_semantic_hedge_repair(
        group, correction_usd=30.0
    )
    assert (etf3, leg3, reason3) == ("CORE", "underlying", None)
    assert change3 == pytest.approx(30.0)


def test_phase3_hedge_scope_excludes_b4_and_repairs_shared_stock_residual():
    from scripts.production_actual_backtest import (
        _delta_adjusted_pair_exposure,
        _hedge_correction_usd,
        _phase3_stock_residual_book,
        _select_live_semantic_hedge_repair,
    )

    shared = pd.DataFrame(
        [
            {
                "Underlying": "SHARED", "sleeve": "core_leveraged",
                "Delta": 2.0, "etf_usd": -100.0, "underlying_usd": 250.0,
                "shares_available": 1000,
            },
            {
                "Underlying": "SHARED", "sleeve": "inverse_decay_bucket4",
                "Delta": -2.0, "etf_usd": -100.0, "underlying_usd": 0.0,
                "shares_available": 1000,
            },
        ],
        index=["CORE", "B4"],
    )
    scoped = _phase3_stock_residual_book(shared)
    assert list(scoped.index) == ["CORE"]
    exp = _delta_adjusted_pair_exposure(
        scoped.at["CORE", "etf_usd"],
        scoped.at["CORE", "underlying_usd"],
        scoped.at["CORE", "Delta"],
    )
    correction = _hedge_correction_usd(
        net_notional=exp[0], reference_gross=exp[1],
        long_trigger_net_pct=0.04, long_target_net_pct=0.01,
        short_trigger_net_pct=0.01, short_target_net_pct=0.0,
    )
    etf, _, _, _ = _select_live_semantic_hedge_repair(
        scoped, correction_usd=correction, etf_prices={"CORE": 10.0}
    )
    assert etf == "CORE"

    cal = pd.bdate_range("2025-01-01", periods=25)
    panel = {"B4": pd.DataFrame({"a_px": 100.0, "b_px": 50.0}, index=cal)}
    b4_plan = normalize_plan(
        pd.DataFrame([{
            "ETF": "B4", "Underlying": "SHARED",
            "sleeve": "inverse_decay_bucket4", "Delta": -2.0,
            "long_usd": 0.0, "short_usd": -100.0,
            "gross_target_usd": 100.0,
        }]),
        source_date=str(cal[0].date()),
    )
    _, _, meta, _, _ = simulate_book_from_plan_timeline(
        {cal[0]: b4_plan}, panel,
        budgets={"inverse_decay_bucket4": 100.0},
        capital_usd=100.0, start=cal[0],
        slippage_bps=0.0, commission_per_share=0.0, margin_rate_annual=0.0,
        scale_sleeves_to_budget=False, min_trade_usd=0.0,
        b4_execution="weekly_plan_legs", turnover_pace_mode="hedge_safe_v1",
        max_daily_turnover_pct=10.0, use_resize_bands=False,
    )
    assert meta["n_hedge_repairs"] == 0


def test_hedge_safe_confirmation_and_drop_require_two_effective_plans():
    cal = pd.bdate_range("2025-01-01", periods=30)
    panel = {"AAA": pd.DataFrame({"a_px": 100.0, "b_px": 50.0}, index=cal)}
    present0 = _simple_plan(cal[0], long_usd=200.0, short_usd=-100.0)
    present0["Delta"] = 2.0
    present1 = present0.copy()
    present1["source_date"] = str(cal[1].date())
    empty = normalize_plan(
        pd.DataFrame([{
            "ETF": "OTHER", "Underlying": "X", "sleeve": "inverse_decay_bucket4",
            "Delta": -2.0, "long_usd": 0.0, "short_usd": 0.0,
            "gross_target_usd": 0.0, "purgatory": True,
        }]),
        source_date=str(cal[4].date()),
    )
    _, _, meta, _, _ = simulate_book_from_plan_timeline(
        {cal[0]: present0, cal[1]: present1, cal[4]: empty, cal[5]: empty.copy()},
        panel,
        budgets={"core_leveraged": 300.0},
        capital_usd=100.0,
        start=cal[0],
        slippage_bps=0.0,
        commission_per_share=0.0,
        margin_rate_annual=0.0,
        scale_sleeves_to_budget=False,
        min_trade_usd=0.0,
        retarget_on_plan_change=True,
        use_resize_bands=False,
        turnover_pace_mode="hedge_safe_v1",
        confirmation_count=2,
        target_blend_alpha=1.0,
        entry_ramp_sessions=1,
        reduction_ramp_sessions=1,
        max_daily_turnover_pct=10.0,
    )
    ledger = meta["pending_target_audit"]
    aaa = ledger[ledger["ETF"] == "AAA"].copy()
    first_trade = pd.Timestamp(aaa.loc[aaa["allocated_turnover_usd"] > 0, "date"].min())
    assert first_trade >= cal[2]  # second plan executes one session later
    drop_rows = aaa[aaa["desired_gross_usd"] == 0.0]
    assert not drop_rows.empty
    assert "drop_confirmation" in set(drop_rows["block_reason"].dropna())
    first_drop_trade = pd.Timestamp(
        drop_rows.loc[drop_rows["allocated_turnover_usd"] > 0, "date"].min()
    )
    assert first_drop_trade >= cal[6]


def test_hedge_safe_distinct_entry_reduction_and_resize_rates_in_ledger():
    from scripts.production_actual_backtest import _advance_pair_atomic

    # Five-session linear entry starts at 20%; three-session reduction at 1/3;
    # ordinary existing resize closes 25% of remaining gap.
    assert _advance_pair_atomic(0, 0, -100, 200, completion_fraction=1 / 5) == pytest.approx(
        (-20.0, 40.0)
    )
    assert _advance_pair_atomic(-100, 200, 0, 0, completion_fraction=1 / 3) == pytest.approx(
        (-66.6666667, 133.3333333)
    )
    assert _advance_pair_atomic(-100, 200, -200, 400, completion_fraction=0.25) == pytest.approx(
        (-125.0, 250.0)
    )


def test_hedge_safe_pending_ledger_has_required_fields_and_reserve():
    cal = pd.bdate_range("2025-01-01", periods=25)
    panel = {"AAA": pd.DataFrame({"a_px": 100.0, "b_px": 50.0}, index=cal)}
    plan = _simple_plan(cal[0], long_usd=250.0, short_usd=-100.0)
    plan["Delta"] = 2.0
    _, _, meta, _, _ = simulate_book_from_plan_timeline(
        {cal[0]: plan},
        panel,
        budgets={"core_leveraged": 350.0},
        capital_usd=100.0,
        start=cal[0],
        slippage_bps=0.0,
        commission_per_share=0.0,
        margin_rate_annual=0.0,
        scale_sleeves_to_budget=False,
        min_trade_usd=0.0,
        turnover_pace_mode="hedge_safe_v1",
        confirmation_count=1,
        max_daily_turnover_pct=0.15,
        hedge_reserve_frac=0.20,
        use_resize_bands=False,
    )
    ledger = meta["pending_target_audit"]
    required = {
        "current_gross_usd", "desired_gross_usd", "next_gross_usd",
        "hedge_net_pct_before", "hedge_net_pct_after", "target_age",
        "allocated_turnover_usd", "deferred_turnover_usd", "block_reason",
        "priority", "hedge_reserve_usd", "tracking_budget_usd",
    }
    assert not ledger.empty
    assert required <= set(ledger.columns)


def test_turnover_mode_legacy_and_off_explicit_parity():
    cal = pd.bdate_range("2025-01-03", periods=25)
    panel = {"AAA": pd.DataFrame({"a_px": 100.0, "b_px": 50.0}, index=cal)}
    plan = _simple_plan(cal[0])
    common = dict(
        budgets={"core_leveraged": 400.0}, capital_usd=100.0, start=cal[0],
        slippage_bps=0.0, commission_per_share=0.0, margin_rate_annual=0.0,
        scale_sleeves_to_budget=False, min_trade_usd=0.0, use_resize_bands=False,
    )
    nav_legacy, audit_legacy, _, _, _ = simulate_book_from_plan_timeline(
        {cal[0]: plan}, panel, turnover_pace_enabled=True, **common
    )
    nav_named, audit_named, _, _, _ = simulate_book_from_plan_timeline(
        {cal[0]: plan}, panel, turnover_pace_mode="legacy", **common
    )
    pd.testing.assert_series_equal(nav_legacy, nav_named)
    pd.testing.assert_frame_equal(audit_legacy, audit_named)

    nav_false, audit_false, _, _, _ = simulate_book_from_plan_timeline(
        {cal[0]: plan}, panel, turnover_pace_enabled=False, **common
    )
    nav_off, audit_off, _, _, _ = simulate_book_from_plan_timeline(
        {cal[0]: plan}, panel, turnover_pace_mode="off", **common
    )
    pd.testing.assert_series_equal(nav_false, nav_off)
    pd.testing.assert_frame_equal(audit_false, audit_off)
