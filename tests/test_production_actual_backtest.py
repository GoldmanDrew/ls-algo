"""Regression tests for production actual backtest audit fixes."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

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


def test_normalize_plan_keeps_purgatory_keep_open():
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
    assert bool(ccc["keep_open"]) is True
    assert float(ccc["gross_target_usd"]) == pytest.approx(0.0)


def test_purgatory_keep_open_does_not_liquidate():
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
    # Later plan: only purgatory keep-open (0 targets) — must not close AAA.
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
    _, audit, _, _, daily = simulate_book_from_plan_timeline(
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
    )
    # After second plan date, position should still be open (gross > 0).
    after = daily[daily["date"] > cal[10]]
    assert len(after) > 0
    assert float(after["n_positions"].iloc[0]) >= 1
    changed = audit[audit["plan_date"] == str(cal[10].date())]
    if len(changed):
        assert float(changed.iloc[0]["turnover_usd"]) == pytest.approx(0.0)


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
