"""Bucket 5 Production B — policy parity, ledger idempotency, and plan tests.

Golden parity: scripts/bucket5_policy.py must reproduce the research
implementation in scripts/bucket5_insurance_bt.py exactly (plan Phase 1 exit
gate: "research and live target engines match on frozen inputs to rounding
tolerance").
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.bucket5_policy import (
    HedgeBudgetParams,
    LotState,
    MonetizeParams,
    RegimeParams,
    cadence_interval_days,
    carry_targets,
    hedge_budget_multiplier,
    intent_id,
    load_b5_config,
    monetization_decision,
    order_ref_for_intent,
    redeploy_sleeve_weight,
    regime_state,
    solve_rung_contracts,
    RungSpec,
)
from scripts.bucket5_ledger import (
    Bucket5Ledger,
    fill_event_id,
    intent_event_id,
    submit_event_id,
    tier_event_id,
)


# =============================================================================
# Golden parity with the research implementation
# =============================================================================

RATIO_GRID = [0.70, 0.85, 0.88, 0.90, 0.94, 1.00, 1.05, 1.30]
VIX_GRID = [10.0, 14.0, 18.77, 25.0, 28.0, 45.0, 80.0]


def test_regime_parity_with_insurance_bt():
    from scripts.bucket5_insurance_bt import RegimePolicy

    research = RegimePolicy()
    ratios = pd.Series(RATIO_GRID, index=pd.date_range("2026-01-01", periods=len(RATIO_GRID)))
    rho_r, gross_r = research.series(ratios)
    for i, r in enumerate(RATIO_GRID):
        rho, gross = regime_state(r, RegimeParams())
        assert rho == pytest.approx(float(rho_r.iloc[i]), abs=1e-12)
        assert gross == pytest.approx(float(gross_r.iloc[i]), abs=1e-12)


def test_hedge_budget_parity_with_insurance_bt():
    from scripts.bucket5_insurance_bt import HedgeBudgetPolicy

    research = HedgeBudgetPolicy()
    for r in RATIO_GRID + [float("nan")]:
        for v in VIX_GRID + [float("nan")]:
            live = hedge_budget_multiplier(r, v, HedgeBudgetParams())
            assert live == pytest.approx(research.multiplier(r, v), abs=1e-12)


def test_contract_solver_parity_with_reverse_solve():
    from scripts.bucket5_insurance_bt import (
        HedgeBudgetPolicy, LADDER_2X, reverse_solve_put_contracts,
    )

    equity, spot, iv, ratio, vix = 240_000.0, 6300.0, 0.1877, 0.914, 18.77
    research = reverse_solve_put_contracts(
        equity_usd=equity, spx_spot=spot, atm_iv=iv,
        rungs=LADDER_2X, hedge_budget=HedgeBudgetPolicy(), ratio=ratio, vix=vix,
    )
    mult = hedge_budget_multiplier(ratio, vix, HedgeBudgetParams())
    assert mult == pytest.approx(research["dynamic_budget_multiplier"], abs=1e-12)
    for rr in research["rungs"]:
        rung = RungSpec(
            otm_pct=rr["otm_pct"], per_roll_frac=0.008, quantity_multiplier=2,
        )
        live = solve_rung_contracts(
            rung=rung, effective_b5_nav=equity, budget_multiplier=mult,
            executable_ask=rr["modeled_put_price"],
            contract_multiplier=rr["contract_multiplier"],
            allow_min_one=True,  # research parity mode reproduces max(1, floor(...))
        )
        assert live["baseline_contracts"] == rr["baseline_contracts"]
        assert live["target_contracts"] == rr["target_contracts"]
        assert live["baseline_budget_usd"] == pytest.approx(rr["baseline_budget_usd"], rel=1e-9)


def test_live_solver_never_rounds_up_beyond_budget():
    rung = RungSpec(otm_pct=0.10, per_roll_frac=0.008, quantity_multiplier=2)
    # Rung budget: 240k * 0.8% = $1,920; a $50 ask -> $5,000/contract > budget.
    out = solve_rung_contracts(
        rung=rung, effective_b5_nav=240_000.0, budget_multiplier=1.0,
        executable_ask=50.0, allow_min_one=False,
    )
    assert out["target_contracts"] == 0
    assert out["under_covered"] is True
    # A $9 ask fits twice in $1,920 baseline -> 2 baseline, 4 target contracts.
    out2 = solve_rung_contracts(
        rung=rung, effective_b5_nav=240_000.0, budget_multiplier=1.0,
        executable_ask=9.0, allow_min_one=False,
    )
    assert out2["baseline_contracts"] == 2
    assert out2["target_contracts"] == 4
    assert out2["premium_used_usd"] <= out2["target_budget_usd"] + 1e-9


def test_carry_targets_formula_and_ceiling():
    t = carry_targets(effective_b5_nav=156_875.0, ratio=0.914)
    # rho and gross from linear interpolation between 0.88 and 1.00
    frac = (0.914 - 0.88) / 0.12
    assert t["rho"] == pytest.approx(1.0 + frac, rel=1e-9)
    assert t["uvix_short_usd"] + t["svix_short_usd"] == pytest.approx(t["carry_gross_usd"], abs=1e-9)
    assert t["svix_short_usd"] == pytest.approx(t["uvix_short_usd"] * t["rho"], rel=1e-9)
    capped = carry_targets(effective_b5_nav=156_875.0, ratio=0.70, max_carry_gross_usd=10_000.0)
    assert capped["carry_gross_usd"] == pytest.approx(10_000.0)


def test_regime_state_fails_closed_on_missing_signal():
    with pytest.raises(ValueError):
        regime_state(float("nan"))


def test_cadence_interval_bounds():
    assert cadence_interval_days(0.70) == 14      # deep contango -> base
    assert cadence_interval_days(1.30) == 2       # stress -> floor
    assert 2 <= cadence_interval_days(0.94) <= 14


def test_redeploy_weight_parity():
    from scripts.bucket5_insurance_bt import RedeployPolicy

    research = RedeployPolicy()
    for r in RATIO_GRID:
        assert redeploy_sleeve_weight(r) == pytest.approx(research.sleeve_weight(r), abs=1e-12)


# =============================================================================
# Monetization state machine
# =============================================================================

def _lot(**kw):
    base = dict(entry_contracts=6, remaining_contracts=6, cost_basis_usd=6_000.0)
    base.update(kw)
    return LotState(**base)


def test_monetization_profit_tier_fires_once_and_respects_runner():
    p = MonetizeParams()
    lot = _lot()
    # 3x on the whole entry: bid value for remaining = 3x cost basis
    d = monetization_decision(lot, executable_bid_value_usd=18_000.0, vix=20.0, p=p)
    assert d["reason"].startswith("profit_tier_3x")
    assert d["sell_contracts"] == 2                    # floor(0.34 * 6)
    assert 3.0 in d["fired_profit_tiers"]
    # Same tier cannot fire twice.
    lot2 = _lot(remaining_contracts=4, profit_tiers_fired=(3.0,))
    d2 = monetization_decision(lot2, executable_bid_value_usd=12_000.0, vix=20.0, p=p)
    assert d2["sell_contracts"] == 0 or not str(d2["reason"] or "").startswith("profit_tier_3x")


def test_monetization_runner_floor_blocks_full_sale():
    p = MonetizeParams(runner_frac=0.15)
    lot = _lot(profit_tiers_fired=(3.0, 5.0))
    # 8x tier is frac=1.0 but the runner floor (floor(0.15*6)=0 -> wait, 0.9 -> 0)
    lot_big = _lot(entry_contracts=10, remaining_contracts=10, cost_basis_usd=10_000.0,
                   profit_tiers_fired=(3.0, 5.0))
    d = monetization_decision(lot_big, executable_bid_value_usd=80_000.0, vix=30.0, p=p)
    # runner floor = floor(0.15*10) = 1 contract must stay
    assert d["sell_contracts"] == 9
    assert d["full_exit"] is False


def test_monetization_runner_release_full_exit():
    p = MonetizeParams(runner_mult=12.0)
    lot = _lot(remaining_contracts=1, profit_tiers_fired=(3.0, 5.0, 8.0))
    d = monetization_decision(lot, executable_bid_value_usd=12_500.0, vix=30.0, p=p)
    assert d["full_exit"] is True
    assert d["sell_contracts"] == 1


def test_monetization_vix_tier_and_giveback():
    p = MonetizeParams()
    lot = _lot()
    d = monetization_decision(lot, executable_bid_value_usd=7_000.0, vix=46.0, p=p)
    assert str(d["reason"]).startswith("vix_tier_45")
    # Giveback: peak 3x, now down 40% from peak with min mult reached.
    lot_gb = _lot(peak_mult=3.0, profit_tiers_fired=(3.0,))
    d2 = monetization_decision(lot_gb, executable_bid_value_usd=6_000.0 * 1.8, vix=20.0, p=p)
    assert "giveback" in str(d2["reason"])


def test_monetization_noop_when_flat():
    d = monetization_decision(_lot(), executable_bid_value_usd=6_100.0, vix=15.0)
    assert d["sell_contracts"] == 0
    assert d["reason"] is None


# =============================================================================
# Ledger idempotency and replay
# =============================================================================

def test_ledger_append_is_idempotent(tmp_path):
    led = Bucket5Ledger(tmp_path / "events.jsonl")
    assert led.append("INTENT_EMITTED", intent_event_id("X1"), {"intent_id": "X1"}) is True
    assert led.append("INTENT_EMITTED", intent_event_id("X1"), {"intent_id": "X1"}) is False
    # A NEW instance (process restart) must also refuse the duplicate.
    led2 = Bucket5Ledger(tmp_path / "events.jsonl")
    assert led2.append("INTENT_EMITTED", intent_event_id("X1"), {"intent_id": "X1"}) is False
    assert len(led2.load_events()) == 1


def test_ledger_lot_replay_buy_sell(tmp_path):
    led = Bucket5Ledger(tmp_path / "events.jsonl")
    led.append("FILL", fill_event_id("I1", "e1"), {
        "intent_id": "I1", "conId": "777", "qty": 4, "price": 10.0,
        "multiplier": 100.0, "fees": 2.0, "expiry": "2027-01-15", "strike": 570.0,
        "right": "P", "local_symbol": "XSP 270115P00570000",
    })
    led.append("FILL", fill_event_id("I2", "e2"), {
        "intent_id": "I2", "conId": "777", "qty": -1, "price": 30.0, "multiplier": 100.0,
    })
    lots = led.build_lots()
    lot = lots["777"]
    assert lot["entry_contracts"] == 4
    assert lot["remaining_contracts"] == 3
    assert lot["cost_basis_usd"] == pytest.approx(4_000.0)
    assert lot["realized_cash_usd"] == pytest.approx(3_000.0)
    assert lot["fees_usd"] == pytest.approx(2.0)
    # Tier idempotency
    assert led.append("TIER_FIRED", tier_event_id("777", "profit", 3.0), {
        "conId": "777", "tier_kind": "profit", "level": 3.0}) is True
    assert led.append("TIER_FIRED", tier_event_id("777", "profit", 3.0), {
        "conId": "777", "tier_kind": "profit", "level": 3.0}) is False
    assert led.build_lots()["777"]["profit_tiers_fired"] == [3.0]


def test_ledger_corrupt_line_fails_closed(tmp_path):
    p = tmp_path / "events.jsonl"
    p.write_text('{"event_id": "ok", "kind": "NOTE"}\nnot-json\n', encoding="utf-8")
    with pytest.raises(ValueError):
        Bucket5Ledger(p).load_events()


def test_intent_id_deterministic():
    a = intent_id(strategy_version="B5PROD-1", asof="2026-07-21", action_type="PUTBUY",
                  instrument_key="XSP|2027-01-15|570.0|P", target_stage="otm10")
    b = intent_id(strategy_version="B5PROD-1", asof="2026-07-21", action_type="PUTBUY",
                  instrument_key="XSP|2027-01-15|570.0|P", target_stage="otm10")
    c = intent_id(strategy_version="B5PROD-1", asof="2026-07-21", action_type="PUTBUY",
                  instrument_key="XSP|2027-01-15|560.0|P", target_stage="otm10")
    assert a == b != c
    assert order_ref_for_intent(a).startswith("B5P|")


# =============================================================================
# Config validation
# =============================================================================

def test_repo_config_loads_and_is_shadow_or_placeholder():
    cfg = load_b5_config(Path(__file__).resolve().parents[1] / "config" / "bucket5_production.yml")
    # Safety: this repo must not ship with production mode enabled by default.
    assert cfg["mode"] in ("placeholder", "shadow")
    assert cfg["kill_mode"] == "normal"
    assert float(cfg["capital"]["b5_allocated_nav"]) > 0
    rungs = (cfg["policy"]["ladder"]["rungs"])
    assert [r["otm_pct"] for r in rungs] == [0.10, 0.20, 0.30]
    assert all(int(r["quantity_multiplier"]) == 2 for r in rungs)


def test_config_rejects_bad_mode(tmp_path):
    p = tmp_path / "b5.yml"
    p.write_text("bucket5_production:\n  mode: paper\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_b5_config(p)


def test_config_rejects_paper_account(tmp_path):
    p = tmp_path / "b5.yml"
    p.write_text("bucket5_production:\n  mode: shadow\n  account: paper\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_b5_config(p)


# =============================================================================
# GTP extension: option intent emission (isolated working dir)
# =============================================================================

@pytest.fixture()
def b5_sandbox(tmp_path, monkeypatch):
    """Isolated cwd with signal caches + config; returns (run_date, cfg_path)."""
    monkeypatch.chdir(tmp_path)
    run_date = "2026-07-21"
    cache = tmp_path / "data" / "cache" / "bucket5"
    cache.mkdir(parents=True)
    dates = pd.bdate_range(end=run_date, periods=30)
    pd.DataFrame({"date": dates, "close": 18.77}).to_csv(cache / "^VIX.csv", index=False)
    pd.DataFrame({"date": dates, "close": 20.54}).to_csv(cache / "^VIX3M.csv", index=False)
    pd.DataFrame({"date": dates, "close": 6300.0}).to_csv(cache / "^GSPC.csv", index=False)
    cfg_path = tmp_path / "b5.yml"
    cfg_path.write_text(
        """
bucket5_production:
  mode: shadow
  account: live
  strategy_version: B5PROD-1
  capital: {b5_allocated_nav: 156875.0, ramp_factor: 1.0, max_carry_gross_usd: 31375.0}
  policy:
    sleeve_frac: 0.20
    regime: {r_lo: 0.88, r_hi: 1.00, rho_contango: 1.0, rho_backwardation: 2.0, gross_contango: 1.0, gross_backwardation: 0.35}
    cadence: {base_days: 14.0, k_stress: 6.0, min_interval: 2, max_interval: 21}
    ladder:
      buy_dte: 126
      roll_dte: 63
      rungs:
        - {otm_pct: 0.10, per_roll_frac: 0.008, quantity_multiplier: 2}
        - {otm_pct: 0.20, per_roll_frac: 0.008, quantity_multiplier: 2}
        - {otm_pct: 0.30, per_roll_frac: 0.008, quantity_multiplier: 2}
    hedge_budget: {enabled: true, contango_mult: 1.20, stress_mult: 0.85, vix_lo: 14.0, vix_hi: 28.0, vix_calm_boost: 1.10}
    monetize: {profit_tiers: [[3.0, 0.34], [5.0, 0.5], [8.0, 1.0]], vix_tiers: [[45.0, 0.5], [65.0, 1.0]], giveback_frac: 0.35, giveback_min_mult: 2.0, bank_frac: 0.6, rearm: true, runner_frac: 0.15, runner_mult: 12.0}
    redeploy: {sleeve_w_contango: 0.20, sleeve_w_backwardation: 0.65}
  options: {instrument: XSP, exchange: SMART, currency: USD, contract_multiplier: 100.0, right: P, pilot_max_contracts_per_intent: 1, pilot_max_total_open_contracts: 6}
  execution: {require_manual_approval: true, order_ref_prefix: B5P, max_spread_frac_of_mid: 0.25, limit_cross_frac: 0.25, timeout_sec: 120}
  signals:
    vix_csv: data/cache/bucket5/^VIX.csv
    vix3m_csv: data/cache/bucket5/^VIX3M.csv
    spx_csv: data/cache/bucket5/^GSPC.csv
    max_signal_age_days: 5
  kill_mode: normal
  reconcile: {max_residual_usd: 15.69}
  paths: {ledger_events: data/bucket5_ledger/events.jsonl, run_subdir: bucket5_production, live_panel_json: risk_dashboard/data/bucket5_live.json}
""",
        encoding="utf-8",
    )
    return run_date, cfg_path


def _fake_plan() -> pd.DataFrame:
    return pd.DataFrame([
        {"ETF": "UVIX", "Underlying": "SVIX", "sleeve": "volatility_etp_bucket5",
         "gross_target_usd": 48_000.0, "long_usd": -31_936.0, "short_usd": -16_064.0,
         "underlying_target_usd": -31_936.0, "etf_target_usd": -16_064.0,
         "optimal_gross_target_usd": 48_000.0, "optimal_long_usd": -31_936.0,
         "optimal_short_usd": -16_064.0, "optimal_underlying_target_usd": -31_936.0,
         "optimal_etf_target_usd": -16_064.0},
        {"ETF": "TQQQ", "Underlying": "QQQ", "sleeve": "core_leveraged",
         "gross_target_usd": 100_000.0, "long_usd": 50_000.0, "short_usd": -50_000.0,
         "underlying_target_usd": 50_000.0, "etf_target_usd": -50_000.0,
         "optimal_gross_target_usd": 100_000.0, "optimal_long_usd": 50_000.0,
         "optimal_short_usd": -50_000.0, "optimal_underlying_target_usd": 50_000.0,
         "optimal_etf_target_usd": -50_000.0},
    ])


def test_gtp_ext_shadow_emits_intents_without_touching_plan(b5_sandbox):
    from scripts.bucket5_gtp_ext import run_b5_gtp_extension

    run_date, cfg_path = b5_sandbox
    plan = _fake_plan()
    plan_path = Path("data") / "runs" / run_date / "proposed_trades.csv"
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan.to_csv(plan_path, index=False)

    out = run_b5_gtp_extension(
        run_date=run_date, proposed=plan, proposed_paths=[plan_path], config_path=cfg_path,
    )
    assert out is None  # shadow never rewrites the plan
    b5_dir = Path("data") / "runs" / run_date / "bucket5_production"
    manifest = json.loads((b5_dir / "decision_manifest.json").read_text(encoding="utf-8"))
    assert manifest["mode"] == "shadow"
    assert manifest["health"] == "green"
    assert manifest["carry"]["carry_gross_usd"] <= 31_375.0 + 1e-6
    intents = pd.read_csv(b5_dir / "option_intents.csv")
    assert len(intents) == 3  # one entry intent per rung, no lots yet
    assert set(intents["rung_id"]) == {"otm10", "otm20", "otm30"}
    assert (intents["instrument"] == "XSP").all()
    assert (intents["order_ref_prefix"] == "B5P").all()
    assert intents["intent_id"].is_unique
    # Sibling plan artifact exists for the rebalancer.
    assert (Path("data") / "runs" / run_date / "proposed_trades_b5_options.csv").exists()
    # Plan CSV untouched in shadow.
    reread = pd.read_csv(plan_path)
    assert "b5_owner" not in reread.columns
    # Ledger recorded every intent exactly once; re-run stays idempotent.
    led = Bucket5Ledger(Path("data") / "bucket5_ledger" / "events.jsonl")
    n_before = len(led.load_events())
    run_b5_gtp_extension(
        run_date=run_date, proposed=plan, proposed_paths=[plan_path], config_path=cfg_path,
    )
    assert len(led.load_events()) == n_before


def test_gtp_ext_production_overrides_carry_and_stamps_owner(b5_sandbox):
    from scripts.bucket5_gtp_ext import run_b5_gtp_extension

    run_date, cfg_path = b5_sandbox
    cfg_path.write_text(cfg_path.read_text(encoding="utf-8").replace("mode: shadow", "mode: production"), encoding="utf-8")
    plan = _fake_plan()
    plan_path = Path("data") / "runs" / run_date / "proposed_trades.csv"
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan.to_csv(plan_path, index=False)

    out = run_b5_gtp_extension(
        run_date=run_date, proposed=plan, proposed_paths=[plan_path], config_path=cfg_path,
    )
    assert out is not None
    reread = pd.read_csv(plan_path)
    b5_row = reread[reread["ETF"] == "UVIX"].iloc[0]
    assert b5_row["b5_owner"] == "production"
    # Policy-sized: ratio 18.77/20.54 = 0.9138 -> gross < calm ceiling.
    manifest = json.loads(
        (Path("data") / "runs" / run_date / "bucket5_production" / "decision_manifest.json").read_text(encoding="utf-8")
    )
    carry = manifest["carry"]
    assert b5_row["gross_target_usd"] == pytest.approx(carry["carry_gross_usd"], rel=1e-9)
    assert abs(b5_row["short_usd"]) == pytest.approx(carry["uvix_short_usd"], rel=1e-9)
    assert abs(b5_row["long_usd"]) == pytest.approx(carry["svix_short_usd"], rel=1e-9)
    # Non-B5 rows untouched.
    tqqq = reread[reread["ETF"] == "TQQQ"].iloc[0]
    assert tqqq["gross_target_usd"] == pytest.approx(100_000.0)
    assert tqqq["b5_owner"] in ("", None) or (isinstance(tqqq["b5_owner"], float) and math.isnan(tqqq["b5_owner"]))


def test_gtp_ext_stale_signals_fail_closed(b5_sandbox):
    from scripts.bucket5_gtp_ext import run_b5_gtp_extension

    run_date, cfg_path = b5_sandbox
    # Rewrite caches ending 10 business days before the run date.
    cache = Path("data") / "cache" / "bucket5"
    old = pd.bdate_range(end=pd.Timestamp(run_date) - pd.tseries.offsets.BDay(10), periods=20)
    for name, px in (("^VIX.csv", 18.77), ("^VIX3M.csv", 20.54), ("^GSPC.csv", 6300.0)):
        pd.DataFrame({"date": old, "close": px}).to_csv(cache / name, index=False)
    plan = _fake_plan()
    out = run_b5_gtp_extension(run_date=run_date, proposed=plan, proposed_paths=[], config_path=cfg_path)
    assert out is None
    manifest = json.loads(
        (Path("data") / "runs" / run_date / "bucket5_production" / "decision_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["health"] == "no_new_risk"
    assert not (Path("data") / "runs" / run_date / "bucket5_production" / "option_intents.csv").exists()


# =============================================================================
# Flex option accounting (contract-safe path)
# =============================================================================

def test_flex_option_books_namespace_isolation(tmp_path):
    from scripts.bucket5_flex_options import build_b5_option_books

    trades_xml = tmp_path / "flex_trades.xml"
    trades_xml.write_text(
        """<FlexQueryResponse><FlexStatements><FlexStatement>
        <Trades>
          <Trade assetCategory="OPT" conid="111" symbol="XSP   270115P00570000" underlyingSymbol="XSP"
                 putCall="P" expiry="2027-01-15" strike="570" multiplier="100" tradingClass="XSP"
                 buySell="BUY" quantity="2" tradePrice="9.5" tradeMoney="-1900" ibCommission="-1.3"
                 fifoPnlRealized="0" orderReference="B5P|B5-PUTBUY-20260721-abc123" openCloseIndicator="O"
                 execId="X1" fxRateToBase="1"/>
          <Trade assetCategory="OPT" conid="222" symbol="SPXW  260720P06300000" underlyingSymbol="SPX"
                 putCall="P" expiry="2026-07-20" strike="6300" multiplier="100" tradingClass="SPXW"
                 buySell="SELL" quantity="-1" tradePrice="4.2" tradeMoney="420" ibCommission="-1.1"
                 fifoPnlRealized="55" orderReference="ODTE|whatever" openCloseIndicator="C"
                 execId="X2" fxRateToBase="1"/>
          <Trade assetCategory="STK" conid="333" symbol="UVIX" buySell="SELL" quantity="-100"
                 tradePrice="60" tradeMoney="-6000" orderReference="ETF_LS|x" fxRateToBase="1"/>
        </Trades></FlexStatement></FlexStatements></FlexQueryResponse>""",
        encoding="utf-8",
    )
    pos_xml = tmp_path / "flex_positions.xml"
    pos_xml.write_text(
        """<FlexQueryResponse><FlexStatements><FlexStatement>
        <OpenPositions>
          <OpenPosition assetCategory="OPT" conid="111" symbol="XSP   270115P00570000" underlyingSymbol="XSP"
                        putCall="P" expiry="2027-01-15" strike="570" multiplier="100" tradingClass="XSP"
                        position="2" markPrice="10.4" positionValue="2080" costBasisMoney="1900" fxRateToBase="1"/>
          <OpenPosition assetCategory="OPT" conid="999" symbol="SPX  270115P05000000" underlyingSymbol="SPX"
                        putCall="P" expiry="2027-01-15" strike="5000" multiplier="100" tradingClass="SPX"
                        position="1" markPrice="50" positionValue="5000" costBasisMoney="4800" fxRateToBase="1"/>
          <OpenPosition assetCategory="STK" conid="444" symbol="SVIX" position="-500"
                        markPrice="41.4" positionValue="-20700" fxRateToBase="1"/>
        </OpenPositions></FlexStatement></FlexStatements></FlexQueryResponse>""",
        encoding="utf-8",
    )
    outdir = tmp_path / "accounting"
    summary = build_b5_option_books(
        flex_trades_path=trades_xml, flex_positions_path=pos_xml, outdir=outdir,
        ledger_conids={"111"},  # only the B5 ledger lot
    )
    # Only the B5P| trade and the ledger-matched position are attributed to B5.
    assert summary["n_b5_option_trades"] == 1
    assert summary["n_b5_option_positions"] == 1
    assert summary["open_contracts"] == 2
    assert summary["put_mark_value_usd"] == pytest.approx(2080.0)
    assert summary["non_b5_option_trades_excluded"] == 1     # the 0DTE SPXW trade
    assert summary["non_b5_option_positions_excluded"] == 1  # the foreign SPX lot
    trades = pd.read_csv(outdir / "b5_option_trades.csv")
    assert trades.iloc[0]["conId"] == 111
    assert str(trades.iloc[0]["orderReference"]).startswith("B5P|")


def test_stock_parsers_exclude_option_rows(tmp_path):
    from ibkr_accounting import parse_open_positions, parse_trade_events

    pos_xml = tmp_path / "flex_positions.xml"
    pos_xml.write_text(
        """<FlexQueryResponse><FlexStatements><FlexStatement>
        <OpenPositions>
          <OpenPosition assetCategory="OPT" conid="111" symbol="XSP   270115P00570000"
                        position="2" markPrice="10.4" positionValue="2080" fxRateToBase="1"/>
          <OpenPosition assetCategory="STK" conid="444" symbol="SVIX" position="-500"
                        markPrice="41.4" positionValue="-20700" fxRateToBase="1"/>
        </OpenPositions></FlexStatement></FlexStatements></FlexQueryResponse>""",
        encoding="utf-8",
    )
    pos = parse_open_positions(pos_xml)
    assert list(pos["symbol"]) == ["SVIX"]

    trades_xml = tmp_path / "flex_trades.xml"
    trades_xml.write_text(
        """<FlexQueryResponse><FlexStatements><FlexStatement>
        <Trades>
          <Trade assetCategory="OPT" symbol="XSP   270115P00570000" buySell="BUY" quantity="2"
                 tradePrice="9.5" orderReference="B5P|x" fxRateToBase="1" dateTime="2026-07-21"/>
          <Trade assetCategory="STK" symbol="UVIX" buySell="SELL" quantity="-100"
                 tradePrice="60" orderReference="ETF_LS|x" fxRateToBase="1" dateTime="2026-07-21"/>
        </Trades></FlexStatement></FlexStatements></FlexQueryResponse>""",
        encoding="utf-8",
    )
    ev = parse_trade_events(trades_xml)
    assert list(ev["symbol"]) == ["UVIX"]
