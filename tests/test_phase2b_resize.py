"""Tests for Phase 2b resize logic in `phase2b_resize.py`.

Covers:

* `_band_decide` pure decision: deadband, hysteresis (enter vs exit
  band), absolute floor (min_trim_usd / min_grow_usd), direction per
  leg side, sign-mismatch guard.
* `build_resize_trades` integration: per-underlying iteration, qty
  capping (cannot SELL more long than held; cannot BTC more than short
  position), purgatory + flow ETF exclusion, no-price skip,
  `skip_underlyings` (newly-established pairs), tax_router hook.

These tests exercise pure logic only — no IBKR / order-routing.
"""

from __future__ import annotations

import pandas as pd
import pytest

from phase2b_resize import (
    ResizeBandConfig,
    ResizeDecision,
    _band_decide,
    build_resize_trades,
)


# -------------------------------------------------------------------
# Pure decision: _band_decide
# -------------------------------------------------------------------

def _cfg(**kw) -> ResizeBandConfig:
    base = dict(
        enabled=True,
        enter_band_pct=0.15,
        exit_band_pct=0.05,
        min_trim_usd=250.0,
        min_grow_usd=250.0,
    )
    base.update(kw)
    return ResizeBandConfig(**base)


class TestBandDecideDeadband:
    """Within enter band -> skip with reason 'within_enter_band'."""

    def test_long_at_target_skips(self):
        d = _band_decide(side="long_under", target_usd=10_000, current_usd=10_000, cfg=_cfg())
        assert d.decision == "skip"
        assert d.reason == "within_enter_band"
        assert d.action is None

    def test_long_within_pct_band_skips(self):
        # 10% drift on 10k -> $1000, < 15% enter band ($1500)
        d = _band_decide(side="long_under", target_usd=10_000, current_usd=11_000, cfg=_cfg())
        assert d.decision == "skip"

    def test_short_within_pct_band_skips(self):
        # short target -5000, current -5500 (10% over)
        d = _band_decide(side="short_etf", target_usd=-5_000, current_usd=-5_500, cfg=_cfg())
        assert d.decision == "skip"


class TestBandDecideAbsoluteFloor:
    """Floor protects small B4 pairs where pct band falls below floor."""

    def test_small_pair_pct_below_floor_uses_floor(self):
        # 1k pair * 15% = 150, but min_trim_usd = 250 -> effective floor = 250
        # drift of 200 should NOT trigger
        d = _band_decide(side="long_under", target_usd=1_000, current_usd=1_200, cfg=_cfg())
        assert d.decision == "skip", f"unexpected: {d}"
        assert d.enter_threshold_usd == 250.0

    def test_small_pair_drift_above_floor_triggers(self):
        # target=1000 (floor=250 dominates pct band).
        # drift=700 > enter_thr 250; trade_usd = 700 - exit_thr(250) = 450 > floor 250.
        d = _band_decide(side="long_under", target_usd=1_000, current_usd=1_700, cfg=_cfg())
        assert d.decision == "trim"
        assert d.action == "SELL"
        assert pytest.approx(d.trade_usd, rel=1e-6) == 450.0


class TestBandDecideHysteresis:
    """Trade only when |drift| > enter; reduce to exit_thr."""

    def test_long_over_target_triggers_trim(self):
        # 20% drift on 10k -> 2000, > 15% enter (1500); exit_thr = 5% -> 500
        d = _band_decide(side="long_under", target_usd=10_000, current_usd=12_000, cfg=_cfg())
        assert d.decision == "trim"
        assert d.action == "SELL"
        # trade_usd = 2000 - 500 = 1500
        assert pytest.approx(d.trade_usd, rel=1e-6) == 1500.0

    def test_long_under_target_triggers_grow(self):
        d = _band_decide(side="long_under", target_usd=10_000, current_usd=8_000, cfg=_cfg())
        assert d.decision == "grow"
        assert d.action == "BUY"
        assert pytest.approx(d.trade_usd, rel=1e-6) == 1500.0

    def test_short_over_target_triggers_btc_trim(self):
        # short tgt -10k, current -12k -> over-shorted; need to BUY back
        d = _band_decide(side="short_etf", target_usd=-10_000, current_usd=-12_000, cfg=_cfg())
        assert d.decision == "trim"
        assert d.action == "BUY"

    def test_short_under_target_triggers_grow_short(self):
        # short tgt -10k, current -8k -> under-shorted; need to SELL more
        d = _band_decide(side="short_etf", target_usd=-10_000, current_usd=-8_000, cfg=_cfg())
        assert d.decision == "grow"
        assert d.action == "SELL"


class TestBandDecideSignMismatch:
    """Sign mismatch between target and current -> skip with explicit reason."""

    def test_long_target_short_position_mismatch(self):
        d = _band_decide(side="long_under", target_usd=5_000, current_usd=-5_000, cfg=_cfg())
        assert d.decision == "skip"
        assert "sign_mismatch" in d.reason

    def test_short_target_long_position_mismatch(self):
        d = _band_decide(side="short_etf", target_usd=-5_000, current_usd=5_000, cfg=_cfg())
        assert d.decision == "skip"
        assert "sign_mismatch" in d.reason


class TestBandDecideFloorOnComputedTrade:
    """Even when drift exceeds enter, the residual trade_usd must beat the floor."""

    def test_marginal_drift_just_above_enter_below_floor_skips(self):
        # Pick numbers where (drift - exit_thr) < floor, with drift > enter_thr.
        # target=10k, enter=1500, exit=500, floor=250
        # If drift = 1600 (just above enter) -> trade_usd = 1100, well above 250 -> trade.
        # Need a configuration where (drift - exit_thr) < floor.
        # Use small target so floors dominate.
        # target=2000, pct enter=300, floor=250, exit_thr=max(100,250)=250
        # drift=400 -> > floor(250) -> triggers; trade_usd = 400-250 = 150 < floor -> SKIP
        d = _band_decide(side="long_under", target_usd=2_000, current_usd=2_400, cfg=_cfg())
        assert d.decision == "skip"
        assert "floor" in d.reason


# -------------------------------------------------------------------
# build_resize_trades integration
# -------------------------------------------------------------------

def _hedgeable(rows):
    return pd.DataFrame(rows)


class TestBuildResizeTrades:

    def test_empty_plan_returns_empty(self):
        trades, decisions = build_resize_trades(
            hedgeable_plan=pd.DataFrame(),
            strat_pos={},
            prices={},
            purgatory_etfs=set(),
            flow_etfs=set(),
            cfg=_cfg(),
        )
        assert trades == []
        assert decisions == []

    def test_within_band_emits_skip_decisions_no_trades(self):
        plan = _hedgeable([
            {"Underlying": "AAPL", "ETF": "AAPU",
             "long_usd": 10_000, "short_usd": -5_000},
        ])
        # current ~ target -> within band
        strat_pos = {"AAPL": 100, "AAPU": -50}
        prices    = {"AAPL": 100.0, "AAPU": 100.0}

        trades, decisions = build_resize_trades(
            hedgeable_plan=plan, strat_pos=strat_pos, prices=prices,
            purgatory_etfs=set(), flow_etfs=set(), cfg=_cfg(),
        )
        assert trades == []
        # Two legs evaluated -> two decisions, both skip
        assert len(decisions) == 2
        assert all(d.decision == "skip" for d in decisions)

    def test_long_overweight_triggers_sell_under(self):
        plan = _hedgeable([
            {"Underlying": "AAPL", "ETF": "AAPU",
             "long_usd": 10_000, "short_usd": -5_000},
        ])
        # AAPL: current 130 sh * 100 = 13_000 (30% over -> trim)
        strat_pos = {"AAPL": 130, "AAPU": -50}
        prices    = {"AAPL": 100.0, "AAPU": 100.0}

        trades, _ = build_resize_trades(
            hedgeable_plan=plan, strat_pos=strat_pos, prices=prices,
            purgatory_etfs=set(), flow_etfs=set(), cfg=_cfg(),
        )
        long_trades = [t for t in trades if t["leg_side"] == "long_under"]
        assert len(long_trades) == 1
        t = long_trades[0]
        assert t["action"] == "SELL"
        assert t["symbol"] == "AAPL"
        # drift=3000, exit_thr=500 (5% of 10k) -> trade_usd ~ 2500 -> 25 sh @ 100
        assert t["qty"] == 25

    def test_short_overweight_triggers_btc_buy_etf(self):
        plan = _hedgeable([
            {"Underlying": "AAPL", "ETF": "AAPU",
             "long_usd": 10_000, "short_usd": -5_000},
        ])
        # AAPU: current -70 sh * 100 = -7000 (40% over-shorted)
        strat_pos = {"AAPL": 100, "AAPU": -70}
        prices    = {"AAPL": 100.0, "AAPU": 100.0}

        trades, _ = build_resize_trades(
            hedgeable_plan=plan, strat_pos=strat_pos, prices=prices,
            purgatory_etfs=set(), flow_etfs=set(), cfg=_cfg(),
        )
        etf_trades = [t for t in trades if t["leg_side"] == "short_etf"]
        assert len(etf_trades) == 1
        t = etf_trades[0]
        assert t["action"] == "BUY"
        assert t["symbol"] == "AAPU"
        # drift=2000, exit_thr=250 (5% of 5k) -> trade_usd=1750 -> 17 sh
        assert t["qty"] == 17

    def test_sell_long_qty_never_exceeds_holdings(self):
        # On a long-leg trim, qty = floor((current_usd - exit_thr)/px), which by
        # construction is always <= current_sh. Verify this property holds even
        # for a maximally aggressive trim (target_usd ~ 0).
        plan = _hedgeable([
            {"Underlying": "AAPL", "ETF": "AAPU",
             "long_usd": 10, "short_usd": -5_000},
        ])
        strat_pos = {"AAPL": 8, "AAPU": -50}
        prices    = {"AAPL": 100.0, "AAPU": 100.0}
        trades, _ = build_resize_trades(
            hedgeable_plan=plan, strat_pos=strat_pos, prices=prices,
            purgatory_etfs=set(), flow_etfs=set(), cfg=_cfg(),
        )
        long_trades = [t for t in trades if t["leg_side"] == "long_under"]
        # If a trim was emitted, its qty must not exceed current shares (8).
        for t in long_trades:
            assert t["qty"] <= 8

    def test_btc_qty_capped_by_short_position(self):
        # Plan wants tiny short, current short much larger -> trim by buying back
        plan = _hedgeable([
            {"Underlying": "AAPL", "ETF": "AAPU",
             "long_usd": 10_000, "short_usd": -10},
        ])
        strat_pos = {"AAPL": 100, "AAPU": -3}  # only 3 shares short
        prices    = {"AAPL": 100.0, "AAPU": 100.0}
        # target=-10, current=-300 -> drift +290 -> trim
        # trade_usd = 290 - 250 = 40 -> 0 shares before cap; qty=0 -> skip
        trades, decisions = build_resize_trades(
            hedgeable_plan=plan, strat_pos=strat_pos, prices=prices,
            purgatory_etfs=set(), flow_etfs=set(), cfg=_cfg(),
        )
        # No trade emitted
        assert all(t["leg_side"] != "short_etf" for t in trades)

    def test_purgatory_etf_may_trim(self):
        plan = _hedgeable([
            {"Underlying": "AAPL", "ETF": "AAPU",
             "long_usd": 10_000, "short_usd": -5_000,
             "model_long_usd": 10_000, "model_short_usd": -5_000,
             "execution_policy": "reduce_only", "purgatory": True},
        ])
        strat_pos = {"AAPL": 130, "AAPU": -70}
        prices    = {"AAPL": 100.0, "AAPU": 100.0}

        trades, decisions = build_resize_trades(
            hedgeable_plan=plan, strat_pos=strat_pos, prices=prices,
            purgatory_etfs={"AAPU"}, flow_etfs=set(), cfg=_cfg(),
        )
        etf_decisions = [d for d in decisions if d.leg_side == "short_etf"]
        assert len(etf_decisions) == 1
        assert etf_decisions[0].decision == "trim"
        assert etf_decisions[0].execution_policy == "reduce_only"
        assert any(t["leg_side"] == "short_etf" and t["action"] == "BUY" for t in trades)

    def test_purgatory_target_above_current_cannot_add_gross(self):
        plan = _hedgeable([
            {"Underlying": "AAPL", "ETF": "AAPU",
             "long_usd": 0, "short_usd": 0,
             "model_long_usd": 20_000, "model_short_usd": -10_000,
             "execution_policy": "reduce_only", "purgatory": True},
        ])
        trades, decisions = build_resize_trades(
            hedgeable_plan=plan,
            strat_pos={"AAPL": 50, "AAPU": -25},
            prices={"AAPL": 100.0, "AAPU": 100.0},
            purgatory_etfs={"AAPU"}, flow_etfs=set(), cfg=_cfg(),
        )
        reduce_decisions = [d for d in decisions if d.execution_policy == "reduce_only"]
        assert reduce_decisions
        assert all(d.pair_allowed_gross_usd <= d.pair_current_gross_usd + 1e-6 for d in reduce_decisions)
        assert not any(t["decision"] == "grow" for t in trades)

    def test_purgatory_allows_hedge_leg_growth_when_pair_gross_falls(self):
        plan = _hedgeable([
            {"Underlying": "AAPL", "ETF": "AAPU",
             "long_usd": 0, "short_usd": 0,
             "model_long_usd": 4_000, "model_short_usd": -4_000,
             "execution_policy": "reduce_only", "purgatory": True},
        ])
        trades, decisions = build_resize_trades(
            hedgeable_plan=plan,
            strat_pos={"AAPL": 10, "AAPU": -100},
            prices={"AAPL": 100.0, "AAPU": 100.0},
            purgatory_etfs={"AAPU"}, flow_etfs=set(),
            cfg=_cfg(enter_band_pct=0.01, exit_band_pct=0.0, min_trim_usd=1, min_grow_usd=1),
        )
        assert any(t["leg_side"] == "long_under" and t["action"] == "BUY" for t in trades)
        assert any(t["leg_side"] == "short_etf" and t["action"] == "BUY" for t in trades)
        assert all(
            d.pair_allowed_gross_usd <= d.pair_current_gross_usd + 1e-6
            for d in decisions if d.execution_policy == "reduce_only"
        )

    def test_flow_etf_skipped(self):
        plan = _hedgeable([
            {"Underlying": "AAPL", "ETF": "AAPU",
             "long_usd": 10_000, "short_usd": -5_000},
        ])
        strat_pos = {"AAPL": 100, "AAPU": -50}
        prices    = {"AAPL": 100.0, "AAPU": 100.0}

        trades, decisions = build_resize_trades(
            hedgeable_plan=plan, strat_pos=strat_pos, prices=prices,
            purgatory_etfs=set(), flow_etfs={"AAPU"}, cfg=_cfg(),
        )
        etf_decisions = [d for d in decisions if d.leg_side == "short_etf"]
        assert len(etf_decisions) == 1
        assert etf_decisions[0].reason == "flow_etf"
        assert all(t["leg_side"] != "short_etf" for t in trades)

    def test_no_price_for_underlying_skips_pair(self):
        plan = _hedgeable([
            {"Underlying": "AAPL", "ETF": "AAPU",
             "long_usd": 10_000, "short_usd": -5_000},
        ])
        strat_pos = {"AAPL": 100, "AAPU": -50}
        prices    = {"AAPU": 100.0}  # no AAPL price

        trades, decisions = build_resize_trades(
            hedgeable_plan=plan, strat_pos=strat_pos, prices=prices,
            purgatory_etfs=set(), flow_etfs=set(), cfg=_cfg(),
        )
        assert trades == []
        assert any(d.reason == "no_price_for_underlying" for d in decisions)

    def test_skip_underlyings_excludes_newly_established(self):
        plan = _hedgeable([
            {"Underlying": "AAPL", "ETF": "AAPU",
             "long_usd": 10_000, "short_usd": -5_000},
        ])
        # Even if drift huge, established_underlyings={"AAPL"} should skip
        strat_pos = {"AAPL": 200, "AAPU": -50}
        prices    = {"AAPL": 100.0, "AAPU": 100.0}

        trades, decisions = build_resize_trades(
            hedgeable_plan=plan, strat_pos=strat_pos, prices=prices,
            purgatory_etfs=set(), flow_etfs=set(), cfg=_cfg(),
            skip_underlyings={"AAPL"},
        )
        assert trades == []
        assert decisions == []  # entire underlying skipped, no rows emitted

    def test_blacklist_excludes_underlying(self):
        plan = _hedgeable([
            {"Underlying": "AAPL", "ETF": "AAPU",
             "long_usd": 10_000, "short_usd": -5_000},
        ])
        strat_pos = {"AAPL": 200, "AAPU": -50}
        prices    = {"AAPL": 100.0, "AAPU": 100.0}

        trades, decisions = build_resize_trades(
            hedgeable_plan=plan, strat_pos=strat_pos, prices=prices,
            purgatory_etfs=set(), flow_etfs=set(),
            blacklist={"AAPL"}, cfg=_cfg(),
        )
        assert trades == []
        assert decisions == []

    def test_multiple_etfs_per_underlying_aggregated(self):
        # Two ETF rows for same underlying; long_usd sums for the under leg
        plan = _hedgeable([
            {"Underlying": "AAPL", "ETF": "AAPU",  "long_usd": 4_000, "short_usd": -2_000},
            {"Underlying": "AAPL", "ETF": "AAPL3", "long_usd": 6_000, "short_usd": -3_000},
        ])
        # Total long target = 10_000; current 100 sh * 100 = 10_000 -> within band
        strat_pos = {"AAPL": 100, "AAPU": -20, "AAPL3": -30}
        prices    = {"AAPL": 100.0, "AAPU": 100.0, "AAPL3": 100.0}

        trades, decisions = build_resize_trades(
            hedgeable_plan=plan, strat_pos=strat_pos, prices=prices,
            purgatory_etfs=set(), flow_etfs=set(), cfg=_cfg(),
        )
        # Under leg should be skipped; ETF legs at exact target -> skipped too
        assert trades == []
        # one long_under decision + two short_etf decisions
        assert sum(1 for d in decisions if d.leg_side == "long_under") == 1
        assert sum(1 for d in decisions if d.leg_side == "short_etf") == 2

    def test_tax_router_hook_invoked(self):
        plan = _hedgeable([
            {"Underlying": "AAPL", "ETF": "AAPU",
             "long_usd": 10_000, "short_usd": -5_000},
        ])
        strat_pos = {"AAPL": 130, "AAPU": -50}
        prices    = {"AAPL": 100.0, "AAPU": 100.0}

        called = {"n": 0}

        def tax_router(trades, decisions):
            called["n"] += 1
            # Drop all trades to verify router can short-circuit
            return [], decisions

        trades, decisions = build_resize_trades(
            hedgeable_plan=plan, strat_pos=strat_pos, prices=prices,
            purgatory_etfs=set(), flow_etfs=set(), cfg=_cfg(),
            tax_router=tax_router,
        )
        assert called["n"] == 1
        assert trades == []  # router dropped them
        assert len(decisions) >= 1


# -------------------------------------------------------------------
# B1 + B4 same-underlying netting
# -------------------------------------------------------------------
#
# Sign convention from generate_trade_plan.py:
#   * core_leveraged / yieldboost (B1, YB):
#       long_usd  > 0  (long underlying notional)
#       short_usd < 0  (short LETF notional)
#   * inverse_decay_bucket4 (B4):
#       long_usd  < 0  (short underlying notional — column name kept for
#                       schema compatibility with B1 plumbing)
#       short_usd < 0  (short inverse-ETF notional)
#
# build_resize_trades groups by Underlying and sums long_usd into a
# single signed underlying-leg target. When Phase 2b is fed a plan that
# includes BOTH B1 and B4 rows for the same Underlying, the underlying
# decision is therefore the *signed net* of the two sleeves.

class TestB1B4Netting:
    """Stage A regression: B1 + B4 same-underlying coexistence.

    Verifies that build_resize_trades, when fed the FULL plan (B1 + YB +
    B4) instead of a B1+YB-only filter, correctly nets the underlying
    leg into a single signed decision while keeping ETF legs independent.
    """

    def test_b1_b4_perfectly_offset_skips_underlying(self):
        # B1 long $10k AAPL underlying + B4 short $10k AAPL underlying.
        # Net target = 0; if we already sit at 0 sh AAPL, no trade.
        plan = _hedgeable([
            # B1: long underlying $10k, short LETF $5k
            {"Underlying": "AAPL", "ETF": "AAPU",
             "long_usd": 10_000, "short_usd": -5_000},
            # B4: short underlying $10k (long_usd negative), short inverse $5k
            {"Underlying": "AAPL", "ETF": "AAPS",
             "long_usd": -10_000, "short_usd": -5_000},
        ])
        # Net flat on the underlying; both ETF shorts already at target.
        strat_pos = {"AAPL": 0, "AAPU": -50, "AAPS": -50}
        prices    = {"AAPL": 100.0, "AAPU": 100.0, "AAPS": 100.0}

        trades, decisions = build_resize_trades(
            hedgeable_plan=plan, strat_pos=strat_pos, prices=prices,
            purgatory_etfs=set(), flow_etfs=set(), cfg=_cfg(),
        )

        # No underlying trade — net target $0, current $0.
        long_under = [t for t in trades if t["leg_side"] == "long_under"]
        assert long_under == [], (
            f"B1+B4 net flat AAPL should not trade underlying; got {long_under}"
        )

        # Exactly one long_under decision (per-underlying), with target_usd=0.
        under_decisions = [d for d in decisions if d.leg_side == "long_under"]
        assert len(under_decisions) == 1
        assert under_decisions[0].target_usd == 0.0
        assert under_decisions[0].decision == "skip"

        # Both ETF legs evaluated independently.
        etf_decisions = [d for d in decisions if d.leg_side == "short_etf"]
        assert {d.etf for d in etf_decisions} == {"AAPU", "AAPS"}

    def test_b1_b4_partial_offset_drives_signed_net_underlying(self):
        # B1 long $10k + B4 short $3k -> net long $7k.
        # Currently long $10k -> over by $3k -> trim long.
        plan = _hedgeable([
            {"Underlying": "AAPL", "ETF": "AAPU",
             "long_usd": 10_000, "short_usd": -5_000},
            {"Underlying": "AAPL", "ETF": "AAPS",
             "long_usd": -3_000, "short_usd": -1_500},
        ])
        strat_pos = {"AAPL": 100, "AAPU": -50, "AAPS": -15}
        prices    = {"AAPL": 100.0, "AAPU": 100.0, "AAPS": 100.0}

        trades, decisions = build_resize_trades(
            hedgeable_plan=plan, strat_pos=strat_pos, prices=prices,
            purgatory_etfs=set(), flow_etfs=set(), cfg=_cfg(),
        )

        long_under = [t for t in trades if t["leg_side"] == "long_under"]
        assert len(long_under) == 1
        t = long_under[0]
        # Net target $7k, current $10k -> drift $3k (~43% of $7k -> over enter
        # band of 15%). exit_thr = max(7000*0.05=350, min_trim=250) = 350.
        # trade_usd = 3000 - 350 = 2650 -> 26 sh @ $100.
        assert t["action"] == "SELL"
        assert t["symbol"] == "AAPL"
        assert t["qty"] == 26
        assert t["target_usd"] == 7_000.0  # signed-net target persisted

        # The underlying decision's target reflects the netted value.
        under_decisions = [d for d in decisions if d.leg_side == "long_under"]
        assert len(under_decisions) == 1
        assert under_decisions[0].target_usd == 7_000.0

    def test_b4_only_produces_short_underlying_target(self):
        # B4 row alone with no B1 -> target is purely short underlying.
        # This validates Phase 2b sees B4 rows at all (Stage A fix).
        plan = _hedgeable([
            {"Underlying": "AAPL", "ETF": "AAPS",
             "long_usd": -10_000, "short_usd": -5_000},
        ])
        # Currently flat AAPL; AAPS short already at target.
        strat_pos = {"AAPL": 0, "AAPU": 0, "AAPS": -50}
        prices    = {"AAPL": 100.0, "AAPS": 100.0}

        trades, decisions = build_resize_trades(
            hedgeable_plan=plan, strat_pos=strat_pos, prices=prices,
            purgatory_etfs=set(), flow_etfs=set(), cfg=_cfg(),
        )

        # Need to short AAPL: target=-$10k, current=$0 -> drift -$10k -> SELL.
        long_under = [t for t in trades if t["leg_side"] == "long_under"]
        assert len(long_under) == 1
        t = long_under[0]
        assert t["action"] == "SELL"
        assert t["symbol"] == "AAPL"
        # exit_thr=max(10000*0.05=500, 250)=500; trade_usd=10000-500=9500 -> 95 sh.
        assert t["qty"] == 95
        assert t["target_usd"] == -10_000.0

    def test_b1_b4_etf_legs_drift_independently(self):
        # B1 + B4: underlying nets to zero (no underlying trade), but
        # both ETF shorts drift over their target -> each trims independently.
        plan = _hedgeable([
            {"Underlying": "AAPL", "ETF": "AAPU",
             "long_usd": 10_000, "short_usd": -5_000},
            {"Underlying": "AAPL", "ETF": "AAPS",
             "long_usd": -10_000, "short_usd": -5_000},
        ])
        # AAPU and AAPS each over-shorted by 40% (-70 sh vs -50 target).
        strat_pos = {"AAPL": 0, "AAPU": -70, "AAPS": -70}
        prices    = {"AAPL": 100.0, "AAPU": 100.0, "AAPS": 100.0}

        trades, decisions = build_resize_trades(
            hedgeable_plan=plan, strat_pos=strat_pos, prices=prices,
            purgatory_etfs=set(), flow_etfs=set(), cfg=_cfg(),
        )

        # No underlying trade.
        assert all(t["leg_side"] != "long_under" for t in trades)

        # Both ETF legs fire BUY-to-cover (BTC trim of over-shorted leg).
        etf_trades = [t for t in trades if t["leg_side"] == "short_etf"]
        assert {t["symbol"] for t in etf_trades} == {"AAPU", "AAPS"}
        for t in etf_trades:
            assert t["action"] == "BUY"
            # drift=2000, exit_thr=250 (5%*5000), trade_usd=1750 -> 17 sh.
            assert t["qty"] == 17

    def test_b4_buy_to_cover_qty_capped_by_short_inventory(self):
        # B4: target=-$5k short AAPL, but currently over-shorted to -$10k.
        # Need BUY-to-cover. Cap should be |current_sh| = 100, not the
        # legacy SELL-only cap which would have left this uncapped.
        plan = _hedgeable([
            {"Underlying": "AAPL", "ETF": "AAPS",
             "long_usd": -5_000, "short_usd": -2_500},
        ])
        # Drift |10k|-|5k| = 5k; exit_thr = max(5%*5k=250, 250) = 250.
        # trade_usd = 5_000 - 250 = 4_750 -> 47 sh @ $100. Cap by |-100|=100. Pass.
        strat_pos = {"AAPL": -100, "AAPS": -25}
        prices    = {"AAPL": 100.0, "AAPS": 100.0}
        trades, _ = build_resize_trades(
            hedgeable_plan=plan, strat_pos=strat_pos, prices=prices,
            purgatory_etfs=set(), flow_etfs=set(), cfg=_cfg(),
        )
        long_under = [t for t in trades if t["leg_side"] == "long_under"]
        assert len(long_under) == 1
        t = long_under[0]
        assert t["action"] == "BUY"
        assert t["qty"] == 47
        # And the qty must never exceed |current short| (100 here).
        assert t["qty"] <= 100

    def test_b4_only_underlying_skipped_when_within_band(self):
        # B4 alone, currently within enter band on the negative target.
        plan = _hedgeable([
            {"Underlying": "AAPL", "ETF": "AAPS",
             "long_usd": -10_000, "short_usd": -5_000},
        ])
        # current = -$11k; target = -$10k; drift = $1k = 10% of |target|
        # < 15% enter band -> skip.
        strat_pos = {"AAPL": -110, "AAPS": -50}
        prices    = {"AAPL": 100.0, "AAPS": 100.0}

        trades, decisions = build_resize_trades(
            hedgeable_plan=plan, strat_pos=strat_pos, prices=prices,
            purgatory_etfs=set(), flow_etfs=set(), cfg=_cfg(),
        )
        assert all(t["leg_side"] != "long_under" for t in trades)
        under_decisions = [d for d in decisions if d.leg_side == "long_under"]
        assert len(under_decisions) == 1
        assert under_decisions[0].decision == "skip"


# -------------------------------------------------------------------
# Config plumbing
# -------------------------------------------------------------------

class TestResizeBandConfig:

    def test_from_dict_defaults(self):
        cfg = ResizeBandConfig.from_dict({})
        assert cfg.enabled is True
        assert cfg.enter_band_pct == 0.15
        assert cfg.exit_band_pct == 0.05
        assert cfg.min_trim_usd == 250.0
        assert cfg.min_grow_usd == 250.0

    def test_from_dict_overrides(self):
        cfg = ResizeBandConfig.from_dict({
            "enabled": False,
            "enter_band_pct": 0.20,
            "exit_band_pct": 0.10,
            "min_trim_usd": 500,
            "min_grow_usd": 100,
        })
        assert cfg.enabled is False
        assert cfg.enter_band_pct == 0.20
        assert cfg.exit_band_pct == 0.10
        assert cfg.min_trim_usd == 500.0
        assert cfg.min_grow_usd == 100.0

    def test_from_dict_none_yields_defaults(self):
        cfg = ResizeBandConfig.from_dict(None)
        assert cfg.enabled is True


@pytest.mark.parametrize(
    ("sleeve", "underlying", "etf", "under_sign"),
    [
        ("core_leveraged", "AAPL", "AAPU", 1),
        ("yieldboost", "MSTR", "MSTY", 1),
        ("inverse_decay_bucket4", "CLSK", "CLSZ", -1),
        ("volatility_etp_bucket5", "SVIX", "UVIX", -1),
    ],
)
def test_reduce_only_invariant_applies_to_all_four_sleeves(
    sleeve, underlying, etf, under_sign
):
    plan = _hedgeable([
        {
            "Underlying": underlying,
            "ETF": etf,
            "sleeve": sleeve,
            "long_usd": 0.0,
            "short_usd": 0.0,
            "model_long_usd": under_sign * 4_000.0,
            "model_short_usd": -2_000.0,
            "execution_policy": "reduce_only",
            "purgatory": True,
        }
    ])
    _, decisions = build_resize_trades(
        hedgeable_plan=plan,
        strat_pos={underlying: under_sign * 80, etf: -40},
        prices={underlying: 100.0, etf: 100.0},
        purgatory_etfs={etf},
        flow_etfs=set(),
        cfg=_cfg(
            enter_band_pct=0.01,
            exit_band_pct=0.0,
            min_trim_usd=1,
            min_grow_usd=1,
        ),
    )
    policy_decisions = [d for d in decisions if d.execution_policy == "reduce_only"]
    assert policy_decisions
    assert all(
        d.pair_allowed_gross_usd <= d.pair_current_gross_usd + 1e-6
        for d in policy_decisions
    )
