"""Tests for `tax_router.py`.

Covers:
* `TaxConfig.from_dict` parsing & defaults
* `classify_trim` decision tree:
    - no_lot_data path
    - GAIN with prefer_lt: full LT inventory -> lt_only_trim with full qty
    - GAIN with prefer_lt: partial LT inventory -> limited qty
    - GAIN no LT inventory -> pure_trim
    - GAIN with prefer_lt=False -> pure_trim
    - LOSS with substitution disabled -> pure_trim
    - LOSS below min -> pure_trim with note
    - LOSS no pool -> skip
    - LOSS with substitute -> harvest_via_sub
* `make_tax_router` end-to-end:
    - GROW trades pass through unchanged
    - TRIM gain with prefer_lt limits qty
    - TRIM loss spawns paired BUY-substitute trade
    - TRIM loss with no price for substitute -> skips (drops original SELL)
    - Decisions are back-annotated with tax fields
* `record_completed_swaps_from_fills`:
    - Records swap when both legs filled
    - Skips when one leg unfilled
    - No-op when sub_engine None
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from phase2b_resize import ResizeDecision
from substitution_engine import SubstitutionConfig, SubstitutionEngine
from tax_lot_view import Lot, LotView
from tax_router import (
    TaxConfig,
    TrimClassification,
    classify_trim,
    make_tax_router,
    record_completed_swaps_from_fills,
)


# -----------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------

ASOF = date(2026, 5, 11)


def _lot(qty, basis, open_date, side="long", lot_id=None):
    return Lot(
        symbol="X", side=side, qty=float(qty),
        open_date=open_date, cost_basis_per_share=float(basis),
        lot_id=lot_id or f"L_{open_date.isoformat()}_{basis}",
    )


def _view(lots, sym="X"):
    return LotView(symbol=sym, lots=list(lots), asof=ASOF)


def _tax_cfg(**kw) -> TaxConfig:
    base = dict(
        enabled=True, lot_method_assumed="HIFO",
        prefer_long_term_lots=True, st_lt_holding_days=365,
    )
    base.update(kw)
    return TaxConfig(**base)


def _sub_engine(tmp_path: Path, pool=None, enabled=True, min_loss=500.0) -> SubstitutionEngine:
    cfg = SubstitutionConfig.from_dict({
        "enabled": enabled,
        "min_loss_usd_to_substitute": min_loss,
        "underlyings": pool or {"IBIT": ["FBTC", "BITB"]},
    })
    return SubstitutionEngine(
        config=cfg,
        state_path=tmp_path / "active.json",
    )


# -----------------------------------------------------------------------
# TaxConfig
# -----------------------------------------------------------------------

class TestTaxConfig:

    def test_from_dict_defaults(self):
        c = TaxConfig.from_dict({})
        assert c.enabled is True
        assert c.lot_method_assumed == "HIFO"
        assert c.prefer_long_term_lots is True
        assert c.st_lt_holding_days == 365

    def test_from_dict_uppercases_lot_method(self):
        c = TaxConfig.from_dict({"lot_method_assumed": "fifo"})
        assert c.lot_method_assumed == "FIFO"


# -----------------------------------------------------------------------
# classify_trim
# -----------------------------------------------------------------------

class TestClassifyTrim:

    def test_no_lot_data_yields_pure_trim(self):
        cls = classify_trim(
            symbol="X", qty=10, current_px=100, leg_side="long_under",
            lot_view=None, tax_cfg=_tax_cfg(), sub_engine=None,
        )
        assert cls.action == "pure_trim"
        assert cls.qty == 10
        assert "no_lot_data" in cls.note

    def test_empty_lot_view_yields_pure_trim(self):
        cls = classify_trim(
            symbol="X", qty=10, current_px=100, leg_side="long_under",
            lot_view=_view([]), tax_cfg=_tax_cfg(), sub_engine=None,
        )
        assert cls.action == "pure_trim"

    # ---- GAIN paths ------------------------------------------------------

    def test_gain_with_full_lt_inventory_yields_lt_only(self):
        # All LT, gain at current_px=110 vs basis=80
        v = _view([_lot(20, 80, date(2024, 1, 1))])
        cls = classify_trim(
            symbol="X", qty=10, current_px=110, leg_side="long_under",
            lot_view=v, tax_cfg=_tax_cfg(), sub_engine=None,
        )
        assert cls.action == "lt_only_trim"
        assert cls.qty == 10
        assert cls.est_pnl_usd == pytest.approx(300.0)  # 10*(110-80)
        assert cls.lt_qty == 10
        assert cls.st_qty == 0

    def test_gain_with_partial_lt_inventory_limits_qty(self):
        # 5 LT shares, 10 ST shares; ask to trim 12 -> limited to LT (5)
        v = _view([
            _lot(5,  80,  date(2024, 1, 1), lot_id="LT"),
            _lot(10, 100, date(2026, 4, 1), lot_id="ST"),
        ])
        cls = classify_trim(
            symbol="X", qty=12, current_px=110, leg_side="long_under",
            lot_view=v, tax_cfg=_tax_cfg(), sub_engine=None,
        )
        assert cls.action == "lt_only_trim"
        assert cls.qty == 5
        assert "limited_to_lt_qty=5" in cls.note

    def test_gain_with_no_lt_inventory_yields_pure_trim(self):
        v = _view([_lot(10, 100, date(2026, 4, 1))])
        cls = classify_trim(
            symbol="X", qty=5, current_px=110, leg_side="long_under",
            lot_view=v, tax_cfg=_tax_cfg(), sub_engine=None,
        )
        assert cls.action == "pure_trim"
        assert cls.qty == 5
        assert "no_lt_inventory" in cls.note

    def test_gain_with_prefer_lt_disabled_yields_pure_trim(self):
        v = _view([_lot(20, 80, date(2024, 1, 1))])  # plenty of LT
        cls = classify_trim(
            symbol="X", qty=10, current_px=110, leg_side="long_under",
            lot_view=v, tax_cfg=_tax_cfg(prefer_long_term_lots=False),
            sub_engine=None,
        )
        assert cls.action == "pure_trim"
        assert "no_lt_preference" in cls.note

    # ---- LOSS paths ------------------------------------------------------

    def test_loss_no_sub_engine_yields_pure_trim(self):
        v = _view([_lot(10, 120, date(2026, 4, 1))])  # high basis -> loss
        cls = classify_trim(
            symbol="X", qty=10, current_px=100, leg_side="long_under",
            lot_view=v, tax_cfg=_tax_cfg(), sub_engine=None,
        )
        assert cls.action == "pure_trim"
        assert "substitution_disabled" in cls.note

    def test_loss_below_min_yields_pure_trim(self, tmp_path):
        # Loss = 10*(100-105) = -50, below 500 floor
        v = _view([_lot(10, 105, date(2026, 4, 1))])
        cls = classify_trim(
            symbol="IBIT", qty=10, current_px=100, leg_side="long_under",
            lot_view=v, tax_cfg=_tax_cfg(),
            sub_engine=_sub_engine(tmp_path, min_loss=500),
        )
        assert cls.action == "pure_trim"
        assert "loss_below_min" in cls.note

    def test_loss_no_pool_yields_skip(self, tmp_path):
        # IBIT has no pool configured here -> skip
        v = _view([_lot(100, 120, date(2026, 4, 1))])  # large loss
        eng = _sub_engine(tmp_path, pool={"OTHER": ["FOO"]})
        cls = classify_trim(
            symbol="IBIT", qty=100, current_px=100, leg_side="long_under",
            lot_view=v, tax_cfg=_tax_cfg(), sub_engine=eng,
        )
        assert cls.action == "skip"
        assert "no_substitute_for_loss_trim" in cls.note

    def test_loss_with_substitute_yields_harvest(self, tmp_path):
        v = _view([_lot(100, 120, date(2026, 4, 1))])  # 100*(100-120)=-2000 loss
        eng = _sub_engine(tmp_path, pool={"IBIT": ["FBTC", "BITB"]})
        cls = classify_trim(
            symbol="IBIT", qty=100, current_px=100, leg_side="long_under",
            lot_view=v, tax_cfg=_tax_cfg(), sub_engine=eng,
        )
        assert cls.action == "harvest_via_sub"
        assert cls.substitute == "FBTC"
        assert cls.est_pnl_usd == pytest.approx(-2000.0)

    def test_qty_capped_at_inventory(self):
        v = _view([_lot(5, 100, date(2026, 4, 1))])  # only 5 shares
        cls = classify_trim(
            symbol="X", qty=20, current_px=100, leg_side="long_under",
            lot_view=v, tax_cfg=_tax_cfg(), sub_engine=None,
        )
        # qty capped at 5; pure_trim path with 0 P&L
        assert cls.qty == 5


# -----------------------------------------------------------------------
# make_tax_router end-to-end transform
# -----------------------------------------------------------------------

def _grow_trade():
    return {
        "underlying": "AAPL", "etf": "AAPU",
        "leg_side": "short_etf", "symbol": "AAPU",
        "action": "SELL", "qty": 10, "ref_price": 100.0,
        "trade_usd_target": 1000.0, "decision": "grow",
        "reason": "...", "target_usd": -5000.0, "current_usd": -4000.0,
    }


def _trim_trade(symbol="IBIT", qty=100, action="SELL", leg="long_under", target=10000):
    return {
        "underlying": "IBIT" if leg == "long_under" else "AAPL",
        "etf":        "" if leg == "long_under" else symbol,
        "leg_side":   leg,
        "symbol":     symbol,
        "action":     action,
        "qty":        qty,
        "ref_price":  100.0,
        "trade_usd_target": qty * 100.0,
        "decision":   "trim",
        "reason":     "abs_drift=...",
        "target_usd": target,
        "current_usd": target + qty * 100.0,
    }


def _decision_for(t):
    return ResizeDecision(
        underlying=t["underlying"], etf=t.get("etf", ""),
        leg_side=t["leg_side"],
        target_usd=t.get("target_usd", 0.0),
        current_usd=t.get("current_usd", 0.0),
        decision=t["decision"], reason=t["reason"],
        action=t.get("action"), qty=t["qty"],
        ref_price=t["ref_price"], trade_usd=t.get("trade_usd_target", 0.0),
    )


class TestMakeTaxRouter:

    def test_disabled_router_passes_through(self):
        t = _trim_trade()
        d = _decision_for(t)
        router = make_tax_router(
            lot_views={}, prices={"IBIT": 100.0},
            tax_cfg=_tax_cfg(enabled=False), sub_engine=None,
        )
        out_trades, out_dec = router([t], [d])
        assert out_trades == [t]
        assert out_dec == [d]

    def test_grow_trades_pass_through(self, tmp_path):
        t = _grow_trade()
        d = _decision_for(t)
        router = make_tax_router(
            lot_views={}, prices={"AAPU": 100.0},
            tax_cfg=_tax_cfg(), sub_engine=_sub_engine(tmp_path),
        )
        out_trades, _ = router([t], [d])
        assert out_trades == [t]

    def test_gain_trim_with_full_lt_limits_qty(self):
        v = _view([_lot(200, 80, date(2024, 1, 1))], sym="IBIT")
        t = _trim_trade(symbol="IBIT", qty=100)
        d = _decision_for(t)
        router = make_tax_router(
            lot_views={"IBIT": v}, prices={"IBIT": 100.0},
            tax_cfg=_tax_cfg(), sub_engine=None,
        )
        out_trades, out_dec = router([t], [d])
        assert len(out_trades) == 1
        assert out_trades[0]["qty"] == 100  # all LT available
        assert out_dec[0].decision == "lt_only_trim"
        assert out_dec[0].est_realized_pnl_usd == pytest.approx(2000.0)

    def test_loss_trim_emits_paired_swap(self, tmp_path):
        v = _view([_lot(100, 120, date(2026, 4, 1))], sym="IBIT")  # loss = -2000
        t = _trim_trade(symbol="IBIT", qty=100)
        d = _decision_for(t)
        eng = _sub_engine(tmp_path, pool={"IBIT": ["FBTC"]})
        router = make_tax_router(
            lot_views={"IBIT": v},
            prices={"IBIT": 100.0, "FBTC": 50.0},
            tax_cfg=_tax_cfg(), sub_engine=eng,
        )
        out_trades, out_dec = router([t], [d])
        # Two trades: SELL IBIT (original) + BUY FBTC (substitute)
        assert len(out_trades) == 2
        sell, buy = out_trades
        assert sell["symbol"] == "IBIT"
        assert sell["action"] == "SELL"
        assert sell["decision"] == "harvest_sub_sell"
        assert sell["swap_with"] == "FBTC"
        assert buy["symbol"] == "FBTC"
        assert buy["action"] == "BUY"
        assert buy["decision"] == "harvest_sub_buy"
        assert buy["substitute_of"] == "IBIT"
        # Substitute notional matches: 100 * 100 / 50 = 200 sh
        assert buy["qty"] == 200
        # Decisions: original updated, plus a synthetic BUY decision appended
        assert len(out_dec) == 2
        assert out_dec[0].decision == "harvest_sub_sell"
        assert out_dec[0].swap_with == "FBTC"
        assert out_dec[0].harvested_loss_usd == pytest.approx(2000.0)
        assert out_dec[1].substitute_of == "IBIT"

    def test_loss_trim_no_substitute_price_skips(self, tmp_path):
        v = _view([_lot(100, 120, date(2026, 4, 1))], sym="IBIT")
        t = _trim_trade(symbol="IBIT", qty=100)
        d = _decision_for(t)
        eng = _sub_engine(tmp_path, pool={"IBIT": ["FBTC"]})
        router = make_tax_router(
            lot_views={"IBIT": v},
            prices={"IBIT": 100.0},  # FBTC absent
            tax_cfg=_tax_cfg(), sub_engine=eng,
        )
        out_trades, out_dec = router([t], [d])
        # No swap emitted; original SELL also dropped
        assert out_trades == []
        assert out_dec[0].decision == "skip"
        assert "sub_no_price" in out_dec[0].reason

    def test_long_under_trade_matches_decision_with_comma_joined_etf(self, tmp_path):
        """Regression: build_resize_trades emits long_under trades with
        etf="" but decisions with etf="<comma-joined ETF list>". The router
        must fall back to (underlying, leg_side) lookup so the original
        decision row gets back-annotated with tax fields.
        """
        v = _view([_lot(100, 120, date(2026, 4, 1))], sym="IBIT")
        # Trade with etf="" (matches build_resize_trades long_under output)
        t = _trim_trade(symbol="IBIT", qty=100)
        t["etf"] = ""
        # Decision with etf="BITX,BITU" (matches build_resize_trades long_under decision)
        d = _decision_for(t)
        d.etf = "BITX,BITU"

        eng = _sub_engine(tmp_path, pool={"IBIT": ["FBTC"]})
        router = make_tax_router(
            lot_views={"IBIT": v},
            prices={"IBIT": 100.0, "FBTC": 50.0},
            tax_cfg=_tax_cfg(), sub_engine=eng,
        )
        out_trades, out_dec = router([t], [d])
        # Original decision should be transformed even though etf strings differ
        assert d.decision == "harvest_sub_sell"
        assert d.swap_with == "FBTC"
        assert d.harvested_loss_usd == pytest.approx(2000.0)
        # Output should contain SELL + BUY substitute pair
        assert len(out_trades) == 2

    def test_skip_decision_drops_trade(self, tmp_path):
        # Loss large, sub engine has no pool for this symbol -> classify -> skip
        v = _view([_lot(100, 120, date(2026, 4, 1))], sym="IBIT")
        t = _trim_trade(symbol="IBIT", qty=100)
        d = _decision_for(t)
        eng = _sub_engine(tmp_path, pool={"OTHER": ["FOO"]})
        router = make_tax_router(
            lot_views={"IBIT": v}, prices={"IBIT": 100.0},
            tax_cfg=_tax_cfg(), sub_engine=eng,
        )
        out_trades, out_dec = router([t], [d])
        assert out_trades == []
        assert out_dec[0].decision == "skip"
        assert "no_substitute" in out_dec[0].reason


# -----------------------------------------------------------------------
# record_completed_swaps_from_fills
# -----------------------------------------------------------------------

class TestRecordCompletedSwaps:

    def test_records_swap_when_both_legs_filled(self, tmp_path):
        eng = _sub_engine(tmp_path, pool={"IBIT": ["FBTC"]})
        sell_trade = {
            "underlying": "IBIT", "etf": "", "leg_side": "long_under",
            "symbol": "IBIT", "action": "SELL", "qty": 100,
            "ref_price": 100.0, "swap_with": "FBTC", "substitute_of": "",
            "decision": "harvest_sub_sell",
        }
        buy_trade = {
            "underlying": "IBIT", "etf": "", "leg_side": "long_under",
            "symbol": "FBTC", "action": "BUY", "qty": 200,
            "ref_price": 50.0, "swap_with": "IBIT", "substitute_of": "IBIT",
            "decision": "harvest_sub_buy",
        }
        fills = [
            {"underlying": "IBIT", "etf": "", "filled_sh_under": -100, "filled_sh_etf": 0},
            {"underlying": "IBIT", "etf": "FBTC", "filled_sh_under": 0, "filled_sh_etf": 200},
        ]
        # Note: in our test fixture, FBTC fill goes through "etf" column
        # because the BUY-substitute trade has etf="" but record_completed_swaps
        # looks up by symbol. Let's adjust the fill schema to match what
        # execute_resize_serial produces.
        fills = [
            {
                "underlying": "IBIT", "etf": "",
                "filled_sh_under": -100, "filled_sh_etf": 0,
            },
            {
                # The BUY-substitute fill: leg_side=long_under so etf="" but symbol="FBTC".
                # The fill record would use symbol via the underlying column for long legs
                # OR via etf for short legs. For long swap, fill goes under "underlying" column.
                "underlying": "FBTC", "etf": "",
                "filled_sh_under": 200, "filled_sh_etf": 0,
            },
        ]
        n = record_completed_swaps_from_fills([sell_trade, buy_trade], fills, eng)
        assert n == 1
        active = eng.active_substitutions()
        assert "IBIT" in active
        assert active["IBIT"].substitute == "FBTC"
        assert active["IBIT"].qty == 200

    def test_skips_when_one_leg_unfilled(self, tmp_path):
        eng = _sub_engine(tmp_path, pool={"IBIT": ["FBTC"]})
        sell_trade = {
            "underlying": "IBIT", "etf": "", "leg_side": "long_under",
            "symbol": "IBIT", "action": "SELL", "qty": 100,
            "swap_with": "FBTC", "substitute_of": "",
        }
        buy_trade = {
            "underlying": "IBIT", "etf": "", "leg_side": "long_under",
            "symbol": "FBTC", "action": "BUY", "qty": 200,
            "swap_with": "IBIT", "substitute_of": "IBIT",
        }
        # Only SELL filled; BUY didn't
        fills = [
            {"underlying": "IBIT", "etf": "", "filled_sh_under": -100, "filled_sh_etf": 0},
        ]
        n = record_completed_swaps_from_fills([sell_trade, buy_trade], fills, eng)
        assert n == 0
        assert eng.active_substitutions() == {}

    def test_no_op_when_engine_none(self):
        n = record_completed_swaps_from_fills([], [], None)
        assert n == 0

    def test_no_op_when_no_swap_trades(self, tmp_path):
        eng = _sub_engine(tmp_path)
        plain_trade = {
            "underlying": "AAPL", "etf": "AAPU", "leg_side": "short_etf",
            "symbol": "AAPU", "action": "SELL", "qty": 10,
            # no swap_with marker
        }
        fills = [{"underlying": "AAPL", "etf": "AAPU",
                  "filled_sh_under": 0, "filled_sh_etf": -10}]
        n = record_completed_swaps_from_fills([plain_trade], fills, eng)
        assert n == 0
