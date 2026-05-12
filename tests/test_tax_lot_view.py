"""Tests for `tax_lot_view.py`.

Covers:
* `Lot`, `LotView`, `EstPnL` dataclasses
* `LotView.estimate_realized_pnl` for HIFO/FIFO/LIFO/MAXLOSS, longs and shorts
* `LotView.st_lt_split` boundary
* `prefer_lt` ordering override
* `_parse_flex_date` tolerance for various formats
* `lots_from_flex_rows` dict-row parsing
* `parse_open_position_lots` XML parsing (synthetic XML)
* `load_lot_views_for_run_dir` file lookup behavior
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from textwrap import dedent

import pytest

from tax_lot_view import (
    EstPnL,
    Lot,
    LotView,
    _parse_flex_date,
    load_lot_views_for_run_dir,
    lots_from_flex_rows,
    parse_open_position_lots,
)


def _lot(qty=10, basis=100.0, open_date=date(2025, 1, 1), side="long", lot_id="L1"):
    return Lot(
        symbol="TEST", side=side, qty=float(qty),
        open_date=open_date, cost_basis_per_share=float(basis), lot_id=lot_id,
    )


# -----------------------------------------------------------------------
# LotView aggregate
# -----------------------------------------------------------------------

class TestLotViewAggregate:

    def test_empty_view_total_qty_zero(self):
        v = LotView(symbol="X", lots=[], asof=date(2026, 5, 11))
        assert v.total_qty == 0
        assert v.avg_cost == 0.0

    def test_total_qty_sums_lots(self):
        lots = [_lot(qty=10, basis=100), _lot(qty=20, basis=110, lot_id="L2")]
        v = LotView(symbol="X", lots=lots, asof=date(2026, 5, 11))
        assert v.total_qty == 30

    def test_avg_cost_is_basis_weighted(self):
        lots = [_lot(qty=10, basis=100), _lot(qty=30, basis=200, lot_id="L2")]
        v = LotView(symbol="X", lots=lots, asof=date(2026, 5, 11))
        # weighted: (10*100 + 30*200) / 40 = 175
        assert v.avg_cost == pytest.approx(175.0)


# -----------------------------------------------------------------------
# ST/LT split
# -----------------------------------------------------------------------

class TestLotViewStLtSplit:

    def test_split_boundary_at_365(self):
        asof = date(2026, 5, 11)
        # Open exactly 365 days ago -> LT (older or equal to boundary)
        lots = [
            _lot(qty=10, basis=100, open_date=asof.replace(year=asof.year - 1, month=5, day=11), lot_id="LT1"),
            _lot(qty=15, basis=110, open_date=date(2026, 4, 1), lot_id="ST1"),
        ]
        v = LotView(symbol="X", lots=lots, asof=asof)
        st, lt = v.st_lt_split(st_lt_holding_days=365)
        assert lt == 10
        assert st == 15

    def test_split_no_st_inventory(self):
        asof = date(2026, 5, 11)
        lots = [_lot(qty=20, open_date=date(2024, 1, 1), lot_id="OLD")]
        v = LotView(symbol="X", lots=lots, asof=asof)
        st, lt = v.st_lt_split(365)
        assert (st, lt) == (0, 20)


# -----------------------------------------------------------------------
# estimate_realized_pnl: longs
# -----------------------------------------------------------------------

class TestEstimateLongs:

    def setup_method(self):
        # Three long lots at different prices and dates
        self.asof = date(2026, 5, 11)
        self.lots = [
            Lot("X", "long", 10, date(2024, 1, 1), 90,  "OLD_CHEAP"),    # LT, low basis
            Lot("X", "long", 10, date(2026, 2, 1), 110, "ST_PRICEY"),    # ST, high basis
            Lot("X", "long", 10, date(2026, 4, 1), 100, "ST_MEDIUM"),    # ST, mid
        ]
        self.view = LotView("X", self.lots, asof=self.asof)

    def test_hifo_picks_highest_basis_first(self):
        # Sell 10 @ $105 -> HIFO uses the 110-basis lot first
        est = self.view.estimate_realized_pnl(qty=10, current_px=105, method="HIFO")
        assert est.qty_consumed == 10
        # 10 * (105 - 110) = -50 (loss)
        assert est.realized_pnl_usd == pytest.approx(-50.0)
        assert est.lots_consumed[0][0] == "ST_PRICEY"

    def test_fifo_picks_oldest_first(self):
        est = self.view.estimate_realized_pnl(qty=10, current_px=105, method="FIFO")
        # 10 * (105 - 90) = +150 (gain) — old cheap lot
        assert est.realized_pnl_usd == pytest.approx(150.0)
        assert est.lots_consumed[0][0] == "OLD_CHEAP"

    def test_lifo_picks_newest_first(self):
        est = self.view.estimate_realized_pnl(qty=10, current_px=105, method="LIFO")
        # 10 * (105 - 100) = +50 — newest mid-basis lot
        assert est.realized_pnl_usd == pytest.approx(50.0)
        assert est.lots_consumed[0][0] == "ST_MEDIUM"

    def test_maxloss_picks_highest_basis_relative_to_current(self):
        # Current px = 105; max-loss-per-share lot is the 110-basis one (-5)
        est = self.view.estimate_realized_pnl(qty=10, current_px=105, method="MAXLOSS")
        assert est.realized_pnl_usd == pytest.approx(-50.0)
        assert est.lots_consumed[0][0] == "ST_PRICEY"

    def test_consumes_across_lots(self):
        # Sell 25 -> consumes more than one lot under HIFO
        est = self.view.estimate_realized_pnl(qty=25, current_px=105, method="HIFO")
        assert est.qty_consumed == 25
        assert len(est.lots_consumed) >= 2

    def test_qty_exceeding_inventory_capped(self):
        est = self.view.estimate_realized_pnl(qty=999, current_px=100, method="FIFO")
        assert est.qty_consumed == 30  # all three lots

    def test_no_inventory_returns_zero(self):
        empty = LotView("X", [], asof=self.asof)
        est = empty.estimate_realized_pnl(qty=5, current_px=100, method="HIFO")
        assert est.realized_pnl_usd == 0
        assert est.qty_consumed == 0

    def test_zero_qty_returns_zero(self):
        est = self.view.estimate_realized_pnl(qty=0, current_px=100, method="HIFO")
        assert est.qty_consumed == 0

    def test_st_lt_breakdown_recorded(self):
        # FIFO: oldest lot (LT) consumed first. Sell 5 -> all LT.
        est = self.view.estimate_realized_pnl(qty=5, current_px=100, method="FIFO")
        assert est.lt_qty == 5
        assert est.st_qty == 0
        # Sell 15 -> 10 LT + 5 ST
        est2 = self.view.estimate_realized_pnl(qty=15, current_px=100, method="FIFO")
        assert est2.lt_qty == 10
        assert est2.st_qty == 5


# -----------------------------------------------------------------------
# estimate_realized_pnl: shorts (P&L sign reversed)
# -----------------------------------------------------------------------

class TestEstimateShorts:

    def test_short_profit_when_current_below_basis(self):
        # Short opened at 100, cover at 90 -> profit of 10/share
        lots = [Lot("Y", "short", 10, date(2026, 1, 1), 100, "S1")]
        v = LotView("Y", lots, asof=date(2026, 5, 11))
        est = v.estimate_realized_pnl(qty=10, current_px=90, method="HIFO")
        assert est.realized_pnl_usd == pytest.approx(100.0)

    def test_short_loss_when_current_above_basis(self):
        lots = [Lot("Y", "short", 10, date(2026, 1, 1), 100, "S1")]
        v = LotView("Y", lots, asof=date(2026, 5, 11))
        est = v.estimate_realized_pnl(qty=10, current_px=110, method="HIFO")
        assert est.realized_pnl_usd == pytest.approx(-100.0)

    def test_maxloss_for_short_picks_lowest_basis(self):
        # Two shorts: opened at 100 and 105. Cover at 110. Max loss = lower basis lot.
        # short1 loss = 110-100 = -10/share; short2 loss = 110-105 = -5/share
        lots = [
            Lot("Y", "short", 10, date(2026, 2, 1), 100, "LOW"),
            Lot("Y", "short", 10, date(2026, 3, 1), 105, "HIGH"),
        ]
        v = LotView("Y", lots, asof=date(2026, 5, 11))
        est = v.estimate_realized_pnl(qty=10, current_px=110, method="MAXLOSS")
        # Should pick LOW first (max loss = -10/sh)
        assert est.lots_consumed[0][0] == "LOW"
        assert est.realized_pnl_usd == pytest.approx(-100.0)


# -----------------------------------------------------------------------
# prefer_lt ordering override
# -----------------------------------------------------------------------

class TestPreferLt:

    def test_prefer_lt_consumes_lt_first_even_when_hifo_would_pick_st(self):
        asof = date(2026, 5, 11)
        lots = [
            Lot("X", "long", 10, date(2024, 1, 1), 80,  "LT_LOW"),    # LT, low basis
            Lot("X", "long", 10, date(2026, 4, 1), 120, "ST_HIGH"),   # ST, high basis
        ]
        v = LotView("X", lots, asof=asof)

        # HIFO without prefer_lt: ST_HIGH first -> gain = 10*(100-120) = -200
        est = v.estimate_realized_pnl(qty=10, current_px=100, method="HIFO")
        assert est.lots_consumed[0][0] == "ST_HIGH"

        # HIFO with prefer_lt: LT_LOW first -> gain = 10*(100-80) = +200
        est_lt = v.estimate_realized_pnl(
            qty=10, current_px=100, method="HIFO", prefer_lt=True,
        )
        assert est_lt.lots_consumed[0][0] == "LT_LOW"
        assert est_lt.realized_pnl_usd == pytest.approx(200.0)
        assert est_lt.lt_qty == 10
        assert est_lt.st_qty == 0


# -----------------------------------------------------------------------
# Date parsing
# -----------------------------------------------------------------------

class TestParseFlexDate:

    def test_yyyymmdd_with_time(self):
        assert _parse_flex_date("20260511;120000") == date(2026, 5, 11)

    def test_yyyymmdd_no_time(self):
        assert _parse_flex_date("20260511") == date(2026, 5, 11)

    def test_yyyymmdd_space_time(self):
        assert _parse_flex_date("20260511 120000") == date(2026, 5, 11)

    def test_iso_yyyy_mm_dd(self):
        assert _parse_flex_date("2026-05-11") == date(2026, 5, 11)

    def test_empty_returns_none(self):
        assert _parse_flex_date("") is None
        assert _parse_flex_date(None) is None

    def test_garbage_returns_none(self):
        assert _parse_flex_date("notadate") is None


# -----------------------------------------------------------------------
# lots_from_flex_rows
# -----------------------------------------------------------------------

class TestLotsFromFlexRows:

    def test_long_lot_basic(self):
        rows = [{
            "symbol": "AAPL",
            "position": "100",
            "costBasisPrice": "150.50",
            "openDateTime": "20260101;093000",
            "originatingOrderID": "ORD1",
        }]
        out = lots_from_flex_rows(rows)
        assert "AAPL" in out
        lots = out["AAPL"]
        assert len(lots) == 1
        assert lots[0].side == "long"
        assert lots[0].qty == 100
        assert lots[0].cost_basis_per_share == 150.50
        assert lots[0].open_date == date(2026, 1, 1)
        assert lots[0].lot_id == "ORD1"

    def test_short_lot_negative_position(self):
        rows = [{
            "symbol": "TSLA",
            "position": "-25",
            "costBasisPrice": "300",
            "openDateTime": "20260301",
        }]
        out = lots_from_flex_rows(rows)
        lots = out["TSLA"]
        assert lots[0].side == "short"
        assert lots[0].qty == 25  # magnitude

    def test_zero_position_skipped(self):
        rows = [{"symbol": "ZZZ", "position": "0", "costBasisPrice": "100"}]
        assert lots_from_flex_rows(rows) == {}

    def test_missing_symbol_skipped(self):
        rows = [{"symbol": "", "position": "100"}]
        assert lots_from_flex_rows(rows) == {}

    def test_synthetic_lot_id_when_missing(self):
        rows = [{
            "symbol": "ABC", "position": "5",
            "costBasisPrice": "50", "openDateTime": "20260101",
        }]
        lots = lots_from_flex_rows(rows)["ABC"]
        assert lots[0].lot_id  # non-empty
        assert "ABC" in lots[0].lot_id

    def test_alternate_field_names(self):
        # Tolerate `qty`, `cost_basis`, `open_date` as inputs
        rows = [{
            "sym": "XYZ",
            "qty": 10,
            "cost_basis": 75,
            "open_date": "2026-02-15",
        }]
        out = lots_from_flex_rows(rows)
        assert "XYZ" in out
        assert out["XYZ"][0].qty == 10
        assert out["XYZ"][0].cost_basis_per_share == 75
        assert out["XYZ"][0].open_date == date(2026, 2, 15)


# -----------------------------------------------------------------------
# XML parsing
# -----------------------------------------------------------------------

class TestParseOpenPositionLots:

    def test_parses_lot_level_rows(self, tmp_path):
        xml = dedent("""
            <FlexQueryResponse>
              <FlexStatements>
                <FlexStatement>
                  <OpenPositions>
                    <OpenPosition
                        symbol="AAPL" position="100"
                        costBasisPrice="150.50" costBasisMoney="15050"
                        markPrice="160" openDateTime="20260101;093000"
                        holdingPeriodDateTime="20260101"
                        originatingOrderID="ORD1"
                        levelOfDetail="LOT"/>
                    <OpenPosition
                        symbol="AAPL" position="50"
                        costBasisPrice="155" costBasisMoney="7750"
                        markPrice="160" openDateTime="20260301"
                        originatingOrderID="ORD2"
                        levelOfDetail="LOT"/>
                  </OpenPositions>
                </FlexStatement>
              </FlexStatements>
            </FlexQueryResponse>
        """).strip()
        p = tmp_path / "flex_positions.xml"
        p.write_text(xml)

        rows = parse_open_position_lots(p)
        assert len(rows) == 2
        assert rows[0]["symbol"] == "AAPL"
        assert rows[0]["originatingOrderID"] == "ORD1"
        assert rows[1]["originatingOrderID"] == "ORD2"

    def test_skips_summary_level_rows(self, tmp_path):
        xml = dedent("""
            <FlexQueryResponse>
              <OpenPositions>
                <OpenPosition symbol="AAPL" position="150" levelOfDetail="SUMMARY"/>
                <OpenPosition symbol="AAPL" position="100" levelOfDetail="LOT"
                  costBasisPrice="150" openDateTime="20260101"/>
              </OpenPositions>
            </FlexQueryResponse>
        """).strip()
        p = tmp_path / "flex_positions.xml"
        p.write_text(xml)
        rows = parse_open_position_lots(p)
        assert len(rows) == 1
        assert rows[0]["position"] == "100"

    def test_missing_open_positions_raises(self, tmp_path):
        xml = "<FlexQueryResponse></FlexQueryResponse>"
        p = tmp_path / "flex_positions.xml"
        p.write_text(xml)
        with pytest.raises(ValueError):
            parse_open_position_lots(p)


# -----------------------------------------------------------------------
# load_lot_views_for_run_dir
# -----------------------------------------------------------------------

class TestLoadLotViewsForRunDir:

    def test_returns_empty_when_no_xml(self, tmp_path):
        out = load_lot_views_for_run_dir(tmp_path)
        assert out == {}

    def test_falls_back_to_search_root(self, tmp_path):
        # Place an XML at tmp_path/runs/2026-04-01/accounting/flex_positions.xml
        nested = tmp_path / "runs" / "2026-04-01" / "accounting"
        nested.mkdir(parents=True)
        xml = dedent("""
            <FlexQueryResponse>
              <OpenPositions>
                <OpenPosition symbol="ABC" position="5" costBasisPrice="50"
                  openDateTime="20260101" originatingOrderID="O1"
                  levelOfDetail="LOT"/>
              </OpenPositions>
            </FlexQueryResponse>
        """).strip()
        (nested / "flex_positions.xml").write_text(xml)

        # Ask for a run_dir that doesn't have the file, with fallback root set
        out = load_lot_views_for_run_dir(
            tmp_path / "runs" / "2026-05-11" / "accounting",
            fallback_search_root=tmp_path / "runs",
            asof=date(2026, 5, 11),
        )
        assert "ABC" in out
        assert out["ABC"].lots[0].qty == 5
        assert out["ABC"].lots[0].open_date == date(2026, 1, 1)

    def test_handles_malformed_xml_gracefully(self, tmp_path):
        (tmp_path / "flex_positions.xml").write_text("<not><valid")
        # Should NOT raise; returns empty dict
        out = load_lot_views_for_run_dir(tmp_path)
        assert out == {}
