"""Tests for PnL attribution long/short split and row builder in run_eod_pnl_email."""

import pandas as pd

from run_eod_pnl_email import build_attribution_row, split_long_short_realized_unrealized


def test_split_long_short_by_position_sign():
    pnl = pd.DataFrame(
        [
            {"symbol": "AAA", "realized_pnl": 100.0, "unrealized_pnl": 50.0},
            {"symbol": "BBB", "realized_pnl": -20.0, "unrealized_pnl": 30.0},
        ]
    )
    pos = pd.DataFrame(
        [
            {"symbol": "AAA", "position": 100.0},
            {"symbol": "BBB", "position": -10.0},
        ]
    )
    lr, lu, sr, su = split_long_short_realized_unrealized(pnl, pos)
    assert lr == 100.0 and lu == 50.0
    assert sr == -20.0 and su == 30.0


def test_split_flat_symbol_defaults_to_long():
    pnl = pd.DataFrame([{"symbol": "ZZZ", "realized_pnl": 5.0, "unrealized_pnl": 0.0}])
    pos = pd.DataFrame()
    lr, lu, sr, su = split_long_short_realized_unrealized(pnl, pos)
    assert lr == 5.0 and sr == 0.0


def test_build_attribution_row_totals_round_trip():
    totals = {
        "total_realized_pnl": 10.0,
        "total_unrealized_pnl": 20.0,
        "total_dividends": 1.0,
        "total_withholding_tax": -0.5,
        "total_pil_dividends": -2.0,
        "total_borrow_fees": -3.0,
        "total_short_credit_interest": 0.5,
        "total_other_fees": -0.25,
        "total_bond_interest": 0.1,
        "total_pnl": 25.85,
        "excluded_cash_interest_base": -99.0,
    }
    pnl = pd.DataFrame(
        [
            {"symbol": "X", "realized_pnl": 10.0, "unrealized_pnl": 20.0},
        ]
    )
    pos = pd.DataFrame([{"symbol": "X", "position": 1.0}])
    row = build_attribution_row("2026-01-01", totals, pnl, pos)
    assert row["gross_realized_pnl"] == 10.0
    assert row["excluded_cash_interest_base"] == -99.0
    assert row["strategy_total_pnl"] == 25.85
    assert row["long_realized_pnl"] + row["short_realized_pnl"] == 10.0
