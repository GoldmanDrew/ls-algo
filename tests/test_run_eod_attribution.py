"""Tests for PnL attribution long/short split and row builder in run_eod_pnl_email."""

import pytest
import pandas as pd

from run_eod_pnl_email import (
    compute_bucket_capital_snapshot,
    format_bucket_return_table,
    build_attribution_row,
    split_long_short_realized_unrealized,
)


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


def test_compute_bucket_capital_snapshot_splits_spot_and_uses_maintenance_margin():
    positions = pd.DataFrame(
        [
            {
                "symbol": "LETF",
                "position": -100.0,
                "markPrice": 10.0,
                "fxRateToBase": 1.0,
                "positionValue_base": -1000.0,
            },
            {
                "symbol": "SPOT",
                "position": 100.0,
                "markPrice": 20.0,
                "fxRateToBase": 1.0,
                "positionValue_base": 2000.0,
            },
        ]
    )
    pnl_symbol = pd.DataFrame(
        [
            {"symbol": "LETF", "bucket": "bucket_1"},
            {"symbol": "SPOT", "bucket": "bucket_1"},
            {"symbol": "SPOT", "bucket": "bucket_2"},
        ]
    )
    screened = pd.DataFrame(
        [
            {
                "ETF": "LETF",
                "bucket": "bucket_1",
                "maint_pct_long": 0.50,
                "maint_pct_short": 0.60,
            }
        ]
    )
    lot_state = pd.DataFrame(
        [
            {
                "underlying": "SPOT",
                "qty_b1": 25.0,
                "qty_b2": 75.0,
                "qty_b4": 0.0,
            }
        ]
    )

    snap = compute_bucket_capital_snapshot(positions, pnl_symbol, screened, lot_state)

    assert snap["net_capital_bucket_1"] == pytest.approx(-500.0)
    assert snap["gross_capital_bucket_1"] == pytest.approx(1500.0)
    assert snap["margin_req_bucket_1"] == pytest.approx(725.0)
    assert snap["net_capital_bucket_2"] == pytest.approx(1500.0)
    assert snap["gross_capital_bucket_2"] == pytest.approx(1500.0)
    assert snap["margin_req_bucket_2"] == pytest.approx(375.0)


def test_format_bucket_return_table_includes_return_metrics():
    table = format_bucket_return_table(
        {"bucket_1": 10.0, "bucket_2": 0.0, "bucket_3": 0.0, "bucket_4": 0.0},
        {
            "net_capital_bucket_1": 100.0,
            "gross_capital_bucket_1": 200.0,
            "margin_req_bucket_1": 50.0,
        },
    )

    assert "ROC" in table
    assert "10.00%" in table
    assert "5.00%" in table
    assert "20.00%" in table
