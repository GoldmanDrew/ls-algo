"""Tests for PnL attribution long/short split and row builder in run_eod_pnl_email."""

import json
from pathlib import Path

import pytest
import pandas as pd

from run_eod_pnl_email import (
    _bucket_pnl_from_totals,
    _eod_bucket_pnl_continuity_enabled,
    _format_subject_bucket_pnl_line,
    _load_flow_universe_sets,
    apply_bucket_pnl_continuity,
    compute_average_bucket_capital,
    compute_bucket_capital_snapshot,
    compute_bucket_daily_deltas,
    compute_period_pnl_deltas,
    format_bucket_email_block,
    format_bucket_pnl_section,
    format_bucket_return_table,
    format_bucket_ytd_headline,
    format_eod_subject,
    format_top_underlying_net_exposure,
    load_position_discrepancies,
    read_bucket_pnl_from_run,
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


def test_compute_bucket_capital_snapshot_excludes_positions_outside_screened_universe():
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
                "symbol": "GTX",
                "position": 1000.0,
                "markPrice": 10.0,
                "fxRateToBase": 1.0,
                "positionValue_base": 10000.0,
            },
        ]
    )
    screened = pd.DataFrame(
        [
            {
                "ETF": "LETF",
                "Underlying": "LETF_U",
                "delta": 2.0,
                "bucket": "bucket_1",
                "maint_pct_long": 0.50,
                "maint_pct_short": 0.60,
            }
        ]
    )
    snap = compute_bucket_capital_snapshot(positions, pd.DataFrame(), screened, pd.DataFrame())
    assert snap["net_capital_bucket_1"] == pytest.approx(-1000.0)
    assert snap["gross_capital_bucket_1"] == pytest.approx(1000.0)
    assert snap["net_capital_bucket_2"] == pytest.approx(0.0)


def test_compute_bucket_capital_snapshot_keeps_bucket5_from_pnl_symbol():
    positions = pd.DataFrame(
        [
            {
                "symbol": "UVIX",
                "position": -25.0,
                "markPrice": 40.0,
                "fxRateToBase": 1.0,
                "positionValue_base": -1000.0,
            },
            {
                "symbol": "OUTSIDE",
                "position": 100.0,
                "markPrice": 10.0,
                "fxRateToBase": 1.0,
                "positionValue_base": 1000.0,
            },
        ]
    )
    pnl_symbol = pd.DataFrame(
        [
            {"symbol": "UVIX", "bucket": "bucket_5"},
            {"symbol": "OUTSIDE", "bucket": "bucket_1"},
        ]
    )
    screened = pd.DataFrame(
        [
            {
                "ETF": "LETF",
                "Underlying": "LETF_U",
                "delta": 2.0,
                "bucket": "bucket_1",
                "maint_pct_long": 0.50,
                "maint_pct_short": 0.60,
            }
        ]
    )

    snap = compute_bucket_capital_snapshot(positions, pnl_symbol, screened, pd.DataFrame())

    assert snap["net_capital_bucket_5"] == pytest.approx(-1000.0)
    assert snap["gross_capital_bucket_5"] == pytest.approx(1000.0)
    assert snap["margin_req_bucket_5"] == pytest.approx(300.0)
    assert snap["net_capital_bucket_1"] == pytest.approx(0.0)


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
                "Underlying": "LETF_U",
                "delta": 2.0,
                "bucket": "bucket_1",
                "maint_pct_long": 0.50,
                "maint_pct_short": 0.60,
            },
            {
                "ETF": "SPOT_ETF",
                "Underlying": "SPOT",
                "delta": 1.0,
                "bucket": "bucket_2",
                "maint_pct_long": 0.25,
                "maint_pct_short": 0.30,
            },
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

    # SPOT ledger attributes 25 + 75 = 100 shares vs the actual 100-share IBKR
    # long line, so no capping. Each bucket gets qty * price directly.
    assert snap["net_capital_bucket_1"] == pytest.approx(-500.0)
    assert snap["gross_capital_bucket_1"] == pytest.approx(1500.0)
    assert snap["margin_req_bucket_1"] == pytest.approx(725.0)
    assert snap["net_capital_bucket_2"] == pytest.approx(1500.0)
    assert snap["gross_capital_bucket_2"] == pytest.approx(1500.0)
    assert snap["margin_req_bucket_2"] == pytest.approx(375.0)


def test_compute_bucket_capital_snapshot_caps_at_position_when_lot_state_is_stale():
    # Stale ledger reports 1,000 shares (10x) but the IBKR line is only 100.
    positions = pd.DataFrame(
        [
            {
                "symbol": "SPOT",
                "position": 100.0,
                "markPrice": 20.0,
                "fxRateToBase": 1.0,
                "positionValue_base": 2000.0,
            }
        ]
    )
    pnl_symbol = pd.DataFrame(
        [
            {"symbol": "SPOT", "bucket": "bucket_1"},
            {"symbol": "SPOT", "bucket": "bucket_2"},
        ]
    )
    screened = pd.DataFrame(
        [
            {
                "ETF": "SPOT_ETF",
                "Underlying": "SPOT",
                "delta": 1.0,
                "bucket": "bucket_1",
                "maint_pct_long": 0.25,
                "maint_pct_short": 0.30,
            }
        ]
    )
    lot_state = pd.DataFrame(
        [
            {
                "underlying": "SPOT",
                "qty_b1": 250.0,
                "qty_b2": 750.0,
                "qty_b4": 0.0,
            }
        ]
    )

    snap = compute_bucket_capital_snapshot(positions, pnl_symbol, screened, lot_state)

    # Total attributed MV scaled down to the actual $2,000 position.
    assert snap["net_capital_bucket_1"] + snap["net_capital_bucket_2"] == pytest.approx(2000.0)
    assert snap["gross_capital_bucket_1"] + snap["gross_capital_bucket_2"] == pytest.approx(2000.0)
    assert snap["net_capital_bucket_1"] == pytest.approx(500.0)
    assert snap["net_capital_bucket_2"] == pytest.approx(1500.0)


def test_compute_bucket_capital_snapshot_excludes_orphan_position_shares():
    # IBKR line is 1,000 shares but ledger only attributes 100 to a bucket
    # (the other 900 are pre-strategy / blacklist holdings). Bucket capital
    # should reflect only the 100 attributed shares (worth $1,000), not the
    # full $10,000 position market value.
    positions = pd.DataFrame(
        [
            {
                "symbol": "SPOT",
                "position": 1000.0,
                "markPrice": 10.0,
                "fxRateToBase": 1.0,
                "positionValue_base": 10000.0,
            }
        ]
    )
    pnl_symbol = pd.DataFrame([{"symbol": "SPOT", "bucket": "bucket_1"}])
    screened = pd.DataFrame(
        [
            {
                "ETF": "SPOT_ETF",
                "Underlying": "SPOT",
                "delta": 1.0,
                "bucket": "bucket_1",
                "maint_pct_long": 0.25,
                "maint_pct_short": 0.30,
            }
        ]
    )
    lot_state = pd.DataFrame(
        [
            {
                "underlying": "SPOT",
                "qty_b1": 100.0,
                "qty_b2": 0.0,
                "qty_b4": 0.0,
            }
        ]
    )

    snap = compute_bucket_capital_snapshot(positions, pnl_symbol, screened, lot_state)

    assert snap["net_capital_bucket_1"] == pytest.approx(1000.0)
    assert snap["gross_capital_bucket_1"] == pytest.approx(1000.0)


def test_format_bucket_return_table_includes_return_metrics():
    table = format_bucket_return_table(
        {
            "bucket_1": 10.0,
            "bucket_2": 0.0,
            "bucket_3": 0.0,
            "bucket_4": 0.0,
        },
        {
            "net_capital_bucket_1": 100.0,
            "gross_capital_bucket_1": 200.0,
            "margin_req_bucket_1": 50.0,
            "net_capital_bucket_2": 0.0,
            "gross_capital_bucket_2": 0.0,
            "margin_req_bucket_2": 0.0,
            "net_capital_bucket_3": -50.0,
            "gross_capital_bucket_3": 200.0,
            "margin_req_bucket_3": 50.0,
            "net_capital_bucket_4": 0.0,
            "gross_capital_bucket_4": 0.0,
            "margin_req_bucket_4": 0.0,
        },
    )

    assert "AVG_NET_CAP" in table
    assert "AVG_GROSS_CAP" in table
    assert "AVG_MAINT_MARGIN" in table
    assert "ROC" in table
    assert "10.00%" in table
    assert "5.00%" in table
    assert "20.00%" in table
    lines = table.splitlines()
    b1_row = next(line for line in lines if "Bucket 1" in line)
    b3_row = next(line for line in lines if "Bucket 3" in line)
    assert "n/a" in b3_row
    assert "n/a" not in b1_row
    assert sum("Bucket" in line for line in lines) == 5


def test_compute_average_bucket_capital_means_daily_history():
    history = pd.DataFrame(
        [
            {
                "date": "2026-02-27",
                "net_capital_bucket_1": 100.0,
                "gross_capital_bucket_1": 200.0,
                "margin_req_bucket_1": 50.0,
            },
            {
                "date": "2026-02-28",
                "net_capital_bucket_1": 300.0,
                "gross_capital_bucket_1": 400.0,
                "margin_req_bucket_1": 150.0,
            },
        ]
    )

    avg = compute_average_bucket_capital(history)

    assert avg["net_capital_bucket_1"] == pytest.approx(200.0)
    assert avg["gross_capital_bucket_1"] == pytest.approx(300.0)
    assert avg["margin_req_bucket_1"] == pytest.approx(100.0)
    assert avg["net_capital_bucket_3"] == pytest.approx(0.0)


def test_compute_average_bucket_capital_skips_nan_legacy_rows():
    history = pd.DataFrame(
        [
            {"date": "2026-02-27", "net_capital_bucket_2": float("nan")},
            {"date": "2026-02-28", "net_capital_bucket_2": 400.0},
            {"date": "2026-03-02", "net_capital_bucket_2": 600.0},
        ]
    )

    avg = compute_average_bucket_capital(history)

    assert avg["net_capital_bucket_2"] == pytest.approx(500.0)


def test_compute_average_bucket_capital_handles_empty_history():
    avg = compute_average_bucket_capital(pd.DataFrame())
    assert avg["net_capital_bucket_1"] == 0.0
    assert avg["net_capital_bucket_3"] == 0.0
    assert avg["margin_req_bucket_3"] == 0.0


def test_compute_period_pnl_deltas_daily_vs_ytd():
    history = pd.DataFrame(
        [
            {
                "date": "2026-05-18",
                "pnl_bucket_1": 100.0,
                "pnl_bucket_2": 200.0,
                "pnl_bucket_3": 50.0,
                "pnl_bucket_4": -10.0,
                "total_pnl": 340.0,
            },
            {
                "date": "2026-05-19",
                "pnl_bucket_1": 90.0,
                "pnl_bucket_2": 180.0,
                "pnl_bucket_3": 55.0,
                "pnl_bucket_4": -5.0,
                "total_pnl": 320.0,
            },
        ]
    )
    daily = compute_period_pnl_deltas(history, "2026-05-19", period="daily")
    assert daily is not None
    assert daily["bucket_1"] == pytest.approx(-10.0)
    assert daily["bucket_2"] == pytest.approx(-20.0)
    assert daily["bucket_3"] == pytest.approx(5.0)
    assert daily["bucket_4"] == pytest.approx(5.0)
    assert daily["total"] == pytest.approx(-20.0)
    assert daily["bucket_1"] != history.iloc[-1]["pnl_bucket_1"]


def test_compute_period_pnl_deltas_no_baseline_returns_zero():
    history = pd.DataFrame(
        [
            {
                "date": "2026-05-19",
                "pnl_bucket_1": 90.0,
                "pnl_bucket_2": 180.0,
                "pnl_bucket_3": 55.0,
                "pnl_bucket_4": -5.0,
                "total_pnl": 320.0,
            },
        ]
    )
    daily = compute_period_pnl_deltas(history, "2026-05-19", period="daily")
    assert daily is not None
    assert daily["bucket_1"] == pytest.approx(0.0)
    assert daily["total"] == pytest.approx(0.0)


def test_apply_bucket_pnl_continuity_spreads_account_daily(tmp_path, monkeypatch):
    runs = tmp_path / "data" / "runs"
    for d, bp, total in (
        (
            "2026-05-18",
            {"bucket_1": 25426.75, "bucket_2": 28552.10, "bucket_3": 7809.71, "bucket_4": -2410.09},
            59378.47,
        ),
        (
            "2026-05-19",
            {"bucket_1": 227261.17, "bucket_2": -180028.37, "bucket_3": 7344.99, "bucket_4": -1964.68},
            52613.11,
        ),
    ):
        acct = runs / d / "accounting"
        acct.mkdir(parents=True)
        (acct / "totals.json").write_text(
            json.dumps({"total_pnl": total, "bucket_pnl": bp}),
            encoding="utf-8",
        )
        pd.DataFrame([{"bucket": "bucket_1", "total_pnl": bp["bucket_1"]}]).to_csv(
            acct / "pnl_by_bucket.csv", index=False
        )
    monkeypatch.setattr("run_eod_pnl_email.RUNS_ROOT", runs)
    monkeypatch.setattr("run_eod_pnl_email._eod_bucket_pnl_continuity_enabled", lambda: True)
    fixed = apply_bucket_pnl_continuity(
        "2026-05-19",
        json.loads((runs / "2026-05-19" / "accounting" / "totals.json").read_text()),
    )
    bp = fixed["bucket_pnl"]
    # B1/B2 outliers re-spread; B3/B4 keep raw daily moves from accounting.
    assert bp["bucket_1"] == pytest.approx(22252.0, abs=50.0)
    assert bp["bucket_2"] == pytest.approx(24980.0, abs=50.0)
    assert bp["bucket_3"] == pytest.approx(7344.99, abs=1.0)
    assert bp["bucket_4"] == pytest.approx(-1964.68, abs=1.0)
    assert bp["bucket_4"] - (-2410.09) == pytest.approx(445.41, abs=1.0)
    assert sum(bp.values()) == pytest.approx(52613.11, abs=0.1)


def test_headline_bucket_pnl_uses_totals_not_misattributed_detail_csv(tmp_path, monkeypatch):
    """EOD subject/history must use post-continuity totals, not pnl_bucket_*.csv sums."""
    runs = tmp_path / "data" / "runs"
    for d, bp, total in (
        (
            "2026-05-18",
            {"bucket_1": 25426.75, "bucket_2": 28552.10, "bucket_3": 7809.71, "bucket_4": -2410.09},
            59378.47,
        ),
        (
            "2026-05-19",
            {"bucket_1": 227261.17, "bucket_2": -180028.37, "bucket_3": 7344.99, "bucket_4": -1964.68},
            52613.11,
        ),
    ):
        acct = runs / d / "accounting"
        acct.mkdir(parents=True)
        (acct / "totals.json").write_text(
            json.dumps({"total_pnl": total, "bucket_pnl": bp}),
            encoding="utf-8",
        )
    monkeypatch.setattr("run_eod_pnl_email.RUNS_ROOT", runs)
    monkeypatch.setattr("run_eod_pnl_email._eod_bucket_pnl_continuity_enabled", lambda: True)
    totals = json.loads((runs / "2026-05-19" / "accounting" / "totals.json").read_text())
    fixed = apply_bucket_pnl_continuity("2026-05-19", totals)
    headline = _bucket_pnl_from_totals(fixed)
    assert headline["bucket_1"] == pytest.approx(22252.0, abs=50.0)
    assert headline["bucket_1"] != pytest.approx(227261.17, abs=1.0)
    assert sum(headline.values()) == pytest.approx(52613.11, abs=0.1)


def test_read_bucket_pnl_from_run_uses_pnl_by_bucket_csv(tmp_path, monkeypatch):
    run_dir = tmp_path / "data" / "runs" / "2026-05-18" / "accounting"
    run_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {"bucket": "bucket_1", "total_pnl": 25426.75},
            {"bucket": "bucket_2", "total_pnl": 28552.10},
            {"bucket": "bucket_3", "total_pnl": 7809.71},
            {"bucket": "bucket_4", "total_pnl": -2410.09},
        ]
    ).to_csv(run_dir / "pnl_by_bucket.csv", index=False)
    (run_dir / "totals.json").write_text(
        json.dumps({"bucket_pnl": {"bucket_1": -999.0, "bucket_2": 0, "bucket_3": 0, "bucket_4": 0}}),
        encoding="utf-8",
    )
    monkeypatch.setattr("run_eod_pnl_email.RUNS_ROOT", tmp_path / "data" / "runs")
    b1, b2, b3, b4, b5 = read_bucket_pnl_from_run("2026-05-18")
    assert b1 == pytest.approx(25426.75)
    assert b2 == pytest.approx(28552.10)
    assert b5 == pytest.approx(0.0)


def test_format_top_underlying_net_exposure_uses_book_rollup():
    df = pd.DataFrame(
        [
            {"underlying": "AMD", "net_notional_usd": 3000.0, "gross_notional_usd": 10000.0},
            {"underlying": "IBIT", "net_notional_usd": -2000.0, "gross_notional_usd": 8000.0},
            {"underlying": "TINY", "net_notional_usd": 100.0, "gross_notional_usd": 500.0},
        ]
    )
    text = format_top_underlying_net_exposure(df, min_abs_net_usd=500.0, max_rows=10)
    assert "Book net: $+1,100" in text
    assert "Book gross: $18,500" in text
    assert "AMD" in text
    assert "IBIT" in text
    assert "TINY" not in text
    assert "Buckets 1, 2 & 4 combined" in text or "stock sleeves" in text.lower()


def test_apply_bucket_pnl_continuity_skipped_when_config_disabled(tmp_path, monkeypatch):
    runs = tmp_path / "data" / "runs"
    for d, bp, total in (
        (
            "2026-05-18",
            {"bucket_1": 25426.75, "bucket_2": 28552.10, "bucket_3": 7809.71, "bucket_4": -2410.09},
            59378.47,
        ),
        (
            "2026-05-19",
            {"bucket_1": 227261.17, "bucket_2": -180028.37, "bucket_3": 7344.99, "bucket_4": -1964.68},
            52613.11,
        ),
    ):
        acct = runs / d / "accounting"
        acct.mkdir(parents=True)
        (acct / "totals.json").write_text(
            json.dumps({"total_pnl": total, "bucket_pnl": bp}),
            encoding="utf-8",
        )
    monkeypatch.setattr("run_eod_pnl_email.RUNS_ROOT", runs)
    monkeypatch.setattr("run_eod_pnl_email._eod_bucket_pnl_continuity_enabled", lambda: False)
    totals = json.loads((runs / "2026-05-19" / "accounting" / "totals.json").read_text())
    fixed = apply_bucket_pnl_continuity("2026-05-19", totals)
    assert fixed["bucket_pnl"]["bucket_1"] == pytest.approx(227261.17)


def test_load_flow_universe_sets_reads_bucket3_flow_low_delta_symbols(tmp_path, monkeypatch):
    runs = tmp_path / "data" / "runs" / "2026-05-30" / "accounting"
    runs.mkdir(parents=True)
    (runs / "totals.json").write_text(
        json.dumps({"bucket3_flow_low_delta_symbols": ["NVYY", "TSYY"]}),
        encoding="utf-8",
    )
    monkeypatch.setattr("run_eod_pnl_email.RUNS_ROOT", tmp_path / "data" / "runs")
    _, flow_low = _load_flow_universe_sets("2026-05-30")
    assert flow_low == {"NVYY", "TSYY"}


def test_format_bucket_ytd_headline_lists_all_buckets():
    text = format_bucket_ytd_headline(
        {
            "bucket_1": 100.0,
            "bucket_2": 200.0,
            "bucket_3": 50.0,
            "bucket_4": -10.0,
            "bucket_5": 7.0,
        }
    )
    assert "B1" in text
    assert "B2" in text
    assert "B5" in text
    assert "Stock sleeves (B1+B2+B4+B5): 297.00" in text


def test_format_eod_subject_lists_all_buckets():
    bucket_pnl = {
        "bucket_1": 22752.45,
        "bucket_2": 73537.07,
        "bucket_3": 9439.03,
        "bucket_4": -3510.76,
        "bucket_5": 125.0,
    }
    subject = format_eod_subject("2026-05-30", bucket_pnl, total_pnl=102217.79)
    assert "B1:" in subject
    assert "B2:" in subject
    assert "B3:" in subject
    assert "B4:" in subject
    assert "B5:" in subject
    assert "Total:" in subject
    line = _format_subject_bucket_pnl_line(bucket_pnl, total_pnl=102217.79)
    assert line.count("|") == 5


def test_format_bucket_pnl_section_matches_headline_for_fixture_run():
    run_dir = Path("data/runs/2026-05-30/accounting")
    if not (run_dir / "pnl_bucket_2.csv").is_file():
        pytest.skip("fixture run 2026-05-30 not on disk")
    totals = json.loads((run_dir / "totals.json").read_text(encoding="utf-8"))
    headline = _bucket_pnl_from_totals(totals)
    _, csv_sum = format_bucket_pnl_section(
        "bucket_2",
        run_dir / "pnl_bucket_2.csv",
        run_dir / "pnl_by_symbol.csv",
    )
    assert csv_sum == pytest.approx(headline["bucket_2"], abs=1.0)


def test_compute_bucket_daily_deltas_from_totals_json(tmp_path, monkeypatch):
    runs = tmp_path / "data" / "runs"
    for d, bp, total in (
        (
            "2026-05-29",
            {"bucket_1": 100.0, "bucket_2": 200.0, "bucket_3": 50.0, "bucket_4": -10.0},
            340.0,
        ),
        (
            "2026-05-30",
            {"bucket_1": 110.0, "bucket_2": 180.0, "bucket_3": 55.0, "bucket_4": -5.0},
            340.0,
        ),
    ):
        acct = runs / d / "accounting"
        acct.mkdir(parents=True)
        (acct / "totals.json").write_text(
            json.dumps({"total_pnl": total, "bucket_pnl": bp}),
            encoding="utf-8",
        )
    monkeypatch.setattr("run_eod_pnl_email.RUNS_ROOT", runs)
    deltas = compute_bucket_daily_deltas(
        "2026-05-30",
        {"bucket_1": 110.0, "bucket_2": 180.0, "bucket_3": 55.0, "bucket_4": -5.0},
    )
    assert deltas["bucket_1"] == pytest.approx(10.0)
    assert deltas["bucket_2"] == pytest.approx(-20.0)
    assert deltas["bucket_3"] == pytest.approx(5.0)
    assert deltas["bucket_4"] == pytest.approx(5.0)


def test_format_bucket_email_block_includes_pnl_and_exposure(tmp_path, monkeypatch):
    run_dir = tmp_path / "data" / "runs" / "2026-05-30" / "accounting"
    run_dir.mkdir(parents=True)
    pd.DataFrame(
        [{"underlying": "NVDA", "total_pnl": 100.0, "realized_pnl": 50.0, "unrealized_pnl": 50.0}]
    ).to_csv(run_dir / "pnl_bucket_2.csv", index=False)
    pd.DataFrame(
        [{"underlying": "NVDA", "net_notional_usd": 1000.0, "gross_notional_usd": 5000.0}]
    ).to_csv(run_dir / "net_exposure_bucket_2.csv", index=False)
    block = format_bucket_email_block(
        "bucket_2",
        headline_pnl=100.0,
        pnl_csv=run_dir / "pnl_bucket_2.csv",
        pnl_symbol_csv=run_dir / "pnl_by_symbol.csv",
        exposure_csv=run_dir / "net_exposure_bucket_2.csv",
        blocked_keys=set(),
    )
    assert "Bucket 2" in block
    assert "PnL detail:" in block
    assert "Net exposure" in block
    assert "NVDA" in block


def test_load_position_discrepancies_uses_optimal_target_usd(tmp_path, monkeypatch):
    run_dir = tmp_path / "data" / "runs" / "2026-05-30"
    flex_dir = run_dir / "ibkr_flex"
    flex_dir.mkdir(parents=True)
    (flex_dir / "flex_positions.xml").write_text(
        """<?xml version="1.0"?>
<FlexQueryResponse>
  <FlexStatements>
    <FlexStatement>
      <OpenPositions>
        <OpenPosition symbol="MTYY" position="-100" markPrice="4.0" fxRateToBase="1"
          assetCategory="STK" description="MTYY" />
      </OpenPositions>
    </FlexStatement>
  </FlexStatements>
</FlexQueryResponse>""",
        encoding="utf-8",
    )
    screened = tmp_path / "data" / "etf_screened_today.csv"
    screened.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"ETF": "MTYY", "Underlying": "MSTR", "delta": 1.0, "bucket": "bucket_2"}]).to_csv(
        screened, index=False
    )
    pd.DataFrame(
        [
            {
                "ETF": "MTYY",
                "Underlying": "MSTR",
                "long_usd": 1000.0,
                "short_usd": -3000.0,
                "optimal_long_usd": 5000.0,
                "optimal_short_usd": -15000.0,
                "strategy_tag": "ls",
                "purgatory": True,
                "purgatory_no_locate": True,
                "purgatory_borrow_band": False,
            }
        ]
    ).to_csv(run_dir / "proposed_trades.csv", index=False)

    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir()
    (cfg_dir / "strategy_config.yml").write_text(
        "strategy:\n  tag: ls\n  blacklist: []\n", encoding="utf-8"
    )

    monkeypatch.setattr("run_eod_pnl_email.PROJECT_ROOT", tmp_path)
    monkeypatch.setattr("run_eod_pnl_email.RUNS_ROOT", tmp_path / "data" / "runs")
    df = load_position_discrepancies("2026-05-30")
    row = df[df["symbol"] == "MTYY"].iloc[0]
    assert row["target_net_usd"] == pytest.approx(-15000.0)
    assert row["discrepancy_blocker"] == "no_locate"
    assert bool(row["purgatory_no_locate"]) is True


def test_load_position_discrepancies_merges_optimal_targets_csv(tmp_path, monkeypatch):
    """When proposed_trades lacks optimal_* columns, merge from optimal_targets.csv (harvest parity)."""
    run_dir = tmp_path / "data" / "runs" / "2026-05-30"
    flex_dir = run_dir / "ibkr_flex"
    flex_dir.mkdir(parents=True)
    (flex_dir / "flex_positions.xml").write_text(
        """<?xml version="1.0"?>
<FlexQueryResponse>
  <FlexStatements>
    <FlexStatement>
      <OpenPositions>
        <OpenPosition symbol="MTYY" position="-100" markPrice="4.0" fxRateToBase="1"
          assetCategory="STK" description="MTYY" />
      </OpenPositions>
    </FlexStatement>
  </FlexStatements>
</FlexQueryResponse>""",
        encoding="utf-8",
    )
    screened = tmp_path / "data" / "etf_screened_today.csv"
    screened.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"ETF": "MTYY", "Underlying": "MSTR", "delta": 1.0, "bucket": "bucket_2"}]).to_csv(
        screened, index=False
    )
    pd.DataFrame(
        [
            {
                "ETF": "MTYY",
                "Underlying": "MSTR",
                "long_usd": 1000.0,
                "short_usd": -3000.0,
                "strategy_tag": "ls",
            }
        ]
    ).to_csv(run_dir / "proposed_trades.csv", index=False)
    pd.DataFrame(
        [
            {
                "ETF": "MTYY",
                "Underlying": "MSTR",
                "optimal_long_usd": 5000.0,
                "optimal_short_usd": -15000.0,
                "optimal_gross_target_usd": 20000.0,
            }
        ]
    ).to_csv(run_dir / "optimal_targets.csv", index=False)

    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir()
    (cfg_dir / "strategy_config.yml").write_text(
        "strategy:\n  tag: ls\n  blacklist: []\n", encoding="utf-8"
    )

    monkeypatch.setattr("run_eod_pnl_email.PROJECT_ROOT", tmp_path)
    monkeypatch.setattr("run_eod_pnl_email.RUNS_ROOT", tmp_path / "data" / "runs")
    df = load_position_discrepancies("2026-05-30")
    row = df[df["symbol"] == "MTYY"].iloc[0]
    assert row["target_net_usd"] == pytest.approx(-15000.0)

