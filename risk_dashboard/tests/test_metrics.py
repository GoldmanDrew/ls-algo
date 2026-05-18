"""Smoke tests for risk_dashboard.metrics."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from risk_dashboard.metrics import (
    DEFAULT_LIMITS,
    SLEEVE_TARGET_WEIGHTS,
    compute_book_summary,
    compute_bucket_detail,
    compute_data_quality,
    compute_scenario_panel,
)


@pytest.fixture
def fake_totals() -> dict:
    return {
        "run_date": "2026-05-15",
        "total_pnl": 48626.58,
        "net_exposure_total": -752462.23,
        "gross_exposure_total": 4_195_127.28,
        "net_exposure_bucket_1": -707601.71,
        "gross_exposure_bucket_1": 3_966_574.48,
        "net_exposure_bucket_2": -44860.52,
        "gross_exposure_bucket_2": 228_552.80,
        "net_exposure_bucket_3": 85047.29,
        "gross_exposure_bucket_3": 86236.36,
        "net_exposure_bucket_4": 437084.68,
        "gross_exposure_bucket_4": 437084.68,
        "bucket_pnl": {
            "bucket_1": 10714.78,
            "bucket_2": 25381.18,
            "bucket_3": 10610.49,
            "bucket_4": 1920.14,
        },
    }


def test_book_summary_pct_nav(fake_totals):
    book = compute_book_summary(
        totals=fake_totals,
        pnl_by_bucket=pd.DataFrame(),
        nav_usd=800_000.0,
    )
    assert book.gross_notional_usd == pytest.approx(4_195_127.28)
    assert book.gross_exposure_pct_nav == pytest.approx(4_195_127.28 / 800_000.0)
    assert book.pnl_today_pct_nav == pytest.approx(48626.58 / 800_000.0)
    assert len(book.sleeve_table) == 4
    b4 = next(r for r in book.sleeve_table if r["bucket"] == "bucket_4")
    assert b4["target_weight"] == 0.25
    assert b4["actual_weight"] == pytest.approx(437084.68 / 4_195_127.28)


def test_book_summary_breach_when_gross_exceeds(fake_totals):
    book = compute_book_summary(
        totals=fake_totals,
        pnl_by_bucket=pd.DataFrame(),
        nav_usd=800_000.0,
    )
    # 4.19M / 800k = 524% -- way above the memo-linked hard limit.
    assert any(b["metric"] == "gross_exposure_pct_nav" for b in book.breaches)
    breach = next(b for b in book.breaches if b["metric"] == "gross_exposure_pct_nav")
    assert breach["status"] == "hard"


def test_compute_bucket_detail_handles_missing_files(tmp_path: Path):
    detail = compute_bucket_detail(
        bucket="bucket_4",
        pnl_csv=tmp_path / "missing_pnl.csv",
        net_exposure_csv=tmp_path / "missing_expo.csv",
    )
    assert detail["bucket"] == "bucket_4"
    assert detail["n_pnl_rows"] == 0
    assert detail["n_exposure_rows"] == 0


def test_compute_bucket_detail_normalizes_grouped_bucket_rows(tmp_path: Path):
    pnl_path = tmp_path / "pnl_bucket_1.csv"
    expo_path = tmp_path / "net_exposure_bucket_1.csv"
    pnl_path.write_text(
        "underlying,symbols,realized_pnl,unrealized_pnl,borrow_fees,short_credit_interest,total_pnl\n"
        "DIA,\"DIA, DXD\",0,100,-1,0,99\n"
        "XLK,XLK,0,-50,0,0,-50\n",
        encoding="utf-8",
    )
    expo_path.write_text(
        "underlying,symbols,net_notional_usd,gross_notional_usd,n_legs\n"
        "DIA,\"DIA, DXD\",1000,1500,2\n",
        encoding="utf-8",
    )

    detail = compute_bucket_detail("bucket_1", pnl_path, expo_path)

    assert detail["winners"][0]["display_name"] == "DIA"
    assert detail["winners"][0]["description"] == "DIA, DXD"
    assert detail["losers"][0]["display_name"] == "XLK"
    assert detail["exposure_rows"][0]["underlying"] == "DIA"
    assert detail["exposure_rows"][0]["symbols"] == "DIA, DXD"


def test_data_quality_counts_no_blank_top_rows(tmp_path: Path):
    accounting = tmp_path / "accounting"
    flex = tmp_path / "ibkr_flex"
    accounting.mkdir()
    flex.mkdir()
    (accounting / "totals.json").write_text("{}", encoding="utf-8")
    for bucket in ("bucket_1", "bucket_2", "bucket_3", "bucket_4"):
        (accounting / f"pnl_{bucket}.csv").write_text(
            "underlying,symbols,total_pnl\nABC,\"ABC, ABCU\",1\n",
            encoding="utf-8",
        )
        (accounting / f"net_exposure_{bucket}.csv").write_text(
            "underlying,symbols,net_notional_usd,gross_notional_usd,n_legs\nABC,ABC,1,1,1\n",
            encoding="utf-8",
        )
    (accounting / "pnl_by_symbol.csv").write_text(
        "symbol,underlying,bucket,total_pnl\nABC,ABC,bucket_1,1\n",
        encoding="utf-8",
    )
    (accounting / "pnl_by_underlying.csv").write_text(
        "underlying,symbols,total_pnl\nABC,ABC,1\n",
        encoding="utf-8",
    )
    (flex / "flex_positions.xml").write_text("<FlexQueryResponse />", encoding="utf-8")
    (flex / "flex_borrow_fee_details.xml").write_text("<FlexQueryResponse />", encoding="utf-8")

    buckets = {
        bucket: compute_bucket_detail(
            bucket,
            accounting / f"pnl_{bucket}.csv",
            accounting / f"net_exposure_{bucket}.csv",
        )
        for bucket in ("bucket_1", "bucket_2", "bucket_3", "bucket_4")
    }
    dq = compute_data_quality(
        accounting_dir=accounting,
        flex_dir=flex,
        buckets=buckets,
        totals={
            "gross_exposure_total": 4.0,
            "net_exposure_total": 4.0,
            "gross_exposure_bucket_1": 1.0,
            "gross_exposure_bucket_2": 1.0,
            "gross_exposure_bucket_3": 1.0,
            "gross_exposure_bucket_4": 1.0,
            "net_exposure_bucket_1": 1.0,
            "net_exposure_bucket_2": 1.0,
            "net_exposure_bucket_3": 1.0,
            "net_exposure_bucket_4": 1.0,
        },
        run_date="2026-05-18",
    )

    assert dq["blank_render_field_count"] == 0
    assert dq["missing_source_count"] == 0
    assert dq["missing_required_column_count"] == 0
    assert dq["status"] == "ok"


def test_compute_scenario_panel_ranks_worst_contributor():
    buckets = {
        "bucket_1": {
            "exposure_rows": [
                {
                    "underlying": "LONG",
                    "symbols": "LONG",
                    "net_notional_usd": 1000.0,
                    "gross_notional_usd": 1000.0,
                },
                {
                    "underlying": "SHORT",
                    "symbols": "SHORT",
                    "net_notional_usd": -500.0,
                    "gross_notional_usd": 500.0,
                },
            ],
            "pnl_rows": [
                {"display_name": "LONG", "symbols": "LONG", "borrow_fees": -10.0}
            ],
        }
    }

    panel = compute_scenario_panel(buckets, nav_usd=10_000.0)
    down_5 = next(s for s in panel["scenarios"] if s["id"] == "market_down_5")

    assert down_5["pnl_usd"] == pytest.approx(-25.0)
    assert down_5["top_contributor"]["underlying"] == "LONG"
    assert panel["worst_shock"]["pnl_usd"] <= down_5["pnl_usd"]


def test_default_limits_are_sane():
    for k, v in DEFAULT_LIMITS.items():
        assert "warn" in v and "hard" in v, k
    # Sleeve targets must sum (within rounding) to roughly 1.0 when bucket_3
    # is excluded (b3 is layered, not a fixed slice).
    fixed = sum(
        v for k, v in SLEEVE_TARGET_WEIGHTS.items() if v is not None and k != "bucket_3"
    )
    assert 0.95 <= fixed <= 1.05
