"""Tests for dashboard data-contract and verification gates."""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from risk_dashboard.flex_parser import parse_flex_nav
from scripts.run_data_contract import (
    daily_pnl_from_history,
    is_broker_nav_source,
    is_config_nav_source,
    patch_totals_nav,
    write_run_manifest,
)
from scripts.trading_calendar import is_us_equity_session, scheduled_run_context
from scripts.verify_dashboard_snapshot import verify_snapshot

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS = PROJECT_ROOT / "data" / "runs"


def test_is_us_equity_session_weekend():
    assert is_us_equity_session(date(2026, 6, 27)) is False  # Saturday


def test_scheduled_run_context_has_run_date():
    ctx = scheduled_run_context()
    assert "run_date" in ctx
    assert "should_run" in ctx


def test_parse_flex_nav_from_percent_of_nav():
    flex = RUNS / "2026-06-30" / "ibkr_flex"
    if not flex.is_dir():
        pytest.skip("no flex dir")
    nav = parse_flex_nav(flex)
    assert nav is not None
    assert float(nav["nav_usd"]) > 1_000_000
    assert "percentOfNAV" in str(nav.get("source", ""))


def test_is_config_nav_source():
    assert is_config_nav_source("config:capital_usd") is True
    assert is_config_nav_source("flex_positions:percentOfNAV_median") is False


def test_is_broker_nav_source():
    assert is_broker_nav_source("flex_positions:percentOfNAV_median") is True
    assert is_broker_nav_source("config:capital_usd") is False


def test_daily_pnl_from_history_finds_prior():
    daily, prior = daily_pnl_from_history("2026-06-26")
    if daily is None:
        pytest.skip("no pnl_history rows")
    assert prior is not None
    assert isinstance(daily, float)


def test_write_run_manifest_and_verify_latest(tmp_path):
    run_date = "2026-01-02"
    run_dir = tmp_path / "runs" / run_date
    acc = run_dir / "accounting"
    flex = run_dir / "ibkr_flex"
    acc.mkdir(parents=True)
    flex.mkdir(parents=True)
    totals = {
        "total_pnl": 100.0,
        "bucket_pnl": {"bucket_1": 100.0, "bucket_2": 0.0, "bucket_3": 0.0, "bucket_4": 0.0, "bucket_5": 0.0},
        "gross_exposure_total": 1_000_000.0,
        "net_exposure_total": 0.0,
        "gross_exposure_bucket_1": 1_000_000.0,
        "gross_exposure_bucket_2": 0.0,
        "gross_exposure_bucket_4": 0.0,
        "net_exposure_bucket_1": 0.0,
        "net_exposure_bucket_2": 0.0,
        "net_exposure_bucket_4": 0.0,
        "net_exposure_unbucketed": 0.0,
    }
    (acc / "totals.json").write_text(json.dumps(totals), encoding="utf-8")
    (acc / "pnl_bucket_1.csv").write_text("underlying,total_pnl\nA,100\n", encoding="utf-8")
    for b in ("bucket_2", "bucket_3", "bucket_4", "bucket_5"):
        (acc / f"pnl_{b}.csv").write_text("underlying,total_pnl\n", encoding="utf-8")
    (flex / "flex_positions.xml").write_text(
        '<?xml version="1.0"?><FlexQueryResponse><FlexStatements><FlexStatement>'
        '<OpenPosition percentOfNAV="10" positionValue="100000" fxRateToBase="1" levelOfDetail="SUMMARY"/>'
        "</FlexStatement></FlexStatements></FlexQueryResponse>",
        encoding="utf-8",
    )
    manifest = write_run_manifest(run_date, runs_root=tmp_path / "runs")
    assert manifest["nav_usd"] == pytest.approx(1_050_000.0)
    assert manifest["nav_source"] == "config:capital_usd"

    out = tmp_path / "risk_dashboard" / "data"
    out.mkdir(parents=True)
    snap = {
        "run_date": run_date,
        "nav_source": manifest["nav_source"],
        "freshness": {"is_latest": True},
        "book": {"pnl_ytd_usd": 100.0, "pnl_today_usd": 100.0},
        "buckets": {
            "bucket_1": {"pnl_rows": [{"total_pnl": 100.0}]},
            "bucket_2": {"pnl_rows": []},
            "bucket_3": {"pnl_rows": []},
            "bucket_4": {"pnl_rows": []},
            "bucket_5": {"pnl_rows": []},
        },
        "data_quality": {"status": "ok"},
        "exposure_reconciliation": {"reconciles": True},
        "borrow_shock_panel": {},
        "drawdown_panel": {},
        "pnl_panel": {},
        "movers_panel": {},
        "display_sleeve_groups": [],
    }
    (out / f"{run_date}.json").write_text(json.dumps(snap), encoding="utf-8")
    (out / "latest.json").write_text(json.dumps(snap), encoding="utf-8")

    errors = verify_snapshot(
        run_date,
        runs_root=tmp_path / "runs",
        snapshot_dir=out,
        require_fresh=False,
    )
    assert errors == []
