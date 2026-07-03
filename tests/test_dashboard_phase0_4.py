"""Unit tests for the Phase 0-4 risk-dashboard upgrades.

Covers: Bucket-5 wiring, NAV-from-config fallback + source labelling, daily/YTD
PnL naming, snapshot freshness, and the new analytical panels (borrow shock,
drawdown, shared-underlying map, top movers, display sleeve groups).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from risk_dashboard import metrics
from risk_dashboard.build_site import _nav_from_config
from risk_dashboard.metrics import (
    BUCKET_KEYS,
    SLEEVE_TARGET_WEIGHTS,
    compute_borrow_shock_panel,
    compute_drawdown_panel,
    compute_movers_panel,
    compute_pnl_panel,
    compute_shared_underlying_panel,
    resolve_nav_usd,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = PROJECT_ROOT / "risk_dashboard" / "data" / "latest.json"


# ── Bucket 5 wiring ────────────────────────────────────────────────────────


def test_bucket_5_is_a_first_class_bucket():
    assert "bucket_5" in BUCKET_KEYS
    assert "bucket_5" in SLEEVE_TARGET_WEIGHTS
    assert "bucket_5" in metrics.BUCKET_LABELS
    assert "bucket_5" in metrics.BUCKET_SLEEVE_KEYS


def test_display_stock_sleeves_include_b5_but_reconcile_set_does_not():
    # Reconcile set mirrors ibkr_accounting (B1+B2+B4); display set adds B5.
    assert "bucket_5" in metrics.DISPLAY_STOCK_SLEEVE_BUCKETS
    assert "bucket_5" not in metrics.RECONCILE_EXPOSURE_BUCKETS


# ── NAV fallback + source labelling ────────────────────────────────────────


def test_resolve_nav_threads_cli_source_label():
    nav, source = resolve_nav_usd({}, cli_fallback=1_050_000, cli_source="config:capital_usd")
    assert nav == pytest.approx(1_050_000)
    assert source == "config:capital_usd"


def test_resolve_nav_still_prefers_totals_over_fallback():
    nav, source = resolve_nav_usd(
        {"nav_usd": 950_000, "nav_source": "test"},
        cli_fallback=1_050_000,
        cli_source="config:capital_usd",
    )
    assert nav == pytest.approx(950_000)
    assert source == "test"


def test_nav_from_config_reads_capital_usd():
    nav, source = _nav_from_config(PROJECT_ROOT / "config" / "strategy_config.yml")
    assert nav > 0
    assert source == "config:capital_usd"


# ── Borrow shock panel ─────────────────────────────────────────────────────


def test_borrow_shock_scales_incremental_carry():
    borrow_panel = {
        "short_etf_rows": [
            {"symbol": "AAA", "short_notional_usd": 100_000, "borrow_rate_pct": 10.0},
            {"symbol": "BBB", "short_notional_usd": -50_000, "borrow_rate_pct": 20.0},
        ]
    }
    panel = compute_borrow_shock_panel(borrow_panel, nav_usd=1_000_000)
    assert panel["available"] is True
    # base carry = 100k*0.10 + 50k*0.20 = 10k + 10k = 20k
    assert panel["current_annual_cost_usd"] == pytest.approx(20_000)
    # x2 shock doubles the carry -> incremental == base
    x2 = next(s for s in panel["scenarios"] if "x2" in s["label"])
    assert x2["incremental_cost_usd"] == pytest.approx(20_000)
    assert x2["incremental_pct_nav"] == pytest.approx(0.02)


def test_borrow_shock_unavailable_when_no_rows():
    panel = compute_borrow_shock_panel({"short_etf_rows": []}, nav_usd=1_000_000)
    assert panel["available"] is False


# ── Drawdown panel ─────────────────────────────────────────────────────────


def test_drawdown_panel_finds_max_drawdown(tmp_path):
    csv = tmp_path / "pnl_history.csv"
    csv.write_text(
        "date,total_pnl\n"
        "2026-01-01,0\n"
        "2026-01-02,10000\n"  # peak equity = 110k
        "2026-01-03,-5000\n"  # equity 95k -> dd from 110k = -15k
        "2026-01-04,2000\n",
        encoding="utf-8",
    )
    panel = compute_drawdown_panel(csv, nav_usd=100_000, run_date="2026-01-04")
    assert panel["available"] is True
    assert panel["max_drawdown_usd"] == pytest.approx(-15_000)
    assert panel["peak_equity_usd"] == pytest.approx(110_000)
    assert panel["current_equity_usd"] == pytest.approx(102_000)


def test_drawdown_panel_filters_future_dates(tmp_path):
    csv = tmp_path / "pnl_history.csv"
    csv.write_text(
        "date,total_pnl\n2026-01-01,0\n2026-01-02,5000\n2026-01-03,99999\n",
        encoding="utf-8",
    )
    panel = compute_drawdown_panel(csv, nav_usd=100_000, run_date="2026-01-02")
    assert panel["n_points"] == 2
    assert panel["current_cum_pnl_usd"] == pytest.approx(5000)


# ── P&L panel ──────────────────────────────────────────────────────────────


def test_pnl_panel_daily_and_weekly_deltas(tmp_path):
    csv = tmp_path / "pnl_history.csv"
    csv.write_text(
        "date,total_pnl,pnl_bucket_1,pnl_bucket_2,pnl_bucket_3,pnl_bucket_4,pnl_bucket_5,pnl_stock_sleeves\n"
        "2026-06-02,0,0,0,0,0,0,0\n"
        "2026-06-03,1000,600,200,0,100,0,100\n"
        "2026-06-04,1500,800,300,0,200,0,200\n"
        "2026-06-09,500,-200,400,0,-500,0,800\n",
        encoding="utf-8",
    )
    panel = compute_pnl_panel(csv, nav_usd=1_000_000, run_date="2026-06-09")
    assert panel["available"] is True
    assert panel["summary"]["daily_usd"] == pytest.approx(-1000)
    assert panel["summary"]["prior_date"] == "2026-06-04"
    assert panel["summary"]["ytd_usd"] == pytest.approx(500)
    last = panel["daily"][-1]
    assert last["buckets"]["bucket_1"] == pytest.approx(-1000)
    assert last["buckets"]["stock_sleeves"] == pytest.approx(600)
    assert panel["weekly"][-1]["daily_usd"] == pytest.approx(-1000)


# ── Shared underlying + movers ─────────────────────────────────────────────


def test_shared_underlying_flags_cross_bucket_names(tmp_path):
    (tmp_path / "net_exposure_bucket_1.csv").write_text(
        "underlying,symbols,n_legs,net_notional_usd,gross_notional_usd\n"
        "SHARED,SHARED,1,1000,1000\nSOLO1,SOLO1,1,500,500\n",
        encoding="utf-8",
    )
    (tmp_path / "net_exposure_bucket_4.csv").write_text(
        "underlying,symbols,n_legs,net_notional_usd,gross_notional_usd\n"
        "SHARED,SHRT,1,-400,400\n",
        encoding="utf-8",
    )
    panel = compute_shared_underlying_panel(tmp_path)
    assert panel["available"] is True
    assert panel["n_shared"] == 1
    assert panel["rows"][0]["underlying"] == "SHARED"
    assert set(panel["rows"][0]["buckets"]) == {"bucket_1", "bucket_4"}


def test_movers_panel_splits_winners_and_losers(tmp_path):
    (tmp_path / "pnl_by_underlying.csv").write_text(
        "underlying,symbols,total_pnl\nA,A,5000\nB,B,-3000\nC,C,100\n",
        encoding="utf-8",
    )
    panel = compute_movers_panel(tmp_path, top_n=5)
    assert panel["available"] is True
    assert panel["winners"][0]["underlying"] == "A"
    assert panel["losers"][0]["underlying"] == "B"


# ── Published snapshot contract (integration) ──────────────────────────────


def _load_snapshot() -> dict:
    if not SNAPSHOT.is_file():
        pytest.skip(f"missing {SNAPSHOT}")
    return json.loads(SNAPSHOT.read_text(encoding="utf-8"))


def test_snapshot_exposes_phase0_4_fields():
    snap = _load_snapshot()
    for key in (
        "freshness",
        "borrow_shock_panel",
        "drawdown_panel",
        "pnl_panel",
        "shared_underlying_panel",
        "movers_panel",
        "display_sleeve_groups",
    ):
        assert key in snap, f"snapshot missing {key}"
    book = snap["book"]
    assert "pnl_ytd_usd" in book and "pnl_daily_usd" in book
    # Bucket 5 must be present everywhere it matters.
    assert "bucket_5" in snap["buckets"]
    assert any(r["bucket"] == "bucket_5" for r in book["sleeve_table"])


def test_snapshot_nav_source_is_labelled():
    snap = _load_snapshot()
    assert snap.get("nav_source")
    assert snap.get("nav_usd", 0) > 0
