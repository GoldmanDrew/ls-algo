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
    compute_bucket_drawdown_panel,
    compute_bucket_movers_panel,
    compute_component_attribution_panel,
    compute_dividend_panel,
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
    assert "periods" in panel
    assert panel["periods"]["today"]["book_usd"] == pytest.approx(-1000)
    assert panel["periods"]["wtd"]["book_usd"] == pytest.approx(-1000)
    assert "recon_ok" in panel["periods"]["today"]
    assert panel["available_dates"] == ["2026-06-03", "2026-06-04", "2026-06-09"]
    assert panel["periods_by_date"]["2026-06-09"]["book_usd"] == pytest.approx(-1000)
    assert len(panel["daily_all"]) == 3


def test_component_attribution_panel_periods(tmp_path):
    csv = tmp_path / "pnl_attribution_history.csv"
    csv.write_text(
        "date,long_realized_pnl,long_unrealized_pnl,short_realized_pnl,short_unrealized_pnl,"
        "gross_realized_pnl,gross_unrealized_pnl,other_fees,borrow_fees,short_credit_interest,"
        "excluded_cash_interest_base,dividends,withholding_tax,pil_dividends,bond_interest,strategy_total_pnl\n"
        "2026-06-02,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0\n"
        "2026-06-03,100,200,0,0,100,200,0,-10,0,0,0,0,0,0,290\n",
        encoding="utf-8",
    )
    panel = compute_component_attribution_panel(csv, nav_usd=1_000_000, run_date="2026-06-03")
    assert panel["available"] is True
    assert panel["periods"]["today"]["total_usd"] == pytest.approx(290)
    assert panel["by_date"]["2026-06-03"]["total_usd"] == pytest.approx(290)


def test_dividend_panel_by_date(tmp_path):
    att = tmp_path / "pnl_attribution_history.csv"
    att.write_text(
        "date,long_realized_pnl,long_unrealized_pnl,short_realized_pnl,short_unrealized_pnl,"
        "gross_realized_pnl,gross_unrealized_pnl,other_fees,borrow_fees,short_credit_interest,"
        "excluded_cash_interest_base,dividends,withholding_tax,pil_dividends,bond_interest,strategy_total_pnl\n"
        "2026-07-01,0,0,0,0,0,0,0,0,0,0,100,0,-1000,0,0\n"
        "2026-07-02,0,0,0,0,0,0,0,0,0,0,103,0,-1329,0,0\n",
        encoding="utf-8",
    )
    div = tmp_path / "dividend_cash_history.csv"
    div.write_text(
        "date,symbol,underlying,bucket,pair,type,category,amount_usd,ex_date,description\n"
        "2026-07-02,CRM,CRM,bucket_1,CRM (spot),Dividends,dividends,3.96,20260702,CRM dividend\n"
        "2026-07-02,MTYY,MSTR,bucket_2,MSTR | MTYY,Payment In Lieu Of Dividends,pil_dividends,-329.35,,PIL\n",
        encoding="utf-8",
    )
    acct = tmp_path / "accounting"
    acct.mkdir()
    (acct / "pnl_by_symbol.csv").write_text(
        "symbol,underlying,bucket,pair,total_pnl\nCRM,CRM,bucket_1,CRM (spot),0\n",
        encoding="utf-8",
    )
    runs = tmp_path / "runs"
    runs.mkdir()
    panel = compute_dividend_panel(
        dividend_cash_history_csv=div,
        attribution_history_csv=att,
        accounting_dir=acct,
        runs_root=runs,
        run_date="2026-07-02",
        nav_usd=1_000_000,
    )
    assert panel["available"] is True
    day = panel["by_date"]["2026-07-02"]
    assert day["net_usd"] == pytest.approx(-325.39)
    assert day["pil_usd"] == pytest.approx(-329.35)
    assert len(day["rows"]) == 2
    assert panel["sparkline"][-1]["net_usd"] == pytest.approx(-325.39)


def test_bucket_movers_by_date(tmp_path):
    hist_csv = tmp_path / "pnl_bucket_underlying_history.csv"
    hist_csv.write_text(
        "date,bucket,underlying,symbols,cum_pnl_usd\n"
        "2026-06-02,bucket_1,AAA,AAA,0\n"
        "2026-06-03,bucket_1,AAA,AAA,500\n"
        "2026-06-04,bucket_1,AAA,AAA,200\n",
        encoding="utf-8",
    )
    pnl_hist = tmp_path / "pnl_history.csv"
    pnl_hist.write_text(
        "date,total_pnl,pnl_bucket_1,pnl_bucket_2,pnl_bucket_3,pnl_bucket_4,pnl_bucket_5,pnl_stock_sleeves\n"
        "2026-06-02,0,0,0,0,0,0,0\n"
        "2026-06-03,500,500,0,0,0,0,500\n"
        "2026-06-04,200,200,0,0,0,0,200\n",
        encoding="utf-8",
    )
    acct = tmp_path / "accounting"
    acct.mkdir()
    for b in ("bucket_1", "bucket_2", "bucket_3", "bucket_4", "bucket_5"):
        (acct / f"pnl_{b}.csv").write_text(
            "underlying,symbols,total_pnl\nAAA,AAA,200\n",
            encoding="utf-8",
        )
        (acct / f"net_exposure_{b}.csv").write_text(
            "underlying,symbols,n_legs,net_notional_usd,gross_notional_usd\n",
            encoding="utf-8",
        )
    panel = compute_bucket_movers_panel(
        acct,
        bucket_underlying_history_csv=hist_csv,
        pnl_history_csv=pnl_hist,
        run_date="2026-06-04",
        top_n=5,
    )
    assert panel["available"] is True
    day = panel["by_date"]["2026-06-04"]
    assert day["by_bucket"]["bucket_1"]["losers"][0]["total_pnl"] == pytest.approx(-300)
    assert panel["book_by_date"]["2026-06-04"]["losers"][0]["underlying"] == "AAA"


def test_bucket_drawdown_panel(tmp_path):
    csv = tmp_path / "pnl_history.csv"
    csv.write_text(
        "date,total_pnl,pnl_bucket_1,pnl_bucket_2,pnl_bucket_3,pnl_bucket_4,pnl_bucket_5,pnl_stock_sleeves\n"
        "2026-06-01,0,0,0,0,0,0,0\n"
        "2026-06-02,100,500,0,0,0,0,500\n"
        "2026-06-03,-200,100,0,0,0,0,100\n",
        encoding="utf-8",
    )
    panel = compute_bucket_drawdown_panel(csv, nav_usd=1_000_000, run_date="2026-06-03")
    assert panel["available"] is True
    b1 = next(r for r in panel["rows"] if r["bucket"] == "bucket_1")
    assert b1["max_drawdown_usd"] == pytest.approx(-400)


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
    (tmp_path / "net_exposure_by_underlying.csv").write_text(
        "underlying,symbols,n_legs,net_notional_usd,gross_notional_usd\n"
        "SHARED,SHARED,2,600,1400\n",
        encoding="utf-8",
    )
    panel = compute_shared_underlying_panel(tmp_path)
    assert panel["available"] is True
    assert panel["n_shared"] == 1
    row = panel["rows"][0]
    assert row["underlying"] == "SHARED"
    assert set(row["buckets"]) == {"bucket_1", "bucket_4"}
    assert row["net_b1_usd"] == pytest.approx(1000.0)
    assert row["net_b4_usd"] == pytest.approx(-400.0)
    assert row["book_net_usd"] == pytest.approx(600.0)
    assert row["recon_diff_usd"] == pytest.approx(0.0)
    assert row["recon_ok"] is True


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
        "bucket_movers_panel",
        "component_attribution_panel",
        "dividend_panel",
        "bucket_drawdown_panel",
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


# ── Site shell: tab layout + B4 book-only ──────────────────────────────────


SITE_ROOT = PROJECT_ROOT / "site"


def test_site_has_dashboard_tab_shell():
    html = (SITE_ROOT / "index.html").read_text(encoding="utf-8")
    for needle in (
        'id="dashboard-tabs"',
        'data-dash-tab="overview"',
        'data-dash-tab="pnl"',
        'data-dash-tab="risk"',
        'data-dash-tab="book"',
        'id="dash-tab-risk"',
        'id="risk-subnav"',
        'id="book-subnav"',
    ):
        assert needle in html, f"index.html missing {needle}"
    assert 'id="subnav"' not in html


def test_app_js_has_lazy_tab_rendering():
    js = (SITE_ROOT / "assets" / "js" / "app.js").read_text(encoding="utf-8")
    for needle in (
        "function switchDashboardTab",
        "function renderTabPanel",
        "function dashParseHash",
        "bindDashboardTabs",
        "filter((p) => p.in_book)",
    ):
        assert needle in js, f"app.js missing {needle}"


def test_b4_risk_sim_build_script_is_book_only():
    src = (PROJECT_ROOT / "scripts" / "build_bucket4_risk_sim.py").read_text(encoding="utf-8")
    assert "proposed book only" in src.lower() or "proposed book only" in src
    assert "gross_target_usd" in src

