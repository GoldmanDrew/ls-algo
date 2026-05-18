"""Smoke tests for risk_dashboard.metrics."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from risk_dashboard.factor_map import lookup_underlying
from risk_dashboard.metrics import (
    DEFAULT_LIMITS,
    SLEEVE_TARGET_WEIGHTS,
    compute_action_queue,
    compute_book_summary,
    compute_bucket_detail,
    compute_concentration_panel,
    compute_data_quality,
    compute_factor_panel,
    compute_scenario_panel,
)


@pytest.fixture
def fake_totals() -> dict:
    # Bucket components (b1+b2+b4) must sum to the book aggregate within
    # 1% so the sleeve attribution gate stays green. Bucket 3 is a
    # beta-normalized OVERLAY and intentionally excluded from the sum
    # (mirrors the upstream accounting reconciliation gate after the
    # Phase G fix in ``ibkr_accounting.py``).
    b1_g, b2_g, b4_g = 3_966_574.48, 228_552.80, 437_084.68
    b1_n, b2_n, b4_n = -707_601.71, -44_860.52, 437_084.68
    return {
        "run_date": "2026-05-15",
        "total_pnl": 48626.58,
        "net_exposure_total": b1_n + b2_n + b4_n,
        "gross_exposure_total": b1_g + b2_g + b4_g,
        "net_exposure_bucket_1": b1_n,
        "gross_exposure_bucket_1": b1_g,
        "net_exposure_bucket_2": b2_n,
        "gross_exposure_bucket_2": b2_g,
        # Bucket 3 is a beta-normalized hedge overlay; NOT included in
        # the gross/net reconciliation sum on purpose.
        "net_exposure_bucket_3": 85047.29,
        "gross_exposure_bucket_3": 86236.36,
        "net_exposure_bucket_4": b4_n,
        "gross_exposure_bucket_4": b4_g,
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
    expected_gross = fake_totals["gross_exposure_total"]
    assert book.gross_notional_usd == pytest.approx(expected_gross)
    assert book.gross_exposure_pct_nav == pytest.approx(expected_gross / 800_000.0)
    assert book.pnl_today_pct_nav == pytest.approx(48626.58 / 800_000.0)
    assert len(book.sleeve_table) == 4
    b4 = next(r for r in book.sleeve_table if r["bucket"] == "bucket_4")
    assert b4["target_weight"] == 0.25
    assert b4["actual_weight"] == pytest.approx(437084.68 / expected_gross)


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
            # b1+b2+b4 must sum to book; b3 is an overlay (not in sum).
            "gross_exposure_total": 3.0,
            "net_exposure_total": 3.0,
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


def test_sleeve_attribution_hidden_when_buckets_dont_reconcile():
    broken = {
        "gross_exposure_total": 1_000_000.0,
        "net_exposure_total": -500_000.0,
        "gross_exposure_bucket_1": 30_000_000.0,
        "gross_exposure_bucket_2": 1.0,
        "gross_exposure_bucket_3": 1.0,
        "gross_exposure_bucket_4": 1.0,
        "net_exposure_bucket_1": -10_000_000.0,
        "net_exposure_bucket_2": 0.0,
        "net_exposure_bucket_3": 0.0,
        "net_exposure_bucket_4": 0.0,
        "bucket_pnl": {"bucket_1": 100.0},
    }
    book = compute_book_summary(
        totals=broken,
        pnl_by_bucket=pd.DataFrame(),
        nav_usd=20_000_000.0,
    )
    assert book.sleeve_attribution_available is False
    assert "do not reconcile" in book.sleeve_attribution_reason
    for row in book.sleeve_table:
        assert row["gross_usd"] is None
        assert row["net_usd"] is None
        assert row["actual_weight"] is None
        assert row["drift_pp"] is None
        assert row["drift_status"] == "unknown"
        assert row["attribution_available"] is False
    pnl_total = sum(r["pnl_usd"] for r in book.sleeve_table)
    assert pnl_total == pytest.approx(100.0)


def test_sleeve_attribution_visible_when_buckets_reconcile(fake_totals):
    book = compute_book_summary(
        totals=fake_totals,
        pnl_by_bucket=pd.DataFrame(),
        nav_usd=800_000.0,
    )
    assert book.sleeve_attribution_available is True
    assert book.sleeve_attribution_reason == ""
    for row in book.sleeve_table:
        assert row["gross_usd"] is not None
        assert row["attribution_available"] is True


def test_scenario_panel_carries_book_only_badge():
    panel = compute_scenario_panel(
        buckets={
            "book": {
                "exposure_rows": [
                    {
                        "underlying": "ABC",
                        "symbols": "ABC",
                        "net_notional_usd": 1000.0,
                        "gross_notional_usd": 1000.0,
                    }
                ],
                "pnl_rows": [],
            }
        },
        nav_usd=10_000.0,
        book_only_mode=True,
        book_only_reason="bucket reconciliation broken",
    )
    assert panel["book_only_mode"] is True
    assert "bucket reconciliation broken" in panel["book_only_reason"]
    market = next(s for s in panel["scenarios"] if s["id"] == "market_down_5")
    assert "book" in market["bucket_pnl"]


def test_factor_panel_computes_beta_weighted_exposure(tmp_path: Path):
    csv = tmp_path / "net_exposure_by_underlying.csv"
    csv.write_text(
        "underlying,symbols,net_notional_usd,gross_notional_usd,n_legs\n"
        "NVDA,\"NVDU, NVDA\",10000,15000,2\n"
        "MSTR,\"MSTU, MSTR\",-5000,8000,2\n"
        "ZZUNK,ZZUNK,1000,1000,1\n",
        encoding="utf-8",
    )
    panel = compute_factor_panel(csv, nav_usd=100_000.0)
    assert panel["available"] is True
    rows = {r["underlying"]: r for r in panel["rows"]}
    assert rows["NVDA"]["beta_to_spy"] == pytest.approx(1.70)
    assert rows["NVDA"]["beta_source"] == "curated"
    assert rows["NVDA"]["beta_weighted_net_usd"] == pytest.approx(10000 * 1.70)
    assert rows["MSTR"]["beta_weighted_net_usd"] == pytest.approx(-5000 * 2.80)
    # Unknown ticker -> default beta + default source.
    assert rows["ZZUNK"]["beta_source"] == "default"
    assert rows["ZZUNK"]["sector"] == "other"
    totals = panel["totals"]
    assert totals["net_beta_to_spy"] == pytest.approx(
        (10000 * 1.70 + -5000 * 2.80 + 1000 * 1.20) / 100_000.0
    )
    assert 0 < totals["beta_coverage_gross_pct"] < 1.0


def test_factor_map_lookup_defaults_safe():
    out = lookup_underlying("DEFINITELY_NOT_A_TICKER_42")
    assert out["sector"] == "other"
    assert out["beta_to_spy"] > 0
    assert out["beta_source"] == "default"


def test_scenario_panel_appends_beta_scenarios(tmp_path: Path):
    csv = tmp_path / "net_exposure_by_underlying.csv"
    csv.write_text(
        "underlying,symbols,net_notional_usd,gross_notional_usd,n_legs\n"
        "AAPL,AAPL,5000,5000,1\n",
        encoding="utf-8",
    )
    factor = compute_factor_panel(csv, nav_usd=100_000.0)
    panel = compute_scenario_panel(
        buckets={"book": {"exposure_rows": [], "pnl_rows": []}},
        nav_usd=100_000.0,
        book_only_mode=True,
        factor_panel=factor,
    )
    beta_ids = [s["id"] for s in panel["scenarios"] if s["id"].startswith("spx_beta_")]
    assert "spx_beta_down_5" in beta_ids
    spx_down_5 = next(s for s in panel["scenarios"] if s["id"] == "spx_beta_down_5")
    assert spx_down_5["pnl_usd"] == pytest.approx(5000 * 1.20 * -0.05)


def test_concentration_panel_flags_single_name_over_cap(tmp_path: Path):
    csv = tmp_path / "net_exposure_by_underlying.csv"
    csv.write_text(
        "underlying,symbols,net_notional_usd,gross_notional_usd,n_legs\n"
        "NVDA,NVDA,90000,90000,1\n"
        "AAPL,AAPL,10000,10000,1\n",
        encoding="utf-8",
    )
    factor = compute_factor_panel(csv, nav_usd=100_000.0)
    panel = compute_concentration_panel(factor, nav_usd=100_000.0)
    assert panel["available"] is True
    nvda = next(r for r in panel["top_names"] if r["underlying"] == "NVDA")
    assert nvda["status"] == "hard"
    assert nvda["pct_nav_gross"] == pytest.approx(0.90)
    metrics_in_breaches = {b["metric"] for b in panel["breaches"]}
    assert "single_name:NVDA" in metrics_in_breaches
    assert "top10_gross_pct_nav" in metrics_in_breaches
    assert panel["totals"]["hhi_underlying"] > 0


def test_action_queue_emits_quantitative_trim(tmp_path: Path):
    csv = tmp_path / "net_exposure_by_underlying.csv"
    csv.write_text(
        "underlying,symbols,net_notional_usd,gross_notional_usd,n_legs\n"
        "NVDA,NVDA,30000,30000,1\n"
        "OTHER,OTHER,10000,10000,1\n",
        encoding="utf-8",
    )
    factor = compute_factor_panel(csv, nav_usd=100_000.0)
    conc = compute_concentration_panel(factor, nav_usd=100_000.0)
    book = compute_book_summary(
        totals={
            "gross_exposure_total": 40_000.0,
            "net_exposure_total": 40_000.0,
            "bucket_pnl": {},
            "gross_exposure_bucket_1": 40_000.0,
            "gross_exposure_bucket_2": 0,
            "gross_exposure_bucket_3": 0,
            "gross_exposure_bucket_4": 0,
            "net_exposure_bucket_1": 40_000.0,
            "net_exposure_bucket_2": 0,
            "net_exposure_bucket_3": 0,
            "net_exposure_bucket_4": 0,
        },
        pnl_by_bucket=pd.DataFrame(),
        nav_usd=100_000.0,
    )
    scenario = compute_scenario_panel(
        buckets={"book": {"exposure_rows": [], "pnl_rows": []}},
        nav_usd=100_000.0,
        book_only_mode=True,
        factor_panel=factor,
    )
    queue = compute_action_queue(
        book=book,
        factor_panel=factor,
        concentration_panel=conc,
        scenario_panel=scenario,
        borrow_panel={"squeeze_rows": []},
        nav_usd=100_000.0,
    )
    nvda_actions = [a for a in queue if "NVDA" in a.get("title", "")]
    assert nvda_actions, queue
    a = nvda_actions[0]
    assert a["status"] == "hard"
    assert "$25,000" in a["detail"]
    assert a["priority"] == 0


def test_data_quality_emits_drilldown_payload_when_sources_missing(tmp_path: Path):
    """Phase H: when files are missing, ``compute_data_quality`` must
    surface enough detail (per-source path, missing columns, blanks)
    for the UI drill-down to render an actionable list."""
    accounting = tmp_path / "accounting"
    flex = tmp_path / "ibkr_flex"
    accounting.mkdir()
    flex.mkdir()
    # Provide only totals.json; everything else is intentionally missing.
    (accounting / "totals.json").write_text("{}", encoding="utf-8")
    # Provide ONE bucket CSV with malformed schema to force a missing-
    # column error in the drill-down list.
    (accounting / "pnl_bucket_1.csv").write_text(
        "wrong_col,another_col\n1,2\n",
        encoding="utf-8",
    )

    dq = compute_data_quality(
        accounting_dir=accounting,
        flex_dir=flex,
        buckets={
            "bucket_1": {
                "n_pnl_rows": 1,
                "n_exposure_rows": 0,
                "winners": [{"display_name": "", "description": ""}],
                "losers": [],
                "exposure_rows": [{"underlying": "", "symbols": ""}],
            }
        },
        totals={"gross_exposure_total": 100.0, "net_exposure_total": 50.0},
        run_date="2026-05-18",
    )

    assert dq["missing_source_count"] >= 1, dq
    src_by_name = {s["name"]: s for s in dq["sources"]}
    missing_paths = [s["path"] for s in dq["sources"] if not s["exists"]]
    assert any("net_exposure_bucket_1.csv" in p for p in missing_paths), missing_paths
    bad_pnl = src_by_name.get("pnl_bucket_1") or {}
    assert "total_pnl" in bad_pnl.get("missing_required_columns", []), bad_pnl
    assert dq["blank_render_field_count"] >= 2
    blank_fields = {b["field"] for b in dq["blank_render_fields"]}
    assert {"display_name", "underlying"}.issubset(blank_fields)


def test_live_snapshot_reconciles_bucket_to_book():
    """Regression: bucket gross/net (b1+b2+b4) must sum to book aggregate
    within 1% on the LIVE on-disk snapshot. This catches plan-aware /
    orphan-underlying double-counting bugs that would silently inflate
    bucket sums vs the book file (see Phase G fix in ibkr_accounting.py)."""
    totals_path = Path("data/runs/2026-05-18/accounting/totals.json")
    if not totals_path.exists():
        pytest.skip(f"live snapshot not present: {totals_path}")
    totals = json.loads(totals_path.read_text(encoding="utf-8"))
    book_g = float(totals.get("gross_exposure_total", 0.0))
    book_n = float(totals.get("net_exposure_total", 0.0))
    bucket_g = sum(
        float(totals.get(f"gross_exposure_bucket_{i}", 0.0)) for i in (1, 2, 4)
    )
    bucket_n = sum(
        float(totals.get(f"net_exposure_bucket_{i}", 0.0)) for i in (1, 2, 4)
    )
    assert abs(book_g) > 0, "snapshot has zero book gross"
    gross_diff_pct = abs(bucket_g - book_g) / abs(book_g)
    net_diff_pct = abs(bucket_n - book_n) / abs(book_g)
    assert gross_diff_pct < 0.01, (
        f"bucket gross {bucket_g:,.0f} does not reconcile to book {book_g:,.0f} "
        f"(diff {gross_diff_pct:.2%})"
    )
    assert net_diff_pct < 0.01, (
        f"bucket net {bucket_n:,.0f} does not reconcile to book {book_n:,.0f} "
        f"(diff {net_diff_pct:.2%})"
    )


def test_default_limits_are_sane():
    for k, v in DEFAULT_LIMITS.items():
        assert "warn" in v and "hard" in v, k
    # Sleeve targets must sum (within rounding) to roughly 1.0 when bucket_3
    # is excluded (b3 is layered, not a fixed slice).
    fixed = sum(
        v for k, v in SLEEVE_TARGET_WEIGHTS.items() if v is not None and k != "bucket_3"
    )
    assert 0.95 <= fixed <= 1.05
