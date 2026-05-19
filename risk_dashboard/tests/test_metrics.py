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
    compute_borrow_shock_panel,
    compute_bucket_detail,
    compute_concentration_panel,
    compute_data_quality,
    compute_factor_panel,
    compute_scenario_panel,
    compute_slide_risk_panel,
    compute_vol_shock_panel,
)


@pytest.fixture
def fake_totals() -> dict:
    # Bucket components (b1+b2+b4) must sum to the book aggregate within
    # 1% so the sleeve attribution gate stays green. Bucket 3 is a
    # delta-normalized OVERLAY and intentionally excluded from the sum
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
        # Bucket 3 is a delta-normalized hedge overlay; NOT included in
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


def test_factor_panel_computes_delta_weighted_exposure(tmp_path: Path):
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
    slide = compute_slide_risk_panel(
        factor_panel=factor,
        nav_usd=100_000.0,
    )
    queue = compute_action_queue(
        book=book,
        factor_panel=factor,
        concentration_panel=conc,
        slide_risk_panel=slide,
        borrow_panel={"squeeze_rows": []},
        nav_usd=100_000.0,
    )
    items = queue["items"]
    nvda_actions = [a for a in items if "NVDA" in a.get("title", "")]
    assert nvda_actions, items
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


# ---------------------------------------------------------------------------
# Phase 1: slide-risk strips
# ---------------------------------------------------------------------------


def _slide_factor_panel_fixture() -> dict:
    """Minimal factor_panel payload for slide / vol tests."""
    return {
        "available": True,
        "rows": [
            {
                "underlying": "NVDA",
                "symbols": "NVDA, NVDU",
                "net_notional_usd": 50_000.0,
                "gross_notional_usd": 50_000.0,
                "n_legs": 2,
                "sector": "semis",
                "beta_to_spy": 1.70,
                "beta_to_ndx": 1.50,
                "beta_to_rut": 1.40,
                "regime_vol_pct": 40.0,
                "beta_source": "computed",
            },
            {
                "underlying": "LLY",
                "symbols": "LLY",
                "net_notional_usd": -10_000.0,
                "gross_notional_usd": 10_000.0,
                "n_legs": 1,
                "sector": "healthcare",
                "beta_to_spy": 0.50,
                "beta_to_ndx": 0.30,
                "beta_to_rut": 0.20,
                "regime_vol_pct": 25.0,
                "beta_source": "computed",
            },
        ],
        "totals": {},
    }


def test_compute_slide_risk_panel_produces_spx_and_vix_strips():
    panel = compute_slide_risk_panel(
        factor_panel=_slide_factor_panel_fixture(),
        nav_usd=100_000.0,
        screener_csv=None,
        flex_positions_xml=None,
    )
    assert panel["available"] is True
    indices = {idx["index"]: idx for idx in panel["indices"]}
    assert set(indices) == {"SPX", "VIX"}
    assert indices["VIX"]["strip_type"] == "vix_pts"
    spx = indices["SPX"]
    # SPX -3%: NVDA loses 50k*1.7*0.03 = 2550, LLY gains 10k*0.5*0.03 = 150
    # net = -2400 (note LLY is SHORT so -1*0.5*-0.03 = +0.015 * 10k = +150)
    m3 = next(r for r in spx["shock_rows"] if abs(r["shock_pct"] + 0.03) < 1e-9)
    assert m3["pnl_usd"] == pytest.approx(-2400.0, abs=1.0)
    assert m3["pnl_pct_nav"] == pytest.approx(-0.024, abs=1e-6)
    assert any(h["horizon_days"] == 0 for h in m3["horizons"])
    assert m3["status"] in ("ok", "warn", "hard")


def test_compute_slide_risk_panel_letf_decay_uses_per_leg(tmp_path: Path):
    """Vol decay should be applied only to LETF legs, not spot legs."""
    screener = tmp_path / "screener.csv"
    pd.DataFrame(
        [
            {
                "ETF": "APLX",
                "Underlying": "APLD",
                "Delta": 1.99,
                "Delta_product_class": "letf_long",
                "vol_etf_annual": 1.4,
                "vol_underlying_annual": 0.7,
                "borrow_fee_annual": 0.0,
            },
            {
                "ETF": "APLD",
                "Underlying": "APLD",
                "Delta": 1.0,
                "Delta_product_class": "passive_low_delta",
                "vol_etf_annual": 0.7,
                "vol_underlying_annual": 0.7,
                "borrow_fee_annual": 0.0,
            },
        ]
    ).to_csv(screener, index=False)
    flex = tmp_path / "flex_positions.xml"
    flex.write_text(
        '<FlexQueryResponse>'
        '<OpenPosition symbol="APLD" position="20000" markPrice="50" '
        'positionValue="1000000" underlyingSymbol="APLD" fxRateToBase="1" multiplier="1" />'
        '<OpenPosition symbol="APLX" position="32000" markPrice="31.25" '
        'positionValue="1000000" underlyingSymbol="APLX" fxRateToBase="1" multiplier="1" />'
        '</FlexQueryResponse>',
        encoding="utf-8",
    )
    panel = compute_slide_risk_panel(
        factor_panel={
            "available": True,
            "rows": [
                {
                    "underlying": "APLD",
                    "symbols": "APLD, APLX",
                    "net_notional_usd": 2_000_000.0,
                    "gross_notional_usd": 2_000_000.0,
                    "beta_to_spy": 1.5,
                    "beta_to_ndx": 1.3,
                    "beta_to_rut": 1.2,
                    "regime_vol_pct": 70.0,
                    "beta_source": "computed",
                }
            ],
        },
        nav_usd=1_000_000.0,
        screener_csv=screener,
        flex_positions_xml=flex,
    )
    assert panel["available"] is True
    spx = next(idx for idx in panel["indices"] if idx["index"] == "SPX")
    # T+20 decay applied only to APLX (k=2, vol=1.4); APLD spot leg contributes 0.
    # Expected: $1M * -0.5*2*1*1.96*(20/252) ~= -$155.5k
    h20 = next(h for h in spx["shock_rows"][0]["horizons"] if h["horizon_days"] == 20)
    assert -200_000 < h20["decay_usd"] < -100_000, h20["decay_usd"]


# ---------------------------------------------------------------------------
# Phase 3: borrow shock sensitivity
# ---------------------------------------------------------------------------


def test_compute_borrow_shock_panel_applies_abs_and_mult_shocks(tmp_path: Path):
    flex = tmp_path / "flex_positions.xml"
    flex.write_text(
        '<FlexQueryResponse>'
        '<OpenPosition symbol="TSLZ" position="-5000" markPrice="20" '
        'positionValue="-100000" underlyingSymbol="TSLA" fxRateToBase="1" multiplier="1" />'
        '<OpenPosition symbol="MSTZ" position="-2000" markPrice="50" '
        'positionValue="-100000" underlyingSymbol="MSTR" fxRateToBase="1" multiplier="1" />'
        '</FlexQueryResponse>',
        encoding="utf-8",
    )
    screener = tmp_path / "screener.csv"
    pd.DataFrame(
        [
            {"ETF": "TSLZ", "borrow_fee_annual": 0.10},
            {"ETF": "MSTZ", "borrow_fee_annual": 0.05},
        ]
    ).to_csv(screener, index=False)

    borrow_panel = {
        "borrow": {"fee_rate_by_symbol": {}},
        "squeeze_rows": [],
    }
    panel = compute_borrow_shock_panel(
        borrow_panel=borrow_panel,
        flex_positions_xml=flex,
        nav_usd=1_000_000.0,
        screener_csv=screener,
        abs_shocks_bp=(100, 500),
        mult_shocks=(2.0,),
    )
    assert panel["available"] is True
    assert panel["current_annual_cost_usd"] == pytest.approx(15_000.0)
    abs_100 = next(r for r in panel["abs_ladder"] if r["shock"] == 100)
    assert abs_100["annual_delta_usd"] == pytest.approx(-2000.0, abs=1.0)
    assert abs_100["persistence_delta_usd"] == pytest.approx(
        abs_100["annual_delta_usd"] / 252.0 * 30.0, rel=1e-9
    )
    mult_2 = next(r for r in panel["mult_ladder"] if r["shock"] == 2.0)
    assert mult_2["annual_delta_usd"] == pytest.approx(-15_000.0, abs=1.0)
    assert mult_2["worst_victims"][0]["symbol"] == "TSLZ"


# ---------------------------------------------------------------------------
# Phase 4: VIX / vol shock sensitivity
# ---------------------------------------------------------------------------


def test_compute_vol_shock_panel_letf_decay_signed_by_net_notional(tmp_path: Path):
    """Long-LETF positions LOSE on vol spike."""
    screener = tmp_path / "screener.csv"
    pd.DataFrame(
        [
            {
                "ETF": "AAPU",
                "Underlying": "AAPL",
                "Delta": 2.0,
                "Delta_product_class": "letf_long",
                "vol_etf_annual": 0.5,
                "vol_underlying_annual": 0.25,
                "borrow_fee_annual": 0.0,
            }
        ]
    ).to_csv(screener, index=False)
    flex = tmp_path / "flex_positions.xml"
    flex.write_text(
        '<FlexQueryResponse>'
        '<OpenPosition symbol="AAPU" position="1000" markPrice="100" '
        'positionValue="100000" underlyingSymbol="AAPL" fxRateToBase="1" multiplier="1" />'
        '</FlexQueryResponse>',
        encoding="utf-8",
    )
    factor_panel = {
        "available": True,
        "rows": [
            {
                "underlying": "AAPL",
                "symbols": "AAPU",
                "net_notional_usd": 100_000.0,
                "gross_notional_usd": 100_000.0,
                "beta_to_spy": 2.0,
                "regime_vol_pct": 25.0,
            }
        ],
    }
    panel = compute_vol_shock_panel(
        factor_panel=factor_panel,
        nav_usd=1_000_000.0,
        screener_csv=screener,
        flex_positions_xml=flex,
        vix_shocks=(5,),
        vol_multipliers=(2.0,),
        decay_horizon_days=20,
    )
    assert panel["available"] is True
    vol2 = panel["vol_ladder"][0]
    assert vol2["pnl_usd"] < 0
    assert vol2["worst_victims"][0]["underlying"] == "AAPL"


# ---------------------------------------------------------------------------
# Phase I: beta loader integration
# ---------------------------------------------------------------------------


def test_compute_factor_panel_uses_computed_betas_when_provided(tmp_path: Path):
    csv = tmp_path / "net_exposure_by_underlying.csv"
    csv.write_text(
        "underlying,symbols,net_notional_usd,gross_notional_usd,n_legs\n"
        "NVDA,NVDA,50000,50000,1\n",
        encoding="utf-8",
    )
    panel = compute_factor_panel(
        csv,
        nav_usd=100_000.0,
        beta_results={
            "NVDA": {
                "provenance": "computed",
                "beta_to_spy": 1.95,
                "beta_to_ndx": 1.70,
                "beta_to_rut": 1.55,
                "delta_se": 0.05,
                "n_obs": 60,
                "r2": 0.72,
                "regime_vol_pct": 38.5,
                "shrinkage_applied": False,
            }
        },
    )
    assert panel["available"] is True
    nvda = next(r for r in panel["rows"] if r["underlying"] == "NVDA")
    assert nvda["beta_to_spy"] == pytest.approx(1.95)
    assert nvda["beta_to_qqq"] == pytest.approx(1.70)
    assert nvda["beta_to_iwm"] == pytest.approx(1.55)
    assert nvda["beta_source"] == "computed"
    assert nvda["regime_vol_pct"] == pytest.approx(38.5)
    assert nvda["beta_n_obs"] == 60
    counts = panel["beta_provenance_counts"]
    assert counts.get("computed") == 1

