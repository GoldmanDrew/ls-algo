"""Tests for the EOD per-pair B4 PnL + hedge chart helpers."""
from __future__ import annotations

import pandas as pd
import pytest

from scripts.bucket4_eod_pair_charts import (
    B5_SLEEVE,
    _build_prices_flexible,
    _pick_etf_px_from_row,
    _reconcile_actual_endpoints,
    _trim_leading_flat_history,
    load_active_b4_pairs_from_proposed,
    load_actual_trade_markers,
    load_b4_pair_leg_history,
    load_pair_gross_and_realized_h,
    load_pair_leg_history,
    make_b4_pair_pnl_hedge_chart,
    resolve_b4_model_inputs,
)


def _write_proposed(root, ds, rows):
    run = root / ds
    run.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(run / "proposed_trades.csv", index=False)


def _write_run(root, ds, *, pair_pnl, etf_pnl, etf_gross=10_000.0, und_gross=4_500.0, delta=-1.0):
    acct = root / ds / "accounting"
    acct.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"etf": "QBTZ", "underlying": "QBTS", "delta": delta, "total_pnl": pair_pnl},
        ]
    ).to_csv(acct / "pnl_bucket_4_by_pair.csv", index=False)
    pd.DataFrame(
        [
            {"symbol": "QBTZ", "total_pnl": etf_pnl},
            {"symbol": "QBTS", "total_pnl": pair_pnl - etf_pnl},
        ]
    ).to_csv(acct / "pnl_bucket_4_by_symbol.csv", index=False)
    pd.DataFrame(
        [
            {"underlying": "QBTS", "symbol": "QBTZ", "leg_type": "etf", "gross_notional_usd": etf_gross},
            {"underlying": "QBTS", "symbol": "QBTS", "leg_type": "underlying", "gross_notional_usd": und_gross},
        ]
    ).to_csv(acct / "net_exposure_bucket_4_detail.csv", index=False)


def _write_b5_run(
    root,
    ds,
    *,
    pair_pnl,
    etf_pnl,
    etf_gross=18_000.0,
    und_gross=22_000.0,
    delta=-1.99,
):
    acct = root / ds / "accounting"
    acct.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"etf": "UVIX", "underlying": "SVIX", "delta": delta, "total_pnl": pair_pnl},
        ]
    ).to_csv(acct / "pnl_bucket_5_by_pair.csv", index=False)
    pd.DataFrame(
        [
            {"symbol": "UVIX", "total_pnl": etf_pnl},
            {"symbol": "SVIX", "total_pnl": pair_pnl - etf_pnl},
        ]
    ).to_csv(acct / "pnl_bucket_5_by_symbol.csv", index=False)
    pd.DataFrame(
        [
            {"underlying": "SVIX", "symbol": "UVIX", "leg_type": "etf", "gross_notional_usd": etf_gross},
            {"underlying": "SVIX", "symbol": "SVIX", "leg_type": "underlying", "gross_notional_usd": und_gross},
        ]
    ).to_csv(acct / "net_exposure_bucket_5_detail.csv", index=False)
    # B4 placeholder row (legacy zero) should be ignored when sleeve is B5.
    pd.DataFrame(
        [
            {"etf": "UVIX", "underlying": "SVIX", "delta": delta, "total_pnl": 0.0},
        ]
    ).to_csv(acct / "pnl_bucket_4_by_pair.csv", index=False)


def _write_flex_trades(root, ds, trades):
    flex = root / ds / "ibkr_flex"
    flex.mkdir(parents=True, exist_ok=True)
    body = "\n".join(
        (
            f'<Trade dateTime="{t["dateTime"]}" symbol="{t["symbol"]}" '
            f'underlyingSymbol="{t.get("underlyingSymbol", "")}" buySell="{t.get("buySell", "SELL")}" '
            f'quantity="{t.get("quantity", 1)}" fifoPnlRealized="0" tradePrice="10" '
            f'fxRateToBase="1" orderReference="{t.get("orderReference", "")}" openCloseIndicator="O" />'
        )
        for t in trades
    )
    (flex / "flex_trades.xml").write_text(f"<FlexQueryResponse><Trades>{body}</Trades></FlexQueryResponse>")


def test_active_pairs_from_proposed_filters_positive_b4_and_dedupes(tmp_path):
    _write_proposed(
        tmp_path,
        "2026-06-22",
        [
            {"ETF": " qbtz ", "Underlying": "qbts", "sleeve": "inverse_decay_bucket4", "gross_target_usd": 100.0, "Delta": -2.0},
            {"ETF": "QBTZ", "Underlying": "QBTS", "sleeve": "inverse_decay_bucket4", "gross_target_usd": 250.0, "Delta": -2.0},
            {"ETF": "ZERO", "Underlying": "ZZZ", "sleeve": "inverse_decay_bucket4", "gross_target_usd": 0.0, "Delta": -1.0},
            {"ETF": "CORE", "Underlying": "AAA", "sleeve": "core_leveraged", "gross_target_usd": 999.0, "Delta": 2.0},
        ],
    )
    active = load_active_b4_pairs_from_proposed("2026-06-22", runs_root=tmp_path)
    assert list(active["pair"]) == ["QBTZ|QBTS"]
    assert active.iloc[0]["gross_target_usd"] == pytest.approx(250.0)
    assert active.iloc[0]["etf"] == "QBTZ"


def test_active_pairs_from_proposed_includes_positive_bucket5(tmp_path):
    _write_proposed(
        tmp_path,
        "2026-06-22",
        [
            {
                "ETF": "UVIX",
                "Underlying": "VIX",
                "sleeve": "volatility_etp_bucket5",
                "gross_target_usd": 100.0,
                "Delta": -1.0,
                "und_trend_ratio_fwd_60d": 0.72,
            },
            {
                "ETF": "SVIX",
                "Underlying": "SHORTVOL",
                "sleeve": "volatility_etp_bucket5",
                "gross_target_usd": 0.0,
                "Delta": -1.0,
            },
        ],
    )
    active = load_active_b4_pairs_from_proposed("2026-06-22", runs_root=tmp_path)
    assert list(active["pair"]) == ["UVIX|VIX"]
    assert active.iloc[0]["sleeve"] == "volatility_etp_bucket5"
    assert active.iloc[0]["sizing_tr_fwd"] == pytest.approx(0.72)


def test_leg_history_splits_pair_into_etf_and_underlying(tmp_path):
    _write_run(tmp_path, "2026-06-01", pair_pnl=1000.0, etf_pnl=700.0)
    _write_run(tmp_path, "2026-06-02", pair_pnl=1500.0, etf_pnl=900.0)
    hist = load_b4_pair_leg_history(tmp_path)
    assert len(hist) == 2
    last = hist.sort_values("date").iloc[-1]
    assert last["pair"] == "QBTZ|QBTS"
    assert last["pair_pnl_cum"] == pytest.approx(1500.0)
    assert last["etf_leg_pnl_cum"] == pytest.approx(900.0)
    assert last["und_leg_pnl_cum"] == pytest.approx(600.0)


def test_b5_leg_history_reads_bucket5_files_not_b4_zeros(tmp_path):
    _write_proposed(
        tmp_path,
        "2026-07-01",
        [
            {
                "ETF": "UVIX",
                "Underlying": "SVIX",
                "sleeve": B5_SLEEVE,
                "gross_target_usd": 1_600_000.0,
                "Delta": -1.99,
            },
        ],
    )
    _write_b5_run(tmp_path, "2026-07-01", pair_pnl=290.0, etf_pnl=1622.0)
    active = load_active_b4_pairs_from_proposed("2026-07-01", runs_root=tmp_path)
    hist = load_pair_leg_history(tmp_path, active=active)
    assert len(hist) == 1
    row = hist.iloc[0]
    assert row["pair"] == "UVIX|SVIX"
    assert row["pair_pnl_cum"] == pytest.approx(290.0)
    assert row["etf_leg_pnl_cum"] == pytest.approx(1622.0)
    assert row["und_leg_pnl_cum"] == pytest.approx(-1332.0)


def test_b5_gross_history_uses_bucket5_detail(tmp_path):
    _write_b5_run(tmp_path, "2026-07-01", pair_pnl=0.0, etf_pnl=0.0, etf_gross=18_684.0, und_gross=22_139.0)
    g = load_pair_gross_and_realized_h(tmp_path, "2026-07-01", sleeve=B5_SLEEVE)
    assert len(g) == 1
    assert g.iloc[0]["pair"] == "UVIX|SVIX"
    assert g.iloc[0]["etf_gross_usd"] == pytest.approx(18_684.0)


def test_b5_reconcile_endpoints_uses_bucket5_pair_file(tmp_path):
    _write_proposed(
        tmp_path,
        "2026-07-01",
        [
            {
                "ETF": "UVIX",
                "Underlying": "SVIX",
                "sleeve": B5_SLEEVE,
                "gross_target_usd": 100.0,
                "Delta": -1.99,
            },
        ],
    )
    _write_b5_run(tmp_path, "2026-07-01", pair_pnl=290.0, etf_pnl=1622.0)
    active = load_active_b4_pairs_from_proposed("2026-07-01", runs_root=tmp_path)
    ok = _reconcile_actual_endpoints(
        {"UVIX|SVIX": 290.0},
        runs_root=tmp_path,
        run_date="2026-07-01",
        active=active,
    )
    assert ok == []


def test_trim_leading_flat_history_keeps_one_zero_anchor():
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2026-01-30", "2026-02-15", "2026-06-06", "2026-06-07"]
            ),
            "pair_pnl_cum": [0.0, 0.0, -1591.0, -1800.0],
        }
    )
    out = _trim_leading_flat_history(df)
    # Drops the long leading flat-zero stretch but keeps a single 0 anchor
    # immediately before the first economically meaningful row.
    assert list(out["date"].dt.strftime("%Y-%m-%d")) == [
        "2026-02-15",
        "2026-06-06",
        "2026-06-07",
    ]
    assert out["pair_pnl_cum"].iloc[0] == pytest.approx(0.0)


def test_trim_leading_flat_history_all_zero_keeps_last():
    df = pd.DataFrame(
        {"date": pd.to_datetime(["2026-01-01", "2026-01-02"]), "pair_pnl_cum": [0.0, 0.0]}
    )
    out = _trim_leading_flat_history(df)
    assert len(out) == 1


def test_reconcile_actual_endpoints_flags_mismatch(tmp_path, capsys):
    _write_run(tmp_path, "2026-06-02", pair_pnl=1500.0, etf_pnl=900.0)
    # Matching endpoint -> no warning; mismatched endpoint -> warning + return.
    ok = _reconcile_actual_endpoints(
        {"QBTZ|QBTS": 1500.0}, runs_root=tmp_path, run_date="2026-06-02"
    )
    assert ok == []
    bad = _reconcile_actual_endpoints(
        {"QBTZ|QBTS": 999.0}, runs_root=tmp_path, run_date="2026-06-02"
    )
    assert len(bad) == 1 and "QBTZ|QBTS" in bad[0]
    assert "endpoint mismatch" in capsys.readouterr().out


def test_realized_h_is_und_gross_over_beta_times_etf_gross(tmp_path):
    _write_run(
        tmp_path,
        "2026-06-01",
        pair_pnl=0.0,
        etf_pnl=0.0,
        etf_gross=10_000.0,
        und_gross=4_500.0,
        delta=-1.0,
    )
    g = load_pair_gross_and_realized_h(tmp_path, "2026-06-01")
    row = g.iloc[0]
    assert row["realized_h"] == pytest.approx(0.45)
    assert row["etf_gross_usd"] == pytest.approx(10_000.0)


def test_actual_trade_markers_map_etf_and_shared_underlying(tmp_path):
    active = pd.DataFrame(
        [
            {"etf": "QBTZ", "underlying": "QBTS", "pair": "QBTZ|QBTS", "gross_target_usd": 1000.0},
            {"etf": "ABCZ", "underlying": "QBTS", "pair": "ABCZ|QBTS", "gross_target_usd": 500.0},
        ]
    )
    _write_flex_trades(
        tmp_path,
        "2026-06-22",
        [
            {"dateTime": "2026-06-22;15:30:00", "symbol": "QBTZ", "underlyingSymbol": "QBTS"},
            {"dateTime": "2026-06-22;15:35:00", "symbol": "QBTS", "underlyingSymbol": "QBTS"},
        ],
    )
    markers = load_actual_trade_markers(tmp_path, active)
    etf_markers = markers[markers["marker_type"].eq("actual_etf_trade")]
    shared = markers[markers["marker_type"].eq("actual_underlying_trade_shared")]
    assert list(etf_markers["pair"]) == ["QBTZ|QBTS"]
    assert set(shared["pair"]) == {"QBTZ|QBTS", "ABCZ|QBTS"}


def test_resolve_b4_model_inputs_snapshots_local_dashboard_data(tmp_path, monkeypatch):
    monkeypatch.delenv("EOD_B4_METRICS_CSV", raising=False)
    monkeypatch.delenv("EOD_B4_VOL_SHAPE_JSON", raising=False)
    dash = tmp_path / "dashboard" / "data"
    dash.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "date": "2026-01-02",
                "ticker": "QBTZ",
                "close_price": 10.0,
                "nav": 10.0,
                "etf_adj_close": 10.0,
                "underlying_adj_close": 20.0,
            }
        ]
    ).to_csv(dash / "etf_metrics_daily.csv", index=False)
    (dash / "vol_shape_history.json").write_text("{}", encoding="utf-8")

    resolved = resolve_b4_model_inputs(
        "2026-06-22",
        runs_root=tmp_path / "runs",
        dashboard_data_dirs=(dash,),
        allow_download=False,
    )

    assert resolved.metrics == tmp_path / "runs" / "2026-06-22" / "model_inputs" / "etf_metrics_daily.csv"
    assert resolved.metrics.exists()
    assert resolved.vol_shape == tmp_path / "runs" / "2026-06-22" / "model_inputs" / "vol_shape_history.json"
    assert resolved.vol_shape.exists()
    assert resolved.metrics_source.startswith("local-dashboard:")
    assert resolved.vol_shape_source.startswith("local-dashboard:")


def test_resolve_b4_model_inputs_preserves_missing_explicit_path(tmp_path, monkeypatch):
    monkeypatch.delenv("EOD_B4_METRICS_CSV", raising=False)
    missing = tmp_path / "missing_metrics.csv"
    resolved = resolve_b4_model_inputs(
        "2026-06-22",
        runs_root=tmp_path / "runs",
        metrics_path=missing,
        dashboard_data_dirs=(tmp_path / "empty",),
        allow_download=False,
    )
    assert resolved.metrics == missing
    assert resolved.metrics_source.startswith("missing-explicit:")
    assert "explicit metrics path missing" in " | ".join(resolved.diagnostics)


def test_chart_builder_fails_soft_on_conflicted_proposed_trades(tmp_path):
    run = tmp_path / "2026-06-01"
    run.mkdir(parents=True)
    (run / "proposed_trades.csv").write_text("<<<<<<< ours\nETF,Underlying\n=======\n>>>>>>> theirs\n")
    pdf, csv = make_b4_pair_pnl_hedge_chart(
        "2026-06-01",
        runs_root=tmp_path,
        out_dir=tmp_path / "out",
    )
    assert pdf is None and csv is None


def test_chart_builder_fails_soft_on_missing_proposed_trade_columns(tmp_path):
    run = tmp_path / "2026-06-01"
    run.mkdir(parents=True)
    pd.DataFrame([{"ETF": "QBTZ", "sleeve": "inverse_decay_bucket4"}]).to_csv(
        run / "proposed_trades.csv",
        index=False,
    )
    pdf, csv = make_b4_pair_pnl_hedge_chart(
        "2026-06-01",
        runs_root=tmp_path,
        out_dir=tmp_path / "out",
    )
    assert pdf is None and csv is None


def test_chart_builder_writes_pdf_and_summary_for_active_proposed_pairs(tmp_path):
    _write_proposed(
        tmp_path,
        "2026-06-02",
        [
            {"ETF": "QBTZ", "Underlying": "QBTS", "sleeve": "inverse_decay_bucket4", "gross_target_usd": 10_000.0, "Delta": -2.0},
            {"ETF": "ZERO", "Underlying": "ZZZ", "sleeve": "inverse_decay_bucket4", "gross_target_usd": 0.0, "Delta": -1.0},
        ],
    )
    _write_run(tmp_path, "2026-06-01", pair_pnl=1000.0, etf_pnl=700.0, delta=-2.0)
    _write_run(tmp_path, "2026-06-02", pair_pnl=1500.0, etf_pnl=900.0, delta=-2.0)
    pdf, csv = make_b4_pair_pnl_hedge_chart(
        "2026-06-02",
        runs_root=tmp_path,
        out_dir=tmp_path / "out",
        metrics_csv=tmp_path / "missing_metrics.csv",
    )
    assert pdf is not None and pdf.suffix == ".pdf" and pdf.exists() and pdf.stat().st_size > 0
    assert csv is not None and csv.exists()
    summary = pd.read_csv(csv)
    assert list(summary["pair"]) == ["QBTZ|QBTS"]
    assert "model_pair_pnl_cum" in summary.columns
    assert "model_status" in summary.columns
    assert "borrow_cost_cum" in summary.columns
    assert "cagr" in summary.columns
    assert "metrics_source" in summary.columns
    assert "vol_shape_source" in summary.columns


def test_pick_etf_px_prefers_nav_when_close_price_is_vix_index_spike():
    row = pd.Series(
        {
            "close_price": 70.0,
            "nav": 3.5,
            "etf_adj_close": 3.24,
        }
    )
    px = _pick_etf_px_from_row(row, prev_a_px=3.5, b_px=22.5, prev_b_px=22.5)
    assert px == pytest.approx(3.5)


def test_pick_etf_px_holds_last_good_on_unanimous_corrupt_jump():
    row = pd.Series(
        {
            "close_price": 62.48,
            "nav": 62.48,
            "etf_adj_close": 62.48,
        }
    )
    px = _pick_etf_px_from_row(row, prev_a_px=3.09, b_px=23.61, prev_b_px=23.30)
    assert px == pytest.approx(3.09)


def test_build_prices_flexible_sanitizes_uvix_style_metrics_corruption():
    metrics = pd.DataFrame(
        [
            {
                "date": "2026-06-29",
                "ticker": "UVIX",
                "close_price": 70.0,
                "nav": 3.5,
                "etf_adj_close": 3.24,
                "underlying_adj_close": 22.5,
            },
            {
                "date": "2026-06-30",
                "ticker": "UVIX",
                "close_price": 64.8,
                "nav": 3.5,
                "etf_adj_close": 3.09,
                "underlying_adj_close": 23.30,
            },
            {
                "date": "2026-07-01",
                "ticker": "UVIX",
                "close_price": 62.48,
                "nav": 62.48,
                "etf_adj_close": 62.48,
                "underlying_adj_close": 23.61,
            },
        ]
        + [
            {
                "date": f"2026-01-{d:02d}",
                "ticker": "UVIX",
                "close_price": 3.0 + 0.01 * i,
                "nav": 3.0 + 0.01 * i,
                "etf_adj_close": 3.0 + 0.01 * i,
                "underlying_adj_close": 20.0 + 0.01 * i,
            }
            for i, d in enumerate(range(2, 22), start=0)
        ]
    )
    prices, reason, nrows = _build_prices_flexible(
        metrics,
        "UVIX",
        pd.Timestamp("2026-01-01"),
        underlying="SVIX",
    )
    assert reason == "ok"
    assert nrows >= 20
    last = prices.iloc[-1]
    assert last["a_px"] == pytest.approx(3.09)
    assert last["b_px"] == pytest.approx(23.61)
