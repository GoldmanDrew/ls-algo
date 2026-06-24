"""Tests for the EOD per-pair B4 PnL + hedge chart helpers."""
from __future__ import annotations

import pandas as pd
import pytest

from scripts.bucket4_eod_pair_charts import (
    load_active_b4_pairs_from_proposed,
    load_actual_trade_markers,
    load_b4_pair_leg_history,
    load_pair_gross_and_realized_h,
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
