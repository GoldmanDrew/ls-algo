"""Tests for the hedged vs unhedged PnL lens (hedged_pnl.py)."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from hedged_pnl import (
    B4_PAIR_CSV_NAME,
    SPLIT_JSON_NAME,
    compute_b4_leg_fractions,
    compute_hedged_split,
)


def _write_run(
    runs_root: Path,
    run_date: str,
    *,
    bucket_pnl: dict[str, float],
    b4_symbol_rows: list[dict] | None = None,
    detail_rows: list[dict] | None = None,
    pair_rows: list[dict] | None = None,
    plan_h_rows: list[dict] | None = None,
) -> None:
    acct = runs_root / run_date / "accounting"
    acct.mkdir(parents=True, exist_ok=True)
    totals = {
        "run_date": run_date,
        "total_pnl": sum(bucket_pnl.values()),
        "bucket_pnl": bucket_pnl,
    }
    (acct / "totals.json").write_text(json.dumps(totals), encoding="utf-8")
    sym_rows = list(b4_symbol_rows or [])
    pd.DataFrame(
        sym_rows, columns=["symbol", "underlying", "bucket", "total_pnl"]
    ).to_csv(acct / "pnl_by_symbol.csv", index=False)
    if detail_rows is not None:
        pd.DataFrame(
            detail_rows,
            columns=["underlying", "symbol", "leg_type", "net_notional_usd", "gross_notional_usd"],
        ).to_csv(acct / "net_exposure_bucket_4_detail.csv", index=False)
    if pair_rows is not None:
        pd.DataFrame(pair_rows, columns=["etf", "underlying", "delta"]).to_csv(
            acct / "bucket4_pairs.csv", index=False
        )
    if plan_h_rows is not None:
        cadence = runs_root / run_date / "b4_hedge_cadence"
        cadence.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(plan_h_rows, columns=["ETF", "Underlying", "hedge_ratio"]).to_csv(
            cadence / "b4_ratchet_targets.csv", index=False
        )


def _b4_sym(symbol: str, underlying: str, pnl: float) -> dict:
    return {"symbol": symbol, "underlying": underlying, "bucket": "bucket_4", "total_pnl": pnl}


def _detail(underlying: str, symbol: str, leg_type: str, gross: float) -> dict:
    return {
        "underlying": underlying,
        "symbol": symbol,
        "leg_type": leg_type,
        "net_notional_usd": -gross,
        "gross_notional_usd": gross,
    }


class TestB4LegFractions:
    def test_under_hedged_pair(self, tmp_path):
        """und = 1000, beta*etf = 2000 -> ETF leg half hedged, und leg fully matched."""
        runs = tmp_path / "runs"
        _write_run(
            runs,
            "2026-01-02",
            bucket_pnl={"bucket_4": 0.0},
            detail_rows=[
                _detail("ABC", "ABC", "underlying", 1000.0),
                _detail("ABC", "XYZ", "etf", 1000.0),
            ],
            pair_rows=[{"etf": "XYZ", "underlying": "ABC", "delta": -2.0}],
        )
        fr = compute_b4_leg_fractions("2026-01-02", runs)
        etf = fr[(fr.leg_type == "etf") & (fr.symbol == "XYZ")].iloc[0]
        und = fr[(fr.leg_type == "underlying") & (fr.underlying == "ABC")].iloc[0]
        assert etf["f_hedged"] == pytest.approx(0.5)
        assert etf["matched_usd"] == pytest.approx(1000.0)
        assert und["f_hedged"] == pytest.approx(1.0)

    def test_over_hedged_pair(self, tmp_path):
        """und = 5000 > beta*etf = 2000 -> ETF fully hedged, excess und unhedged."""
        runs = tmp_path / "runs"
        _write_run(
            runs,
            "2026-01-02",
            bucket_pnl={"bucket_4": 0.0},
            detail_rows=[
                _detail("ABC", "ABC", "underlying", 5000.0),
                _detail("ABC", "XYZ", "etf", 1000.0),
            ],
            pair_rows=[{"etf": "XYZ", "underlying": "ABC", "delta": -2.0}],
        )
        fr = compute_b4_leg_fractions("2026-01-02", runs)
        etf = fr[(fr.leg_type == "etf")].iloc[0]
        und = fr[(fr.leg_type == "underlying")].iloc[0]
        assert etf["f_hedged"] == pytest.approx(1.0)
        assert und["f_hedged"] == pytest.approx(2000.0 / 5000.0)

    def test_multi_etf_underlying_pro_rata(self, tmp_path):
        """Underlying gross allocated by ETF gross share (same as realized-h convention)."""
        runs = tmp_path / "runs"
        _write_run(
            runs,
            "2026-01-02",
            bucket_pnl={"bucket_4": 0.0},
            detail_rows=[
                _detail("U", "U", "underlying", 3000.0),
                _detail("U", "E1", "etf", 1000.0),
                _detail("U", "E2", "etf", 2000.0),
            ],
            pair_rows=[
                {"etf": "E1", "underlying": "U", "delta": -2.0},
                {"etf": "E2", "underlying": "U", "delta": -1.0},
            ],
        )
        fr = compute_b4_leg_fractions("2026-01-02", runs)
        e1 = fr[fr.symbol == "E1"].iloc[0]
        e2 = fr[fr.symbol == "E2"].iloc[0]
        und = fr[fr.leg_type == "underlying"].iloc[0]
        # E1: alloc 1000 vs beta_equiv 2000 -> 0.5; E2: alloc 2000 vs beta_equiv 2000 -> 1.0
        assert e1["f_hedged"] == pytest.approx(0.5)
        assert e2["f_hedged"] == pytest.approx(1.0)
        assert und["f_hedged"] == pytest.approx(1.0)

    def test_no_underlying_leg_is_unhedged(self, tmp_path):
        runs = tmp_path / "runs"
        _write_run(
            runs,
            "2026-01-02",
            bucket_pnl={"bucket_4": 0.0},
            detail_rows=[_detail("ABC", "XYZ", "etf", 1000.0)],
            pair_rows=[{"etf": "XYZ", "underlying": "ABC", "delta": -2.0}],
        )
        fr = compute_b4_leg_fractions("2026-01-02", runs)
        assert fr[fr.symbol == "XYZ"].iloc[0]["f_hedged"] == pytest.approx(0.0)


class TestComputeHedgedSplit:
    def test_seed_run_split_and_invariant(self, tmp_path):
        runs = tmp_path / "runs"
        ledger = tmp_path / "ledger" / "hedged_pnl_history.csv"
        _write_run(
            runs,
            "2026-01-02",
            bucket_pnl={
                "bucket_1": 10.0,
                "bucket_2": 20.0,
                "bucket_3": 5.0,
                "bucket_4": 150.0,
                "bucket_5": 3.0,
            },
            b4_symbol_rows=[_b4_sym("XYZ", "ABC", 100.0), _b4_sym("ABC", "ABC", 50.0)],
            detail_rows=[
                _detail("ABC", "ABC", "underlying", 1000.0),
                _detail("ABC", "XYZ", "etf", 1000.0),
            ],
            pair_rows=[{"etf": "XYZ", "underlying": "ABC", "delta": -2.0}],
        )
        res = compute_hedged_split(
            "2026-01-02", runs_root=runs, ledger_path=ledger, prev_run_date=None
        )
        # Hedged = B1 + B2 + 0.5*etf(100) + 1.0*und(50) = 130; unhedged = 5+3+50 = 58.
        assert res["seeded"] is True
        assert res["hedged_pnl_ytd"] == pytest.approx(130.0)
        assert res["unhedged_pnl_ytd"] == pytest.approx(58.0)
        assert res["hedged_pnl_ytd"] + res["unhedged_pnl_ytd"] == pytest.approx(
            res["total_pnl_ytd"]
        )
        assert res["reconciliation"]["ok"] is True
        acct = runs / "2026-01-02" / "accounting"
        assert (acct / SPLIT_JSON_NAME).is_file()
        assert (acct / B4_PAIR_CSV_NAME).is_file()

    def test_daily_delta_uses_todays_fractions_and_ties_out(self, tmp_path):
        runs = tmp_path / "runs"
        ledger = tmp_path / "ledger" / "hedged_pnl_history.csv"
        common = dict(
            detail_rows=[
                _detail("ABC", "ABC", "underlying", 1000.0),
                _detail("ABC", "XYZ", "etf", 1000.0),
            ],
            pair_rows=[{"etf": "XYZ", "underlying": "ABC", "delta": -2.0}],
        )
        _write_run(
            runs,
            "2026-01-02",
            bucket_pnl={"bucket_1": 10.0, "bucket_3": 5.0, "bucket_4": 150.0},
            b4_symbol_rows=[_b4_sym("XYZ", "ABC", 100.0), _b4_sym("ABC", "ABC", 50.0)],
            **common,
        )
        # Day 2: ETF leg +40, und leg -10; B1 +5, B3 +1.
        _write_run(
            runs,
            "2026-01-03",
            bucket_pnl={"bucket_1": 15.0, "bucket_3": 6.0, "bucket_4": 180.0},
            b4_symbol_rows=[_b4_sym("XYZ", "ABC", 140.0), _b4_sym("ABC", "ABC", 40.0)],
            **common,
        )
        compute_hedged_split("2026-01-02", runs_root=runs, ledger_path=ledger, prev_run_date=None)
        res = compute_hedged_split(
            "2026-01-03", runs_root=runs, ledger_path=ledger, prev_run_date="2026-01-02"
        )
        # hedged_daily = 5 (B1) + 0.5*40 (etf) + 1.0*(-10) (und) = 15
        assert res["hedged_daily"] == pytest.approx(15.0)
        # unhedged_daily = 1 (B3) + 0.5*40 = 21
        assert res["unhedged_daily"] == pytest.approx(21.0)
        assert res["total_daily"] == pytest.approx(36.0)
        led = pd.read_csv(ledger)
        assert len(led) == 2
        last = led.iloc[-1]
        assert last["hedged_pnl"] + last["unhedged_pnl"] == pytest.approx(201.0)  # 15+6+180

    def test_carry_forward_when_exposure_missing(self, tmp_path):
        """Pair closed today (no detail row) -> prior run's fraction is carried."""
        runs = tmp_path / "runs"
        ledger = tmp_path / "ledger" / "hedged_pnl_history.csv"
        _write_run(
            runs,
            "2026-01-02",
            bucket_pnl={"bucket_4": 100.0},
            b4_symbol_rows=[_b4_sym("XYZ", "ABC", 100.0)],
            detail_rows=[
                _detail("ABC", "ABC", "underlying", 1000.0),
                _detail("ABC", "XYZ", "etf", 1000.0),
            ],
            pair_rows=[{"etf": "XYZ", "underlying": "ABC", "delta": -2.0}],
        )
        _write_run(
            runs,
            "2026-01-03",
            bucket_pnl={"bucket_4": 160.0},
            b4_symbol_rows=[_b4_sym("XYZ", "ABC", 160.0)],
            detail_rows=[],  # empty detail: no exposure today
            pair_rows=[{"etf": "XYZ", "underlying": "ABC", "delta": -2.0}],
        )
        compute_hedged_split("2026-01-02", runs_root=runs, ledger_path=ledger, prev_run_date=None)
        res = compute_hedged_split(
            "2026-01-03", runs_root=runs, ledger_path=ledger, prev_run_date="2026-01-02"
        )
        # ETF delta +60 at carried f=0.5 -> hedged 30 / unhedged 30
        assert res["components"]["b4_hedged_daily"] == pytest.approx(30.0)
        assert res["components"]["b4_unhedged_daily"] == pytest.approx(30.0)
        audit = pd.read_csv(runs / "2026-01-03" / "accounting" / B4_PAIR_CSV_NAME)
        assert audit.iloc[0]["f_source"] == "carry"

    def test_plan_h_fallback_for_new_pair(self, tmp_path):
        """No exposure and no prior fraction -> plan hedge ratio drives the split."""
        runs = tmp_path / "runs"
        ledger = tmp_path / "ledger" / "hedged_pnl_history.csv"
        _write_run(
            runs,
            "2026-01-02",
            bucket_pnl={"bucket_4": 100.0},
            b4_symbol_rows=[_b4_sym("XYZ", "ABC", 100.0)],
            detail_rows=[],
            pair_rows=[{"etf": "XYZ", "underlying": "ABC", "delta": -2.0}],
            plan_h_rows=[{"ETF": "XYZ", "Underlying": "ABC", "hedge_ratio": 0.6}],
        )
        res = compute_hedged_split(
            "2026-01-02", runs_root=runs, ledger_path=ledger, prev_run_date=None
        )
        assert res["components"]["b4_hedged_daily"] == pytest.approx(60.0)
        assert res["components"]["b4_unhedged_daily"] == pytest.approx(40.0)
        audit = pd.read_csv(runs / "2026-01-02" / "accounting" / B4_PAIR_CSV_NAME)
        assert audit.iloc[0]["f_source"] == "plan"

    def test_unhedged_default_without_any_source(self, tmp_path):
        runs = tmp_path / "runs"
        ledger = tmp_path / "ledger" / "hedged_pnl_history.csv"
        _write_run(
            runs,
            "2026-01-02",
            bucket_pnl={"bucket_4": 100.0},
            b4_symbol_rows=[_b4_sym("XYZ", "ABC", 100.0)],
            detail_rows=[],
            pair_rows=[{"etf": "XYZ", "underlying": "ABC", "delta": -2.0}],
        )
        res = compute_hedged_split(
            "2026-01-02", runs_root=runs, ledger_path=ledger, prev_run_date=None
        )
        assert res["components"]["b4_hedged_daily"] == pytest.approx(0.0)
        assert res["components"]["b4_unhedged_daily"] == pytest.approx(100.0)

    def test_symbol_vs_bucket_gap_goes_to_unhedged(self, tmp_path):
        """Bucket-level B4 diff not matched by symbol rows still ties out (residual -> unhedged)."""
        runs = tmp_path / "runs"
        ledger = tmp_path / "ledger" / "hedged_pnl_history.csv"
        _write_run(
            runs,
            "2026-01-02",
            bucket_pnl={"bucket_4": 100.0},
            b4_symbol_rows=[_b4_sym("XYZ", "ABC", 75.0)],  # 25 unexplained at bucket level
            detail_rows=[],
            pair_rows=[{"etf": "XYZ", "underlying": "ABC", "delta": -2.0}],
        )
        res = compute_hedged_split(
            "2026-01-02", runs_root=runs, ledger_path=ledger, prev_run_date=None
        )
        assert res["hedged_pnl_ytd"] + res["unhedged_pnl_ytd"] == pytest.approx(100.0)

    def test_idempotent_rerun_same_date(self, tmp_path):
        runs = tmp_path / "runs"
        ledger = tmp_path / "ledger" / "hedged_pnl_history.csv"
        _write_run(
            runs,
            "2026-01-02",
            bucket_pnl={"bucket_1": 10.0, "bucket_4": 0.0},
            b4_symbol_rows=[],
            detail_rows=[],
            pair_rows=[],
        )
        r1 = compute_hedged_split("2026-01-02", runs_root=runs, ledger_path=ledger, prev_run_date=None)
        r2 = compute_hedged_split("2026-01-02", runs_root=runs, ledger_path=ledger, prev_run_date=None)
        assert r1["hedged_pnl_ytd"] == pytest.approx(r2["hedged_pnl_ytd"])
        assert len(pd.read_csv(ledger)) == 1


class TestEmailHeadline:
    def test_headline_block_includes_hedged_lines(self):
        from run_eod_pnl_email import format_headline_pnl_block

        split = {
            "hedged_pnl_ytd": 110_000.0,
            "unhedged_pnl_ytd": -20_000.0,
            "hedged_daily": 1_500.0,
            "unhedged_daily": -266.0,
            "total_daily": 1_234.0,
        }
        block = format_headline_pnl_block(
            {"bucket_1": 1.0, "bucket_2": 2.0, "bucket_3": 3.0, "bucket_4": 4.0, "bucket_5": 5.0},
            total_pnl=90_000.0,
            hedged_split=split,
        )
        assert "Hedged PnL:" in block
        assert "Unhedged PnL:" in block
        assert "Total PnL:" in block
        assert "110,000.00" in block
        assert "B1 + B2 + B4 matched" in block

    def test_headline_block_without_split_unchanged(self):
        from run_eod_pnl_email import format_headline_pnl_block

        block = format_headline_pnl_block(
            {"bucket_1": 1.0, "bucket_2": 2.0, "bucket_3": 3.0, "bucket_4": 4.0, "bucket_5": 5.0},
            total_pnl=15.0,
        )
        assert "Hedged PnL:" not in block
        assert "Total: 15.00" in block


class TestDashboardPanel:
    def test_panel_from_split_artifact(self, tmp_path):
        from risk_dashboard.metrics import compute_hedged_pnl_panel

        acct = tmp_path / "accounting"
        acct.mkdir()
        (acct / "hedged_pnl_split.json").write_text(
            json.dumps(
                {
                    "run_date": "2026-01-02",
                    "hedged_pnl_ytd": 100.0,
                    "unhedged_pnl_ytd": -40.0,
                    "hedged_daily": 10.0,
                    "unhedged_daily": -4.0,
                    "total_pnl_ytd": 60.0,
                    "total_daily": 6.0,
                    "components": {},
                    "reconciliation": {"residual_daily": 0.0, "ok": True},
                }
            ),
            encoding="utf-8",
        )
        panel = compute_hedged_pnl_panel(
            acct,
            hedged_history_csv=tmp_path / "missing.csv",
            nav_usd=1000.0,
            run_date="2026-01-02",
        )
        assert panel["available"] is True
        assert panel["hedged_ytd_usd"] == pytest.approx(100.0)
        assert panel["hedged_ytd_pct_nav"] == pytest.approx(0.1)
        assert panel["reconciliation"]["ties_out"] is True
        json.dumps(panel, allow_nan=False)

    def test_panel_unavailable_when_no_sources(self, tmp_path):
        from risk_dashboard.metrics import compute_hedged_pnl_panel

        panel = compute_hedged_pnl_panel(
            tmp_path,
            hedged_history_csv=tmp_path / "missing.csv",
            nav_usd=1000.0,
        )
        assert panel["available"] is False
