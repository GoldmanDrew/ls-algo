from __future__ import annotations

import pandas as pd
import pytest

from ibkr_accounting import (
    _is_etf_leg,
    _normalize_bucket_triple,
    apply_plan_b4_spot_pnl_override,
    apply_yieldboost_spot_b2_override,
    build_bucket4_pair_registry,
    build_underlying_realized_bucket_ratio_map,
    compute_bucket4_pair_exposure,
    compute_plan_b4_structural_qty,
    held_exposure_bucket124_weights,
    load_plan_sleeve_bucket_usd,
    merge_plan_etf_metadata,
    resolve_b4_plan_exposure_underlyings,
)


def test_is_etf_leg_spot_underlying_not_etf() -> None:
    """Spot rows (symbol == underlying) must update FIFO ledger, not ETF pos qty.

    Regression: when complete_etf_maps adds MSTR->MSTR self-map, the old
    ``symbol in etf_to_under`` check skipped all MSTR spot trades and froze
    the share ledger while IBKR position kept growing (orphan shares).
    """
    etf_map = {"MSTR": "MSTR", "MTYY": "MSTR", "MSTU": "MSTR"}
    assert _is_etf_leg("MSTR", "MSTR", etf_map) is False
    assert _is_etf_leg("MTYY", "MSTR", etf_map) is True
    assert _is_etf_leg("SPY", "SPY", {"SPY": "SPY"}) is False


def test_normalize_bucket_triple() -> None:
    assert _normalize_bucket_triple(2, 2, 4) == (0.25, 0.25, 0.5)


def test_build_bucket4_pair_registry_excludes_flow_shorts(tmp_path) -> None:
    screened = tmp_path / "screened.csv"
    screened.write_text(
        "ETF,Underlying,Delta\n"
        "APLZ,APLD,-2.0\n"
        "SH,S&P500,-1.0\n"
        "TQQQ,QQQ,3.0\n",
        encoding="utf-8",
    )
    reg = build_bucket4_pair_registry(screened, flow_short_syms={"SH"})
    etfs = set(reg["etf"].tolist())
    assert "APLZ" in etfs
    assert "SH" not in etfs
    assert "TQQQ" not in etfs


def test_held_exposure_bucket124_weights_b4_only() -> None:
    pos = pd.DataFrame(
        [
            {
                "symbol": "APLZ",
                "position": -100,
                "markPrice": 10.0,
                "fxRateToBase": 1.0,
                "positionValue_base": -1000.0,
            },
            {
                "symbol": "APLD",
                "position": -50,
                "markPrice": 20.0,
                "fxRateToBase": 1.0,
                "positionValue_base": -1000.0,
            },
        ]
    )
    etf_to_under = {"APLZ": "APLD"}
    etf_to_delta = {"APLZ": -2.0}
    w1, w2, w4 = held_exposure_bucket124_weights(
        "APLD",
        pos,
        etf_to_under,
        etf_to_delta,
        b4_etf_syms={"APLZ"},
    )
    assert w4 == 1.0
    assert w1 == 0.0
    assert w2 == 0.0


def test_compute_bucket4_pair_exposure_net_near_hedged() -> None:
    registry = pd.DataFrame([{"etf": "APLZ", "underlying": "APLD", "delta": -2.0, "partial_hedge_ratio": 0.75}])
    pos = pd.DataFrame(
        [
            {"symbol": "APLZ", "position": -100, "markPrice": 10.0, "fxRateToBase": 1.0},
            {"symbol": "APLD", "position": -200, "markPrice": 10.0, "fxRateToBase": 1.0},
        ]
    )
    by_under, detail = compute_bucket4_pair_exposure(pos, registry)
    assert len(by_under) == 1
    net = float(by_under["net_notional_usd"].iloc[0])
    gross = float(by_under["gross_notional_usd"].iloc[0])
    assert gross > 0
    assert abs(net) < gross * 0.25
    assert len(detail) == 2


def test_pair_exposure_uses_exact_b4_share_count() -> None:
    registry = pd.DataFrame([{"etf": "APLZ", "underlying": "APLD", "delta": -2.0, "partial_hedge_ratio": 1.0}])
    pos = pd.DataFrame(
        [
            {"symbol": "APLZ", "position": -100, "markPrice": 10.0, "fxRateToBase": 1.0},
            {"symbol": "APLD", "position": -400, "markPrice": 10.0, "fxRateToBase": 1.0},
        ]
    )
    full, _ = compute_bucket4_pair_exposure(pos, registry)
    exact, _ = compute_bucket4_pair_exposure(pos, registry, underlying_b4_qty={"APLD": -100.0})
    assert float(exact["net_notional_usd"].iloc[0]) != float(full["net_notional_usd"].iloc[0])
    assert abs(float(exact["net_notional_usd"].iloc[0])) < abs(float(full["net_notional_usd"].iloc[0]))


def test_realized_ratio_map_includes_bucket_4() -> None:
    trades = pd.DataFrame(
        [
            {
                "dateTime": "20260115103000",
                "symbol": "APLD",
                "underlyingSymbol": "APLD",
                "quantity": -10.0,
                "fifoPnlRealized_base": 100.0,
                "tradePrice_base": 10.0,
                "orderReference": "ETF_LS|APLD__GROUP|APLZ|UNDER",
            },
        ]
    )
    etf_to_under = {"APLZ": "APLD"}
    etf_to_delta = {"APLZ": -2.0}
    m = build_underlying_realized_bucket_ratio_map(trades, etf_to_under, etf_to_delta)
    assert m["APLD"]["b4"] == 1.0


def test_load_plan_sleeve_bucket_usd_buckets_by_delta(tmp_path) -> None:
    """Plan rows are aggregated into b1/b2/b4 based on ETF delta sign/size."""
    plan = tmp_path / "proposed_trades.csv"
    plan.write_text(
        "ETF,Underlying,sleeve,long_usd\n"
        "MSTU,MSTR,core_leveraged,500\n"
        "MSTX,MSTR,core_leveraged,900\n"
        "MTYY,MSTR,yieldboost,15000\n"
        "MSTZ,MSTR,inverse_decay_bucket4,-1700\n"
        "MSDD,MSTR,inverse_decay_bucket4,-2600\n",
        encoding="utf-8",
    )
    etf_to_delta = {
        "MSTU": 2.0,
        "MSTX": 2.0,
        "MTYY": 0.37,
        "MSTZ": -2.0,
        "MSDD": -2.0,
    }
    out = load_plan_sleeve_bucket_usd(plan, etf_to_delta)
    assert "MSTR" in out
    assert out["MSTR"]["b1"] == 1400.0
    assert out["MSTR"]["b2"] == 15000.0
    assert out["MSTR"]["b4"] == -4300.0


def test_load_plan_sleeve_bucket_usd_missing_file_returns_empty(tmp_path) -> None:
    out = load_plan_sleeve_bucket_usd(tmp_path / "missing.csv", {"APLZ": -2.0})
    assert out == {}


def test_load_plan_sleeve_bucket_usd_sleeve_first_without_screened_delta(tmp_path) -> None:
    """Yieldboost rows must count toward b2 even when absent from etf_to_delta map."""
    plan = tmp_path / "proposed_trades.csv"
    plan.write_text(
        "ETF,Underlying,sleeve,long_usd,Delta,is_yieldboost\n"
        "MSTU,MSTR,core_leveraged,500,2.0,False\n"
        "MTYY,MSTR,yieldboost,15000,0.37,True\n"
        "MSTZ,MSTR,inverse_decay_bucket4,-1700,-2.0,False\n",
        encoding="utf-8",
    )
    out = load_plan_sleeve_bucket_usd(plan, {}, sleeve_first=True)
    assert out["MSTR"]["b1"] == 500.0
    assert out["MSTR"]["b2"] == 15000.0
    assert out["MSTR"]["b4"] == -1700.0


def test_merge_plan_etf_metadata_fills_missing_delta(tmp_path) -> None:
    plan = tmp_path / "proposed_trades.csv"
    plan.write_text(
        "ETF,Underlying,Delta\n"
        "MTYY,MSTR,0.37\n",
        encoding="utf-8",
    )
    under, delta = merge_plan_etf_metadata(plan, {}, {})
    assert under["MTYY"] == "MSTR"
    assert delta["MTYY"] == pytest.approx(0.37)


def test_inject_slice_preserves_b2_lot_share() -> None:
    """B4 plan slice should not zero out FIFO bucket-2 spot PnL."""
    df = pd.DataFrame(
        [
            {
                "symbol": "MSTR",
                "underlying": "MSTR",
                "realized_pnl": 100.0,
                "unrealized_pnl": 1000.0,
            }
        ]
    )
    lot: dict = {
        "MSTR": {
            "bucket_1": {"realized_pnl": 100.0, "unrealized_pnl": 490.0},
            "bucket_2": {"realized_pnl": 0.0, "unrealized_pnl": 510.0},
            "bucket_4": {"realized_pnl": 0.0, "unrealized_pnl": 0.0},
        }
    }
    plan_ratio = {"b1": 0.08, "b2": 0.71, "b4": 0.21}
    apply_plan_b4_spot_pnl_override(
        lot_components=lot,
        underlying="MSTR",
        df=df,
        plan_ratio=plan_ratio,
        spot_carry_cols=set(),
        etf_to_delta_map={"MTYY": 0.37},
        mode="inject_slice",
        ledger_r_b1=0.49,
        ledger_r_b2=0.51,
    )
    assert lot["MSTR"]["bucket_4"]["unrealized_pnl"] == pytest.approx(210.0)
    assert lot["MSTR"]["bucket_2"]["unrealized_pnl"] == pytest.approx(402.9)
    assert lot["MSTR"]["bucket_1"]["unrealized_pnl"] == pytest.approx(387.1)
    total = sum(lot["MSTR"][b]["unrealized_pnl"] for b in lot["MSTR"])
    assert total == pytest.approx(1000.0)


def test_yieldboost_spot_b2_override_uses_held_ratios() -> None:
    """Yieldboost spot PnL should roll into bucket 2 using held-exposure split."""
    df = pd.DataFrame(
        [
            {
                "symbol": "SMCI",
                "underlying": "SMCI",
                "realized_pnl": -100.0,
                "unrealized_pnl": 1000.0,
                "borrow_fees": -10.0,
            }
        ]
    )
    lot: dict = {
        "SMCI": {
            "bucket_1": {"realized_pnl": -100.0, "unrealized_pnl": 1000.0, "borrow_fees": -10.0},
            "bucket_2": {"realized_pnl": 0.0, "unrealized_pnl": 0.0, "borrow_fees": 0.0},
            "bucket_4": {"realized_pnl": 0.0, "unrealized_pnl": 0.0, "borrow_fees": 0.0},
        }
    }
    apply_yieldboost_spot_b2_override(
        lot_components=lot,
        underlying="SMCI",
        df=df,
        spot_carry_cols={"borrow_fees"},
        r_b1=0.04,
        r_b2=0.96,
    )
    assert lot["SMCI"]["bucket_2"]["unrealized_pnl"] == pytest.approx(960.0)
    assert lot["SMCI"]["bucket_1"]["unrealized_pnl"] == pytest.approx(40.0)
    assert lot["SMCI"]["bucket_2"]["borrow_fees"] == pytest.approx(-9.6)
    assert lot["SMCI"]["bucket_1"]["borrow_fees"] == pytest.approx(-0.4)


def test_inject_slice_pure_b4_goes_to_bucket_4() -> None:
    df = pd.DataFrame(
        [{"symbol": "APLD", "underlying": "APLD", "realized_pnl": 0.0, "unrealized_pnl": 500.0}]
    )
    lot: dict = {
        "APLD": {
            "bucket_1": {"realized_pnl": 0.0, "unrealized_pnl": 500.0},
            "bucket_2": {"realized_pnl": 0.0, "unrealized_pnl": 0.0},
            "bucket_4": {"realized_pnl": 0.0, "unrealized_pnl": 0.0},
        }
    }
    apply_plan_b4_spot_pnl_override(
        lot_components=lot,
        underlying="APLD",
        df=df,
        plan_ratio={"b1": 0.0, "b2": 0.0, "b4": 1.0},
        spot_carry_cols=set(),
        etf_to_delta_map={"APLZ": -2.0},
        mode="inject_slice",
        ledger_r_b1=0.0,
        ledger_r_b2=0.0,
    )
    assert lot["APLD"]["bucket_4"]["unrealized_pnl"] == pytest.approx(500.0)
    assert lot["APLD"]["bucket_1"]["unrealized_pnl"] == pytest.approx(0.0)
    assert lot["APLD"]["bucket_2"]["unrealized_pnl"] == pytest.approx(0.0)


def test_compute_plan_b4_structural_qty_signed_short() -> None:
    plan = {"MSTR": {"b1": 1000.0, "b2": 2000.0, "b4": -3000.0}}
    qty = compute_plan_b4_structural_qty(plan, "MSTR", 100.0)
    assert qty == pytest.approx(-30.0)


def test_plan_structural_underlying_short_on_net_long_spot() -> None:
    """Plan-implied B4 qty is negative even when IBKR net spot is long."""
    registry = pd.DataFrame(
        [{"etf": "MSTZ", "underlying": "MSTR", "delta": -2.0, "partial_hedge_ratio": 1.0}]
    )
    pos = pd.DataFrame(
        [
            {"symbol": "MSTR", "position": 332, "markPrice": 400.0, "fxRateToBase": 1.0},
            {"symbol": "MSTZ", "position": -100, "markPrice": 30.0, "fxRateToBase": 1.0},
        ]
    )
    plan_qty = -5414.0 / 400.0
    _, detail = compute_bucket4_pair_exposure(
        pos, registry, underlying_b4_qty={"MSTR": plan_qty}
    )
    under = detail[(detail["underlying"] == "MSTR") & (detail["leg_type"] == "underlying")]
    assert len(under) == 1
    assert float(under["net_notional_usd"].iloc[0]) == pytest.approx(plan_qty * 400.0)
    assert float(under["net_notional_usd"].iloc[0]) < 0


def test_resolve_b4_plan_exposure_underlyings_auto() -> None:
    plan = {
        "MSTR": {"b1": 1.0, "b2": 2.0, "b4": -5.0},
        "GDX": {"b1": 100.0, "b2": 0.0, "b4": 0.0},
    }
    out = resolve_b4_plan_exposure_underlyings(
        mode="plan_structural",
        explicit=set(),
        b4_underlyings={"MSTR", "GDX", "NBIS"},
        plan_sleeve_usd=plan,
        min_usd=1.0,
    )
    assert out == {"MSTR"}


def test_pair_exposure_emits_underlying_once_per_underlying() -> None:
    """Multiple inverse ETFs for one underlying must not duplicate the under leg."""
    registry = pd.DataFrame(
        [
            {"etf": "MSDD", "underlying": "MSTR", "delta": -2.0, "partial_hedge_ratio": 1.0},
            {"etf": "MSTZ", "underlying": "MSTR", "delta": -2.0, "partial_hedge_ratio": 1.0},
            {"etf": "SMST", "underlying": "MSTR", "delta": -2.0, "partial_hedge_ratio": 1.0},
        ]
    )
    pos = pd.DataFrame(
        [
            {"symbol": "MSTR", "position": 100, "markPrice": 100.0, "fxRateToBase": 1.0},
            {"symbol": "MSDD", "position": -50, "markPrice": 20.0, "fxRateToBase": 1.0},
            {"symbol": "MSTZ", "position": -25, "markPrice": 10.0, "fxRateToBase": 1.0},
        ]
    )
    _, detail = compute_bucket4_pair_exposure(
        pos, registry, underlying_b4_qty={"MSTR": -10.0}
    )
    under_rows = detail[
        (detail["underlying"] == "MSTR") & (detail["leg_type"] == "underlying")
    ]
    assert len(under_rows) == 1
    assert float(under_rows["net_notional_usd"].iloc[0]) == -1000.0


def test_ledger_full_replay_trade_filter() -> None:
    """Full-replay underlyings keep pre-cutoff trades; others stay incremental."""
    from ibkr_accounting import canonical_symbol

    trades = pd.DataFrame(
        {
            "dateTime": ["20260515000000", "20260519000000", "20260521000000"],
            "symbol": ["MSTR", "RCAT", "RCAT"],
            "underlyingSymbol": ["MSTR", "RCAT", "RCAT"],
            "quantity": [100.0, 50.0, 10.0],
        }
    )
    trades["date"] = trades["dateTime"].astype(str).str.slice(0, 8)
    trades["_ledger_u"] = trades.apply(
        lambda r: canonical_symbol(str(r.get("underlyingSymbol") or r.get("symbol") or "")),
        axis=1,
    )
    prev_cutoff_ymd = "20260520"
    full_replay = {"MSTR"}
    kept = trades[
        (trades["date"] > prev_cutoff_ymd) | trades["_ledger_u"].isin(full_replay)
    ]
    assert set(kept["symbol"].tolist()) == {"MSTR", "RCAT"}
    assert len(kept) == 2
