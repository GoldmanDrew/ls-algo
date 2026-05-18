from __future__ import annotations

import pandas as pd

from ibkr_accounting import (
    _normalize_bucket_triple,
    build_bucket4_pair_registry,
    build_underlying_realized_bucket_ratio_map,
    compute_bucket4_pair_exposure,
    held_exposure_bucket124_weights,
    load_plan_sleeve_bucket_usd,
)


def test_normalize_bucket_triple() -> None:
    assert _normalize_bucket_triple(2, 2, 4) == (0.25, 0.25, 0.5)


def test_build_bucket4_pair_registry_excludes_flow_shorts(tmp_path) -> None:
    screened = tmp_path / "screened.csv"
    screened.write_text(
        "ETF,Underlying,Beta\n"
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
    etf_to_beta = {"APLZ": -2.0}
    w1, w2, w4 = held_exposure_bucket124_weights(
        "APLD",
        pos,
        etf_to_under,
        etf_to_beta,
        b4_etf_syms={"APLZ"},
    )
    assert w4 == 1.0
    assert w1 == 0.0
    assert w2 == 0.0


def test_compute_bucket4_pair_exposure_net_near_hedged() -> None:
    registry = pd.DataFrame([{"etf": "APLZ", "underlying": "APLD", "beta": -2.0, "partial_hedge_ratio": 0.75}])
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
    registry = pd.DataFrame([{"etf": "APLZ", "underlying": "APLD", "beta": -2.0, "partial_hedge_ratio": 1.0}])
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
    etf_to_beta = {"APLZ": -2.0}
    m = build_underlying_realized_bucket_ratio_map(trades, etf_to_under, etf_to_beta)
    assert m["APLD"]["b4"] == 1.0


def test_load_plan_sleeve_bucket_usd_buckets_by_beta(tmp_path) -> None:
    """Plan rows are aggregated into b1/b2/b4 based on ETF beta sign/size."""
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
    etf_to_beta = {
        "MSTU": 2.0,
        "MSTX": 2.0,
        "MTYY": 0.37,
        "MSTZ": -2.0,
        "MSDD": -2.0,
    }
    out = load_plan_sleeve_bucket_usd(plan, etf_to_beta)
    assert "MSTR" in out
    assert out["MSTR"]["b1"] == 1400.0
    assert out["MSTR"]["b2"] == 15000.0
    assert out["MSTR"]["b4"] == -4300.0


def test_load_plan_sleeve_bucket_usd_missing_file_returns_empty(tmp_path) -> None:
    out = load_plan_sleeve_bucket_usd(tmp_path / "missing.csv", {"APLZ": -2.0})
    assert out == {}


def test_pair_exposure_emits_underlying_once_per_underlying() -> None:
    """Multiple inverse ETFs for one underlying must not duplicate the under leg."""
    registry = pd.DataFrame(
        [
            {"etf": "MSDD", "underlying": "MSTR", "beta": -2.0, "partial_hedge_ratio": 1.0},
            {"etf": "MSTZ", "underlying": "MSTR", "beta": -2.0, "partial_hedge_ratio": 1.0},
            {"etf": "SMST", "underlying": "MSTR", "beta": -2.0, "partial_hedge_ratio": 1.0},
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
