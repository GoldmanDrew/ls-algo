from __future__ import annotations

import pandas as pd

from daily_screener import (
    apply_strategy_blacklist_to_universe,
    build_full_universe,
    load_strategy_blacklist,
)
from execute_trade_plan import (
    build_long_spot_bucket_token,
    classify_plan_leg_bucket,
)
from ibkr_accounting import (
    _bucket_hint_from_order_reference,
    _normalize_bucket_triple,
    bucket_weights_from_order_reference,
    normalize_plan_etf_ticker,
    resolve_ledger_full_replay_all,
    spot_trade_bucket_weights,
)
from run_eod_pnl_email import format_period_pnl_summary


# ── Phase 3: explicit B1/B2 long-spot split tagged at execution ──────────────


def test_long_spot_bucket_token_permille_round_trip() -> None:
    tok = build_long_spot_bucket_token(6000.0, 4000.0)
    assert tok == "LSB:600:400"
    assert bucket_weights_from_order_reference(f"ETF_LS|MSTR__ESTABLISH|MSTR|UNDER|{tok}") == (
        0.6,
        0.4,
        0.0,
    )


def test_long_spot_bucket_token_single_sleeve() -> None:
    assert build_long_spot_bucket_token(12000.0, 0.0) == "LSB:1000:0"
    assert build_long_spot_bucket_token(0.0, 0.0) == ""


def test_bucket_weights_bare_single_bucket_tokens() -> None:
    assert bucket_weights_from_order_reference("ETF_LS|FOO|B1") == (1.0, 0.0, 0.0)
    assert bucket_weights_from_order_reference("ETF_LS|FOO|B2") == (0.0, 1.0, 0.0)
    assert bucket_weights_from_order_reference("ETF_LS|FOO|B4") == (0.0, 0.0, 1.0)
    assert bucket_weights_from_order_reference("ETF_LS|FOO__ESTABLISH|FOO|UNDER") is None


def test_classify_plan_leg_bucket_sleeve_then_delta() -> None:
    assert classify_plan_leg_bucket(sleeve="core_leveraged") == "b1"
    assert classify_plan_leg_bucket(sleeve="yieldboost") == "b2"
    assert classify_plan_leg_bucket(sleeve="inverse_decay_bucket4") == "b4"
    assert classify_plan_leg_bucket(sleeve=None, delta=2.0) == "b1"
    assert classify_plan_leg_bucket(sleeve=None, delta=0.5) == "b2"
    assert classify_plan_leg_bucket(sleeve=None, delta=-2.0) == "b4"
    assert classify_plan_leg_bucket(sleeve=None, delta=None, long_usd=-100.0) == "b4"


def test_resolve_ledger_full_replay_all_triggers() -> None:
    # Off by default.
    assert resolve_ledger_full_replay_all(False, ["MSTR", "COIN"], "") is False
    # Config boolean.
    assert resolve_ledger_full_replay_all(True, [], "") is True
    # Sentinel inside the explicit replay list.
    assert resolve_ledger_full_replay_all(False, ["MSTR", "*"], "") is True
    assert resolve_ledger_full_replay_all(False, ["all"], "") is True
    # Env var.
    assert resolve_ledger_full_replay_all(False, [], "1") is True
    assert resolve_ledger_full_replay_all(False, [], "true") is True
    assert resolve_ledger_full_replay_all(False, [], "no") is False


def test_explicit_split_overrides_inferred_weights() -> None:
    # MSTR holds a B1 LETF (MSTU β=2) and a B2 yieldBOOST (MSTY β=0.4).
    etf_to_under = {"MSTU": "MSTR", "MSTY": "MSTR"}
    etf_to_delta = {"MSTU": 2.0, "MSTY": 0.4}
    etf_pos_qty = {"MSTU": -100.0, "MSTY": -100.0}
    ref = "ETF_LS|MSTR__ESTABLISH|MSTR|UNDER|LSB:700:300"
    w1, w2, w4 = spot_trade_bucket_weights(
        "MSTR",
        ref,
        etf_to_under,
        etf_to_delta,
        etf_pos_qty,
    )
    assert (round(w1, 3), round(w2, 3), round(w4, 3)) == (0.7, 0.3, 0.0)


def test_order_ref_standalone_etf_token_classifies_bucket_2() -> None:
    etf_to_delta = {"FBYY": 0.4, "FBL": 2.0}
    ref = "ETF_LS|META__GROUP|FBYY|ETF_DELTA|att1|ADAPTIVE_MKT"
    assert _bucket_hint_from_order_reference(ref, etf_to_delta) == "bucket_2"


def test_order_ref_standalone_etf_token_classifies_bucket_1() -> None:
    etf_to_delta = {"FBYY": 0.4, "FBL": 2.0}
    ref = "ETF_LS|META__GROUP|FBL|ETF_DELTA|att1|ADAPTIVE_MKT"
    assert _bucket_hint_from_order_reference(ref, etf_to_delta) == "bucket_1"


def test_order_ref_standalone_inverse_token_classifies_bucket_4() -> None:
    etf_to_delta = {"APLZ": -2.0}
    ref = "ETF_LS|APLD__GROUP|APLZ|ETF_DELTA|att1|ADAPTIVE_MKT"
    assert _bucket_hint_from_order_reference(ref, etf_to_delta) == "bucket_4"


def test_order_ref_under_with_inverse_etf_classifies_bucket_4() -> None:
    etf_to_delta = {"APLZ": -2.0}
    ref = "ETF_LS|APLD__ESTABLISH|APLZ|UNDER"
    assert _bucket_hint_from_order_reference(ref, etf_to_delta) == "bucket_4"


def test_normalize_bucket_triple_sums_to_one() -> None:
    assert sum(_normalize_bucket_triple(1, 2, 3)) == 1.0


def test_normalize_plan_etf_btfl_to_keex() -> None:
    assert normalize_plan_etf_ticker("BTFL") == "KEEX"
    assert normalize_plan_etf_ticker("KEEX") == "KEEX"


def test_screener_keex_maps_to_keel() -> None:
    universe = build_full_universe(skip_scrape=True, skip_inverse=True)
    row = universe.loc[universe["ETF"] == "KEEX"]
    assert not row.empty
    assert set(row["Underlying"]) == {"KEEL"}


def test_blacklist_loader_and_filter_drop_derivatives(tmp_path) -> None:
    txt = tmp_path / "extra_blacklist.txt"
    txt.write_text("META\n", encoding="utf-8")
    cfg = {
        "strategy": {"blacklist": ["APLD"]},
        "paths": {"blacklist_txt": txt.name},
    }
    blacklist = load_strategy_blacklist(cfg, base_dir=tmp_path)
    assert blacklist == {"APLD", "META"}

    universe = pd.DataFrame(
        [
            {"ETF": "APLZ", "Underlying": "APLD"},
            {"ETF": "FBYY", "Underlying": "META"},
            {"ETF": "KEEX", "Underlying": "KEEL"},
        ]
    )
    out = apply_strategy_blacklist_to_universe(universe, blacklist)
    assert out.to_dict("records") == [{"ETF": "KEEX", "Underlying": "KEEL"}]


def test_format_period_pnl_summary_from_cumulative_history() -> None:
    hist = pd.DataFrame(
        [
            {"date": "2026-04-24", "pnl_stock_sleeves": 30.0, "pnl_bucket_3": 0.0, "total_pnl": 30.0},
            {"date": "2026-04-27", "pnl_stock_sleeves": 40.0, "pnl_bucket_3": 1.0, "total_pnl": 41.0},
            {"date": "2026-04-28", "pnl_stock_sleeves": 58.0, "pnl_bucket_3": 2.0, "total_pnl": 60.0},
        ]
    )

    out = format_period_pnl_summary(hist, "2026-04-28")

    assert "Daily: Stock: 18.00 | B3: 1.00 | Total: 19.00" in out
    assert "Week-to-date: Stock: 28.00 | B3: 2.00 | Total: 30.00" in out
    assert "Month-to-date: Stock: 58.00 | B3: 2.00 | Total: 60.00" in out
