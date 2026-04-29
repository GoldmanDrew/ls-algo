from __future__ import annotations

import pandas as pd

from daily_screener import (
    apply_strategy_blacklist_to_universe,
    build_full_universe,
    load_strategy_blacklist,
)
from ibkr_accounting import _bucket_hint_from_order_reference
from run_eod_pnl_email import format_period_pnl_summary


def test_order_ref_standalone_etf_token_classifies_bucket_2() -> None:
    etf_to_beta = {"FBYY": 0.4, "FBL": 2.0}
    ref = "ETF_LS|META__GROUP|FBYY|ETF_DELTA|att1|ADAPTIVE_MKT"
    assert _bucket_hint_from_order_reference(ref, etf_to_beta) == "bucket_2"


def test_order_ref_standalone_etf_token_classifies_bucket_1() -> None:
    etf_to_beta = {"FBYY": 0.4, "FBL": 2.0}
    ref = "ETF_LS|META__GROUP|FBL|ETF_DELTA|att1|ADAPTIVE_MKT"
    assert _bucket_hint_from_order_reference(ref, etf_to_beta) == "bucket_1"


def test_order_ref_standalone_inverse_token_classifies_bucket_4() -> None:
    etf_to_beta = {"APLZ": -2.0}
    ref = "ETF_LS|APLD__GROUP|APLZ|ETF_DELTA|att1|ADAPTIVE_MKT"
    assert _bucket_hint_from_order_reference(ref, etf_to_beta) == "bucket_4"


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
            {"date": "2026-04-24", "pnl_bucket_1": 10.0, "pnl_bucket_2": 20.0, "pnl_bucket_3": 0.0, "pnl_bucket_4": 0.0, "total_pnl": 30.0},
            {"date": "2026-04-27", "pnl_bucket_1": 15.0, "pnl_bucket_2": 25.0, "pnl_bucket_3": 1.0, "pnl_bucket_4": 0.0, "total_pnl": 41.0},
            {"date": "2026-04-28", "pnl_bucket_1": 18.0, "pnl_bucket_2": 35.0, "pnl_bucket_3": 2.0, "pnl_bucket_4": 5.0, "total_pnl": 60.0},
        ]
    )

    out = format_period_pnl_summary(hist, "2026-04-28")

    assert "Daily: B1: 3.00 | B2: 10.00 | B3: 1.00 | B4: 5.00 | Total: 19.00" in out
    assert "Week-to-date: B1: 8.00 | B2: 15.00 | B3: 2.00 | B4: 5.00 | Total: 30.00" in out
    assert "Month-to-date: B1: 18.00 | B2: 35.00 | B3: 2.00 | B4: 5.00 | Total: 60.00" in out
