"""Schema smoke tests for Bucket 5 product dashboard exporters."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.bucket5_product_export import (  # noqa: E402
    build_b5_insurance_strategy_guide,
    build_run_payload,
    export_daily_rows,
    export_day_marks,
    export_events_by_date,
)


def _toy_res():
    idx = pd.bdate_range("2024-01-02", periods=5)
    carry = pd.DataFrame(
        {
            "uvix": [20.0, 21.0, 19.0, 18.0, 17.5],
            "svix": [30.0, 29.0, 31.0, 32.0, 33.0],
            "rho": [1.0] * 5,
            "gross_mult": [0.2] * 5,
            "cash": [1e6] * 5,
            "u_notional": [-100_000.0] * 5,
            "s_notional": [-100_000.0] * 5,
            "gross": [200_000.0] * 5,
            "equity": [1e6] * 5,
            "beta_notional": [0.0] * 5,
            "borrow_cost": [10.0] * 5,
            "financing_pnl": [-10.0] * 5,
            "rebalance": [True, False, False, True, False],
            "rebalance_friction": [5.0, 0, 0, 5.0, 0],
            "ret": [0.0, 0.001, -0.002, 0.001, 0.0],
            "drawdown": [0.0, 0.0, -0.002, -0.001, -0.001],
        },
        index=idx,
    )
    ladder = pd.DataFrame(
        {
            "put_mtm": [1000.0, 1200.0, 5000.0, 3000.0, 2500.0],
            "put_cash_flow": [0.0, 0.0, 2000.0, 0.0, 0.0],
            "realized": [0.0, 0.0, 2000.0, 0.0, 0.0],
        },
        index=idx,
    )
    ladder.attrs["monetize_events"] = [
        {"date": "2024-01-04", "kind": "profit_3x", "usd": 2000.0, "otm_pct": 0.1, "mult": 3.1, "vix": 28.0, "contracts_sold": 2}
    ]
    ladder.attrs["realized_total"] = 2000.0
    base = pd.Series([1e6, 1.001e6, 0.999e6, 1.0e6, 1.001e6], index=idx)
    bt = pd.DataFrame(
        {
            "ratio": [0.9] * 5,
            "rho": [1.0] * 5,
            "gross_frac": [0.2] * 5,
            "sleeve_equity": carry["equity"],
            "base_equity": base,
            "put_mtm": ladder["put_mtm"],
            "put_cash_cum": ladder["put_cash_flow"].cumsum(),
            "realized_cum": ladder["realized"].cumsum(),
            "redeploy_extra": [0.0] * 5,
            "combined_equity": base + ladder["put_mtm"] + ladder["put_cash_flow"].cumsum(),
        },
        index=idx,
    )
    bt["combined_ret"] = bt["combined_equity"].pct_change().fillna(0.0)
    bt["drawdown"] = bt["combined_equity"].div(bt["combined_equity"].cummax()).sub(1.0)
    bt.attrs["rebalances"] = 2
    from scripts.bucket5_insurance_bt import production_config

    cfg = production_config()
    return {
        "bt": bt,
        "carry": carry,
        "ladder": ladder,
        "rebal": pd.DatetimeIndex([idx[0], idx[3]]),
        "cfg": cfg,
    }


def test_strategy_guide_shape():
    g = build_b5_insurance_strategy_guide(
        {"combined_CAGR": 0.14, "combined_Sharpe": 1.0, "combined_MaxDD": -0.2, "realized_$": 1e5},
        meta={"start": "2008-01-02", "end": "2026-07-13", "synthetic_days": 100},
    )
    assert g["title"]
    assert g["sections"]
    assert g["results"]
    assert all("title" in s for s in g["sections"])


def test_daily_and_marks_export():
    res = _toy_res()
    daily = export_daily_rows(res)
    assert len(daily) == 5
    assert {"date", "combined_equity", "rho", "put_mtm", "realized_day", "u_notional", "s_notional", "uvix_px"} <= set(
        daily[0]
    )
    marks = export_day_marks(res)
    assert "2024-01-04" in marks
    kinds = {m["kind"] for m in marks["2024-01-04"]}
    assert "carry_leg" in kinds
    assert "put_overlay" in kinds
    events = export_events_by_date(res)
    assert events["2024-01-04"][0]["kind"] == "profit_3x"


def test_product_ui_assets_exist():
    css = REPO / "site" / "assets" / "css" / "bucket5_product.css"
    js = REPO / "site" / "assets" / "js" / "bucket5_product.js"
    assert css.is_file()
    assert js.is_file()
    text = css.read_text(encoding="utf-8")
    assert ".b5p-root" in text and ".b5p-cards" in text and ".b5p-tabs" in text
    assert "b5p-tabs" in js.read_text(encoding="utf-8")


def test_build_run_payload_schema():
    res = _toy_res()
    panel = pd.DataFrame({"vix": [18.0] * 5, "ratio": [0.9] * 5}, index=res["bt"].index)
    payload = build_run_payload(
        run_id="toy",
        label="Toy",
        res=res,
        panel=panel,
        summary={"combined_CAGR": 0.1, "combined_Sharpe": 0.8, "combined_MaxDD": -0.1, "realized_$": 2000.0},
        crash={"crash_mild_-20%": 0.3},
        meta={"start": "2024-01-02", "end": "2024-01-08", "era": "live", "synthetic_days": 0, "rebalances": 2},
        assumptions={"sleeve_frac": 0.2, "tbill_rate": 0.043, "borrow_uvix_annual": 0.03, "borrow_svix_annual": 0.03},
        include_guide=True,
    )
    assert payload["id"] == "toy"
    assert payload["meta"]["strategy_guide"]["sections"]
    assert payload["daily"]
    assert payload["events_by_date"]["2024-01-04"]
    assert payload["regime_panels"]["ratio"]
