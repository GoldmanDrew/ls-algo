"""QBTS trade ledger from Flex + daily realized attributed to B4."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ibkr_accounting import parse_trade_events  # noqa: E402

DATES = [
    "2026-05-15",
    "2026-05-16",
    "2026-05-18",
    "2026-05-19",
    "2026-05-20",
    "2026-05-21",
    "2026-05-22",
    "2026-05-23",
]
QBTS_SYMS = {"QBTS", "QBTZ", "QBTX", "QBY"}


def main() -> None:
    rows: list[dict] = []
    for d in DATES:
        flex = ROOT / "data" / "runs" / d / "ibkr_flex" / "flex_trades.xml"
        if not flex.exists():
            continue
        te = parse_trade_events(flex)
        if te.empty:
            continue
        day = te[te["symbol"].isin(QBTS_SYMS) | (te["underlyingSymbol"] == "QBTS")].copy()
        if day.empty:
            continue
        day["run_date"] = d
        day["trade_date"] = day["dateTime"].astype(str).str[:10]
        rows.extend(day.to_dict("records"))

    if not rows:
        print("No QBTS trades found.")
        return

    df = pd.DataFrame(rows)
    df = df.sort_values(["dateTime", "symbol"]).reset_index(drop=True)

    pd.set_option("display.width", 240)
    pd.set_option("display.max_rows", 200)
    pd.set_option("display.float_format", lambda x: f"{x:,.2f}")

    print("=== QBTS complex trades (all dates in window, cumulative Flex) ===")
    print(
        df[
            [
                "run_date",
                "trade_date",
                "dateTime",
                "symbol",
                "buySell",
                "quantity",
                "tradePrice_base",
                "fifoPnlRealized_base",
                "openCloseIndicator",
                "orderReference",
            ]
        ].to_string(index=False)
    )

    print("\n=== Daily QBTS realized (trade_date, symbol) ===")
    daily = (
        df.groupby(["trade_date", "symbol"], as_index=False)
        .agg(qty=("quantity", "sum"), realized=("fifoPnlRealized_base", "sum"), n_trades=("quantity", "count"))
        .sort_values(["trade_date", "symbol"])
    )
    print(daily.to_string(index=False))

    print("\n=== Daily QBTS spot (QBTS) realized only ===")
    spot = df[df["symbol"] == "QBTS"].groupby("trade_date", as_index=False).agg(
        qty=("quantity", "sum"),
        realized=("fifoPnlRealized_base", "sum"),
        n_trades=("quantity", "count"),
    )
    print(spot.to_string(index=False))

    # Compare to B4 attributed realized on QBTS from accounting outputs
    print("\n=== QBTS B4 bucket attributed PnL (from pnl_bucket_4.csv) ===")
    pnl_rows = []
    for d in DATES:
        p = ROOT / "data" / "runs" / d / "accounting" / "pnl_bucket_4.csv"
        if not p.exists():
            continue
        b4 = pd.read_csv(p)
        row = b4[b4["underlying"] == "QBTS"]
        if row.empty:
            continue
        r = row.iloc[0]
        pnl_rows.append(
            {
                "run_date": d,
                "b4_realized": float(r.get("realized_pnl", 0)),
                "b4_unrealized": float(r.get("unrealized_pnl", 0)),
                "b4_total": float(r.get("total_pnl", 0)),
            }
        )
    pnl_df = pd.DataFrame(pnl_rows)
    print(pnl_df.to_string(index=False))

    # Timing state for inject_slice fraction on 05-21/22
    for d in ["2026-05-21", "2026-05-22", "2026-05-23"]:
        ts = ROOT / "data" / "runs" / d / "accounting" / "bucket_timing_state.csv"
        if not ts.exists():
            continue
        t = pd.read_csv(ts)
        q = t[t["underlying"] == "QBTS"]
        if q.empty:
            continue
        r = q.iloc[0]
        print(f"\n=== QBTS bucket_timing_state {d} ===")
        for col in [
            "plan_b4_frac",
            "ratio_spot_b4",
            "ratio_realized_b4",
            "pnl_split_mode",
            "exposure_b4_qty_source",
            "plan_b4_qty",
            "ibkr_qty",
        ]:
            if col in r:
                print(f"  {col}: {r[col]}")


if __name__ == "__main__":
    main()
