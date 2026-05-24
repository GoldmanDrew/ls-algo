"""Trace B1/B2/B4 PnL and exposure over recent trading days."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "data" / "runs"
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


def load_totals(d: str) -> dict:
    p = RUNS / d / "accounting" / "totals.json"
    return json.loads(p.read_text()) if p.exists() else {}


def main() -> None:
    rows = []
    for d in DATES:
        t = load_totals(d)
        if not t:
            continue
        bp = t.get("bucket_pnl") or {}
        rows.append(
            {
                "date": d,
                "ytd_total": float(t.get("total_pnl", 0)),
                "ytd_b1": float(bp.get("bucket_1", 0)),
                "ytd_b2": float(bp.get("bucket_2", 0)),
                "ytd_b3": float(bp.get("bucket_3", 0)),
                "ytd_b4": float(bp.get("bucket_4", 0)),
                "net_b1": t.get("net_exposure_bucket_1"),
                "net_b2": t.get("net_exposure_bucket_2"),
                "net_b4": t.get("net_exposure_bucket_4"),
                "net_b4_pair": t.get("net_exposure_bucket_4_pair"),
                "gross_b4_pair": t.get("gross_exposure_bucket_4_pair"),
                "b4_attr": t.get("b4_underlying_attribution", "legacy"),
            }
        )
    df = pd.DataFrame(rows)
    for col in ["ytd_total", "ytd_b1", "ytd_b2", "ytd_b3", "ytd_b4"]:
        df[f"d_{col.replace('ytd_', '')}"] = df[col].diff()
    for col in ["net_b1", "net_b2", "net_b4", "net_b4_pair", "gross_b4_pair"]:
        df[f"d_{col}"] = pd.to_numeric(df[col], errors="coerce").diff()

    pd.set_option("display.width", 220)
    pd.set_option("display.float_format", lambda x: f"{x:,.0f}")

    print("=== Daily PnL (diff of YTD in each run's totals.json) ===")
    print(
        df[
            [
                "date",
                "d_total",
                "d_b1",
                "d_b2",
                "d_b3",
                "d_b4",
                "ytd_b4",
                "b4_attr",
            ]
        ].to_string(index=False)
    )

    print("\n=== Exposure levels and day-over-day delta ===")
    print(
        df[
            [
                "date",
                "net_b1",
                "d_net_b1",
                "net_b2",
                "d_net_b2",
                "net_b4",
                "d_net_b4",
                "net_b4_pair",
                "gross_b4_pair",
                "b4_attr",
            ]
        ].to_string(index=False)
    )

    hist = pd.read_csv(ROOT / "data" / "ledger" / "pnl_history.csv")
    hist = hist[hist["date"].isin(DATES)].copy()
    for c in ["pnl_bucket_1", "pnl_bucket_2", "pnl_bucket_4", "total_pnl"]:
        hist[f"d_{c}"] = hist[c].diff()
    print("\n=== pnl_history.csv (EOD continuity-adjusted YTD diffs) ===")
    print(
        hist[
            [
                "date",
                "d_total_pnl",
                "d_pnl_bucket_1",
                "d_pnl_bucket_2",
                "d_pnl_bucket_4",
                "pnl_bucket_4",
            ]
        ].to_string(index=False)
    )

    for prev, curr in [("2026-05-21", "2026-05-22"), ("2026-05-22", "2026-05-23")]:
        p1 = RUNS / prev / "accounting" / "pnl_bucket_4.csv"
        p2 = RUNS / curr / "accounting" / "pnl_bucket_4.csv"
        if not p1.exists() or not p2.exists():
            continue
        m = pd.read_csv(p1)[["underlying", "total_pnl"]].merge(
            pd.read_csv(p2)[["underlying", "total_pnl"]],
            on="underlying",
            suffixes=("_prev", "_curr"),
        )
        m["delta"] = m["total_pnl_curr"] - m["total_pnl_prev"]
        m = m.sort_values("delta")
        print(f"\n=== B4 underlying delta {prev} -> {curr} ===")
        print(f"Sum delta: {m['delta'].sum():,.0f}")
        print("Worst:")
        print(m.head(10).to_string(index=False, float_format=lambda x: f"{x:,.0f}"))
        print("Best:")
        print(m.tail(5).sort_values("delta", ascending=False).to_string(index=False, float_format=lambda x: f"{x:,.0f}"))

        d1 = pd.read_csv(RUNS / prev / "accounting" / "net_exposure_bucket_4_detail.csv")
        d2 = pd.read_csv(RUNS / curr / "accounting" / "net_exposure_bucket_4_detail.csv")
        if not d1.empty and not d2.empty:
            e = d1.groupby("underlying")["net_notional_usd"].sum().reset_index(name="net_prev")
            e = e.merge(
                d2.groupby("underlying")["net_notional_usd"].sum().reset_index(name="net_curr"),
                on="underlying",
            )
            e["delta"] = e["net_curr"] - e["net_prev"]
            e = e[e["delta"].abs() > 500].sort_values("delta")
            print(f"\n=== B4 pair net exposure delta {prev} -> {curr} (abs delta > 500) ===")
            print(e.head(12).to_string(index=False, float_format=lambda x: f"{x:,.0f}"))


if __name__ == "__main__":
    main()
