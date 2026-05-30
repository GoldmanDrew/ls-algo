"""Compare bucket PnL: yieldboost override vs FIFO-only."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1] / "data" / "runs" / "2026-05-25" / "accounting"


def main() -> None:
    old_t = json.loads((ROOT / "totals_yieldboost_override.json").read_text(encoding="utf-8"))
    new_t = json.loads((ROOT / "totals.json").read_text(encoding="utf-8"))
    old_b = pd.read_csv(ROOT / "pnl_by_bucket_yieldboost_override.csv").set_index("bucket")
    new_b = pd.read_csv(ROOT / "pnl_by_bucket.csv").set_index("bucket")

    print("=== Per-bucket total PnL (YTD) ===")
    print(f"{'Bucket':<10} {'Override ON':>14} {'FIFO only':>14} {'Delta':>12}")
    for b in ["bucket_1", "bucket_2", "bucket_3", "bucket_4"]:
        o = float(old_b.loc[b, "total_pnl"])
        n = float(new_b.loc[b, "total_pnl"])
        print(f"{b:<10} ${o:>12,.0f} ${n:>12,.0f} ${n - o:>+10,.0f}")
    print(
        f"{'BOOK':<10} ${old_t['total_pnl']:>12,.0f} ${new_t['total_pnl']:>12,.0f} "
        f"${new_t['total_pnl'] - old_t['total_pnl']:>+10,.0f}"
    )

    names = ["IONQ", "SMCI", "IBIT", "COIN", "MSTR", "HOOD"]
    for label, fname in [("Override ON", "pnl_bucket_2.csv"), ("FIFO only", "pnl_bucket_2.csv")]:
        pass

    old2 = pd.read_csv(ROOT / "pnl_bucket_2.csv") if label == "FIFO only" else None
    # reload from backup - we overwrote pnl_bucket_2; use totals only for top names from sym
    sym_new = pd.read_csv(ROOT / "pnl_by_symbol.csv")
    print("\n=== Top-6 B2 underlyings (rollup) — FIFO only ===")
    b2 = pd.read_csv(ROOT / "pnl_bucket_2.csv").sort_values("total_pnl", ascending=False)
    print(b2.head(8)[["underlying", "symbols", "total_pnl"]].to_string(index=False))

    print("\n=== Spot PnL shift for yieldboost names (FIFO only) ===")
    for u in ["IONQ", "SMCI", "IBIT"]:
        b1s = sym_new[(sym_new["underlying"] == u) & (sym_new["bucket"] == "bucket_1") & (sym_new["symbol"] == u)][
            "total_pnl"
        ].sum()
        b2s = sym_new[(sym_new["underlying"] == u) & (sym_new["bucket"] == "bucket_2") & (sym_new["symbol"] == u)][
            "total_pnl"
        ].sum()
        b2e = sym_new[(sym_new["underlying"] == u) & (sym_new["bucket"] == "bucket_2") & (sym_new["symbol"] != u)][
            "total_pnl"
        ].sum()
        print(f"{u}: B1 spot=${b1s:,.0f}  B2 spot=${b2s:,.0f}  B2 ETFs=${b2e:,.0f}")


if __name__ == "__main__":
    main()
