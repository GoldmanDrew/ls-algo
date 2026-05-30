"""One-off: top-6 bucket-2 PnL with B1/B4 comparison."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1] / "data" / "runs" / "2026-05-25" / "accounting"


def _print_bucket_lines(sym: pd.DataFrame, underlying: str, bucket: str, label: str) -> None:
    det = sym[(sym["underlying"] == underlying) & (sym["bucket"] == bucket)].sort_values(
        "total_pnl", ascending=False
    )
    if det.empty:
        print(f"  {label}: (no row)")
        return
    for _, r in det.iterrows():
        leg = "spot" if r["symbol"] == underlying else "etf"
        print(
            f"    {label} {str(r['symbol']):8} {leg:4}  "
            f"total={r['total_pnl']:>10,.0f}  "
            f"real={r['realized_pnl']:>8,.0f}  "
            f"unreal={r['unrealized_pnl']:>10,.0f}"
        )


def main() -> None:
    b2 = pd.read_csv(ROOT / "pnl_bucket_2.csv").sort_values("total_pnl", ascending=False)
    b1 = pd.read_csv(ROOT / "pnl_bucket_1.csv")
    b4 = pd.read_csv(ROOT / "pnl_bucket_4.csv")
    sym = pd.read_csv(ROOT / "pnl_by_symbol.csv")

    top6 = b2.head(6)["underlying"].tolist()
    print(f"B2 bucket total: ${b2['total_pnl'].sum():,.0f}")
    print(f"Top 6 names: {', '.join(top6)}")
    print(f"Top 6 sum: ${b2.head(6)['total_pnl'].sum():,.0f} ({100*b2.head(6)['total_pnl'].sum()/b2['total_pnl'].sum():.1f}% of B2)")

    for u in top6:
        print("\n" + "=" * 72)
        print(f"UNDERLYING: {u}")
        row2 = b2[b2["underlying"] == u].iloc[0]
        print(
            f"  B2 rollup: {row2['symbols']}  "
            f"total=${row2['total_pnl']:,.0f}  "
            f"(real=${row2['realized_pnl']:,.0f} unreal=${row2['unrealized_pnl']:,.0f} "
            f"pil=${row2['pil_dividends']:,.0f} borrow=${row2['borrow_fees']:,.0f})"
        )
        _print_bucket_lines(sym, u, "bucket_2", "B2")

        r1 = b1[b1["underlying"] == u]
        if not r1.empty:
            r1 = r1.iloc[0]
            print(
                f"  B1 rollup: {r1['symbols']}  total=${r1['total_pnl']:,.0f}"
            )
            _print_bucket_lines(sym, u, "bucket_1", "B1")
        else:
            print("  B1 rollup: —")

        r4 = b4[b4["underlying"] == u]
        if not r4.empty:
            r4 = r4.iloc[0]
            print(
                f"  B4 rollup: {r4['symbols']}  total=${r4['total_pnl']:,.0f}"
            )
            _print_bucket_lines(sym, u, "bucket_4", "B4")
        else:
            print("  B4 rollup: —")


if __name__ == "__main__":
    main()
