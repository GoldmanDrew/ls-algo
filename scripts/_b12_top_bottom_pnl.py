"""Print top/bottom 5 underlying PnL for buckets 1 and 2."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "data/runs/2026-05-25/accounting"


def show_bucket(bucket: int) -> None:
    df = pd.read_csv(RUN / f"pnl_bucket_{bucket}.csv")
    df = df.sort_values("total_pnl", ascending=False)
    total = float(df["total_pnl"].sum())
    cols = [
        "underlying",
        "realized_pnl",
        "unrealized_pnl",
        "pil_dividends",
        "borrow_fees",
        "total_pnl",
    ]
    print(f"=== BUCKET {bucket} YTD: ${total:,.0f} ({len(df)} underlyings) ===")
    print("Top 5:")
    for _, r in df.head(5)[cols].iterrows():
        print(
            f"  {r['underlying']:8}  ${r['total_pnl']:>11,.0f}  "
            f"realized ${r['realized_pnl']:>10,.0f}  "
            f"unreal ${r['unrealized_pnl']:>10,.0f}  "
            f"PIL ${r['pil_dividends']:>9,.0f}  "
            f"borrow ${r['borrow_fees']:>8,.0f}"
        )
    print("Bottom 5:")
    for _, r in df.tail(5).sort_values("total_pnl")[cols].iterrows():
        print(
            f"  {r['underlying']:8}  ${r['total_pnl']:>11,.0f}  "
            f"realized ${r['realized_pnl']:>10,.0f}  "
            f"unreal ${r['unrealized_pnl']:>10,.0f}  "
            f"PIL ${r['pil_dividends']:>9,.0f}  "
            f"borrow ${r['borrow_fees']:>8,.0f}"
        )
    print()


def main() -> None:
    show_bucket(1)
    show_bucket(2)


if __name__ == "__main__":
    main()
