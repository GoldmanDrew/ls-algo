"""Print HOYY's final allocation in three reference frames."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd


def main() -> None:
    df = pd.read_csv("data/runs/2026-05-14/proposed_trades.csv")
    yb = df[df["sleeve"] == "yieldboost"].copy()
    yb["gross"] = yb["long_usd"] + (-yb["short_usd"])
    yb_total = float(yb["gross"].sum())
    target_book = 3_200_000.0
    yb_target = 1_600_000.0
    yb["frac_of_target_book"] = yb["gross"] / target_book
    yb["frac_of_yb_target"] = yb["gross"] / yb_target
    yb["frac_of_yb_deployed"] = yb["gross"] / yb_total
    print("YB sleeve final allocation:")
    print(
        yb[
            [
                "ETF",
                "borrow_current",
                "gross",
                "frac_of_target_book",
                "frac_of_yb_target",
                "frac_of_yb_deployed",
            ]
        ]
        .sort_values("gross", ascending=False)
        .to_string(index=False)
    )
    print()
    h = yb[yb["ETF"] == "HOYY"]
    if not h.empty:
        g = float(h["gross"].iloc[0])
        b = float(h["borrow_current"].iloc[0])
        print(
            f"HOYY: borrow={b:.1%} | gross=${g:,.0f}  "
            f"({g/target_book:.2%} of $3.2M book, "
            f"{g/yb_target:.2%} of $1.6M YB target, "
            f"{g/yb_total:.1%} of $%.0f YB deployed)" % yb_total
        )


if __name__ == "__main__":
    main()
