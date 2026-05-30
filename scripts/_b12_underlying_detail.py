"""B1/B2 symbol-level breakdown for selected underlyings."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "data/runs/2026-05-25/accounting"
NAMES = ["COIN", "IONQ", "SMCI", "HOOD"]


def main() -> None:
    sym = pd.read_csv(RUN / "pnl_by_symbol.csv")
    exp = pd.read_csv(RUN / "bucket_exposure_detail.csv")
    comb = pd.read_csv(RUN / "pnl_by_underlying_b12.csv").set_index("underlying")
    screened = pd.read_csv(ROOT / "data/etf_screened_today.csv")
    etf_info = screened[screened["Underlying"].isin(NAMES)][
        ["ETF", "Underlying", "Delta", "is_yieldboost"]
    ].copy()
    etf_info["ETF"] = etf_info["ETF"].str.upper()

    for u in NAMES:
        row = comb.loc[u]
        print("=" * 72)
        print(
            f"{u}  combined B1+B2: ${row['total_pnl']:,.0f}  "
            f"(realized ${row['realized_pnl']:,.0f}  "
            f"unreal ${row['unrealized_pnl']:,.0f}  "
            f"PIL ${row['pil_dividends']:,.0f}  "
            f"borrow ${row['borrow_fees']:,.0f})"
        )
        print("ETFs in universe (by beta):")
        for _, e in etf_info[etf_info["Underlying"] == u].sort_values(
            "Delta", ascending=False
        ).iterrows():
            b = "B1 levered" if e["Delta"] > 1.5 else "B2 standard"
            yb = (
                " [yieldboost]"
                if str(e.get("is_yieldboost", "")).lower() in ("true", "1")
                else ""
            )
            print(f"  {e['ETF']:6}  beta={e['Delta']:.2f}  {b}{yb}")

        sub = sym[sym["underlying"] == u].copy()
        sub["leg"] = sub["description"].apply(
            lambda d: "spot" if "(spot)" in str(d) else "etf"
        )
        for bkt, label in [("bucket_1", "B1"), ("bucket_2", "B2")]:
            b = sub[sub["bucket"] == bkt]
            print(f"  {label} total: ${b['total_pnl'].sum():,.0f}")
            for _, r in b.sort_values("total_pnl", ascending=False).iterrows():
                print(
                    f"    {r['symbol']:6} {r['leg']:4}  "
                    f"${r['total_pnl']:>10,.0f}  "
                    f"real ${r['realized_pnl']:>9,.0f}  "
                    f"unrl ${r['unrealized_pnl']:>9,.0f}  "
                    f"PIL ${r['pil_dividends']:>8,.0f}  "
                    f"borrow ${r['borrow_fees']:>7,.0f}"
                )

        print("  Exposure (net USD, beta-normalized):")
        ex = exp[exp["underlying"] == u]
        leg_col = "leg_class" if "leg_class" in ex.columns else "leg_type"
        for _, r in ex.sort_values(leg_col).iterrows():
            print(
                f"    {r['symbol']:6} {str(r[leg_col]):18}  "
                f"net ${r['net_notional_usd']:>12,.0f}"
            )
        print()


if __name__ == "__main__":
    main()
