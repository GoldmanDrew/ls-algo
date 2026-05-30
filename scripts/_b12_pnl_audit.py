from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1] / "data" / "runs" / "2026-05-25" / "accounting"
sym = pd.read_csv(ROOT / "pnl_by_symbol.csv")
timing = pd.read_csv(ROOT / "bucket_timing_state.csv")
names = ["IONQ", "COIN", "MSTR", "SMCI", "HOOD", "AMZN"]
yb_list = {"SMCI", "IONQ", "IBIT"}

print("=== Ledger vs PnL mode (spot) ===")
for u in names:
    t = timing.loc[timing["underlying"] == u].iloc[0]
    print(
        f"{u}: in_yb_list={u in yb_list} pnl_mode={t['pnl_split_mode']} "
        f"qty_b1={t['curr_qty_b1']:.0f} qty_b2={t['curr_qty_b2']:.0f} "
        f"orphan={t['orphan_qty']:.0f} ratio_spot_b2={t['ratio_spot_b2']:.2f}"
    )

print("\n=== B2 PnL: spot vs ETF legs ===")
for u in names:
    s = sym[(sym["underlying"] == u) & (sym["bucket"] == "bucket_2")]
    tot = s["total_pnl"].sum()
    spot = s.loc[s["symbol"] == u, "total_pnl"].sum()
    etf = s.loc[s["symbol"] != u, "total_pnl"].sum()
    print(f"{u}: B2 total=${tot:,.0f}  spot=${spot:,.0f}  etf=${etf:,.0f}  pil=${s['pil_dividends'].sum():,.0f}")

print("\n=== B1 spot (competing bucket) ===")
for u in names:
    b1s = sym[(sym["underlying"] == u) & (sym["bucket"] == "bucket_1") & (sym["symbol"] == u)][
        "total_pnl"
    ].sum()
    b2s = sym[(sym["underlying"] == u) & (sym["bucket"] == "bucket_2") & (sym["symbol"] == u)][
        "total_pnl"
    ].sum()
    print(f"{u}: B1_spot=${b1s:,.0f}  B2_spot=${b2s:,.0f}")
