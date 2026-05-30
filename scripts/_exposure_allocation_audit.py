#!/usr/bin/env python3
"""Leg-level exposure allocation audit for B1/B2 ratio-split."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

RUN = Path(__file__).resolve().parents[1] / "data" / "runs" / "2026-05-25" / "accounting"
NAMES = ["IONQ", "SMCI", "COIN", "HOOD", "IBIT", "MSTR", "AMD", "META"]


def main() -> None:
    detail = pd.read_csv(RUN / "bucket_exposure_detail.csv")
    timing = pd.read_csv(RUN / "bucket_timing_state.csv")
    by_u = pd.read_csv(RUN / "net_exposure_by_underlying.csv")
    b1 = pd.read_csv(RUN / "net_exposure_bucket_1.csv")
    b2 = pd.read_csv(RUN / "net_exposure_bucket_2.csv")
    b4r = pd.read_csv(RUN / "net_exposure_bucket_4.csv")
    b4d = pd.read_csv(RUN / "net_exposure_bucket_4_detail.csv")

    detail["net"] = pd.to_numeric(detail["net_notional_usd"], errors="coerce").fillna(0)
    for c in ("_ratio_b1", "_ratio_b2", "_ratio_b4"):
        detail[c] = pd.to_numeric(detail[c], errors="coerce").fillna(0)
    detail["to_b1"] = detail["net"] * detail["_ratio_b1"]
    detail["to_b2"] = detail["net"] * detail["_ratio_b2"]
    detail["to_b4"] = detail["net"] * detail["_ratio_b4"]

    print("=" * 88)
    print("BOOK vs COMBINED UNDERLYING (b1+b2+b4 ratio-split + b3 separate)")
    with (RUN / "totals.json").open(encoding="utf-8") as f:
        t = json.load(f)
    print(f"  Book net: {t['net_exposure_total']:,.0f}")
    print(f"  B1 ratio-split net: {t['net_exposure_bucket_1']:,.0f}")
    print(f"  B2 ratio-split net: {t['net_exposure_bucket_2']:,.0f}")
    print(f"  B4 ratio-split net: {t['net_exposure_bucket_4']:,.0f}")
    print(f"  B4 pair view net: {t.get('net_exposure_bucket_4_pair', 0):,.0f}")
    print()

    for u in NAMES:
        if u not in detail["underlying"].values:
            continue
        g = detail[detail["underlying"] == u].copy()
        tr = timing[timing["underlying"] == u]
        comb = by_u[by_u["underlying"] == u]
        print("=" * 88)
        print(f"UNDERLYING: {u}")
        if not comb.empty:
            print(
                f"  Combined (all legs, unscaled): net={comb.iloc[0]['net_notional_usd']:,.0f}  "
                f"gross={comb.iloc[0]['gross_notional_usd']:,.0f}  symbols={comb.iloc[0]['symbols']}"
            )
        if not tr.empty:
            r = tr.iloc[0]
            print(
                f"  Timing: ratio_current b1/b2/b4 = "
                f"{r['ratio_current_b1']:.1%}/{r['ratio_current_b2']:.1%}/{r['ratio_current_b4']:.1%}  |  "
                f"ratio_spot (PnL FIFO) = {r['ratio_spot_b1']:.1%}/{r['ratio_spot_b2']:.1%}  |  "
                f"ratio_spot_exposure = {r.get('ratio_spot_exposure_b1', r['ratio_spot_b1']):.1%}/"
                f"{r.get('ratio_spot_exposure_b2', r['ratio_spot_b2']):.1%}  "
                f"({r.get('ratio_spot_exposure_source', r['ratio_spot_source'])})"
            )
            print(
                f"  Ledger qty: b1={r['curr_qty_b1']:.0f} b2={r['curr_qty_b2']:.0f} b4={r['curr_qty_b4']:.0f}  "
                f"ibkr={r['ibkr_qty']:.0f}"
            )
        n1 = float(b1.loc[b1["underlying"] == u, "net_notional_usd"].sum()) if u in b1["underlying"].values else 0.0
        n2 = float(b2.loc[b2["underlying"] == u, "net_notional_usd"].sum()) if u in b2["underlying"].values else 0.0
        n4r = float(b4r.loc[b4r["underlying"] == u, "net_notional_usd"].sum()) if u in b4r["underlying"].values else 0.0
        n4d = float(b4d.loc[b4d["underlying"] == u, "net_notional_usd"].sum()) if u in b4d["underlying"].values else 0.0
        print(
            f"  Published bucket nets: B1={n1:,.0f}  B2={n2:,.0f}  "
            f"B4(ratio)={n4r:,.0f}  B4(pair detail)={n4d:,.0f}  "
            f"B1+B2={n1+n2:,.0f}"
        )
        print()
        print(
            f"  {'symbol':8s} {'class':18s} {'leg_net':>12s} "
            f"{'r_b1':>6s} {'r_b2':>6s} {'r_b4':>6s} "
            f"{'->B1':>12s} {'->B2':>12s} {'->B4':>12s}"
        )
        print("  " + "-" * 84)
        for _, row in g.sort_values("net", ascending=False).iterrows():
            print(
                f"  {row['symbol']:8s} {row['leg_class']:18s} {row['net']:>12,.0f} "
                f"{row['_ratio_b1']:>6.2f} {row['_ratio_b2']:>6.2f} {row['_ratio_b4']:>6.2f} "
                f"{row['to_b1']:>12,.0f} {row['to_b2']:>12,.0f} {row['to_b4']:>12,.0f}"
            )
        print("  " + "-" * 84)
        print(
            f"  {'TOTAL':8s} {'':18s} {g['net'].sum():>12,.0f} "
            f"{'':>6s} {'':>6s} {'':>6s} "
            f"{g['to_b1'].sum():>12,.0f} {g['to_b2'].sum():>12,.0f} {g['to_b4'].sum():>12,.0f}"
        )
        print()

    # Portfolio: why B1/B2 are huge but names are flat
    print("=" * 88)
    print("PORTFOLIO: per-underlying combined net vs B1+B2 sleeve (top flat names)")
    rows = []
    for u in by_u["underlying"].astype(str):
        comb = float(by_u.loc[by_u["underlying"] == u, "net_notional_usd"].iloc[0])
        n1 = float(b1.loc[b1["underlying"] == u, "net_notional_usd"].sum()) if u in b1["underlying"].values else 0.0
        n2 = float(b2.loc[b2["underlying"] == u, "net_notional_usd"].sum()) if u in b2["underlying"].values else 0.0
        rows.append({"u": u, "combined": comb, "b1": n1, "b2": n2, "b12": n1 + n2})
    audit = pd.DataFrame(rows).sort_values("combined", key=abs, ascending=False)
    print(audit.head(20).to_string(index=False, formatters={
        "combined": "{:,.0f}".format,
        "b1": "{:,.0f}".format,
        "b2": "{:,.0f}".format,
        "b12": "{:,.0f}".format,
    }))


if __name__ == "__main__":
    main()
