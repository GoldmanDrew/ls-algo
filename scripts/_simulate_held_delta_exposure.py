#!/usr/bin/env python3
"""Simulate B1/B2 net exposure if spot uses held-delta weights (2026-05-25)."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

RUN = Path(__file__).resolve().parents[1] / "data" / "runs" / "2026-05-25" / "accounting"


def main() -> None:
    detail = pd.read_csv(RUN / "bucket_exposure_detail.csv")
    timing = pd.read_csv(RUN / "bucket_timing_state.csv")
    with (RUN / "totals.json").open(encoding="utf-8") as f:
        totals = json.load(f)

    spot = detail[detail["leg_class"] == "spot"].copy()
    t = timing[
        [
            "underlying",
            "ratio_current_b1",
            "ratio_current_b2",
            "ratio_current_b4",
            "ratio_spot_b1",
            "ratio_spot_b2",
        ]
    ]
    spot = spot.merge(t, on="underlying", how="left")

    spot["net"] = pd.to_numeric(spot["net_notional_usd"], errors="coerce").fillna(0)
    spot["fifo_b1"] = pd.to_numeric(spot["_ratio_b1"], errors="coerce").fillna(0)
    spot["fifo_b2"] = pd.to_numeric(spot["_ratio_b2"], errors="coerce").fillna(0)
    spot["held_b1"] = pd.to_numeric(spot["ratio_current_b1"], errors="coerce").fillna(0)
    spot["held_b2"] = pd.to_numeric(spot["ratio_current_b2"], errors="coerce").fillna(0)
    s12 = spot["held_b1"] + spot["held_b2"]
    mask = s12 > 1e-12
    spot.loc[mask, "held_b1"] = spot.loc[mask, "held_b1"] / s12
    spot.loc[mask, "held_b2"] = spot.loc[mask, "held_b2"] / s12

    spot["net_fifo_b1"] = spot["net"] * spot["fifo_b1"]
    spot["net_fifo_b2"] = spot["net"] * spot["fifo_b2"]
    spot["net_held_b1"] = spot["net"] * spot["held_b1"]
    spot["net_held_b2"] = spot["net"] * spot["held_b2"]

    b1 = pd.read_csv(RUN / "net_exposure_bucket_1.csv")
    b2 = pd.read_csv(RUN / "net_exposure_bucket_2.csv")

    def sum_net(df: pd.DataFrame) -> float:
        return float(pd.to_numeric(df["net_notional_usd"], errors="coerce").fillna(0).sum())

    cur_b1 = sum_net(b1)
    cur_b2 = sum_net(b2)
    cur_spot_b1 = float(spot["net_fifo_b1"].sum())
    cur_spot_b2 = float(spot["net_fifo_b2"].sum())
    cur_etf_b1 = cur_b1 - cur_spot_b1
    cur_etf_b2 = cur_b2 - cur_spot_b2

    prop_spot_b1 = float(spot["net_held_b1"].sum())
    prop_spot_b2 = float(spot["net_held_b2"].sum())
    prop_b1 = cur_etf_b1 + prop_spot_b1
    prop_b2 = cur_etf_b2 + prop_spot_b2

    print("=== Book (unchanged) ===")
    print(f"net_exposure_total: {totals['net_exposure_total']:,.0f}")
    print(f"gross_exposure_total: {totals['gross_exposure_total']:,.0f}")
    print()
    print("=== Current (FIFO spot on ratio-split) ===")
    print(f"Bucket 1 net: {cur_b1:,.0f}  (spot {cur_spot_b1:,.0f} + non-spot {cur_etf_b1:,.0f})")
    print(f"Bucket 2 net: {cur_b2:,.0f}  (spot {cur_spot_b2:,.0f} + ETF {cur_etf_b2:,.0f})")
    print(f"B1+B2 net: {cur_b1 + cur_b2:,.0f}")
    print(f"Bucket 4 net (ratio-split): {totals['net_exposure_bucket_4']:,.0f}")
    print()
    print("=== Proposed (held-delta spot, ETF legs unchanged) ===")
    print(f"Bucket 1 net: {prop_b1:,.0f}  (spot {prop_spot_b1:,.0f} + non-spot {cur_etf_b1:,.0f})")
    print(f"Bucket 2 net: {prop_b2:,.0f}  (spot {prop_spot_b2:,.0f} + ETF {cur_etf_b2:,.0f})")
    print(f"B1+B2 net: {prop_b1 + prop_b2:,.0f}")
    print()
    print("=== Delta (proposed - current) ===")
    print(f"B1: {prop_b1 - cur_b1:+,.0f}")
    print(f"B2: {prop_b2 - cur_b2:+,.0f}")
    print()
    print("=== Top spot movers to B2 ===")
    spot["d_b2"] = spot["net_held_b2"] - spot["net_fifo_b2"]
    top = spot.reindex(spot["d_b2"].abs().sort_values(ascending=False).index).head(15)
    for _, r in top.iterrows():
        print(
            f"{r['underlying']:6s} spot={r['net']:>11,.0f}  "
            f"fifo_b2={r['fifo_b2']:5.1%} held_b2={r['held_b2']:5.1%}  dB2={r['d_b2']:+,.0f}"
        )


if __name__ == "__main__":
    main()
