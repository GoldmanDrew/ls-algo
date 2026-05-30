#!/usr/bin/env python3
"""Print per-underlying spot PnL sleeve attribution (B2/B4) for one accounting run."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> int:
    run_date = sys.argv[1] if len(sys.argv) > 1 else "2026-05-29"
    run = PROJECT_ROOT / "data" / "runs" / run_date / "accounting"
    sym = pd.read_csv(run / "pnl_by_symbol.csv")
    timing = pd.read_csv(run / "bucket_timing_state.csv")
    detail = pd.read_csv(run / "bucket_exposure_detail.csv")

    spot = sym[sym["symbol"].astype(str) == sym["underlying"].astype(str)].copy()
    spot = spot[spot["bucket"].isin(["bucket_1", "bucket_2", "bucket_4"])]

    rows: list[dict] = []
    for _, tr in timing.iterrows():
        u = str(tr["underlying"])
        s = spot[spot.underlying == u]
        if s.empty:
            continue
        pnl_b2 = float(s.loc[s.bucket == "bucket_2", "total_pnl"].sum())
        pnl_b4 = float(s.loc[s.bucket == "bucket_4", "total_pnl"].sum())
        if abs(pnl_b2) < 0.01 and abs(pnl_b4) < 0.01:
            continue

        pnl_total = float(s["total_pnl"].sum())
        pnl_b1 = float(s.loc[s.bucket == "bucket_1", "total_pnl"].sum())

        g = detail[detail.underlying.astype(str) == u]
        spot_net = (
            float(g.loc[g.symbol.astype(str) == u, "net_notional_usd"].sum())
            if not g.empty
            else np.nan
        )

        etf_b2 = etf_b4 = 0.0
        for _, r in g.iterrows():
            if str(r["symbol"]) == u:
                continue
            net = float(r["net_notional_usd"])
            if float(r.get("_ratio_b2", 0)) > 0.5:
                etf_b2 += net
            if float(r.get("_ratio_b4", 0)) > 0.5:
                etf_b4 += net

        struct_usd = float(tr.get("plan_b4_usd", 0) or 0)
        b4_frac = struct_usd / spot_net if spot_net and abs(spot_net) > 1 else np.nan
        if (b4_frac != b4_frac or abs(b4_frac) < 1e-9) and not g.empty:
            sr = g.loc[g.symbol.astype(str) == u]
            if not sr.empty and "_ratio_b4" in sr.columns:
                b4_frac = float(sr["_ratio_b4"].iloc[0])

        hr_b1 = float(tr["ratio_spot_exposure_b1"])
        hr_b2 = float(tr["ratio_spot_exposure_b2"])

        if b4_frac == b4_frac and abs(b4_frac) > 1e-12:
            r4 = b4_frac
            rem = 1.0 - r4
            s12 = hr_b1 + hr_b2
            r1 = rem * hr_b1 / s12 if s12 > 1e-12 else rem
            r2 = rem * hr_b2 / s12 if s12 > 1e-12 else 0.0
            method = "B2 hedge + B4 structural"
        else:
            r1, r2, r4 = hr_b1, hr_b2, 0.0
            s = r1 + r2 + r4
            if s < 1.0 - 1e-9:
                r1 += 1.0 - s
            method = "B2 hedge + orphan to B1" if s < 1.0 - 1e-9 else "B2 hedge only"

        rows.append(
            {
                "underlying": u,
                "spot_net_usd": spot_net,
                "etf_b2_net_usd": etf_b2,
                "etf_b4_net_usd": etf_b4,
                "struct_short_usd": struct_usd,
                "hr_b2": hr_b2,
                "b4_signed_frac": b4_frac,
                "pnl_frac_b1": r1,
                "pnl_frac_b2": r2,
                "pnl_frac_b4": r4,
                "spot_pnl_total": pnl_total,
                "spot_pnl_b1": pnl_b1,
                "spot_pnl_b2": pnl_b2,
                "spot_pnl_b4": pnl_b4,
                "method": method,
            }
        )

    df = pd.DataFrame(rows)
    out = run / "spot_sleeve_pnl_attribution.csv"
    df.sort_values("underlying").to_csv(out, index=False)
    print(f"Wrote {out} ({len(df)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
