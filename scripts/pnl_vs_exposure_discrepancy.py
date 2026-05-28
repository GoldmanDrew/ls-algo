"""PnL vs exposure spot-ratio discrepancy (ledger_fifo PnL vs exposure methods)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.compare_spot_exposure_methods import (  # noqa: E402
    _load_run,
    _per_underlying_leg_stats,
    _structural_usd_from_detail,
    spot_ratios_option_a,
    spot_ratios_current_hr,
)

YB = [
    "AMD", "BABA", "COIN", "HIMS", "HOOD", "IBIT", "IONQ", "MARA",
    "META", "MSTR", "QBTS", "RIOT", "SMCI",
]


def main() -> None:
    run_date = "2026-05-25"
    detail, timing, totals = _load_run(run_date)
    timing = timing.set_index("underlying")
    legs = pd.read_csv(ROOT / "data" / "runs" / run_date / "accounting" / "bucket_leg_classification.csv")
    recon = pd.read_csv(ROOT / "data" / "runs" / run_date / "accounting" / "bucket_ratio_reconciliation.csv")

    d2 = detail.copy()
    leg_map = {"core_levered_etf": 2.0, "yieldboost_etf": 1.0, "flow_low_delta": 1.0, "inverse_b4_etf": -1.0}
    d2["_beta"] = d2["leg_class"].map(leg_map).fillna(1.0)
    etf_to_under = {str(r["symbol"]): str(r["underlying"]) for _, r in d2.iterrows()}
    stats = _per_underlying_leg_stats(d2, etf_to_under, set(), set())
    b4_b12 = set(totals.get("b4_plan_exposure_underlyings") or []) | set(
        totals.get("b4_etf_implied_underlyings") or []
    )

    # Spot-only PnL by bucket per underlying
    spot_pnl: dict[str, dict[str, float]] = {}
    for u, grp in legs[legs["symbol"] == legs["underlying"]].groupby("underlying"):
        u = str(u)
        spot_pnl[u] = {
            "b1": float(grp.loc[grp["bucket"] == "bucket_1", "total_pnl"].sum()),
            "b2": float(grp.loc[grp["bucket"] == "bucket_2", "total_pnl"].sum()),
            "b4": float(grp.loc[grp["bucket"] == "bucket_4", "total_pnl"].sum()),
        }

    rows = []
    for u in timing.index:
        u = str(u)
        if u not in stats:
            continue
        st = stats[u]
        if abs(st["spot_net"]) < 500 and u not in YB:
            continue
        tr = timing.loc[u]
        r_pnl_b1 = float(tr.get("ratio_spot_b1", 1) or 0)
        r_pnl_b2 = float(tr.get("ratio_spot_b2", 0) or 0)
        r_exp_b1 = float(tr.get("ratio_spot_exposure_b1", r_pnl_b1) or 0)
        r_exp_b2 = float(tr.get("ratio_spot_exposure_b2", r_pnl_b2) or 0)
        sr_a = spot_ratios_option_a(st, b4_b12_only=(u in b4_b12))
        sr_h = spot_ratios_current_hr(st, b4_b12_only=(u in b4_b12))
        sp = spot_pnl.get(u, {"b1": 0.0, "b2": 0.0, "b4": 0.0})
        pnl_tot = sp["b1"] + sp["b2"] + sp["b4"]
        rows.append({
            "u": u,
            "pnl_spot_total": pnl_tot,
            "pnl_b1": sp["b1"],
            "pnl_b2": sp["b2"],
            "r_pnl_b1": r_pnl_b1,
            "r_pnl_b2": r_pnl_b2,
            "r_exp_hr_b1": sr_h.b1,
            "r_exp_hr_b2": sr_h.b2,
            "r_exp_sb_b1": sr_a.b1,
            "r_exp_sb_b2": sr_a.b2,
            "diff_pnl_vs_hr_b2": abs(r_pnl_b2 - sr_h.b2),
            "diff_pnl_vs_sb_b2": abs(r_pnl_b2 - sr_a.b2),
            "diff_hr_vs_sb_b2": abs(sr_h.b2 - sr_a.b2),
        })

    df = pd.DataFrame(rows)
    print("=== Published totals (2026-05-25) ===")
    bp = totals.get("bucket_pnl", {})
    print(f"PnL  B1=${bp.get('bucket_1',0):,.0f}  B2=${bp.get('bucket_2',0):,.0f}  B4=${bp.get('bucket_4',0):,.0f}")
    print(
        f"Exp  B1=${totals['net_exposure_bucket_1']:,.0f}  B2=${totals['net_exposure_bucket_2']:,.0f}  "
        f"B4=${totals['net_exposure_bucket_4']:,.0f}  (hedge_ratio)"
    )
    print(f"Gate spot_ratio_max_diff_pnl={totals.get('spot_ratio_max_diff_pnl')} (spot PnL share vs hedge_ratio map)")

    print("\n=== Spot ratio max |diff| (underlyings with spot) ===")
    print(f"PnL ledger vs exposure hedge_ratio:  max diff_b2 = {df['diff_pnl_vs_hr_b2'].max():.3f}")
    print(f"PnL ledger vs exposure sleeve_balance: max diff_b2 = {df['diff_pnl_vs_sb_b2'].max():.3f}")
    print(f"hedge_ratio vs sleeve_balance exposure: max diff_b2 = {df['diff_hr_vs_sb_b2'].max():.3f}")

    # Counterfactual: re-split *spot-only* PnL using exposure ratios
    def counterfactual(pnl_u: dict, r1: float, r2: float) -> tuple[float, float]:
        t = pnl_u["b1"] + pnl_u["b2"] + pnl_u["b4"]
        if abs(t) < 1e-6:
            return 0.0, 0.0
        r4 = 0.0
        rem = 1.0 - r1 - r2 - r4
        # unbucketed spot PnL not assigned
        b1 = t * r1
        b2 = t * r2
        return b1, b2

    cf_hr_b1 = cf_hr_b2 = cf_sb_b1 = cf_sb_b2 = act_b1 = act_b2 = 0.0
    unattrib_sb = 0.0
    for _, r in df.iterrows():
        u = r["u"]
        sp = spot_pnl.get(u, {"b1": 0.0, "b2": 0.0, "b4": 0.0})
        act_b1 += sp["b1"]
        act_b2 += sp["b2"]
        h1, h2 = counterfactual(sp, r["r_exp_hr_b1"], r["r_exp_hr_b2"])
        s1, s2 = counterfactual(sp, r["r_exp_sb_b1"], r["r_exp_sb_b2"])
        cf_hr_b1 += h1
        cf_hr_b2 += h2
        cf_sb_b1 += s1
        cf_sb_b2 += s2
        t = sp["b1"] + sp["b2"] + sp["b4"]
        unattrib_sb += t * max(0.0, 1.0 - r["r_exp_sb_b1"] - r["r_exp_sb_b2"])

    print("\n=== Spot-line PnL only (symbol == underlying) ===")
    print(f"Actual ledger_fifo:     B1 spot PnL=${act_b1:,.0f}  B2 spot PnL=${act_b2:,.0f}")
    print(f"If split like hedge_ratio: B1=${cf_hr_b1:,.0f}  B2=${cf_hr_b2:,.0f}")
    print(f"If split like sleeve_bal: B1=${cf_sb_b1:,.0f}  B2=${cf_sb_b2:,.0f}  (unattrib=${unattrib_sb:,.0f})")
    print(f"Delta spot PnL B2 (sleeve_bal vs ledger): ${cf_sb_b2 - act_b2:,.0f}")

    # Full book PnL: ETF legs fixed; only spot would move
    etf_b1 = float(legs.loc[legs["bucket"] == "bucket_1", "total_pnl"].sum()) - act_b1
    etf_b2 = float(legs.loc[legs["bucket"] == "bucket_2", "total_pnl"].sum()) - act_b2
    # approximate - ETF pnl = total bucket - spot only is wrong because spot is subset

    b1_total = float(legs.loc[legs["bucket"] == "bucket_1", "total_pnl"].sum())
    b2_total = float(legs.loc[legs["bucket"] == "bucket_2", "total_pnl"].sum())
    spot_only_b1 = act_b1
    spot_only_b2 = act_b2
    etf_only_b1 = b1_total - spot_only_b1
    etf_only_b2 = b2_total - spot_only_b2
    print("\n=== Full bucket PnL (ETF + spot, ledger) ===")
    print(f"B1 total=${b1_total:,.0f}  (ETF ~${etf_only_b1:,.0f} + spot ${spot_only_b1:,.0f})")
    print(f"B2 total=${b2_total:,.0f}  (ETF ~${etf_only_b2:,.0f} + spot ${spot_only_b2:,.0f})")
    hypo_b1 = etf_only_b1 + cf_sb_b1
    hypo_b2 = etf_only_b2 + cf_sb_b2
    print(f"If spot used sleeve_balance shares:")
    print(f"  Hypo B1=${hypo_b1:,.0f}  (delta ${hypo_b1 - b1_total:,.0f})")
    print(f"  Hypo B2=${hypo_b2:,.0f}  (delta ${hypo_b2 - b2_total:,.0f})")

    print("\n=== 13 yieldboost: spot ratio diffs ===")
    ydf = df[df["u"].isin(YB)].copy()
    print(ydf[["u", "r_pnl_b1", "r_pnl_b2", "r_exp_hr_b1", "r_exp_hr_b2", "r_exp_sb_b1", "r_exp_sb_b2"]].to_string(index=False))

    print("\n=== Worst PnL vs sleeve_balance spot ratio (|diff| on r2) ===")
    ydf2 = ydf.sort_values("diff_pnl_vs_sb_b2", ascending=False)
    print(ydf2[["u", "pnl_spot_total", "diff_pnl_vs_sb_b2", "diff_pnl_vs_hr_b2", "diff_hr_vs_sb_b2"]].head(8).to_string(index=False))


if __name__ == "__main__":
    main()
