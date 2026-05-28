"""Per-underlying Option A exposure for the 13 yieldboost names."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.compare_spot_exposure_methods import (  # noqa: E402
    _load_run,
    _per_underlying_leg_stats,
    _structural_usd_from_detail,
    bucket_nets,
    spot_ratios_current_hr,
    spot_ratios_option_a,
)

YB = [
    "AMD",
    "BABA",
    "COIN",
    "HIMS",
    "HOOD",
    "IBIT",
    "IONQ",
    "MARA",
    "META",
    "MSTR",
    "QBTS",
    "RIOT",
    "SMCI",
]


def main() -> None:
    run_date = "2026-05-25"
    detail, _timing, totals = _load_run(run_date)
    structural = _structural_usd_from_detail(detail)
    d2 = detail.copy()
    leg = {
        "core_levered_etf": 2.0,
        "yieldboost_etf": 1.0,
        "flow_low_delta": 1.0,
        "inverse_b4_etf": -1.0,
    }
    d2["_beta"] = d2["leg_class"].map(leg).fillna(1.0)
    etf_to_under = {str(r["symbol"]): str(r["underlying"]) for _, r in d2.iterrows()}
    flow = set(totals.get("bucket3_flow_program_symbols") or [])
    b4_b12 = set(totals.get("b4_plan_exposure_underlyings") or []) | set(
        totals.get("b4_etf_implied_underlyings") or []
    )
    stats = _per_underlying_leg_stats(d2, etf_to_under, flow, set())

    print(f"Option A sleeve-balance spot split — {run_date}\n")
    hdr = (
        f"{'U':6s} {'Book':>10s} {'B1_A':>10s} {'B2_A':>10s} {'B4_A':>10s} "
        f"{'Unattrib':>10s} {'B1_cur':>10s} {'B2_cur':>10s}  "
        f"{'r1':>6s} {'r2':>6s} {'orph%':>6s}"
    )
    print(hdr)
    print("-" * len(hdr))

    tot_book = tot_b1a = tot_b2a = tot_b4a = tot_un = tot_b1c = tot_b2c = 0.0

    for u in YB:
        st = stats.get(u)
        if not st:
            print(f"{u:6s}  (no positions)")
            continue
        b12 = u in b4_b12
        sr_a = spot_ratios_option_a(st, b4_b12_only=b12)
        sr_c = spot_ratios_current_hr(st, b4_b12_only=b12)
        sub = d2[d2["underlying"] == u]
        na = bucket_nets(
            sub,
            {u: sr_a},
            {k: v for k, v in structural.items() if k == u},
            etf_to_under,
            flow,
            set(),
            normalize_orphan_to_b1=False,
        )
        nc = bucket_nets(
            sub,
            {u: sr_c},
            {k: v for k, v in structural.items() if k == u},
            etf_to_under,
            flow,
            set(),
            normalize_orphan_to_b1=True,
        )
        book = (
            st["spot_net"] + st["etf_b1"] + st["etf_b2"] + st["etf_b4"]
        )
        orphan_pct = max(0.0, 1.0 - sr_a.b1 - sr_a.b2 - sr_a.b4) * 100.0
        print(
            f"{u:6s} {book:10,.0f} {na['b1']:10,.0f} {na['b2']:10,.0f} {na['b4']:10,.0f} "
            f"{na['unattributed_spot']:10,.0f} {nc['b1']:10,.0f} {nc['b2']:10,.0f}  "
            f"{sr_a.b1:6.3f} {sr_a.b2:6.3f} {orphan_pct:5.1f}%"
        )
        tot_book += book
        tot_b1a += na["b1"]
        tot_b2a += na["b2"]
        tot_b4a += na["b4"]
        tot_un += na["unattributed_spot"]
        tot_b1c += nc["b1"]
        tot_b2c += nc["b2"]

        print(f"       spot={st['spot_net']:,.0f}  need_b1={st['need_b1']:,.0f}  need_b2={st['need_b2']:,.0f}")
        yb = sub[sub["leg_class"] == "yieldboost_etf"][["symbol", "net_notional_usd"]]
        b1 = sub[sub["leg_class"] == "core_levered_etf"][["symbol", "net_notional_usd"]]
        b4e = sub[sub["leg_class"].astype(str).str.contains("inverse", case=False, na=False)][
            ["symbol", "net_notional_usd"]
        ]
        if not yb.empty:
            print("       B2 ETF:", ", ".join(f"{r.symbol} {r.net_notional_usd:,.0f}" for r in yb.itertuples()))
        if not b1.empty:
            print("       B1 ETF:", ", ".join(f"{r.symbol} {r.net_notional_usd:,.0f}" for r in b1.itertuples()))
        if not b4e.empty:
            print("       B4 ETF:", ", ".join(f"{r.symbol} {r.net_notional_usd:,.0f}" for r in b4e.itertuples()))
        if structural.get(u):
            print(f"       B4 structural spot carve: {structural[u]:,.0f}")
        print()

    print("TOTAL (13 names)")
    print(
        f"       book={tot_book:,.0f}  B1_A={tot_b1a:,.0f}  B2_A={tot_b2a:,.0f}  "
        f"B4_A={tot_b4a:,.0f}  unattrib={tot_un:,.0f}"
    )
    print(f"       current: B1={tot_b1c:,.0f}  B2={tot_b2c:,.0f}  B1+B2={tot_b1c+tot_b2c:,.0f}")


if __name__ == "__main__":
    main()
