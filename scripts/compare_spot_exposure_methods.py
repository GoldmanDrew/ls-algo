"""Compare B1/B2/B4 ratio-split net exposure under spot allocation alternatives."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ibkr_accounting import (  # noqa: E402
    SpotBucketRatios,
    _b12_hedge_spot_usd_for_etf,
    apply_b4_structural_short_to_exposure_detail,
    classify_etf_leg_bucket,
    held_etf_bucket_flags_from_positions,
    ledger_spot_bucket_ratios,
    _hedge_ratio_finalize_spot_ratios,
    _is_etf_leg,
)


def _load_run(run_date: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    base = ROOT / "data" / "runs" / run_date / "accounting"
    detail = pd.read_csv(base / "bucket_exposure_detail.csv")
    timing = pd.read_csv(base / "timing" / "bucket_timing_state.csv") if (base / "timing" / "bucket_timing_state.csv").exists() else pd.read_csv(base / "bucket_timing_state.csv")
    totals = json.loads((base / "totals.json").read_text(encoding="utf-8"))
    return detail, timing, totals


def _structural_usd_from_detail(detail: pd.DataFrame) -> dict[str, float]:
    out: dict[str, float] = {}
    spot = detail[
        (detail["symbol"].astype(str) == detail["underlying"].astype(str))
        & (detail.get("leg_class", pd.Series(dtype=str)).astype(str) == "spot_b4_structural")
    ]
    for _, row in spot.iterrows():
        u = str(row["underlying"])
        net = float(row["net_notional_usd"])
        r4 = float(row["_ratio_b4"])
        if abs(r4) > 1e-12:
            out[u] = net * r4
    return out


def _per_underlying_leg_stats(
    detail: pd.DataFrame,
    etf_to_under: dict[str, str],
    flow_short: set[str],
    b4_syms: set[str],
) -> dict[str, dict]:
    d = detail.copy()
    d["_is_etf"] = d.apply(
        lambda r: _is_etf_leg(str(r["symbol"]), str(r["underlying"]), etf_to_under),
        axis=1,
    )
    stats: dict[str, dict] = {}
    for u, grp in d.groupby("underlying"):
        u = str(u)
        spot_net = etf_b1 = etf_b2 = etf_b4 = 0.0
        need_b1 = need_b2 = 0.0
        for _, row in grp.iterrows():
            sym = str(row["symbol"])
            net = float(row["net_notional_usd"])
            if not bool(row["_is_etf"]) and sym == u:
                spot_net += net
                continue
            if not bool(row["_is_etf"]):
                continue
            beta = float(row.get("_beta", 1.0))
            bkt, _ = classify_etf_leg_bucket(sym, beta, flow_short_set=flow_short, b4_etf_syms=b4_syms)
            if bkt == "bucket_1":
                etf_b1 += net
                need_b1 += _b12_hedge_spot_usd_for_etf(net, beta, bucket=bkt, delta_floor=0.25)
            elif bkt == "bucket_2":
                etf_b2 += net
                need_b2 += _b12_hedge_spot_usd_for_etf(net, beta, bucket=bkt, delta_floor=0.25)
            elif bkt == "bucket_4":
                etf_b4 += net
        stats[u] = {
            "spot_net": spot_net,
            "etf_b1": etf_b1,
            "etf_b2": etf_b2,
            "etf_b4": etf_b4,
            "need_b1": need_b1,
            "need_b2": need_b2,
            "has_b1": abs(etf_b1) > 1e-12,
            "has_b2": abs(etf_b2) > 1e-12,
            "has_b4": abs(etf_b4) > 1e-12,
        }
    return stats


def spot_ratios_option_a(st: dict, *, b4_b12_only: bool) -> SpotBucketRatios:
    spot = float(st["spot_net"])
    if abs(spot) <= 1e-12:
        return SpotBucketRatios(1.0, 0.0, 0.0, "option_a_flat")
    e1, e2 = float(st["etf_b1"]), float(st["etf_b2"])
    r1 = r2 = 0.0
    if e2 < -1e-12 and st["has_b2"]:
        r2 = min(1.0, max(0.0, -e2 / spot))
    if e1 < -1e-12 and st["has_b1"]:
        r1 = min(1.0, max(0.0, -e1 / spot))
    s = r1 + r2
    if s > 1.0 + 1e-9:
        r1, r2 = r1 / s, r2 / s
    r4 = 0.0
    if b4_b12_only and (r1 + r2) > 1e-12:
        r1, r2 = r1 / (r1 + r2), r2 / (r1 + r2)
    return SpotBucketRatios(r1, r2, r4, "option_a")


def spot_ratios_option_e(st: dict, *, b4_b12_only: bool) -> SpotBucketRatios:
    spot = float(st["spot_net"])
    need_b1, need_b2 = float(st["need_b1"]), float(st["need_b2"])
    if abs(spot) <= 1e-12:
        return SpotBucketRatios(1.0, 0.0, 0.0, "option_e_flat")
    total_need = need_b1 + need_b2
    if total_need <= 1e-12:
        return SpotBucketRatios(0.0, 0.0, 0.0, "option_e_unbucketed")
    if spot >= total_need - 1e-9:
        # Paired slice: proportional to needs; no orphan forced to B1
        r1 = need_b1 / spot
        r2 = need_b2 / spot
        r4 = 0.0
    else:
        # Shortage: proportional split of all spot
        r1 = need_b1 / total_need
        r2 = need_b2 / total_need
        r4 = 0.0
    if b4_b12_only and (r1 + r2) > 1e-12:
        r1, r2 = r1 / (r1 + r2), r2 / (r1 + r2)
    return SpotBucketRatios(r1, r2, r4, "option_e")


def spot_ratios_current_hr(st: dict, *, b4_b12_only: bool) -> SpotBucketRatios:
    return _hedge_ratio_finalize_spot_ratios(
        spot_net=float(st["spot_net"]),
        need_b1=float(st["need_b1"]),
        need_b2=float(st["need_b2"]),
        has_b1=st["has_b1"],
        has_b2=st["has_b2"],
        has_b4=st["has_b4"],
        b4_b12_only=b4_b12_only,
    )[0]


def spot_ratios_option_d(
    u: str,
    ibkr_qty: float,
    ledger: dict[str, float],
    *,
    b4_b12_only: bool,
    has_b1: bool,
    has_b2: bool,
) -> SpotBucketRatios:
    return ledger_spot_bucket_ratios(
        ibkr_qty,
        ledger,
        b12_only=b4_b12_only,
        has_b1_etf=has_b1,
        has_b2_etf=has_b2,
        has_b4_etf=False,
    )


def bucket_nets(
    detail: pd.DataFrame,
    spot_map: dict[str, SpotBucketRatios],
    structural: dict[str, float],
    etf_to_under: dict[str, str],
    flow_short: set[str],
    b4_syms: set[str],
    *,
    normalize_orphan_to_b1: bool,
) -> dict[str, float]:
    d = detail.copy()
    d["_is_etf"] = d.apply(
        lambda r: _is_etf_leg(str(r["symbol"]), str(r["underlying"]), etf_to_under),
        axis=1,
    )
    ratios: list[tuple[float, float, float]] = []
    for _, row in d.iterrows():
        sym = str(row["symbol"])
        u = str(row["underlying"])
        net = float(row["net_notional_usd"])
        if bool(row["_is_etf"]):
            beta = float(row.get("_beta", 1.0))
            bkt, _ = classify_etf_leg_bucket(sym, beta, flow_short_set=flow_short, b4_etf_syms=b4_syms)
            if bkt == "bucket_1":
                ratios.append((1.0, 0.0, 0.0))
            elif bkt == "bucket_2":
                ratios.append((0.0, 1.0, 0.0))
            elif bkt == "bucket_4":
                ratios.append((0.0, 0.0, 1.0))
            else:
                ratios.append((0.0, 0.0, 0.0))
        elif sym == u:
            sr = spot_map.get(u, SpotBucketRatios(1.0, 0.0, 0.0, "default"))
            ratios.append((sr.b1, sr.b2, sr.b4))
        else:
            ratios.append((0.0, 0.0, 0.0))
    d["_ratio_b1"] = [r[0] for r in ratios]
    d["_ratio_b2"] = [r[1] for r in ratios]
    d["_ratio_b4"] = [r[2] for r in ratios]
    if structural:
        d = apply_b4_structural_short_to_exposure_detail(d, structural, hedge_spot_ratio_map=spot_map)
    if normalize_orphan_to_b1:
        for idx, row in d.iterrows():
            sym = str(row["symbol"])
            if sym != str(row["underlying"]) or bool(row["_is_etf"]):
                continue
            r1, r2, r4 = float(row["_ratio_b1"]), float(row["_ratio_b2"]), float(row["_ratio_b4"])
            if r4 < -1e-12:
                continue
            s = r1 + r2 + r4
            if 1e-12 < s < 1.0 - 1e-9:
                d.at[idx, "_ratio_b1"] = r1 + (1.0 - s)
    b1 = (d["net_notional_usd"] * d["_ratio_b1"]).sum()
    b2 = (d["net_notional_usd"] * d["_ratio_b2"]).sum()
    b4 = (d["net_notional_usd"] * d["_ratio_b4"]).sum()
    book_b124 = d["net_notional_usd"].sum()
    unattributed = 0.0
    for _, row in d.iterrows():
        sym = str(row["symbol"])
        if sym != str(row["underlying"]) or bool(row["_is_etf"]):
            continue
        net = float(row["net_notional_usd"])
        r1, r2, r4 = float(row["_ratio_b1"]), float(row["_ratio_b2"]), float(row["_ratio_b4"])
        unattributed += net * max(0.0, 1.0 - r1 - r2 - r4)
    return {
        "b1": float(b1),
        "b2": float(b2),
        "b4": float(b4),
        "book_b124": float(book_b124),
        "unattributed_spot": float(unattributed),
        "b1_b2_b4": float(b1 + b2 + b4),
    }


def main() -> None:
    run_date = "2026-05-25"
    detail, timing, totals = _load_run(run_date)
    structural = _structural_usd_from_detail(detail)

    # Minimal maps from detail (beta column not in csv — infer bucket from leg_class)
    etf_to_under: dict[str, str] = {}
    for _, row in detail.iterrows():
        etf_to_under[str(row["symbol"])] = str(row["underlying"])

    flow_short: set[str] = set(totals.get("bucket3_flow_program_symbols") or [])
    b4_plan = set(totals.get("b4_plan_exposure_underlyings") or [])
    b4_implied = set(totals.get("b4_etf_implied_underlyings") or [])
    b4_b12_only = b4_plan | b4_implied

    b4_syms = set(
        detail.loc[detail["leg_class"].astype(str).str.contains("inverse", case=False, na=False), "symbol"].astype(str)
    )
    # Rebuild stats with beta from leg_class proxy
    d2 = detail.copy()
    leg_to_beta = {
        "core_levered_etf": 2.0,
        "yieldboost_etf": 1.0,
        "flow_low_delta": 1.0,
        "inverse_b4_etf": -1.0,
        "flow_inverse": -1.0,
    }
    d2["_beta"] = d2["leg_class"].map(leg_to_beta).fillna(1.0)
    d2["_is_etf"] = d2["symbol"].astype(str) != d2["underlying"].astype(str)
    stats = _per_underlying_leg_stats(d2, etf_to_under, flow_short, b4_syms)  # d2 has _beta from leg_class

    timing_u = timing.set_index("underlying") if "underlying" in timing.columns else timing.set_index(timing.columns[1])

    spot_maps: dict[str, dict[str, SpotBucketRatios]] = {
        "current_hedge_ratio": {},
        "option_a_sleeve_balance": {},
        "option_e_proportional": {},
        "option_d_ledger_fifo": {},
    }
    for u, st in stats.items():
        b12 = u in b4_b12_only
        spot_maps["current_hedge_ratio"][u] = spot_ratios_current_hr(st, b4_b12_only=b12)
        spot_maps["option_a_sleeve_balance"][u] = spot_ratios_option_a(st, b4_b12_only=b12)
        spot_maps["option_e_proportional"][u] = spot_ratios_option_e(st, b4_b12_only=b12)
        if u in timing_u.index:
            tr = timing_u.loc[u]
            ledger = {
                "bucket_1": float(tr.get("curr_qty_b1", 0) or 0),
                "bucket_2": float(tr.get("curr_qty_b2", 0) or 0),
                "bucket_4": float(tr.get("curr_qty_b4", 0) or 0),
            }
            ibkr = float(tr.get("ibkr_qty", 0) or 0)
        else:
            ledger = {"bucket_1": 0.0, "bucket_2": 0.0, "bucket_4": 0.0}
            ibkr = 0.0
        spot_maps["option_d_ledger_fifo"][u] = spot_ratios_option_d(
            u, ibkr, ledger, b4_b12_only=b12, has_b1=st["has_b1"], has_b2=st["has_b2"]
        )

    results = {}
    for name, smap in spot_maps.items():
        norm = name == "current_hedge_ratio"
        results[name] = bucket_nets(
            d2,
            smap,
            structural,
            etf_to_under,
            flow_short,
            b4_syms,
            normalize_orphan_to_b1=norm,
        )

    # Published totals
    results["published_2026_05_25"] = {
        "b1": totals["net_exposure_bucket_1"],
        "b2": totals["net_exposure_bucket_2"],
        "b4": totals["net_exposure_bucket_4"],
        "book_b124": totals["net_exposure_total"],
    }

    print(json.dumps(results, indent=2))

    # Yieldboost subset (13 names)
    yb = {"AMD", "BABA", "COIN", "HIMS", "HOOD", "IBIT", "IONQ", "MARA", "META", "MSTR", "QBTS", "RIOT", "SMCI"}
    print("\n--- Yieldboost underlyings (spot+etf, B1/B2 only) ---")
    for name in ["current_hedge_ratio", "option_a_sleeve_balance", "option_e_proportional", "option_d_ledger_fifo"]:
        smap = spot_maps[name]
        b1 = b2 = 0.0
        for u in yb:
            if u not in stats:
                continue
            sub = d2[d2["underlying"] == u]
            nets = bucket_nets(
                sub,
                {u: smap[u]},
                {k: v for k, v in structural.items() if k == u},
                etf_to_under,
                flow_short,
                b4_syms,
                normalize_orphan_to_b1=(name == "current_hedge_ratio"),
            )
            b1 += nets["b1"]
            b2 += nets["b2"]
        print(f"{name}: B1={b1:,.0f} B2={b2:,.0f} sum={b1+b2:,.0f}")

    # Names where methods differ most on spot ratios
    print("\n--- Largest dB1 vs current (option A) ---")
    diffs = []
    for u, st in stats.items():
        if abs(st["spot_net"]) < 1000:
            continue
        cur = spot_maps["current_hedge_ratio"][u]
        a = spot_maps["option_a_sleeve_balance"][u]
        diffs.append((u, (a.b1 - cur.b1) * st["spot_net"], st["spot_net"], cur.b1, cur.b2, a.b1, a.b2))
    diffs.sort(key=lambda x: abs(x[1]), reverse=True)
    for row in diffs[:12]:
        print(
            f"{row[0]:6s} spot={row[2]:>10,.0f}  "
            f"cur r1={row[3]:.3f} r2={row[4]:.3f}  A r1={row[5]:.3f} r2={row[6]:.3f}  "
            f"dB1_usd={row[1]:>10,.0f}"
        )

    print("\n--- Canonical names (IONQ / SMCI / QBTS / MSTR) ---")
    for u in ["IONQ", "SMCI", "QBTS", "MSTR"]:
        if u not in stats:
            continue
        st = stats[u]
        b12 = u in b4_b12_only
        for label, smap in [
            ("current", spot_maps["current_hedge_ratio"]),
            ("A", spot_maps["option_a_sleeve_balance"]),
            ("E", spot_maps["option_e_proportional"]),
            ("D", spot_maps["option_d_ledger_fifo"]),
        ]:
            nets = bucket_nets(
                d2[d2["underlying"] == u],
                {u: smap[u]},
                {k: v for k, v in structural.items() if k == u},
                etf_to_under,
                flow_short,
                b4_syms,
                normalize_orphan_to_b1=(label == "current"),
            )
            sr = smap[u]
            print(
                f"{u:5s} {label:7s}  B1={nets['b1']:>10,.0f}  B2={nets['b2']:>10,.0f}  "
                f"B4={nets['b4']:>10,.0f}  book={nets['book_b124']:>8,.0f}  "
                f"unattrib={nets['unattributed_spot']:>8,.0f}  r1={sr.b1:.3f} r2={sr.b2:.3f}"
            )


if __name__ == "__main__":
    main()
