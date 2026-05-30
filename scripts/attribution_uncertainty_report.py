"""Attribution uncertainty report (Phase 1).

The physical underlying stock line is a single netted IBKR position, but the
strategy intends up to three economic claims on it (B1 long hedge, B2 long
hedge, B4 structural short). The published bucket numbers pick ONE attribution
method; this report quantifies the *model risk* of that choice by recomputing
the per-bucket split under every method the accounting engine already produces
and reporting the spread as an explicit error bar.

It is strictly read-only: it consumes existing ``data/runs/<date>/accounting``
outputs and writes two report CSVs (plus a console summary). It never mutates
the share ledger or any accounting output.

The honest finding is asymmetric, so the two splits are reported differently:

* B1 vs B2 long-spot split: there is effectively only ONE defensible method
  (``sleeve_balance`` -- long spot offsets the short ETF sleeves by delta-dollar
  notional). There is no independent second source to cross-check it, because
  (a) the underlying trade is netted with no sleeve tag, and (b) the FIFO share
  ledger carries ZERO B2 spot tags (proven by ``ledger_b2_spot_usd`` below). So
  the deliverable here is the *exposure at stake*: how much long-spot notional
  on shared names rests entirely on the sleeve_balance assumption, with no
  data-based way to validate the split. This is the "blast radius", not a spread.

* B4 structural short notional: here we DO have two genuinely disagreeing
  methods (``plan`` vs ``etf_implied``), so a real per-name and total spread is
  reported.

* PnL attribution: the spot PnL on shared names inherits the same single-method
  dependence; ``spot_pnl_model_determined`` flags the PnL that rides on it.

Usage::

    python scripts/attribution_uncertainty_report.py [--run-date YYYY-MM-DD]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = PROJECT_ROOT / "data" / "runs"

STOCK_BUCKETS = ("bucket_1", "bucket_2", "bucket_4")


def _latest_run_date() -> str:
    dates = sorted(
        p.name
        for p in RUNS_ROOT.iterdir()
        if p.is_dir() and (p / "accounting" / "totals.json").exists()
    )
    if not dates:
        raise SystemExit(f"No run with totals.json found under {RUNS_ROOT}")
    return dates[-1]


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _norm_pair(a: float, b: float) -> tuple[float, float]:
    """Normalize a 2-vector of magnitudes to fractions; (0,0) -> (0,0)."""
    a, b = abs(float(a)), abs(float(b))
    tot = a + b
    if tot <= 1e-12:
        return 0.0, 0.0
    return a / tot, b / tot


def build_report(run_date: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    acct = RUNS_ROOT / run_date / "accounting"
    if not acct.exists():
        raise SystemExit(f"No accounting dir for {run_date}: {acct}")

    spot = _read_csv(acct / "net_exposure_spot_by_underlying.csv")
    timing = _read_csv(acct / "bucket_timing_state.csv")
    b4recon = _read_csv(acct / "b4_plan_ledger_reconciliation.csv")
    pnlsym = _read_csv(acct / "pnl_by_symbol.csv")
    totals = json.loads((acct / "totals.json").read_text(encoding="utf-8"))

    if spot.empty:
        raise SystemExit("net_exposure_spot_by_underlying.csv missing/empty")

    timing_ix = timing.set_index("underlying") if not timing.empty else pd.DataFrame()
    b4_ix = b4recon.set_index("underlying") if not b4recon.empty else pd.DataFrame()

    # Spot PnL per underlying per bucket (the spot leg == underlying symbol).
    spot_pnl: dict[str, dict[str, float]] = {}
    if not pnlsym.empty:
        sp = pnlsym[pnlsym["symbol"] == pnlsym["underlying"]]
        for u, grp in sp.groupby("underlying"):
            spot_pnl[str(u)] = {
                str(b): float(grp.loc[grp["bucket"] == b, "total_pnl"].sum())
                for b in STOCK_BUCKETS
            }

    SHARE_THR = 0.02  # min ratio to count a sleeve as a live claim on the spot
    rows: list[dict] = []
    for _, r in spot.iterrows():
        u = str(r["underlying"])
        net = float(r.get("net_notional_usd", 0.0) or 0.0)
        if abs(net) <= 1e-9:
            continue

        # --- Published long-spot split (sleeve_balance, delta-dollar weighted) ---
        pub_r1 = float(r.get("ratio_spot_b1", 0.0) or 0.0)
        pub_r2 = float(r.get("ratio_spot_b2", 0.0) or 0.0)
        pub_b1, pub_b2 = pub_r1 * net, pub_r2 * net

        # --- Ledger coverage diagnostic (proves there is no 2nd source) ---
        led_r1 = led_r2 = 0.0
        if u in getattr(timing_ix, "index", []):
            t = timing_ix.loc[u]
            led_r1, led_r2 = _norm_pair(t.get("curr_qty_b1", 0.0), t.get("curr_qty_b2", 0.0))
        ledger_b2_spot_usd = led_r2 * net

        # A name is "shared B1/B2" when both sleeves hold a live claim on the
        # same long spot -> the entire division is model-determined.
        shared_b1b2 = (pub_r1 > SHARE_THR) and (pub_r2 > SHARE_THR)
        model_determined_usd = abs(net) if shared_b1b2 else 0.0

        # --- B4 structural short notional: plan vs etf-implied (real spread) ---
        plan_b4 = impl_b4 = 0.0
        b4_source = "none"
        if u in getattr(b4_ix, "index", []):
            b = b4_ix.loc[u]
            plan_b4 = float(b.get("plan_b4_usd", 0.0) or 0.0)
            impl_b4 = float(b.get("etf_implied_short_usd", 0.0) or 0.0)
            b4_source = str(b.get("b4_source", "none"))
        b4_cands = [v for v in (plan_b4, impl_b4) if abs(v) > 1e-9]
        b4_spread = (max(b4_cands) - min(b4_cands)) if len(b4_cands) == 2 else 0.0

        # --- PnL riding on the single-method split ---
        sp_pnl = spot_pnl.get(u, {})
        spot_pnl_total = sum(sp_pnl.values())
        spot_pnl_model_determined = spot_pnl_total if shared_b1b2 else 0.0

        rows.append(
            {
                "underlying": u,
                "spot_net_usd": round(net, 2),
                "shared_b1b2": shared_b1b2,
                # single-method published exposure split
                "b1_net_exp_usd": round(pub_b1, 2),
                "b2_net_exp_usd": round(pub_b2, 2),
                "b1_share": round(pub_r1, 4),
                "b2_share": round(pub_r2, 4),
                # exposure resting on the sleeve_balance assumption alone
                "b1b2_model_determined_usd": round(model_determined_usd, 2),
                "ledger_b2_spot_usd": round(ledger_b2_spot_usd, 2),  # ~0 everywhere
                # B4 structural short (two methods, real disagreement)
                "b4_short_plan_usd": round(plan_b4, 2),
                "b4_short_etf_implied_usd": round(impl_b4, 2),
                "b4_short_chosen": b4_source,
                "b4_short_spread_usd": round(b4_spread, 2),
                # pnl exposed to the single-method split
                "spot_pnl_total": round(spot_pnl_total, 2),
                "spot_pnl_model_determined": round(spot_pnl_model_determined, 2),
            }
        )

    by_under = pd.DataFrame(rows)
    if by_under.empty:
        raise SystemExit("No spot underlyings with exposure to report")

    by_under["rank_key"] = by_under[["b1b2_model_determined_usd", "b4_short_spread_usd"]].max(axis=1)
    by_under = by_under.sort_values("rank_key", ascending=False).reset_index(drop=True)
    by_under = by_under.drop(columns=["rank_key"])

    # --- Bucket-level rollup ---
    b1b2_at_stake = float(by_under["b1b2_model_determined_usd"].sum())
    n_shared = int(by_under["shared_b1b2"].sum())
    ledger_b2_total = float(by_under["ledger_b2_spot_usd"].sum())
    b4_plan_total = float(by_under["b4_short_plan_usd"].sum())
    b4_impl_total = float(by_under["b4_short_etf_implied_usd"].sum())
    pnl_model_determined = float(by_under["spot_pnl_model_determined"].sum())

    def _net(b: str) -> float:
        return float(totals.get(f"net_exposure_{b}", 0.0) or 0.0)

    bp = totals.get("bucket_pnl", {})
    summary_rows = [
        {
            "bucket": "bucket_1",
            "published_net_exposure_usd": round(_net("bucket_1"), 2),
            "published_pnl_usd": round(float(bp.get("bucket_1", 0.0)), 2),
            "attribution_basis": "single method (sleeve_balance); no data cross-check",
            "exposure_at_stake_usd": round(b1b2_at_stake, 2),
            "pnl_at_stake_usd": round(pnl_model_determined, 2),
            "note": "shared B1/B2 long-spot division is model-only; ledger B2 spot=$0",
        },
        {
            "bucket": "bucket_2",
            "published_net_exposure_usd": round(_net("bucket_2"), 2),
            "published_pnl_usd": round(float(bp.get("bucket_2", 0.0)), 2),
            "attribution_basis": "single method (sleeve_balance); no data cross-check",
            "exposure_at_stake_usd": round(b1b2_at_stake, 2),
            "pnl_at_stake_usd": round(pnl_model_determined, 2),
            "note": "shared B1/B2 long-spot division is model-only; ledger B2 spot=$0",
        },
        {
            "bucket": "bucket_4",
            "published_net_exposure_usd": round(_net("bucket_4"), 2),
            "published_pnl_usd": round(float(bp.get("bucket_4", 0.0)), 2),
            "attribution_basis": "two methods (plan vs etf_implied) -- real spread",
            "exposure_at_stake_usd": round(abs(b4_plan_total - b4_impl_total), 2),
            "pnl_at_stake_usd": float("nan"),
            "note": f"plan total {b4_plan_total:,.0f} vs etf_implied {b4_impl_total:,.0f}",
        },
    ]
    summary = pd.DataFrame(summary_rows)
    meta = {
        "run_date": run_date,
        "n_shared_b1b2_names": n_shared,
        "b1b2_exposure_at_stake_usd": round(b1b2_at_stake, 2),
        "b1b2_pnl_at_stake_usd": round(pnl_model_determined, 2),
        "ledger_b2_spot_total_usd": round(ledger_b2_total, 2),
        "b4_short_plan_total_usd": round(b4_plan_total, 2),
        "b4_short_etf_implied_total_usd": round(b4_impl_total, 2),
        "b4_short_total_spread_usd": round(abs(b4_plan_total - b4_impl_total), 2),
        "n_underlyings": int(len(by_under)),
    }
    return by_under, summary, meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-date", default=None, help="YYYY-MM-DD (default: latest run)")
    args = ap.parse_args()
    run_date = args.run_date or _latest_run_date()

    by_under, summary, meta = build_report(run_date)
    acct = RUNS_ROOT / run_date / "accounting"
    by_under_path = acct / "attribution_uncertainty_by_underlying.csv"
    summary_path = acct / "attribution_uncertainty_bucket_summary.csv"
    by_under.to_csv(by_under_path, index=False)
    summary.to_csv(summary_path, index=False)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 40)
    print(f"\n=== Attribution uncertainty report - {run_date} ===\n")
    print("Bucket-level summary:")
    print(summary.to_string(index=False))
    print("\nTop 12 shared B1/B2 names (whole long-spot division is model-only):")
    sh = by_under[by_under["shared_b1b2"]]
    cols_b12 = [
        "underlying", "spot_net_usd", "b1_net_exp_usd", "b2_net_exp_usd",
        "b1_share", "b2_share", "ledger_b2_spot_usd", "spot_pnl_total",
    ]
    print(sh[cols_b12].head(12).to_string(index=False))
    print("\nTop 12 B4 names by plan-vs-implied short disagreement:")
    b4 = by_under[by_under["b4_short_spread_usd"] > 0].sort_values(
        "b4_short_spread_usd", ascending=False
    )
    cols_b4 = [
        "underlying", "b4_short_plan_usd", "b4_short_etf_implied_usd",
        "b4_short_chosen", "b4_short_spread_usd",
    ]
    print(b4[cols_b4].head(12).to_string(index=False))
    print("\nHeadline:")
    for k, v in meta.items():
        print(f"  {k}: {v}")
    print(f"\nWrote:\n  {by_under_path}\n  {summary_path}")


if __name__ == "__main__":
    main()
