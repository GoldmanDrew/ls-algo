"""Show per-ticker B4 rebalance cadence under k_tr = +2.25, 0, -1."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.bucket4_hedge_cadence import build_rebal_dates, compute_pair_policy  # noqa: E402
from scripts.bucket4_vol_shape_signals import get_pair_signal  # noqa: E402
from scripts.sizing_tilt_cadence_bt import (  # noqa: E402
    knobs_from_yaml,
    load_price_panel,
    load_universe,
    make_knobs,
)


def fmt_date(d) -> str:
    return pd.Timestamp(d).strftime("%Y-%m-%d")


def summarize_ticker(etf, und, px, blk, start, warmup=60, n_examples=8):
    cal = pd.DatetimeIndex([d for d in px.index if d >= pd.Timestamp(start)])
    if len(cal) < warmup + 30:
        return None
    sig = get_pair_signal(etf, und, cal, history={}, underlying_prices=px["b_px"],
                          window=60, lookahead_shift=1)

    rows = []
    for k_tr, label in [(2.25, "k+2.25"), (0.0, "k0"), (-1.0, "k-1")]:
        knobs = make_knobs(blk, cadence_signal_col="tr_est", k_tr=float(k_tr))
        rb, diag = build_rebal_dates(sig, cal, knobs=knobs, warmup_bdays=warmup)
        if diag.empty:
            continue
        iv = pd.to_numeric(diag["interval_days"], errors="coerce")
        tr = pd.to_numeric(diag["cadence_signal"], errors="coerce")
        vcr = pd.to_numeric(diag["vcr"], errors="coerce")
        rows.append({
            "k_tr": label,
            "n_rebal": len(rb),
            "mean_interval": float(iv.mean()),
            "min_interval": int(iv.min()),
            "max_interval": int(iv.max()),
            "mean_tr_est": float(tr.mean()),
            "mean_vcr": float(vcr.mean()),
            "first_rebal": fmt_date(rb[0]),
            "last_rebal": fmt_date(rb[-1]),
        })

        # Pick example rebalance situations: first, last, and spread across history
        idxs = sorted(set([0, 1, 2, len(diag) // 4, len(diag) // 2, 3 * len(diag) // 4,
                           len(diag) - 3, len(diag) - 2, len(diag) - 1]))
        idxs = [i for i in idxs if 0 <= i < len(diag)][:n_examples]
        for i in idxs:
            r = diag.iloc[i]
            t = float(r["cadence_signal"]) if pd.notna(r["cadence_signal"]) else np.nan
            v = float(r["vcr"]) if pd.notna(r["vcr"]) else np.nan
            vm = float(r["vcr_med"]) if pd.notna(r["vcr_med"]) else np.nan
            pol = compute_pair_policy(t, v, vm, knobs=knobs, cadence_signal_col="tr_est")
            rows.append({
                "k_tr": label,
                "example": fmt_date(r["date"]),
                "tr_est": round(t, 3) if np.isfinite(t) else None,
                "vcr": round(v, 3) if np.isfinite(v) else None,
                "vcr_med": round(vm, 3) if np.isfinite(vm) else None,
                "interval_days": int(r["interval_days"]),
                "denom": round(pol.denom, 3),
                "next_in_d": int(r["interval_days"]),
            })
    return rows


def main():
    run_date = "2026-06-25"
    start = "2024-01-01"
    min_days = 60  # show all pairs with any meaningful history

    uni = load_universe(run_date)
    panel = load_price_panel(run_date)
    blk = knobs_from_yaml()

    # B5 uses bucket5_carry_bt, not B4 dynamic-h — cadence by ticker is B4-only.
    df = uni[uni["sleeve"].isin(["inverse_decay_bucket4"])].copy()
    df = df[df["ETF"].isin(panel.keys())].reset_index(drop=True)

    print(f"B4 eligible universe ({run_date}): {len(df)} pairs\n")
    print(f"Production cadence: tr_est, base_days={blk.get('base_days', 12)}, "
          f"m_vcr={blk.get('m_vcr', 2.5)}, min/max={blk.get('min_interval', 1)}/{blk.get('max_interval', 21)}\n")

    all_summary = []
    all_examples = []

    for _, row in df.iterrows():
        etf, und = row["ETF"], row["Underlying"]
        px = panel[etf]
        cal_len = len(px[px.index >= pd.Timestamp(start)])
        gross = float(row.get("gross_target_usd") or 0)
        borrow = float(pd.to_numeric(row.get("borrow_current"), errors="coerce") or 0)
        print("=" * 88)
        print(f"{etf:6s} / {und:6s}  |  gross=${gross:,.0f}  borrow={borrow:.1%}  "
              f"delta={row['Delta']:.3f}  history={cal_len}d")
        if cal_len < min_days:
            print("  SKIP — insufficient history\n")
            continue

        out = summarize_ticker(etf, und, px, blk, start)
        if not out:
            print("  SKIP — no schedule\n")
            continue

        summary = pd.DataFrame([r for r in out if "n_rebal" in r])
        examples = pd.DataFrame([r for r in out if "example" in r])
        summary["etf"] = etf
        summary["und"] = und
        examples["etf"] = etf
        examples["und"] = und
        all_summary.append(summary)
        all_examples.append(examples)

        print("\n  Schedule summary:")
        print(summary[["k_tr", "n_rebal", "mean_interval", "min_interval", "max_interval",
                       "first_rebal", "last_rebal"]].to_string(index=False))

        print("\n  Example rebalance situations (signal at each rebalance -> next interval):")
        for k in ["k+2.25", "k0", "k-1"]:
            sub = examples[examples["k_tr"] == k]
            if sub.empty:
                continue
            print(f"\n    [{k}]")
            for _, e in sub.iterrows():
                tr_s = f"{e['tr_est']:.3f}" if e["tr_est"] is not None else "nan"
                vcr_s = f"{e['vcr']:.3f}" if e["vcr"] is not None else "nan"
                print(f"      {e['example']}  tr_est={tr_s}  vcr={vcr_s}  "
                      f"denom={e['denom']:.3f}  -> rebalance every {e['interval_days']}d")
        print()

    outdir = REPO / "notebooks/output/sizing_tilt_cadence_bt"
    outdir.mkdir(parents=True, exist_ok=True)
    if all_summary:
        s = pd.concat(all_summary, ignore_index=True)
        e = pd.concat(all_examples, ignore_index=True)
        s.to_csv(outdir / "b4_cadence_by_ticker_summary.csv", index=False)
        e.to_csv(outdir / "b4_cadence_by_ticker_examples.csv", index=False)
        print(f"\nSaved -> {outdir / 'b4_cadence_by_ticker_summary.csv'}")
        print(f"Saved -> {outdir / 'b4_cadence_by_ticker_examples.csv'}")


if __name__ == "__main__":
    main()
