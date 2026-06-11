"""Bucket 4 parameter scorecard: standardized theta-grid comparison (Phase 1, WS6).

Replays the real pair backtest (run_bucket4_backtest_dynamic_h + TR/VCR cadence)
over a small grid around the current production knobs and scores every
theta = (k_tr, m_vcr, base_days) on the SAME fixed metrics, so a human can decide
whether to nudge a knob -- the "SGD-style" loop:

  metrics per theta (equal-weight across pairs):
    mean/median CAGR, winsorized mean CAGR, mean vol, mean max_dd,
    mean rebalances per pair, mean interval, pct of pairs whose mean interval
    hit the cap, composite rank (higher CAGR + lower vol/dd is better)

Current YAML knobs are always included and marked ``is_current=True``.

Outputs:
  notebooks/output/b4_param_scorecard.csv      (one row per theta, ranked)
  data/b4_param_scorecard_history.jsonl        (best + current theta per run, for trend)

Usage:
  python -m scripts.bucket4_param_scorecard --quick          # 1-step neighborhood (7 thetas)
  python -m scripts.bucket4_param_scorecard                  # full 3x3x3 grid
  python -m scripts.bucket4_param_scorecard --start 2026-03-01 --max-pairs 10
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.bucket4_base_days_frequency_sweep import (  # noqa: E402
    build_prices,
    norm_sym,
    pair_metrics,
)
from scripts.bucket4_phase345_backtest import load_metrics_filtered  # noqa: E402
from scripts.bucket4_dynamic_bt import run_bucket4_backtest_dynamic_h  # noqa: E402
from scripts.bucket4_vol_shape_signals import (  # noqa: E402
    get_pair_signal,
    load_vol_shape_history,
    policy_continuous_interval,
)

TRADING_DAYS = 252
SLIPPAGE_BPS = 20.0
FEE_BPS = 0.0
SIGNAL_WINDOW = 45
CAP_DAYS = 21


def current_knobs() -> dict:
    try:
        from strategy_config import load_config
        cfg = load_config()
        hcp = (
            cfg.get("portfolio", {}).get("sleeves", {})
            .get("inverse_decay_bucket4", {}).get("rules", {})
            .get("bucket4_weekly_opt2", {}).get("hedge_cadence_policy", {})
        )
        return {
            "k_tr": float(hcp.get("k_tr", 2.25)),
            "m_vcr": float(hcp.get("m_vcr", 2.5)),
            "base_days": float(hcp.get("base_days", 10.0)),
            "max_interval": int(hcp.get("max_interval", CAP_DAYS)),
        }
    except Exception:
        return {"k_tr": 2.25, "m_vcr": 2.5, "base_days": 10.0, "max_interval": CAP_DAYS}


def latest_pairs_csv(runs_root: Path) -> Path | None:
    cands = sorted(runs_root.glob("*/accounting/bucket4_pairs.csv"))
    return cands[-1] if cands else None


def theta_grid(cur: dict, *, quick: bool, eps_ktr: float, eps_mvcr: float, eps_bd: float) -> list[dict]:
    k0, m0, b0 = cur["k_tr"], cur["m_vcr"], cur["base_days"]
    if quick:
        thetas = [
            (k0, m0, b0),
            (k0 - eps_ktr, m0, b0), (k0 + eps_ktr, m0, b0),
            (k0, m0 - eps_mvcr, b0), (k0, m0 + eps_mvcr, b0),
            (k0, m0, b0 - eps_bd), (k0, m0, b0 + eps_bd),
        ]
    else:
        thetas = list(itertools.product(
            [k0 - eps_ktr, k0, k0 + eps_ktr],
            [m0 - eps_mvcr, m0, m0 + eps_mvcr],
            [b0 - eps_bd, b0, b0 + eps_bd],
        ))
    out = []
    for k, m, b in thetas:
        out.append({"k_tr": round(float(k), 4), "m_vcr": round(float(m), 4),
                    "base_days": round(float(max(b, 1.0)), 4)})
    # de-dupe preserving order
    seen, uniq = set(), []
    for t in out:
        key = (t["k_tr"], t["m_vcr"], t["base_days"])
        if key not in seen:
            seen.add(key)
            uniq.append(t)
    return uniq


def run_theta(
    theta: dict,
    pair_data: list[dict],
    *,
    max_interval: int,
) -> dict | None:
    rows = []
    for pd_ in pair_data:
        prices, sig = pd_["prices"], pd_["sig"]
        cal = prices.index
        rd, _ = policy_continuous_interval(
            cal, sig,
            base_days=float(theta["base_days"]),
            k_tr=float(theta["k_tr"]),
            m_vcr=float(theta["m_vcr"]),
            min_interval=1,
            max_interval=int(max_interval),
        )
        rd = pd.DatetimeIndex(rd).intersection(cal)
        if len(rd) == 0:
            rd = pd.DatetimeIndex([cal[0]])
        if str(pd_.get("hedge_model", "v7")) == "v7":
            # mirror production hedge_ratio_model: v7 (adopted in the Phase 3-5 lab)
            from scripts.bucket4_phase345_backtest import build_h_series
            h = build_h_series(pd_, h_model="v7", beta_mode="static", hyst=0.0, regime_bump=0.0)
        else:
            h = pd.Series(float(pd_["partial_h"]), index=cal)
        try:
            bt = run_bucket4_backtest_dynamic_h(
                prices, h, rd,
                beta_a=float(pd_["beta_a"]),
                beta_b=1.0,
                borrow_a_annual=float(pd_["borrow_a"]),
                fee_bps=FEE_BPS,
                slippage_bps=SLIPPAGE_BPS,
                opt2_h_base=float(pd_["partial_h"]),
            )
        except Exception:
            continue
        if bt is None or bt.empty:
            continue
        m = pair_metrics(bt)
        m["pair"] = pd_["pair"]
        rows.append(m)
    if not rows:
        return None
    df = pd.DataFrame(rows)
    cagr = df["CAGR"].dropna()
    lo, hi = cagr.quantile(0.05), cagr.quantile(0.95)
    at_cap = (df["mean_interval_days"] >= (int(theta.get("max_interval", CAP_DAYS)) - 0.5)).mean()
    return {
        **theta,
        "n_pairs": int(len(df)),
        "ew_mean_cagr": float(cagr.mean()),
        "ew_median_cagr": float(cagr.median()),
        "winsor_mean_cagr": float(cagr.clip(lo, hi).mean()),
        "ew_mean_vol": float(df["vol"].mean(skipna=True)),
        "ew_mean_max_dd": float(df["max_dd"].mean(skipna=True)),
        "mean_rebalances": float(df["n_rebalances"].mean(skipna=True)),
        "mean_interval_days": float(df["mean_interval_days"].mean(skipna=True)),
        "pct_pairs_at_cap": float(at_cap) if np.isfinite(at_cap) else np.nan,
        "total_borrow_paid": float(df["borrow_paid"].sum(skipna=True)),
        "total_slippage_paid": float(df["slippage_paid"].sum(skipna=True)),
    }


def add_composite_rank(score: pd.DataFrame) -> pd.DataFrame:
    s = score.copy()
    for col, asc in [
        ("winsor_mean_cagr", False),
        ("ew_median_cagr", False),
        ("ew_mean_vol", True),
        ("ew_mean_max_dd", False),    # max_dd is negative; less negative (shallower) ranks better
    ]:
        s[f"rank_{col}"] = s[col].rank(ascending=asc)
    rank_cols = [c for c in s.columns if c.startswith("rank_")]
    s["composite_rank"] = s[rank_cols].mean(axis=1)
    return s.sort_values("composite_rank")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Bucket 4 cadence-parameter scorecard.")
    ap.add_argument("--pairs", type=Path, default=None,
                    help="bucket4_pairs.csv (default: latest under data/runs)")
    ap.add_argument("--metrics", type=Path,
                    default=REPO.parent / "Levered ETFs/etf-dashboard/data/etf_metrics_daily.csv")
    ap.add_argument("--vol-shape", type=Path,
                    default=REPO.parent / "Levered ETFs/etf-dashboard/data/vol_shape_history.json")
    ap.add_argument("--screened", type=Path, default=REPO / "data/etf_screened_today.csv")
    ap.add_argument("--outdir", type=Path, default=REPO / "notebooks/output")
    ap.add_argument("--start", default="2025-10-07", help="backtest window start")
    ap.add_argument("--max-pairs", type=int, default=0, help="0 = all")
    ap.add_argument("--quick", action="store_true", help="1-step neighborhood instead of full grid")
    ap.add_argument("--eps-ktr", type=float, default=0.5)
    ap.add_argument("--eps-mvcr", type=float, default=0.5)
    ap.add_argument("--eps-bd", type=float, default=2.0)
    ap.add_argument("--hedge-model", choices=["v7", "fixed"], default="v7",
                    help="h used while scoring cadence thetas (default mirrors production v7)")
    ap.add_argument("--history-jsonl", type=Path, default=REPO / "data/b4_param_scorecard_history.jsonl")
    args = ap.parse_args(argv)

    pairs_csv = args.pairs or latest_pairs_csv(REPO / "data/runs")
    if pairs_csv is None or not pairs_csv.is_file():
        print("[b4-scorecard] no bucket4_pairs.csv found")
        return 1

    cur = current_knobs()
    thetas = theta_grid(cur, quick=args.quick, eps_ktr=args.eps_ktr,
                        eps_mvcr=args.eps_mvcr, eps_bd=args.eps_bd)
    print(f"[b4-scorecard] pairs={pairs_csv}")
    print(f"[b4-scorecard] current knobs: {cur}")
    print(f"[b4-scorecard] thetas to score: {len(thetas)} (quick={args.quick})")

    pairs = pd.read_csv(pairs_csv)
    pairs["etf"] = pairs["etf"].map(norm_sym)
    pairs["underlying"] = pairs["underlying"].map(norm_sym)
    metrics = load_metrics_filtered(args.metrics, set(pairs["etf"]))
    vs_hist = load_vol_shape_history(args.vol_shape)
    screened = pd.read_csv(args.screened)
    screened["ETF"] = screened["ETF"].map(norm_sym)
    borrow_map = {
        r["ETF"]: float(r.get("borrow_current") or np.nan)
        for _, r in screened.iterrows()
        if pd.notna(r.get("borrow_current", np.nan))
    }

    start = pd.Timestamp(args.start)
    keys = list(zip(pairs["etf"], pairs["underlying"], pairs["delta"], pairs["partial_hedge_ratio"]))
    if args.max_pairs > 0:
        keys = keys[: args.max_pairs]

    # Load prices + signals ONCE per pair (shared across thetas)
    pair_data: list[dict] = []
    for etf, und, delta, ph in keys:
        prices = build_prices(metrics, etf, start)
        if prices is None:
            continue
        sig = get_pair_signal(
            etf, und, prices.index, history=vs_hist,
            underlying_prices=prices["b_px"],
            window=SIGNAL_WINDOW, lookahead_shift=1,
            prefer_underlying_recompute=True, norm_sym=norm_sym,
        )
        pair_data.append({
            "pair": f"{etf}/{und}",
            "prices": prices,
            "sig": sig,
            "beta_a": float(delta),
            "borrow_a": float(borrow_map.get(etf, 0.0)),
            "partial_h": float(ph) if pd.notna(ph) else 0.75,
            "hedge_model": args.hedge_model,
        })
    print(f"[b4-scorecard] pairs with data: {len(pair_data)}")
    if not pair_data:
        return 1

    rows = []
    for i, theta in enumerate(thetas, 1):
        res = run_theta(theta, pair_data, max_interval=cur["max_interval"])
        if res is not None:
            res["is_current"] = (
                theta["k_tr"] == cur["k_tr"]
                and theta["m_vcr"] == cur["m_vcr"]
                and theta["base_days"] == cur["base_days"]
            )
            rows.append(res)
        print(f"  ... theta {i}/{len(thetas)} done", flush=True)

    if not rows:
        print("[b4-scorecard] no thetas completed")
        return 1

    score = add_composite_rank(pd.DataFrame(rows))
    args.outdir.mkdir(parents=True, exist_ok=True)
    out_csv = args.outdir / "b4_param_scorecard.csv"
    score.to_csv(out_csv, index=False)

    show_cols = ["k_tr", "m_vcr", "base_days", "is_current", "ew_mean_cagr", "ew_median_cagr",
                 "winsor_mean_cagr", "ew_mean_vol", "ew_mean_max_dd", "mean_interval_days",
                 "pct_pairs_at_cap", "composite_rank"]
    disp = score[show_cols].copy()
    for c in ("ew_mean_cagr", "ew_median_cagr", "winsor_mean_cagr", "ew_mean_vol",
              "ew_mean_max_dd", "pct_pairs_at_cap"):
        disp[c] = (disp[c] * 100).round(2)
    print("\n=== Scorecard (best composite first; CAGR/vol/dd in %) ===")
    with pd.option_context("display.width", 220):
        print(disp.to_string(index=False))
    print(f"\n[b4-scorecard] wrote {out_csv}")

    best = score.iloc[0]
    cur_row = score[score["is_current"]]
    rec = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "window_start": str(args.start),
        "n_pairs": int(score["n_pairs"].max()),
        "current": cur,
        "current_score": (
            {k: float(cur_row.iloc[0][k]) for k in
             ("winsor_mean_cagr", "ew_median_cagr", "ew_mean_vol", "composite_rank")}
            if not cur_row.empty else None
        ),
        "best": {k: float(best[k]) for k in ("k_tr", "m_vcr", "base_days")},
        "best_score": {k: float(best[k]) for k in
                       ("winsor_mean_cagr", "ew_median_cagr", "ew_mean_vol", "composite_rank")},
    }
    args.history_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with open(args.history_jsonl, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec) + "\n")
    print(f"[b4-scorecard] appended history -> {args.history_jsonl}")

    if not cur_row.empty and float(best["composite_rank"]) < float(cur_row.iloc[0]["composite_rank"]):
        print(
            f"\n[b4-scorecard] SUGGESTION: best theta k_tr={best['k_tr']} m_vcr={best['m_vcr']} "
            f"base_days={best['base_days']} outranks current "
            f"({best['composite_rank']:.2f} vs {float(cur_row.iloc[0]['composite_rank']):.2f}). "
            f"Nudge ONE knob at most, shadow it for a week before promoting to YAML."
        )
    else:
        print("\n[b4-scorecard] current knobs are at/near the local optimum -- HOLD.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
