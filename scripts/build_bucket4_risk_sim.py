"""Generate the Bucket-4 risk-simulator dataset for the risk dashboard.

Writes ``risk_dashboard/data/bucket4_risk_sim.json``: the gross-weighted B4
portfolio daily-return series under the CURRENT production cadence (read live
from ``config/strategy_config.yml``), fat-tailed distribution fits, realized
stats, and a reference Monte-Carlo drawdown tail. ``build_site.py`` merges this
into the snapshot so the dashboard's client-side simulator can resample it.

Run
---
    python scripts/build_bucket4_risk_sim.py
    python scripts/build_bucket4_risk_sim.py --run-date 2026-06-25 --start 2024-01-01
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.bucket4_dynamic_bt import run_bucket4_backtest_dynamic_h  # noqa: E402
from scripts.bucket4_hedge_cadence import build_h_series, build_rebal_dates  # noqa: E402
from scripts.bucket4_cadence_risk_opt import (  # noqa: E402
    HORIZON,
    mc_block_bootstrap,
    mc_parametric,
    perf_from_returns,
    port_returns,
    tail_stats,
)
from scripts.bucket4_risk_sim_universe import FORCE_INCLUDE_ETFS, load_risk_sim_universe  # noqa: E402
from scripts.sizing_tilt_cadence_bt import (  # noqa: E402
    knobs_from_yaml,
    load_price_panel,
    make_knobs,
)
from scripts.bucket4_vol_shape_signals import get_pair_signal  # noqa: E402

OUT = REPO / "risk_dashboard/data/bucket4_risk_sim.json"


def _finite_float(val, default: float = 0.0) -> float:
    """Coerce to float; NaN/inf become *default* (``x or 0`` fails for NaN)."""
    v = float(pd.to_numeric(val, errors="coerce"))
    return default if not np.isfinite(v) else v


def build_portfolio(uni, panel, start, min_days, *, min_days_short: int = 60):
    """Per-pair returns under the CURRENT production cadence; gross-weighted."""
    blk = knobs_from_yaml()
    knobs = make_knobs(blk)  # production knobs straight from YAML
    df = uni[uni["sleeve"].isin(["inverse_decay_bucket4", "volatility_etp_bucket5"])].copy()
    df = df[df["ETF"].isin(panel.keys())].reset_index(drop=True)

    ret_cols, weights, pairs = {}, {}, []
    for _, row in df.iterrows():
        etf, und = row["ETF"], row["Underlying"]
        px = panel[etf]
        cal = pd.DatetimeIndex([d for d in px.index if d >= pd.Timestamp(start)])
        etf_u = str(etf).upper()
        need_days = min_days_short if (etf_u in FORCE_INCLUDE_ETFS or bool(row.get("low_n_included"))) else min_days
        if len(cal) < need_days:
            continue
        sig = get_pair_signal(etf, und, cal, history={}, underlying_prices=px["b_px"],
                              window=60, lookahead_shift=1)
        h_daily = build_h_series(sig, cal, knobs=knobs)
        rb, _ = build_rebal_dates(sig, cal, knobs=knobs, warmup_bdays=60)
        borrow = _finite_float(row.get("borrow_current"), 0.0)
        beta = _finite_float(row.get("Delta"), -2.0)
        bt = run_bucket4_backtest_dynamic_h(
            px.reindex(cal), h_daily, rb, initial_capital=1.0, gross_multiplier=1.0,
            beta_a=-abs(beta), beta_b=1.0, borrow_a_annual=borrow, slippage_bps=20.0,
        )
        ret_cols[etf] = bt["ret"]
        w = _finite_float(row.get("sim_gross_usd"), 0.0)
        weights[etf] = w
        proposed = _finite_float(row.get("gross_target_usd"), 0.0)
        optimal = _finite_float(row.get("optimal_gross_target_usd"), 0.0)
        pairs.append({
            "etf": etf,
            "und": und,
            "n_days": int(len(cal)),
            "gross_usd": round(w, 2),
            "borrow": round(borrow, 4),
            "weight_source": str(row.get("weight_source", "proposed")),
            "in_book": bool(row.get("in_book", False)),
            "locate_ok": bool(row.get("locate_ok", False)),
            "blacklisted": bool(row.get("blacklisted", False)),
            "proposed_gross_usd": round(proposed, 2),
            "optimal_gross_usd": round(optimal, 2) if optimal > 0 else None,
        })
    if not ret_cols:
        return None
    ret_df = pd.DataFrame(ret_cols).reindex(
        sorted(set().union(*[set(s.index) for s in ret_cols.values()]))
    )
    gross_w = pd.Series(weights)
    pr = port_returns(ret_df, gross_w)
    wnorm = (gross_w / gross_w.sum()).round(4)
    for p in pairs:
        p["weight"] = float(wnorm[p["etf"]])
    return pr, pairs, blk, ret_df


def main(argv=None) -> int:
    from scipy import stats

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-date", default="2026-06-25")
    ap.add_argument("--start", default="2024-01-01")
    ap.add_argument("--min-days", type=int, default=120)
    ap.add_argument("--min-days-short", type=int, default=60, help="Min history for force-included / low-N names")
    ap.add_argument("--n-mc", type=int, default=10000)
    ap.add_argument("--block-len", type=int, default=10)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args(argv)

    uni = load_risk_sim_universe(args.run_date)
    panel = load_price_panel(args.run_date)
    built = build_portfolio(uni, panel, args.start, args.min_days, min_days_short=args.min_days_short)
    if built is None:
        print("[risk-sim] no eligible B4 pairs", file=sys.stderr)
        return 1
    pr, pairs, blk, ret_df = built
    prv = pr.dropna()
    arr = prv.to_numpy(dtype=float)
    sim_dates = [d.strftime("%Y-%m-%d") for d in prv.index]
    pair_returns = []
    for p in pairs:
        etf = p["etf"]
        if etf not in ret_df.columns:
            continue
        ser = ret_df[etf].reindex(prv.index).fillna(0.0)
        pair_returns.append({
            "etf": etf,
            "und": p["und"],
            "weight": p["weight"],
            "returns": [round(float(x), 6) for x in ser.to_numpy(dtype=float)],
        })

    perf = perf_from_returns(prv)
    t_df, t_loc, t_scale = stats.t.fit(arr)
    t_df = float(max(2.05, min(t_df, 50.0)))
    l_loc, l_scale = stats.laplace.fit(arr)

    rng = np.random.default_rng(args.seed)
    boot = mc_block_bootstrap(arr, args.n_mc, HORIZON, args.block_len, rng)
    tdd = mc_parametric(arr, args.n_mc, HORIZON, rng, "t")
    ldd = mc_parametric(arr, args.n_mc, HORIZON, rng, "laplace")

    payload = {
        "schema": "bucket4_risk_sim.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_date": args.run_date,
        "window_start": args.start,
        "cadence": {
            "cadence_signal_col": blk.get("cadence_signal_col"),
            "base_days": blk.get("base_days"),
            "k_tr": blk.get("k_tr"),
            "m_vcr": blk.get("m_vcr"),
            "min_interval": blk.get("min_interval"),
            "max_interval": blk.get("max_interval"),
        },
        "n_pairs": len(pairs),
        "pairs": pairs,
        "n_obs": int(len(arr)),
        "mean_daily": float(np.mean(arr)),
        "sim_dates": sim_dates,
        "port_daily_returns": [round(float(x), 6) for x in arr],
        "pair_returns": pair_returns,
        "weight_policy": {
            "description": "Uses proposed gross when >0; else optimal gross when short locate available; "
            "else structural/screener proxy for eligible stress names (incl. SMZ/CBRZ/APLZ).",
            "force_include_etfs": sorted(FORCE_INCLUDE_ETFS),
        },
        "fit_student_t": {"df": round(t_df, 3), "loc": round(float(t_loc), 6),
                          "scale": round(float(t_scale), 6)},
        "fit_laplace": {"loc": round(float(l_loc), 6), "scale": round(float(l_scale), 6)},
        "realized": {"cagr": round(perf["cagr"], 4), "ann_vol": round(perf["vol"], 4),
                     "sharpe": round(perf["sharpe"], 3), "hist_maxdd": round(perf["maxdd"], 4),
                     "calmar": round(perf["calmar"], 3) if np.isfinite(perf["calmar"]) else None,
                     "cvar95_daily": round(perf["cvar95_daily"], 5)},
        "reference_mc": {
            "horizon_days": HORIZON, "n_sims": args.n_mc, "block_len": args.block_len,
            "block_bootstrap": {k.replace("boot_", ""): round(v, 4) for k, v in tail_stats(boot, "boot").items()},
            "student_t": {k.replace("t_", ""): round(v, 4) for k, v in tail_stats(tdd, "t").items()},
            "laplace": {k.replace("lap_", ""): round(v, 4) for k, v in tail_stats(ldd, "lap").items()},
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, allow_nan=False))
    print(f"[risk-sim] wrote {OUT}")
    print(f"[risk-sim] pairs={len(pairs)} obs={len(arr)} cadence base_days={blk.get('base_days')} "
          f"k_tr={blk.get('k_tr')}  realized CAGR={perf['cagr']:.1%} maxDD={perf['maxdd']:.1%}")
    print(f"[risk-sim] bootstrap 1y dd p95={payload['reference_mc']['block_bootstrap'].get('dd_p95')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
