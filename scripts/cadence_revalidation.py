"""Re-validation of the Bucket-4 hedge/rebalance CADENCE backtest.

Why this exists
---------------
The first cadence pass (``sizing_tilt_cadence_bt.run_cadence_backtest``) summed
per-pair DOLLAR equities into one portfolio NAV. For a short book that NAV can
cross zero when a single pair blows up, which makes ``pct_change`` (Sharpe) and
the CAGR ratio meaningless. This script fixes the aggregation and is explicit
about every assumption so the cadence comparison can be trusted (or correctly
distrusted).

What is held FIXED across all cadence variants (so only cadence is tested)
--------------------------------------------------------------------------
* Pair universe          - the names SELECTED in the live trade plan
                           (proposed_trades.csv, gross_target_usd > 0) that have
                           enough price history to simulate.
* Hedge-ratio series h_t - production v7 cadence hedge (build_h_series with the
                           YAML knobs); identical series fed to every variant.
* Leg betas / borrow     - from the live plan (Delta, borrow_current).
* Costs                  - 20 bps slippage, ETF-leg borrow drag.
* Engine                 - run_bucket4_backtest_dynamic_h (solvent 2-leg book).

What VARIES
-----------
Only ``build_rebal_dates`` inputs: cadence_signal_col (tr_est vs raw tr), k_tr
(sign/strength of the trend tilt), and base_days (mean spacing). Production is
cadence_signal_col=tr_est, k_tr=+2.25, base_days=12, m_vcr=2.5.

Robust metrics (avoid the zero-crossing trap)
---------------------------------------------
For each variant we run every pair SEPARATELY funded (1.0 capital) and report:
* per-pair CAGR / Sharpe / MaxDD, then the cross-pair MEAN and MEDIAN
  (the convention the B4 scorecard scripts use), plus
* n_ruin   - pairs whose equity hit <= 0 (a cadence that causes fewer ruins is
             better), and
* a BOUNDED portfolio: gross-weighted average of per-pair DAILY RETURNS
  (each pair's return floored at -95%/day), compounded. This NAV stays > 0, so
  its CAGR/Sharpe are well defined.
Plus cadence diagnostics (mean interval days, # rebalances) to PROVE each knob
set actually changes the schedule -> confirms we tested the right values.

Run
---
    python scripts/cadence_revalidation.py
    python scripts/cadence_revalidation.py --min-days 150 --start 2018-01-01
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.bucket4_dynamic_bt import run_bucket4_backtest_dynamic_h  # noqa: E402
from scripts.bucket4_hedge_cadence import build_h_series, build_rebal_dates  # noqa: E402
from scripts.bucket4_vol_shape_signals import get_pair_signal  # noqa: E402
from scripts.sizing_tilt_cadence_bt import (  # noqa: E402
    knobs_from_yaml,
    load_price_panel,
    load_universe,
    make_knobs,
    tr_est_series,
)

TRADING_DAYS = 252


# Cadence variants: name -> dict of knob overrides on the production block.
# Production: cadence_signal_col=tr_est, k_tr=+2.25, base_days=12, m_vcr=2.5.
CADENCE_VARIANTS: dict[str, dict] = {
    "PROD trEst k+2.25 b12":      {"cadence_signal_col": "tr_est", "k_tr": 2.25, "base_days": 12.0},
    "TRagnostic k0 b12":          {"cadence_signal_col": "tr_est", "k_tr": 0.0, "base_days": 12.0},
    "contrarian k-1.0 b12":       {"cadence_signal_col": "tr_est", "k_tr": -1.0, "base_days": 12.0},
    "contrarian k-2.25 b12":      {"cadence_signal_col": "tr_est", "k_tr": -2.25, "base_days": 12.0},
    "rawTR k+2.25 b12":           {"cadence_signal_col": "tr", "k_tr": 2.25, "base_days": 12.0},
    "rawTR contrarian k-2.25 b12": {"cadence_signal_col": "tr", "k_tr": -2.25, "base_days": 12.0},
    # base_days sensitivity (TR-agnostic so spacing is the only driver)
    "fixed ~8d (k0 b8)":          {"cadence_signal_col": "tr_est", "k_tr": 0.0, "base_days": 8.0},
    "fixed ~12d (k0 b12)":        {"cadence_signal_col": "tr_est", "k_tr": 0.0, "base_days": 12.0},
    "fixed ~16d (k0 b16)":        {"cadence_signal_col": "tr_est", "k_tr": 0.0, "base_days": 16.0},
}


def pair_perf(eq: pd.Series) -> dict:
    """Per-pair CAGR/Sharpe/MaxDD from a single dollar-equity curve.

    Equity that touches <= 0 is flagged as ruin and CAGR set to -100%.
    """
    eq = eq.dropna()
    out = {"cagr": np.nan, "sharpe": np.nan, "maxdd": np.nan, "ruin": False}
    if len(eq) < 30:
        return out
    e0, emin, e1 = float(eq.iloc[0]), float(eq.min()), float(eq.iloc[-1])
    ruin = emin <= 0.0
    out["ruin"] = ruin
    years = max((eq.index[-1] - eq.index[0]).days / 365.25, 1e-6)
    if e0 <= 0:
        return out
    if ruin or e1 <= 0:
        out["cagr"] = -1.0
    else:
        out["cagr"] = (e1 / e0) ** (1 / years) - 1.0
    rets = eq.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    if ruin:
        # truncate the series at first ruin for a meaningful vol/dd
        first_ruin = eq[eq <= 0].index[0]
        rets = eq.loc[:first_ruin].pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    vol = float(rets.std(ddof=1) * np.sqrt(TRADING_DAYS)) if len(rets) > 2 else np.nan
    out["sharpe"] = float(rets.mean() * TRADING_DAYS / vol) if vol and vol > 0 else np.nan
    out["maxdd"] = float((eq / eq.cummax() - 1.0).min())
    return out


def port_perf_from_returns(ret_df: pd.DataFrame, weights: pd.Series) -> dict:
    """Bounded portfolio: gross-weighted mean of per-pair daily returns.

    Per-pair daily returns are floored at -95% so one blown-up pair cannot send
    the compounded NAV through zero; the NAV is therefore always > 0 and its
    CAGR/Sharpe are well defined.
    """
    w = weights.reindex(ret_df.columns).fillna(0.0).to_numpy(dtype=float)
    if w.sum() <= 0:
        w = np.ones(len(w))
    w = w / w.sum()
    r = ret_df.clip(lower=-0.95, upper=0.95).fillna(0.0).to_numpy(dtype=float)
    port_r = r @ w
    nav = pd.Series(np.cumprod(1.0 + port_r), index=ret_df.index)
    years = max((nav.index[-1] - nav.index[0]).days / 365.25, 1e-6)
    cagr = (float(nav.iloc[-1]) / float(nav.iloc[0])) ** (1 / years) - 1.0
    pr = pd.Series(port_r, index=ret_df.index)
    vol = float(pr.std(ddof=1) * np.sqrt(TRADING_DAYS))
    sharpe = float(pr.mean() * TRADING_DAYS / vol) if vol > 0 else np.nan
    maxdd = float((nav / nav.cummax() - 1.0).min())
    return {"port_cagr": cagr, "port_sharpe": sharpe, "port_maxdd": maxdd}


def build_pairs(uni: pd.DataFrame, panel: dict, start: str, min_days: int):
    """Pre-compute fixed (signal, production hedge) context per eligible pair."""
    blk = knobs_from_yaml()
    prod_knobs = make_knobs(blk)
    # B5 uses bucket5_carry_bt, not B4 dynamic-h — keep cadence research B4-only.
    df = uni[uni["sleeve"].isin(["inverse_decay_bucket4"])].copy()
    df = df[df["ETF"].isin(panel.keys())].reset_index(drop=True)

    ctx, dropped = [], []
    for _, row in df.iterrows():
        etf, und = row["ETF"], row["Underlying"]
        px = panel[etf]
        cal = pd.DatetimeIndex([d for d in px.index if d >= pd.Timestamp(start)])
        if len(cal) < min_days:
            dropped.append((etf, len(cal)))
            continue
        sig = get_pair_signal(etf, und, cal, history={}, underlying_prices=px["b_px"],
                              window=60, lookahead_shift=1)
        tr_est_ok = float(pd.to_numeric(sig.get("tr_est"), errors="coerce").notna().mean()) if "tr_est" in sig else 0.0
        h_daily = build_h_series(sig, cal, knobs=prod_knobs)
        borrow = float(pd.to_numeric(row.get("borrow_current"), errors="coerce") or 0.0)
        ctx.append({
            "etf": etf, "und": und, "px": px.reindex(cal), "sig": sig, "h": h_daily,
            "beta_a": float(row["Delta"]), "borrow": borrow,
            "gross": float(pd.to_numeric(row["gross_target_usd"], errors="coerce") or 0.0),
            "cal": cal, "n_days": len(cal), "tr_est_ok": tr_est_ok,
        })
    return ctx, dropped, blk


def run(uni, panel, start, min_days):
    ctx, dropped, blk = build_pairs(uni, panel, start, min_days)
    print(f"[reval] eligible pairs (>= {min_days}d): {len(ctx)}")
    for c in ctx:
        print(f"        {c['etf']:6s} und={c['und']:6s} days={c['n_days']:4d} "
              f"tr_est_valid={c['tr_est_ok']:.0%} borrow={c['borrow']:.1%}")
    for etf, n in dropped:
        print(f"        DROP {etf:6s} days={n}")
    if not ctx:
        return pd.DataFrame()

    gross_w = pd.Series({c["etf"]: c["gross"] for c in ctx})

    rows = []
    for vname, over in CADENCE_VARIANTS.items():
        knobs = make_knobs(blk, **over)
        per_pair, ret_cols, intervals, nrebs = [], {}, [], []
        for c in ctx:
            rb, diag = build_rebal_dates(c["sig"], c["cal"], knobs=knobs, warmup_bdays=60)
            bt = run_bucket4_backtest_dynamic_h(
                c["px"], c["h"], rb, initial_capital=1.0,
                gross_multiplier=1.0, beta_a=-abs(c["beta_a"]), beta_b=1.0,
                borrow_a_annual=c["borrow"], slippage_bps=20.0,
            )
            pp = pair_perf(bt["equity"])
            pp["etf"] = c["etf"]
            per_pair.append(pp)
            ret_cols[c["etf"]] = bt["ret"]
            nrebs.append(len(rb))
            try:
                iv = pd.to_numeric(pd.Series(diag.get("interval_days")), errors="coerce")
                if iv.notna().any():
                    intervals.append(float(iv.mean()))
            except Exception:
                pass
        pp_df = pd.DataFrame(per_pair).set_index("etf")
        ret_df = pd.DataFrame(ret_cols).reindex(
            sorted(set().union(*[set(s.index) for s in ret_cols.values()]))
        )
        port = port_perf_from_returns(ret_df, gross_w)
        n_ruin = int(pp_df["ruin"].sum())
        rows.append({
            "variant": vname,
            "mean_cagr": float(pp_df["cagr"].mean(skipna=True)),
            "median_cagr": float(pp_df["cagr"].median(skipna=True)),
            "mean_sharpe": float(pp_df["sharpe"].mean(skipna=True)),
            "median_sharpe": float(pp_df["sharpe"].median(skipna=True)),
            "mean_maxdd": float(pp_df["maxdd"].mean(skipna=True)),
            "n_ruin": n_ruin,
            "port_cagr": port["port_cagr"],
            "port_sharpe": port["port_sharpe"],
            "port_maxdd": port["port_maxdd"],
            "mean_interval_d": float(np.nanmean(intervals)) if intervals else np.nan,
            "mean_n_rebal": float(np.mean(nrebs)) if nrebs else np.nan,
        })
    return pd.DataFrame(rows).set_index("variant")


def run_b4_sizing(uni, panel, start, min_days):
    """Test TR/VCR-aware PAIR SIZING on the real B4 pairs, cadence held at PROD.

    Per-pair daily returns are produced once (production cadence + v7 hedge), so
    they are IDENTICAL across weight schemes -> any difference is attributable to
    the weighting alone. Pair characteristics use the median point-in-time tr_est
    and vcr over each pair's window (cross-sectional rank within the live book).

    Schemes (multiplier on the current gross-target weights):
      risk_only        - current plan gross weights, no TR/VCR tilt (baseline)
      lowTR            - up-weight choppy (low tr_est) underlyings (more daily-
                         reset decay to harvest)
      vcr_penalty      - down-weight high-VCR (jump/gap-prone) underlyings (the
                         ruin / path-risk driver)
      lowTR+vcr_pen    - both
    """
    ctx, dropped, blk = build_pairs(uni, panel, start, min_days)
    if len(ctx) < 2:
        return pd.DataFrame()
    prod_knobs = make_knobs(blk)

    # Per-pair returns (production cadence) + characteristic tr_est / vcr.
    ret_cols, chars = {}, {}
    for c in ctx:
        rb, _ = build_rebal_dates(c["sig"], c["cal"], knobs=prod_knobs, warmup_bdays=60)
        bt = run_bucket4_backtest_dynamic_h(
            c["px"], c["h"], rb, initial_capital=1.0, gross_multiplier=1.0,
            beta_a=-abs(c["beta_a"]), beta_b=1.0, borrow_a_annual=c["borrow"], slippage_bps=20.0,
        )
        ret_cols[c["etf"]] = bt["ret"]
        sig = c["sig"]
        tr_med = float(pd.to_numeric(sig.get("tr_est"), errors="coerce").median())
        vcr_med = float(pd.to_numeric(sig.get("vcr"), errors="coerce").median())
        chars[c["etf"]] = {"tr": tr_med, "vcr": vcr_med, "gross": c["gross"]}
    ret_df = pd.DataFrame(ret_cols).reindex(
        sorted(set().union(*[set(s.index) for s in ret_cols.values()]))
    )
    ch = pd.DataFrame(chars).T
    tr_rank = ch["tr"].rank(pct=True)
    vcr_rank = ch["vcr"].rank(pct=True)
    base = ch["gross"] / ch["gross"].sum()

    def clip_norm(mult):
        w = (base * mult.reindex(base.index).fillna(1.0)).clip(lower=0)
        return w / w.sum()

    schemes = {
        "risk_only (current)": base,
        "lowTR a1.0":          clip_norm((1 + 1.0 * (0.5 - tr_rank)).clip(0.4, 1.6)),
        "vcr_penalty a1.0":    clip_norm((1 - 1.0 * (vcr_rank - 0.5)).clip(0.4, 1.6)),
        "lowTR+vcr_pen":       clip_norm(((1 + 1.0 * (0.5 - tr_rank)) * (1 - 1.0 * (vcr_rank - 0.5))).clip(0.4, 1.6)),
    }
    rows = []
    for name, w in schemes.items():
        p = port_perf_from_returns(ret_df, w)
        r = {"scheme": name, **p}
        rows.append(r)
    print("\n        pair characteristics (median over window):")
    print(ch.assign(tr_rank=tr_rank.round(2), vcr_rank=vcr_rank.round(2),
                    base_w=base.round(3)).round(3).to_string())
    return pd.DataFrame(rows).set_index("scheme")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-date", default="2026-06-25")
    ap.add_argument("--start", default="2018-01-01")
    ap.add_argument("--min-days", type=int, default=150)
    ap.add_argument("--outdir", type=Path, default=REPO / "notebooks/output/sizing_tilt_cadence_bt")
    args = ap.parse_args(argv)
    args.outdir.mkdir(parents=True, exist_ok=True)

    print("[reval] loading universe + prices ...")
    uni = load_universe(args.run_date)
    panel = load_price_panel(args.run_date)
    res = run(uni, panel, args.start, args.min_days)
    if res.empty:
        print("[reval] no eligible pairs")
        return 1
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    print("\n=== CADENCE RE-VALIDATION (per-pair robust metrics) ===")
    print(res[["mean_cagr", "median_cagr", "mean_sharpe", "median_sharpe",
               "n_ruin", "port_cagr", "port_sharpe", "port_maxdd",
               "mean_interval_d", "mean_n_rebal"]].round(3).to_string())
    res.to_csv(args.outdir / "cadence_revalidation.csv")

    print("\n=== B4 TR/VCR PAIR-SIZING (cadence + hedge held at PROD) ===")
    sz = run_b4_sizing(uni, panel, args.start, args.min_days)
    if not sz.empty:
        print(sz.round(4).to_string())
        sz.to_csv(args.outdir / "b4_sizing_trvcr.csv")
    print(f"\n[reval] DONE -> {args.outdir / 'cadence_revalidation.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
