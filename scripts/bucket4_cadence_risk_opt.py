"""Cadence search for Bucket 4: maximize PnL subject to low tail-drawdown risk.

Goal
----
Find the rebalance-cadence design that maximizes a real B4 portfolio's PnL
while keeping the probability of a SUBSTANTIAL drawdown very low. We score
every cadence variant on (a) realized historical performance on the live B4
proposed positions, and (b) a fat-tailed Monte-Carlo / block-bootstrap tail of
the portfolio max-drawdown distribution.

Universe / weights
------------------
The candidates SELECTED in the current trade plan
(``data/runs/<date>/proposed_trades.csv``, B4/B5 sleeves, ``gross_target_usd>0``)
that have enough price history. Portfolio weights = those gross targets,
renormalized. Per-pair returns come from the solvent dynamic engine
(``run_bucket4_backtest_dynamic_h``) with the production v7 hedge series held
FIXED across variants, so only the rebalance schedule changes.

Cadence families tested
------------------------
1. linear        production parametrization  interval = base/(1 + k_tr*(tr_est-1)
                 + m_vcr*(vcr-vcr_med)); grid over base_days, k_tr, m_vcr.
2. vcr_insurance asymmetric "drawdown insurance": only SPEED UP when VCR is
                 elevated, never stretch past base when calm (k_tr=0, one-sided
                 m_vcr). Targets jump/gap risk directly.
3. vol_target    rebalance faster when realized vol (rv_daily) is high relative
                 to its own expanding median -> tighter hedge tracking in fast
                 markets.
4. drift_gate    event-driven: dense base schedule, execute only when the hedge
                 has drifted > threshold of gross, with a clock-floor force.
                 Directly bounds hedge mismatch (the mechanical DD source).
5. drift+vcr     drift gate on top of a VCR-accelerated schedule.

Risk-aware objective
---------------------
Primary: Calmar (CAGR / |maxDD|) and a CVaR-penalized return
``score = port_CAGR - LAMBDA * MC_dd_p95`` (LAMBDA configurable). We also report
the full MC tail so a risk appetite can be chosen explicitly.

Monte-Carlo tail (fat-tailed)
-----------------------------
On each variant's portfolio daily-return series we estimate the 1-year
max-drawdown distribution three ways:
  * stationary block bootstrap (preserves vol clustering / autocorrelation),
  * parametric Student-t fit (fat tails),
  * parametric Laplace fit (fat tails, lighter than t).
We report median / p95 / p99 / p99.9 max-drawdown and P(maxDD > thresholds).

Run
---
    python scripts/bucket4_cadence_risk_opt.py
    python scripts/bucket4_cadence_risk_opt.py --start 2024-01-01 --n-mc 20000
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
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
)

TRADING_DAYS = 252
HORIZON = 252           # 1-year drawdown horizon for Monte Carlo
RET_FLOOR = -0.95       # daily per-pair return floor (keeps bounded NAV > 0)
DD_THRESHOLDS = (0.15, 0.25, 0.40)  # "substantial" drawdown levels to report


# ---------------------------------------------------------------------------
# Cadence schedule builders
# ---------------------------------------------------------------------------
@dataclass
class Variant:
    name: str
    family: str
    # linear-family knob overrides on the production block
    overrides: dict = field(default_factory=dict)
    # custom-family parameters
    params: dict = field(default_factory=dict)


def _step_schedule(cal: pd.DatetimeIndex, interval_at, warmup: int) -> pd.DatetimeIndex:
    """Walk the calendar, stepping by interval_at(date) business days."""
    cal = pd.DatetimeIndex(cal).sort_values().unique()
    if warmup > 0:
        cal = cal[warmup:]
    dates, i, n = [], 0, len(cal)
    while i < n:
        d = pd.Timestamp(cal[i])
        dates.append(d)
        iv = interval_at(d)
        i += max(1, int(iv))
    return pd.DatetimeIndex(dates)


def schedule_for_variant(v: Variant, sig: pd.DataFrame, cal: pd.DatetimeIndex,
                         blk: dict, warmup: int):
    """Return (rebal_dates, engine_kwargs) for a variant.

    engine_kwargs may carry drift_threshold_share_of_gross / force_rebalance_after_days.
    """
    base = float(v.params.get("base_days", blk.get("base_days", 12.0)))
    mn = int(blk.get("min_interval", 1))
    mx = int(blk.get("max_interval", 21))

    if v.family == "linear":
        knobs = make_knobs(blk, **v.overrides)
        rb, _ = build_rebal_dates(sig, cal, knobs=knobs, warmup_bdays=warmup)
        return rb, {}

    if v.family == "vcr_insurance":
        m_vcr = float(v.params.get("m_vcr", 5.0))
        vcr = pd.to_numeric(sig.get("vcr"), errors="coerce")
        vmed = pd.to_numeric(sig.get("vcr_med"), errors="coerce")

        def iv_at(d):
            x = float(vcr.get(d, np.nan)); m = float(vmed.get(d, np.nan))
            denom = 1.0
            if np.isfinite(x) and np.isfinite(m):
                denom += m_vcr * max(0.0, x - m)      # one-sided: only speed up
            return int(np.clip(round(base / denom), mn, mx))

        return _step_schedule(cal, iv_at, warmup), {}

    if v.family == "vol_target":
        g = float(v.params.get("g", 1.0))
        rv = pd.to_numeric(sig.get("rv_daily"), errors="coerce")
        rv_ref = rv.expanding(min_periods=20).median()

        def iv_at(d):
            x = float(rv.get(d, np.nan)); r = float(rv_ref.get(d, np.nan))
            denom = 1.0
            if np.isfinite(x) and np.isfinite(r) and r > 0:
                denom += g * (x / r - 1.0)
            return int(np.clip(round(base / denom), mn, mx))

        return _step_schedule(cal, iv_at, warmup), {}

    if v.family in ("drift_gate", "drift_vcr"):
        thr = float(v.params.get("drift_thr", 0.05))
        floor_days = int(v.params.get("force_after", 21))
        if v.family == "drift_gate":
            # dense schedule = every business day; engine gate decides execution
            cal2 = pd.DatetimeIndex(cal).sort_values().unique()
            if warmup > 0:
                cal2 = cal2[warmup:]
            rb = cal2
        else:
            # VCR-accelerated base schedule, then drift-gated execution
            m_vcr = float(v.params.get("m_vcr", 5.0))
            vcr = pd.to_numeric(sig.get("vcr"), errors="coerce")
            vmed = pd.to_numeric(sig.get("vcr_med"), errors="coerce")

            def iv_at(d):
                x = float(vcr.get(d, np.nan)); m = float(vmed.get(d, np.nan))
                denom = 1.0
                if np.isfinite(x) and np.isfinite(m):
                    denom += m_vcr * max(0.0, x - m)
                return int(np.clip(round(base / denom), mn, mx))

            rb = _step_schedule(cal, iv_at, warmup)
        return rb, {"drift_threshold_share_of_gross": thr, "force_rebalance_after_days": floor_days}

    raise ValueError(f"unknown family {v.family}")


def build_variants() -> list[Variant]:
    out: list[Variant] = []
    # 1. linear grid (signal = tr_est)
    for base in (10.0, 12.0, 14.0):
        for k in (2.25, 0.0, -1.0):
            for m in (2.5, 5.0):
                tag = f"lin b{int(base)} k{k:+.2f} m{m:.1f}"
                out.append(Variant(tag, "linear",
                                   overrides={"cadence_signal_col": "tr_est",
                                              "k_tr": k, "m_vcr": m, "base_days": base}))
    # 2. VCR insurance (asymmetric)
    for base in (12.0, 14.0):
        for m in (4.0, 6.0, 8.0):
            out.append(Variant(f"vcr_ins b{int(base)} m{m:.0f}", "vcr_insurance",
                               params={"base_days": base, "m_vcr": m}))
    # 3. vol-targeted
    for base in (12.0,):
        for g in (0.75, 1.5):
            out.append(Variant(f"vol_tgt b{int(base)} g{g:.2f}", "vol_target",
                               params={"base_days": base, "g": g}))
    # 4. drift-gated event-driven
    for thr in (0.03, 0.05, 0.08):
        out.append(Variant(f"drift {thr:.0%} f21", "drift_gate",
                           params={"drift_thr": thr, "force_after": 21}))
    # 5. drift + VCR hybrid
    out.append(Variant("drift 5% + vcr m6", "drift_vcr",
                       params={"drift_thr": 0.05, "force_after": 21, "base_days": 12.0, "m_vcr": 6.0}))
    return out


# ---------------------------------------------------------------------------
# Portfolio + performance
# ---------------------------------------------------------------------------
def port_returns(ret_df: pd.DataFrame, weights: pd.Series) -> pd.Series:
    w = weights.reindex(ret_df.columns).fillna(0.0).to_numpy(dtype=float)
    if w.sum() <= 0:
        w = np.ones(len(w))
    w = w / w.sum()
    r = ret_df.clip(lower=RET_FLOOR, upper=0.95).fillna(0.0).to_numpy(dtype=float)
    return pd.Series(r @ w, index=ret_df.index)


def perf_from_returns(pr: pd.Series) -> dict:
    pr = pr.dropna()
    nav = (1.0 + pr).cumprod()
    years = max((nav.index[-1] - nav.index[0]).days / 365.25, 1e-6)
    cagr = float(nav.iloc[-1]) ** (1 / years) - 1.0
    vol = float(pr.std(ddof=1) * np.sqrt(TRADING_DAYS))
    sharpe = float(pr.mean() * TRADING_DAYS / vol) if vol > 0 else np.nan
    maxdd = float((nav / nav.cummax() - 1.0).min())
    calmar = cagr / abs(maxdd) if maxdd < 0 else np.nan
    # daily CVaR 95 (expected shortfall of daily returns)
    q = np.nanpercentile(pr.to_numpy(), 5)
    cvar95 = float(pr[pr <= q].mean()) if np.isfinite(q) else np.nan
    return {"cagr": cagr, "vol": vol, "sharpe": sharpe, "maxdd": maxdd,
            "calmar": calmar, "cvar95_daily": cvar95}


# ---------------------------------------------------------------------------
# Monte Carlo / fat-tailed drawdown distribution
# ---------------------------------------------------------------------------
def _maxdd_from_path(r: np.ndarray) -> float:
    eq = np.cumprod(1.0 + np.clip(r, RET_FLOOR, None))
    peak = np.maximum.accumulate(eq)
    return float((eq / peak - 1.0).min())


def mc_block_bootstrap(pr: np.ndarray, n: int, horizon: int, block: int, rng) -> np.ndarray:
    """Stationary (circular) block bootstrap of max-drawdown over `horizon`."""
    m = len(pr)
    if m < block + 5:
        return np.array([])
    n_blocks = int(np.ceil(horizon / block))
    dds = np.empty(n)
    for i in range(n):
        starts = rng.integers(0, m, size=n_blocks)
        path = np.concatenate([np.take(pr, range(s, s + block), mode="wrap") for s in starts])[:horizon]
        dds[i] = _maxdd_from_path(path)
    return dds


def mc_parametric(pr: np.ndarray, n: int, horizon: int, rng, kind: str) -> np.ndarray:
    """Fit Student-t or Laplace to daily returns, simulate max-drawdown paths."""
    from scipy import stats
    if kind == "t":
        df, loc, scale = stats.t.fit(pr)
        df = max(2.05, min(df, 50.0))  # keep finite variance, avoid degenerate fit
        draws = stats.t.rvs(df, loc=loc, scale=scale, size=(n, horizon), random_state=rng)
    elif kind == "laplace":
        loc, scale = stats.laplace.fit(pr)
        draws = stats.laplace.rvs(loc=loc, scale=scale, size=(n, horizon), random_state=rng)
    else:
        raise ValueError(kind)
    eq = np.cumprod(1.0 + np.clip(draws, RET_FLOOR, None), axis=1)
    peak = np.maximum.accumulate(eq, axis=1)
    return (eq / peak - 1.0).min(axis=1)


def tail_stats(dds: np.ndarray, prefix: str) -> dict:
    if dds.size == 0:
        return {}
    a = np.abs(dds)  # work with positive drawdown magnitudes
    out = {
        f"{prefix}_dd_med": float(np.median(a)),
        f"{prefix}_dd_p95": float(np.percentile(a, 95)),
        f"{prefix}_dd_p99": float(np.percentile(a, 99)),
        f"{prefix}_dd_p999": float(np.percentile(a, 99.9)),
    }
    for thr in DD_THRESHOLDS:
        out[f"{prefix}_P(dd>{int(thr*100)})"] = float((a > thr).mean())
    return out


# ---------------------------------------------------------------------------
def build_pair_returns(uni, panel, start, min_days):
    """Per-pair daily returns (prod cadence hedge) + gross weights for live B4."""
    blk = knobs_from_yaml()
    prod_knobs = make_knobs(blk)
    df = uni[uni["sleeve"].isin(["inverse_decay_bucket4", "volatility_etp_bucket5"])].copy()
    df = df[df["ETF"].isin(panel.keys())].reset_index(drop=True)

    ctx, weights = [], {}
    for _, row in df.iterrows():
        etf, und = row["ETF"], row["Underlying"]
        px = panel[etf]
        cal = pd.DatetimeIndex([d for d in px.index if d >= pd.Timestamp(start)])
        if len(cal) < min_days:
            continue
        sig = get_pair_signal(etf, und, cal, history={}, underlying_prices=px["b_px"],
                              window=60, lookahead_shift=1)
        h_daily = build_h_series(sig, cal, knobs=prod_knobs)
        borrow = float(pd.to_numeric(row.get("borrow_current"), errors="coerce") or 0.0)
        ctx.append({"etf": etf, "und": und, "px": px.reindex(cal), "sig": sig, "h": h_daily,
                    "beta_a": float(row["Delta"]), "borrow": borrow, "cal": cal})
        weights[etf] = float(pd.to_numeric(row["gross_target_usd"], errors="coerce") or 0.0)
    return ctx, pd.Series(weights), blk


def run(uni, panel, start, min_days, n_mc, block_len, lam, seed):
    ctx, gross_w, blk = build_pair_returns(uni, panel, start, min_days)
    print(f"[risk-opt] eligible pairs (>= {min_days}d): {len(ctx)}  "
          f"weights=gross_target_usd")
    for c in ctx:
        print(f"          {c['etf']:6s} und={c['und']:6s} days={len(c['cal']):4d} "
              f"gross=${gross_w[c['etf']]:,.0f} borrow={c['borrow']:.1%}")
    if len(ctx) < 2:
        print("[risk-opt] too few pairs")
        return pd.DataFrame()

    variants = build_variants()
    rng = np.random.default_rng(seed)
    rows = []
    for v in variants:
        ret_cols, n_reb = {}, []
        for c in ctx:
            rb, ekw = schedule_for_variant(v, c["sig"], c["cal"], blk, warmup=60)
            bt = run_bucket4_backtest_dynamic_h(
                c["px"], c["h"], rb, initial_capital=1.0, gross_multiplier=1.0,
                beta_a=-abs(c["beta_a"]), beta_b=1.0, borrow_a_annual=c["borrow"],
                slippage_bps=20.0, **ekw,
            )
            ret_cols[c["etf"]] = bt["ret"]
            n_reb.append(int(bt["rebalance"].sum()))
        ret_df = pd.DataFrame(ret_cols).reindex(
            sorted(set().union(*[set(s.index) for s in ret_cols.values()]))
        )
        pr = port_returns(ret_df, gross_w)
        perf = perf_from_returns(pr)

        prv = pr.dropna().to_numpy()
        boot = mc_block_bootstrap(prv, n_mc, HORIZON, block_len, rng)
        tstat = mc_parametric(prv, n_mc, HORIZON, rng, "t")
        lstat = mc_parametric(prv, n_mc, HORIZON, rng, "laplace")

        row = {"variant": v.name, "family": v.family, "mean_n_rebal": float(np.mean(n_reb)),
               **perf}
        row.update(tail_stats(boot, "boot"))
        row.update(tail_stats(tstat, "t"))
        row.update(tail_stats(lstat, "lap"))
        # risk-aware score: PnL penalized by bootstrap p95 1y drawdown
        row["score"] = perf["cagr"] - lam * row.get("boot_dd_p95", np.nan)
        rows.append(row)

    res = pd.DataFrame(rows).set_index("variant")
    return res.sort_values("score", ascending=False)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-date", default="2026-06-25")
    ap.add_argument("--start", default="2024-01-01")
    ap.add_argument("--min-days", type=int, default=150)
    ap.add_argument("--n-mc", type=int, default=10000)
    ap.add_argument("--block-len", type=int, default=10)
    ap.add_argument("--lam", type=float, default=0.5, help="CVaR/drawdown penalty weight in score")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--outdir", type=Path, default=REPO / "notebooks/output/sizing_tilt_cadence_bt")
    args = ap.parse_args(argv)
    args.outdir.mkdir(parents=True, exist_ok=True)

    uni = load_universe(args.run_date)
    panel = load_price_panel(args.run_date)
    res = run(uni, panel, args.start, args.min_days, args.n_mc, args.block_len, args.lam, args.seed)
    if res.empty:
        return 1

    pd.set_option("display.width", 240)
    pd.set_option("display.max_columns", 50)
    print("\n=== CADENCE RISK-OPT: realized performance (ranked by risk-aware score) ===")
    print(res[["family", "cagr", "vol", "sharpe", "maxdd", "calmar", "mean_n_rebal", "score"]].round(3).to_string())
    print("\n=== Monte-Carlo 1y max-drawdown tail (block bootstrap) ===")
    print(res[["boot_dd_med", "boot_dd_p95", "boot_dd_p99", "boot_dd_p999",
               "boot_P(dd>15)", "boot_P(dd>25)", "boot_P(dd>40)"]].round(3).to_string())
    print("\n=== Fat-tailed parametric tail (Student-t / Laplace), p99 max-DD ===")
    print(res[["t_dd_p95", "t_dd_p99", "t_dd_p999", "lap_dd_p95", "lap_dd_p99", "lap_dd_p999"]].round(3).to_string())

    res.to_csv(args.outdir / "cadence_risk_opt.csv")
    print(f"\n[risk-opt] DONE -> {args.outdir / 'cadence_risk_opt.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
