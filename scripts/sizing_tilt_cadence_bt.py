"""Backtest: sizing-tilt direction/strength (Section B) + cadence mean-reversion (#3).

Research-only. Modifies no production config or data. Writes to
``notebooks/output/sizing_tilt_cadence_bt/``.

What this answers
-----------------
B (sizing tilt)  Does the cross-sectional trend tilt help book CAGR, and does
    up-weighting HIGH-TR (trendier) underlyings beat the production tilt, which
    up-weights LOW-TR (choppier) underlyings? Sweeps direction (alpha sign),
    strength (|alpha|), asymmetric clip bands, and a net-edge interaction.

#3 (cadence)  Trend ratio mean-reverts (Section A). The production cadence
    rebalances FASTER when tr_est is high ("trending -> keep trending"). Test
    contrarian / TR-agnostic cadence variants against production on B4 CAGR.

Eligibility (hard constraint)
-----------------------------
Universe = the candidates actually selected in the CURRENT trade plan
(``data/runs/<date>/proposed_trades.csv`` rows with a sleeve and
``gross_target_usd > 0``). By construction those already passed every
production gate (min net edge, max borrow, purgatory, vol floor, blacklist,
sleeve masks), so every backtest here only trades names that would be in the
live book today.

Isolation of effects
---------------------
* Tilt test: per-pair return series and the weekly rebalance schedule are held
  fixed; only the cross-sectional weight function changes between variants, so
  CAGR differences are attributable to the tilt alone. Base (un-tilted)
  decay-score weights use the production ``weighting`` block.
* Cadence test: hedge-ratio series and pair set are held fixed; only the
  rebalance-date schedule changes between variants.

Prices: offline ``etf_metrics_daily.parquet`` (aligned ``etf_adj_close`` /
``underlying_adj_close`` per ETF). Trend signal recomputed point-in-time from
the underlying series (1-day shift, no look-ahead).

Run
---
    python scripts/sizing_tilt_cadence_bt.py
    python scripts/sizing_tilt_cadence_bt.py --start 2024-06-01
"""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from generate_trade_plan import _decay_score_weights, _trend_percentile_signal  # noqa: E402
from vol_shape import build_underlying_vol_shape_history  # noqa: E402
from scripts.bucket4_dynamic_bt import run_bucket4_backtest_dynamic_h  # noqa: E402
from scripts.bucket4_hedge_cadence import (  # noqa: E402
    HedgeCadenceKnobs,
    build_h_series,
    build_rebal_dates,
)
from scripts.bucket4_vol_shape_signals import get_pair_signal  # noqa: E402

TRADING_DAYS = 252


def _norm(x) -> str:
    return str(x).strip().upper().replace(".", "-")


# ---------------------------------------------------------------------------
# Config / universe
# ---------------------------------------------------------------------------
def load_weighting(sleeve: str) -> dict:
    """Merged ``weighting`` block for a sleeve (resolves the YAML anchor)."""
    raw = yaml.safe_load((REPO / "config/strategy_config.yml").read_text())
    sleeves = raw["portfolio"]["sleeves"]
    w = dict(sleeves[sleeve].get("weighting") or {})
    # PyYAML resolves ``<<`` merge keys, but guard in case of a plain dict.
    if "method" not in w:
        defaults = raw["portfolio"].get("_weighting_defaults", {})
        w = {**defaults, **w}
    return w


def load_universe(run_date: str) -> pd.DataFrame:
    pt = pd.read_csv(REPO / f"data/runs/{run_date}/proposed_trades.csv")
    sel = pt[pd.to_numeric(pt["gross_target_usd"], errors="coerce") > 0].copy()
    sel["ETF"] = sel["ETF"].map(_norm)
    sel["Underlying"] = sel["Underlying"].map(_norm)
    sel["delta_abs"] = pd.to_numeric(sel["Delta"], errors="coerce").abs()
    return sel


def load_price_panel(run_date: str) -> dict[str, pd.DataFrame]:
    """Per-ETF aligned (etf_adj_close, underlying_adj_close) from the parquet."""
    pq = REPO / f"data/runs/{run_date}/model_inputs/etf_metrics_daily.parquet"
    md = pd.read_parquet(pq, columns=["date", "ticker", "etf_adj_close", "underlying_adj_close"])
    md["ticker"] = md["ticker"].map(_norm)
    md["date"] = pd.to_datetime(md["date"], errors="coerce").dt.normalize()
    md = md.dropna(subset=["date"]).sort_values(["ticker", "date"])
    out: dict[str, pd.DataFrame] = {}
    for etf, g in md.groupby("ticker"):
        g = g.dropna(subset=["etf_adj_close", "underlying_adj_close"])
        if len(g) < 80:
            continue
        df = pd.DataFrame({
            "a_px": g["etf_adj_close"].to_numpy(dtype=float),
            "b_px": g["underlying_adj_close"].to_numpy(dtype=float),
        }, index=pd.DatetimeIndex(g["date"]))
        df = df[~df.index.duplicated(keep="last")]
        out[etf] = df
    return out


def tr_est_series(prices_b: pd.Series, window: int = 60) -> pd.Series:
    """Point-in-time forward trend ratio (shifted 1 day; no look-ahead)."""
    hist = build_underlying_vol_shape_history(prices_b, window=window, max_points=0)
    ser = hist.get("series") or []
    if not ser:
        return pd.Series(dtype=float)
    df = pd.DataFrame(ser)
    s = pd.Series(
        pd.to_numeric(df["trend_ratio_fwd"], errors="coerce").to_numpy(),
        index=pd.to_datetime(df["date"]).dt.normalize(),
    )
    return s.shift(1)  # signal known only as of the prior close


# ---------------------------------------------------------------------------
# Per-pair daily return per gross dollar (legs at current plan fractions)
# ---------------------------------------------------------------------------
def pair_daily_returns(row: pd.Series, px: pd.DataFrame, borrow_on_etf: bool) -> pd.Series:
    gross = abs(float(row["long_usd"])) + abs(float(row["short_usd"]))
    if gross <= 0:
        return pd.Series(dtype=float)
    w_a = float(row["long_usd"]) / gross   # ETF leg fraction (signed)
    w_b = float(row["short_usd"]) / gross  # underlying leg fraction (signed)
    r_a = px["a_px"].pct_change()
    r_b = px["b_px"].pct_change()
    ret = (w_a * r_a + w_b * r_b).fillna(0.0)
    if borrow_on_etf:
        borrow = float(pd.to_numeric(row.get("borrow_current"), errors="coerce") or 0.0)
        ret = ret - borrow * abs(w_a) / TRADING_DAYS  # short-ETF borrow drag
    return ret


# ---------------------------------------------------------------------------
# Tilt variants
# ---------------------------------------------------------------------------
def tilt_cfg(alpha: float, floor: float = 0.5, ceiling: float = 1.5) -> dict:
    return {
        "enabled": True,
        "column": "und_trend_ratio_fwd_60d",
        "percentile_mode": "cross_sectional",
        "alpha": alpha,
        "neutral_pctile": 0.5,
        "floor": floor,
        "ceiling": ceiling,
        "missing": "neutral",
    }


# alpha > 0 -> production direction (LOW TR up-weighted); alpha < 0 -> HIGH TR up-weighted.
TILT_VARIANTS: dict[str, dict] = {
    "no_tilt": {"enabled": False},
    "lowTR_prod_a1.0": tilt_cfg(1.0),
    "lowTR_a0.5": tilt_cfg(0.5),
    "lowTR_a1.5": tilt_cfg(1.5),
    "highTR_a0.5": tilt_cfg(-0.5),
    "highTR_a1.0": tilt_cfg(-1.0),
    "highTR_a1.5": tilt_cfg(-1.5),
    "highTR_a2.0": tilt_cfg(-2.0),
    "highTR_asym_c2.0": tilt_cfg(-1.0, floor=0.5, ceiling=2.0),
}


def perf(nav: pd.Series) -> dict:
    nav = nav.dropna()
    if len(nav) < 30:
        return {"cagr": np.nan, "vol": np.nan, "sharpe": np.nan, "maxdd": np.nan}
    rets = nav.pct_change().dropna()
    years = max((nav.index[-1] - nav.index[0]).days / 365.25, 1e-6)
    n0, n1 = float(nav.iloc[0]), float(nav.iloc[-1])
    if n0 <= 0:
        cagr = np.nan
    elif n1 <= 0:
        cagr = -1.0  # capital wiped out
    else:
        cagr = (n1 / n0) ** (1 / years) - 1.0
    vol = float(rets.std(ddof=1) * np.sqrt(TRADING_DAYS))
    sharpe = float(rets.mean() * TRADING_DAYS / vol) if vol > 0 else np.nan
    maxdd = float((nav / nav.cummax() - 1).min())
    return {"cagr": float(cagr), "vol": vol, "sharpe": sharpe, "maxdd": maxdd}


def portfolio_tilt_sim(etfs, R, TR, base_w, cal, rebal_days):
    """Run all TILT_VARIANTS on a fixed per-name daily-return matrix R.

    R, TR: DataFrames indexed by ``cal`` with one column per ETF (daily return,
    point-in-time tr_est). Only the cross-sectional weight function varies.
    """
    navs: dict[str, pd.Series] = {}
    rows = []
    for vname, vcfg in TILT_VARIANTS.items():
        nav = pd.Series(index=cal, dtype=float)
        equity = 1.0
        cur_w = pd.Series(0.0, index=etfs)
        turnover = 0.0
        n_rebal = 0
        for i, d in enumerate(cal):
            if d in rebal_days or i == 0:
                active = [e for e in etfs if np.isfinite(R.at[d, e]) and np.isfinite(TR.at[d, e])]
                if active:
                    bw = base_w.reindex(active).fillna(0.0).to_numpy(dtype=float)
                    if bw.sum() <= 0:
                        bw = np.ones(len(active))
                    if vcfg.get("enabled", False):
                        sub = pd.DataFrame({"und_trend_ratio_fwd_60d": [TR.at[d, e] for e in active]})
                        _p, mult = _trend_percentile_signal(sub, {"trend_percentile_multiplier": vcfg})
                        bw = bw * mult
                    w = bw / bw.sum() if bw.sum() > 0 else np.ones(len(active)) / len(active)
                    new_w = pd.Series(0.0, index=etfs)
                    new_w.loc[active] = w
                    turnover += float(np.abs(new_w - cur_w).sum())
                    cur_w = new_w
                    n_rebal += 1
            r = float(np.nansum(cur_w.to_numpy() * np.nan_to_num(R.loc[d].to_numpy())))
            equity *= (1.0 + r)
            nav.iloc[i] = equity
        navs[vname] = nav
        m = perf(nav)
        m.update({"variant": vname, "n_names": len(etfs),
                  "turnover_per_rebal": turnover / max(n_rebal, 1)})
        rows.append(m)
    return pd.DataFrame(rows).set_index("variant"), navs


def run_tilt_backtest(
    sleeve: str,
    uni: pd.DataFrame,
    panel: dict[str, pd.DataFrame],
    start: str,
    rebalance: str = "W-FRI",
) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    df = uni[uni["sleeve"] == sleeve].copy().reset_index(drop=True)
    df = df[df["ETF"].isin(panel.keys())].reset_index(drop=True)
    if df.empty:
        return pd.DataFrame(), {}

    borrow_on_etf = sleeve in ("inverse_decay_bucket4", "volatility_etp_bucket5")

    # Per-pair daily returns + point-in-time tr_est, aligned to a common calendar.
    daily_ret: dict[str, pd.Series] = {}
    tr_est: dict[str, pd.Series] = {}
    for _, row in df.iterrows():
        etf = row["ETF"]
        px = panel[etf]
        dr = pair_daily_returns(row, px, borrow_on_etf)
        if dr.empty:
            continue
        daily_ret[etf] = dr
        tr_est[etf] = tr_est_series(px["b_px"])
    df = df[df["ETF"].isin(daily_ret.keys())].reset_index(drop=True)
    if df.empty:
        return pd.DataFrame(), {}

    # Base (un-tilted) decay-score weights from the production weighting block.
    weighting = load_weighting(sleeve)
    base_cfg = copy.deepcopy(weighting)
    base_cfg["trend_percentile_multiplier"] = {"enabled": False}
    base_w = pd.Series(_decay_score_weights(df, base_cfg, sleeve_name=sleeve), index=df["ETF"].to_numpy())

    # Calendar = union of trading days from start.
    all_idx = sorted(set().union(*[set(s.index) for s in daily_ret.values()]))
    cal = pd.DatetimeIndex([d for d in all_idx if d >= pd.Timestamp(start)])
    rebal_days = pd.DatetimeIndex(
        pd.Series(1, index=cal).resample(rebalance).last().index
    ).intersection(cal)
    if len(rebal_days) < 8:
        return pd.DataFrame(), {}

    etfs = df["ETF"].to_numpy()
    # Daily return matrix and as-of tr_est matrix on the calendar.
    R = pd.DataFrame({e: daily_ret[e].reindex(cal) for e in etfs})
    TR = pd.DataFrame({e: tr_est[e].reindex(cal).ffill() for e in etfs})
    res, navs = portfolio_tilt_sim(etfs, R, TR, base_w, cal, rebal_days)
    res = res.assign(sleeve=sleeve)
    return res, navs


def run_tilt_backtest_b4_dynamic(uni, panel, start, rebalance="W-FRI"):
    """B4 sizing-tilt test using the SOLVENT dynamic engine for per-pair returns.

    Per-pair daily returns come from ``run_bucket4_backtest_dynamic_h`` (production
    cadence + v7 hedge), so the short book cannot spuriously cross zero. Only the
    cross-sectional tilt weighting then varies.
    """
    blk = knobs_from_yaml()
    prod_knobs = make_knobs(blk)
    df = uni[uni["sleeve"].isin(["inverse_decay_bucket4", "volatility_etp_bucket5"])].copy()
    df = df[df["ETF"].isin(panel.keys())].reset_index(drop=True)
    if df.empty:
        return pd.DataFrame(), {}

    daily_ret, tr_est = {}, {}
    keep = []
    for _, row in df.iterrows():
        etf, und = row["ETF"], row["Underlying"]
        px = panel[etf]
        cal = pd.DatetimeIndex([d for d in px.index if d >= pd.Timestamp(start)])
        if len(cal) < 120:
            continue
        sig = get_pair_signal(etf, und, cal, history={}, underlying_prices=px["b_px"],
                              window=60, lookahead_shift=1)
        h_daily = build_h_series(sig, cal, knobs=prod_knobs)
        rb, _ = build_rebal_dates(sig, cal, knobs=prod_knobs, warmup_bdays=60)
        borrow = float(pd.to_numeric(row.get("borrow_current"), errors="coerce") or 0.0)
        bt = run_bucket4_backtest_dynamic_h(
            px.reindex(cal), h_daily, rb, initial_capital=1_000_000.0,
            beta_a=-abs(float(row["Delta"])), beta_b=1.0,
            borrow_a_annual=borrow, slippage_bps=20.0,
        )
        daily_ret[etf] = bt["ret"]
        tr_est[etf] = tr_est_series(px["b_px"])
        keep.append(etf)
    df = df[df["ETF"].isin(keep)].reset_index(drop=True)
    if df.empty:
        return pd.DataFrame(), {}

    weighting = load_weighting("inverse_decay_bucket4")
    base_cfg = copy.deepcopy(weighting)
    base_cfg["trend_percentile_multiplier"] = {"enabled": False}
    base_w = pd.Series(_decay_score_weights(df, base_cfg, sleeve_name="inverse_decay_bucket4"),
                       index=df["ETF"].to_numpy())

    all_idx = sorted(set().union(*[set(s.index) for s in daily_ret.values()]))
    cal = pd.DatetimeIndex([d for d in all_idx if d >= pd.Timestamp(start)])
    rebal_days = pd.DatetimeIndex(pd.Series(1, index=cal).resample(rebalance).last().index).intersection(cal)
    etfs = df["ETF"].to_numpy()
    R = pd.DataFrame({e: daily_ret[e].reindex(cal) for e in etfs})
    TR = pd.DataFrame({e: tr_est[e].reindex(cal).ffill() for e in etfs})
    res, navs = portfolio_tilt_sim(etfs, R, TR, base_w, cal, rebal_days)
    res = res.assign(sleeve="inverse_decay_bucket4_dynamic")
    return res, navs


# ---------------------------------------------------------------------------
# #3 Cadence backtest (B4, production engine; only rebalance schedule varies)
# ---------------------------------------------------------------------------
def knobs_from_yaml() -> dict:
    raw = yaml.safe_load((REPO / "config/strategy_config.yml").read_text())
    blk = (raw["portfolio"]["sleeves"]["inverse_decay_bucket4"]["rules"]
           ["bucket4_weekly_opt2"]["hedge_cadence_policy"])
    return blk


def make_knobs(base_blk: dict, **over) -> HedgeCadenceKnobs:
    fields = {f: base_blk[f] for f in HedgeCadenceKnobs.__dataclass_fields__ if f in base_blk}
    fields.update(over)
    return HedgeCadenceKnobs(**fields)


CADENCE_VARIANTS = {
    # name -> (cadence_signal_col, k_tr)  [base_days, m_vcr held at production]
    "prod_trEst_k+2.25": ("tr_est", 2.25),
    "contrarian_k-2.25": ("tr_est", -2.25),
    "contrarian_k-1.0": ("tr_est", -1.0),
    "TRagnostic_k0": ("tr_est", 0.0),
    "rawTR_k+2.25": ("tr", 2.25),
    "rawTR_contrarian_k-2.25": ("tr", -2.25),
}


def run_cadence_backtest(uni: pd.DataFrame, panel: dict[str, pd.DataFrame], start: str
                         ) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    blk = knobs_from_yaml()
    df = uni[uni["sleeve"].isin(["inverse_decay_bucket4", "volatility_etp_bucket5"])].copy()
    df = df[df["ETF"].isin(panel.keys())].reset_index(drop=True)
    if df.empty:
        return pd.DataFrame(), {}

    gross = pd.to_numeric(df["gross_target_usd"], errors="coerce").to_numpy()
    wts = gross / gross.sum()

    # Pre-compute per-pair signal + a FIXED production hedge series (shared across variants).
    pair_ctx = []
    prod_knobs = make_knobs(blk)
    for (_, row), wt in zip(df.iterrows(), wts):
        etf, und = row["ETF"], row["Underlying"]
        px = panel[etf]
        cal = pd.DatetimeIndex([d for d in px.index if d >= pd.Timestamp(start)])
        if len(cal) < 120:
            continue
        sig = get_pair_signal(etf, und, cal, history={}, underlying_prices=px["b_px"],
                              window=60, lookahead_shift=1)
        h_daily = build_h_series(sig, cal, knobs=prod_knobs)
        borrow = float(pd.to_numeric(row.get("borrow_current"), errors="coerce") or 0.0)
        pair_ctx.append({
            "etf": etf, "px": px.reindex(cal), "sig": sig, "h": h_daily,
            "beta_a": float(row["Delta"]), "borrow": borrow, "w": float(wt), "cal": cal,
        })
    if not pair_ctx:
        return pd.DataFrame(), {}

    rows, navs = [], {}
    for vname, (sig_col, k_tr) in CADENCE_VARIANTS.items():
        knobs = make_knobs(blk, cadence_signal_col=sig_col, k_tr=float(k_tr))
        pair_navs, intervals = [], []
        for ctx in pair_ctx:
            rb, diag = build_rebal_dates(ctx["sig"], ctx["cal"], knobs=knobs, warmup_bdays=60)
            bt = run_bucket4_backtest_dynamic_h(
                ctx["px"], ctx["h"], rb,
                initial_capital=ctx["w"] * 1_000_000.0,
                gross_multiplier=1.0, beta_a=-abs(ctx["beta_a"]), beta_b=1.0,
                borrow_a_annual=ctx["borrow"], slippage_bps=20.0,
            )
            pair_navs.append(bt["equity"])
            if "interval_days" in diag:
                intervals.append(float(pd.to_numeric(diag["interval_days"], errors="coerce").mean()))
        port = pd.concat(pair_navs, axis=1, sort=True).ffill().sum(axis=1)
        navs[vname] = port
        m = perf(port)
        m.update({"variant": vname, "mean_interval_days": float(np.nanmean(intervals)) if intervals else np.nan,
                  "n_pairs": len(pair_navs)})
        rows.append(m)
    return pd.DataFrame(rows).set_index("variant"), navs


# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-date", default="2026-06-25")
    ap.add_argument("--start", default="2024-01-01")
    ap.add_argument("--outdir", type=Path, default=REPO / "notebooks/output/sizing_tilt_cadence_bt")
    args = ap.parse_args(argv)
    args.outdir.mkdir(parents=True, exist_ok=True)

    print("[bt] loading universe + prices ...")
    uni = load_universe(args.run_date)
    panel = load_price_panel(args.run_date)
    print(f"[bt] selected names: {len(uni)}  priced ETFs: {len(panel)}")
    for s, g in uni.groupby("sleeve"):
        print(f"      {s}: {len(g)}")

    # ---- Sizing tilt (B): B1 (core) primary, B4 secondary (dynamic engine) ----
    all_tilt = []
    print("[bt] tilt backtest: core_leveraged ...")
    res_b1, _ = run_tilt_backtest("core_leveraged", uni, panel, args.start)
    if not res_b1.empty:
        res_b1.to_csv(args.outdir / "tilt_core_leveraged.csv")
        all_tilt.append(res_b1)
        print(res_b1[["cagr", "vol", "sharpe", "maxdd", "n_names"]].round(4).to_string())

    print("[bt] tilt backtest: inverse_decay_bucket4 (dynamic engine) ...")
    res_b4, _ = run_tilt_backtest_b4_dynamic(uni, panel, args.start)
    if not res_b4.empty:
        res_b4.to_csv(args.outdir / "tilt_inverse_decay_bucket4.csv")
        all_tilt.append(res_b4)
        print(res_b4[["cagr", "vol", "sharpe", "maxdd", "n_names"]].round(4).to_string())
    if all_tilt:
        pd.concat(all_tilt, sort=False).to_csv(args.outdir / "tilt_all.csv")

    # ---- Cadence (#3): B4 ----
    print("[bt] cadence backtest (B4) ...")
    cad, cad_navs = run_cadence_backtest(uni, panel, args.start)
    if not cad.empty:
        cad.to_csv(args.outdir / "cadence_b4.csv")
        print(cad[["cagr", "vol", "sharpe", "maxdd", "mean_interval_days", "n_pairs"]].round(4).to_string())

    print(f"[bt] DONE -> {args.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
