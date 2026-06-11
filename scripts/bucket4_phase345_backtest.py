"""Bucket 4 Phases 3-5 backtest lab: hedge model, cadence trigger, portfolio construction.

Three sequential stages, each scored on the same equal-weight metrics so results
are comparable and the best configuration per stage feeds the next:

  Stage A -- HEDGE MODEL (WS2)
    h source        : fixed 0.75  vs  v7 closed-form  h = clip(h_mid + k_vcr*(VCR-VCR_med))
    beta            : static screener |beta|  vs  realized 20d regression beta (shifted)
    h hysteresis    : 0 (off) vs 0.05 (h target only moves when it shifts > 5pts)
    regime overlay  : 0 (off) vs +0.10 toward h_max when trailing rv percentile > 0.8,
                      sag toward h_mid when < 0.3 (proxy for the book-level overlay)

  Stage B -- CADENCE TRIGGER (WS3), using the best Stage A hedge config
    clock           : production TR/VCR continuous interval (base 10, cap 21)
    drift+clock     : daily schedule, trade only when leg-share drift > thr
                      (thr in {4,6,8,12}%), forced after {10,21} trading days

  Stage C -- PORTFOLIO (WS4/WS5), using best Stage A+B per-pair equity curves
    selection       : all pairs vs top-N by (net_edge - borrow) / underlying vol
    weights         : equal vs Kelly-lite (trailing 63bd realized return / var),
                      blend lambda in {0, 0.5, 1.0}, max name weight 0.20
    cluster cap     : crypto cluster capped at 35% of sleeve gross (on/off)

Outputs (notebooks/output/):
  b4_phase345_stageA.csv / stageB.csv / stageC.csv
  b4_phase345_summary.md

Usage:
  python -m scripts.bucket4_phase345_backtest [--max-pairs 0] [--start 2025-10-07]
"""
from __future__ import annotations

import argparse
import sys
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
BASE_DAYS = 10.0
K_TR = 2.25
M_VCR = 2.5
H_FIXED = 0.75
H_MID, H_MIN, H_MAX, K_VCR = 0.55, 0.30, 0.80, 1.0
EMA_ALPHA = 0.25

CRYPTO_CLUSTER = {"MSTR", "COIN", "IBIT", "CLSK", "IREN", "BMNR", "CRCL",
                  "RIOT", "MARA", "GBTC", "ETHA", "BITX", "XRPZ", "BLSH"}


# ---------------------------------------------------------------------------
# Pair data prep
# ---------------------------------------------------------------------------
def load_metrics_filtered(path: Path, tickers: set[str]) -> pd.DataFrame:
    """Chunked variant of the sweep's load_metrics (full file can OOM the C parser)."""
    usecols = ["date", "ticker", "close_price", "nav", "etf_adj_close", "underlying_adj_close"]
    chunks = []
    for chunk in pd.read_csv(path, usecols=usecols, low_memory=False, chunksize=500_000):
        chunk["ticker"] = chunk["ticker"].map(norm_sym)
        chunks.append(chunk[chunk["ticker"].isin(tickers)])
    df = pd.concat(chunks, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for c in ("close_price", "nav", "etf_adj_close", "underlying_adj_close"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    etf = df["etf_adj_close"].where(df["etf_adj_close"] > 0)
    etf = etf.fillna(df["close_price"].where(df["close_price"] > 0))
    etf = etf.fillna(df["nav"].where(df["nav"] > 0))
    df["etf_px"] = etf
    return df


def load_pair_data(args) -> tuple[list[dict], pd.DataFrame]:
    pairs_csv = args.pairs
    if pairs_csv is None:
        cands = sorted((REPO / "data/runs").glob("*/accounting/bucket4_pairs.csv"))
        pairs_csv = cands[-1] if cands else None
    if pairs_csv is None:
        raise SystemExit("no bucket4_pairs.csv found")
    pairs = pd.read_csv(pairs_csv)
    pairs["etf"] = pairs["etf"].map(norm_sym)
    pairs["underlying"] = pairs["underlying"].map(norm_sym)

    metrics = load_metrics_filtered(args.metrics, set(pairs["etf"]))
    vs_hist = load_vol_shape_history(args.vol_shape)
    screened = pd.read_csv(args.screened)
    screened["ETF"] = screened["ETF"].map(norm_sym)
    screened["Underlying"] = screened["Underlying"].map(norm_sym)
    scr = screened.set_index("ETF")

    start = pd.Timestamp(args.start)
    keys = list(zip(pairs["etf"], pairs["underlying"], pairs["delta"], pairs["partial_hedge_ratio"]))
    if args.max_pairs > 0:
        keys = keys[: args.max_pairs]

    out: list[dict] = []
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
        beta_static = abs(float(delta))
        # realized 20d regression beta of ETF returns on underlying returns (shifted 1d)
        r_etf = prices["a_px"].pct_change()
        r_und = prices["b_px"].pct_change()
        cov = r_etf.rolling(20).cov(r_und)
        var = r_und.rolling(20).var()
        beta_real = (cov / var).abs().shift(1)
        beta_real = beta_real.clip(lower=0.5 * beta_static, upper=1.5 * beta_static)
        beta_real = beta_real.fillna(beta_static)

        # trailing rv percentile (expanding rank of rv_daily) as the regime proxy
        rv = sig.get("rv_daily")
        if rv is None or rv.dropna().empty:
            rv = r_und.rolling(20).std() * np.sqrt(TRADING_DAYS)
        rv_pct = rv.expanding(min_periods=20).rank(pct=True)

        row = scr.loc[etf] if etf in scr.index else None
        out.append({
            "pair": f"{etf}/{und}",
            "etf": etf, "underlying": und,
            "prices": prices, "sig": sig,
            "beta_static": beta_static,
            "beta_real": beta_real,
            "rv_pct": rv_pct,
            "partial_h": float(ph) if pd.notna(ph) else H_FIXED,
            "borrow_a": float(row.get("borrow_current", 0.0)) if row is not None and pd.notna(row.get("borrow_current", np.nan)) else 0.0,
            "net_edge": float(row.get("bucket4_net_edge_annual", np.nan)) if row is not None else np.nan,
            "und_vol": float(row.get("vol_underlying_annual", np.nan)) if row is not None else np.nan,
        })
    return out, screened


# ---------------------------------------------------------------------------
# Hedge model variants (Stage A)
# ---------------------------------------------------------------------------
def build_h_series(
    pd_: dict,
    *,
    h_model: str,           # "fixed" | "v7"
    beta_mode: str,         # "static" | "realized"
    hyst: float,            # 0 = off
    regime_bump: float,     # 0 = off
) -> pd.Series:
    cal = pd_["prices"].index
    sig = pd_["sig"]
    if h_model == "fixed":
        h = pd.Series(float(pd_["partial_h"]), index=cal)
    else:
        vcr = pd.to_numeric(sig.get("vcr"), errors="coerce")
        vcr_med = pd.to_numeric(sig.get("vcr_med"), errors="coerce")
        h = (H_MID + K_VCR * (vcr - vcr_med)).clip(H_MIN, H_MAX)
        h = h.fillna(float(pd_["partial_h"]))
        if EMA_ALPHA > 0:
            h = h.ewm(alpha=EMA_ALPHA, adjust=False).mean()

    if regime_bump > 0:
        pct = pd_["rv_pct"].reindex(cal).ffill()
        hot = (pct >= 0.8).fillna(False)
        cold = (pct <= 0.3).fillna(False)
        h = h.where(~hot, (h + regime_bump).clip(upper=H_MAX))
        h = h.where(~cold, h + (H_MID - h) * 0.5)

    if beta_mode == "realized":
        # engine uses static |beta|; emulate time-varying beta via h scaling
        scale = (pd_["beta_real"].reindex(cal).ffill() / pd_["beta_static"]).clip(0.5, 1.5)
        h = (h * scale).clip(0.15, 1.0)

    if hyst > 0:
        vals = h.to_numpy(copy=True)
        applied = vals[0]
        for i in range(len(vals)):
            if np.isfinite(vals[i]) and abs(vals[i] - applied) > hyst:
                applied = vals[i]
            vals[i] = applied
        h = pd.Series(vals, index=cal)
    return h


def run_pair(
    pd_: dict,
    h: pd.Series,
    rebal_dates: pd.DatetimeIndex,
    *,
    drift_thr: float | None = None,
    clock_floor: int | None = None,
) -> dict | None:
    prices = pd_["prices"]
    rd = pd.DatetimeIndex(rebal_dates).intersection(prices.index)
    if len(rd) == 0:
        rd = pd.DatetimeIndex([prices.index[0]])
    try:
        bt = run_bucket4_backtest_dynamic_h(
            prices, h, rd,
            beta_a=-pd_["beta_static"],
            beta_b=1.0,
            borrow_a_annual=pd_["borrow_a"],
            fee_bps=FEE_BPS,
            slippage_bps=SLIPPAGE_BPS,
            opt2_h_base=float(pd_["partial_h"]),
            drift_threshold_share_of_gross=drift_thr,
            force_rebalance_after_days=clock_floor,
        )
    except Exception:
        return None
    if bt is None or bt.empty:
        return None
    m = pair_metrics(bt)
    m["pair"] = pd_["pair"]
    m["n_trades"] = int(bt["rebalance"].astype(bool).sum())
    m["equity_curve"] = bt["equity"]
    return m


def production_schedule(pd_: dict) -> pd.DatetimeIndex:
    rd, _ = policy_continuous_interval(
        pd_["prices"].index, pd_["sig"],
        base_days=BASE_DAYS, k_tr=K_TR, m_vcr=M_VCR,
        min_interval=1, max_interval=CAP_DAYS,
    )
    return rd


def ew_score(rows: list[dict], label: dict) -> dict | None:
    if not rows:
        return None
    df = pd.DataFrame([{k: v for k, v in r.items() if k != "equity_curve"} for r in rows])
    cagr = df["CAGR"].dropna()
    if cagr.empty:
        return None
    lo, hi = cagr.quantile(0.05), cagr.quantile(0.95)
    return {
        **label,
        "n_pairs": int(len(df)),
        "ew_mean_cagr": float(cagr.mean()),
        "ew_median_cagr": float(cagr.median()),
        "winsor_mean_cagr": float(cagr.clip(lo, hi).mean()),
        "ew_mean_vol": float(df["vol"].mean(skipna=True)),
        "ew_mean_max_dd": float(df["max_dd"].mean(skipna=True)),
        "mean_trades_per_pair": float(df["n_trades"].mean(skipna=True)),
        "total_slippage": float(df["slippage_paid"].sum(skipna=True)),
        "total_borrow": float(df["borrow_paid"].sum(skipna=True)),
    }


def rank_composite(df: pd.DataFrame) -> pd.DataFrame:
    """Rank on return and RISK-ADJUSTED return (ret/vol, calmar) -- raw vol/dd only
    enter through the ratios, so a config that loses money slowly can't outrank one
    that makes money with moderate vol."""
    s = df.copy()
    s["ret_over_vol"] = s["ew_mean_cagr"] / s["ew_mean_vol"].clip(lower=0.01)
    s["calmar_ew"] = s["ew_mean_cagr"] / s["ew_mean_max_dd"].abs().clip(lower=0.01)
    for col in ("winsor_mean_cagr", "ew_median_cagr", "ret_over_vol", "calmar_ew"):
        s[f"rank_{col}"] = s[col].rank(ascending=False)
    s["composite_rank"] = s[[c for c in s.columns if c.startswith("rank_")]].mean(axis=1)
    return s.sort_values("composite_rank").drop(columns=[c for c in s.columns if c.startswith("rank_")])


# ---------------------------------------------------------------------------
# Stage C: portfolio construction on per-pair equity curves
# ---------------------------------------------------------------------------
def portfolio_metrics(port_ret: pd.Series, label: dict) -> dict:
    nav = (1 + port_ret.fillna(0.0)).cumprod()
    span_days = max(int((nav.index[-1] - nav.index[0]).days), 1)
    years = max(span_days / 365.25, 1 / 365.25)
    cagr = float(nav.iloc[-1] ** (1 / years) - 1)
    vol = float(port_ret.std(ddof=1) * np.sqrt(TRADING_DAYS))
    dd = float((nav / nav.cummax() - 1).min())
    return {**label, "cagr": cagr, "vol": vol, "max_dd": dd,
            "calmar": cagr / abs(dd) if dd < -1e-9 else np.nan}


def kelly_lite_weights(
    rets: pd.DataFrame,
    *,
    lam: float,
    lookback: int = 63,
    rebal_every: int = 21,
    max_w: float = 0.20,
    cluster_map: dict[str, str] | None = None,
    cluster_cap: float | None = None,
) -> pd.DataFrame:
    """Walk-forward weights: lam * kelly-lite + (1-lam) * equal, monthly refresh."""
    cols = rets.columns
    n = len(cols)
    w_eq = np.ones(n) / n
    weights = pd.DataFrame(index=rets.index, columns=cols, dtype=float)
    current = w_eq.copy()
    for i, dt in enumerate(rets.index):
        if i % rebal_every == 0 and i >= lookback:
            window = rets.iloc[i - lookback:i]
            mu = window.mean() * TRADING_DAYS          # annualized realized return
            var = (window.std(ddof=1) ** 2).clip(lower=1e-4) * TRADING_DAYS
            score = (mu.clip(lower=0.0) / var).to_numpy()
            if score.sum() > 1e-9:
                w_k = score / score.sum()
            else:
                w_k = w_eq.copy()
            w = lam * w_k + (1 - lam) * w_eq
            w = np.minimum(w, max_w)
            w = w / w.sum()
            if cluster_cap is not None and cluster_map:
                in_cl = np.array([cluster_map.get(c, "") == "crypto" for c in cols])
                cl_w = w[in_cl].sum()
                if cl_w > cluster_cap:
                    w[in_cl] *= cluster_cap / cl_w
                    out_w = w[~in_cl].sum()
                    if out_w > 1e-9:
                        w[~in_cl] *= (1 - cluster_cap) / out_w
            current = w
        weights.iloc[i] = current
    return weights


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Bucket 4 Phase 3-5 backtest lab.")
    ap.add_argument("--pairs", type=Path, default=None)
    ap.add_argument("--metrics", type=Path,
                    default=REPO.parent / "Levered ETFs/etf-dashboard/data/etf_metrics_daily.csv")
    ap.add_argument("--vol-shape", type=Path,
                    default=REPO.parent / "Levered ETFs/etf-dashboard/data/vol_shape_history.json")
    ap.add_argument("--screened", type=Path, default=REPO / "data/etf_screened_today.csv")
    ap.add_argument("--outdir", type=Path, default=REPO / "notebooks/output")
    ap.add_argument("--start", default="2025-10-07")
    ap.add_argument("--max-pairs", type=int, default=0)
    args = ap.parse_args(argv)
    args.outdir.mkdir(parents=True, exist_ok=True)

    pair_data, screened = load_pair_data(args)
    print(f"[phase345] pairs with data: {len(pair_data)}")
    if not pair_data:
        return 1

    # Pre-compute the production cadence schedule once per pair
    for pd_ in pair_data:
        pd_["sched_prod"] = production_schedule(pd_)

    # ---------------- Stage A: hedge model grid ----------------
    print("\n=== Stage A: hedge model (WS2) ===")
    grid_a = []
    for h_model in ("fixed", "v7"):
        for beta_mode in ("static", "realized"):
            for hyst in (0.0, 0.05):
                for bump in (0.0, 0.10):
                    grid_a.append({"h_model": h_model, "beta_mode": beta_mode,
                                   "h_hyst": hyst, "regime_bump": bump})
    rows_a = []
    stageA_curves: dict[tuple, list[dict]] = {}
    for gi, g in enumerate(grid_a, 1):
        pair_rows = []
        for pd_ in pair_data:
            h = build_h_series(pd_, h_model=g["h_model"], beta_mode=g["beta_mode"],
                               hyst=g["h_hyst"], regime_bump=g["regime_bump"])
            m = run_pair(pd_, h, pd_["sched_prod"])
            if m:
                pair_rows.append(m)
        sc = ew_score(pair_rows, g)
        if sc:
            rows_a.append(sc)
            stageA_curves[tuple(g.values())] = pair_rows
        print(f"  A {gi}/{len(grid_a)} {g} done", flush=True)
    dfa = rank_composite(pd.DataFrame(rows_a))
    dfa.to_csv(args.outdir / "b4_phase345_stageA.csv", index=False)
    best_a = dfa.iloc[0][["h_model", "beta_mode", "h_hyst", "regime_bump"]].to_dict()
    print(dfa.head(8).to_string(index=False))
    print(f"[phase345] best hedge config: {best_a}")

    # ---------------- Stage B: cadence trigger ----------------
    print("\n=== Stage B: cadence trigger (WS3) ===")
    rows_b = []
    stageB_curves: dict[str, list[dict]] = {}

    def run_cadence(tag: str, *, daily: bool, drift: float | None, floor: int | None):
        pair_rows = []
        for pd_ in pair_data:
            h = build_h_series(pd_, h_model=best_a["h_model"], beta_mode=best_a["beta_mode"],
                               hyst=float(best_a["h_hyst"]), regime_bump=float(best_a["regime_bump"]))
            sched = pd_["prices"].index if daily else pd_["sched_prod"]
            m = run_pair(pd_, h, sched, drift_thr=drift, clock_floor=floor)
            if m:
                pair_rows.append(m)
        sc = ew_score(pair_rows, {"cadence": tag, "drift_thr": drift if drift is not None else np.nan,
                                  "clock_floor": floor if floor is not None else np.nan})
        if sc:
            rows_b.append(sc)
            stageB_curves[tag] = pair_rows
        print(f"  B {tag} done", flush=True)

    run_cadence("clock_prod_base10", daily=False, drift=None, floor=None)
    for thr in (0.04, 0.06, 0.08, 0.12):
        for floor in (10, 21):
            run_cadence(f"drift{int(thr*100)}_floor{floor}", daily=True, drift=thr, floor=floor)
    dfb = rank_composite(pd.DataFrame(rows_b))
    dfb.to_csv(args.outdir / "b4_phase345_stageB.csv", index=False)
    best_b_tag = str(dfb.iloc[0]["cadence"])
    print(dfb.to_string(index=False))
    print(f"[phase345] best cadence: {best_b_tag}")

    # ---------------- Stage C: portfolio construction ----------------
    print("\n=== Stage C: portfolio (WS4/WS5) ===")
    best_rows = stageB_curves[best_b_tag]
    rets = {}
    for r in best_rows:
        eq = r["equity_curve"]
        rets[r["pair"]] = eq.pct_change()
    rmat = pd.DataFrame(rets).dropna(how="all")
    # static selection score from the screener: (net_edge - borrow) / und vol
    meta = {p["pair"]: p for p in pair_data}
    sel_score = {}
    for pair in rmat.columns:
        p = meta.get(pair)
        if p is None:
            continue
        edge = p["net_edge"] if np.isfinite(p["net_edge"]) else 0.0
        volu = p["und_vol"] if np.isfinite(p["und_vol"]) and p["und_vol"] > 0.05 else 0.6
        sel_score[pair] = (edge - p["borrow_a"]) / volu
    ranked_pairs = sorted(sel_score, key=sel_score.get, reverse=True)
    cluster_map = {pair: ("crypto" if pair.split("/")[1] in CRYPTO_CLUSTER else "")
                   for pair in rmat.columns}

    rows_c = []
    for n_sel in (len(ranked_pairs), 30, 20, 15, 10):
        chosen = ranked_pairs[: int(n_sel)]
        sub = rmat[chosen].fillna(0.0)
        for lam in (0.0, 0.5, 1.0):
            for cap in (None, 0.35):
                w = kelly_lite_weights(sub, lam=lam, cluster_map=cluster_map, cluster_cap=cap)
                port = (sub * w).sum(axis=1)
                rows_c.append(portfolio_metrics(port, {
                    "n_pairs": int(n_sel), "kelly_lambda": lam,
                    "cluster_cap": cap if cap is not None else np.nan,
                }))
        print(f"  C n={n_sel} done", flush=True)
    dfc = pd.DataFrame(rows_c)
    dfc["rank_cagr"] = dfc["cagr"].rank(ascending=False)
    dfc["rank_calmar"] = dfc["calmar"].rank(ascending=False)
    dfc["composite_rank"] = dfc[["rank_cagr", "rank_calmar"]].mean(axis=1)
    dfc = dfc.sort_values("composite_rank").drop(columns=["rank_cagr", "rank_calmar"])
    dfc.to_csv(args.outdir / "b4_phase345_stageC.csv", index=False)
    print(dfc.head(12).to_string(index=False))

    # ---------------- Summary ----------------
    def block(df: pd.DataFrame) -> str:
        return "```\n" + df.to_string(index=False) + "\n```"

    md = ["# Bucket 4 Phase 3-5 backtest results\n",
          f"Window start: {args.start} | pairs: {len(pair_data)} | slippage: {SLIPPAGE_BPS}bps\n",
          "## Stage A -- hedge model (top 5)\n",
          block(dfa.head(5)),
          f"\n**Best:** `{best_a}`\n",
          "\n## Stage B -- cadence (all)\n",
          block(dfb),
          f"\n**Best:** `{best_b_tag}`\n",
          "\n## Stage C -- portfolio (top 10)\n",
          block(dfc.head(10)),
          "\n"]
    (args.outdir / "b4_phase345_summary.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\n[phase345] wrote {args.outdir / 'b4_phase345_summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
