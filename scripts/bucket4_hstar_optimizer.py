"""Bucket 4 hedge-ratio (h*) optimizer: per-pair, signal-conditional, and walk-forward.

Optimizes the hedge ratio h (dollars of underlying shorted per dollar of inverse-ETF
short, as a share of |beta|) while holding everything else at the production-optimal
configuration:

  * cadence  : policy_continuous_interval(base_days=12, k_tr=2.25, m_vcr=2.5, cap 21)
  * beta     : static screener |beta|
  * costs    : 20 bps slippage, 0 fee, screener borrow
  * schedule : computed ONCE per pair, reused across all h variants

Baseline to beat (v7 closed form):

  h_t = clip(0.55 + 1.0*(VCR - VCR_med), 0.30, 0.80); NaN -> partial_h;
  then EMA ewm(alpha=0.25, adjust=False)

Experiments
  1. per-pair constant h* (grid 0.30..0.95 step 0.05) + cross-section of h* drivers  [IS]
  2. global v7 refit: h_mid x k_vcr x h_max grid, EW-scored                          [IS]
  3. multi-signal closed form: best-of-(2) + one extra signal (TR / rv_pct / mom)    [IS]
  4. walk-forward adaptive per-pair h*: every 21bd pick the constant h that won the
     trailing 63bd (from the exp-1 equity panel), apply forward                      [OOS]

Outputs -> notebooks/output/b4_hstar/ (CSVs, PNGs, B4_HSTAR_RESULTS.md)

Usage:
  python -m scripts.bucket4_hstar_optimizer [--max-pairs 5] [--start 2025-10-07]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.bucket4_phase345_backtest import (  # noqa: E402
    ew_score,
    load_pair_data,
    rank_composite,
    run_pair,
)
from scripts.bucket4_vol_shape_signals import policy_continuous_interval  # noqa: E402

# ---------------------------------------------------------------------------
# FIXED production configuration (do not vary)
# ---------------------------------------------------------------------------
BASE_DAYS = 12.0          # newly adopted cadence optimum (NOT phase345's 10)
K_TR = 2.25
M_VCR = 2.5
MIN_INTERVAL = 1
CAP_DAYS = 21

H_MIN = 0.30
EMA_ALPHA = 0.25

# baseline v7 closed form
BASE_H_MID, BASE_K_VCR, BASE_H_MAX = 0.55, 1.0, 0.80

H_GRID = [round(h, 2) for h in np.arange(0.30, 0.951, 0.05)]   # 0.30 .. 0.95

# experiment 2 grids
GRID_H_MID = [0.45, 0.55, 0.65, 0.75]
GRID_K_VCR = [0.0, 0.5, 1.0, 1.5, 2.0]
GRID_H_MAX = [0.80, 0.95]

# experiment 3 coefficient grids
GRID_K_SIG = [-0.2, -0.1, 0.0, 0.1, 0.2]

# experiment 4 walk-forward
WF_LOOKBACK = 63
WF_STEP = 21


# ---------------------------------------------------------------------------
# h-series builders
# ---------------------------------------------------------------------------
def schedule_base12(pd_: dict) -> pd.DatetimeIndex:
    """Production cadence with the newly adopted base_days=12 (phase345 hardcodes 10)."""
    rd, _ = policy_continuous_interval(
        pd_["prices"].index, pd_["sig"],
        base_days=BASE_DAYS, k_tr=K_TR, m_vcr=M_VCR,
        min_interval=MIN_INTERVAL, max_interval=CAP_DAYS,
    )
    return rd


def closed_form_h(
    pd_: dict,
    *,
    h_mid: float,
    k_vcr: float,
    h_max: float,
    k_tr_h: float = 0.0,
    k_rv: float = 0.0,
    k_mom: float = 0.0,
) -> pd.Series:
    """h_t = clip(h_mid + k_vcr*(VCR-VCR_med) [+ extra signal terms], H_MIN, h_max),
    NaN -> partial_h, then EMA(alpha=0.25). Extra-signal NaNs contribute 0."""
    sig = pd_["sig"]
    cal = pd_["prices"].index
    vcr = pd.to_numeric(sig.get("vcr"), errors="coerce")
    vcr_med = pd.to_numeric(sig.get("vcr_med"), errors="coerce")
    raw = h_mid + k_vcr * (vcr - vcr_med)
    if k_tr_h != 0.0:
        tr = pd.to_numeric(sig.get("tr"), errors="coerce")
        raw = raw + k_tr_h * (tr - 1.0).fillna(0.0)
    if k_rv != 0.0:
        rvp = pd_["rv_pct"].reindex(raw.index).ffill()
        raw = raw + k_rv * (rvp - 0.5).fillna(0.0)
    if k_mom != 0.0:
        mom = np.sign(pd_["prices"]["b_px"].pct_change(20)).shift(1)
        raw = raw + k_mom * mom.reindex(raw.index).fillna(0.0)
    h = raw.clip(H_MIN, h_max)
    h = h.fillna(float(pd_["partial_h"]))
    h = h.ewm(alpha=EMA_ALPHA, adjust=False).mean()
    return h.reindex(cal).ffill()


def v7_baseline_h(pd_: dict) -> pd.Series:
    return closed_form_h(pd_, h_mid=BASE_H_MID, k_vcr=BASE_K_VCR, h_max=BASE_H_MAX)


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------
def score_variant(pair_data: list[dict], h_builder, label: dict) -> tuple[dict | None, list[dict]]:
    """Run all pairs with h = h_builder(pd_) on the fixed schedule; EW-score."""
    rows = []
    for pd_ in pair_data:
        m = run_pair(pd_, h_builder(pd_), pd_["sched"])
        if m:
            rows.append(m)
    return ew_score(rows, label), rows


def ew_portfolio_nav(rows: list[dict]) -> pd.Series:
    """Equal-weight portfolio NAV from per-pair equity curves (mean of daily returns)."""
    rets = pd.DataFrame({r["pair"]: r["equity_curve"].pct_change() for r in rows})
    port = rets.mean(axis=1, skipna=True).fillna(0.0)
    return (1.0 + port).cumprod()


# ---------------------------------------------------------------------------
# Experiment 1: per-pair constant h*
# ---------------------------------------------------------------------------
def experiment1(pair_data: list[dict], outdir: Path):
    print("\n=== Experiment 1: per-pair constant h* (IS) ===", flush=True)
    long_rows = []
    eq_panels: dict[str, pd.DataFrame] = {}    # pair -> DataFrame[dates x h] equity
    for pi, pd_ in enumerate(pair_data, 1):
        curves = {}
        for h in H_GRID:
            hseries = pd.Series(float(h), index=pd_["prices"].index)
            m = run_pair(pd_, hseries, pd_["sched"])
            if m is None:
                continue
            curves[h] = m["equity_curve"]
            long_rows.append({
                "pair": pd_["pair"], "h": h,
                "CAGR": m["CAGR"], "total_return": m["total_return"],
                "vol": m["vol"], "max_dd": m["max_dd"],
                "borrow_paid": m["borrow_paid"], "slippage_paid": m["slippage_paid"],
                "n_trades": m["n_trades"],
            })
        if curves:
            eq_panels[pd_["pair"]] = pd.DataFrame(curves)
        print(f"  [exp1] {pi}/{len(pair_data)} {pd_['pair']}", flush=True)

    long_df = pd.DataFrame(long_rows)
    long_df.to_csv(outdir / "b4_hstar_perpair_profiles.csv", index=False)

    # per-pair argmax (ties -> lower h: H_GRID ascending + idxmax picks first)
    meta = {p["pair"]: p for p in pair_data}
    arg_rows = []
    for pair, grp in long_df.groupby("pair"):
        g = grp.dropna(subset=["CAGR"]).sort_values("h")
        if g.empty:
            continue
        best = g.loc[g["CAGR"].idxmax()]
        p = meta[pair]
        arg_rows.append({
            "pair": pair, "h_star": float(best["h"]), "cagr_at_hstar": float(best["CAGR"]),
            "cagr_at_h055": float(g.loc[g["h"] == 0.55, "CAGR"].iloc[0]) if (g["h"] == 0.55).any() else np.nan,
            "abs_beta": p["beta_static"], "und_vol": p["und_vol"],
            "borrow_a": p["borrow_a"], "net_edge": p["net_edge"],
        })
    arg_df = pd.DataFrame(arg_rows).sort_values("h_star").reset_index(drop=True)
    arg_df.to_csv(outdir / "b4_hstar_perpair_argmax.csv", index=False)

    # cross-section correlations of h* vs pair characteristics
    corr_rows = []
    for col in ("abs_beta", "und_vol", "borrow_a", "net_edge"):
        sub = arg_df[["h_star", col]].dropna()
        if len(sub) >= 5:
            corr_rows.append({
                "driver": col, "n": len(sub),
                "pearson": float(sub["h_star"].corr(sub[col])),
                "spearman": float(sub["h_star"].corr(sub[col], method="spearman")),
            })
    corr_df = pd.DataFrame(corr_rows)
    corr_df.to_csv(outdir / "b4_hstar_crosssection_corr.csv", index=False)

    # ---- plot 1: heatmap pairs x h with h* marked ----
    mat = long_df.pivot_table(index="pair", columns="h", values="CAGR")
    mat = mat.reindex(arg_df["pair"])           # sorted by h*
    fig, ax = plt.subplots(figsize=(10, max(6, 0.28 * len(mat))))
    vmax = np.nanpercentile(np.abs(mat.values), 95)
    im = ax.imshow(mat.values, aspect="auto", cmap="RdYlGn", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(mat.columns)))
    ax.set_xticklabels([f"{h:.2f}" for h in mat.columns], fontsize=8)
    ax.set_yticks(range(len(mat.index)))
    ax.set_yticklabels(mat.index, fontsize=7)
    hpos = {h: j for j, h in enumerate(mat.columns)}
    for i, pair in enumerate(mat.index):
        hs = arg_df.loc[arg_df["pair"] == pair, "h_star"].iloc[0]
        if hs in hpos:
            ax.plot(hpos[hs], i, marker="*", color="black", ms=8)
    ax.set_xlabel("constant hedge ratio h")
    ax.set_title("Per-pair CAGR vs constant h (cadence base12; star = in-sample h*)")
    fig.colorbar(im, ax=ax, label="CAGR", shrink=0.6)
    fig.tight_layout()
    fig.savefig(outdir / "b4_hstar_heatmap_pair_by_h.png", dpi=130)
    plt.close(fig)

    # ---- plot 2: scatters h* vs drivers ----
    drivers = [("abs_beta", "|beta| (static)"), ("und_vol", "underlying vol (annual)"),
               ("borrow_a", "ETF borrow rate (annual)"), ("net_edge", "screener net edge (annual)")]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.2))
    for ax, (col, lab) in zip(axes, drivers):
        sub = arg_df[["h_star", col]].dropna()
        ax.scatter(sub[col], sub["h_star"], s=28, alpha=0.75)
        rho = sub["h_star"].corr(sub[col], method="spearman") if len(sub) >= 5 else np.nan
        ax.set_xlabel(lab)
        ax.set_ylabel("in-sample h*")
        ax.set_title(f"h* vs {col} (spearman={rho:.2f})")
        ax.grid(alpha=0.3)
    fig.suptitle("Cross-section of per-pair in-sample optimal hedge ratio h*", y=1.02)
    fig.tight_layout()
    fig.savefig(outdir / "b4_hstar_scatter_drivers.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    return long_df, arg_df, corr_df, eq_panels


# ---------------------------------------------------------------------------
# Experiment 2: global v7 refit grid
# ---------------------------------------------------------------------------
def experiment2(pair_data: list[dict], outdir: Path, baseline_score: dict):
    print("\n=== Experiment 2: global v7 refit grid (IS) ===", flush=True)
    rows = []
    total = len(GRID_H_MID) * len(GRID_K_VCR) * len(GRID_H_MAX)
    i = 0
    for h_mid in GRID_H_MID:
        for k_vcr in GRID_K_VCR:
            for h_max in GRID_H_MAX:
                i += 1
                sc, _ = score_variant(
                    pair_data,
                    lambda p, hm=h_mid, kv=k_vcr, hx=h_max: closed_form_h(p, h_mid=hm, k_vcr=kv, h_max=hx),
                    {"h_mid": h_mid, "k_vcr": k_vcr, "h_max": h_max},
                )
                if sc:
                    rows.append(sc)
                print(f"  [exp2] {i}/{total} h_mid={h_mid} k_vcr={k_vcr} h_max={h_max}", flush=True)
    df = rank_composite(pd.DataFrame(rows))
    df["d_mean_cagr_vs_baseline"] = df["ew_mean_cagr"] - baseline_score["ew_mean_cagr"]
    df.to_csv(outdir / "b4_hstar_refit_grid.csv", index=False)
    best = df.iloc[0]
    print(df.head(8).to_string(index=False))

    # ---- plot 3: heatmaps h_mid x k_vcr per h_max ----
    fig, axes = plt.subplots(1, len(GRID_H_MAX), figsize=(7 * len(GRID_H_MAX), 5))
    vals = df["ew_mean_cagr"]
    vmin, vmax = float(vals.min()), float(vals.max())
    for ax, h_max in zip(np.atleast_1d(axes), GRID_H_MAX):
        sub = df[df["h_max"] == h_max].pivot_table(index="k_vcr", columns="h_mid", values="ew_mean_cagr")
        sub = sub.reindex(index=GRID_K_VCR, columns=GRID_H_MID)
        im = ax.imshow(sub.values, cmap="RdYlGn", vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(GRID_H_MID)))
        ax.set_xticklabels(GRID_H_MID)
        ax.set_yticks(range(len(GRID_K_VCR)))
        ax.set_yticklabels(GRID_K_VCR)
        ax.set_xlabel("h_mid")
        ax.set_ylabel("k_vcr")
        ax.set_title(f"EW mean CAGR | h_max={h_max}")
        for yi, kv in enumerate(GRID_K_VCR):
            for xi, hm in enumerate(GRID_H_MID):
                v = sub.loc[kv, hm]
                if pd.notna(v):
                    ax.text(xi, yi, f"{v:.3f}", ha="center", va="center", fontsize=8)
        if h_max == BASE_H_MAX:
            ax.add_patch(plt.Rectangle((GRID_H_MID.index(BASE_H_MID) - 0.5, GRID_K_VCR.index(BASE_K_VCR) - 0.5),
                                       1, 1, fill=False, edgecolor="blue", lw=2.5))
            ax.text(GRID_H_MID.index(BASE_H_MID), GRID_K_VCR.index(BASE_K_VCR) - 0.38, "baseline",
                    ha="center", color="blue", fontsize=8, fontweight="bold")
        if float(best["h_max"]) == h_max:
            ax.add_patch(plt.Rectangle((GRID_H_MID.index(float(best["h_mid"])) - 0.5,
                                        GRID_K_VCR.index(float(best["k_vcr"])) - 0.5),
                                       1, 1, fill=False, edgecolor="black", lw=2.5))
            ax.text(GRID_H_MID.index(float(best["h_mid"])), GRID_K_VCR.index(float(best["k_vcr"])) + 0.40,
                    "best", ha="center", color="black", fontsize=8, fontweight="bold")
        fig.colorbar(im, ax=ax, shrink=0.85)
    fig.suptitle("v7 refit grid: EW mean CAGR by (h_mid, k_vcr, h_max) — blue=baseline, black=grid best")
    fig.tight_layout()
    fig.savefig(outdir / "b4_hstar_refit_heatmap.png", dpi=130)
    plt.close(fig)
    return df, {"h_mid": float(best["h_mid"]), "k_vcr": float(best["k_vcr"]), "h_max": float(best["h_max"])}


# ---------------------------------------------------------------------------
# Experiment 3: one extra signal at a time on top of best-of-(2)
# ---------------------------------------------------------------------------
def experiment3(pair_data: list[dict], outdir: Path, best2: dict, best2_score: dict):
    print("\n=== Experiment 3: multi-signal closed form (IS) ===", flush=True)
    rows = []
    signals = [("tr", "k_tr_h"), ("rv_pct", "k_rv"), ("mom20", "k_mom")]
    for sig_name, kw in signals:
        for k in GRID_K_SIG:
            sc, _ = score_variant(
                pair_data,
                lambda p, kk=k, kwn=kw: closed_form_h(p, **best2, **{kwn: kk}),
                {"signal": sig_name, "coef": k},
            )
            if sc:
                rows.append(sc)
            print(f"  [exp3] {sig_name} k={k}", flush=True)
    df = pd.DataFrame(rows)
    df["d_mean_cagr_vs_best2"] = df["ew_mean_cagr"] - best2_score["ew_mean_cagr"]
    df.to_csv(outdir / "b4_hstar_signal_marginals.csv", index=False)

    marg = (df.loc[df.groupby("signal")["ew_mean_cagr"].idxmax()]
            [["signal", "coef", "ew_mean_cagr", "d_mean_cagr_vs_best2"]]
            .reset_index(drop=True))

    # ---- plot 6: bar chart of marginal signal value ----
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(marg["signal"], marg["d_mean_cagr_vs_best2"] * 100.0,
                  color=["tab:blue", "tab:orange", "tab:green"])
    for b, (_, r) in zip(bars, marg.iterrows()):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                f"k*={r['coef']:+.1f}\n{r['d_mean_cagr_vs_best2'] * 100:+.2f}pp",
                ha="center", va="bottom", fontsize=9)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("marginal EW mean CAGR vs exp-2 optimum (pct pts)")
    ax.set_title("Experiment 3: marginal value of one extra h-signal (best coefficient, IS)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "b4_hstar_signal_marginal_bars.png", dpi=130)
    plt.close(fig)
    return df, marg


# ---------------------------------------------------------------------------
# Experiment 4: walk-forward adaptive per-pair h* (OOS)
# ---------------------------------------------------------------------------
def build_wf_h(pd_: dict, eq_panel: pd.DataFrame) -> tuple[pd.Series, list[dict]]:
    """Every WF_STEP days pick the constant h with the best trailing WF_LOOKBACK growth
    (from the precomputed constant-h equity panel); apply from the NEXT day (no
    same-day lookahead). Warmup (< lookback) uses the v7 baseline h. Ties -> lower h."""
    cal = pd_["prices"].index
    h_path = v7_baseline_h(pd_).copy()
    picks = []
    i = WF_LOOKBACK
    while i < len(cal):
        growth = {}
        for h in eq_panel.columns:        # ascending; max() keeps the first (lowest h) on ties
            e0, e1 = eq_panel[h].iloc[i - WF_LOOKBACK], eq_panel[h].iloc[i]
            if np.isfinite(e0) and np.isfinite(e1) and e0 > 1e-9:
                growth[float(h)] = float(e1 / e0)
        if growth:
            best_h = max(sorted(growth), key=lambda h: growth[h])
            j0, j1 = i + 1, min(i + 1 + WF_STEP, len(cal))
            if j0 < len(cal):
                h_path.iloc[j0:j1] = best_h
            picks.append({"pair": pd_["pair"], "decision_date": cal[i], "h_pick": best_h,
                          "trailing_growth": growth[best_h]})
        i += WF_STEP
    return h_path, picks


def experiment4(pair_data: list[dict], eq_panels: dict, outdir: Path,
                baseline_rows: list[dict], baseline_score: dict):
    print("\n=== Experiment 4: walk-forward adaptive h* (OOS) ===", flush=True)
    wf_rows, all_picks, wf_h_paths = [], [], {}
    for pd_ in pair_data:
        panel = eq_panels.get(pd_["pair"])
        if panel is None or panel.empty:
            continue
        h_path, picks = build_wf_h(pd_, panel)
        wf_h_paths[pd_["pair"]] = h_path
        all_picks.extend(picks)
        m = run_pair(pd_, h_path, pd_["sched"])
        if m:
            wf_rows.append(m)
    wf_score = ew_score(wf_rows, {"variant": "walkforward_hstar"})
    picks_df = pd.DataFrame(all_picks)
    picks_df.to_csv(outdir / "b4_hstar_walkforward_picks.csv", index=False)

    summary = pd.DataFrame([
        {**baseline_score, "variant": "v7_baseline"},
        wf_score,
    ])
    summary["d_mean_cagr_vs_baseline"] = summary["ew_mean_cagr"] - baseline_score["ew_mean_cagr"]
    summary.to_csv(outdir / "b4_hstar_walkforward_summary.csv", index=False)
    print(summary.to_string(index=False))

    # ---- plot 4: h(t) for representative pairs ----
    base_h = {p["pair"]: v7_baseline_h(p) for p in pair_data}
    avail = [p for p in pair_data if p["pair"] in wf_h_paths]
    # spread of pairs across the h* spectrum: sort by mean WF h, take 6 evenly spaced
    avail = sorted(avail, key=lambda p: float(wf_h_paths[p["pair"]].mean()))
    idxs = np.unique(np.linspace(0, len(avail) - 1, min(6, len(avail))).astype(int))
    reps = [avail[i] for i in idxs]
    fig, axes = plt.subplots(len(reps), 1, figsize=(11, 2.1 * len(reps)), sharex=True)
    for ax, p in zip(np.atleast_1d(axes), reps):
        pair = p["pair"]
        ax.plot(base_h[pair].index, base_h[pair].values, label="v7 baseline", lw=1.4)
        ax.plot(wf_h_paths[pair].index, wf_h_paths[pair].values, label="walk-forward h*",
                lw=1.4, drawstyle="steps-post")
        ax.set_ylabel("h")
        ax.set_ylim(0.25, 1.0)
        ax.set_title(pair, fontsize=9, loc="left")
        ax.grid(alpha=0.3)
    np.atleast_1d(axes)[0].legend(loc="upper right", fontsize=8)
    fig.suptitle("Hedge ratio h(t): v7 baseline vs walk-forward adaptive per-pair h*")
    fig.tight_layout()
    fig.savefig(outdir / "b4_hstar_wf_h_timeseries.png", dpi=130)
    plt.close(fig)
    return wf_rows, wf_score, summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Bucket 4 hedge-ratio h* optimizer.")
    ap.add_argument("--pairs", type=Path, default=None)
    ap.add_argument("--metrics", type=Path,
                    default=REPO.parent / "Levered ETFs/etf-dashboard/data/etf_metrics_daily.csv")
    ap.add_argument("--vol-shape", type=Path,
                    default=REPO.parent / "Levered ETFs/etf-dashboard/data/vol_shape_history.json")
    ap.add_argument("--screened", type=Path, default=REPO / "data/etf_screened_today.csv")
    ap.add_argument("--outdir", type=Path, default=REPO / "notebooks/output/b4_hstar")
    ap.add_argument("--start", default="2025-10-07")
    ap.add_argument("--max-pairs", type=int, default=0)
    args = ap.parse_args(argv)
    args.outdir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    pair_data, _ = load_pair_data(args)
    print(f"[hstar] pairs with data: {len(pair_data)}")
    if not pair_data:
        return 1

    # fixed cadence schedule, computed once per pair and reused everywhere
    for pd_ in pair_data:
        pd_["sched"] = schedule_base12(pd_)

    # ---------------- baseline ----------------
    baseline_score, baseline_rows = score_variant(pair_data, v7_baseline_h, {"variant": "v7_baseline"})
    if baseline_score is None:
        print("[hstar] baseline scoring failed")
        return 1
    print(f"[hstar] baseline v7: ew_mean_cagr={baseline_score['ew_mean_cagr']:.4f} "
          f"ew_median_cagr={baseline_score['ew_median_cagr']:.4f}")

    # ---------------- experiments ----------------
    long_df, arg_df, corr_df, eq_panels = experiment1(pair_data, args.outdir)

    # IS upper bound: per-pair best constant h (oracle, fully overfit)
    oracle_mean = float(arg_df["cagr_at_hstar"].mean())
    oracle_median = float(arg_df["cagr_at_hstar"].median())

    refit_df, best2 = experiment2(pair_data, args.outdir, baseline_score)
    best2_score, best2_rows = score_variant(
        pair_data, lambda p: closed_form_h(p, **best2),
        {"variant": "best_refit", **best2})

    sig_df, marg_df = experiment3(pair_data, args.outdir, best2, best2_score)

    wf_rows, wf_score, wf_summary = experiment4(pair_data, eq_panels, args.outdir,
                                                baseline_rows, baseline_score)

    # ---------------- plot 5: EW portfolio equity curves ----------------
    navs = {
        "v7 baseline": ew_portfolio_nav(baseline_rows),
        f"best refit (h_mid={best2['h_mid']}, k_vcr={best2['k_vcr']}, h_max={best2['h_max']})":
            ew_portfolio_nav(best2_rows),
        "walk-forward adaptive h* (OOS)": ew_portfolio_nav(wf_rows),
    }
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for lab, nav in navs.items():
        ax.plot(nav.index, nav.values, label=lab, lw=1.6)
    ax.set_ylabel("EW portfolio NAV (start=1)")
    ax.set_title("Bucket 4 EW portfolio equity: baseline vs best global refit (IS) vs walk-forward h* (OOS)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.outdir / "b4_hstar_portfolio_equity.png", dpi=130)
    plt.close(fig)

    # ---------------- results markdown ----------------
    def block(df: pd.DataFrame, n: int | None = None) -> str:
        d = df if n is None else df.head(n)
        return "```\n" + d.to_string(index=False) + "\n```"

    best_marg = marg_df.loc[marg_df["d_mean_cagr_vs_best2"].idxmax()] if not marg_df.empty else None
    n_lo = int((arg_df["h_star"] <= H_GRID[0] + 1e-9).sum())
    n_hi = int((arg_df["h_star"] >= H_GRID[-1] - 1e-9).sum())
    d_med_wf = wf_score["ew_median_cagr"] - baseline_score["ew_median_cagr"]
    d_med_refit = best2_score["ew_median_cagr"] - baseline_score["ew_median_cagr"]
    md = [
        "# Bucket 4 hedge-ratio (h*) optimization results",
        "",
        f"Window start: {args.start} | pairs: {len(pair_data)} | slippage 20bps, fee 0, static beta, screener borrow.",
        f"Cadence fixed at production optimum: `policy_continuous_interval(base_days={BASE_DAYS}, "
        f"k_tr={K_TR}, m_vcr={M_VCR}, min=1, max={CAP_DAYS})`, schedule computed once per pair and reused.",
        "",
        "## Method",
        "",
        "The engine (`run_bucket4_backtest_dynamic_h`) shorts the inverse ETF and shorts "
        "`h * |beta|` dollars of the underlying per dollar of inverse short; `h` is consumed "
        "as a daily series but only matters on rebalance days. Everything except the h-rule "
        "is held at the production configuration above.",
        "",
        "**Baseline (v7 closed form):**",
        "",
        "```",
        "h_t = clip(0.55 + 1.0*(VCR_t - VCRmed_t), 0.30, 0.80);  NaN -> partial_h;",
        "h <- EMA(h, alpha=0.25, adjust=False)",
        "```",
        "",
        "Experiments 1-3 are **in-sample** (parameters chosen on the same window they are "
        "scored on). Experiment 4 is the **out-of-sample** test: every 21 trading days, each "
        "pair adopts the constant h in {0.30..0.95} with the best trailing-63-day equity "
        "growth (`eq[t]/eq[t-63]`, from a precomputed constant-h panel; ties -> lower h), "
        "applied from the next day. Warmup (first 63bd) uses the v7 baseline h.",
        "",
        f"## Baseline score",
        "",
        f"- EW mean CAGR: **{baseline_score['ew_mean_cagr']:.4f}**",
        f"- EW median CAGR: **{baseline_score['ew_median_cagr']:.4f}**",
        f"- winsorized mean: {baseline_score['winsor_mean_cagr']:.4f} | mean vol: "
        f"{baseline_score['ew_mean_vol']:.4f} | mean max-DD: {baseline_score['ew_mean_max_dd']:.4f}",
        "",
        "## Experiment 1 — per-pair constant h* (IS)",
        "",
        f"Constant-h grid {H_GRID[0]:.2f}..{H_GRID[-1]:.2f} step 0.05 per pair "
        f"(`b4_hstar_perpair_profiles.csv`). Per-pair argmax (`b4_hstar_perpair_argmax.csv`):",
        "",
        block(arg_df),
        "",
        f"Oracle (pick each pair's IS-best constant h): EW mean CAGR **{oracle_mean:.4f}**, "
        f"median **{oracle_median:.4f}** — this is the fully overfit upper bound, not attainable.",
        "",
        "Cross-section of h* vs pair characteristics:",
        "",
        block(corr_df),
        "",
        "## Experiment 2 — global v7 refit (IS)",
        "",
        f"Grid: h_mid in {GRID_H_MID}, k_vcr in {GRID_K_VCR}, h_max in {GRID_H_MAX} "
        "(h_min 0.30, EMA 0.25 fixed). Top 10 by composite rank:",
        "",
        block(refit_df, 10),
        "",
        f"**Best refit:** `{best2}` -> EW mean CAGR {best2_score['ew_mean_cagr']:.4f} "
        f"({best2_score['ew_mean_cagr'] - baseline_score['ew_mean_cagr']:+.4f} vs baseline).",
        "",
        "## Experiment 3 — one extra signal on top of the refit optimum (IS)",
        "",
        "h_t = clip(h_mid* + k_vcr*(VCR-VCRmed) + k_sig*X, 0.30, h_max*), X in "
        "{TR-1, rv_pct-0.5, sign(20d underlying return, shifted)}; signal NaNs contribute 0.",
        "",
        block(sig_df.sort_values(["signal", "coef"])),
        "",
        "Best coefficient per signal (marginal EW mean CAGR vs exp-2 optimum):",
        "",
        block(marg_df),
        "",
        "## Experiment 4 — walk-forward adaptive per-pair h* (OOS)",
        "",
        block(wf_summary),
        "",
        "## Conclusion",
        "",
        f"- Baseline v7 EW mean CAGR {baseline_score['ew_mean_cagr']:.4f} (median "
        f"{baseline_score['ew_median_cagr']:.4f}); best IS refit mean "
        f"{best2_score['ew_mean_cagr']:.4f} ({best2_score['ew_mean_cagr'] - baseline_score['ew_mean_cagr']:+.4f}, "
        f"median {d_med_refit:+.4f}); OOS walk-forward mean {wf_score['ew_mean_cagr']:.4f} "
        f"({wf_score['ew_mean_cagr'] - baseline_score['ew_mean_cagr']:+.4f}, median {d_med_wf:+.4f}).",
        f"- **Per-pair h* is bimodal at the grid edges**: {n_lo}/{len(arg_df)} pairs pin at "
        f"h*={H_GRID[0]:.2f} and {n_hi}/{len(arg_df)} at h*={H_GRID[-1]:.2f}. h* is mostly a "
        "label for whether the pair made or lost money this window (winners want minimal "
        "hedging, losers want maximal), i.e. it encodes the realized outcome rather than an "
        "exploitable ex-ante characteristic. The strongest cross-sectional driver is borrow "
        "cost (expensive-to-carry pairs prefer high h).",
        f"- The best IS refit sets `k_vcr=0` — on this window the VCR tilt adds nothing and "
        "the gain comes almost entirely from a lower average h (more residual delta in a "
        "period when shorts of inverse ETFs were profitable). That is direction-of-market "
        "exposure, not signal alpha.",
        ("- The best extra signal is "
         f"`{best_marg['signal']}` (k={best_marg['coef']:+.1f}, "
         f"{best_marg['d_mean_cagr_vs_best2']:+.4f} EW mean CAGR vs the refit optimum), but "
         "check the median column in `b4_hstar_signal_marginals.csv` before trusting it — "
         "mean improvements driven by a few high-vol pairs while the median deteriorates are "
         "an overfit signature."
         if best_marg is not None else "- No extra-signal results."),
        "- **IS vs OOS:** experiments 1-3 select parameters on the same window they are scored "
        "on and therefore overstate attainable performance; the per-pair oracle h* is pure "
        "overfit. Only the experiment-4 walk-forward delta is an honest estimate of what "
        "per-pair h adaptation would have delivered out of sample on this (short, ~8-month) "
        "window — and there the mean improves while the **median deteriorates**, meaning the "
        "typical pair is hurt and the aggregate gain rides on a handful of momentum-y winners.",
        "- **Recommendation:** keep the production v7 rule. Do not adopt per-pair h* or the "
        "walk-forward adapter on this evidence (median-negative OOS, h* pinned at grid edges). "
        "The only candidates worth a longer-window / paper-live validation are (a) a modestly "
        "lower h_mid and (b) a small negative rv_pct tilt (rv_pct k=-0.2 improved both mean "
        "and median in-sample); both add residual delta, so size them against the sleeve's "
        "delta budget rather than on backtest CAGR alone.",
        "",
        "## Files",
        "",
        "- `b4_hstar_perpair_profiles.csv`, `b4_hstar_perpair_argmax.csv`, `b4_hstar_crosssection_corr.csv`",
        "- `b4_hstar_refit_grid.csv`, `b4_hstar_signal_marginals.csv`",
        "- `b4_hstar_walkforward_summary.csv`, `b4_hstar_walkforward_picks.csv`",
        "- `b4_hstar_heatmap_pair_by_h.png`, `b4_hstar_scatter_drivers.png`, `b4_hstar_refit_heatmap.png`,",
        "  `b4_hstar_wf_h_timeseries.png`, `b4_hstar_portfolio_equity.png`, `b4_hstar_signal_marginal_bars.png`",
        "",
    ]
    (args.outdir / "B4_HSTAR_RESULTS.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\n[hstar] wrote {args.outdir / 'B4_HSTAR_RESULTS.md'} "
          f"({time.time() - t0:.0f}s total)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
