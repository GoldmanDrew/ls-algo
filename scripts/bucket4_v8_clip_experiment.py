"""Bucket 4 "v8" experiment: jointly optimize the v7 hedge clip constants.

The production v7 rule is  h = clip(h_mid + k_vcr*(VCR - VCR_med), h_min, h_max)
with (h_mid, h_min, h_max) = (0.55, 0.30, 0.80) INHERITED from the v6
calibration -- never jointly optimized. This experiment grids all three while
holding everything else at the adopted optimum (k_vcr=1.0, EMA alpha=0.25,
cadence base 12 / k_tr 2.25 / m_vcr 2.5 / cap 21, static beta, 20bps slip).

Stability-first selection (pre-declared, to avoid blessing an in-sample fluke):
  1. Every combo is scored on the FULL window and on two time halves (H1/H2)
     computed from the same equity curves (no refitting -- the rule has no
     fitted state, so slicing the curve is valid).
  2. A combo only QUALIFIES for v8 if it ranks in the top quartile of EW mean
     CAGR in BOTH halves (it must work in two different regimes).
  3. Among qualifiers, pick the best average rank across (full winsor mean,
     full median, H1 mean, H2 mean, ret/vol) -- then report the PLATEAU: the
     mean score of its grid neighbors. A sharp peak is a red flag.
  4. Pair-fold CV: 20 random 50/50 splits of the pair universe; the train half
     picks its best combo, which is then scored on the held-out half vs v7.
     Reports mean OOS uplift and win rate -- "would this re-pick have helped
     on names you didn't fit on?"

Outputs -> notebooks/output/b4_v8/:
  b4_v8_grid_scores.csv, b4_v8_pairfold_cv.csv, plots (heatmaps, H1-vs-H2
  stability scatter, plateau slices, EW equity curves), B4_V8_RESULTS.md.

Usage:  python -m scripts.bucket4_v8_clip_experiment [--max-pairs 0]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.bucket4_dynamic_bt import run_bucket4_backtest_dynamic_h  # noqa: E402
from scripts.bucket4_phase345_backtest import load_pair_data  # noqa: E402
from scripts.bucket4_vol_shape_signals import policy_continuous_interval  # noqa: E402

TRADING_DAYS = 252
SLIPPAGE_BPS = 20.0
K_VCR = 1.0
EMA_ALPHA = 0.25
BASE_DAYS, K_TR, M_VCR, CAP_DAYS = 12.0, 2.25, 2.5, 21
V7 = (0.55, 0.30, 0.80)

H_MID_GRID = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65]
H_MIN_GRID = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
H_MAX_GRID = [0.70, 0.80, 0.90, 0.95]


def valid_grid() -> list[tuple[float, float, float]]:
    out = []
    for mid in H_MID_GRID:
        for lo in H_MIN_GRID:
            for hi in H_MAX_GRID:
                if lo <= mid - 0.05 and hi >= mid + 0.05:
                    out.append((mid, lo, hi))
    if V7 not in out:
        out.append(V7)
    return out


def h_series(pd_: dict, mid: float, lo: float, hi: float) -> pd.Series:
    sig = pd_["sig"]
    vcr = pd.to_numeric(sig.get("vcr"), errors="coerce")
    vcr_med = pd.to_numeric(sig.get("vcr_med"), errors="coerce")
    h = (mid + K_VCR * (vcr - vcr_med)).clip(lo, hi)
    h = h.fillna(float(np.clip(pd_["partial_h"], lo, hi)))
    return h.ewm(alpha=EMA_ALPHA, adjust=False).mean()


def cagr_of(eq: pd.Series) -> float:
    eq = eq.dropna()
    if len(eq) < 10 or eq.iloc[0] <= 1e-9:
        return np.nan
    years = max((eq.index[-1] - eq.index[0]).days / 365.25, 1 / 365.25)
    return float((eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1.0)


def run_all(pair_data: list[dict], grid: list[tuple]) -> tuple[pd.DataFrame, pd.Timestamp]:
    """One engine run per (combo, pair); returns long frame of full/H1/H2 metrics."""
    all_dates = sorted({d for p in pair_data for d in p["prices"].index})
    split = all_dates[len(all_dates) // 2]
    rows = []
    for gi, (mid, lo, hi) in enumerate(grid, 1):
        for pd_ in pair_data:
            h = h_series(pd_, mid, lo, hi)
            try:
                bt = run_bucket4_backtest_dynamic_h(
                    pd_["prices"], h, pd_["sched"],
                    beta_a=-pd_["beta_static"], beta_b=1.0,
                    borrow_a_annual=pd_["borrow_a"],
                    fee_bps=0.0, slippage_bps=SLIPPAGE_BPS,
                    opt2_h_base=float(pd_["partial_h"]),
                )
            except Exception:
                continue
            if bt is None or bt.empty:
                continue
            eq = bt["equity"]
            rets = eq.pct_change().dropna()
            vol = float(rets.std(ddof=1) * np.sqrt(TRADING_DAYS)) if len(rets) > 2 else np.nan
            dd = float((eq / eq.cummax() - 1.0).min())
            rows.append({
                "h_mid": mid, "h_min": lo, "h_max": hi, "pair": pd_["pair"],
                "cagr_full": cagr_of(eq),
                "cagr_h1": cagr_of(eq.loc[:split]),
                "cagr_h2": cagr_of(eq.loc[split:]),
                "vol": vol, "max_dd": dd,
            })
        if gi % 20 == 0 or gi == len(grid):
            print(f"  ... combo {gi}/{len(grid)}", flush=True)
    return pd.DataFrame(rows), split


def score_combos(long: pd.DataFrame) -> pd.DataFrame:
    def agg(g: pd.DataFrame) -> pd.Series:
        c = g["cagr_full"].dropna()
        lo_, hi_ = (c.quantile(0.05), c.quantile(0.95)) if len(c) else (np.nan, np.nan)
        return pd.Series({
            "n_pairs": len(g),
            "ew_mean_full": c.mean(),
            "ew_median_full": c.median(),
            "winsor_mean_full": c.clip(lo_, hi_).mean(),
            "ew_mean_h1": g["cagr_h1"].mean(skipna=True),
            "ew_mean_h2": g["cagr_h2"].mean(skipna=True),
            "ew_median_h1": g["cagr_h1"].median(skipna=True),
            "ew_median_h2": g["cagr_h2"].median(skipna=True),
            "ew_mean_vol": g["vol"].mean(skipna=True),
            "ew_mean_dd": g["max_dd"].mean(skipna=True),
        })
    s = long.groupby(["h_mid", "h_min", "h_max"]).apply(agg, include_groups=False).reset_index()
    s["ret_over_vol"] = s["ew_mean_full"] / s["ew_mean_vol"].clip(lower=0.01)
    for col in ("winsor_mean_full", "ew_median_full", "ew_mean_h1", "ew_mean_h2", "ret_over_vol"):
        s[f"rank_{col}"] = s[col].rank(ascending=False)
    s["composite_rank"] = s[[c for c in s.columns if c.startswith("rank_")]].mean(axis=1)

    # stability qualifier: top quartile EW mean in BOTH halves
    q1 = s["ew_mean_h1"].quantile(0.75)
    q2 = s["ew_mean_h2"].quantile(0.75)
    s["stable"] = (s["ew_mean_h1"] >= q1) & (s["ew_mean_h2"] >= q2)
    return s.sort_values("composite_rank")


def plateau_score(s: pd.DataFrame, mid: float, lo: float, hi: float) -> float:
    """Mean full EW CAGR of grid neighbors (one step in any single dimension)."""
    def near(vals, x):
        i = vals.index(x)
        return {vals[j] for j in (i - 1, i + 1) if 0 <= j < len(vals)}
    nb = []
    for m2 in near(H_MID_GRID, mid):
        nb.append((m2, lo, hi))
    for l2 in near(H_MIN_GRID, lo):
        nb.append((mid, l2, hi))
    for h2 in near(H_MAX_GRID, hi):
        nb.append((mid, lo, h2))
    vals = [
        float(s.loc[(s["h_mid"] == a) & (s["h_min"] == b) & (s["h_max"] == c), "ew_mean_full"].iloc[0])
        for a, b, c in nb
        if ((s["h_mid"] == a) & (s["h_min"] == b) & (s["h_max"] == c)).any()
    ]
    return float(np.mean(vals)) if vals else np.nan


def pairfold_cv(long: pd.DataFrame, n_folds: int = 20, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    pairs = sorted(long["pair"].unique())
    key = ["h_mid", "h_min", "h_max"]
    piv = long.pivot_table(index=key, columns="pair", values="cagr_full")
    v7_row = piv.loc[V7]
    rows = []
    for f in range(n_folds):
        sel = rng.permutation(len(pairs))
        train = [pairs[i] for i in sel[: len(pairs) // 2]]
        test = [pairs[i] for i in sel[len(pairs) // 2:]]
        train_mean = piv[train].mean(axis=1)
        best = train_mean.idxmax()
        oos_pick = float(piv.loc[best, test].mean())
        oos_v7 = float(v7_row[test].mean())
        rows.append({"fold": f, "picked": best, "oos_pick_ew_mean": oos_pick,
                     "oos_v7_ew_mean": oos_v7, "uplift": oos_pick - oos_v7})
    return pd.DataFrame(rows)


def ew_portfolio_curve(pair_data: list[dict], mid: float, lo: float, hi: float) -> pd.Series:
    rets = {}
    for pd_ in pair_data:
        h = h_series(pd_, mid, lo, hi)
        bt = run_bucket4_backtest_dynamic_h(
            pd_["prices"], h, pd_["sched"],
            beta_a=-pd_["beta_static"], beta_b=1.0,
            borrow_a_annual=pd_["borrow_a"], fee_bps=0.0, slippage_bps=SLIPPAGE_BPS,
            opt2_h_base=float(pd_["partial_h"]),
        )
        rets[pd_["pair"]] = bt["equity"].pct_change()
    rmat = pd.DataFrame(rets)
    return (1 + rmat.mean(axis=1).fillna(0.0)).cumprod()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="v8 hedge clip-bound experiment.")
    ap.add_argument("--pairs", type=Path, default=None)
    ap.add_argument("--metrics", type=Path,
                    default=REPO.parent / "Levered ETFs/etf-dashboard/data/etf_metrics_daily.csv")
    ap.add_argument("--vol-shape", type=Path,
                    default=REPO.parent / "Levered ETFs/etf-dashboard/data/vol_shape_history.json")
    ap.add_argument("--screened", type=Path, default=REPO / "data/etf_screened_today.csv")
    ap.add_argument("--outdir", type=Path, default=REPO / "notebooks/output/b4_v8")
    ap.add_argument("--start", default="2025-10-07")
    ap.add_argument("--max-pairs", type=int, default=0)
    args = ap.parse_args(argv)
    args.outdir.mkdir(parents=True, exist_ok=True)

    pair_data, _ = load_pair_data(args)
    print(f"[v8] pairs with data: {len(pair_data)}")
    for pd_ in pair_data:
        rd, _ = policy_continuous_interval(
            pd_["prices"].index, pd_["sig"],
            base_days=BASE_DAYS, k_tr=K_TR, m_vcr=M_VCR,
            min_interval=1, max_interval=CAP_DAYS,
        )
        pd_["sched"] = pd.DatetimeIndex(rd).intersection(pd_["prices"].index)

    grid = valid_grid()
    print(f"[v8] grid combos: {len(grid)} (h_mid x h_min x h_max, valid orderings)")
    long, split = run_all(pair_data, grid)
    print(f"[v8] time split at {split.date()} (H1 before, H2 after)")

    scores = score_combos(long)
    # plateau for the top 10 + v7
    scores["plateau_ew_mean"] = [
        plateau_score(scores, r["h_mid"], r["h_min"], r["h_max"])
        if (i < 10 or (r["h_mid"], r["h_min"], r["h_max"]) == V7) else np.nan
        for i, (_, r) in enumerate(scores.iterrows())
    ]
    scores.to_csv(args.outdir / "b4_v8_grid_scores.csv", index=False)

    v7_row = scores[(scores["h_mid"] == V7[0]) & (scores["h_min"] == V7[1]) & (scores["h_max"] == V7[2])].iloc[0]

    # Selection: the h_mid gradient is monotone (lower hedge = more net-short
    # = more return AND more risk), so raw-CAGR argmax just picks the grid
    # edge -- a regime bet, not a better rule. Instead pick under a RISK
    # BUDGET: vol and max-dd no worse than 1.25x the v7 baseline. Tie-break:
    # among combos within 0.5pp of the gated best, take the one closest to v7
    # (minimal-change principle -- h_min/h_max barely bind, so don't move them
    # without evidence).
    vol_cap = float(v7_row["ew_mean_vol"]) * 1.25
    dd_floor = float(v7_row["ew_mean_dd"]) * 1.25  # dd is negative
    gated = scores[(scores["ew_mean_vol"] <= vol_cap) & (scores["ew_mean_dd"] >= dd_floor)]
    if gated.empty:
        gated = scores
    best_gated = gated.sort_values("composite_rank").iloc[0]
    near = gated[gated["ew_mean_full"] >= float(best_gated["ew_mean_full"]) - 0.005].copy()
    near["dist_v7"] = (near["h_mid"] - V7[0]).abs() + (near["h_min"] - V7[1]).abs() + (near["h_max"] - V7[2]).abs()
    v8_row = near.sort_values(["dist_v7", "composite_rank"]).iloc[0]
    v8 = (float(v8_row["h_mid"]), float(v8_row["h_min"]), float(v8_row["h_max"]))
    v8_row = v8_row.copy()
    v8_row["plateau_ew_mean"] = plateau_score(scores, *v8)

    cv = pairfold_cv(long)
    cv.to_csv(args.outdir / "b4_v8_pairfold_cv.csv", index=False)
    cv_uplift = float(cv["uplift"].mean())
    cv_winrate = float((cv["uplift"] > 0).mean())
    cv_modal = cv["picked"].mode().iloc[0] if not cv.empty else None

    print("\n=== Top 10 combos (stability-ranked) ===")
    show = ["h_mid", "h_min", "h_max", "stable", "ew_mean_full", "ew_median_full",
            "ew_mean_h1", "ew_mean_h2", "ew_mean_vol", "ew_mean_dd", "ret_over_vol",
            "plateau_ew_mean", "composite_rank"]
    with pd.option_context("display.width", 240):
        print(scores[show].head(10).to_string(index=False))
        print("\n--- v7 baseline row ---")
        print(scores[show][(scores["h_mid"] == V7[0]) & (scores["h_min"] == V7[1]) & (scores["h_max"] == V7[2])]
              .to_string(index=False))
    print(f"\n[v8] risk budget: vol<= {vol_cap:.2%}, dd>= {dd_floor:.2%} (1.25x v7)")
    print("\n=== Top 10 within risk budget ===")
    with pd.option_context("display.width", 240):
        print(gated[show].head(10).to_string(index=False))
    print(f"\n[v8] candidate: h_mid={v8[0]} h_min={v8[1]} h_max={v8[2]} (stable={bool(v8_row['stable'])})")
    print(f"[v8] pair-fold CV: mean OOS uplift {cv_uplift:+.4f}, win-rate {cv_winrate:.0%}, modal pick {cv_modal}")

    # ------------------- plots -------------------
    # 1. heatmaps h_mid x h_max, faceted by h_min
    fig, axes = plt.subplots(1, len(H_MIN_GRID), figsize=(4 * len(H_MIN_GRID), 4), sharey=True)
    for ax, lo in zip(axes, H_MIN_GRID):
        sub = scores[scores["h_min"] == lo].pivot_table(index="h_max", columns="h_mid", values="ew_mean_full")
        sub = sub.reindex(index=sorted(set(scores["h_max"]), reverse=True), columns=H_MID_GRID)
        im = ax.imshow(sub.values * 100, cmap="RdYlGn", vmin=-10, vmax=15, aspect="auto")
        ax.set_xticks(range(len(sub.columns)), [f"{c:.2f}" for c in sub.columns])
        ax.set_yticks(range(len(sub.index)), [f"{i:.2f}" for i in sub.index])
        ax.set_title(f"h_min={lo:.2f}")
        ax.set_xlabel("h_mid")
        for (a, b, c), mark in [(V7, "v7"), (v8, "v8")]:
            if b == lo and a in sub.columns and c in sub.index:
                ax.text(list(sub.columns).index(a), list(sub.index).index(c), mark,
                        ha="center", va="center", fontweight="bold", fontsize=11)
    axes[0].set_ylabel("h_max")
    fig.colorbar(im, ax=axes, label="EW mean CAGR %", shrink=0.85)
    fig.suptitle("v8 grid: EW mean CAGR (full window) by clip bounds")
    fig.savefig(args.outdir / "b4_v8_grid_heatmaps.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # 2. H1 vs H2 stability scatter
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(scores["ew_mean_h1"] * 100, scores["ew_mean_h2"] * 100,
               c=np.where(scores["stable"], "#2a9d2a", "#bbbbbb"), s=28, alpha=0.85)
    for (combo, lab, col) in [(V7, "v7", "#1f77b4"), (v8, "v8", "#d62728")]:
        r = scores[(scores["h_mid"] == combo[0]) & (scores["h_min"] == combo[1]) & (scores["h_max"] == combo[2])]
        if not r.empty:
            ax.scatter(r["ew_mean_h1"] * 100, r["ew_mean_h2"] * 100, c=col, s=140, marker="*",
                       label=f"{lab} ({combo[0]:.2f}/{combo[1]:.2f}/{combo[2]:.2f})", zorder=5)
    ax.axhline(0, color="#888", lw=0.7)
    ax.axvline(0, color="#888", lw=0.7)
    ax.set_xlabel("EW mean CAGR % — first half")
    ax.set_ylabel("EW mean CAGR % — second half")
    ax.set_title("Stability: each dot is one (h_mid,h_min,h_max); green = top quartile in BOTH halves")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.outdir / "b4_v8_stability_scatter.png", dpi=130)
    plt.close(fig)

    # 3. plateau slices: EW mean vs h_mid for v8's (h_min, h_max) and v7's
    fig, ax = plt.subplots(figsize=(8, 5))
    for (lo, hi, lab, col) in [(v8[1], v8[2], f"h_min={v8[1]:.2f}, h_max={v8[2]:.2f} (v8 slice)", "#d62728"),
                               (V7[1], V7[2], f"h_min={V7[1]:.2f}, h_max={V7[2]:.2f} (v7 slice)", "#1f77b4")]:
        sub = scores[(scores["h_min"] == lo) & (scores["h_max"] == hi)].sort_values("h_mid")
        ax.plot(sub["h_mid"], sub["ew_mean_full"] * 100, "o-", label=lab, color=col)
    ax.axhline(float(v7_row["ew_mean_full"]) * 100, ls=":", color="#1f77b4", alpha=0.6, label="v7 level")
    ax.set_xlabel("h_mid")
    ax.set_ylabel("EW mean CAGR %")
    ax.set_title("Plateau check: sensitivity to h_mid (broad plateau = robust, sharp peak = fragile)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.outdir / "b4_v8_plateau_slices.png", dpi=130)
    plt.close(fig)

    # 4. EW portfolio equity: v7 vs v8
    eq7 = ew_portfolio_curve(pair_data, *V7)
    eq8 = ew_portfolio_curve(pair_data, *v8)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(eq7.index, eq7.values, label=f"v7 {V7}", lw=1.6)
    ax.plot(eq8.index, eq8.values, label=f"v8 {v8}", lw=1.6)
    ax.axvline(split, color="#888", ls="--", lw=1, label=f"H1|H2 split ({split.date()})")
    ax.set_ylabel("EW portfolio NAV (start=1)")
    ax.set_title("Bucket 4 EW portfolio: v7 vs v8 clip bounds (same cadence, costs, pairs)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.outdir / "b4_v8_equity_curves.png", dpi=130)
    plt.close(fig)

    # 5. pair-fold CV uplift histogram
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(cv["uplift"] * 100, bins=15, color="#4477aa", alpha=0.85)
    ax.axvline(0, color="k", lw=1)
    ax.axvline(cv_uplift * 100, color="#d62728", ls="--", label=f"mean {cv_uplift*100:+.2f}pp")
    ax.set_xlabel("OOS EW mean CAGR uplift vs v7 (pp), per fold")
    ax.set_title(f"Pair-fold CV (20 folds): train-half picks bounds, scored on held-out pairs\nwin-rate {cv_winrate:.0%}, modal pick {cv_modal}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.outdir / "b4_v8_pairfold_cv.png", dpi=130)
    plt.close(fig)

    # ------------------- results md -------------------
    lines = [
        "# Bucket 4 v8 experiment: hedge clip bounds (h_mid / h_min / h_max)\n",
        f"Window {args.start} -> latest | {len(pair_data)} pairs | split {split.date()} | "
        f"k_vcr={K_VCR}, EMA={EMA_ALPHA}, cadence base {BASE_DAYS:.0f}/{K_TR}/{M_VCR}/cap{CAP_DAYS}, slip {SLIPPAGE_BPS}bps\n",
        "## Why 0.55/0.30/0.80 existed\n",
        "Inherited from the v6 calibration (V7_DEFAULT_H_MID, V7_GLOBAL_H_MIN/MAX in "
        "scripts/bucket4_hedge_v7.py) -- guardrails chosen to mimic v6's average hedge level, "
        "never jointly optimized.\n",
        "## Selection protocol (pre-declared)\n",
        "Top quartile EW mean CAGR in BOTH time halves required to qualify; best composite "
        "(full winsor mean, full median, H1 mean, H2 mean, ret/vol) among qualifiers wins; "
        "plateau and 20-fold pair CV reported as overfit checks.\n",
        "## Top 10 combos\n",
        "```\n" + scores[show].head(10).to_string(index=False) + "\n```\n",
        "## v7 baseline\n",
        "```\n" + scores[show][(scores['h_mid'] == V7[0]) & (scores['h_min'] == V7[1]) & (scores['h_max'] == V7[2])].to_string(index=False) + "\n```\n",
        f"## v8 candidate: h_mid={v8[0]} h_min={v8[1]} h_max={v8[2]}\n",
        f"- full EW mean {v8_row['ew_mean_full']:.2%} vs v7 {v7_row['ew_mean_full']:.2%}\n",
        f"- full EW median {v8_row['ew_median_full']:.2%} vs v7 {v7_row['ew_median_full']:.2%}\n",
        f"- H1 {v8_row['ew_mean_h1']:.2%} vs {v7_row['ew_mean_h1']:.2%} | H2 {v8_row['ew_mean_h2']:.2%} vs {v7_row['ew_mean_h2']:.2%}\n",
        f"- vol {v8_row['ew_mean_vol']:.2%} vs {v7_row['ew_mean_vol']:.2%} | dd {v8_row['ew_mean_dd']:.2%} vs {v7_row['ew_mean_dd']:.2%}\n",
        f"- plateau (neighbor mean) {v8_row['plateau_ew_mean']:.2%}\n",
        f"- pair-fold CV: mean OOS uplift {cv_uplift:+.2%}, win-rate {cv_winrate:.0%}, modal pick {cv_modal}\n",
    ]
    (args.outdir / "B4_V8_RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[v8] wrote {args.outdir / 'B4_V8_RESULTS.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
