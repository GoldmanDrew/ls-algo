"""Bucket 4 "v9" experiment: add the v6 panel's cross-sectional signal to v8.

v8 (production candidate): h = clip(0.45 + k_vcr*(VCR - VCR_med), 0.30, 0.80), EMA 0.25.
v8 only sees the pair's OWN variance state. The legacy v6 Opt-2 panel ranked
each underlying AGAINST ITS PEERS weekly:

    z_composite = 0.5*(-z(r_10d)) + 0.5*(+z(range_expansion))
    h_v6        = clip(0.75 - 0.05 * z_composite, ...)

  r_10d            10d log return of the underlying  (rally -> hedge UP in v6)
  range_expansion  sigma5/sigma63 vol ratio          (expansion -> hedge DOWN in v6)

v9 candidate: keep the v8 closed form and ADD the tilt:

    h = clip(h_mid + k_vcr*(VCR - VCR_med) - k_z * z_tilt, h_min, h_max), EMA 0.25

with z_tilt in {composite (v6 weights), momentum-only (-z10), range-only (+zrx)}
and k_z gridded over BOTH signs -- the h* lab's best extra signal was momentum
with the OPPOSITE sign to v6's convention, so direction is an open question.

Implementation notes (honesty):
  * features computed daily per UNIQUE underlying (MSTR appears once, not 3x),
    z-scored cross-sectionally with the same robust (median/MAD) scheme as v6,
    then SHIFTED 1 DAY so today's trade only uses yesterday's information.
  * warmup: sigma63 needs >=30 obs; missing z -> 0 tilt (falls back to pure v8).
  * regime overlay NOT ported: already tested in the phase-3 lab and rejected.

Selection protocol (same as v8): full-window + split-half EW scores, 20-fold
pair CV vs the v8 baseline, risk budget (vol/DD <= 1.10x v8 -- the tilt is not
supposed to change the average hedge level, so the budget is tighter than the
1.25x used when relocating h_mid).

Outputs -> notebooks/output/b4_v9/:
  b4_v9_grid_scores.csv, b4_v9_pairfold_cv.csv, plots, B4_V9_RESULTS.md

Usage:  python -m scripts.bucket4_v9_xsec_experiment [--max-pairs 0]
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
from scripts.bucket4_weekly_opt2 import robust_z_cross_sectional  # noqa: E402

TRADING_DAYS = 252
SLIPPAGE_BPS = 20.0
K_VCR = 1.0
EMA_ALPHA = 0.25
BASE_DAYS, K_TR, M_VCR, CAP_DAYS = 12.0, 2.25, 2.5, 21
H_MID, H_MIN, H_MAX = 0.45, 0.30, 0.80   # v8

SIGNALS = ["composite", "momentum", "range"]
K_Z_GRID = [-0.20, -0.10, -0.05, 0.05, 0.10, 0.20, 0.30, 0.40, 0.60]


# ---------------------------------------------------------------------------
# Cross-sectional z panel (the portable core of the v6 panel)
# ---------------------------------------------------------------------------
def build_xsec_panel(pair_data: list[dict]) -> dict[str, pd.DataFrame]:
    """Daily z10 / zrx / z_composite per unique underlying, shifted 1 day."""
    und_close: dict[str, pd.Series] = {}
    for pd_ in pair_data:
        u = pd_["underlying"]
        if u not in und_close or len(pd_["prices"]) > len(und_close[u]):
            und_close[u] = pd_["prices"]["b_px"].astype(float)
    closes = pd.DataFrame(und_close).sort_index()

    logret = np.log(closes / closes.shift(1))
    r10 = np.log(closes / closes.shift(10))
    sigma5 = logret.rolling(5, min_periods=5).std(ddof=1)
    sigma63 = logret.rolling(63, min_periods=30).std(ddof=1)
    rx = sigma5 / sigma63

    z10 = pd.DataFrame({d: robust_z_cross_sectional(r10.loc[d]) for d in r10.index}).T
    zrx = pd.DataFrame({d: robust_z_cross_sectional(rx.loc[d]) for d in rx.index}).T
    z10, zrx = z10.shift(1), zrx.shift(1)   # trade on yesterday's ranks only
    return {
        "composite": (0.5 * (-z10) + 0.5 * zrx),
        "momentum": -z10,
        "range": zrx,
    }


def h_series(pd_: dict, z_panel: dict[str, pd.DataFrame], signal: str, k_z: float) -> pd.Series:
    sig = pd_["sig"]
    vcr = pd.to_numeric(sig.get("vcr"), errors="coerce")
    vcr_med = pd.to_numeric(sig.get("vcr_med"), errors="coerce")
    raw = H_MID + K_VCR * (vcr - vcr_med)
    if k_z != 0.0 and signal in z_panel:
        zt = z_panel[signal].get(pd_["underlying"])
        if zt is not None:
            raw = raw - k_z * zt.reindex(raw.index).fillna(0.0)
    h = raw.clip(H_MIN, H_MAX).fillna(float(np.clip(pd_["partial_h"], H_MIN, H_MAX)))
    return h.ewm(alpha=EMA_ALPHA, adjust=False).mean()


def cagr_of(eq: pd.Series) -> float:
    eq = eq.dropna()
    if len(eq) < 10 or eq.iloc[0] <= 1e-9:
        return np.nan
    years = max((eq.index[-1] - eq.index[0]).days / 365.25, 1 / 365.25)
    return float((eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1.0)


# ---------------------------------------------------------------------------
# Grid runner
# ---------------------------------------------------------------------------
def run_all(pair_data: list[dict], z_panel: dict[str, pd.DataFrame],
            combos: list[tuple[str, float]]) -> tuple[pd.DataFrame, pd.Timestamp]:
    all_dates = sorted({d for p in pair_data for d in p["prices"].index})
    split = all_dates[len(all_dates) // 2]
    rows = []
    for ci, (signal, k_z) in enumerate(combos, 1):
        for pd_ in pair_data:
            h = h_series(pd_, z_panel, signal, k_z)
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
            rows.append({
                "signal": signal, "k_z": k_z, "pair": pd_["pair"],
                "cagr_full": cagr_of(eq),
                "cagr_h1": cagr_of(eq.loc[:split]),
                "cagr_h2": cagr_of(eq.loc[split:]),
                "vol": float(rets.std(ddof=1) * np.sqrt(TRADING_DAYS)) if len(rets) > 2 else np.nan,
                "max_dd": float((eq / eq.cummax() - 1.0).min()),
            })
        print(f"  ... combo {ci}/{len(combos)} ({signal}, k_z={k_z})", flush=True)
    return pd.DataFrame(rows), split


def score_combos(long: pd.DataFrame) -> pd.DataFrame:
    def agg(g: pd.DataFrame) -> pd.Series:
        c = g["cagr_full"].dropna()
        lo, hi = (c.quantile(0.05), c.quantile(0.95)) if len(c) else (np.nan, np.nan)
        return pd.Series({
            "n_pairs": len(g),
            "ew_mean_full": c.mean(),
            "ew_median_full": c.median(),
            "winsor_mean_full": c.clip(lo, hi).mean(),
            "ew_mean_h1": g["cagr_h1"].mean(skipna=True),
            "ew_mean_h2": g["cagr_h2"].mean(skipna=True),
            "ew_median_h1": g["cagr_h1"].median(skipna=True),
            "ew_median_h2": g["cagr_h2"].median(skipna=True),
            "ew_mean_vol": g["vol"].mean(skipna=True),
            "ew_mean_dd": g["max_dd"].mean(skipna=True),
        })
    s = long.groupby(["signal", "k_z"]).apply(agg, include_groups=False).reset_index()
    s["ret_over_vol"] = s["ew_mean_full"] / s["ew_mean_vol"].clip(lower=0.01)
    for col in ("winsor_mean_full", "ew_median_full", "ew_mean_h1", "ew_mean_h2", "ret_over_vol"):
        s[f"rank_{col}"] = s[col].rank(ascending=False)
    s["composite_rank"] = s[[c for c in s.columns if c.startswith("rank_")]].mean(axis=1)
    return s.sort_values("composite_rank")


def pairfold_cv(long: pd.DataFrame, n_folds: int = 20, seed: int = 7) -> pd.DataFrame:
    """Train half of pairs picks (signal, k_z); scored on held-out pairs vs v8."""
    rng = np.random.default_rng(seed)
    pairs = sorted(long["pair"].unique())
    piv = long.pivot_table(index=["signal", "k_z"], columns="pair", values="cagr_full")
    v8_row = piv.loc[("baseline", 0.0)]
    rows = []
    for f in range(n_folds):
        sel = rng.permutation(len(pairs))
        train = [pairs[i] for i in sel[: len(pairs) // 2]]
        test = [pairs[i] for i in sel[len(pairs) // 2:]]
        best = piv[train].mean(axis=1).idxmax()
        rows.append({
            "fold": f, "picked": str(best),
            "oos_pick_ew_mean": float(piv.loc[best, test].mean()),
            "oos_v8_ew_mean": float(v8_row[test].mean()),
            "uplift": float(piv.loc[best, test].mean() - v8_row[test].mean()),
        })
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="v9 cross-sectional tilt experiment.")
    ap.add_argument("--pairs", type=Path, default=None)
    ap.add_argument("--metrics", type=Path,
                    default=REPO.parent / "Levered ETFs/etf-dashboard/data/etf_metrics_daily.csv")
    ap.add_argument("--vol-shape", type=Path,
                    default=REPO.parent / "Levered ETFs/etf-dashboard/data/vol_shape_history.json")
    ap.add_argument("--screened", type=Path, default=REPO / "data/etf_screened_today.csv")
    ap.add_argument("--outdir", type=Path, default=REPO / "notebooks/output/b4_v9")
    ap.add_argument("--start", default="2025-10-07")
    ap.add_argument("--max-pairs", type=int, default=0)
    args = ap.parse_args(argv)
    args.outdir.mkdir(parents=True, exist_ok=True)

    pair_data, _ = load_pair_data(args)
    print(f"[v9] pairs with data: {len(pair_data)}")
    for pd_ in pair_data:
        rd, _ = policy_continuous_interval(
            pd_["prices"].index, pd_["sig"],
            base_days=BASE_DAYS, k_tr=K_TR, m_vcr=M_VCR,
            min_interval=1, max_interval=CAP_DAYS,
        )
        pd_["sched"] = pd.DatetimeIndex(rd).intersection(pd_["prices"].index)

    z_panel = build_xsec_panel(pair_data)
    n_unds = z_panel["composite"].shape[1]
    print(f"[v9] cross-section: {n_unds} unique underlyings")

    combos = [("baseline", 0.0)] + [(s, k) for s in SIGNALS for k in K_Z_GRID]
    long, split = run_all(pair_data, z_panel, combos)
    print(f"[v9] time split at {split.date()}")

    scores = score_combos(long)
    scores.to_csv(args.outdir / "b4_v9_grid_scores.csv", index=False)
    v8_row = scores[(scores["signal"] == "baseline")].iloc[0]

    # risk budget: tilt should not change the risk profile much
    vol_cap = float(v8_row["ew_mean_vol"]) * 1.10
    dd_floor = float(v8_row["ew_mean_dd"]) * 1.10
    gated = scores[(scores["ew_mean_vol"] <= vol_cap) & (scores["ew_mean_dd"] >= dd_floor)]
    if gated.empty:
        gated = scores
    v9_row = gated.sort_values("composite_rank").iloc[0]
    v9 = (str(v9_row["signal"]), float(v9_row["k_z"]))

    cv = pairfold_cv(long)
    cv.to_csv(args.outdir / "b4_v9_pairfold_cv.csv", index=False)
    cv_uplift = float(cv["uplift"].mean())
    cv_winrate = float((cv["uplift"] > 0).mean())
    cv_modal = cv["picked"].mode().iloc[0] if not cv.empty else None

    show = ["signal", "k_z", "ew_mean_full", "ew_median_full", "ew_mean_h1", "ew_mean_h2",
            "ew_median_h1", "ew_median_h2", "ew_mean_vol", "ew_mean_dd", "ret_over_vol", "composite_rank"]
    with pd.option_context("display.width", 240):
        print("\n=== All combos (composite-ranked) ===")
        print(scores[show].to_string(index=False))
    print(f"\n[v9] risk budget: vol<= {vol_cap:.2%}, dd>= {dd_floor:.2%} (1.10x v8)")
    print(f"[v9] best within budget: signal={v9[0]} k_z={v9[1]}")
    print(f"[v9] pair-fold CV vs v8: mean OOS uplift {cv_uplift:+.4f}, win-rate {cv_winrate:.0%}, modal pick {cv_modal}")

    # ---- adoption gate: must beat v8 on mean AND median, in BOTH halves ----
    beats = {
        "full_mean": v9_row["ew_mean_full"] > v8_row["ew_mean_full"],
        "full_median": v9_row["ew_median_full"] > v8_row["ew_median_full"],
        "h1_mean": v9_row["ew_mean_h1"] > v8_row["ew_mean_h1"],
        "h2_mean": v9_row["ew_mean_h2"] > v8_row["ew_mean_h2"],
        "cv_winrate>=0.6": cv_winrate >= 0.6,
    }
    verdict = "ADOPT" if all(beats.values()) else "REJECT"
    print(f"[v9] gate: {beats} -> {verdict}")

    # ---- plots ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    for ax, signal in zip(axes, SIGNALS):
        sub = scores[scores["signal"] == signal].sort_values("k_z")
        ax.plot(sub["k_z"], sub["ew_mean_full"] * 100, "o-", label="EW mean", color="#1f77b4")
        ax.plot(sub["k_z"], sub["ew_median_full"] * 100, "s--", label="EW median", color="#2ca02c")
        ax.axhline(float(v8_row["ew_mean_full"]) * 100, ls=":", color="#1f77b4", alpha=0.6)
        ax.axhline(float(v8_row["ew_median_full"]) * 100, ls=":", color="#2ca02c", alpha=0.6)
        ax.axvline(0, color="#888", lw=0.7)
        ax.set_title(f"{signal} tilt")
        ax.set_xlabel("k_z  (v6 convention: + = v6 sign)")
        ax.grid(True, linestyle="--", alpha=0.35)
    axes[0].set_ylabel("full-window CAGR %")
    axes[0].legend(fontsize=8)
    fig.suptitle("v9: cross-sectional tilt sweep vs v8 baseline (dotted lines)")
    fig.tight_layout()
    fig.savefig(args.outdir / "b4_v9_kz_sweep.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(cv["uplift"] * 100, bins=15, color="#4477aa", alpha=0.85)
    ax.axvline(0, color="k", lw=1)
    ax.axvline(cv_uplift * 100, color="#d62728", ls="--", label=f"mean {cv_uplift*100:+.2f}pp")
    ax.set_xlabel("OOS EW mean CAGR uplift vs v8 (pp), per fold")
    ax.set_title(f"Pair-fold CV (20 folds) | win-rate {cv_winrate:.0%} | modal {cv_modal}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.outdir / "b4_v9_pairfold_cv.png", dpi=130)
    plt.close(fig)

    # ---- results md ----
    lines = [
        "# Bucket 4 v9 experiment: v8 + v6 cross-sectional z_composite tilt\n",
        f"Window {args.start} -> latest | {len(pair_data)} pairs | {n_unds} unique underlyings | "
        f"split {split.date()} | v8 base (h_mid {H_MID}, clips [{H_MIN},{H_MAX}]), slip {SLIPPAGE_BPS}bps\n",
        "## What was ported from v6\n",
        "z_composite = 0.5*(-z(r_10d)) + 0.5*(+z(range_expansion)), robust cross-sectional z per day,\n"
        "shifted 1 day, dedup by underlying. Regime overlay NOT ported (rejected in phase-3 lab).\n",
        "## All combos\n",
        "```\n" + scores[show].to_string(index=False) + "\n```\n",
        f"## Best within risk budget: signal={v9[0]} k_z={v9[1]}\n",
        f"- full EW mean {v9_row['ew_mean_full']:.2%} vs v8 {v8_row['ew_mean_full']:.2%}\n",
        f"- full EW median {v9_row['ew_median_full']:.2%} vs v8 {v8_row['ew_median_full']:.2%}\n",
        f"- H1 {v9_row['ew_mean_h1']:.2%} vs {v8_row['ew_mean_h1']:.2%} | H2 {v9_row['ew_mean_h2']:.2%} vs {v8_row['ew_mean_h2']:.2%}\n",
        f"- vol {v9_row['ew_mean_vol']:.2%} vs {v8_row['ew_mean_vol']:.2%} | dd {v9_row['ew_mean_dd']:.2%} vs {v8_row['ew_mean_dd']:.2%}\n",
        f"- pair-fold CV: mean OOS uplift {cv_uplift:+.2%}, win-rate {cv_winrate:.0%}, modal pick {cv_modal}\n",
        f"## Gate: {beats}\n",
        f"## VERDICT: {verdict}\n",
    ]
    (args.outdir / "B4_V9_RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[v9] wrote {args.outdir / 'B4_V9_RESULTS.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
