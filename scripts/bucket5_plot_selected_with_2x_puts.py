#!/usr/bin/env python3
"""
Selected Bucket 5 paths including 2.00x puts, with robust-DD metrics chart.

SPY idle: 0% (all cash), 20%, 30%, 40%
Puts: 1.00x, 1.50x, 2.00x
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.bucket5_data import load_series, load_vol_panel  # noqa: E402
from scripts.bucket5_insurance_bt import EXTENDED_START, production_config  # noqa: E402
from scripts.bucket5_put_overlay import load_spx_spot  # noqa: E402
from scripts.bucket5_spy_put_grid_bt import (  # noqa: E402
    _attach_puts,
    _build_idle_base,
    _scale_rungs,
)
from scripts.bucket5_plot_robust_dd_metrics import robust_maxdd, ann_stats  # noqa: E402

OUT_DIR = (
    REPO
    / "data"
    / "runs"
    / pd.Timestamp.today().strftime("%Y-%m-%d")
    / "bucket5"
    / "spy_put_grid"
)
EXISTING = OUT_DIR / "selected_spy_puts_equity.csv"
GRID_EQUITY = OUT_DIR / "grid_equity.csv"

SPY_FRACS = (0.0, 0.20, 0.30, 0.40)
PUT_MULTS = (1.00, 1.50, 2.00)

LABELS = {
    "spy00_puts1.00x": "All cash\nputs 1.00x",
    "spy00_puts1.50x": "All cash\nputs 1.50x",
    "spy00_puts2.00x": "All cash\nputs 2.00x",
    "spy20_puts1.00x": "20% SPY\nputs 1.00x",
    "spy20_puts1.50x": "20% SPY\nputs 1.50x",
    "spy20_puts2.00x": "20% SPY\nputs 2.00x",
    "spy30_puts1.00x": "30% SPY\nputs 1.00x",
    "spy30_puts1.50x": "30% SPY\nputs 1.50x",
    "spy30_puts2.00x": "30% SPY\nputs 2.00x",
    "spy40_puts1.00x": "40% SPY\nputs 1.00x",
    "spy40_puts1.50x": "40% SPY\nputs 1.50x",
    "spy40_puts2.00x": "40% SPY\nputs 2.00x",
}
COLORS = {
    "spy00_puts1.00x": "#cbd5e1",
    "spy00_puts1.50x": "#64748b",
    "spy00_puts2.00x": "#0f172a",
    "spy20_puts1.00x": "#93c5fd",
    "spy20_puts1.50x": "#3b82f6",
    "spy20_puts2.00x": "#1e3a8a",
    "spy30_puts1.00x": "#5eead4",
    "spy30_puts1.50x": "#14b8a6",
    "spy30_puts2.00x": "#115e59",
    "spy40_puts1.00x": "#fcd34d",
    "spy40_puts1.50x": "#f59e0b",
    "spy40_puts2.00x": "#92400e",
}
LINESTYLES = {
    1.00: "-",
    1.50: "--",
    2.00: ":",
}


def _key(spy_frac: float, put_mult: float) -> str:
    return f"spy{int(round(spy_frac * 100)):02d}_puts{put_mult:.2f}x"


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    series: dict[str, pd.Series] = {}

    if EXISTING.is_file():
        old = pd.read_csv(EXISTING, index_col=0, parse_dates=True)
        for c in old.columns:
            series[c] = old[c]
    if GRID_EQUITY.is_file():
        grid = pd.read_csv(GRID_EQUITY, index_col=0, parse_dates=True)
        for c in grid.columns:
            if c not in series:
                series[c] = grid[c]

    need = [_key(s, p) for s in SPY_FRACS for p in PUT_MULTS if _key(s, p) not in series]
    if need:
        print(f"Computing missing paths: {need}")
        panel = load_vol_panel(start=EXTENDED_START, use_synthetic=True)
        spx = load_spx_spot(panel.index.min().strftime("%Y-%m-%d"))
        spy = load_series("SPY", start=panel.index.min().strftime("%Y-%m-%d")).rename("spy")
        iv = (panel["vix"] / 100.0).rename("iv")
        base_cfg = production_config()

        bases = {}
        for spy_frac in sorted({float(k.split("_")[0].replace("spy", "")) / 100.0 for k in need}):
            print(f"  base path SPY idle {spy_frac:.0%} ...")
            bases[spy_frac] = _build_idle_base(panel, spy, spy_idle_frac=spy_frac, cfg=base_cfg)

        for key in need:
            spy_frac = int(key[3:5]) / 100.0
            put_mult = float(key.split("puts")[1].replace("x", ""))
            print(f"  -> {key} ...")
            cfg = replace(base_cfg, rungs=_scale_rungs(base_cfg.rungs, put_mult))
            res = _attach_puts(panel, spx, iv, bases[spy_frac], cfg)
            series[key] = res["bt"]["combined_equity"]
            print(f"     final=${series[key].iloc[-1]:,.0f}")

    cols = [_key(s, p) for s in SPY_FRACS for p in PUT_MULTS]
    plot_df = pd.DataFrame({c: series[c] for c in cols}).dropna(how="any")
    equity_csv = OUT_DIR / "selected_spy_puts_equity_with_2x.csv"
    plot_df.to_csv(equity_csv)

    # ----- equity + drawdown chart -----
    fig, axes = plt.subplots(
        2, 1, figsize=(12.5, 8.5), constrained_layout=True, sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.0]},
    )
    ax = axes[0]
    for spy_frac in SPY_FRACS:
        for put_mult in PUT_MULTS:
            col = _key(spy_frac, put_mult)
            spy_lab = "All cash" if spy_frac == 0 else f"{int(spy_frac*100)}% SPY"
            ax.plot(
                plot_df.index,
                plot_df[col] / 1e6,
                label=f"{spy_lab} · puts {put_mult:.2f}×",
                color=COLORS[col],
                ls=LINESTYLES[put_mult],
                lw=1.55,
            )
    ax.set_ylabel("Equity ($M)")
    ax.set_title(
        "Bucket 5 — all-cash / 20/30/40% SPY idle × puts 1.00× / 1.50× / 2.00×\n"
        f"{plot_df.index.min().date()} → {plot_df.index.max().date()}"
    )
    ax.legend(loc="upper left", fontsize=7.5, ncol=3)
    ax.grid(alpha=0.3)

    ax2 = axes[1]
    for spy_frac in SPY_FRACS:
        for put_mult in PUT_MULTS:
            col = _key(spy_frac, put_mult)
            dd = plot_df[col].div(plot_df[col].cummax()).sub(1.0) * 100
            ax2.plot(plot_df.index, dd, color=COLORS[col], ls=LINESTYLES[put_mult], lw=1.05)
    ax2.set_ylabel("Drawdown (%)")
    ax2.set_title("Raw drawdowns (includes Feb 2018 spike artifact)")
    ax2.grid(alpha=0.3)
    equity_png = OUT_DIR / "selected_spy_puts_equity_with_2x.png"
    fig.savefig(equity_png, dpi=150)
    plt.close(fig)

    # ----- metrics with robust DD -----
    rows = []
    for col in cols:
        eq = plot_df[col]
        cagr, vol, maxdd_raw = ann_stats(eq)
        maxdd_rob, _, _ = robust_maxdd(eq)
        rows.append(
            {
                "key": col,
                "label": LABELS[col],
                "spy_idle_frac": int(col[3:5]) / 100.0,
                "put_mult": float(col.split("puts")[1].replace("x", "")),
                "CAGR": cagr,
                "Vol": vol,
                "MaxDD_raw": maxdd_raw,
                "MaxDD_robust": maxdd_rob,
                "Calmar_raw": cagr / abs(maxdd_raw) if maxdd_raw < 0 else np.nan,
                "Calmar_robust": cagr / abs(maxdd_rob) if maxdd_rob < 0 else np.nan,
            }
        )
    stats = pd.DataFrame(rows)
    stats_csv = OUT_DIR / "selected_spy_puts_metrics_with_2x.csv"
    stats.to_csv(stats_csv, index=False)

    keys = cols
    x = np.arange(len(keys))
    bar_colors = [COLORS[k] for k in keys]
    short = [LABELS[k] for k in keys]

    fig, axes = plt.subplots(2, 2, figsize=(14.5, 8.2), constrained_layout=True)

    ax = axes[0, 0]
    vals = stats["CAGR"].values * 100
    bars = ax.bar(x, vals, color=bar_colors, edgecolor="white", width=0.75)
    ax.set_title("CAGR")
    ax.set_ylabel("%")
    ax.set_xticks(x)
    ax.set_xticklabels(short, fontsize=6.8)
    ax.grid(axis="y", alpha=0.3)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.12, f"{v:.1f}", ha="center", va="bottom", fontsize=6.5)

    ax = axes[0, 1]
    vals = stats["MaxDD_robust"].values * 100
    bars = ax.bar(x, vals, color=bar_colors, edgecolor="white", width=0.75)
    ax.set_title("Adjusted MaxDD (spike/crash neutralized)")
    ax.set_ylabel("%")
    ax.set_xticks(x)
    ax.set_xticklabels(short, fontsize=6.8)
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(0, color="#94a3b8", lw=0.8)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v - 0.45, f"{v:.1f}", ha="center", va="top", fontsize=6.5)

    ax = axes[1, 0]
    vals = stats["Vol"].values * 100
    bars = ax.bar(x, vals, color=bar_colors, edgecolor="white", width=0.75)
    ax.set_title("Volatility")
    ax.set_ylabel("%")
    ax.set_xticks(x)
    ax.set_xticklabels(short, fontsize=6.8)
    ax.grid(axis="y", alpha=0.3)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.15, f"{v:.1f}", ha="center", va="bottom", fontsize=6.5)

    ax = axes[1, 1]
    vals = stats["Calmar_robust"].values
    bars = ax.bar(x, vals, color=bar_colors, edgecolor="white", width=0.75)
    ax.set_title("Calmar (CAGR / |adjusted MaxDD|)")
    ax.set_ylabel("ratio")
    ax.set_xticks(x)
    ax.set_xticklabels(short, fontsize=6.8)
    ax.grid(axis="y", alpha=0.3)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=6.5)

    fig.suptitle(
        "Bucket 5 selected strategies incl. 2.00× puts — 2008-01-02 to 2026-07-13\n"
        "Adjusted MaxDD neutralizes Feb 2018 spike/crash artifact",
        fontsize=11,
        fontweight="bold",
    )
    metrics_png = OUT_DIR / "selected_spy_puts_metrics_with_2x.png"
    fig.savefig(metrics_png, dpi=150)
    plt.close(fig)

    # console table
    show = stats.copy()
    show["spy"] = show["spy_idle_frac"].map(lambda v: "All cash" if v == 0 else f"{100*v:.0f}% SPY")
    show["puts"] = show["put_mult"].map(lambda v: f"{v:.2f}×")
    show["CAGR"] = show["CAGR"].map(lambda v: f"{100*v:.2f}%")
    show["Vol"] = show["Vol"].map(lambda v: f"{100*v:.2f}%")
    show["Adj MaxDD"] = show["MaxDD_robust"].map(lambda v: f"{100*v:.2f}%")
    show["Calmar"] = show["Calmar_robust"].map(lambda v: f"{v:.2f}")
    print("\n=== Metrics (adjusted MaxDD) ===")
    print(show[["spy", "puts", "CAGR", "Adj MaxDD", "Vol", "Calmar"]].to_string(index=False))
    print(f"\nsaved equity  -> {equity_png}")
    print(f"saved metrics -> {metrics_png}")
    print(f"saved csv     -> {stats_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
