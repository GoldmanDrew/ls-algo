#!/usr/bin/env python3
"""
Plot selected Bucket 5 paths: all-cash + 20/30/40% SPY idle, at baseline and +50% puts.

Uses cached grid equity where available; computes missing all-cash (0% SPY) paths.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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

GRID_EQUITY = (
    REPO / "data" / "runs" / "2026-07-13" / "bucket5" / "spy_put_grid" / "grid_equity.csv"
)
OUT_DIR = REPO / "data" / "runs" / pd.Timestamp.today().strftime("%Y-%m-%d") / "bucket5" / "spy_put_grid"


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cached = pd.read_csv(GRID_EQUITY, index_col=0, parse_dates=True)

    # Need all-cash (0% SPY) at 1.00x and 1.50x puts
    print("Computing all-cash (0% SPY) paths for puts 1.00x and 1.50x ...")
    panel = load_vol_panel(start=EXTENDED_START, use_synthetic=True)
    spx = load_spx_spot(panel.index.min().strftime("%Y-%m-%d"))
    spy = load_series("SPY", start=panel.index.min().strftime("%Y-%m-%d")).rename("spy")
    iv = (panel["vix"] / 100.0).rename("iv")
    base_cfg = production_config()

    base0 = _build_idle_base(panel, spy, spy_idle_frac=0.0, cfg=base_cfg)
    series = {}
    for put_mult in (1.00, 1.50):
        cfg = replace(base_cfg, rungs=_scale_rungs(base_cfg.rungs, put_mult))
        res = _attach_puts(panel, spx, iv, base0, cfg)
        key = f"spy00_puts{put_mult:.2f}x"
        series[key] = res["bt"]["combined_equity"]
        print(f"  {key}: final=${series[key].iloc[-1]:,.0f}")

    # Pull requested SPY idle paths from cache
    wanted = [
        ("spy00_puts1.00x", "All cash · puts 1.00×", "#64748b", "-"),
        ("spy00_puts1.50x", "All cash · puts 1.50×", "#334155", "--"),
        ("spy20_puts1.00x", "20% SPY · puts 1.00×", "#2563eb", "-"),
        ("spy20_puts1.50x", "20% SPY · puts 1.50×", "#1d4ed8", "--"),
        ("spy30_puts1.00x", "30% SPY · puts 1.00×", "#0f766e", "-"),
        ("spy30_puts1.50x", "30% SPY · puts 1.50×", "#115e59", "--"),
        ("spy40_puts1.00x", "40% SPY · puts 1.00×", "#b45309", "-"),
        ("spy40_puts1.50x", "40% SPY · puts 1.50×", "#92400e", "--"),
    ]

    plot_df = pd.DataFrame(series)
    for col, _, _, _ in wanted:
        if col.startswith("spy00"):
            continue
        plot_df[col] = cached[col]
    plot_df = plot_df.dropna(how="any")

    # Equity
    fig, axes = plt.subplots(2, 1, figsize=(12, 8.2), constrained_layout=True, sharex=True,
                             gridspec_kw={"height_ratios": [2.2, 1.0]})
    ax = axes[0]
    for col, label, color, ls in wanted:
        ax.plot(plot_df.index, plot_df[col] / 1e6, label=label, color=color, ls=ls, lw=1.6)
    ax.set_ylabel("Equity ($M)")
    ax.set_title(
        "Bucket 5 — all-cash vs 20/30/40% SPY idle, baseline vs +50% puts\n"
        f"{plot_df.index.min().date()} → {plot_df.index.max().date()} (extended / synthetic pre-2022)"
    )
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    ax.set_xlim(plot_df.index.min(), plot_df.index.max())

    # Drawdowns
    ax2 = axes[1]
    for col, label, color, ls in wanted:
        dd = plot_df[col].div(plot_df[col].cummax()).sub(1.0) * 100
        ax2.plot(plot_df.index, dd, label=label, color=color, ls=ls, lw=1.1)
    ax2.set_ylabel("Drawdown (%)")
    ax2.set_title("Drawdowns")
    ax2.grid(alpha=0.3)
    ax2.set_xlim(plot_df.index.min(), plot_df.index.max())

    out = OUT_DIR / "selected_spy_puts_equity.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)

    # also save the series used
    series_out = OUT_DIR / "selected_spy_puts_equity.csv"
    plot_df[[c for c, *_ in wanted]].to_csv(series_out)
    print(f"saved plot   -> {out}")
    print(f"saved series -> {series_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
