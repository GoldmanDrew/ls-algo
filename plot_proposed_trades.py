#!/usr/bin/env python3
"""
plot_proposed_trades.py — Visualise proposed_trades.csv for a given run date.

Pipeline:
  daily_screener.py → etf_screened_today.csv
  → generate_trade_plan.py → proposed_trades.csv
  → plot_proposed_trades.py → proposed_trades_plot.png   ← this script

Usage:
  python plot_proposed_trades.py                  # uses today's date
  python plot_proposed_trades.py 2026-03-03       # specific run date
  python plot_proposed_trades.py --show           # open plot window after saving
"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless by default; --show switches to interactive
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
RUNS_DIR     = PROJECT_ROOT / "data" / "runs"

_SLEEVE_COLORS: dict[str, str] = {
    "core_leveraged":  "steelblue",
    "whitelist_stock": "coral",
}
_SLEEVE_DEFAULT = "grey"


# ──────────────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────────────

def load_proposed(run_date: str) -> pd.DataFrame:
    path = RUNS_DIR / run_date / "proposed_trades.csv"
    if not path.exists():
        raise FileNotFoundError(f"proposed_trades.csv not found for {run_date}: {path}")
    df = pd.read_csv(path)

    numeric_cols = [
        "target_long_usd", "target_short_usd",
        "borrow_current", "gross_decay_annual",
        "expected_decay_annual", "net_decay_annual",
        "vol_underlying_annual", "vol_etf_annual", "Beta",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "purgatory" in df.columns:
        df["purgatory"] = df["purgatory"].astype(str).str.strip().str.lower().map(
            {"true": True, "false": False, "1": True, "0": False}
        ).fillna(False)

    return df


def active_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Rows with a non-zero allocation (purgatory=False and at least one leg)."""
    return df[
        (df["purgatory"] == False) &  # noqa: E712
        ((df["target_long_usd"].fillna(0).abs() > 1) |
         (df["target_short_usd"].fillna(0).abs() > 1))
    ].copy()


# ──────────────────────────────────────────────────────────────────────────────
# Panel helpers
# ──────────────────────────────────────────────────────────────────────────────

def _bar_label(ax, val: float, label: str, *, side: str, fontsize: int = 7,
               pad: float = 200) -> None:
    """Place a ticker label just beyond the end of a bar."""
    ha  = "left"  if side == "right" else "right"
    x   = val + pad if side == "right" else val - pad
    ax.text(x, 0, label, ha=ha, va="center", fontsize=fontsize, clip_on=True)


def plot_allocation(ax: plt.Axes, df: pd.DataFrame) -> None:
    """
    Panel 1 — diverging bars.
    Right (steelblue): target_long_usd  = underlying we are buying.
    Left  (tomato):    target_short_usd = ETF we are shorting (already negative).
    """
    df = df.sort_values("target_long_usd", ascending=True).reset_index(drop=True)
    y  = np.arange(len(df))

    # Determine bar padding from data range
    xmax = max(df["target_long_usd"].abs().max(),
               df["target_short_usd"].abs().max(), 1)
    pad  = xmax * 0.012

    # Long (underlying) bars
    ax.barh(y, df["target_long_usd"], color="steelblue", alpha=0.85, height=0.7)
    # Short (ETF) bars
    ax.barh(y, df["target_short_usd"], color="tomato",    alpha=0.85, height=0.7)

    # Ticker labels at bar ends
    for i, row in df.iterrows():
        # underlying label on the right
        if abs(row["target_long_usd"]) > 1:
            ax.text(row["target_long_usd"] + pad, i,
                    str(row["Underlying"]),
                    ha="left", va="center", fontsize=6.5, color="steelblue")
        # ETF label on the left
        if abs(row["target_short_usd"]) > 1:
            ax.text(row["target_short_usd"] - pad, i,
                    str(row["ETF"]),
                    ha="right", va="center", fontsize=6.5, color="tomato")

    # Y-axis: "ETF → UNDERLYING"
    labels = [f"{r.ETF} → {r.Underlying}" for _, r in df.iterrows()]
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7.5)

    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Notional (USD)", fontsize=9)
    ax.set_title("Allocation — ETF Short (left)  vs  Underlying Long (right)", fontsize=11, pad=6)
    ax.xaxis.grid(True, alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    long_patch  = mpatches.Patch(color="steelblue", alpha=0.85, label="Underlying (long)")
    short_patch = mpatches.Patch(color="tomato",    alpha=0.85, label="ETF (short)")
    ax.legend(handles=[long_patch, short_patch], fontsize=8, loc="lower right")


def plot_decay_score(ax: plt.Axes, df: pd.DataFrame) -> None:
    """
    Panel 2 — single horizontal bar chart of net_decay_annual (the sizing score).
    NaN → 0.
    """
    df = df.copy()
    df["_score"] = df["net_decay_annual"].fillna(0) * 100
    df = df.sort_values("_score", ascending=True).reset_index(drop=True)
    y  = np.arange(len(df))

    bars = ax.barh(y, df["_score"], color="seagreen", alpha=0.85, height=0.7)

    # Value labels
    xmax = df["_score"].max() if df["_score"].max() > 0 else 1
    pad  = xmax * 0.012
    for i, val in enumerate(df["_score"]):
        if val > 0:
            ax.text(val + pad, i, f"{val:.1f}%",
                    ha="left", va="center", fontsize=6.5)

    ax.set_yticks(y)
    ax.set_yticklabels(df["ETF"].astype(str), fontsize=7.5)
    ax.set_xlabel("Net Decay Score (% annual)", fontsize=9)
    ax.set_title("Net Decay Score — Position Sizing Score (Annual %)", fontsize=11, pad=6)
    ax.xaxis.grid(True, alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)


def plot_borrow(ax: plt.Axes, df: pd.DataFrame) -> None:
    """Panel 3 — borrow rate by ETF."""
    df = df.copy()
    df["_borrow"] = df["borrow_current"].fillna(0) * 100
    df = df.sort_values("_borrow", ascending=True).reset_index(drop=True)
    y  = np.arange(len(df))

    ax.barh(y, df["_borrow"], color="mediumpurple", alpha=0.85, height=0.7)

    xmax = df["_borrow"].max() if df["_borrow"].max() > 0 else 1
    pad  = xmax * 0.012
    for i, val in enumerate(df["_borrow"]):
        ax.text(val + pad, i, f"{val:.1f}%",
                ha="left", va="center", fontsize=6.5)

    ax.set_yticks(y)
    ax.set_yticklabels(df["ETF"].astype(str), fontsize=7.5)
    ax.set_xlabel("Borrow Rate (% annual)", fontsize=9)
    ax.set_title("Current Borrow Rate by ETF", fontsize=11, pad=6)
    ax.xaxis.grid(True, alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def make_plot(run_date: str, *, show: bool = False) -> Path:
    df  = load_proposed(run_date)
    act = active_rows(df)

    if act.empty:
        print(f"[WARN] No active (non-purgatory, non-zero) rows found for {run_date}.")

    n = len(act)
    # Scale height: ~0.22in per row, minimum 12in
    row_height = 0.22
    panel_h    = max(12, n * row_height)
    fig, axes  = plt.subplots(3, 1, figsize=(22, panel_h * 3 + 2))

    plot_allocation(axes[0], act)
    plot_decay_score(axes[1], act)
    plot_borrow(axes[2], act)

    fig.suptitle(
        f"Proposed Trades — {run_date}   ({n} active positions)",
        fontsize=13, fontweight="bold", y=1.002,
    )
    fig.tight_layout(pad=2.5)

    out = RUNS_DIR / run_date / "proposed_trades_plot.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[OK] Saved -> {out}")

    if show:
        matplotlib.use("TkAgg")  # switch to interactive backend
        plt.show()
    else:
        plt.close(fig)

    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "run_date", nargs="?",
        default=date.today().isoformat(),
        help="Run date YYYY-MM-DD (default: today)",
    )
    parser.add_argument(
        "--show", action="store_true",
        help="Open the plot in an interactive window after saving",
    )
    args = parser.parse_args()

    try:
        make_plot(args.run_date, show=args.show)
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
