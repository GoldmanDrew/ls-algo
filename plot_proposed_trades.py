#!/usr/bin/env python3
"""
plot_proposed_trades.py — Visualise bucket-level proposed trades for a run date.

Outputs:
  - proposed_trades_bucket_1_plot.png   (beta > 1.5 from proposed_trades.csv)
  - proposed_trades_bucket_2_plot.png   (0 < beta <= 1.5 from proposed_trades.csv)
"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
RUNS_DIR = PROJECT_ROOT / "data" / "runs"


def _to_bool_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().map(
        {"true": True, "false": False, "1": True, "0": False}
    ).fillna(False)


def load_proposed(run_date: str) -> pd.DataFrame:
    path = RUNS_DIR / run_date / "proposed_trades.csv"
    if not path.exists():
        raise FileNotFoundError(f"proposed_trades.csv not found for {run_date}: {path}")
    df = pd.read_csv(path)

    for col in ("long_usd", "short_usd", "borrow_current", "net_decay_annual", "Beta"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = np.nan

    if "purgatory" in df.columns:
        df["purgatory"] = _to_bool_series(df["purgatory"])
    else:
        df["purgatory"] = False

    return df


def active_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df[
        (df["purgatory"] == False)  # noqa: E712
        & ((df["long_usd"].fillna(0).abs() > 1) | (df["short_usd"].fillna(0).abs() > 1))
    ].copy()


def split_stock_buckets(df_active: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    b = pd.to_numeric(df_active["Beta"], errors="coerce")
    b1 = df_active[b > 1.5].copy()
    b2 = df_active[(b > 0) & (b <= 1.5)].copy()
    return b1, b2


def plot_allocation(ax: plt.Axes, df: pd.DataFrame, title: str) -> None:
    if df.empty:
        ax.text(0.5, 0.5, "No proposed trades in this bucket", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title, fontsize=11, pad=6)
        ax.set_axis_off()
        return

    df = df.sort_values("long_usd", ascending=True).reset_index(drop=True)
    y = np.arange(len(df))
    xmax = max(df["long_usd"].abs().max(), df["short_usd"].abs().max(), 1)
    pad = xmax * 0.012

    ax.barh(y, df["long_usd"], color="steelblue", alpha=0.85, height=0.7)
    ax.barh(y, df["short_usd"], color="tomato", alpha=0.85, height=0.7)

    for i, row in df.iterrows():
        if abs(row["long_usd"]) > 1:
            ax.text(row["long_usd"] + pad, i, str(row.get("Underlying", "")),
                    ha="left", va="center", fontsize=6.5, color="steelblue")
        if abs(row["short_usd"]) > 1:
            ax.text(row["short_usd"] - pad, i, str(row.get("ETF", "")),
                    ha="right", va="center", fontsize=6.5, color="tomato")

    labels = [f"{r.ETF} -> {r.Underlying}" for _, r in df.iterrows()]
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Notional (USD)", fontsize=9)
    ax.set_title(title, fontsize=11, pad=6)
    ax.xaxis.grid(True, alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    long_patch = mpatches.Patch(color="steelblue", alpha=0.85, label="Underlying (long)")
    short_patch = mpatches.Patch(color="tomato", alpha=0.85, label="ETF (short)")
    ax.legend(handles=[long_patch, short_patch], fontsize=8, loc="lower right")


def plot_decay_score(ax: plt.Axes, df: pd.DataFrame) -> None:
    if df.empty:
        ax.text(0.5, 0.5, "No decay data", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return

    d = df.copy()
    d["_score"] = d["net_decay_annual"].fillna(0) * 100
    d = d.sort_values("_score", ascending=True).reset_index(drop=True)
    y = np.arange(len(d))
    ax.barh(y, d["_score"], color="seagreen", alpha=0.85, height=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(d["ETF"].astype(str), fontsize=7.5)
    ax.set_xlabel("Net Decay Score (% annual)", fontsize=9)
    ax.set_title("Net Decay Score", fontsize=11, pad=6)
    ax.xaxis.grid(True, alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)


def plot_borrow(ax: plt.Axes, df: pd.DataFrame) -> None:
    if df.empty:
        ax.text(0.5, 0.5, "No borrow data", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return

    d = df.copy()
    d["_borrow"] = d["borrow_current"].fillna(0) * 100
    d = d.sort_values("_borrow", ascending=True).reset_index(drop=True)
    y = np.arange(len(d))
    ax.barh(y, d["_borrow"], color="mediumpurple", alpha=0.85, height=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(d["ETF"].astype(str), fontsize=7.5)
    ax.set_xlabel("Borrow Rate (% annual)", fontsize=9)
    ax.set_title("Current Borrow Rate by ETF", fontsize=11, pad=6)
    ax.xaxis.grid(True, alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)


def render_stock_bucket(run_date: str, bucket_df: pd.DataFrame, bucket_label: str, out_path: Path) -> Path:
    n = len(bucket_df)
    row_height = 0.22
    panel_h = max(7, n * row_height)
    fig, ax = plt.subplots(1, 1, figsize=(20, panel_h + 1.5))

    # Keep only the proposed position allocation panel.
    plot_allocation(ax, bucket_df, f"{bucket_label} Allocation — ETF short vs Underlying long")

    fig.suptitle(f"{bucket_label} Proposed Trades — {run_date} ({n} rows)", fontsize=13, fontweight="bold", y=1.002)
    fig.tight_layout(pad=2.2)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def make_bucket_plots(run_date: str) -> list[Path]:
    proposed = load_proposed(run_date)
    act = active_rows(proposed)
    b1, b2 = split_stock_buckets(act)

    out_dir = RUNS_DIR / run_date
    out_dir.mkdir(parents=True, exist_ok=True)
    out_b1 = out_dir / "proposed_trades_bucket_1_plot.png"
    out_b2 = out_dir / "proposed_trades_bucket_2_plot.png"

    render_stock_bucket(run_date, b1, "Bucket 1 (Beta > 1.5)", out_b1)
    render_stock_bucket(run_date, b2, "Bucket 2 (0 < Beta <= 1.5)", out_b2)

    return [out_b1, out_b2]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_date", nargs="?", default=date.today().isoformat(),
                        help="Run date YYYY-MM-DD (default: today)")
    parser.add_argument("--show", action="store_true",
                        help="Retained for compatibility; plotting is saved to files in headless mode.")
    args = parser.parse_args()

    try:
        outputs = make_bucket_plots(args.run_date)
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    for p in outputs:
        print(f"[OK] Saved -> {p}")
    if args.show:
        print("[INFO] --show requested; script runs headless and saves PNG files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
