#!/usr/bin/env python3
"""
plot_proposed_trades.py — Visualise bucket-level proposed trades for a run date.

Outputs:
  - proposed_trades_bucket_1_plot.png   (beta > 1.5, stock sleeves only)
  - proposed_trades_bucket_2_plot.png   (yieldboost sleeve)
  - proposed_trades_bucket_4_plot.png   (inverse_decay_bucket4 sleeve: short ETF + hedge)
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
B4_SLEEVE = "inverse_decay_bucket4"


def _to_bool_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().map(
        {"true": True, "false": False, "1": True, "0": False}
    ).fillna(False)


def load_proposed(run_date: str) -> pd.DataFrame:
    path = RUNS_DIR / run_date / "proposed_trades.csv"
    if not path.exists():
        raise FileNotFoundError(f"proposed_trades.csv not found for {run_date}: {path}")
    df = pd.read_csv(path)

    numeric_cols = (
        "long_usd",
        "short_usd",
        "borrow_current",
        "net_decay_annual",
        "Beta",
        # Optional optimal-target columns (added by generate_trade_plan dual pipeline). When
        # absent (older runs), these default to NaN; render falls back to executable-only bars.
        "optimal_long_usd",
        "optimal_short_usd",
        "optimal_gross_target_usd",
        "gross_target_usd",
        "liquidity_gap_usd",
        "executable_pct_of_optimal",
    )
    for col in numeric_cols:
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
    """Rows that are either being traded **today** (executable nonzero) OR have a
    standing **optimal** position (so we render the hatched bar even when day-liquidity
    pushed the executable to zero)."""
    long_exe = df["long_usd"].fillna(0).abs() > 1
    short_exe = df["short_usd"].fillna(0).abs() > 1
    long_opt = df.get("optimal_long_usd", pd.Series(0.0, index=df.index)).fillna(0).abs() > 1
    short_opt = df.get("optimal_short_usd", pd.Series(0.0, index=df.index)).fillna(0).abs() > 1
    return df[
        (df["purgatory"] == False)  # noqa: E712
        & (long_exe | short_exe | long_opt | short_opt)
    ].copy()


def split_buckets(df_active: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if "sleeve" not in df_active.columns:
        stock = df_active.copy()
        b4 = df_active.iloc[0:0].copy()
        b = pd.to_numeric(stock["Beta"], errors="coerce")
        b1 = stock[b > 1.5].copy()
        b2 = stock[(b > 0) & (b <= 1.5)].copy()
        return b1, b2, b4

    is_b4 = df_active["sleeve"].astype(str).eq(B4_SLEEVE)
    stock = df_active[~is_b4].copy()
    b4 = df_active[is_b4].copy()
    slv = stock["sleeve"].astype(str)
    b1 = stock[slv.eq("core_leveraged")].copy()
    b2 = stock[slv.eq("yieldboost")].copy()
    return b1, b2, b4


def _has_optimal(df: pd.DataFrame) -> bool:
    if "optimal_long_usd" not in df.columns or "optimal_short_usd" not in df.columns:
        return False
    o_l = pd.to_numeric(df["optimal_long_usd"], errors="coerce").abs().fillna(0.0)
    o_s = pd.to_numeric(df["optimal_short_usd"], errors="coerce").abs().fillna(0.0)
    return bool((o_l + o_s).sum() > 1.0)


def plot_allocation(ax: plt.Axes, df: pd.DataFrame, title: str, *, bucket4: bool = False) -> None:
    if df.empty:
        ax.text(0.5, 0.5, "No proposed trades in this bucket", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title, fontsize=11, pad=6)
        ax.set_axis_off()
        return

    show_optimal = _has_optimal(df)
    if bucket4:
        # Both legs stored as negative USD; plot magnitudes symmetric about zero like B1/B2.
        long_usd = -pd.to_numeric(df["long_usd"], errors="coerce").fillna(0)
        short_usd = pd.to_numeric(df["short_usd"], errors="coerce").fillna(0)
        opt_long_usd = (
            -pd.to_numeric(df["optimal_long_usd"], errors="coerce").fillna(0) if show_optimal
            else pd.Series(0.0, index=df.index)
        )
        opt_short_usd = (
            pd.to_numeric(df["optimal_short_usd"], errors="coerce").fillna(0) if show_optimal
            else pd.Series(0.0, index=df.index)
        )
        df = df.assign(_plot_long=long_usd, _plot_short=short_usd,
                       _plot_opt_long=opt_long_usd, _plot_opt_short=opt_short_usd)
        sort_col = "_plot_opt_long" if show_optimal else "_plot_long"
        legend_long = mpatches.Patch(color="steelblue", alpha=0.85, label="Executable underlying hedge")
        legend_short = mpatches.Patch(color="tomato", alpha=0.85, label="Executable inverse ETF")
    else:
        df = df.assign(
            _plot_long=pd.to_numeric(df["long_usd"], errors="coerce").fillna(0),
            _plot_short=pd.to_numeric(df["short_usd"], errors="coerce").fillna(0),
            _plot_opt_long=pd.to_numeric(df.get("optimal_long_usd", 0.0), errors="coerce").fillna(0),
            _plot_opt_short=pd.to_numeric(df.get("optimal_short_usd", 0.0), errors="coerce").fillna(0),
        )
        sort_col = "_plot_opt_long" if show_optimal else "long_usd"
        legend_long = mpatches.Patch(color="steelblue", alpha=0.85, label="Executable Underlying (long)")
        legend_short = mpatches.Patch(color="tomato", alpha=0.85, label="Executable ETF (short)")
    df = df.sort_values(sort_col, ascending=True).reset_index(drop=True)
    y = np.arange(len(df))
    xmax = max(
        df["_plot_long"].abs().max(),
        df["_plot_short"].abs().max(),
        df["_plot_opt_long"].abs().max(),
        df["_plot_opt_short"].abs().max(),
        1,
    )
    pad = xmax * 0.012

    if show_optimal:
        # Optimal bar is wider, hatched, drawn behind. Executable bar is narrower, solid, in front.
        ax.barh(y, df["_plot_opt_long"], color="none", edgecolor="steelblue", linewidth=1.0,
                hatch="///", height=0.85, alpha=0.55)
        ax.barh(y, df["_plot_opt_short"], color="none", edgecolor="tomato", linewidth=1.0,
                hatch="///", height=0.85, alpha=0.55)
        ax.barh(y, df["_plot_long"], color="steelblue", alpha=0.95, height=0.55)
        ax.barh(y, df["_plot_short"], color="tomato", alpha=0.95, height=0.55)
    else:
        ax.barh(y, df["_plot_long"], color="steelblue", alpha=0.85, height=0.7)
        ax.barh(y, df["_plot_short"], color="tomato", alpha=0.85, height=0.7)

    for i, row in df.iterrows():
        lu = row["_plot_long"]
        su = row["_plot_short"]
        opt_lu = row.get("_plot_opt_long", 0.0)
        opt_su = row.get("_plot_opt_short", 0.0)
        # Use the wider (optimal) extent for label placement so labels don't collide with the
        # outer hatched bar. Falls back to executable when optimal is unavailable.
        long_extent = opt_lu if show_optimal and abs(opt_lu) > abs(lu) else lu
        short_extent = opt_su if show_optimal and abs(opt_su) > abs(su) else su
        if bucket4:
            lu_abs = abs(float(row["long_usd"])) if pd.notna(row["long_usd"]) else 0.0
            su_abs = abs(float(row["short_usd"])) if pd.notna(row["short_usd"]) else 0.0
            if lu_abs > 1 or abs(opt_lu) > 1:
                ax.text(long_extent + pad, i, str(row.get("Underlying", "")),
                        ha="left", va="center", fontsize=6.5, color="steelblue")
            if su_abs > 1 or abs(opt_su) > 1:
                ax.text(short_extent - pad, i, str(row.get("ETF", "")),
                        ha="right", va="center", fontsize=6.5, color="tomato")
        else:
            if abs(row["long_usd"]) > 1 or abs(opt_lu) > 1:
                ax.text(long_extent + pad, i, str(row.get("Underlying", "")),
                        ha="left", va="center", fontsize=6.5, color="steelblue")
            if abs(row["short_usd"]) > 1 or abs(opt_su) > 1:
                ax.text(short_extent - pad, i, str(row.get("ETF", "")),
                        ha="right", va="center", fontsize=6.5, color="tomato")

    labels = []
    for _, r in df.iterrows():
        base = f"{r.ETF} -> {r.Underlying}"
        # Append daily liquidity-gap annotation when present (proposed plan calls for visible
        # gap shading; matplotlib hatch above visually shows extent, this label gives the dollar).
        gap = float(r.get("liquidity_gap_usd", 0.0) or 0.0)
        if show_optimal and abs(gap) > 1000:
            base += f"  [gap ${gap/1000:,.0f}k]"
        labels.append(base)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Notional (USD)", fontsize=9)
    ax.set_title(title, fontsize=11, pad=6)
    ax.xaxis.grid(True, alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    handles = [legend_long, legend_short]
    if show_optimal:
        handles.extend([
            mpatches.Patch(facecolor="none", edgecolor="steelblue", hatch="///",
                           label="Optimal Underlying (structural-only)"),
            mpatches.Patch(facecolor="none", edgecolor="tomato", hatch="///",
                           label="Optimal ETF (structural-only)"),
        ])
    ax.legend(handles=handles, fontsize=8, loc="lower right")

    if show_optimal:
        # Bottom annotation: total executable vs optimal gross (sum of |long| + |short|) for the bucket.
        exe_total = float(df["_plot_long"].abs().sum() + df["_plot_short"].abs().sum())
        opt_total = float(df["_plot_opt_long"].abs().sum() + df["_plot_opt_short"].abs().sum())
        pct = 100.0 * exe_total / max(opt_total, 1e-9)
        ax.text(
            0.005,
            -0.10,
            f"Bucket totals — executable=${exe_total/1000:,.0f}k / optimal=${opt_total/1000:,.0f}k ({pct:.0f}%)",
            transform=ax.transAxes,
            fontsize=8,
            ha="left",
            va="top",
            color="dimgray",
        )


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


def render_stock_bucket(
    run_date: str,
    bucket_df: pd.DataFrame,
    bucket_label: str,
    out_path: Path,
    *,
    bucket4: bool = False,
) -> Path:
    n = len(bucket_df)
    row_height = 0.22
    panel_h = max(7, n * row_height)
    fig, ax = plt.subplots(1, 1, figsize=(20, panel_h + 1.5))

    # Keep only the proposed position allocation panel.
    subtitle = (
        "Allocation — inverse ETF (short) vs underlying hedge (short)"
        if bucket4
        else "Allocation — ETF short vs Underlying long"
    )
    plot_allocation(ax, bucket_df, f"{bucket_label} {subtitle}", bucket4=bucket4)

    fig.suptitle(f"{bucket_label} Proposed Trades — {run_date} ({n} rows)", fontsize=13, fontweight="bold", y=1.002)
    fig.tight_layout(pad=2.2)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def make_bucket_plots(run_date: str) -> list[Path]:
    proposed = load_proposed(run_date)
    act = active_rows(proposed)
    b1, b2, b4 = split_buckets(act)

    out_dir = RUNS_DIR / run_date
    out_dir.mkdir(parents=True, exist_ok=True)
    out_b1 = out_dir / "proposed_trades_bucket_1_plot.png"
    out_b2 = out_dir / "proposed_trades_bucket_2_plot.png"
    out_b4 = out_dir / "proposed_trades_bucket_4_plot.png"

    render_stock_bucket(run_date, b1, "Bucket 1 (core_leveraged)", out_b1)
    render_stock_bucket(run_date, b2, "Bucket 2 (yieldboost sleeve)", out_b2)
    render_stock_bucket(run_date, b4, "Bucket 4 (inverse decay)", out_b4, bucket4=True)

    return [out_b1, out_b2, out_b4]


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
