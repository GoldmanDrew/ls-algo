#!/usr/bin/env python3
"""
Recompute selected-strategy metrics with spike/crash-robust MaxDD.

The extended-era book has a Feb 2018 put-pricing artifact (~+50% in one day,
then a sharp giveback). Raw MaxDD uses that fake ATH as the high-water mark
all the way into 2020. For DD estimation we neutralize spike-and-crash episodes:
any day with return > spike_cap, plus the following days until equity is back
at/below the pre-spike level (or a short max window).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "data/runs/2026-07-13/bucket5/spy_put_grid/selected_spy_puts_equity.csv"
OUT_DIR = SRC.parent
SPIKE_CAP = 0.20  # one-day gain that triggers episode neutralization
MAX_CRASH_DAYS = 15  # max trading days to unwind after a spike

LABELS = {
    "spy00_puts1.00x": "All cash\nputs 1.00x",
    "spy00_puts1.50x": "All cash\nputs 1.50x",
    "spy20_puts1.00x": "20% SPY\nputs 1.00x",
    "spy20_puts1.50x": "20% SPY\nputs 1.50x",
    "spy30_puts1.00x": "30% SPY\nputs 1.00x",
    "spy30_puts1.50x": "30% SPY\nputs 1.50x",
    "spy40_puts1.00x": "40% SPY\nputs 1.00x",
    "spy40_puts1.50x": "40% SPY\nputs 1.50x",
}
COLORS = {
    "spy00_puts1.00x": "#94a3b8",
    "spy00_puts1.50x": "#334155",
    "spy20_puts1.00x": "#60a5fa",
    "spy20_puts1.50x": "#1d4ed8",
    "spy30_puts1.00x": "#2dd4bf",
    "spy30_puts1.50x": "#0f766e",
    "spy40_puts1.00x": "#fbbf24",
    "spy40_puts1.50x": "#b45309",
}


def ann_stats(eq: pd.Series) -> tuple[float, float, float]:
    eq = eq.dropna()
    ret = eq.pct_change().fillna(0.0)
    n = len(eq)
    total = float(eq.iloc[-1] / eq.iloc[0] - 1.0)
    cagr = (1.0 + total) ** (252.0 / (n - 1)) - 1.0
    vol = float(ret.std() * np.sqrt(252))
    maxdd = float(eq.div(eq.cummax()).sub(1.0).min())
    return cagr, vol, maxdd


def neutralize_spike_crash(
    eq: pd.Series,
    *,
    spike_cap: float = SPIKE_CAP,
    max_crash_days: int = MAX_CRASH_DAYS,
) -> tuple[pd.Series, list[tuple[pd.Timestamp, pd.Timestamp]]]:
    """Zero out spike day + subsequent crash until back to pre-spike level."""
    ret = eq.pct_change().fillna(0.0).copy()
    values = eq.to_numpy(dtype=float)
    idx = eq.index
    neutralized: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    i = 1
    n = len(ret)
    while i < n:
        if float(ret.iloc[i]) > spike_cap:
            pre = values[i - 1]
            start = i
            j = i
            end = min(n - 1, i + max_crash_days)
            # include spike day and keep going while still above pre-spike
            while j <= end:
                ret.iloc[j] = 0.0
                # stop early once path would be back at/below pre-spike using
                # original levels as the unwind signal
                if j > i and values[j] <= pre:
                    break
                j += 1
            neutralized.append((idx[start], idx[min(j, n - 1)]))
            i = j + 1
            continue
        i += 1
    path = (1.0 + ret).cumprod() * float(eq.iloc[0])
    path.name = eq.name
    return path, neutralized


def robust_maxdd(eq: pd.Series) -> tuple[float, pd.Series, list]:
    path, episodes = neutralize_spike_crash(eq)
    dd = float(path.div(path.cummax()).sub(1.0).min())
    return dd, path, episodes


def main() -> int:
    df = pd.read_csv(SRC, index_col=0, parse_dates=True)

    # diagnose one series
    sample = df["spy30_puts1.00x"].dropna()
    _, _, episodes = robust_maxdd(sample)
    print("Neutralized spike/crash episodes (30% SPY / puts 1.00x):")
    for a, b in episodes:
        print(f"  {a.date()} -> {b.date()}")

    rows = []
    robust_paths = {}
    for col, lab in LABELS.items():
        eq = df[col].dropna()
        cagr, vol, maxdd_raw = ann_stats(eq)
        maxdd_rob, path, _ = robust_maxdd(eq)
        robust_paths[col] = path
        rows.append(
            {
                "key": col,
                "label": lab,
                "CAGR": cagr,
                "Vol": vol,
                "MaxDD_raw": maxdd_raw,
                "MaxDD_robust": maxdd_rob,
                "Calmar_raw": cagr / abs(maxdd_raw) if maxdd_raw < 0 else np.nan,
                "Calmar_robust": cagr / abs(maxdd_rob) if maxdd_rob < 0 else np.nan,
            }
        )
    stats = pd.DataFrame(rows)
    stats.to_csv(OUT_DIR / "selected_spy_puts_metrics_robust_dd.csv", index=False)
    pd.DataFrame(robust_paths).to_csv(OUT_DIR / "selected_spy_puts_equity_robust_dd_path.csv")

    keys = list(LABELS)
    x = np.arange(len(keys))
    bar_colors = [COLORS[k] for k in keys]
    short = [LABELS[k] for k in keys]

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 7.8), constrained_layout=True)

    ax = axes[0, 0]
    vals = stats["CAGR"].values * 100
    bars = ax.bar(x, vals, color=bar_colors, edgecolor="white", width=0.72)
    ax.set_title("CAGR (unchanged)")
    ax.set_ylabel("%")
    ax.set_xticks(x)
    ax.set_xticklabels(short, fontsize=7.5)
    ax.grid(axis="y", alpha=0.3)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.15, f"{v:.1f}%", ha="center", va="bottom", fontsize=7.5)

    ax = axes[0, 1]
    vals = stats["MaxDD_robust"].values * 100
    bars = ax.bar(x, vals, color=bar_colors, edgecolor="white", width=0.72)
    ax.set_title("Max Drawdown (spike/crash episodes neutralized)")
    ax.set_ylabel("%")
    ax.set_xticks(x)
    ax.set_xticklabels(short, fontsize=7.5)
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(0, color="#94a3b8", lw=0.8)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v - 0.5, f"{v:.1f}%", ha="center", va="top", fontsize=7.5)

    ax = axes[1, 0]
    vals = stats["Vol"].values * 100
    bars = ax.bar(x, vals, color=bar_colors, edgecolor="white", width=0.72)
    ax.set_title("Volatility (unchanged)")
    ax.set_ylabel("%")
    ax.set_xticks(x)
    ax.set_xticklabels(short, fontsize=7.5)
    ax.grid(axis="y", alpha=0.3)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.2, f"{v:.1f}%", ha="center", va="bottom", fontsize=7.5)

    ax = axes[1, 1]
    vals = stats["Calmar_robust"].values
    bars = ax.bar(x, vals, color=bar_colors, edgecolor="white", width=0.72)
    ax.set_title("Calmar (using robust MaxDD)")
    ax.set_ylabel("ratio")
    ax.set_xticks(x)
    ax.set_xticklabels(short, fontsize=7.5)
    ax.grid(axis="y", alpha=0.3)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=7.5)

    fig.suptitle(
        "Bucket 5 — DD ignores spike-then-crash artifacts (e.g. Feb 2018 +50% day)\n"
        "2008-01-02 to 2026-07-13",
        fontsize=11,
        fontweight="bold",
    )
    out = OUT_DIR / "selected_spy_puts_metrics_robust_dd.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)

    # raw vs robust DD comparison
    fig, ax = plt.subplots(figsize=(11.5, 4.8), constrained_layout=True)
    w = 0.38
    ax.bar(x - w / 2, stats["MaxDD_raw"] * 100, width=w, color="#94a3b8", label="Raw MaxDD", edgecolor="white")
    ax.bar(
        x + w / 2,
        stats["MaxDD_robust"] * 100,
        width=w,
        color=bar_colors,
        label="Robust MaxDD (spike+crash neutralized)",
        edgecolor="white",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(short, fontsize=7.5)
    ax.set_ylabel("%")
    ax.set_title("Max drawdown: raw vs spike/crash-robust")
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(0, color="#94a3b8", lw=0.8)
    for i, (raw, rob) in enumerate(zip(stats["MaxDD_raw"] * 100, stats["MaxDD_robust"] * 100)):
        ax.text(i - w / 2, raw - 1.0, f"{raw:.1f}", ha="center", va="top", fontsize=6.5, color="#475569")
        ax.text(i + w / 2, rob - 1.0, f"{rob:.1f}", ha="center", va="top", fontsize=6.5, color="#0f172a")
    cmp = OUT_DIR / "selected_spy_puts_maxdd_raw_vs_robust.png"
    fig.savefig(cmp, dpi=150)
    plt.close(fig)

    print(stats[["label", "CAGR", "Vol", "MaxDD_raw", "MaxDD_robust", "Calmar_raw", "Calmar_robust"]].to_string(index=False))
    print(f"saved {out}")
    print(f"saved {cmp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
