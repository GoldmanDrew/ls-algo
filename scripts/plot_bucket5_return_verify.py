#!/usr/bin/env python3
"""Create diagnostic plots from a B5 return-verification output directory.

Example:
    python scripts/plot_bucket5_return_verify.py --input data/runs/2026-07-18/b5_return_verify
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def read_csv(path: Path) -> pd.DataFrame | None:
    """Return a CSV frame, printing a clear skip message when unavailable."""
    if not path.exists():
        print(f"Skipping {path.name}: file not found.")
        return None
    try:
        return pd.read_csv(path)
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        print(f"Skipping {path.name}: {exc}")
        return None


def numeric(frame: pd.DataFrame, column: str) -> pd.Series | None:
    if column not in frame:
        return None
    values = pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return values if not values.empty else None


def save(fig: plt.Figure, output: Path) -> None:
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(output)


def plot_theta_scatter(frame: pd.DataFrame, output: Path) -> None:
    observed, model = numeric(frame, "observed_mid"), numeric(frame, "model_price")
    if observed is None or model is None:
        print("Skipping theta scatter: observed_mid or model_price is unavailable.")
        return
    valid = frame.index.intersection(observed.index).intersection(model.index)
    observed, model = observed.loc[valid], model.loc[valid]
    fig, ax = plt.subplots(figsize=(7, 6))
    vix = numeric(frame, "vix")
    if vix is not None:
        valid = valid.intersection(vix.index)
        observed, model, vix = observed.loc[valid], model.loc[valid], vix.loc[valid]
        points = ax.scatter(observed, model, c=vix, cmap="viridis", s=28, alpha=0.8, edgecolors="none")
        fig.colorbar(points, ax=ax, label="VIX")
    else:
        ax.scatter(observed, model, color="#4477aa", s=28, alpha=0.8, edgecolors="none")
    limits = (min(observed.min(), model.min()), max(observed.max(), model.max()))
    ax.plot(limits, limits, color="#cc6677", lw=1.6, label="45°: model = observed")
    ax.set(xlabel="Observed option mid", ylabel="Model price", title="Theta option-mark replay", xlim=limits, ylim=limits)
    ax.legend(frameon=False, loc="upper left")
    ax.grid(alpha=0.2)
    save(fig, output)


def plot_error_distribution(frame: pd.DataFrame, output: Path) -> None:
    errors = numeric(frame, "abs_error")
    if errors is None:
        print("Skipping error distribution: abs_error is unavailable.")
        return
    median, p95 = errors.median(), errors.quantile(0.95)
    fig, (hist, ecdf) = plt.subplots(1, 2, figsize=(11, 4.5))
    hist.hist(errors, bins="auto", color="#4477aa", alpha=0.85)
    x = np.sort(errors)
    ecdf.step(x, np.arange(1, len(x) + 1) / len(x), where="post", color="#4477aa", lw=1.8)
    for ax in (hist, ecdf):
        ax.axvline(median, color="#228833", ls="--", label=f"Median {median:.1f}")
        ax.axvline(p95, color="#cc6677", ls="--", label=f"p95 {p95:.1f}")
        ax.legend(frameon=False)
        ax.grid(alpha=0.2)
        ax.set_xlabel("Absolute model-minus-quote error")
    hist.set(title="Absolute error histogram", ylabel="Observations")
    ecdf.set(title="Absolute error ECDF", ylabel="Cumulative fraction", ylim=(0, 1.02))
    save(fig, output)


def plot_error_drivers(frame: pd.DataFrame, output: Path) -> None:
    candidates = [("vix", "VIX"), ("dte", "Days to expiry")]
    if {"strike", "spot"}.issubset(frame.columns):
        frame = frame.copy()
        frame["moneyness"] = pd.to_numeric(frame["strike"], errors="coerce") / pd.to_numeric(frame["spot"], errors="coerce")
        candidates.insert(1, ("moneyness", "Strike / spot"))
    available = [(name, label) for name, label in candidates if numeric(frame, name) is not None]
    errors = numeric(frame, "abs_error")
    if errors is None or not available:
        print("Skipping error drivers: no compatible error and driver columns.")
        return
    fig, axes = plt.subplots(1, len(available), figsize=(5 * len(available), 4.5), squeeze=False)
    for ax, (name, label) in zip(axes[0], available):
        driver = numeric(frame, name)
        valid = errors.index.intersection(driver.index)
        ax.scatter(driver.loc[valid], errors.loc[valid], color="#4477aa", s=24, alpha=0.65, edgecolors="none")
        ax.set(title=f"Absolute error vs {label}", xlabel=label, ylabel="Absolute error")
        ax.grid(alpha=0.2)
    save(fig, output)


def plot_sensitivity(frame: pd.DataFrame, output: Path) -> None:
    required = {"slippage_bps", "borrow_multiple", "CAGR", "MaxDD"}
    if not required.issubset(frame.columns):
        print("Skipping sensitivity heatmaps: required columns unavailable.")
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, metric, title, cmap in zip(
        axes, ("CAGR", "MaxDD"), ("CAGR by costs", "Maximum drawdown by costs"), ("YlGnBu", "YlOrRd")
    ):
        grid = frame.pivot_table(index="borrow_multiple", columns="slippage_bps", values=metric, aggfunc="mean")
        image = ax.imshow(grid.values, aspect="auto", cmap=cmap)
        ax.set(
            title=title,
            xlabel="ETP slippage (bps)",
            ylabel="Borrow multiple",
            xticks=np.arange(len(grid.columns)),
            xticklabels=[f"{x:g}" for x in grid.columns],
            yticks=np.arange(len(grid.index)),
            yticklabels=[f"{x:g}×" for x in grid.index],
        )
        for row in range(grid.shape[0]):
            for col in range(grid.shape[1]):
                ax.text(col, row, f"{grid.iloc[row, col]:.2%}", ha="center", va="center", fontsize=9)
        fig.colorbar(image, ax=ax, format=lambda x, _: f"{x:.1%}")
    save(fig, output)


def choose_stress_packet(input_dir: Path) -> Path | None:
    preferred = input_dir / "event_aug_2024.csv"
    if preferred.exists():
        return preferred
    packets = sorted(input_dir.glob("event_*.csv"))
    if not packets:
        return None
    return max(packets, key=lambda path: pd.read_csv(path).shape[0])


def plot_stress_packet(path: Path | None, output_dir: Path) -> None:
    if path is None:
        print("Skipping stress packet: no event_*.csv files found.")
        return
    frame = read_csv(path)
    if frame is None or "date" not in frame:
        print(f"Skipping stress packet {path.name}: date column unavailable.")
        return
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["date"]).sort_values("date")
    panels = [
        ("combined_nav", "NAV", "#4477aa"),
        ("put_mtm", "Put MTM", "#228833"),
        ("cash_realized", "Realized put cash", "#cc6677"),
        ("uvix_exposure", "UVIX exposure", "#aa4499"),
        ("svix_exposure", "SVIX exposure", "#ddaa33"),
    ]
    panels = [(col, label, color) for col, label, color in panels if numeric(frame, col) is not None]
    if not panels:
        print(f"Skipping stress packet {path.name}: no plottable columns.")
        return
    fig, axes = plt.subplots(len(panels), 1, figsize=(10, 2.4 * len(panels)), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, (column, label, color) in zip(axes, panels):
        ax.plot(frame["date"], pd.to_numeric(frame[column], errors="coerce"), color=color, marker="o", ms=3, lw=1.7)
        ax.set_ylabel(label)
        ax.grid(alpha=0.2)
    axes[0].set_title(f"Stress packet: {path.stem.removeprefix('event_').replace('_', ' ')}")
    axes[-1].set_xlabel("Date")
    save(fig, output_dir / f"stress_packet_{path.stem.removeprefix('event_')}.png")


def plot_attribution(frame: pd.DataFrame, output: Path) -> None:
    if "date" not in frame:
        print("Skipping attribution: date column unavailable.")
        return
    columns = [col for col in ("uvix_pnl", "svix_pnl", "tbill_pnl", "put_mtm_change", "put_cash_flow", "redeployment_pnl") if col in frame]
    if len(columns) < 2:
        print("Skipping attribution: not enough P&L components.")
        return
    dates = pd.to_datetime(frame["date"], errors="coerce")
    values = frame[columns].apply(pd.to_numeric, errors="coerce").fillna(0)
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.stackplot(dates, *[values[col] for col in columns], labels=[col.replace("_", " ") for col in columns], alpha=0.78)
    ax.axhline(0, color="black", lw=0.8)
    ax.set(title="Daily return attribution", xlabel="Date", ylabel="Daily P&L")
    ax.legend(frameon=False, ncol=3, loc="upper left", fontsize=8)
    ax.grid(alpha=0.2)
    save(fig, output)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="B5 return-verification run directory")
    parser.add_argument("--output", type=Path, default=None, help="Plot directory (defaults to <input>/plots)")
    args = parser.parse_args()
    input_dir = args.input.resolve()
    output_dir = (args.output or input_dir / "plots").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("default")

    theta = read_csv(input_dir / "theta_mark_replay.csv")
    if theta is not None:
        plot_theta_scatter(theta, output_dir / "theta_model_vs_observed.png")
        plot_error_distribution(theta, output_dir / "theta_abs_error_distribution.png")
        plot_error_drivers(theta, output_dir / "theta_abs_error_drivers.png")
    sensitivity = read_csv(input_dir / "sensitivity.csv")
    if sensitivity is not None:
        plot_sensitivity(sensitivity, output_dir / "sensitivity_heatmaps.png")
    plot_stress_packet(choose_stress_packet(input_dir), output_dir)
    attribution = read_csv(input_dir / "daily_attribution.csv")
    if attribution is not None:
        plot_attribution(attribution, output_dir / "daily_attribution.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
