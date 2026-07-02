"""Matplotlib charts for Bucket 5 Streamlit lab."""

from __future__ import annotations

from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from scripts.bucket5_lab_helpers import backwardation_mask, harvest_event_dates, series_to_frame


def _shade_spans(ax, spans: list[tuple[str, str]], *, alpha: float = 0.12, color: str = "#991b1b") -> None:
    for s, e in spans:
        ax.axvspan(pd.Timestamp(s), pd.Timestamp(e), color=color, alpha=alpha, lw=0)


def fig_attribution(series: dict, *, r_lo: float = 0.88) -> plt.Figure:
    """Decompose combined equity into carry, bills, puts, harvest, redeploy."""
    idx = None
    parts: dict[str, pd.Series] = {}
    for key in ("sleeve_equity", "base_equity", "put_mtm", "realized_cum", "redeploy_extra", "combined_equity"):
        df = series_to_frame(series, key)
        if df is not None:
            parts[key] = df[key]
            idx = df.index if idx is None else idx.union(df.index)
    if idx is None or "combined_equity" not in parts:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, "No attribution series", ha="center", va="center")
        ax.axis("off")
        return fig

    idx = idx.sort_values()
    sleeve = parts.get("sleeve_equity", pd.Series(0.0, index=idx)).reindex(idx).ffill().fillna(0)
    base = parts.get("base_equity", sleeve).reindex(idx).ffill().fillna(0)
    tbill = (base - sleeve).clip(lower=0)
    puts = parts.get("put_mtm", pd.Series(0.0, index=idx)).reindex(idx).ffill().fillna(0)
    realized = parts.get("realized_cum", pd.Series(0.0, index=idx)).reindex(idx).ffill().fillna(0)
    redeploy = parts.get("redeploy_extra", pd.Series(0.0, index=idx)).reindex(idx).ffill().fillna(0)
    combined = parts["combined_equity"].reindex(idx).ffill().fillna(0)

    fig, axes = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]})
    ax0, ax1 = axes
    spans = backwardation_mask(series, r_lo=r_lo)
    _shade_spans(ax0, spans)
    _shade_spans(ax1, spans)

    ax0.plot(combined.index, combined / 1e6, color="#0f766e", lw=2.0, label="Combined")
    ax0.plot(base.index, base / 1e6, color="#2563eb", lw=1.2, alpha=0.85, label="Sleeve + T-bills")
    ax0.plot((base + puts).index, (base + puts) / 1e6, color="#7c3aed", lw=1.0, alpha=0.8, label="+ Put MTM")
    ax0.plot((base + puts + realized + redeploy).index, (base + puts + realized + redeploy) / 1e6,
             color="#16a34a", lw=0.9, alpha=0.7, ls="--", label="+ Harvest + redeploy (components)")

    for d in harvest_event_dates(series):
        ax0.axvline(pd.Timestamp(d), color="#16a34a", lw=0.8, alpha=0.55)

    ax0.set_ylabel("Equity ($M)")
    ax0.set_title("Component attribution (red shading = backwardation / stress)")
    ax0.legend(loc="upper left", fontsize=8)
    ax0.grid(alpha=0.25)

    ax1.stackplot(
        idx,
        (sleeve / 1e6).values,
        (tbill / 1e6).values,
        (puts / 1e6).values,
        (realized / 1e6).values,
        (redeploy / 1e6).values,
        labels=["Carry sleeve", "T-bill yield", "Put MTM", "Harvested (cum)", "Redeploy extra"],
        alpha=0.75,
    )
    ax1.set_ylabel("Layer ($M)")
    ax1.legend(loc="upper left", fontsize=7, ncol=2)
    ax1.grid(alpha=0.25)
    ax1.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax1.xaxis.get_major_locator()))
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def fig_equity_drawdown(series: dict, *, r_lo: float = 0.88) -> plt.Figure:
    eq = series_to_frame(series, "combined_equity")
    dd = series_to_frame(series, "drawdown")
    fig, axes = plt.subplots(2, 1, figsize=(11, 5), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    spans = backwardation_mask(series, r_lo=r_lo)
    if eq is not None:
        _shade_spans(axes[0], spans)
        axes[0].plot(eq.index, eq["combined_equity"] / 1e6, color="#0f766e", lw=1.8)
        for d in harvest_event_dates(series):
            axes[0].axvline(pd.Timestamp(d), color="#16a34a", lw=0.8, alpha=0.6)
        axes[0].set_ylabel("Equity ($M)")
        axes[0].set_title("Combined equity (green ticks = monetization harvests)")
        axes[0].grid(alpha=0.25)
    if dd is not None:
        _shade_spans(axes[1], spans)
        axes[1].fill_between(dd.index, dd["drawdown"] * 100, 0, color="#991b1b", alpha=0.35)
        axes[1].plot(dd.index, dd["drawdown"] * 100, color="#991b1b", lw=1.0)
        axes[1].set_ylabel("Drawdown (%)")
        axes[1].grid(alpha=0.25)
    fig.tight_layout()
    return fig
