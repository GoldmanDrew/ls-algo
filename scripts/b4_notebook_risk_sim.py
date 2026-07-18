"""Notebook-facing B4 Monte Carlo / tail risk charts (dashboard-style)."""
from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.bucket4_cadence_risk_opt import (
    HORIZON,
    RET_FLOOR,
    mc_block_bootstrap,
    mc_parametric,
    perf_from_returns,
    port_returns,
    tail_stats,
)
from scripts.build_bucket4_risk_sim import build_portfolio
from scripts.bucket4_risk_sim_universe import load_risk_sim_universe
from scripts.sizing_tilt_cadence_bt import load_price_panel


def run_b4_notebook_risk_sim(
    *,
    run_date: str,
    start: str = "2025-01-01",
    n_mc: int = 2000,
    block_len: int = 10,
    seed: int = 7,
    min_days: int = 30,
) -> dict[str, Any] | None:
    """Build portfolio returns + MC drawdown tails for notebook plots."""
    try:
        uni = load_risk_sim_universe(run_date)
    except Exception:
        # Fallback: empty proposed book — try panel-only universe from pair stats later
        return None
    panel = load_price_panel(run_date)
    built = build_portfolio(uni, panel, start, min_days, min_days_short=min_days)
    if built is None:
        return None
    pr, pairs, blk, ret_df, book_gross = built
    prv = pr.dropna()
    arr = prv.to_numpy(dtype=float)
    if len(arr) < 40:
        return None
    perf = perf_from_returns(prv)
    rng = np.random.default_rng(seed)
    boot = mc_block_bootstrap(arr, n_mc, HORIZON, block_len, rng)
    lap = mc_parametric(arr, n_mc, HORIZON, rng, "laplace")

    # Equity fan: block-bootstrap cumulative paths (subset for speed).
    n_fan = min(400, n_mc)
    m = len(arr)
    horizon = min(HORIZON, max(60, m))
    paths = np.empty((n_fan, horizon))
    for i in range(n_fan):
        starts = rng.integers(0, m, size=int(np.ceil(horizon / block_len)))
        path = np.concatenate(
            [np.take(arr, range(s, s + block_len), mode="wrap") for s in starts]
        )[:horizon]
        paths[i] = np.cumprod(1.0 + np.clip(path, RET_FLOOR, None))

    return {
        "port_returns": prv,
        "pairs": pairs,
        "cadence": blk,
        "book_gross": book_gross,
        "perf": perf,
        "boot_dds": boot,
        "lap_dds": lap,
        "boot_stats": tail_stats(boot, "boot"),
        "lap_stats": tail_stats(lap, "lap"),
        "fan_paths": paths,
        "n_mc": n_mc,
        "horizon": horizon,
    }


def plot_b4_risk_sim(result: dict[str, Any]) -> plt.Figure | None:
    if not result:
        return None
    boot = np.asarray(result["boot_dds"], dtype=float)
    lap = np.asarray(result["lap_dds"], dtype=float)
    paths = np.asarray(result["fan_paths"], dtype=float)
    perf = result.get("perf") or {}

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))

    ax = axes[0]
    if boot.size:
        ax.hist(np.abs(boot), bins=40, color="#4c72b0", alpha=0.75, density=True, label="block bootstrap")
    if lap.size:
        ax.hist(np.abs(lap), bins=40, color="#c44e52", alpha=0.45, density=True, label="Laplace MC")
    ax.set_xlabel("|max drawdown| over horizon")
    ax.set_ylabel("density")
    ax.set_title(
        f"B4 MC maxDD (horizon={result.get('horizon')}d, n={result.get('n_mc')})\n"
        f"realized CAGR={perf.get('cagr', float('nan')):.1%}  "
        f"hist maxDD={perf.get('maxdd', float('nan')):.1%}"
    )
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    bs = result.get("boot_stats") or {}
    ax.text(
        0.98,
        0.98,
        f"boot p95={bs.get('boot_dd_p95', float('nan')):.1%}\n"
        f"boot p99={bs.get('boot_dd_p99', float('nan')):.1%}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        family="monospace",
    )

    ax = axes[1]
    if paths.size:
        q = np.nanpercentile(paths, [5, 25, 50, 75, 95], axis=0)
        t = np.arange(paths.shape[1])
        ax.fill_between(t, q[0], q[4], color="#9ecae1", alpha=0.5, label="5–95%")
        ax.fill_between(t, q[1], q[3], color="#6baed6", alpha=0.55, label="25–75%")
        ax.plot(t, q[2], color="#08519c", lw=1.6, label="median")
        ax.axhline(1.0, color="#9ca3af", lw=0.8)
    ax.set_xlabel("path day")
    ax.set_ylabel("equity (start=1)")
    ax.set_title("B4 equity fan (block-bootstrap paths)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig
