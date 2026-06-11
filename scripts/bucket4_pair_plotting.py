"""Per-pair Bucket 4 backtest chart helpers (metrics + returns panels).

Used by ``scripts/bucket4_pair_backtest_plots.py``. Conventions match
``bucket4_pair_diagnostics`` (inception / funding step stripped on day 1).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TRADING_DAYS = 252


def trading_daily_pnl(bt: pd.DataFrame, *, initial_capital: float) -> pd.Series:
    """Daily mark-to-market PnL with the pair's entry-day diff zeroed."""
    eq = bt["equity"].astype(float)
    deq = eq.diff().fillna(0.0)
    if len(deq):
        deq = deq.copy()
        deq.iloc[0] = 0.0
    return deq


def trading_return_index(bt: pd.DataFrame, *, initial_capital: float) -> pd.Series:
    """Cumulative return index starting at 1.0 (entry step stripped)."""
    deq = trading_daily_pnl(bt, initial_capital=initial_capital)
    return 1.0 + deq.cumsum() / float(initial_capital)


def leg_mtm_daily(bt: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Economic MTM PnL per short leg: shares × Δprice."""
    a = bt["a_shares"].astype(float) * bt["a_px"].astype(float).diff()
    b = bt["b_shares"].astype(float) * bt["b_px"].astype(float).diff()
    first = bt.index[0]
    a = a.fillna(0.0)
    b = b.fillna(0.0)
    if first in a.index:
        a = a.copy()
        b = b.copy()
        a.loc[first] = 0.0
        b.loc[first] = 0.0
    return a, b


def leg_return_index(bt: pd.DataFrame, *, initial_capital: float) -> tuple[pd.Series, pd.Series]:
    """Cumulative leg return indices (1.0 + cum leg PnL / initial capital)."""
    la, lb = leg_mtm_daily(bt)
    ic = float(initial_capital)
    return 1.0 + la.cumsum() / ic, 1.0 + lb.cumsum() / ic


def expanding_cagr(bt: pd.DataFrame, *, initial_capital: float, min_days: int = 20) -> pd.Series:
    """Point-in-time annualized CAGR from pair inception (trading PnL only).

    Suppressed for the first ``min_days`` observations — early-sample CAGR
    explodes numerically when a few good days get annualized.
    """
    idx = trading_return_index(bt, initial_capital=initial_capital)
    out = pd.Series(np.nan, index=bt.index, dtype=float)
    for i, (dt, nav) in enumerate(idx.items()):
        days = i + 1
        if days < min_days or nav <= 1e-9:
            continue
        years = days / TRADING_DAYS
        out.loc[dt] = float(nav ** (1.0 / years) - 1.0)
    return out


def rolling_ann_vol(bt: pd.DataFrame, window: int, *, initial_capital: float) -> pd.Series:
    """Trailing annualized vol of trading daily returns."""
    deq = trading_daily_pnl(bt, initial_capital=initial_capital)
    ret = deq / float(initial_capital)
    return ret.rolling(window, min_periods=max(5, window // 3)).std(ddof=1) * np.sqrt(TRADING_DAYS)


def full_window_metrics(bt: pd.DataFrame, *, initial_capital: float) -> dict[str, float]:
    """Static summary stats for one pair backtest."""
    deq = trading_daily_pnl(bt, initial_capital=initial_capital)
    ret = deq / float(initial_capital)
    eq = bt["equity"].astype(float)
    n = max(len(eq) - 1, 1)
    years = max(n / TRADING_DAYS, 1 / TRADING_DAYS)
    nav_end = float(trading_return_index(bt, initial_capital=initial_capital).iloc[-1])
    cagr = float(nav_end ** (1.0 / years) - 1.0) if nav_end > 1e-9 else np.nan
    vol = float(ret.std(ddof=1) * np.sqrt(TRADING_DAYS)) if len(ret) > 2 else np.nan
    dd = float(bt["drawdown"].min()) if "drawdown" in bt.columns else np.nan
    h = bt["h_used"].astype(float) if "h_used" in bt.columns else pd.Series(dtype=float)
    return {
        "cagr": cagr,
        "max_dd": dd,
        "vol": vol,
        "ret_over_vol": cagr / vol if np.isfinite(vol) and vol > 1e-9 else np.nan,
        "final_equity": float(eq.iloc[-1]),
        "n_rebalances": int(bt["rebalance"].sum()) if "rebalance" in bt.columns else 0,
        "mean_h": float(h.mean()) if len(h) else np.nan,
        "pct_h_at_min": float((h <= h.min() + 1e-6).mean()) if len(h) else np.nan,
        "pct_h_at_max": float((h >= h.max() - 1e-6).mean()) if len(h) else np.nan,
    }


def reconcile_legs(bt: pd.DataFrame, *, initial_capital: float) -> float:
    """Max |Δequity − (leg_a + leg_b + borrow + financing − fees)| per day."""
    deq = bt["equity"].astype(float).diff().fillna(0.0)
    la, lb = leg_mtm_daily(bt)
    borrow = bt.get("borrow_cost", pd.Series(0.0, index=bt.index)).astype(float)
    fin = bt.get("financing_pnl", pd.Series(0.0, index=bt.index)).astype(float)
    fees = (
        bt.get("rebalance_fee", pd.Series(0.0, index=bt.index)).astype(float)
        + bt.get("slippage_cost", pd.Series(0.0, index=bt.index)).astype(float)
    )
    resid = (deq - (la + lb - borrow + fin - fees)).abs()
    return float(resid.max()) if len(resid) else 0.0


def _mark_rebalances(ax, bt: pd.DataFrame) -> int:
    """Draw vertical lines on actual rebalance days; return count."""
    if "rebalance" not in bt.columns:
        return 0
    dates = bt.index[bt["rebalance"].astype(bool)]
    for i, dt in enumerate(dates):
        color = "#e67e22" if i > 0 else "#9b59b6"
        ax.axvline(dt, color=color, lw=0.7, alpha=0.55, zorder=1)
    if "rebalance_skipped_below_drift" in bt.columns:
        skipped = bt.index[bt["rebalance_skipped_below_drift"].astype(bool)]
        for dt in skipped:
            ax.axvline(dt, color="#bbbbbb", lw=0.5, ls=":", alpha=0.45, zorder=1)
    return len(dates)


def plot_pair_metrics_panel(
    bt: pd.DataFrame,
    *,
    pair_label: str,
    etf: str,
    underlying: str,
    h_signal: pd.Series | None,
    run_label: str,
    out_path: Path,
    rolling_vol: int = 21,
    initial_capital: float = 100_000.0,
) -> dict[str, float]:
    """4-row panel: h, expanding CAGR, drawdown, rolling vol."""
    metrics = full_window_metrics(bt, initial_capital=initial_capital)
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True, constrained_layout=True)

    h_used = bt["h_used"].astype(float)
    ax = axes[0]
    if h_signal is not None:
        hs = h_signal.reindex(bt.index).ffill()
        ax.plot(bt.index, hs.values, color="#aec7e8", lw=0.9, alpha=0.7, label="signal h (pre-trade)")
    ax.plot(bt.index, h_used.values, color="#1f77b4", lw=1.3, label="h_used")
    n_reb = _mark_rebalances(ax, bt)
    ax.set_ylabel("Hedge ratio h")
    ax.set_title(f"{pair_label} — risk & hedge  |  {run_label}")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.35)

    ax = axes[1]
    cagr = expanding_cagr(bt, initial_capital=initial_capital) * 100
    ax.plot(bt.index, cagr.values, color="#2ca02c", lw=1.2)
    _mark_rebalances(ax, bt)
    ax.axhline(0, color="#888", lw=0.6)
    ax.set_ylabel("Expanding CAGR %")

    ax = axes[2]
    dd = bt["drawdown"].astype(float) * 100
    ax.fill_between(bt.index, dd.values, 0, color="#d62728", alpha=0.25)
    ax.plot(bt.index, dd.values, color="#d62728", lw=1.0)
    _mark_rebalances(ax, bt)
    ax.set_ylabel("Drawdown %")

    ax = axes[3]
    vol = rolling_ann_vol(bt, rolling_vol, initial_capital=initial_capital) * 100
    ax.plot(bt.index, vol.values, color="#9467bd", lw=1.2, label=f"{rolling_vol}bd ann. vol")
    _mark_rebalances(ax, bt)
    ax.set_ylabel("Volatility %")
    ax.set_xlabel("Date")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.35)

    fig.text(
        0.01, 0.01,
        f"full CAGR {metrics['cagr']:.1%} | max DD {metrics['max_dd']:.1%} | "
        f"vol {metrics['vol']:.1%} | mean h {metrics['mean_h']:.2f} | "
        f"rebalances {n_reb} (orange=trade, purple=entry)",
        fontsize=8, color="#444",
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return metrics


def plot_pair_returns_panel(
    bt: pd.DataFrame,
    *,
    pair_label: str,
    etf: str,
    underlying: str,
    run_label: str,
    out_path: Path,
    initial_capital: float = 100_000.0,
) -> float:
    """Returns panel: pair index + leg MTM indices."""
    pair_idx = trading_return_index(bt, initial_capital=initial_capital)
    leg_a, leg_b = leg_return_index(bt, initial_capital=initial_capital)
    max_resid = reconcile_legs(bt, initial_capital=initial_capital)

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, constrained_layout=True)

    ax = axes[0]
    ax.plot(bt.index, (pair_idx - 1.0).values * 100, color="#1f77b4", lw=1.4,
            label="pair cumulative return %")
    n_reb = _mark_rebalances(ax, bt)
    ax.axhline(0, color="#888", lw=0.6)
    ax.set_ylabel("Return %")
    ax.set_title(f"{pair_label} — returns  |  {run_label}")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.35)

    ax = axes[1]
    ax.plot(bt.index, (leg_a - 1.0).values * 100, color="#ff7f0e", lw=1.2,
            label=f"{etf} short leg")
    ax.plot(bt.index, (leg_b - 1.0).values * 100, color="#2ca02c", lw=1.2,
            label=f"{underlying} short leg")
    _mark_rebalances(ax, bt)
    ax.axhline(0, color="#888", lw=0.6)
    ax.set_ylabel("Leg return %")
    ax.set_xlabel("Date")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.35)

    fig.text(
        0.01, 0.01,
        f"{n_reb} rebalances | leg reconcile max |residual| ${max_resid:.2f}/day",
        fontsize=8, color="#444",
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return max_resid
