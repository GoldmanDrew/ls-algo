"""
Bucket 4 per-pair diagnostics and ETF-level attribution (same conventions as
``notebooks/Buckets1-4_v2.ipynb`` cell 8).

**Inception vs trading PnL (important):** Portfolio ``nav`` is the *sum of separate
sub-account book values* (``aggregate_tail_risk_weighted_portfolio``). When a pair
first appears on the union calendar its book steps from 0 to ~initial capital—that
is **capital funding the sleeve**, not mark-to-market trading gains. Raw
``equity.diff()`` would count that step as “daily PnL” and allocate it to tickers,
producing **spurious vertical jumps** in cumulative attribution whenever a new pair
incepts. We therefore **zero each pair’s first-day ``diff``** on the union index
before splitting by gross (see :func:`build_b4_attribution_from_pair_bts`). After
this, attributed daily flows sum to ``nav.diff()`` **excluding** those inception
injections; total ``nav`` level still steps up when new pairs fund (correct for “sum
of books”, confusing if misread as “cumulative alpha”).

The ticker split remains a **gross-notional heuristic**: each pair's **trading**
daily ``equity`` change is allocated to leg tickers in proportion to that day's
|shares|×|px| on each leg. **Do not read positive flows on both inverse ETF and**
**underlying as “two independent shorts both made money.”** They are fixed fractions
of one portfolio Δ (same sign whenever the pair book gains); factor-style attribution
would require a different decomposition (e.g. returns × signed notionals).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def build_b4_attribution_from_pair_bts(
    bt_by_pair: dict[tuple[str, str], pd.DataFrame],
    *,
    index: pd.DatetimeIndex,
) -> dict[str, pd.DataFrame]:
    """Stacked gross / cumulative **trading** PnL by ticker (inception funding zeroed).

    See module docstring: first calendar row where a pair is live used to carry a
    step 0→book value; that diff is **not** trading PnL and is zeroed before the
    gross-weight split.

    Per-ticker ``daily`` increments sum across the portfolio and reconcile to trading
    NAV—but each column is **only** that ticker’s share of ``Δequity`` from pairs it
    touches, not economic exposure PnL unless you replace this allocator.
    """
    if not bt_by_pair:
        return {}
    idx = pd.DatetimeIndex(index)
    gross_cols: dict[str, pd.Series] = {}
    pnl_cols: dict[str, pd.Series] = {}

    def _add(name: str, ser: pd.Series) -> None:
        ser = ser.reindex(idx).fillna(0.0).astype(float)
        gross_cols.setdefault(name, pd.Series(0.0, index=idx))
        gross_cols[name] = gross_cols[name].add(ser, fill_value=0.0)

    def _add_pnl(name: str, ser: pd.Series) -> None:
        ser = ser.reindex(idx).fillna(0.0).astype(float)
        pnl_cols.setdefault(name, pd.Series(0.0, index=idx))
        pnl_cols[name] = pnl_cols[name].add(ser, fill_value=0.0)

    for (e, u), sub in bt_by_pair.items():
        if sub is None or sub.empty:
            continue
        first_p = pd.Timestamp(sub.index[0])
        active = idx >= first_p
        sx = sub.reindex(idx)
        eq = sx["equity"].astype(float).where(active, 0.0).ffill()
        a_sh = sx["a_shares"].astype(float).where(active, 0.0).ffill()
        b_sh = sx["b_shares"].astype(float).where(active, 0.0).ffill()
        ap = sx["a_px"].astype(float).where(active, 0.0).ffill()
        bp = sx["b_px"].astype(float).where(active, 0.0).ffill()
        g_a = (a_sh.abs() * ap.abs()).fillna(0.0)
        g_b = (b_sh.abs() * bp.abs()).fillna(0.0)
        _add(str(e), g_a)
        _add(str(u), g_b)
        deq = eq.diff().fillna(0.0)
        if first_p in deq.index:
            # Funding a new sub-account is not mark-to-market PnL (see module docstring).
            deq = deq.copy()
            deq.loc[first_p] = 0.0
        tot = (g_a + g_b).astype(float)
        safe = tot.replace(0.0, np.nan)
        sh_a = (g_a / safe).fillna(1.0)
        sh_b = (g_b / safe).fillna(0.0)
        _add_pnl(str(e), deq * sh_a)
        _add_pnl(str(u), deq * sh_b)

    gross_df = pd.DataFrame(gross_cols, index=idx).fillna(0.0)
    gross_df = gross_df.loc[:, gross_df.abs().sum() > 1e-9]
    if gross_df.empty:
        return {}
    pnl_df = pd.DataFrame({c: pnl_cols.get(c, pd.Series(0.0, index=idx)) for c in gross_df.columns}, index=idx).fillna(
        0.0
    )
    daily_pair_pnl_sum = pnl_df.sum(axis=1).astype(float)
    return {
        "gross_by_etf": gross_df,
        # Cumulative **trading** flows only (pair inception steps removed per pair).
        "cum_pnl_by_etf": pnl_df.cumsum(),
        # Sum of ticker legs each day (equals row-sum of ``pnl_df``); reconciles portfolio trading Δ vs NAV.
        "daily_pair_pnl_sum": daily_pair_pnl_sum,
    }


def pair_total_gross_usd(sub: pd.DataFrame) -> pd.Series:
    """Per-date total gross notional: |a_shares|×|a_px| + |b_shares|×|b_px|."""
    a = sub["a_shares"].astype(float).abs() * sub["a_px"].astype(float).abs()
    b = sub["b_shares"].astype(float).abs() * sub["b_px"].astype(float).abs()
    return (a + b).astype(float)


def pair_cumulative_trading_pnl_on_union(
    eq_on_union: pd.Series,
    *,
    union_idx: pd.DatetimeIndex,
    first_live: pd.Timestamp,
) -> pd.Series:
    """Cumulative mark-to-market PnL for one pair on *union_idx*, inception funding stripped.

    Mirrors :func:`build_b4_attribution_from_pair_bts`: the first diff when equity steps from
    pre-live zeros into the funded book is **not** trading return and is zeroed before cumsum.
    """
    eq = eq_on_union.astype(float).reindex(union_idx)
    fp = pd.Timestamp(first_live)
    eq = eq.where(union_idx >= fp, np.nan).ffill().fillna(0.0)
    deq = eq.diff().fillna(0.0)
    if fp in deq.index:
        deq = deq.copy()
        deq.loc[fp] = 0.0
    return deq.cumsum()


def _inception_flow_series(
    bt_by_pair: dict[tuple[str, str], pd.DataFrame],
    idx: pd.DatetimeIndex,
) -> pd.Series:
    """Per date, sum of (0→book) steps when pairs join the union calendar (matches stripped attrib)."""
    out = pd.Series(0.0, index=idx, dtype=float)
    for sub in bt_by_pair.values():
        if sub is None or sub.empty:
            continue
        fp = pd.Timestamp(sub.index[0])
        if fp not in idx:
            continue
        active = idx >= fp
        eq = (
            sub["equity"]
            .astype(float)
            .reindex(idx)
            .where(active, 0.0)
            .ffill()
        )
        raw_d = eq.diff().fillna(0.0)
        if fp in raw_d.index:
            out.loc[fp] = out.loc[fp] + float(raw_d.loc[fp])
    return out


def b4_trading_nav_series(
    nav: pd.Series,
    bt_by_pair: dict[tuple[str, str], pd.DataFrame],
) -> pd.Series:
    """Level NAV for **charts**: remove pair inception (0→book) steps from daily changes.

    Portfolio ``nav`` is the sum of independent sub-accounts. When a pair first
    appears on the union calendar, ``nav.diff()`` includes a one-time step equal
    to funding that book — not mark-to-market return. For growth / drawdown plots,
    use this series so vertical jumps are not misread as performance.

    Returns ``nav`` unchanged if *bt_by_pair* is empty.
    """
    if not bt_by_pair:
        return nav.astype(float)
    idx = pd.DatetimeIndex(nav.index)
    n = nav.astype(float).reindex(idx)
    inception = _inception_flow_series(bt_by_pair, idx)
    td = n.diff().fillna(0.0) - inception.reindex(idx).fillna(0.0)
    base = float(n.iloc[0]) if len(n) else 0.0
    return pd.Series(base + td.cumsum(), index=idx, dtype=float)


def diagnose_b4_attribution_vs_nav(
    bt_pf: pd.DataFrame,
    bt_by_pair: dict[tuple[str, str], pd.DataFrame],
    *,
    tol_usd: float = 1.0,
) -> dict[str, Any]:
    """Check attributed **trading** daily sum vs ``nav.diff()`` minus pair inception flows."""
    nav = bt_pf.get("nav", bt_pf.get("equity"))
    if nav is None or not bt_by_pair:
        return {"ok": False, "reason": "missing nav or pairs"}
    idx = pd.DatetimeIndex(bt_pf.index)
    nav = nav.astype(float).reindex(idx)
    attr = build_b4_attribution_from_pair_bts(bt_by_pair, index=idx)
    pnl = attr.get("cum_pnl_by_etf")
    if pnl is None or pnl.empty:
        return {"ok": False, "reason": "empty attribution"}
    daily_ticker = pnl.diff()
    if len(pnl) > 0:
        daily_ticker.iloc[0] = pnl.iloc[0]
    daily_ticker = daily_ticker.fillna(0.0).sum(axis=1)
    daily_nav = nav.diff().fillna(0.0)
    inception = _inception_flow_series(bt_by_pair, idx)
    expected_trading_nav = daily_nav - inception
    diff = (daily_ticker - expected_trading_nav).abs()
    tail = diff.iloc[1:] if len(diff) > 1 else diff
    mx = float(tail.max()) if len(tail) else float(diff.max())
    return {
        "ok": mx <= tol_usd,
        "max_abs_daily_usd": mx,
        "tol_usd": tol_usd,
        "mean_abs_daily_usd": float(diff.mean()),
    }


def plot_bucket4_per_pair_equity_and_gross(
    bt_pf: pd.DataFrame,
    bt_by_pair: dict[tuple[str, str], pd.DataFrame],
    *,
    run_label: str = "",
) -> None:
    """Two-row panel per pair: **cumulative trading PnL** (USD), then total gross (USD).

    Top line sums daily equity changes with the pair's inception (0→funded book) diff zeroed,
    matching attribution — no vertical jump from capital deployment. Bottom panel unchanged.
    """
    import matplotlib.pyplot as plt

    if not bt_by_pair:
        print("[B4 per-pair] skip: no per-pair backtests.")
        return
    idx = pd.DatetimeIndex(bt_pf.index)
    nav = bt_pf.get("nav", bt_pf.get("equity"))
    if nav is None:
        print("[B4 per-pair] skip: portfolio frame has no nav/equity.")
        return
    nav = nav.astype(float).reindex(idx)

    keys_sorted = sorted(
        bt_by_pair.keys(),
        key=lambda k: float(pair_total_gross_usd(bt_by_pair[k]).mean()),
        reverse=True,
    )
    n = len(keys_sorted)
    fig, axes = plt.subplots(2 * n, 1, figsize=(13, max(7.0, 2.6 * n)), sharex=True, constrained_layout=True)
    if n == 1:
        axes = np.array([axes])
    diag = diagnose_b4_attribution_vs_nav(bt_pf, bt_by_pair, tol_usd=max(5.0, 1e-6 * float(nav.iloc[-1])))
    fig.suptitle(
        f"Bucket 4 per-pair cumulative trading PnL & gross — {run_label}\n"
        f"attrib trading daily vs (nav.diff − inception): max|err|=${diag.get('max_abs_daily_usd', float('nan')):.2f} "
        f"(ok={diag.get('ok')})",
        fontsize=11,
    )

    for i, key in enumerate(keys_sorted):
        sub = bt_by_pair[key]
        if sub is None or sub.empty:
            continue
        lab = f"{key[0]} / {key[1]}"
        first_p = pd.Timestamp(sub.index[0])
        eq = (
            sub["equity"]
            .astype(float)
            .reindex(idx)
            .where(idx >= first_p, np.nan)
            .ffill()
            .fillna(0.0)
        )
        cum_trade = pair_cumulative_trading_pnl_on_union(eq, union_idx=idx, first_live=first_p)
        gr = (
            pair_total_gross_usd(sub)
            .reindex(idx)
            .where(idx >= first_p, np.nan)
            .ffill()
            .fillna(0.0)
        )
        ax_e = axes[2 * i]
        ax_g = axes[2 * i + 1]
        ax_e.plot(
            idx,
            cum_trade.values,
            color="tab:blue",
            lw=1.2,
            label="cum. trading PnL $ (inception stripped)",
        )
        ax_e.axhline(0.0, color="black", lw=0.6, alpha=0.35)
        ax_e.set_ylabel("Trading PnL $")
        ax_e.set_title(lab)
        ax_e.legend(loc="upper left", fontsize=8)
        ax_e.grid(True, linestyle="--", alpha=0.4)
        ax_g.fill_between(idx, 0.0, gr.values, color="tab:gray", alpha=0.35, step=None)
        ax_g.plot(idx, gr.values, color="tab:purple", lw=1.0, label="|ETF leg|+|hedge leg| gross $")
        ax_g.set_ylabel("Gross $")
        ax_g.legend(loc="upper left", fontsize=8)
        ax_g.grid(True, linestyle="--", alpha=0.35)

    axes[-1].set_xlabel("Date")
