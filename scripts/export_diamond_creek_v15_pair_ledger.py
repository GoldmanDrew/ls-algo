"""
Diamond Creek v15 — per-pair daily ledger Excel export.

README (run after backtest in notebooks/Diamond_Creek_Backtest_v15.ipynb):
- ref leverage: pass ALL_PAIR_DAILY / ALL_BT key (e.g. max(LEVERAGE_RUNS)).
- export_start / export_end: default **2023-01-01 .. 2023-01-31** (January 2023 only). Override to expand.
- ATTRIBUTION_BASE_CAPITAL: same as Joel White Bay attribution / CFG capital (default 10M).
- Reconciliation: max |sum(pair daily_net) - NAV daily change| over dates should be <~1 USD
  (floating noise); portfolio margin flows match ALL_BT cum_margin_debit/cum_margin_credit
  incremental flows used in Diamond_Creek_PnL_Attribution.xlsx.

v15 allocates book margin debit to pairs pro-rata by long notional; pair daily_net already
includes that allocation. The portfolio_* sheet repeats book-level margin debit/credit flows
from ALL_BT for DC ETF Attribution-style tie-out (single margin line vs sum of pair debits).

Transaction costs: pair-level daily_txn_cost_usd does not sum to book (cross-symbol netting).
For attribution, use daily_pair_net_ex_txn_usd (gross − borrow − net financing, no txn) and
subtract book_daily_txn_cost_usd once — matches book_daily_net_pnl_from_components_usd exactly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Default ledger window: January 2023 only (all sheets / rows clipped to this range).
LEDGER_EXPORT_START_DEFAULT = pd.Timestamp("2023-01-01")
LEDGER_EXPORT_END_DEFAULT = pd.Timestamp("2023-01-31")

# Reference column order (ALL_PAIRS + per-pair sheets); extras follow.
LEDGER_BASE_COLS: list[str] = [
    "date",
    "etf",
    "under",
    "long_sh",
    "short_sh",
    "long_notional_usd",
    "short_notional_usd",
    "gross_notional_usd",
    "net_notional_usd",
    "borrow_rate_annual",
    "daily_long_pnl_usd",
    "daily_short_pnl_usd",
    "daily_borrow_cost_usd",
    "daily_margin_debit_cost_usd",
    "daily_short_credit_income_usd",
    "daily_net_financing_cost_usd",
    "daily_txn_cost_usd",
    "daily_turnover_usd",
    "fed_funds_rate",
    "is_rebal",
    "daily_pair_gross_trading_pnl_usd",
    "daily_total_cost_usd",
    "daily_pair_net_pnl_usd",
    "daily_total_cost_ex_txn_usd",
    "daily_pair_net_ex_txn_usd",
    "pair",
    "cum_long_pnl_usd",
    "cum_short_pnl_usd",
    "cum_pair_gross_trading_pnl_usd",
    "cum_txn_cost_usd",
    "cum_borrow_cost_usd",
    "cum_margin_debit_cost_usd",
    "cum_short_credit_income_usd",
    "cum_net_financing_cost_usd",
    "cum_total_cost_usd",
    "cum_pair_net_pnl_usd",
    "cum_total_cost_ex_txn_usd",
    "cum_pair_net_ex_txn_usd",
    "cum_turnover_usd",
    "underlying_price",
    "etf_price",
]


def _daily_flow_from_cum(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce").ffill().fillna(0.0)
    flow = x.diff().fillna(0.0)
    if len(flow) > 0:
        flow.iloc[0] = float(x.iloc[0])
    return flow


def _loc_pos(idx: pd.DatetimeIndex, t: pd.Timestamp) -> int:
    pos = idx.get_loc(t)
    if isinstance(pos, slice):
        pos = idx.get_indexer([t], method=None)[0]
    elif isinstance(pos, np.ndarray):
        pos = int(pos[0]) if len(pos) else -1
    return int(pos)


def _incremental_cum_flow_on_dates(cum: pd.Series, dates: pd.DatetimeIndex) -> np.ndarray:
    """Increment vs prior row in full series (first row of full series: flow = cum[0] - 0)."""
    cum = pd.to_numeric(cum, errors="coerce").ffill()
    idx = cum.index
    out = np.empty(len(dates), dtype=float)
    for i, t in enumerate(dates):
        if t not in idx:
            out[i] = np.nan
            continue
        pos = _loc_pos(idx, t)
        cur = float(cum.iloc[pos])
        prev = 0.0 if pos == 0 else float(cum.iloc[pos - 1])
        out[i] = cur - prev
    return out


def _nav_daily_change_on_dates(
    nav: pd.Series, dates: pd.DatetimeIndex, attribution_base: float
) -> np.ndarray:
    """Aligned to dates; first day of full backtest uses nav[0] - attribution_base."""
    nav = pd.to_numeric(nav, errors="coerce").ffill()
    idx = nav.index
    out = np.empty(len(dates), dtype=float)
    for i, t in enumerate(dates):
        if t not in idx:
            out[i] = np.nan
            continue
        pos = _loc_pos(idx, t)
        cur = float(nav.iloc[pos])
        if pos == 0:
            out[i] = cur - float(attribution_base)
        else:
            out[i] = cur - float(nav.iloc[pos - 1])
    return out


def _safe_sheet_name(name: str) -> str:
    out = str(name).strip()
    for c in r"[]:*?/\\":
        out = out.replace(c, "-")
    return out[:31] if len(out) > 31 else out


def _implied_prices(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Close-implied prices for standard long-underlying / short-ETF pairs; NaN if ambiguous (e.g. both legs short)."""
    ls = pd.to_numeric(df["long_sh"], errors="coerce")
    ss = pd.to_numeric(df["short_sh"], errors="coerce")
    ln = pd.to_numeric(df["long_notional_usd"], errors="coerce")
    sn = pd.to_numeric(df["short_notional_usd"], errors="coerce")
    up = pd.Series(np.nan, index=df.index, dtype=float)
    ep = pd.Series(np.nan, index=df.index, dtype=float)
    m_long_und = ls > 0
    m_short_etf = ss < 0
    both_short = (ls < 0) & (ss < 0)
    up.loc[m_long_und] = (ln / ls).loc[m_long_und]
    ep.loc[m_short_etf & ~both_short] = (sn / np.abs(ss)).loc[m_short_etf & ~both_short]
    up.loc[both_short] = np.nan
    ep.loc[both_short] = np.nan
    return up, ep


def build_pair_ledger_frames(
    pair_daily: pd.DataFrame,
    bt: pd.DataFrame,
    *,
    export_start: pd.Timestamp | None = None,
    export_end: pd.Timestamp | None = None,
    attribution_base_capital: float | None = None,
) -> dict[str, Any]:
    """
    Returns dict with:
      all_pairs: enriched long-form DataFrame
      portfolio: daily portfolio financing + NAV bridge from ALL_BT
      reconciliation: one-row metrics + optional daily diff table

    Default date window: January 2023 only (LEDGER_EXPORT_START_DEFAULT / LEDGER_EXPORT_END_DEFAULT).
    Cumulative columns on pair rows are recomputed within that window (month-to-date in export).
    Book flows for the portfolio sheet use incremental deltas from full ALL_BT on the same calendar dates.
    """
    if pair_daily is None or pair_daily.empty:
        raise ValueError("pair_daily is empty")
    export_start = (
        pd.to_datetime(export_start) if export_start is not None else LEDGER_EXPORT_START_DEFAULT
    )
    export_end = (
        pd.to_datetime(export_end) if export_end is not None else LEDGER_EXPORT_END_DEFAULT
    )

    d = pair_daily.copy()
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    d = d[d["date"].notna()].sort_values(["date", "etf"]).reset_index(drop=True)
    d = d.loc[(d["date"] >= export_start) & (d["date"] <= export_end)].copy()
    if d.empty:
        raise ValueError(
            f"No pair rows after clipping to [{export_start.date()} .. {export_end.date()}]"
        )

    d["under"] = d["under"].astype(str).str.strip()
    d["etf"] = d["etf"].astype(str).str.strip()
    d["pair"] = d["under"] + "_" + d["etf"]

    gcols = [
        "daily_long_pnl_usd",
        "daily_short_pnl_usd",
        "daily_borrow_cost_usd",
        "daily_margin_debit_cost_usd",
        "daily_short_credit_income_usd",
        "daily_txn_cost_usd",
        "daily_turnover_usd",
        "daily_pair_gross_trading_pnl_usd",
        "daily_total_cost_usd",
        "daily_pair_net_pnl_usd",
    ]
    for c in gcols:
        if c not in d.columns:
            d[c] = 0.0
        d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0.0)

    if "daily_net_financing_cost_usd" in d.columns:
        d["daily_net_financing_cost_usd"] = pd.to_numeric(
            d["daily_net_financing_cost_usd"], errors="coerce"
        ).fillna(0.0)
    else:
        d["daily_net_financing_cost_usd"] = (
            d["daily_margin_debit_cost_usd"] - d["daily_short_credit_income_usd"]
        )

    # PnL before txn: matches book net when summed and book txn subtracted once (no pair-txn netting gap).
    d["daily_total_cost_ex_txn_usd"] = (
        d["daily_borrow_cost_usd"] + d["daily_net_financing_cost_usd"]
    )
    d["daily_pair_net_ex_txn_usd"] = (
        d["daily_pair_gross_trading_pnl_usd"] - d["daily_total_cost_ex_txn_usd"]
    )

    cum_specs = [
        ("cum_long_pnl_usd", "daily_long_pnl_usd"),
        ("cum_short_pnl_usd", "daily_short_pnl_usd"),
        ("cum_txn_cost_usd", "daily_txn_cost_usd"),
        ("cum_borrow_cost_usd", "daily_borrow_cost_usd"),
        ("cum_margin_debit_cost_usd", "daily_margin_debit_cost_usd"),
        ("cum_short_credit_income_usd", "daily_short_credit_income_usd"),
        ("cum_net_financing_cost_usd", "daily_net_financing_cost_usd"),
        ("cum_pair_gross_trading_pnl_usd", "daily_pair_gross_trading_pnl_usd"),
        ("cum_total_cost_usd", "daily_total_cost_usd"),
        ("cum_total_cost_ex_txn_usd", "daily_total_cost_ex_txn_usd"),
        ("cum_pair_net_pnl_usd", "daily_pair_net_pnl_usd"),
        ("cum_pair_net_ex_txn_usd", "daily_pair_net_ex_txn_usd"),
        ("cum_turnover_usd", "daily_turnover_usd"),
    ]
    for c_out, c_in in cum_specs:
        d[c_out] = d.groupby("etf", sort=False)[c_in].cumsum()

    for c in ("long_sh", "short_sh", "long_notional_usd", "short_notional_usd"):
        if c not in d.columns:
            d[c] = 0.0

    up, ep = _implied_prices(d)
    d["underlying_price"] = up
    d["etf_price"] = ep

    # --- Portfolio-level series (full ALL_BT; flows are correct increments on export window dates) ---
    b_full = bt.copy().sort_index()
    b_full.index = pd.to_datetime(b_full.index)
    win_dates = b_full.index[
        (b_full.index >= export_start) & (b_full.index <= export_end)
    ]
    if len(win_dates) == 0:
        raise ValueError(
            f"No ALL_BT rows in [{export_start.date()} .. {export_end.date()}]; cannot build portfolio sheet."
        )

    nav_s = pd.to_numeric(b_full["nav"], errors="coerce")
    base = attribution_base_capital
    if base is None or not np.isfinite(base):
        base = float(nav_s.iloc[0]) if len(nav_s) else float("nan")

    daily_nav_pnl = _nav_daily_change_on_dates(nav_s, win_dates, base)
    margin_debit_flow = _incremental_cum_flow_on_dates(b_full["cum_margin_debit"], win_dates)
    margin_credit_flow = _incremental_cum_flow_on_dates(b_full["cum_margin_credit"], win_dates)
    borrow_flow = _incremental_cum_flow_on_dates(b_full["cum_borrow"], win_dates)
    txn_flow = _incremental_cum_flow_on_dates(b_full["cum_costs"], win_dates)
    long_leg = _incremental_cum_flow_on_dates(b_full["cum_long_pnl"], win_dates)
    short_leg = _incremental_cum_flow_on_dates(b_full["cum_short_pnl"], win_dates)
    gross_trading_book = long_leg + short_leg
    nav_win = nav_s.reindex(win_dates).values

    portfolio = pd.DataFrame(
        {
            "date": win_dates,
            "nav": nav_win,
            "daily_nav_change_usd": daily_nav_pnl,
            "book_daily_gross_trading_pnl_usd": gross_trading_book,
            "book_daily_txn_cost_usd": txn_flow,
            "book_daily_borrow_cost_usd": borrow_flow,
            "book_daily_margin_debit_usd": margin_debit_flow,
            "book_daily_short_credit_income_usd": margin_credit_flow,
            "book_daily_net_margin_paid_excess_interest_usd": margin_debit_flow - margin_credit_flow,
            "book_daily_total_costs_usd": txn_flow
            + borrow_flow
            + (margin_debit_flow - margin_credit_flow),
            "book_daily_net_pnl_from_components_usd": gross_trading_book
            - txn_flow
            - borrow_flow
            - (margin_debit_flow - margin_credit_flow),
        }
    )

    pairs_sum = (
        d.groupby("date", as_index=False)
        .agg(
            pairs_daily_gross=("daily_pair_gross_trading_pnl_usd", "sum"),
            pairs_daily_txn=("daily_txn_cost_usd", "sum"),
            pairs_daily_borrow=("daily_borrow_cost_usd", "sum"),
            pairs_daily_margin_debit=("daily_margin_debit_cost_usd", "sum"),
            pairs_daily_short_credit=("daily_short_credit_income_usd", "sum"),
            pairs_daily_net_financing=("daily_net_financing_cost_usd", "sum"),
            pairs_daily_net=("daily_pair_net_pnl_usd", "sum"),
            pairs_daily_net_ex_txn=("daily_pair_net_ex_txn_usd", "sum"),
        )
        .sort_values("date")
    )

    merged = portfolio.merge(pairs_sum, on="date", how="left")
    merged["pairs_daily_gross"] = merged["pairs_daily_gross"].fillna(0.0)
    merged["pairs_daily_txn"] = merged["pairs_daily_txn"].fillna(0.0)
    merged["pairs_daily_borrow"] = merged["pairs_daily_borrow"].fillna(0.0)
    merged["pairs_daily_margin_debit"] = merged["pairs_daily_margin_debit"].fillna(0.0)
    merged["pairs_daily_short_credit"] = merged["pairs_daily_short_credit"].fillna(0.0)
    merged["pairs_daily_net_financing"] = merged["pairs_daily_net_financing"].fillna(0.0)
    merged["pairs_daily_net"] = merged["pairs_daily_net"].fillna(0.0)
    merged["pairs_daily_net_ex_txn"] = merged["pairs_daily_net_ex_txn"].fillna(0.0)

    # Book-level txn once: sum(pair net ex txn) - book txn == book net from cum flows (identity).
    merged["attribution_daily_net_usd"] = (
        merged["pairs_daily_net_ex_txn"] - merged["book_daily_txn_cost_usd"]
    )
    merged["diff_attribution_net_vs_book_components_usd"] = (
        merged["attribution_daily_net_usd"] - merged["book_daily_net_pnl_from_components_usd"]
    )

    merged["diff_net_pnl_vs_nav_usd"] = merged["pairs_daily_net"] - merged["daily_nav_change_usd"]
    merged["diff_attribution_net_vs_nav_usd"] = (
        merged["attribution_daily_net_usd"] - merged["daily_nav_change_usd"]
    )
    merged["diff_gross_vs_book_usd"] = (
        merged["pairs_daily_gross"] - merged["book_daily_gross_trading_pnl_usd"]
    )
    merged["diff_txn_vs_book_usd"] = merged["pairs_daily_txn"] - merged["book_daily_txn_cost_usd"]
    merged["diff_borrow_vs_book_usd"] = (
        merged["pairs_daily_borrow"] - merged["book_daily_borrow_cost_usd"]
    )
    merged["diff_margin_debit_vs_book_usd"] = (
        merged["pairs_daily_margin_debit"] - merged["book_daily_margin_debit_usd"]
    )
    merged["diff_short_credit_vs_book_usd"] = (
        merged["pairs_daily_short_credit"] - merged["book_daily_short_credit_income_usd"]
    )

    tol = 1.0
    tol_book = 1e-6
    metrics = {
        "max_abs_diff_attribution_vs_book_net_usd": float(
            merged["diff_attribution_net_vs_book_components_usd"].abs().max()
        ),
        "max_abs_diff_net_vs_nav_usd_legacy_pair_txn": float(
            merged["diff_net_pnl_vs_nav_usd"].abs().max()
        ),
        "max_abs_diff_attribution_net_vs_nav_usd": float(
            merged["diff_attribution_net_vs_nav_usd"].abs().max()
        ),
        "max_abs_diff_gross_usd": float(merged["diff_gross_vs_book_usd"].abs().max()),
        "max_abs_diff_txn_pair_sum_vs_book_usd": float(
            merged["diff_txn_vs_book_usd"].abs().max()
        ),
        "max_abs_diff_borrow_usd": float(merged["diff_borrow_vs_book_usd"].abs().max()),
        "max_abs_diff_margin_debit_usd": float(merged["diff_margin_debit_vs_book_usd"].abs().max()),
        "max_abs_diff_short_credit_usd": float(
            merged["diff_short_credit_vs_book_usd"].abs().max()
        ),
        "reconciliation_tolerance_usd": tol,
        "reconciliation_tolerance_book_identity_usd": tol_book,
        "attribution_base_capital_usd": float(base) if np.isfinite(base) else None,
        "export_start": str(export_start.date()),
        "export_end": str(export_end.date()),
    }
    metrics["pass_tol_attribution_vs_book_net"] = (
        metrics["max_abs_diff_attribution_vs_book_net_usd"] <= tol_book
    )
    metrics["pass_tol_net_vs_nav"] = metrics["max_abs_diff_net_vs_nav_usd_legacy_pair_txn"] <= tol

    # Column order: base + remaining
    extra = [c for c in d.columns if c not in LEDGER_BASE_COLS]
    ordered = [c for c in LEDGER_BASE_COLS if c in d.columns] + extra
    all_pairs = d[ordered].copy()

    return {
        "all_pairs": all_pairs,
        "portfolio": portfolio,
        "reconciliation_daily": merged,
        "metrics": metrics,
    }


def _unique_sheet_names(names: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    out: list[str] = []
    for raw in names:
        base = _safe_sheet_name(raw)
        cnt = seen.get(base, 0)
        seen[base] = cnt + 1
        if cnt == 0:
            out.append(base)
        else:
            suf = f"~{cnt}"
            out.append((base[: 31 - len(suf)] + suf) if len(base) + len(suf) > 31 else base + suf)
    return out


def export_v15_pair_ledger_to_excel(
    pair_daily: pd.DataFrame,
    bt: pd.DataFrame,
    out_path: str | Path,
    *,
    export_start: pd.Timestamp | None = None,
    export_end: pd.Timestamp | None = None,
    attribution_base_capital: float | None = None,
    ref_leverage: float | None = None,
    include_per_pair_sheets: bool = True,
) -> dict[str, Any]:
    """
    Writes ALL_PAIRS, portfolio_financing, reconciliation_daily, summary; optional one sheet per pair.
    Returns metrics dict from build_pair_ledger_frames.

    Defaults: January 2023 only (see LEDGER_EXPORT_*_DEFAULT).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    built = build_pair_ledger_frames(
        pair_daily,
        bt,
        export_start=export_start,
        export_end=export_end,
        attribution_base_capital=attribution_base_capital,
    )
    all_pairs = built["all_pairs"]
    portfolio = built["portfolio"]
    recon = built["reconciliation_daily"]
    metrics = built["metrics"]
    metrics["ref_leverage"] = ref_leverage

    summary = pd.DataFrame([{**metrics, "output_path": str(out_path.resolve())}])

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        all_pairs.to_excel(writer, sheet_name="ALL_PAIRS", index=False)
        portfolio.to_excel(writer, sheet_name="portfolio_financing", index=False)
        recon.to_excel(writer, sheet_name="reconciliation_daily", index=False)
        summary.to_excel(writer, sheet_name="reconciliation_summary", index=False)

        if include_per_pair_sheets:
            groups = list(all_pairs.groupby("pair", sort=True))
            sheet_names = _unique_sheet_names([str(pid) for pid, _ in groups])
            for (pair_id, sub), sh in zip(groups, sheet_names):
                sub.to_excel(writer, sheet_name=sh, index=False)

    return metrics
