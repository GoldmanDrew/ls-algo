"""
Excel export for the v15 Diamond Creek **per-pair daily** ledger + reconciliation.

Uses explicit ``underlying_price`` / ``etf_price`` from the engine when present
(see ``scripts.apply_pair_ledger_explicit_prices.merge_implied_with_explicit``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts.align_pair_bt_to_fund_attribution import align_pair_bt_to_fund_attribution
from scripts.apply_pair_ledger_explicit_prices import merge_implied_with_explicit


def _jan_2023_mask(dates: pd.Series) -> pd.Series:
    t = pd.to_datetime(dates).dt.normalize()
    return (t >= pd.Timestamp("2023-01-01")) & (t <= pd.Timestamp("2023-01-31"))


def _nav_change_usd(bt: pd.DataFrame, attribution_base_capital: float) -> pd.Series:
    nav = pd.to_numeric(bt["nav"], errors="coerce")
    ch = nav.diff()
    if len(ch):
        ch.iloc[0] = float(nav.iloc[0]) - float(attribution_base_capital)
    return ch.reindex(bt.index)


def _prep_pair_daily(pair_daily: pd.DataFrame) -> pd.DataFrame:
    d = pair_daily.copy()
    if d.empty:
        return d
    d["date"] = pd.to_datetime(d["date"]).dt.normalize()
    lsh = pd.to_numeric(d["long_sh"], errors="coerce")
    ssh = pd.to_numeric(d["short_sh"], errors="coerce")
    ln = pd.to_numeric(d["long_notional_usd"], errors="coerce")
    sn = pd.to_numeric(d["short_notional_usd"], errors="coerce")
    implied_u = ln / lsh.replace(0, np.nan)
    implied_e = sn / ssh.replace(0, np.nan)
    u_m, e_m = merge_implied_with_explicit(d, implied_u, implied_e)
    d["underlying_price"] = u_m
    d["etf_price"] = e_m
    if "long_margin_basis_usd" not in d.columns:
        pu = pd.to_numeric(d["underlying_price"], errors="coerce")
        d["long_margin_basis_usd"] = np.maximum(lsh.fillna(0.0), 0.0) * pu.fillna(0.0)
    if "under_borrow_rate_annual" not in d.columns:
        br = pd.to_numeric(d.get("borrow_rate_annual"), errors="coerce")
        d["under_borrow_rate_annual"] = br
    return d


def _active_pairs_mask(df: pd.DataFrame, *, min_abs_gross: float) -> pd.Series:
    s = df.groupby("etf")["gross_notional_usd"].apply(
        lambda x: pd.to_numeric(x, errors="coerce").fillna(0.0).abs().sum()
    )
    ok = set(s[s >= min_abs_gross].index)
    return df["etf"].isin(ok)


def build_reconciliation_daily(
    pair_daily: pd.DataFrame,
    bt: pd.DataFrame,
    *,
    attribution_base_capital: float,
) -> pd.DataFrame:
    b = bt.sort_index().copy()
    b.index = pd.to_datetime(b.index).normalize()

    p = pair_daily.copy()
    p["date"] = pd.to_datetime(p["date"]).dt.normalize()
    if p.empty:
        return pd.DataFrame()

    gcols = {
        "daily_long_pnl_usd": "sum_pair_long_pnl_usd",
        "daily_short_pnl_usd": "sum_pair_short_pnl_usd",
        "daily_borrow_cost_usd": "sum_pair_borrow_usd",
        "daily_margin_debit_cost_usd": "sum_pair_margin_debit_usd",
        "daily_short_credit_income_usd": "sum_pair_short_credit_usd",
        "daily_txn_cost_usd": "sum_pair_txn_usd",
        "daily_net_financing_cost_usd": "sum_pair_net_financing_usd",
        "daily_pair_net_pnl_usd": "sum_pair_net_pnl_usd",
    }
    for c in gcols:
        if c not in p.columns:
            p[c] = 0.0
    agg = p.groupby("date", as_index=False).agg({k: "sum" for k in gcols})
    agg = agg.rename(columns=gcols)

    # Book-side frames: use an explicit `date` column only. If `bt` uses a DatetimeIndex
    # named "date", assigning nav_ch["date"] = nav_ch.index makes `date` both an index
    # level and a column, and merge(..., on="date") raises ValueError.
    dates = pd.DatetimeIndex(b.index).normalize()
    nav_ch = pd.DataFrame(
        {
            "date": dates,
            "book_nav_change_usd": _nav_change_usd(b, attribution_base_capital).to_numpy(),
        }
    )

    txn_book = pd.DataFrame(
        {
            "date": dates,
            "book_daily_txn_usd": pd.to_numeric(b["cum_costs"], errors="coerce")
            .diff()
            .reindex(b.index)
            .fillna(0.0)
            .to_numpy(),
        }
    )

    br_book = pd.DataFrame(
        {
            "date": dates,
            "book_daily_borrow_usd": pd.to_numeric(b["daily_borrow"], errors="coerce")
            .reindex(b.index)
            .fillna(0.0)
            .to_numpy(),
        }
    )

    md_book = pd.to_numeric(b["cum_margin_debit"], errors="coerce").diff().reindex(b.index).fillna(0.0)
    mc_book = pd.to_numeric(b["cum_margin_credit"], errors="coerce").diff().reindex(b.index).fillna(0.0)
    fin = pd.DataFrame(
        {
            "date": dates,
            "book_daily_margin_debit_usd": md_book.to_numpy(),
            "book_daily_margin_credit_usd": mc_book.to_numpy(),
        }
    )

    long_short = pd.DataFrame(
        {
            "date": dates,
            "book_daily_long_short_pnl_usd": (
                pd.to_numeric(b["daily_long_pnl"], errors="coerce").fillna(0.0)
                + pd.to_numeric(b["daily_short_pnl"], errors="coerce").fillna(0.0)
            ).to_numpy(),
        }
    )

    out = (
        nav_ch.merge(txn_book, on="date", how="outer")
        .merge(br_book, on="date", how="outer")
        .merge(fin, on="date", how="outer")
        .merge(long_short, on="date", how="outer")
        .merge(agg, on="date", how="outer")
        .sort_values("date")
    )
    for c in out.columns:
        if c != "date":
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)

    out["sum_pair_gross_ls_usd"] = out["sum_pair_long_pnl_usd"] + out["sum_pair_short_pnl_usd"]
    out["sum_pair_ex_txn_net_usd"] = out["sum_pair_net_pnl_usd"] + out["sum_pair_txn_usd"]

    out["diff_gross_usd"] = out["sum_pair_gross_ls_usd"] - out["book_daily_long_short_pnl_usd"]
    out["diff_borrow_usd"] = out["sum_pair_borrow_usd"] - out["book_daily_borrow_usd"]
    out["diff_txn_pair_sum_vs_book_usd"] = out["sum_pair_txn_usd"] - out["book_daily_txn_usd"]

    rhs_attr = out["book_nav_change_usd"] + out["book_daily_txn_usd"]
    out["diff_attribution_vs_book_net_usd"] = out["sum_pair_ex_txn_net_usd"] - rhs_attr

    fin_book = out["book_daily_margin_debit_usd"] - out["book_daily_margin_credit_usd"]
    out["diff_margin_net_usd"] = out["sum_pair_net_financing_usd"] - fin_book

    out["diff_net_vs_nav_legacy_pair_txn_usd"] = (
        out["sum_pair_ex_txn_net_usd"] - out["book_daily_txn_usd"] - out["book_nav_change_usd"]
    )
    out["diff_attribution_net_vs_nav_usd"] = out["sum_pair_net_pnl_usd"] - out["book_nav_change_usd"]

    out = out.sort_values("date")
    cal = pd.to_datetime(p["date"]).dt.normalize().unique()
    out = out.loc[out["date"].isin(cal)].reset_index(drop=True)
    return out


def _ledger_metrics(recon: pd.DataFrame) -> dict[str, Any]:
    def _maxabs(col: str) -> float:
        if col not in recon.columns:
            return float("nan")
        return float(pd.to_numeric(recon[col], errors="coerce").abs().max())

    tol = 1.0
    tol_book = 1e-6
    m = {
        "max_abs_diff_attribution_vs_book_net_usd": _maxabs("diff_attribution_vs_book_net_usd"),
        "max_abs_diff_net_vs_nav_usd_legacy_pair_txn": _maxabs("diff_net_vs_nav_legacy_pair_txn_usd"),
        "max_abs_diff_attribution_net_vs_nav_usd": _maxabs("diff_attribution_net_vs_nav_usd"),
        "max_abs_diff_gross_usd": _maxabs("diff_gross_usd"),
        "max_abs_diff_txn_pair_sum_vs_book_usd": _maxabs("diff_txn_pair_sum_vs_book_usd"),
        "max_abs_diff_borrow_usd": _maxabs("diff_borrow_usd"),
        "max_abs_diff_margin_debit_usd": _maxabs("diff_margin_net_usd"),
        "max_abs_diff_short_credit_usd": 0.0,
        "reconciliation_tolerance_usd": tol,
        "reconciliation_tolerance_book_identity_usd": tol_book,
    }
    if not recon.empty and "date" in recon.columns:
        m["export_start"] = str(recon["date"].min().date())
        m["export_end"] = str(recon["date"].max().date())
    m["pass_tol_attribution_vs_book_net"] = bool(m["max_abs_diff_attribution_vs_book_net_usd"] < tol_book)
    m["pass_tol_net_vs_nav"] = bool(m["max_abs_diff_attribution_net_vs_nav_usd"] < tol)
    return m


def _all_pairs_columns() -> list[str]:
    """
    Column order for ``ALL_PAIRS``. **Long/short “books”** for margin and borrow
    are ``long_margin_basis_usd`` / ``short_financing_basis_usd``; **signed**
    leg notionals are ``long_notional_usd`` / ``short_notional_usd`` (shares×price);
    do not mix the two in downstream “gross” labels (see
    ``daily_pair_workbook_formulas`` / ``export_dc_etf_arb_pairwise_workbook``).
    """
    return [
        "date",
        "etf",
        "under",
        "long_sh",
        "short_sh",
        "underlying_price",
        "etf_price",
        "long_margin_basis_usd",
        "short_financing_basis_usd",
        "long_notional_usd",
        "short_notional_usd",
        "gross_notional_usd",
        "net_notional_usd",
        "borrow_rate_annual",
        "under_borrow_rate_annual",
        "daily_long_pnl_usd",
        "daily_short_pnl_usd",
        "daily_borrow_cost_usd",
        "daily_underlying_borrow_cost_usd",
        "daily_margin_debit_cost_usd",
        "daily_short_credit_income_usd",
        "daily_net_financing_cost_usd",
        "daily_txn_cost_usd",
        "daily_turnover_usd",
        "daily_pair_gross_trading_pnl_usd",
        "daily_pair_net_pnl_usd",
        "daily_pair_net_ex_txn_usd",
        "fed_funds_rate",
        "is_rebal",
    ]


def export_v15_pair_ledger_to_excel(
    pair_daily: pd.DataFrame,
    bt: pd.DataFrame,
    out_xlsx: Path | str,
    *,
    full_date_range: bool = False,
    include_only_active_pairs: bool = True,
    active_min_abs_gross_sum_usd: float = 0.5,
    attribution_base_capital: float,
    ref_leverage: float,
    include_per_pair_sheets: bool = True,
    return_artifacts: bool = False,
    fund_attribution_xlsx: Path | str | None = None,
    align_fund_attribution: bool = False,
    clip_fund_trading_calendar: bool = True,
) -> dict[str, Any] | None:
    out_path = Path(out_xlsx)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    bt_work = bt.copy()
    pd_work = _prep_pair_daily(pair_daily)

    alignment_report: dict | None = None
    if align_fund_attribution and fund_attribution_xlsx is not None:
        bt_work, pd_work, alignment_report = align_pair_bt_to_fund_attribution(
            bt_work,
            pd_work,
            fund_attribution_xlsx,
            clip_fund_trading_calendar=clip_fund_trading_calendar,
        )

    if not full_date_range:
        m = _jan_2023_mask(pd_work["date"])
        pd_work = pd_work.loc[m]

    if include_only_active_pairs and not pd_work.empty:
        pd_work = pd_work.loc[_active_pairs_mask(pd_work, min_abs_gross=active_min_abs_gross_sum_usd)]

    if not pd_work.empty:
        tn = pd.to_numeric(pd_work["daily_txn_cost_usd"], errors="coerce").fillna(0.0)
        nn = pd.to_numeric(pd_work["daily_pair_net_pnl_usd"], errors="coerce").fillna(0.0)
        pd_work["daily_pair_net_ex_txn_usd"] = nn + tn

    recon = build_reconciliation_daily(pd_work, bt_work, attribution_base_capital=attribution_base_capital)
    if not full_date_range and not recon.empty:
        t = pd.to_datetime(recon["date"])
        recon = recon.loc[(t >= pd.Timestamp("2023-01-01")) & (t <= pd.Timestamp("2023-01-31"))].reset_index(
            drop=True
        )
    metrics = _ledger_metrics(recon)
    metrics["attribution_base_capital_usd"] = float(attribution_base_capital)
    metrics["ref_leverage"] = float(ref_leverage)

    export_cols = _all_pairs_columns()
    all_pairs = pd.DataFrame()
    if not pd_work.empty:
        for c in export_cols:
            if c not in pd_work.columns:
                if c == "short_financing_basis_usd":
                    pd_work[c] = pd.to_numeric(pd_work.get("short_financing_basis_usd"), errors="coerce")
                elif c == "daily_underlying_borrow_cost_usd":
                    pd_work[c] = pd.to_numeric(pd_work.get("daily_underlying_borrow_cost_usd"), errors="coerce")
                else:
                    pd_work[c] = np.nan
        all_pairs = pd_work[export_cols].copy()

    book_cols = ["date"] + [c for c in recon.columns if c.startswith("book_")]
    portfolio_financing = recon[book_cols].copy() if not recon.empty else recon

    with pd.ExcelWriter(out_path, engine="openpyxl") as w:
        all_pairs.to_excel(w, sheet_name="ALL_PAIRS", index=False)
        recon.to_excel(w, sheet_name="reconciliation_daily", index=False)
        portfolio_financing.to_excel(w, sheet_name="portfolio_financing", index=False)

        if include_per_pair_sheets and not all_pairs.empty:
            for sym, grp in all_pairs.groupby("etf"):
                name = str(sym)[:31].replace("/", "-")
                grp.to_excel(w, sheet_name=name, index=False)

    artifacts: dict[str, Any] = {
        "path": out_path,
        "metrics": metrics,
        "reconciliation_daily": recon,
        "all_pairs": all_pairs,
    }
    if alignment_report is not None:
        artifacts["alignment_report"] = alignment_report
        artifacts["aligned_bt"] = bt_work
        artifacts["aligned_pair_daily"] = pd_work
    if return_artifacts:
        return artifacts
    return None
