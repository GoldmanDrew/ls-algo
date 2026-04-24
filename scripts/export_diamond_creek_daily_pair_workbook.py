"""
Write the **Daily Pair** workbook (``ALL_PAIRS`` + ``portfolio_ledger``).

Column **H** in ``ALL_PAIRS`` is ``long_margin_basis_usd`` (debit / margin pool
basis), consistent with ``SUMIFS`` aggregation in ``portfolio_ledger``.

When ``use_portfolio_excel_formulas=True``, a post-pass
(``scripts/daily_pair_workbook_formulas.py``) writes:

- **``book_daily``** (values): book ``nav``, ``nav`` change, ``cum_costs`` txn
  flow, borrow, and long+short trading P&L — single source for
  ``VLOOKUP``/checks.
- **``ALL_PAIRS``** (formulas): algebraic notionals, margin/financing bases,
  gross/net, ``daily_net_financing_cost_usd``, pair gross/net/ex-txn from engine
  inputs in ``O:P:Q:S:T:V``.
- **``portfolio_ledger``** (formulas): ``SUMIFS`` / ``AVERAGEIFS`` into
  ``ALL_PAIRS`` for long/short/net pair PnL roll-ups, margin, and txn; book
  columns from ``book_daily`` via ``VLOOKUP`` (see ``PORTFOLIO_LEDGER_COLUMNS``).
- **``portfolio_ledger_dc``** + **``dc_workbook_settings``** (optional): legacy
  nine-column portfolio layout aligned with ``Diamond_Creek_Daily_Pairs.xlsx``
  semantics (see mapping in ``daily_pair_workbook_formulas`` module docstring).
- **``checks``** (formulas): date rollups vs ``book_daily`` (gross, borrow,
  txn, attribution-style ``ex_txn`` vs ``nav_change + txn``).
- **Per-pair sheets** (formulas): ``SUMIFS`` back to ``ALL_PAIRS`` for the same
  fields plus running ``cum_borrow_cost_usd``.

Fund daily series in ``fund_vs_sim_daily`` stay value-only (external workbook).
"""

from __future__ import annotations

import re
from pathlib import Path
import numpy as np
import pandas as pd

from scripts.align_pair_bt_to_fund_attribution import align_pair_bt_to_fund_attribution
from scripts.compare_dc_etf_attribution import load_fund_daily
from scripts.daily_pair_workbook_formulas import (
    PORTFOLIO_LEDGER_COLUMNS,
    PORTFOLIO_LEDGER_DC_HEADERS,
    _ensure_cum_borrow_cost_usd,
    apply_portfolio_level_cost_model,
    build_book_daily_dataframe,
    fill_portfolio_ledger_values,
    inject_daily_pair_workbook_formulas,
    normalize_series_for_book_merge,
)
from scripts.export_diamond_creek_v15_pair_ledger import (
    _jan_2023_mask,
    _prep_pair_daily,
    _active_pairs_mask,
    _all_pairs_columns,
    build_reconciliation_daily,
)


def _excel_safe_sheet_name(name: str) -> str:
    s = re.sub(r'[\[\]:*?/\\]', "-", str(name).strip())[:31]
    return s or "PAIR"


def _pair_tab_key(under: str, etf: str) -> str:
    """``UNDER_ETF`` (same convention as ``Diamond_Creek_Daily_Pair_new.xlsx``)."""
    return _excel_safe_sheet_name(f"{under}_{etf}")


def _per_pair_breakdown_sheet(df_one: pd.DataFrame) -> pd.DataFrame:
    """
    One pair, sorted by date: rename price columns, append ``cum_borrow_cost_usd``
    and optional Excel flow columns.
    """
    out = df_one.sort_values("date").copy()
    u = str(out["under"].iloc[0])
    e = str(out["etf"].iloc[0])
    col_u = f"{u} Price"
    col_e = f"{e} Price"
    out = out.rename(columns={"underlying_price": col_u, "etf_price": col_e})
    b = pd.to_numeric(out.get("daily_borrow_cost_usd"), errors="coerce").fillna(0.0)
    out["cum_borrow_cost_usd"] = b.cumsum()
    # Column order: legacy daily-pair layout first, then remaining metrics.
    head = [
        "date",
        "etf",
        "under",
        col_u,
        col_e,
        "long_sh",
        "short_sh",
        "long_notional_usd",
        "short_notional_usd",
        "long_margin_basis_usd",
        "short_financing_basis_usd",
        "gross_notional_usd",
        "net_notional_usd",
        "borrow_rate_annual",
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
        "cum_borrow_cost_usd",
    ]
    tail = [
        c
        for c in (
            "excel_long_pnl_level_usd",
            "excel_short_pnl_level_usd",
            "excel_long_trades_usd",
            "excel_short_trades_usd",
        )
        if c in out.columns
    ]
    cols = [c for c in head if c in out.columns] + [c for c in out.columns if c not in head + tail] + tail
    return out[cols]


def build_fund_vs_sim_daily_january(
    bt: pd.DataFrame,
    fund_attribution_xlsx: Path | str,
    pair_daily_jan_all_pairs: pd.DataFrame,
    *,
    attribution_base_capital: float,
) -> pd.DataFrame:
    """
    Merge fund *Daily PnL* with sim ``book_nav_change_usd`` (January 2023).

    ``pair_daily_jan_all_pairs`` should include **every** pair row for the month
    (not the \"active pairs only\" export filter) so summed pair diagnostics in
    ``reconciliation_daily`` match the book.
    """
    fund = load_fund_daily(fund_attribution_xlsx)
    fdj = fund.loc[(fund["date"] >= pd.Timestamp("2023-01-01")) & (fund["date"] <= pd.Timestamp("2023-01-31"))]
    pd_j = pair_daily_jan_all_pairs.loc[_jan_2023_mask(pair_daily_jan_all_pairs["date"])].copy()
    recon = build_reconciliation_daily(
        pd_j,
        bt,
        attribution_base_capital=attribution_base_capital,
    )
    m = fdj.merge(recon[["date", "book_nav_change_usd"]], on="date", how="outer").sort_values("date")
    m["diff_fund_minus_sim_nav_usd"] = pd.to_numeric(m["fund_daily_pnl_usd"], errors="coerce").fillna(0.0) - pd.to_numeric(
        m["book_nav_change_usd"], errors="coerce"
    ).fillna(0.0)
    m["fund_cum_pnl_usd"] = pd.to_numeric(m["fund_daily_pnl_usd"], errors="coerce").fillna(0.0).cumsum()
    m["sim_cum_nav_change_usd"] = pd.to_numeric(m["book_nav_change_usd"], errors="coerce").fillna(0.0).cumsum()
    m["diff_cum_usd"] = m["fund_cum_pnl_usd"] - m["sim_cum_nav_change_usd"]
    return m.reset_index(drop=True)


def export_diamond_creek_daily_pair_workbook(
    pair_daily: pd.DataFrame,
    bt: pd.DataFrame,
    out_xlsx: Path | str,
    *,
    full_date_range: bool = True,
    include_only_active_pairs: bool = False,
    attribution_base_capital: float = 10_000_000.0,
    fund_attribution_xlsx: Path | str | None = None,
    include_all_pairs_sheet: bool = True,
    include_per_pair_sheets: bool = True,
    include_fund_vs_sim_daily: bool | None = None,
    use_portfolio_excel_formulas: bool = False,
    align_to_fund_attribution: bool = False,
    clip_fund_trading_calendar: bool = True,
    pair_borrow_column_formula: bool = False,
    margin_spread_annual: float = 0.0045,
    financing_daycount: float = 360.0,
    debit_margin_rate_annual: float = 0.0432,
    include_portfolio_ledger_dc: bool = True,
    trading_days: float = 252.0,
) -> Path:
    """
    When ``use_portfolio_excel_formulas`` and ``include_portfolio_ledger_dc`` are
    true, appends ``cum_borrow_cost_usd`` / ``pnl_net_of_borrow_usd`` on
    ``ALL_PAIRS``, writes ``dc_workbook_settings`` (**B1** day-count, **B2**
    debit rate), and ``portfolio_ledger_dc`` (reference-style nine columns).
    """
    del pair_borrow_column_formula  # reserved for future formula mode
    out_path = Path(out_xlsx)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    bt_w = bt.copy()
    pd_all = _prep_pair_daily(pair_daily)

    if align_to_fund_attribution and fund_attribution_xlsx is not None:
        bt_w, pd_all, _ = align_pair_bt_to_fund_attribution(
            bt_w, pd_all, fund_attribution_xlsx, clip_fund_trading_calendar=clip_fund_trading_calendar
        )

    if not full_date_range:
        pd_jan = pd_all.loc[_jan_2023_mask(pd_all["date"])].copy()
    else:
        pd_jan = pd_all.copy()
    # Keep full ``bt_w`` so cumulative diffs (e.g. transaction costs) are correct.

    pd_w = pd_jan
    if include_only_active_pairs and not pd_w.empty:
        pd_w = pd_w.loc[_active_pairs_mask(pd_w, min_abs_gross=0.5)]

    cols = _all_pairs_columns()
    for c in cols:
        if c not in pd_w.columns:
            pd_w[c] = np.nan
    all_pairs = pd_w[cols].copy() if not pd_w.empty else pd.DataFrame(columns=cols)
    if not all_pairs.empty:
        for c in ("excel_long_trades_usd", "excel_short_trades_usd"):
            if c in pd_w.columns and c not in all_pairs.columns:
                all_pairs[c] = pd_w[c]
    if not all_pairs.empty and "date" in all_pairs.columns:
        all_pairs = all_pairs.copy()
        all_pairs["date"] = normalize_series_for_book_merge(all_pairs["date"])
        all_pairs = apply_portfolio_level_cost_model(all_pairs, verify_identities=True)
    if use_portfolio_excel_formulas and not all_pairs.empty:
        all_pairs = all_pairs.sort_values(["under", "etf", "date"], kind="mergesort").reset_index(drop=True)
    if use_portfolio_excel_formulas and include_portfolio_ledger_dc and not all_pairs.empty:
        all_pairs = _ensure_cum_borrow_cost_usd(all_pairs)
        all_pairs["pnl_net_of_borrow_usd"] = np.nan

    book_daily = pd.DataFrame()
    if not all_pairs.empty:
        book_daily = build_book_daily_dataframe(
            bt_w,
            all_pairs["date"].unique(),
            attribution_base_capital=attribution_base_capital,
        )

    fund_by_date: dict[pd.Timestamp, float] | None = None
    if fund_attribution_xlsx is not None:
        fdf = load_fund_daily(fund_attribution_xlsx)
        if not fdf.empty and "fund_daily_pnl_usd" in fdf.columns:
            fund_by_date = {}
            for d, v in zip(
                pd.to_datetime(fdf["date"], errors="coerce").dt.normalize(),
                pd.to_numeric(fdf["fund_daily_pnl_usd"], errors="coerce"),
            ):
                if pd.isna(d):
                    continue
                fund_by_date[pd.Timestamp(d)] = float(v) if pd.notna(v) else np.nan

    portfolio = pd.DataFrame(columns=list(PORTFOLIO_LEDGER_COLUMNS))
    if not all_pairs.empty:
        portfolio = fill_portfolio_ledger_values(
            all_pairs,
            book_daily,
            margin_spread_annual=margin_spread_annual,
            financing_daycount=financing_daycount,
            fund_by_date=fund_by_date,
        )

    if include_fund_vs_sim_daily is None:
        include_fund_vs_sim_daily = bool(fund_attribution_xlsx) and (not full_date_range)

    fund_vs_sim = pd.DataFrame()
    if include_fund_vs_sim_daily and fund_attribution_xlsx is not None:
        pd_jan_full = pd_jan.copy()
        for c in cols:
            if c not in pd_jan_full.columns:
                pd_jan_full[c] = np.nan
        if not pd_jan_full.empty:
            pd_jan_full = apply_portfolio_level_cost_model(pd_jan_full, verify_identities=True)
        fund_vs_sim = build_fund_vs_sim_daily_january(
            bt_w,
            fund_attribution_xlsx,
            pd_jan_full,
            attribution_base_capital=attribution_base_capital,
        )

    used_names: set[str] = set()

    with pd.ExcelWriter(out_path, engine="openpyxl") as w:
        if include_all_pairs_sheet or not all_pairs.empty:
            all_pairs.to_excel(w, sheet_name="ALL_PAIRS", index=False)
            used_names.add("ALL_PAIRS")
        if not book_daily.empty:
            book_daily.to_excel(w, sheet_name="book_daily", index=False)
            used_names.add("book_daily")
        if not fund_vs_sim.empty:
            fund_vs_sim.to_excel(w, sheet_name="fund_vs_sim_daily", index=False)
            used_names.add("fund_vs_sim_daily")
        portfolio.to_excel(w, sheet_name="portfolio_ledger", index=False)
        used_names.add("portfolio_ledger")
        if not portfolio.empty:
            portfolio_daily = portfolio[
                [
                    "date",
                    "daily_txn_pairs_sum_usd",
                    "daily_portfolio_margin_usd",
                    "book_daily_txn_cost_usd",
                    "book_daily_net_margin_usd",
                    "book_daily_nav_change_usd",
                ]
            ].copy()
            portfolio_daily = portfolio_daily.rename(
                columns={
                    "daily_txn_pairs_sum_usd": "portfolio_daily_txn_cost_usd",
                    "daily_portfolio_margin_usd": "portfolio_daily_long_margin_cost_usd",
                }
            )
            portfolio_daily.to_excel(w, sheet_name="portfolio_daily", index=False)
            used_names.add("portfolio_daily")

        if use_portfolio_excel_formulas and not all_pairs.empty:
            if include_portfolio_ledger_dc and not portfolio.empty:
                dc_settings = pd.DataFrame(
                    [
                        ["financing_daycount", float(financing_daycount)],
                        ["debit_margin_rate_annual", float(debit_margin_rate_annual)],
                        ["trading_days", float(trading_days)],
                    ]
                )
            else:
                dc_settings = pd.DataFrame([["trading_days", float(trading_days)]])
            dc_settings.to_excel(w, sheet_name="dc_workbook_settings", index=False, header=False)
            used_names.add("dc_workbook_settings")
        if use_portfolio_excel_formulas and include_portfolio_ledger_dc and not portfolio.empty and not all_pairs.empty:
            dc_port = pd.DataFrame(
                np.nan,
                index=np.arange(len(portfolio), dtype=int),
                columns=list(PORTFOLIO_LEDGER_DC_HEADERS),
            )
            dc_port["Date"] = portfolio["date"].values
            dc_port.to_excel(w, sheet_name="portfolio_ledger_dc", index=False)
            used_names.add("portfolio_ledger_dc")

        if include_per_pair_sheets and not all_pairs.empty:
            for (under, etf), grp in all_pairs.groupby(["under", "etf"], sort=False):
                tab = _pair_tab_key(str(under), str(etf))
                base = tab
                n = 1
                while tab in used_names:
                    n += 1
                    tab = _excel_safe_sheet_name(f"{base[:27]}_{n}")
                used_names.add(tab)
                _per_pair_breakdown_sheet(grp).to_excel(w, sheet_name=tab, index=False)

    if use_portfolio_excel_formulas and not all_pairs.empty:
        # See ``scripts/daily_pair_workbook_formulas.py`` for which cells are formula-driven.
        inject_daily_pair_workbook_formulas(
            out_path,
            all_pairs_data_rows=len(all_pairs),
            portfolio_data_rows=len(portfolio),
            checks_data_rows=len(portfolio),
            inject_per_pair_sumifs=True,
            verbose=True,
            financing_daycount=financing_daycount,
            include_portfolio_ledger_dc=include_portfolio_ledger_dc,
        )

    return out_path
