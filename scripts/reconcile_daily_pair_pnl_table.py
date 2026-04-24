"""
Build a **day-by-day reconciliation** of pair-level P&L parts vs book vs fund.

Reads ``ALL_PAIRS`` + ``book_daily`` (+ optional ``fund_vs_sim_daily``) from a
Daily Pair workbook (e.g. ``Diamond_Creek_Daily_Pair_fund_aligned.xlsx``).

``daily_pair_net_pnl_usd`` may be blank in the file if Excel never recalculated
formulas; this script **reconstructs** pair net from value columns using the same
identity as ``scripts/daily_pair_workbook_formulas.py``:

    net = (long_pnl + short_pnl) - borrow - (margin_debit - short_credit) - txn

``daily_portfolio_margin_usd`` (synthetic **F**) is reconstructed from
``max(long_sh,0) * underlying_price`` (long margin basis), mean ``fed_funds_rate``,
``margin_spread_annual``, and ``financing_daycount`` — matching the injector.

Usage::

    python scripts/reconcile_daily_pair_pnl_table.py \\
        --aligned-xlsx notebooks/data/backtest/Diamond_Creek_Daily_Pair_fund_aligned.xlsx \\
        --out-csv notebooks/data/backtest/daily_pnl_reconciliation.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _coerce_num(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def build_reconciliation_table(
    aligned_xlsx: Path | str,
    *,
    margin_spread_annual: float = 0.0045,
    financing_daycount: float = 360.0,
) -> pd.DataFrame:
    p = Path(aligned_xlsx)
    ap = pd.read_excel(p, sheet_name="ALL_PAIRS")
    ap["date"] = pd.to_datetime(ap["date"], errors="coerce").dt.normalize()

    need = [
        "long_sh",
        "short_sh",
        "underlying_price",
        "etf_price",
        "daily_long_pnl_usd",
        "daily_short_pnl_usd",
        "daily_borrow_cost_usd",
        "daily_margin_debit_cost_usd",
        "daily_short_credit_income_usd",
        "daily_txn_cost_usd",
        "fed_funds_rate",
        "daily_pair_net_pnl_usd",
        "daily_pair_gross_trading_pnl_usd",
    ]
    for c in need:
        if c not in ap.columns:
            ap[c] = np.nan
    ap = _coerce_num(ap, need)

    lsh = ap["long_sh"].fillna(0.0)
    pu = ap["underlying_price"].fillna(0.0)
    ap["_lmb"] = np.maximum(lsh, 0.0) * pu

    g = (
        ap.groupby("date", sort=True)
        .agg(
            n_pairs=("etf", "count"),
            sum_long_pnl_usd=("daily_long_pnl_usd", "sum"),
            sum_short_pnl_usd=("daily_short_pnl_usd", "sum"),
            sum_borrow_usd=("daily_borrow_cost_usd", "sum"),
            sum_margin_debit_usd=("daily_margin_debit_cost_usd", "sum"),
            sum_short_credit_usd=("daily_short_credit_income_usd", "sum"),
            sum_txn_usd=("daily_txn_cost_usd", "sum"),
            sum_lmb_usd=("_lmb", "sum"),
            mean_fed_funds=("fed_funds_rate", "mean"),
            sum_pair_net_excel=("daily_pair_net_pnl_usd", "sum"),
            sum_gross_excel=("daily_pair_gross_trading_pnl_usd", "sum"),
        )
        .reset_index()
    )

    g["sum_gross_trading_usd"] = g["sum_long_pnl_usd"] + g["sum_short_pnl_usd"]
    g["sum_net_financing_usd"] = g["sum_margin_debit_usd"] - g["sum_short_credit_usd"]
    g["reconstructed_sum_pair_net_usd"] = (
        g["sum_gross_trading_usd"]
        - g["sum_borrow_usd"]
        - g["sum_net_financing_usd"]
        - g["sum_txn_usd"]
    )
    g["effective_margin_rate"] = g["mean_fed_funds"] + float(margin_spread_annual)
    g["synthetic_portfolio_margin_usd"] = (
        g["sum_lmb_usd"] * g["effective_margin_rate"] / float(financing_daycount)
    )
    g["reconstructed_net_after_margin_usd"] = (
        g["reconstructed_sum_pair_net_usd"] - g["synthetic_portfolio_margin_usd"]
    )
    g["sum_pair_net_excel_minus_recon_usd"] = g["sum_pair_net_excel"] - g["reconstructed_sum_pair_net_usd"]

    bd = pd.read_excel(p, sheet_name="book_daily")
    bd["date"] = pd.to_datetime(bd["date"], errors="coerce").dt.normalize()
    bd = _coerce_num(
        bd,
        [
            "book_nav_change_usd",
            "book_daily_txn_usd",
            "book_daily_borrow_usd",
            "book_daily_long_short_pnl_usd",
            "book_daily_net_margin_usd",
        ],
    )
    g = g.merge(bd, on="date", how="left")

    if "fund_vs_sim_daily" in pd.ExcelFile(p).sheet_names:
        fv = pd.read_excel(p, sheet_name="fund_vs_sim_daily")
        fv["date"] = pd.to_datetime(fv["date"], errors="coerce").dt.normalize()
        fv = _coerce_num(fv, ["fund_daily_pnl_usd", "book_nav_change_usd", "diff_fund_minus_sim_nav_usd"])
        g = g.merge(
            fv[["date", "fund_daily_pnl_usd", "diff_fund_minus_sim_nav_usd"]].rename(
                columns={"diff_fund_minus_sim_nav_usd": "fund_minus_book_nav_from_sheet_usd"}
            ),
            on="date",
            how="left",
        )
    else:
        g["fund_daily_pnl_usd"] = np.nan
        g["fund_minus_book_nav_from_sheet_usd"] = np.nan

    g["diff_gross_vs_book_ls_usd"] = g["sum_gross_trading_usd"] - g["book_daily_long_short_pnl_usd"]
    g["diff_borrow_vs_book_usd"] = g["sum_borrow_usd"] - g["book_daily_borrow_usd"]
    g["diff_txn_pair_sum_vs_book_usd"] = g["sum_txn_usd"] - g["book_daily_txn_usd"]
    g["diff_pair_net_vs_book_nav_usd"] = g["reconstructed_sum_pair_net_usd"] - g["book_nav_change_usd"]
    g["diff_pair_net_vs_fund_usd"] = g["reconstructed_sum_pair_net_usd"] - g["fund_daily_pnl_usd"]
    g["diff_net_after_vs_fund_usd"] = g["reconstructed_net_after_margin_usd"] - g["fund_daily_pnl_usd"]
    g["diff_net_after_vs_book_nav_usd"] = g["reconstructed_net_after_margin_usd"] - g["book_nav_change_usd"]

    # Ex-txn pair net (for attribution-style checks): net + txn
    g["reconstructed_sum_pair_net_ex_txn_usd"] = g["reconstructed_sum_pair_net_usd"] + g["sum_txn_usd"]
    g["book_nav_change_plus_book_txn_usd"] = (
        pd.to_numeric(g["book_nav_change_usd"], errors="coerce").fillna(0.0)
        + pd.to_numeric(g["book_daily_txn_usd"], errors="coerce").fillna(0.0)
    )
    g["diff_ex_txn_net_vs_nav_plus_txn_usd"] = (
        g["reconstructed_sum_pair_net_ex_txn_usd"] - g["book_nav_change_plus_book_txn_usd"]
    )

    col_order = [
        "date",
        "n_pairs",
        "sum_long_pnl_usd",
        "sum_short_pnl_usd",
        "sum_gross_trading_usd",
        "book_daily_long_short_pnl_usd",
        "diff_gross_vs_book_ls_usd",
        "sum_borrow_usd",
        "book_daily_borrow_usd",
        "diff_borrow_vs_book_usd",
        "sum_margin_debit_usd",
        "sum_short_credit_usd",
        "sum_net_financing_usd",
        "book_daily_net_margin_usd",
        "sum_txn_usd",
        "book_daily_txn_usd",
        "diff_txn_pair_sum_vs_book_usd",
        "reconstructed_sum_pair_net_usd",
        "sum_pair_net_excel",
        "sum_pair_net_excel_minus_recon_usd",
        "sum_lmb_usd",
        "mean_fed_funds",
        "effective_margin_rate",
        "synthetic_portfolio_margin_usd",
        "reconstructed_net_after_margin_usd",
        "book_nav_change_usd",
        "fund_daily_pnl_usd",
        "fund_minus_book_nav_from_sheet_usd",
        "diff_pair_net_vs_book_nav_usd",
        "diff_pair_net_vs_fund_usd",
        "diff_net_after_vs_book_nav_usd",
        "diff_net_after_vs_fund_usd",
        "reconstructed_sum_pair_net_ex_txn_usd",
        "book_nav_change_plus_book_txn_usd",
        "diff_ex_txn_net_vs_nav_plus_txn_usd",
    ]
    return g[[c for c in col_order if c in g.columns]].sort_values("date").reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--aligned-xlsx",
        type=Path,
        default=Path("notebooks/data/backtest/Diamond_Creek_Daily_Pair_fund_aligned.xlsx"),
    )
    ap.add_argument(
        "--out-csv",
        type=Path,
        default=Path("notebooks/data/backtest/daily_pnl_reconciliation.csv"),
    )
    ap.add_argument("--margin-spread", type=float, default=0.0045)
    ap.add_argument("--financing-daycount", type=float, default=360.0)
    args = ap.parse_args()

    df = build_reconciliation_table(
        args.aligned_xlsx,
        margin_spread_annual=args.margin_spread,
        financing_daycount=args.financing_daycount,
    )
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 40)
    print(df.to_string())
    print(f"\nWrote {args.out_csv.resolve()}")


if __name__ == "__main__":
    main()
