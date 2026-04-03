from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def build_audit_tabs(xlsx_path: Path) -> tuple[int, int, float, float]:
    all_df = pd.read_excel(xlsx_path, sheet_name="ALL_PAIRS")
    all_df["date"] = pd.to_datetime(all_df["date"], errors="coerce")
    all_df = all_df[all_df["date"].notna()].copy()

    num_cols = [
        "daily_pair_gross_trading_pnl_usd",
        "daily_txn_cost_usd",
        "daily_borrow_cost_usd",
        "daily_margin_debit_cost_usd",
        "daily_short_credit_income_usd",
        "daily_net_financing_cost_usd",
        "daily_total_cost_usd",
        "daily_pair_net_pnl_usd",
        "cum_pair_gross_trading_pnl_usd",
        "cum_txn_cost_usd",
        "cum_borrow_cost_usd",
        "cum_margin_debit_cost_usd",
        "cum_short_credit_income_usd",
        "cum_net_financing_cost_usd",
        "cum_total_cost_usd",
        "cum_pair_net_pnl_usd",
    ]
    for c in num_cols:
        if c in all_df.columns:
            all_df[c] = pd.to_numeric(all_df[c], errors="coerce").fillna(0.0)

    # Pair identity audit (TSLA/TSLL).
    pair_name = "TSLA/TSLL"
    pair_df = all_df[all_df["pair"] == pair_name].copy().sort_values("date")
    if pair_df.empty:
        raise RuntimeError("No TSLA/TSLL rows found in ALL_PAIRS.")

    a = pair_df[
        [
            "date",
            "pair",
            "daily_pair_gross_trading_pnl_usd",
            "daily_txn_cost_usd",
            "daily_borrow_cost_usd",
            "daily_margin_debit_cost_usd",
            "daily_short_credit_income_usd",
            "daily_net_financing_cost_usd",
            "daily_total_cost_usd",
            "daily_pair_net_pnl_usd",
        ]
    ].copy()
    a["daily_net_recalc_usd"] = a["daily_pair_gross_trading_pnl_usd"] - (
        a["daily_txn_cost_usd"] + a["daily_borrow_cost_usd"] + a["daily_net_financing_cost_usd"]
    )
    a["daily_identity_diff_usd"] = a["daily_pair_net_pnl_usd"] - a["daily_net_recalc_usd"]
    a["cum_gross_usd"] = a["daily_pair_gross_trading_pnl_usd"].cumsum()
    a["cum_txn_usd"] = a["daily_txn_cost_usd"].cumsum()
    a["cum_borrow_usd"] = a["daily_borrow_cost_usd"].cumsum()
    a["cum_debit_usd"] = a["daily_margin_debit_cost_usd"].cumsum()
    a["cum_credit_usd"] = a["daily_short_credit_income_usd"].cumsum()
    a["cum_net_financing_usd"] = a["daily_net_financing_cost_usd"].cumsum()
    a["cum_net_recalc_usd"] = a["daily_net_recalc_usd"].cumsum()
    a["cum_net_script_usd"] = a["daily_pair_net_pnl_usd"].cumsum()
    a["cum_identity_diff_usd"] = a["cum_net_script_usd"] - a["cum_net_recalc_usd"]

    # Book vs sum(pairs) daily reconciliation.
    pairs_daily = (
        all_df.groupby("date", as_index=False)[
            [
                "daily_pair_gross_trading_pnl_usd",
                "daily_txn_cost_usd",
                "daily_borrow_cost_usd",
                "daily_margin_debit_cost_usd",
                "daily_short_credit_income_usd",
                "daily_net_financing_cost_usd",
                "daily_pair_net_pnl_usd",
            ]
        ]
        .sum()
        .sort_values("date")
    )

    bt_path = Path("notebooks/data/backtest/v8_ew_nav_4.5x.csv")
    if not bt_path.exists():
        bt_path = Path("data/backtest/v8_ew_nav_4.5x.csv")
    if not bt_path.exists():
        raise FileNotFoundError("Could not find v8_ew_nav_4.5x.csv in notebooks/data/backtest or data/backtest.")

    bt = pd.read_csv(bt_path, parse_dates=["date"]).sort_values("date")
    for c in ["cum_long_pnl", "cum_short_pnl", "cum_costs", "cum_borrow", "cum_margin_debit", "cum_margin_credit", "nav"]:
        bt[c] = pd.to_numeric(bt[c], errors="coerce").fillna(0.0)

    b = pd.DataFrame({"date": bt["date"]})
    b["book_daily_gross_trading_pnl_usd"] = bt["cum_long_pnl"].diff().fillna(bt["cum_long_pnl"]) + bt["cum_short_pnl"].diff().fillna(bt["cum_short_pnl"])
    b["book_daily_txn_cost_usd"] = bt["cum_costs"].diff().fillna(bt["cum_costs"])
    b["book_daily_borrow_cost_usd"] = bt["cum_borrow"].diff().fillna(bt["cum_borrow"])
    b["book_daily_margin_debit_cost_usd"] = bt["cum_margin_debit"].diff().fillna(bt["cum_margin_debit"])
    b["book_daily_short_credit_income_usd"] = bt["cum_margin_credit"].diff().fillna(bt["cum_margin_credit"])
    b["book_daily_net_financing_cost_usd"] = b["book_daily_margin_debit_cost_usd"] - b["book_daily_short_credit_income_usd"]
    b["book_daily_net_pnl_usd"] = bt["nav"].diff().fillna(bt["nav"] - bt["nav"].iloc[0])

    r = b.merge(pairs_daily, on="date", how="inner").rename(
        columns={
            "daily_pair_gross_trading_pnl_usd": "pairs_daily_gross_trading_pnl_usd",
            "daily_txn_cost_usd": "pairs_daily_txn_cost_usd",
            "daily_borrow_cost_usd": "pairs_daily_borrow_cost_usd",
            "daily_margin_debit_cost_usd": "pairs_daily_margin_debit_cost_usd",
            "daily_short_credit_income_usd": "pairs_daily_short_credit_income_usd",
            "daily_net_financing_cost_usd": "pairs_daily_net_financing_cost_usd",
            "daily_pair_net_pnl_usd": "pairs_daily_net_pnl_usd",
        }
    )
    for k in ["gross_trading_pnl", "txn_cost", "borrow_cost", "margin_debit_cost", "short_credit_income", "net_financing_cost", "net_pnl"]:
        r[f"diff_{k}_usd"] = r[f"pairs_daily_{k}_usd"] - r[f"book_daily_{k}_usd"]
        r[f"cum_diff_{k}_usd"] = r[f"diff_{k}_usd"].cumsum()

    with pd.ExcelWriter(xlsx_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
        a.to_excel(writer, sheet_name="AUDIT_TSLA_TSLL", index=False)
        r.to_excel(writer, sheet_name="AUDIT_BOOK_VS_PAIRS", index=False)

    max_txn = float(r["diff_txn_cost_usd"].abs().max()) if len(r) else 0.0
    max_credit = float(r["diff_short_credit_income_usd"].abs().max()) if len(r) else 0.0
    return len(a), len(r), max_txn, max_credit


def main() -> None:
    ap = argparse.ArgumentParser(description="Add audit tabs to daily pair ledger workbook.")
    ap.add_argument("--xlsx", required=True, help="Path to daily_pair_ledger_4.5x.xlsx")
    args = ap.parse_args()

    xlsx_path = Path(args.xlsx)
    if not xlsx_path.exists():
        raise FileNotFoundError(xlsx_path)

    n_pair, n_recon, max_txn, max_credit = build_audit_tabs(xlsx_path)
    print(f"Updated: {xlsx_path}")
    print(f"AUDIT_TSLA_TSLL rows: {n_pair}")
    print(f"AUDIT_BOOK_VS_PAIRS rows: {n_recon}")
    print(f"Max abs txn diff (book vs pairs): {max_txn:,.2f}")
    print(f"Max abs short-credit diff (book vs pairs): {max_credit:,.2f}")


if __name__ == "__main__":
    main()
