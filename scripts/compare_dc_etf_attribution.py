"""
Load **DC ETF Arb Attribution.xlsx** (sheets ``Daily PnL`` and ``Monthly Attribution``)
and reconcile fund rows with the Diamond Creek backtest outputs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

JAN_2023_START = pd.Timestamp("2023-01-01")
JAN_2023_END = pd.Timestamp("2023-01-31")
JAN_2023_PERIOD = pd.Period("2023-01", freq="M")


def load_fund_daily(xlsx: Path | str) -> pd.DataFrame:
    p = Path(xlsx)
    d = pd.read_excel(p, sheet_name="Daily PnL", header=1)
    d = d.loc[pd.to_datetime(d["Date"], errors="coerce").notna()].copy()
    d["date"] = pd.to_datetime(d["Date"]).dt.normalize()
    out = d.rename(columns={"PnL": "fund_daily_pnl_usd"})[["date", "fund_daily_pnl_usd"]].copy()
    out["fund_daily_pnl_usd"] = pd.to_numeric(out["fund_daily_pnl_usd"], errors="coerce")
    return out.reset_index(drop=True)


def load_fund_monthly(xlsx: Path | str) -> pd.DataFrame:
    """Parse the wide *Monthly Attribution* layout (merged headers in rows 1–2)."""
    p = Path(xlsx)
    raw = pd.read_excel(p, sheet_name="Monthly Attribution", header=None)
    rows: list[dict[str, Any]] = []
    for i in range(3, len(raw)):
        month_cell = raw.iloc[i, 1]
        if pd.isna(month_cell):
            continue
        try:
            m = pd.Period(str(month_cell), freq="M")
        except (ValueError, TypeError):
            continue
        net = raw.iloc[i, 19]
        if pd.isna(net):
            continue
        rows.append({"date": m.to_timestamp(how="start"), "year_month": str(m), "fund_net_pnl_usd": float(net)})
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def filter_fund_to_jan_2023(
    fund_m: pd.DataFrame, fund_d: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    mj = fund_m.loc[fund_m["year_month"] == str(JAN_2023_PERIOD)].copy()
    dj = fund_d.loc[(fund_d["date"] >= JAN_2023_START) & (fund_d["date"] <= JAN_2023_END)].copy()
    return mj.reset_index(drop=True), dj.reset_index(drop=True)


def fund_excel_tieout(
    fund_d: pd.DataFrame, fund_m: pd.DataFrame, *, year_month: str = "2023-01"
) -> dict[str, float]:
    per = pd.Period(year_month, freq="M")
    dsum = float(
        fund_d.loc[fund_d["date"].dt.to_period("M") == per, "fund_daily_pnl_usd"].sum()
    )
    mslice = fund_m.loc[fund_m["year_month"] == year_month, "fund_net_pnl_usd"]
    mnet = float(mslice.iloc[0]) if not mslice.empty else float("nan")
    return {
        "fund_daily_sum_usd": dsum,
        "fund_monthly_net_usd": mnet,
        "abs_diff_usd": abs(dsum - mnet) if np.isfinite(mnet) else float("nan"),
    }


def check_fund_daily_rolls_to_monthly_pre_fee(
    fund_m_j: pd.DataFrame, fund_d_j: pd.DataFrame, *, tol_usd: float = 1.0
) -> bool:
    if fund_m_j.empty or fund_d_j.empty:
        return False
    dsum = float(fund_d_j["fund_daily_pnl_usd"].sum())
    mnet = float(fund_m_j["fund_net_pnl_usd"].iloc[0])
    return abs(dsum - mnet) <= tol_usd


def aggregate_sim_from_bt(
    bt: pd.DataFrame, *, attribution_base_capital: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (daily attribution frame indexed by date, monthly sums)."""
    d = bt.sort_index().copy()
    d.index = pd.to_datetime(d.index)
    nav = pd.to_numeric(d["nav"], errors="coerce")
    nav_change = nav.diff()
    if len(nav_change):
        nav_change.iloc[0] = float(nav.iloc[0]) - float(attribution_base_capital)
    daily = pd.DataFrame(
        {
            "date": d.index,
            "nav_change_usd": nav_change.values,
            "attribution_daily_net": nav_change.values,
        }
    )
    daily = daily.set_index("date")
    monthly = (
        daily["attribution_daily_net"].resample("ME").sum().to_frame("sim_net_pnl_usd").reset_index()
    )
    return daily.reset_index(), monthly


def build_sim_daily_for_compare(
    bt: pd.DataFrame,
    pair_daily: pd.DataFrame,
    *,
    attribution_base_capital: float,
) -> pd.DataFrame:
    """Merge book NAV change with same-day pair P&L aggregates."""
    d, _ = aggregate_sim_from_bt(bt, attribution_base_capital=attribution_base_capital)
    pd_ = pair_daily.copy()
    if pd_.empty:
        return d
    pd_["date"] = pd.to_datetime(pd_["date"]).dt.normalize()
    gcols = [
        "daily_long_pnl_usd",
        "daily_short_pnl_usd",
        "daily_borrow_cost_usd",
        "daily_margin_debit_cost_usd",
        "daily_short_credit_income_usd",
        "daily_txn_cost_usd",
        "daily_pair_net_pnl_usd",
    ]
    for c in gcols:
        if c not in pd_.columns:
            pd_[c] = 0.0
    agg = (
        pd_.groupby("date", as_index=False)[gcols]
        .sum(numeric_only=True)
        .rename(
            columns={
                "daily_long_pnl_usd": "sum_pair_long_pnl_usd",
                "daily_short_pnl_usd": "sum_pair_short_pnl_usd",
                "daily_borrow_cost_usd": "sum_pair_borrow_usd",
                "daily_margin_debit_cost_usd": "sum_pair_margin_debit_usd",
                "daily_short_credit_income_usd": "sum_pair_short_credit_usd",
                "daily_txn_cost_usd": "sum_pair_txn_usd",
                "daily_pair_net_pnl_usd": "sum_pair_net_pnl_usd",
            }
        )
    )
    out = d.merge(agg, on="date", how="left")
    for c in list(agg.columns):
        if c != "date" and c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)
    return out


def sim_recon_window(sim_d: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    s = pd.to_datetime(sim_d["date"])
    return sim_d.loc[(s >= start) & (s <= end)].copy()


def reconcile_daily(fund_d: pd.DataFrame, sim_d: pd.DataFrame) -> pd.DataFrame:
    a = fund_d.rename(columns={"fund_daily_pnl_usd": "fund_pnl_usd"}).copy()
    b = sim_d.rename(columns={"attribution_daily_net": "sim_nav_change_usd"}).copy()
    m = a.merge(b, on="date", how="outer").sort_values("date")
    m["diff_usd"] = pd.to_numeric(m["fund_pnl_usd"], errors="coerce").fillna(0.0) - pd.to_numeric(
        m["sim_nav_change_usd"], errors="coerce"
    ).fillna(0.0)
    return m.reset_index(drop=True)


def reconcile_monthly(fund_m: pd.DataFrame, monthly_sim: pd.DataFrame) -> pd.DataFrame:
    fs = fund_m.copy()
    fs["ym"] = pd.to_datetime(fs["date"]).dt.to_period("M").astype(str)
    ss = monthly_sim.copy()
    ss["ym"] = pd.to_datetime(ss["date"]).dt.to_period("M").astype(str)
    m = fs.merge(ss, on="ym", how="outer", suffixes=("_fund", "_sim"))
    if "fund_net_pnl_usd" in m.columns and "sim_net_pnl_usd" in m.columns:
        m["diff_usd"] = pd.to_numeric(m["fund_net_pnl_usd"], errors="coerce").fillna(0.0) - pd.to_numeric(
            m["sim_net_pnl_usd"], errors="coerce"
        ).fillna(0.0)
    return m.sort_values("ym").reset_index(drop=True)


def self_check_sim_reconciliation_daily(recon: pd.DataFrame) -> str:
    """Summarise ``reconciliation_daily`` from ``export_v15_pair_ledger_to_excel``."""
    if recon is None or (isinstance(recon, pd.DataFrame) and recon.empty):
        return "reconciliation_daily is empty"
    lines = [f"rows={len(recon)}", f"columns={list(recon.columns)}"]
    for c in recon.columns:
        if "diff" in c.lower() or "abs" in c.lower():
            s = pd.to_numeric(recon[c], errors="coerce")
            lines.append(f"max_abs({c}) = {s.abs().max():.6g}")
    return "\n".join(lines)


def fund_and_sim_tieout(recon: pd.DataFrame, fund_d_j: pd.DataFrame) -> dict[str, float]:
    """Coarse Jan totals: fund daily vs summed sim columns if present."""
    out: dict[str, float] = {}
    if fund_d_j is not None and not fund_d_j.empty:
        out["fund_jan_sum_usd"] = float(pd.to_numeric(fund_d_j["fund_daily_pnl_usd"], errors="coerce").sum())
    if recon is None or recon.empty:
        return out
    if "book_nav_change_usd" in recon.columns:
        out["sim_book_nav_jan_sum_usd"] = float(pd.to_numeric(recon["book_nav_change_usd"], errors="coerce").sum())
    if "sum_pair_ex_txn_net_usd" in recon.columns:
        out["sim_pair_ex_txn_jan_sum_usd"] = float(
            pd.to_numeric(recon["sum_pair_ex_txn_net_usd"], errors="coerce").sum()
        )
    return out
