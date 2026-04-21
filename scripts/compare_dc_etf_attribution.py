"""
Compare Diamond Creek ETF Arb fund workbook (DC ETF Arb Attribution.xlsx) to simulated ALL_BT / pair ledger.

Fund workbook sheets: Monthly Attribution, Daily PnL.
Sim sources: ALL_BT (reference leverage), optional ALL_PAIR_DAILY for active pair counts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _daily_flow_from_cum(cum: pd.Series) -> pd.Series:
    s = pd.to_numeric(cum, errors="coerce").ffill().fillna(0.0)
    d = s.diff().fillna(0.0)
    if len(d) > 0:
        d.iloc[0] = float(s.iloc[0])
    return d


def _nav_daily_change_full_series(
    nav: pd.Series, attribution_base_capital: float | None
) -> pd.Series:
    """First row = NAV[0] − base (same convention as export_diamond_creek_v15_pair_ledger)."""
    nav = pd.to_numeric(nav, errors="coerce").ffill()
    out = pd.Series(index=nav.index, dtype=float)
    if len(nav) == 0:
        return out
    base = attribution_base_capital
    if base is None or not np.isfinite(float(base)):
        base = float(nav.iloc[0])
    out.iloc[0] = float(nav.iloc[0]) - float(base)
    if len(nav) > 1:
        out.iloc[1:] = nav.iloc[1:].to_numpy(dtype=float) - nav.iloc[:-1].to_numpy(dtype=float)
    return out


# DC ETF Arb Attribution.xlsx — "Monthly Attribution" data row layout (openpyxl):
# Col 1 = month, 3 = benchmark rate, then **blank col 4**, then long/short notional (5–6),
# gross long/short (7–8), **blank col 9**, T-costs (10), borrow (11), margin (12), pre-fee (13),
# blanks, mgmt/incentive/total fees (15–17), net (19). Using 4,5,6,7,9,10… put notionals in the wrong fields.
_FUND_MONTHLY_COLS: list[int] = [1, 3, 5, 6, 7, 8, 10, 11, 12, 13, 15, 16, 17, 19]


def load_fund_monthly(path: str | Path) -> pd.DataFrame:
    """Parse Monthly Attribution (header rows 1–2, data from row 3)."""
    path = Path(path)
    raw = pd.read_excel(path, sheet_name="Monthly Attribution", header=None)
    sub = raw.iloc[3:, _FUND_MONTHLY_COLS].copy()
    sub.columns = [
        "month",
        "benchmark_rate",
        "long_notional",
        "short_notional",
        "gross_long",
        "gross_short",
        "t_costs",
        "borrow",
        "margin",
        "pre_fee_pnl",
        "mgmt_fee",
        "incentive_fee",
        "total_fees",
        "net_pnl",
    ]
    sub = sub.dropna(subset=["month"], how="all")
    sub["month"] = pd.to_datetime(sub["month"].astype(str), errors="coerce")
    sub = sub[sub["month"].notna()]
    for c in sub.columns:
        if c == "month":
            continue
        sub[c] = pd.to_numeric(sub[c], errors="coerce")
    sub["month_period"] = sub["month"].dt.to_period("M")
    return sub.reset_index(drop=True)


def load_fund_daily(path: str | Path) -> pd.DataFrame:
    raw = pd.read_excel(path, sheet_name="Daily PnL", header=None)
    sub = raw.iloc[2:, [1, 2, 3]].copy()
    sub.columns = ["date", "pnl_usd", "active_pairs"]
    sub = sub.dropna(subset=["date"], how="all")
    sub["date"] = pd.to_datetime(sub["date"], errors="coerce")
    sub["pnl_usd"] = pd.to_numeric(sub["pnl_usd"], errors="coerce")
    sub["active_pairs"] = pd.to_numeric(sub["active_pairs"], errors="coerce")
    return sub[sub["date"].notna()].reset_index(drop=True)


def aggregate_sim_from_bt(
    bt: pd.DataFrame,
    *,
    attribution_base_capital: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Daily flows from ALL_BT cum columns; monthly sums aligned to calendar month (period M).

    ``attribution_base_capital``: starting NAV / capital (e.g. ``CFG['capital_usd']``). Required for
    daily ``nav_change`` to match fund daily PnL (first day = NAV[0] − base). If omitted, first-day
    NAV change uses base = NAV[0] (no jump), which misaligns vs fund files.

    Returns:
      daily: index date — long_gross, short_gross, gross_trading, txn, borrow,
             margin_net (debit − short credit), pre_fee (gross − txn − borrow − margin_net),
             nav_change
      monthly: one row per month_period with same fields summed + end-of-month notionals if present
    """
    b = bt.copy().sort_index()
    b.index = pd.to_datetime(b.index)

    long_g = _daily_flow_from_cum(b["cum_long_pnl"])
    short_g = _daily_flow_from_cum(b["cum_short_pnl"])
    txn = _daily_flow_from_cum(b["cum_costs"])
    borrow = _daily_flow_from_cum(b["cum_borrow"])
    mdeb = _daily_flow_from_cum(b["cum_margin_debit"])
    mcred = _daily_flow_from_cum(b["cum_margin_credit"])
    margin_net = mdeb - mcred
    nav = pd.to_numeric(b["nav"], errors="coerce").ffill()
    nav_ch = _nav_daily_change_full_series(nav, attribution_base_capital)

    gross_trading = long_g + short_g
    pre_fee = gross_trading - txn - borrow - margin_net

    daily = pd.DataFrame(
        {
            "long_gross": long_g,
            "short_gross": short_g,
            "gross_trading": gross_trading,
            "txn": txn,
            "borrow": borrow,
            "margin_net": margin_net,
            "pre_fee_pnl": pre_fee,
            "nav_change": nav_ch,
        }
    )
    daily.index.name = "date"

    g = daily.groupby(pd.Grouper(freq="ME")).sum(numeric_only=True)
    g["month_period"] = g.index.to_period("M")

    long_n = b.get("long_notional")
    short_n = b.get("short_notional")
    if long_n is not None and short_n is not None:
        ln = pd.to_numeric(long_n, errors="coerce")
        sn = pd.to_numeric(short_n, errors="coerce")
        agg_n = pd.DataFrame({"long_notional": ln, "short_notional": sn})
        last_n = agg_n.groupby(pd.Grouper(freq="ME")).last()
        g = g.join(last_n, how="left")

    monthly = g.reset_index()
    return daily, monthly


def count_active_pairs_daily(pair_daily: pd.DataFrame, min_gross_usd: float = 1.0) -> pd.Series:
    """Per date: count pairs with |gross_notional| or similar above threshold."""
    d = pair_daily.copy()
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    if "gross_notional_usd" in d.columns:
        g = pd.to_numeric(d["gross_notional_usd"], errors="coerce").fillna(0.0).abs()
    else:
        ln = pd.to_numeric(d.get("long_notional_usd", 0.0), errors="coerce").fillna(0.0)
        sn = pd.to_numeric(d.get("short_notional_usd", 0.0), errors="coerce").fillna(0.0)
        g = (ln.abs() + sn.abs()) * 0.5
    d["_g"] = g
    return (
        d[d["_g"] >= min_gross_usd]
        .groupby("date")["etf"]
        .nunique()
        .rename("active_pairs_sim")
    )


def reconcile_monthly(fund: pd.DataFrame, sim_monthly: pd.DataFrame) -> pd.DataFrame:
    """Side-by-side monthly comparison; diff_* = sim − fund (same economic line)."""
    f = fund.copy()
    f["month_period"] = f["month"].dt.to_period("M")

    s = sim_monthly.copy()
    s["month_period"] = pd.to_datetime(s["date"]).dt.to_period("M")
    s = s.rename(
        columns={
            "long_notional": "long_notional_sim",
            "short_notional": "short_notional_sim",
            "long_gross": "gross_long_sim",
            "short_gross": "gross_short_sim",
            "txn": "t_costs_sim",
            "borrow": "borrow_sim",
            "margin_net": "margin_sim",
            "pre_fee_pnl": "pre_fee_pnl_sim",
            "nav_change": "nav_change_sim",
        }
    )

    fund_cols = [
        "month_period",
        "benchmark_rate",
        "long_notional",
        "short_notional",
        "gross_long",
        "gross_short",
        "t_costs",
        "borrow",
        "margin",
        "pre_fee_pnl",
        "mgmt_fee",
        "incentive_fee",
        "total_fees",
        "net_pnl",
    ]
    sim_cols = [
        "month_period",
        "long_notional_sim",
        "short_notional_sim",
        "gross_long_sim",
        "gross_short_sim",
        "t_costs_sim",
        "borrow_sim",
        "margin_sim",
        "pre_fee_pnl_sim",
        "nav_change_sim",
    ]
    out = f[[c for c in fund_cols if c in f.columns]].merge(
        s[[c for c in sim_cols if c in s.columns]],
        on="month_period",
        how="outer",
    )

    if "gross_long_sim" in out.columns and "gross_long" in out.columns:
        out["diff_gross_long"] = out["gross_long_sim"] - out["gross_long"]
    if "gross_short_sim" in out.columns and "gross_short" in out.columns:
        out["diff_gross_short"] = out["gross_short_sim"] - out["gross_short"]
    if "t_costs_sim" in out.columns and "t_costs" in out.columns:
        out["diff_t_costs"] = out["t_costs_sim"] - out["t_costs"]
    if "borrow_sim" in out.columns and "borrow" in out.columns:
        out["diff_borrow"] = out["borrow_sim"] - out["borrow"]
    if "margin_sim" in out.columns and "margin" in out.columns:
        out["diff_margin"] = out["margin_sim"] - out["margin"]
    if "pre_fee_pnl_sim" in out.columns and "pre_fee_pnl" in out.columns:
        out["diff_pre_fee"] = out["pre_fee_pnl_sim"] - out["pre_fee_pnl"]
    if "nav_change_sim" in out.columns and "net_pnl" in out.columns:
        out["diff_net_nav_vs_fund_net"] = out["nav_change_sim"] - out["net_pnl"]

    # Fund workbook internal bridge (should be ~0 when columns parse correctly):
    # pre_fee = gross_long + gross_short − T-costs − borrow − margin
    _need = ("gross_long", "gross_short", "t_costs", "borrow", "margin")
    if all(c in out.columns for c in _need) and "pre_fee_pnl" in out.columns:
        out["fund_pre_fee_from_components"] = (
            pd.to_numeric(out["gross_long"], errors="coerce")
            + pd.to_numeric(out["gross_short"], errors="coerce")
            - pd.to_numeric(out["t_costs"], errors="coerce")
            - pd.to_numeric(out["borrow"], errors="coerce")
            - pd.to_numeric(out["margin"], errors="coerce")
        )
        out["fund_pre_fee_vs_components"] = pd.to_numeric(out["pre_fee_pnl"], errors="coerce") - out[
            "fund_pre_fee_from_components"
        ]

    return out.sort_values("month_period")


def reconcile_daily(fund_daily: pd.DataFrame, sim_daily: pd.DataFrame) -> pd.DataFrame:
    fund_daily = fund_daily.copy()
    fund_daily["date"] = pd.to_datetime(fund_daily["date"]).dt.normalize()
    sim_daily = sim_daily.copy()
    sim_daily["date"] = pd.to_datetime(sim_daily["date"]).dt.normalize()
    m = fund_daily.merge(sim_daily, on="date", how="outer", suffixes=("_fund", "_sim"))
    fund_pnl = "pnl_usd_fund" if "pnl_usd_fund" in m.columns else "pnl_usd"
    if fund_pnl in m.columns and "nav_change_sim" in m.columns:
        m["diff_pnl"] = m["nav_change_sim"] - m[fund_pnl]
    return m.sort_values("date")


def largest_mismatch_summary(monthly_recon: pd.DataFrame, top_n: int = 5) -> dict[str, Any]:
    rows: list[tuple[str, float, Any]] = []
    mp_col = "month_period" if "month_period" in monthly_recon.columns else monthly_recon.columns[0]
    for col in monthly_recon.columns:
        if not col.startswith("diff_"):
            continue
        s = pd.to_numeric(monthly_recon[col], errors="coerce").abs()
        if s.empty or not s.notna().any():
            continue
        j = int(s.idxmax())
        rows.append((col, float(s.iloc[j]), monthly_recon.loc[j, mp_col]))
    rows.sort(key=lambda x: -x[1])
    return {
        "top_abs_diffs": rows[:top_n],
        "notes": [
            "If fund_pre_fee_vs_components is not ~0, the workbook columns may not match _FUND_MONTHLY_COLS (template drift).",
            "Borrow: model vs live locate/broker marks often diverge.",
            "Fees: fund mgmt/incentive crystallization (quarterly/HWM) vs sim NAV path.",
            "Timing: month-end vs rebalance cut; T+N settlement not in daily sim.",
            "Rounding: notionals and per-leg prints.",
        ],
    }


def build_sim_daily_for_compare(
    bt: pd.DataFrame,
    pair_daily: pd.DataFrame | None = None,
    *,
    attribution_base_capital: float | None = None,
) -> pd.DataFrame:
    """Daily table aligned to fund Daily PnL (date, nav_change vs fund pnl_usd)."""
    daily_bt, _ = aggregate_sim_from_bt(bt, attribution_base_capital=attribution_base_capital)
    d = daily_bt.reset_index()
    d["date"] = pd.to_datetime(d["date"]).dt.normalize()
    d["nav_change_sim"] = d["nav_change"]
    if pair_daily is not None and not pair_daily.empty:
        ap = count_active_pairs_daily(pair_daily)
        d = d.merge(ap.reset_index(), on="date", how="left")
    return d


# --- January 2023: sim ledger (book-level t-costs + margin) vs fund workbook + internal checks ---

JAN_2023_START = pd.Timestamp("2023-01-01")
JAN_2023_END = pd.Timestamp("2023-01-31")
JAN_2023_PERIOD = pd.Period("2023-01", "M")


def sim_recon_window(
    recon_daily: pd.DataFrame,
    start: str | pd.Timestamp = "2023-01-01",
    end: str | pd.Timestamp = "2023-01-31",
) -> pd.DataFrame:
    o = recon_daily.copy()
    o["date"] = pd.to_datetime(o["date"], errors="coerce").dt.normalize()
    t0, t1 = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    return o[(o["date"] >= t0) & (o["date"] <= t1)]


def fund_excel_tieout(
    fund_daily: pd.DataFrame,
    fund_monthly: pd.DataFrame,
    *,
    year_month: str = "2023-01",
) -> dict[str, float | bool | None]:
    """
    For one calendar month, sum fund **Daily PnL** and compare to the same month row on
    **Monthly Attribution** (pre-fee, net, and component bridge if present).
    """
    p = pd.Period(year_month, "M")
    if fund_monthly.empty or "month_period" not in fund_monthly.columns:
        return {"ok": False, "error": "fund_monthly missing month_period", "n_rows": 0.0}
    m = fund_monthly.loc[fund_monthly["month_period"] == p]
    if m.empty or len(m) > 1:
        return {
            "ok": False,
            "error": f"no single monthly row for {p}",
            "n_rows": float(len(m)),
        }
    row = m.iloc[0]
    d0, d1 = p.start_time, p.end_time
    fd = fund_daily.copy()
    fd["date"] = pd.to_datetime(fd["date"], errors="coerce")
    w = (fd["date"] >= d0) & (fd["date"] <= d1)
    s_daily = float(pd.to_numeric(fd.loc[w, "pnl_usd"], errors="coerce").fillna(0.0).sum())
    pre = float(row["pre_fee_pnl"]) if "pre_fee_pnl" in row and pd.notna(row.get("pre_fee_pnl")) else np.nan
    net = float(row["net_pnl"]) if "net_pnl" in row and pd.notna(row.get("net_pnl")) else np.nan
    mfee = float(row["total_fees"]) if "total_fees" in row and pd.notna(row.get("total_fees", np.nan)) else np.nan
    return {
        "ok": True,
        "sum_fund_daily_pnl_in_month": s_daily,
        "monthly_pre_fee_pnl": pre,
        "monthly_net_pnl": net,
        "sum_daily_minus_m_pre": s_daily - pre if not np.isnan(pre) else np.nan,
        "sum_daily_minus_m_net": s_daily - net if not np.isnan(net) else np.nan,
        "monthly_fees": mfee,
        "pass_daily_sums_to_pre_fees": (abs(s_daily - pre) < 1.0) if not np.isnan(pre) else False,
        "pass_daily_sums_to_net": (abs(s_daily - net) < 1.0) if not np.isnan(net) else False,
    }


def fund_and_sim_tieout(
    recon_jan: pd.DataFrame,
    fund_daily_jan: pd.DataFrame,
) -> dict[str, Any]:
    """
    **Sim** = ``attribution_daily_net`` (sum pair ex-txn, minus **one** book T-cost; margin in book
    and pair financing is consistent with export). **Fund** = same-calendar-day daily P&L. Compare
    month sums: gap = model vs live.
    """
    if recon_jan.empty or fund_daily_jan.empty:
        return {"ok": False, "reason": "empty"}
    s_sim = float(recon_jan["attribution_daily_net_usd"].sum()) if "attribution_daily_net_usd" in recon_jan else np.nan
    s_nav = float(recon_jan["daily_nav_change_usd"].sum()) if "daily_nav_change_usd" in recon_jan else np.nan
    s_fund = float(
        pd.to_numeric(fund_daily_jan.get("pnl_usd", pd.Series(dtype=float)), errors="coerce")
        .fillna(0.0)
        .sum()
    )
    return {
        "sum_sim_attribution_ex_txn_minus_book_txn": s_sim,
        "sum_sim_nav_change": s_nav,
        "sum_fund_daily_pnl": s_fund,
        "diff_fund_sum_minus_sim_attribution": s_fund - s_sim,
    }


def filter_fund_to_jan_2023(
    fund_m: pd.DataFrame, fund_d: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Subsets fund monthly to 2023-01 and fund daily to calendar Jan 2023 (inclusive)."""
    m = fund_m.copy()
    if "month_period" in m.columns:
        m = m[m["month_period"] == JAN_2023_PERIOD]
    else:
        m = m[pd.to_datetime(m["month"], errors="coerce").dt.to_period("M") == JAN_2023_PERIOD]
    d = fund_d.copy()
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    d = d[(d["date"] >= JAN_2023_START) & (d["date"] <= JAN_2023_END)]
    return m.reset_index(drop=True), d.reset_index(drop=True)


def self_check_sim_reconciliation_daily(
    recon: pd.DataFrame,
) -> dict[str, Any]:
    """
    On ``reconciliation_daily`` from the ledger: confirm **ex-txn** sum minus **book t-costs** matches
    book and NAV (T-costs and margin are **not** in pair sums; margin is in pair financing until book
    overwrites the comparison via ``book_*`` series).

    - ``attribution_daily_net_usd`` = sum(pairs) ex-txn – ``book_daily_txn_cost_usd`` (one book t-costs line).
    """
    need = [
        "date",
        "attribution_daily_net_usd",
        "book_daily_net_pnl_from_components_usd",
        "daily_nav_change_usd",
        "diff_attribution_net_vs_book_components_usd",
        "diff_attribution_net_vs_nav_usd",
    ]
    miss = [c for c in need if c not in recon.columns]
    if miss:
        raise ValueError(f"reconciliation_daily missing columns: {miss}")

    return {
        "max_abs_attribution_vs_book": float(
            pd.to_numeric(recon["diff_attribution_net_vs_book_components_usd"], errors="coerce")
            .abs()
            .max()
        ),
        "max_abs_attribution_vs_nav": float(
            pd.to_numeric(recon["diff_attribution_net_vs_nav_usd"], errors="coerce").abs().max()
        ),
        "sum_attribution": float(
            pd.to_numeric(recon["attribution_daily_net_usd"], errors="coerce").sum()
        ),
        "sum_daily_nav": float(
            pd.to_numeric(recon["daily_nav_change_usd"], errors="coerce").sum()
        ),
        "sum_book_net_from_components": float(
            pd.to_numeric(recon["book_daily_net_pnl_from_components_usd"], errors="coerce").sum()
        ),
    }


def check_fund_daily_rolls_to_monthly_pre_fee(
    fund_m_jan: pd.DataFrame, fund_d_jan: pd.DataFrame
) -> dict[str, float]:
    """``sum`` of fund daily ``pnl_usd`` should match monthly ``pre_fee_pnl`` for 2023-01 (if daily is pre-fee)."""
    s = float(pd.to_numeric(fund_d_jan["pnl_usd"], errors="coerce").sum())
    if fund_m_jan.empty or "pre_fee_pnl" not in fund_m_jan.columns:
        return {
            "sum_fund_daily_pnl_jan": s,
            "monthly_pre_fee_jan": float("nan"),
            "diff_rollup_fund": float("nan"),
        }
    p = float(pd.to_numeric(fund_m_jan["pre_fee_pnl"], errors="coerce").sum())
    return {
        "sum_fund_daily_pnl_jan": s,
        "monthly_pre_fee_jan": p,
        "diff_rollup_fund": s - p,
    }
