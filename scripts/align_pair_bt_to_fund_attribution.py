"""
Clip / reindex sim outputs to the fund *Daily PnL* calendar.

Full return-matching rescaling is strategy-specific; exports only need a shared
calendar for fair **daily** comparison with ``DC ETF Arb Attribution.xlsx``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.compare_dc_etf_attribution import load_fund_daily


def align_pair_bt_to_fund_attribution(
    bt: pd.DataFrame,
    pair_daily: pd.DataFrame,
    fund_attribution_xlsx: Path | str,
    *,
    clip_fund_trading_calendar: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Returns ``(bt_out, pair_daily_out, report)``.

    When ``clip_fund_trading_calendar`` is True, keep only dates present in the
    fund *Daily PnL* sheet (normalized to midnight UTC-naive).
    """
    bt_o = bt.sort_index().copy()
    bt_o.index = pd.to_datetime(bt_o.index).normalize()
    pd_o = pair_daily.copy()
    if not pd_o.empty:
        pd_o["date"] = pd.to_datetime(pd_o["date"]).dt.normalize()

    report: dict = {"n_bt_before": len(bt_o), "n_pair_before": len(pd_o)}
    if clip_fund_trading_calendar:
        fund = load_fund_daily(fund_attribution_xlsx)
        cal = set(fund["date"].dt.normalize())
        bt_o = bt_o.loc[bt_o.index.isin(cal)]
        if not pd_o.empty:
            pd_o = pd_o.loc[pd_o["date"].isin(cal)]
    report.update({"n_bt_after": len(bt_o), "n_pair_after": len(pd_o)})
    return bt_o, pd_o, report
