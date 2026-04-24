"""V15 LP 2/20 style fees: quarterly management, annual performance (from Diamond_Creek_Backtest_v15)."""

from __future__ import annotations

import numpy as np
import pandas as pd

def _normalize_index_to_date(s: pd.Series) -> pd.Series:
    s = s.copy()
    ix = pd.DatetimeIndex(pd.to_datetime(s.index, errors="coerce"))
    if ix.tz is not None:
        ix = ix.tz_convert("UTC").tz_localize(None)
    s.index = ix.normalize()
    return s.sort_index()


# ============================================================
# LP fee wrapper (quarterly mgmt, annual promote, quarterly allocation)
# ============================================================
def apply_lp_fees_quarterly(
    gross_daily_ret: pd.Series,
    mgmt_fee_q: float = 0.005,
    incentive_fee: float = 0.20,
    crystallize_freq: str = "Q",
) -> tuple[pd.Series, pd.DataFrame]:
    r = gross_daily_ret.replace([np.inf, -np.inf], np.nan).dropna().copy()
    if r.empty:
        return r, pd.DataFrame()

    r = _normalize_index_to_date(r)
    idx = pd.DatetimeIndex(r.index)

    nav = 1.0
    nav_series = pd.Series(index=idx, dtype=float)

    periods = idx.to_period(crystallize_freq)
    q_starts = (
        pd.Series(idx, index=idx)
        .groupby(periods)
        .min()
        .sort_values()
        .to_list()
    )
    q_starts = [pd.Timestamp(x) for x in q_starts]
    q_start_set = set(q_starts)

    q_ends = (
        pd.Series(idx, index=idx)
        .groupby(periods)
        .max()
        .sort_values()
        .to_list()
    )
    q_ends = [pd.Timestamp(x) for x in q_ends]
    q_end_set = set(q_ends)

    fee_rows = []
    q_start_nav = nav
    q_mgmt_fee_amt = 0.0

    # Pass 1: apply quarterly management fee at BEGINNING of each quarter,
    # then apply daily gross returns.
    for t in idx:
        if t in q_start_set:
            q_start_nav = nav
            q_mgmt_fee_amt = mgmt_fee_q * q_start_nav
            nav = q_start_nav - q_mgmt_fee_amt
        nav *= (1.0 + float(r.loc[t]))
        nav_series.loc[t] = nav

        if t in q_end_set:
            nav_pre_fees = nav
            mgmt_fee_amt = q_mgmt_fee_amt
            nav_after_mgmt = nav_pre_fees

            fee_rows.append(
                {
                    "QuarterEnd": t,
                    "StartNAV": q_start_nav,
                    "EndNAV_preFees": nav_pre_fees,
                    "MgmtFee_amt": mgmt_fee_amt,
                    "EndNAV_postMgmt": nav_after_mgmt,
                    "HurdleNAV": np.nan,
                    "PerfFee_amt": 0.0,
                    "EndNAV_postFees": nav_after_mgmt,
                    "HWM_postFees": np.nan,
                    "QuarterGrossRet": (nav_pre_fees / q_start_nav) - 1.0,
                    "QuarterNetRet": (nav_after_mgmt / q_start_nav) - 1.0,
                }
            )
            q_start_nav = nav_after_mgmt

    # Pass 2: annual incentive fee test (NO hurdle):
    # compute promote after full-year management fees, then allocate back to quarter-ends
    # using each quarter's net-of-mgmt contribution.
    perf_fee_by_q = {q: 0.0 for q in q_ends}
    hwm = 1.0
    years = sorted(idx.year.unique())

    for y in years:
        y_mask = idx.year == y
        y_idx = idx[y_mask]
        if len(y_idx) < 2:
            continue

        y_start = y_idx[0]
        y_end = y_idx[-1]
        y_start_nav = float(nav_series.loc[y_start])
        y_end_nav_pre_perf = float(nav_series.loc[y_end])
        is_full_year = (pd.Timestamp(y_end).month == 12)
        # Do not crystallize annual promote for an incomplete trailing year (e.g., only Q1 YTD).
        if (y == years[-1]) and (not is_full_year):
            continue

        gate = hwm
        incentive_base = max(y_end_nav_pre_perf - gate, 0.0)
        total_perf_fee = float(incentive_fee) * incentive_base

        y_q_ends = [q for q in q_ends if (q.year == y and q in nav_series.index)]
        if total_perf_fee > 0 and y_q_ends:
            # Allocate by quarter contribution after management fee:
            # (EndNAV_postMgmt - StartNAV), clipped at 0 to avoid negative weights.
            y_fee_rows = [r for r in fee_rows if pd.Timestamp(r.get("QuarterEnd")).year == y]
            pnl_map = {}
            for r in y_fee_rows:
                q = pd.Timestamp(r.get("QuarterEnd"))
                q_pnl_after_mgmt = float(r.get("EndNAV_postMgmt", np.nan)) - float(r.get("StartNAV", np.nan))
                pnl_map[q] = max(0.0, q_pnl_after_mgmt) if np.isfinite(q_pnl_after_mgmt) else 0.0
            w = pd.Series({q: pnl_map.get(q, 0.0) for q in y_q_ends}, dtype=float)
            if float(w.sum()) <= 0:
                w = nav_series.loc[y_q_ends].astype(float).clip(lower=1e-12)
            w = w / w.sum()

            # Base quarterly fee allocation (before calibration).
            base_alloc = {q: float(total_perf_fee * w.loc[q]) for q in y_q_ends}
            target_end_nav = float(y_end_nav_pre_perf - total_perf_fee)

            def _apply_alloc(alpha: float):
                nav_tmp = nav_series.copy()
                applied = {q: 0.0 for q in y_q_ends}
                for q in y_q_ends:
                    q_nav = float(nav_tmp.loc[q])
                    fee_amt = max(0.0, float(alpha * base_alloc[q]))
                    fee_amt = min(fee_amt, max(0.0, q_nav - 1e-12))
                    applied[q] = fee_amt
                    scale = max(0.0, 1.0 - fee_amt / max(q_nav, 1e-12))
                    nav_tmp.loc[q:] = nav_tmp.loc[q:] * scale
                return float(nav_tmp.loc[y_end]), applied, nav_tmp

            # Calibrate alpha so quarterly deductions are equivalent to year-end 20% take.
            lo, hi = 0.0, 1.0
            end_hi, _, _ = _apply_alloc(hi)
            while end_hi > target_end_nav and hi < 10.0:
                hi *= 1.5
                end_hi, _, _ = _apply_alloc(hi)

            for _ in range(50):
                mid = 0.5 * (lo + hi)
                end_mid, _, _ = _apply_alloc(mid)
                if end_mid > target_end_nav:
                    lo = mid
                else:
                    hi = mid

            _, applied_map, nav_cal = _apply_alloc(hi)
            nav_series = nav_cal
            for q, fee_amt in applied_map.items():
                perf_fee_by_q[q] += float(fee_amt)

        hwm = max(hwm, float(nav_series.loc[y_end]))

    # Build fee diagnostics with annual HWM-only gate + allocated perf fees.
    fee_diag = pd.DataFrame(fee_rows)
    if not fee_diag.empty:
        fee_diag["QuarterEnd"] = pd.to_datetime(fee_diag["QuarterEnd"])
        fee_diag["PerfFee_amt"] = fee_diag["QuarterEnd"].map(perf_fee_by_q).fillna(0.0)
        fee_diag["EndNAV_postFees"] = fee_diag["QuarterEnd"].map(nav_series).astype(float)

        # Hurdle disabled in this simplified fee model.
        y_start_nav_map = {}
        for y in years:
            y_idx = idx[idx.year == y]
            if len(y_idx) > 0:
                y_start_nav_map[y] = float(nav_series.loc[y_idx[0]])
        fee_diag["HurdleNAV"] = fee_diag["QuarterEnd"].dt.year.map(
            lambda y: np.nan
        )

        fee_diag = fee_diag.sort_values("QuarterEnd").reset_index(drop=True)
        fee_diag["HWM_postFees"] = fee_diag["EndNAV_postFees"].cummax()
        fee_diag["QuarterNetRet"] = (fee_diag["EndNAV_postFees"] / fee_diag["StartNAV"]) - 1.0

    nav_series = nav_series.dropna()
    # Keep first available trading day with a 0% return so reported start is earliest feasible date.
    net_daily_ret = nav_series.pct_change().fillna(0.0)
    net_daily_ret.name = "LP_NetRet"

    return net_daily_ret, fee_diag


def build_lp_fee_daily_cashflow_usd(
    nav_series: pd.Series,
    index_dates: pd.DatetimeIndex | list,
    *,
    attribution_base_capital: float,
    management_fee_annual: float,
    incentive_fee: float,
) -> pd.DataFrame:
    """
    **Management fee (USD cashflow, one per calendar quarter)**

    * **Amount:** ``(management_fee_annual/4) × N``, where *N* is the NAV at the
      **end of the day before** the **first** trading day of the quarter
      (``nprev`` at the quarter’s first index date — same as start-of-quarter
      opening NAV for a continuous series).
    * **Posting date:** the **only** day in the series with a positive
      ``mgmt_usd`` for that quarter: the **last** trading day in the index
      that belongs to the quarter. For standard calendar *Q* periods that is
      always a date in **March, June, September, or December** (the final month
      of the quarter), e.g. last *business* day in March for Q1.

    **Performance fee:** ``perf_usd`` on quarter-end dates; dollar scale uses
    ``attribution_base_capital`` × pass-2 ``PerfFee_amt`` from
    :func:`apply_lp_fees_quarterly` (unchanged).
    """
    base = float(attribution_base_capital)
    idx = pd.DatetimeIndex(pd.to_datetime(index_dates, errors="coerce")).sort_values()
    idx = idx[~idx.isna()]
    if len(idx) == 0:
        return pd.DataFrame(
            {"mgmt_usd": [], "perf_usd": []},
            index=pd.DatetimeIndex([], dtype="datetime64[ns]"),
        )
    nav = (
        pd.to_numeric(nav_series, errors="coerce")
        .reindex(idx, method="ffill")
        .ffill()
        .bfill()
    )
    g = nav.pct_change()
    g.iloc[0] = (float(nav.iloc[0]) - base) / base
    g = g.replace([np.inf, -np.inf], np.nan)
    g = _normalize_index_to_date(g)
    g = g.dropna()
    if g.empty:
        return pd.DataFrame(0.0, index=idx, columns=["mgmt_usd", "perf_usd"])
    mgq = float(management_fee_annual) / 4.0
    _net, fee_diag = apply_lp_fees_quarterly(
        g, mgmt_fee_q=mgq, incentive_fee=float(incentive_fee), crystallize_freq="Q"
    )
    mg = pd.Series(0.0, index=idx, dtype=float)
    pr = pd.Series(0.0, index=idx, dtype=float)
    nprev = nav.shift(1).fillna(base)
    byq = g.index.to_series().groupby(g.index.to_period("Q"))
    dfirst = byq.min()
    dlast = byq.max()
    for p, t0 in dfirst.items():
        t0 = pd.Timestamp(t0)
        t1 = pd.Timestamp(dlast[p])
        post = t1
        if post in mg.index:
            nv0 = float(nprev.loc[t0]) if t0 in nprev.index else base
            mg.loc[post] = mgq * (nv0 if np.isfinite(nv0) else base)
    if not fee_diag.empty:
        for _, r in fee_diag.iterrows():
            p_rel = float(r.get("PerfFee_amt", 0.0) or 0.0)
            qe = pd.to_datetime(r.get("QuarterEnd", pd.NaT), errors="coerce")
            if not pd.isna(qe) and p_rel and qe.normalize() in pr.index:
                pr.loc[qe.normalize()] = p_rel * base
    out = pd.DataFrame(
        {
            "mgmt_usd": mg.reindex(idx, fill_value=0.0).values,
            "perf_usd": pr.reindex(idx, fill_value=0.0).values,
        },
        index=idx,
    )
    return out.fillna(0.0)
