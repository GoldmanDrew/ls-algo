"""One-off: extract apply_lp_fees + _normalize from v15 nb into scripts/lp_fees_v15.py body."""
import json
import pathlib

root = pathlib.Path(__file__).resolve().parent.parent
nbp = root / "notebooks" / "Diamond_Creek_Backtest_v15.ipynb"
text = nbp.read_text(encoding="utf-8")
nb = json.loads(text)
out = []
for c in nb["cells"]:
    src = "".join(c.get("source", []))
    if "def apply_lp_fees_quarterly" in src and "def _normalize_index_to_date" in src:
        i0 = src.find("def _normalize_index_to_date")
        i1 = src.find("if \"ALL_BT\" not in globals()")
        if i1 < 0:
            i1 = len(src)
        out.append(src[i0:i1].rstrip())
        break
if not out:
    raise SystemExit("cell not found")
p = root / "scripts" / "lp_fees_v15.py"
p.write_text(
    '"""V15 LP 2/20 style fees: quarterly management, annual performance (from Diamond_Creek_Backtest_v15)."""\n\n'
    "from __future__ import annotations\n\n"
    "import numpy as np\nimport pandas as pd\n\n"
    + out[0]
    + "\n\n\ndef build_lp_fee_daily_cashflow_usd(\n"
    + """    nav_series: pd.Series,
    index_dates: pd.DatetimeIndex | list,
    *,
    attribution_base_capital: float,
    management_fee_annual: float,
    incentive_fee: float,
) -> pd.DataFrame:
    \"\"\"
    One row per ``index_dates`` (trading days): ``mgmt_usd`` and ``perf_usd`` cash
    (zeros except on fee dates), consistent with :func:`apply_lp_fees_quarterly` in
    *relative* NAV space, amounts scaled to **dollars** via ``attribution_base_capital``.

    - **Management**: (mgmt_annual/4) × **prior** close NAV (from ``nav_series``) on
      the first day of each quarter in ``index_dates`` (v15 pass 1 logic).
    - **Performance**: v15 pass 2 allocated perf at quarter-ends, USD = ``PerfFee`` × base
      when the notebook sim would start at 1.0 (same as ``apply_lp_fees_quarterly`` diag
      scaled for dollar tracking).
    \"\"\"
    idx = pd.DatetimeIndex(pd.to_datetime(index_dates, errors="coerce")).sort_values()
    nav = (
        pd.to_numeric(nav_series.reindex(idx, method="ffill"), errors="coerce")
        .ffill()
        .bfill()
    )
    if len(nav) == 0 or nav.isna().all():
        return pd.DataFrame(
            {"mgmt_usd": [], "perf_usd": []}, index=pd.DatetimeIndex([], dtype="datetime64[ns]")
        )
    g = nav.pct_change()
    g.iloc[0] = (float(nav.iloc[0]) - float(attribution_base_capital)) / float(attribution_base_capital)
    g = g.replace([np.inf, -np.inf], np.nan).dropna()
    g = _normalize_index_to_date(g)
    MGMT_FEE_Q = float(management_fee_annual) / 4.0
    _net, fee_diag = apply_lp_fees_quarterly(
        g,
        mgmt_fee_q=MGMT_FEE_Q,
        incentive_fee=float(incentive_fee),
        crystallize_freq="Q",
    )
    outi = idx.normalize()
    mg = pd.Series(0.0, index=outi, dtype=float)
    pr = pd.Series(0.0, index=outi, dtype=float)
    base = float(attribution_base_capital)
    q_starts = g.index.to_period("Q").to_timestamp(how="start")
    uq, inv = np.unique(
        g.index.to_period("Q"), return_inverse=True
    )
    first_in_q: dict[pd._libs.period.Period, pd.Timestamp] = {}
    for t in g.index:
        per = t.to_period("Q")
        if per not in first_in_q or t < first_in_q[per]:
            first_in_q[per] = t
    nprev = float(base)
    for t in g.index:
        if t in first_in_q.values() and first_in_q.get(t.to_period("Q"), t) == t:
            nav_p = nprev
            m_amt = MGMT_FEE_Q * nav_p
            if t in mg.index:
                mg.loc[t] = m_amt
        nprev = float(nav.loc[t]) if t in nav.index and pd.notna(nav.loc[t]) else nprev
    for _, r in (fee_diag if not fee_diag.empty else []).iterrows() if not fee_diag.empty:
        p_rel = float(r.get("PerfFee_amt", 0.0) or 0.0)
        qe = pd.Timestamp(r.get("QuarterEnd", pd.NaT))
        if not pd.isna(qe) and p_rel and qe in pr.index:
            pr.loc[qe] = pr.get(qe, 0.0) + p_rel * base
    mgrid = mg.reindex(outi, fill_value=0.0)
    prd = pr.reindex(outi, fill_value=0.0)
    for i, t in enumerate(outi):
        if t in g.index and t in first_in_q.values():
            nvp = float(nav.shift(1).loc[t]) if t in nav.index else float(
                nav.iloc[max(0, i - 1)]
            ) if i else base
            if t == first_in_q.get(t.to_period("Q"), None):
                mgrid.loc[t] = MGMT_FEE_Q * (nvp if np.isfinite(nvp) else base)
    mgrid, prd = mgrid.reindex(outi, fill_value=0.0), prd.reindex(outi, fill_value=0.0)
    return pd.DataFrame({"mgmt_usd": mgrid.values, "perf_usd": prd.values}, index=outi)
""",
    encoding="utf-8",
)
print("Wrote", p, "chars", p.stat().st_size)
