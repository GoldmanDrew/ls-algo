"""
Bucket 5 Insurance Backtest Lab — interactive parameter exploration.

Run::

    streamlit run scripts/bucket5_lab.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.bucket5_backtest_api import (  # noqa: E402
    config_from_preset,
    list_presets,
    run_backtest,
    run_compare,
    _dc_to_dict,
)
from scripts.bucket5_insurance_bt import build_ladder, production_config  # noqa: E402

st.set_page_config(
    page_title="Bucket 5 Insurance Lab",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .metric-card { background: #0f172a; padding: 12px 16px; border-radius: 8px; border: 1px solid #1e293b; }
    .stApp { background: #0b1220; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Bucket 5 — UVIX/SVIX Insurance Backtest Lab")
st.caption("Short UVIX + short SVIX carry sleeve · SPX put hedge · adaptive cadence · monetize + redeploy")

if "pinned" not in st.session_state:
    st.session_state.pinned = []

with st.sidebar:
    st.header("Run settings")
    era = st.selectbox("History era", ["live", "extended"], format_func=lambda x: "Live (2022+)" if x == "live" else "Extended (2008+)")
    preset_options = ["custom"] + list(list_presets().keys())
    preset = st.selectbox("Preset", preset_options, index=preset_options.index("F_dynamic_deep30") if "F_dynamic_deep30" in preset_options else 0)

    st.subheader("Sleeve")
    sleeve_frac = st.slider("Sleeve gross fraction", 0.05, 0.40, 0.20, 0.01)
    tbill_rate = st.number_input("T-bill yield (annual)", 0.0, 0.10, 0.043, 0.001, format="%.3f")
    uvix_slip = st.number_input("ETP slippage (bps)", 0.0, 50.0, 5.0, 0.5)
    fee_bps = st.number_input("Commission (bps)", 0.0, 10.0, 1.0, 0.5)

    st.subheader("Borrow & costs")
    borrow_mode = st.selectbox("Borrow mode", ["fixed", "etf_history"])
    borrow_uvix = st.number_input("UVIX borrow %/yr", 0.0, 20.0, 2.84, 0.01)
    borrow_svix = st.number_input("SVIX borrow %/yr", 0.0, 20.0, 3.47, 0.01)
    borrow_stress = st.slider("Borrow stress multiplier", 1.0, 3.0, 1.0, 0.05)

    st.subheader("Hedge ladder")
    total_budget = st.slider("Total premium %/roll", 0.5, 4.0, 3.0 if preset == "F_dynamic_deep30" else 2.4, 0.1) / 100.0
    w_deep = st.slider("Deep rung weight (35% OTM)", 0.10, 0.70, 0.50, 0.05)
    w_mid = st.slider("Mid rung weight (25% OTM)", 0.10, 0.50, 0.35, 0.05)
    w_near = max(0.05, 1.0 - w_deep - w_mid)
    st.caption(f"Near rung (10% OTM) weight: {w_near:.0%}")
    dynamic_budget = st.checkbox("Dynamic hedge budget", value=preset in ("B_production", "F_dynamic_deep30"))

    st.subheader("Cadence")
    base_days = st.number_input("Base rebalance days", 5.0, 30.0, 14.0, 1.0)
    cadence_k = st.number_input("Cadence stress k", 1.0, 15.0, 6.0, 0.5)

    use_cache = st.checkbox("Use result cache", value=True)
    run_btn = st.button("Run backtest", type="primary", use_container_width=True)
    pin_btn = st.button("Pin result for compare", use_container_width=True)

methodology = {
    "borrow_mode": borrow_mode,
    "borrow_uvix": borrow_uvix / 100.0,
    "borrow_svix": borrow_svix / 100.0,
    "borrow_stress_mult": borrow_stress,
}


def _build_params() -> dict:
    if preset != "custom":
        base = _dc_to_dict(config_from_preset(preset))
    else:
        base = _dc_to_dict(production_config())
    rungs = build_ladder([(0.10, w_near), (0.25, w_mid), (0.35, w_deep)], total_budget)
    base["sleeve_frac"] = sleeve_frac
    base["tbill_rate"] = tbill_rate
    base["uvix_slip_bps"] = uvix_slip
    base["fee_bps"] = fee_bps
    base["base_days"] = base_days
    base["cadence_k"] = cadence_k
    base["rungs"] = [{"otm_pct": r.otm_pct, "per_roll_frac": r.per_roll_frac} for r in rungs]
    if dynamic_budget:
        base["hedge_budget"] = {"enabled": True}
    else:
        base["hedge_budget"] = None
    return base


@st.cache_data(show_spinner=False, ttl=3600)
def _cached_run(params_json: str, era_: str, methodology_json: str, use_cache_: bool):
    params = json.loads(params_json)
    methodology_ = json.loads(methodology_json)
    return run_backtest(params=params, era=era_, methodology=methodology_, use_cache=use_cache_)


if run_btn or pin_btn:
    params = _build_params()
    with st.spinner("Running backtest…"):
        result = _cached_run(
            json.dumps(params, sort_keys=True),
            era,
            json.dumps(methodology, sort_keys=True),
            use_cache,
        )
    st.session_state.last_result = result
    st.session_state.last_label = preset if preset != "custom" else "custom"
    if pin_btn:
        st.session_state.pinned.append({"label": st.session_state.last_label, "result": result})

result = st.session_state.get("last_result")

if result is None:
    st.info("Configure parameters in the sidebar and click **Run backtest**.")
    st.markdown(
        """
        **Presets:** A baseline · B production ★ · C deep-skew · F dynamic deep 3%

        **Lab features:** era toggle, borrow modes, ladder weights, dynamic budget, compare pinned runs.

        **Production dashboard:** results also ship to the ls-algo risk dashboard and etf-dashboard via nightly build.
        """
    )
    st.stop()

m = result["metrics"]
meta = result["meta"]
crash = result.get("crash", {})

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("CAGR", f"{m.get('combined_CAGR', 0) * 100:.2f}%")
c2.metric("Vol", f"{m.get('combined_Vol', 0) * 100:.2f}%")
c3.metric("Max DD", f"{m.get('combined_MaxDD', 0) * 100:.2f}%")
c4.metric("Sharpe", f"{m.get('combined_Sharpe', 0):.2f}")
c5.metric("Realized $", f"${m.get('realized_$', 0):,.0f}")
c6.metric("Rebalances", meta.get("rebalances", "—"))

st.caption(
    f"{meta.get('start')} → {meta.get('end')} · {meta.get('n_days')} days · "
    f"pricing={meta.get('pricing_mode')} · borrow UVIX/SVIX "
    f"{meta.get('borrow_uvix_annual', 0) * 100:.2f}% / {meta.get('borrow_svix_annual', 0) * 100:.2f}%"
    + (f" · cached" if result.get("cached") else "")
)

series = result.get("series", {})
if series.get("combined_equity"):
    eq_df = pd.DataFrame(series["combined_equity"], columns=["date", "equity"])
    eq_df["date"] = pd.to_datetime(eq_df["date"])
    eq_df = eq_df.set_index("date")
    dd_df = pd.DataFrame(series.get("drawdown", []), columns=["date", "drawdown"])
    if not dd_df.empty:
        dd_df["date"] = pd.to_datetime(dd_df["date"])
        dd_df = dd_df.set_index("date")

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Combined equity")
        st.line_chart(eq_df["equity"] / 1e6, height=280)
        st.caption("Equity ($M)")
    with col_b:
        st.subheader("Drawdown")
        if not dd_df.empty:
            st.area_chart(dd_df["drawdown"] * 100, height=280)
            st.caption("Drawdown (%)")

    st.subheader("Crash scenario payoffs (fraction of equity)")
    crash_df = pd.DataFrame([{"scenario": k, "payoff": v} for k, v in crash.items()])
    if not crash_df.empty:
        st.bar_chart(crash_df.set_index("scenario")["payoff"], height=200)

if st.session_state.pinned:
    st.subheader("Pinned comparisons")
    rows = []
    for pin in st.session_state.pinned:
        pm = pin["result"]["metrics"]
        rows.append({
            "label": pin["label"],
            "CAGR": pm.get("combined_CAGR"),
            "Vol": pm.get("combined_Vol"),
            "MaxDD": pm.get("combined_MaxDD"),
            "Sharpe": pm.get("combined_Sharpe"),
        })
    st.dataframe(pd.DataFrame(rows).assign(
        CAGR=lambda d: (d["CAGR"] * 100).round(2),
        Vol=lambda d: (d["Vol"] * 100).round(2),
        MaxDD=lambda d: (d["MaxDD"] * 100).round(2),
        Sharpe=lambda d: d["Sharpe"].round(2),
    ), use_container_width=True)
    if st.button("Clear pinned"):
        st.session_state.pinned = []
        st.rerun()

with st.expander("Resolved config JSON"):
    st.json(result.get("config", {}))

with st.expander("Full metrics"):
    st.json(m)
