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

from scripts.bucket5_backtest_api import config_from_preset, list_presets, run_backtest, _dc_to_dict  # noqa: E402
from scripts.bucket5_insurance_bt import build_ladder, production_config  # noqa: E402
from scripts.bucket5_lab_charts import fig_attribution, fig_equity_drawdown  # noqa: E402
from scripts.bucket5_lab_helpers import (  # noqa: E402
    PARAM_SPECS,
    TARGET_METRICS,
    apply_override,
    current_value,
    metric_value,
    pack_lab_preset,
)

st.set_page_config(page_title="Bucket 5 Insurance Lab", page_icon="🛡️", layout="wide", initial_sidebar_state="expanded")

st.markdown(
    """
    <style>
    .stApp { background: #0b1220; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Bucket 5 — UVIX/SVIX Insurance Backtest Lab")
st.caption("Short UVIX + short SVIX carry · SPX put hedge · adaptive cadence · monetize + redeploy")

if "pinned" not in st.session_state:
    st.session_state.pinned = []


def _fmt(value, fmt: str) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "—"
    if fmt == "pct":
        return f"{value * 100:.2f}%"
    if fmt == "usd":
        return f"${value:,.0f}"
    return f"{value:.3f}"


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Run settings")
    era = st.selectbox("History era", ["live", "extended"], format_func=lambda x: "Live (2022+)" if x == "live" else "Extended (2008+)")
    preset_options = ["custom"] + list(list_presets().keys())
    preset = st.selectbox("Preset", preset_options, index=preset_options.index("F_dynamic_deep30") if "F_dynamic_deep30" in preset_options else 0)

    uploaded = st.file_uploader("Load saved preset JSON", type=["json"])
    if uploaded is not None:
        try:
            blob = json.loads(uploaded.read().decode("utf-8"))
            st.session_state.loaded_preset = blob
            st.success(f"Loaded preset: {blob.get('label', 'custom')}")
        except Exception as exc:
            st.error(f"Invalid preset file: {exc}")

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
    hedge_kind = st.selectbox("Hedge kind", ["ladder", "backspread", "hybrid"])
    total_budget = st.slider("Total premium %/roll", 0.5, 4.0, 3.0 if preset == "F_dynamic_deep30" else 2.4, 0.1) / 100.0
    w_deep = st.slider("Deep rung weight (35% OTM)", 0.10, 0.70, 0.50, 0.05)
    w_mid = st.slider("Mid rung weight (25% OTM)", 0.10, 0.50, 0.35, 0.05)
    w_near = max(0.05, 1.0 - w_deep - w_mid)
    st.caption(f"Near rung (10% OTM) weight: {w_near:.0%}")
    dynamic_budget = st.checkbox("Dynamic hedge budget", value=preset in ("B_production", "F_dynamic_deep30"))

    with st.expander("Dynamic budget policy", expanded=False):
        hb_contango_mult = st.slider("Contango premium mult", 0.5, 2.0, 1.20, 0.05)
        hb_stress_mult = st.slider("Stress premium mult", 0.3, 1.5, 0.85, 0.05)
        hb_vix_lo = st.number_input("VIX low anchor", 8.0, 30.0, 14.0, 1.0)
        hb_vix_hi = st.number_input("VIX high anchor", 18.0, 60.0, 28.0, 1.0)
        hb_calm_boost = st.slider("Calm-VIX boost", 1.0, 1.5, 1.10, 0.05)

    with st.expander("Hybrid / backspread", expanded=False):
        hybrid_ladder_frac = st.slider("Hybrid: ladder share of budget", 0.0, 1.0, 0.70, 0.05)
        bs_otm_near = st.slider("Backspread near OTM", 0.05, 0.25, 0.12, 0.01)
        bs_otm_far = st.slider("Backspread far OTM", 0.15, 0.50, 0.30, 0.01)
        bs_far_ratio = st.number_input("Backspread far ratio", 1, 6, 3, 1)
        bs_premium_frac = st.slider("Backspread premium %/roll", 0.005, 0.05, 0.024, 0.001)

    st.subheader("Cadence")
    base_days = st.number_input("Base rebalance days", 5.0, 30.0, 14.0, 1.0)
    cadence_k = st.number_input("Cadence stress k", 1.0, 15.0, 6.0, 0.5)

    with st.expander("Regime policy", expanded=False):
        rho_contango = st.slider("Rho (contango)", 0.5, 2.0, 1.0, 0.05)
        rho_backwardation = st.slider("Rho (backwardation)", 1.0, 3.0, 2.0, 0.05)
        gross_contango = st.slider("Gross (contango)", 0.5, 1.5, 1.0, 0.05)
        gross_backwardation = st.slider("Gross (backwardation)", 0.10, 1.00, 0.35, 0.05)

    with st.expander("Monetize policy", expanded=False):
        st.caption("Profit tiers: sell fraction when put multiple hits level")
        pt1_m, pt1_f = st.columns(2)
        tier1_mult = pt1_m.number_input("Tier 1 multiple", 1.5, 10.0, 3.0, 0.5)
        tier1_frac = pt1_f.slider("Tier 1 sell frac", 0.0, 1.0, 0.34, 0.01)
        pt2_m, pt2_f = st.columns(2)
        tier2_mult = pt2_m.number_input("Tier 2 multiple", 2.0, 12.0, 5.0, 0.5)
        tier2_frac = pt2_f.slider("Tier 2 sell frac", 0.0, 1.0, 0.50, 0.01)
        pt3_m, pt3_f = st.columns(2)
        tier3_mult = pt3_m.number_input("Tier 3 multiple", 4.0, 20.0, 8.0, 0.5)
        tier3_frac = pt3_f.slider("Tier 3 sell frac", 0.0, 1.0, 1.0, 0.01)
        vx1, vx2 = st.columns(2)
        vix1_lvl = vx1.number_input("VIX tier 1", 20.0, 80.0, 45.0, 1.0)
        vix1_frac = vx2.slider("VIX tier 1 sell", 0.0, 1.0, 0.50, 0.05)
        vx3, vx4 = st.columns(2)
        vix2_lvl = vx3.number_input("VIX tier 2", 30.0, 100.0, 65.0, 1.0)
        vix2_frac = vx4.slider("VIX tier 2 sell", 0.0, 1.0, 1.0, 0.05)
        mon_giveback = st.slider("Giveback frac", 0.10, 0.60, 0.35, 0.05)
        mon_giveback_min = st.slider("Giveback min multiple", 1.0, 5.0, 2.0, 0.5)
        mon_bank = st.slider("Bank frac on full exit", 0.20, 1.00, 0.60, 0.05)
        mon_runner = st.slider("Runner frac (patience)", 0.0, 0.50, 0.15, 0.05)
        mon_runner_mult = st.slider("Runner harvest multiple", 4.0, 20.0, 12.0, 1.0)
        mon_rearm = st.checkbox("Re-arm fresh puts on full exit", value=True)

    with st.expander("Redeploy policy", expanded=False):
        rd_contango = st.slider("Sleeve wt (contango)", 0.0, 0.60, 0.20, 0.05)
        rd_backwardation = st.slider("Sleeve wt (backwardation)", 0.20, 1.00, 0.65, 0.05)

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
    base.update({
        "sleeve_frac": sleeve_frac,
        "tbill_rate": tbill_rate,
        "uvix_slip_bps": uvix_slip,
        "fee_bps": fee_bps,
        "base_days": base_days,
        "cadence_k": cadence_k,
        "hedge_kind": hedge_kind,
        "hybrid_ladder_frac": hybrid_ladder_frac,
        "rungs": [{"otm_pct": r.otm_pct, "per_roll_frac": r.per_roll_frac} for r in rungs],
        "regime": {
            "rho_contango": rho_contango, "rho_backwardation": rho_backwardation,
            "gross_contango": gross_contango, "gross_backwardation": gross_backwardation,
            "r_lo": 0.88, "r_hi": 1.00,
        },
        "monetize": {
            "profit_tiers": [[tier1_mult, tier1_frac], [tier2_mult, tier2_frac], [tier3_mult, tier3_frac]],
            "vix_tiers": [[vix1_lvl, vix1_frac], [vix2_lvl, vix2_frac]],
            "giveback_frac": mon_giveback, "giveback_min_mult": mon_giveback_min,
            "bank_frac": mon_bank, "runner_frac": mon_runner, "runner_mult": mon_runner_mult,
            "rearm": mon_rearm,
        },
        "redeploy": {
            "sleeve_w_contango": rd_contango, "sleeve_w_backwardation": rd_backwardation,
            "r_lo": 0.88, "r_hi": 1.00,
        },
        "backspread": {
            "otm_near": bs_otm_near, "otm_far": bs_otm_far,
            "far_ratio": int(bs_far_ratio), "premium_frac": bs_premium_frac,
        },
        "hedge_budget": {
            "enabled": True, "contango_mult": hb_contango_mult, "stress_mult": hb_stress_mult,
            "vix_lo": hb_vix_lo, "vix_hi": hb_vix_hi, "vix_calm_boost": hb_calm_boost,
            "r_lo": 0.88, "r_hi": 1.00,
        } if dynamic_budget else None,
    })
    return base


@st.cache_data(show_spinner=False, ttl=3600)
def _cached_run(params_json: str, era_: str, methodology_json: str, use_cache_: bool):
    return run_backtest(
        params=json.loads(params_json), era=era_,
        methodology=json.loads(methodology_json), use_cache=use_cache_,
    )


@st.cache_data(show_spinner=False, ttl=3600)
def _cached_metric(params_json: str, era_: str, methodology_json: str):
    r = run_backtest(
        params=json.loads(params_json), era=era_, methodology=json.loads(methodology_json),
        include_series=False, use_cache=True,
    )
    return {"metrics": r.get("metrics", {}), "crash": r.get("crash", {})}


def _run_for_metric(params: dict, meth: dict, target_label: str):
    r = _cached_metric(json.dumps(params, sort_keys=True), era, json.dumps(meth, sort_keys=True))
    return metric_value(r, target_label)


if run_btn or pin_btn:
    params = _build_params()
    with st.spinner("Running backtest…"):
        result = _cached_run(json.dumps(params, sort_keys=True), era, json.dumps(methodology, sort_keys=True), use_cache)
    st.session_state.last_result = result
    st.session_state.last_params = params
    st.session_state.last_label = preset if preset != "custom" else "custom"
    if pin_btn:
        st.session_state.pinned.append({"label": st.session_state.last_label, "result": result})

result = st.session_state.get("last_result")

if result is None:
    st.info("Configure parameters in the sidebar and click **Run backtest**.")
    st.markdown(
        "**Lab features:** full knob panel, profit/VIX tier editor, preset save/load, "
        "component attribution + regime shading, sweeps, tornado, pinned compare."
    )
    st.stop()

m, meta, crash = result["metrics"], result["meta"], result.get("crash", {})
params_now = st.session_state.get("last_params") or _build_params()

st.download_button(
    "Download preset JSON",
    data=json.dumps(pack_lab_preset(era=era, preset=preset, params=params_now, methodology=methodology, label=st.session_state.get("last_label", preset)), indent=2),
    file_name=f"bucket5_preset_{st.session_state.get('last_label', 'custom')}.json",
    mime="application/json",
)

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("CAGR", f"{m.get('combined_CAGR', 0) * 100:.2f}%")
c2.metric("Vol", f"{m.get('combined_Vol', 0) * 100:.2f}%")
c3.metric("Max DD", f"{m.get('combined_MaxDD', 0) * 100:.2f}%")
c4.metric("Sharpe", f"{m.get('combined_Sharpe', 0):.2f}")
c5.metric("Realized $", f"${m.get('realized_$', 0):,.0f}")
c6.metric("Rebalances", meta.get("rebalances", "—"))

st.caption(
    f"{meta.get('start')} → {meta.get('end')} · {meta.get('n_days')} days · pricing={meta.get('pricing_mode')} · "
    f"borrow {meta.get('borrow_uvix_annual', 0) * 100:.2f}% / {meta.get('borrow_svix_annual', 0) * 100:.2f}%"
    + (" · cached" if result.get("cached") else "")
)
st.markdown(
    f"> **In plain English:** compounded at **{m.get('combined_CAGR', 0) * 100:.1f}%/yr**, "
    f"worst drawdown **{m.get('combined_MaxDD', 0) * 100:.1f}%**, harvested **${m.get('realized_$', 0):,.0f}** from puts."
)

series = result.get("series", {})
if series.get("combined_equity"):
    st.subheader("Equity & attribution")
    st.pyplot(fig_equity_drawdown(series), clear_figure=True)
    st.pyplot(fig_attribution(series), clear_figure=True)

    crash_df = pd.DataFrame([{"scenario": k, "payoff": v} for k, v in crash.items()])
    if not crash_df.empty:
        st.subheader("Crash scenario payoffs")
        st.bar_chart(crash_df.set_index("scenario")["payoff"], height=200)

st.markdown("---")
st.header("What-if analysis")
base_params = params_now
tab_sweep, tab_tornado = st.tabs(["One-factor sweep", "Tornado sensitivity"])

with tab_sweep:
    sc1, sc2, sc3 = st.columns([2, 1, 1])
    sweep_param = sc1.selectbox("Parameter", list(PARAM_SPECS.keys()), key="sweep_param")
    target_label = sc2.selectbox("Target metric", list(TARGET_METRICS.keys()), key="sweep_target")
    n_steps = sc3.slider("Steps", 3, 15, 7, key="sweep_steps")
    lo_default, hi_default = PARAM_SPECS[sweep_param]["range"]
    r1, r2 = st.columns(2)
    lo = r1.number_input("Range low", value=float(lo_default), format="%.4f", key="sweep_lo")
    hi = r2.number_input("Range high", value=float(hi_default), format="%.4f", key="sweep_hi")
    if st.button("Run sweep", type="primary"):
        values = [lo + (hi - lo) * i / (n_steps - 1) for i in range(n_steps)] if n_steps > 1 else [lo]
        rows = []
        prog = st.progress(0.0)
        for i, v in enumerate(values):
            p, mth = apply_override(base_params, methodology, sweep_param, v)
            rows.append({sweep_param: v, target_label: _run_for_metric(p, mth, target_label)})
            prog.progress((i + 1) / len(values))
        prog.empty()
        st.session_state.sweep_result = {"df": pd.DataFrame(rows).set_index(sweep_param), "param": sweep_param, "target": target_label, "current": current_value(base_params, methodology, sweep_param)}
    sr = st.session_state.get("sweep_result")
    if sr is not None:
        st.line_chart(sr["df"], height=300)
        st.dataframe(sr["df"].assign(**{sr["target"]: sr["df"][sr["target"]].map(lambda x: _fmt(x, TARGET_METRICS[sr["target"]][2]))}))

with tab_tornado:
    tor_params = st.multiselect("Parameters", list(PARAM_SPECS.keys()), default=list(PARAM_SPECS.keys())[:6], key="tor_params")
    tor_target = st.selectbox("Target metric", list(TARGET_METRICS.keys()), key="tor_target")
    perturb = st.slider("Perturbation ±%", 5, 50, 20, 5, key="tor_perturb") / 100.0
    if st.button("Run tornado", type="primary") and tor_params:
        base_metric = _run_for_metric(base_params, methodology, tor_target)
        rows = []
        prog = st.progress(0.0)
        for i, label in enumerate(tor_params):
            cur = current_value(base_params, methodology, label)
            lo_v, hi_v = (PARAM_SPECS[label]["range"] if cur in (None, 0) else (cur * (1 - perturb), cur * (1 + perturb)))
            v_lo = _run_for_metric(*apply_override(base_params, methodology, label, lo_v), tor_target)
            v_hi = _run_for_metric(*apply_override(base_params, methodology, label, hi_v), tor_target)
            rows.append({"parameter": label, "low": (v_lo or 0) - (base_metric or 0), "high": (v_hi or 0) - (base_metric or 0), "swing": abs((v_hi or 0) - (v_lo or 0))})
            prog.progress((i + 1) / len(tor_params))
        prog.empty()
        st.session_state.tornado_result = {"df": pd.DataFrame(rows).sort_values("swing").set_index("parameter"), "target": tor_target, "base": base_metric, "perturb": perturb}
    tr = st.session_state.get("tornado_result")
    if tr is not None:
        st.bar_chart(tr["df"][["low", "high"]], height=340)

if st.session_state.pinned:
    st.markdown("---")
    st.subheader("Pinned comparisons")
    rows = [{"label": f"{p['label']} #{i+1}", **{k: p["result"]["metrics"].get(k) for k in ("combined_CAGR", "combined_Vol", "combined_MaxDD", "combined_Sharpe", "realized_$")}, "Crash-30%": p["result"].get("crash", {}).get("crash_severe_-30%")} for i, p in enumerate(st.session_state.pinned)]
    st.dataframe(pd.DataFrame(rows).set_index("label"), use_container_width=True)
    if st.button("Clear pinned"):
        st.session_state.pinned = []
        st.rerun()

with st.expander("Resolved config JSON"):
    st.json(result.get("config", {}))
