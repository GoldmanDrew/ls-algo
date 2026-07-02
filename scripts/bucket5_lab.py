"""
Bucket 5 Insurance Backtest Lab — interactive parameter exploration.

Run::

    streamlit run scripts/bucket5_lab.py

Three ways to explore how a change helps or hurts the strategy:
  * **Single run**  — set every knob in the sidebar and hit *Run backtest*.
  * **One-factor sweep** — pick one knob, a range, and step count; the lab reruns
    across the range and plots the target metric so you see the response curve.
  * **Tornado sensitivity** — perturb several knobs by +/- a percentage around the
    current run and rank them by how much they move the target metric.

Every run is cached (by resolved config + era + costs), so sweeps and tornados
reuse work and stay fast after the first pass.
"""

from __future__ import annotations

import copy
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

# ---------------------------------------------------------------------------
# Target metrics: friendly label -> (source, key, formatter, higher_is_better)
# ---------------------------------------------------------------------------
TARGET_METRICS = {
    "CAGR": ("metrics", "combined_CAGR", "pct", True),
    "Vol": ("metrics", "combined_Vol", "pct", False),
    "Max drawdown": ("metrics", "combined_MaxDD", "pct", True),  # negative; closer to 0 better
    "Sharpe": ("metrics", "combined_Sharpe", "num", True),
    "Calmar": ("metrics", "combined_Calmar", "num", True),
    "Realized $ (harvested)": ("metrics", "realized_$", "usd", True),
    "Crash payoff -20%": ("crash", "crash_mild_-20%", "pct", True),
    "Crash payoff -30%": ("crash", "crash_severe_-30%", "pct", True),
    "Crash payoff -40%": ("crash", "crash_volmageddon_-40%", "pct", True),
}

# ---------------------------------------------------------------------------
# Sweepable parameters: label -> spec.
#   loc: "params" (into config dict) | "methodology" | "total_premium" (special)
#   path: nested key path into the target dict
#   range: (lo, hi) default slider range for a sweep
#   fmt: display format
# ---------------------------------------------------------------------------
PARAM_SPECS = {
    "Sleeve gross fraction": {"loc": "params", "path": ["sleeve_frac"], "range": (0.05, 0.40), "fmt": "pct"},
    "Total premium %/roll": {"loc": "total_premium", "path": None, "range": (0.010, 0.040), "fmt": "pct"},
    "Cadence base days": {"loc": "params", "path": ["base_days"], "range": (5.0, 30.0), "fmt": "num"},
    "Cadence stress k": {"loc": "params", "path": ["cadence_k"], "range": (1.0, 15.0), "fmt": "num"},
    "T-bill yield": {"loc": "params", "path": ["tbill_rate"], "range": (0.0, 0.08), "fmt": "pct"},
    "Regime gross (backwardation)": {"loc": "params", "path": ["regime", "gross_backwardation"], "range": (0.10, 1.00), "fmt": "pct"},
    "Regime gross (contango)": {"loc": "params", "path": ["regime", "gross_contango"], "range": (0.50, 1.50), "fmt": "num"},
    "Regime rho (backwardation)": {"loc": "params", "path": ["regime", "rho_backwardation"], "range": (1.0, 3.0), "fmt": "num"},
    "Monetize giveback frac": {"loc": "params", "path": ["monetize", "giveback_frac"], "range": (0.10, 0.60), "fmt": "pct"},
    "Monetize runner frac": {"loc": "params", "path": ["monetize", "runner_frac"], "range": (0.0, 0.50), "fmt": "pct"},
    "Monetize bank frac": {"loc": "params", "path": ["monetize", "bank_frac"], "range": (0.20, 1.00), "fmt": "pct"},
    "Redeploy sleeve wt (backwardation)": {"loc": "params", "path": ["redeploy", "sleeve_w_backwardation"], "range": (0.20, 1.00), "fmt": "pct"},
    "Redeploy sleeve wt (contango)": {"loc": "params", "path": ["redeploy", "sleeve_w_contango"], "range": (0.0, 0.60), "fmt": "pct"},
    "Borrow stress multiplier": {"loc": "methodology", "path": ["borrow_stress_mult"], "range": (1.0, 3.0), "fmt": "num"},
}


def _fmt(value, fmt: str) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "—"
    if fmt == "pct":
        return f"{value * 100:.2f}%"
    if fmt == "usd":
        return f"${value:,.0f}"
    return f"{value:.3f}"


def _get_path(d: dict, path: list[str]):
    cur = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def _set_path(d: dict, path: list[str], value) -> None:
    cur = d
    for k in path[:-1]:
        cur = cur.setdefault(k, {})
    cur[path[-1]] = value


def _apply_override(base_params: dict, methodology: dict, label: str, value: float):
    """Return (params, methodology) copies with the given parameter overridden."""
    spec = PARAM_SPECS[label]
    params = copy.deepcopy(base_params)
    meth = copy.deepcopy(methodology)
    if spec["loc"] == "methodology":
        _set_path(meth, spec["path"], float(value))
    elif spec["loc"] == "total_premium":
        rungs = params.get("rungs") or []
        cur_total = sum(r.get("per_roll_frac", 0.0) for r in rungs) or 1e-9
        scale = float(value) / cur_total
        for r in rungs:
            r["per_roll_frac"] = r.get("per_roll_frac", 0.0) * scale
    else:
        _set_path(params, spec["path"], float(value))
    return params, meth


def _current_value(base_params: dict, methodology: dict, label: str):
    spec = PARAM_SPECS[label]
    if spec["loc"] == "methodology":
        return _get_path(methodology, spec["path"])
    if spec["loc"] == "total_premium":
        return sum(r.get("per_roll_frac", 0.0) for r in (base_params.get("rungs") or []))
    return _get_path(base_params, spec["path"])


# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
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
        bs_far_ratio = st.number_input("Backspread far ratio (long per short)", 1, 6, 3, 1)
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
    base["sleeve_frac"] = sleeve_frac
    base["tbill_rate"] = tbill_rate
    base["uvix_slip_bps"] = uvix_slip
    base["fee_bps"] = fee_bps
    base["base_days"] = base_days
    base["cadence_k"] = cadence_k
    base["hedge_kind"] = hedge_kind
    base["hybrid_ladder_frac"] = hybrid_ladder_frac
    base["rungs"] = [{"otm_pct": r.otm_pct, "per_roll_frac": r.per_roll_frac} for r in rungs]
    base["regime"] = {
        "rho_contango": rho_contango,
        "rho_backwardation": rho_backwardation,
        "gross_contango": gross_contango,
        "gross_backwardation": gross_backwardation,
        "r_lo": 0.88,
        "r_hi": 1.00,
    }
    base["monetize"] = {
        "giveback_frac": mon_giveback,
        "giveback_min_mult": mon_giveback_min,
        "bank_frac": mon_bank,
        "runner_frac": mon_runner,
        "runner_mult": mon_runner_mult,
        "rearm": mon_rearm,
    }
    base["redeploy"] = {
        "sleeve_w_contango": rd_contango,
        "sleeve_w_backwardation": rd_backwardation,
        "r_lo": 0.88,
        "r_hi": 1.00,
    }
    base["backspread"] = {
        "otm_near": bs_otm_near,
        "otm_far": bs_otm_far,
        "far_ratio": int(bs_far_ratio),
        "premium_frac": bs_premium_frac,
    }
    if dynamic_budget:
        base["hedge_budget"] = {
            "enabled": True,
            "contango_mult": hb_contango_mult,
            "stress_mult": hb_stress_mult,
            "vix_lo": hb_vix_lo,
            "vix_hi": hb_vix_hi,
            "vix_calm_boost": hb_calm_boost,
            "r_lo": 0.88,
            "r_hi": 1.00,
        }
    else:
        base["hedge_budget"] = None
    return base


@st.cache_data(show_spinner=False, ttl=3600)
def _cached_run(params_json: str, era_: str, methodology_json: str, use_cache_: bool):
    params = json.loads(params_json)
    methodology_ = json.loads(methodology_json)
    return run_backtest(params=params, era=era_, methodology=methodology_, use_cache=use_cache_)


@st.cache_data(show_spinner=False, ttl=3600)
def _cached_metric(params_json: str, era_: str, methodology_json: str):
    """Lightweight run (no series) that returns the metrics + crash dicts only."""
    params = json.loads(params_json)
    methodology_ = json.loads(methodology_json)
    r = run_backtest(
        params=params, era=era_, methodology=methodology_,
        include_series=False, use_cache=True,
    )
    return {"metrics": r.get("metrics", {}), "crash": r.get("crash", {})}


def _metric_value(result: dict, target_label: str):
    src, key, _fmt_, _hib = TARGET_METRICS[target_label]
    return (result.get(src) or {}).get(key)


def _run_for_metric(params: dict, meth: dict, target_label: str):
    r = _cached_metric(json.dumps(params, sort_keys=True), era, json.dumps(meth, sort_keys=True))
    return _metric_value(r, target_label)


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
    st.session_state.last_params = params
    st.session_state.last_label = preset if preset != "custom" else "custom"
    if pin_btn:
        st.session_state.pinned.append({"label": st.session_state.last_label, "result": result})

result = st.session_state.get("last_result")

if result is None:
    st.info("Configure parameters in the sidebar and click **Run backtest**.")
    st.markdown(
        """
        **Presets:** A baseline · B production ★ · C deep-skew · F dynamic deep 3%

        **Lab features:** era toggle, borrow modes, full ladder / regime / monetize / redeploy /
        backspread knobs, dynamic budget, one-factor sweeps, tornado sensitivity, pinned compare.

        **Production dashboard:** results also ship to the ls-algo risk dashboard and etf-dashboard via nightly build.
        """
    )
    st.stop()

m = result["metrics"]
meta = result["meta"]
crash = result.get("crash", {})

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("CAGR", f"{m.get('combined_CAGR', 0) * 100:.2f}%", help="Compound annual growth rate of combined equity.")
c2.metric("Vol", f"{m.get('combined_Vol', 0) * 100:.2f}%", help="Annualized volatility of daily returns.")
c3.metric("Max DD", f"{m.get('combined_MaxDD', 0) * 100:.2f}%", help="Worst peak-to-trough drawdown over the backtest.")
c4.metric("Sharpe", f"{m.get('combined_Sharpe', 0):.2f}", help="Return per unit of volatility (higher is better).")
c5.metric("Realized $", f"${m.get('realized_$', 0):,.0f}", help="Cash harvested from monetizing the put hedge.")
c6.metric("Rebalances", meta.get("rebalances", "—"), help="Number of sleeve rebalances (adaptive cadence).")

st.caption(
    f"{meta.get('start')} → {meta.get('end')} · {meta.get('n_days')} days · "
    f"pricing={meta.get('pricing_mode')} · borrow UVIX/SVIX "
    f"{meta.get('borrow_uvix_annual', 0) * 100:.2f}% / {meta.get('borrow_svix_annual', 0) * 100:.2f}%"
    + (f" · cached" if result.get("cached") else "")
)

# Plain-language takeaway
_cagr = m.get("combined_CAGR", 0) * 100
_dd = m.get("combined_MaxDD", 0) * 100
_real = m.get("realized_$", 0)
st.markdown(
    f"> **In plain English:** this configuration compounded at **{_cagr:.1f}%/yr** with a worst "
    f"drawdown of **{_dd:.1f}%**, and harvested **${_real:,.0f}** from the put hedge during vol spikes."
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

# ---------------------------------------------------------------------------
# What-if tools: one-factor sweep + tornado sensitivity
# ---------------------------------------------------------------------------
st.markdown("---")
st.header("What-if analysis")
st.caption("See how a change would improve or deteriorate the strategy. Both tools reuse the run cache, so repeated points are fast.")

base_params = st.session_state.get("last_params") or _build_params()

tab_sweep, tab_tornado = st.tabs(["One-factor sweep", "Tornado sensitivity"])

with tab_sweep:
    sc1, sc2, sc3 = st.columns([2, 1, 1])
    sweep_param = sc1.selectbox("Parameter to sweep", list(PARAM_SPECS.keys()), key="sweep_param")
    target_label = sc2.selectbox("Target metric", list(TARGET_METRICS.keys()), key="sweep_target")
    n_steps = sc3.slider("Steps", 3, 15, 7, key="sweep_steps")
    spec = PARAM_SPECS[sweep_param]
    lo_default, hi_default = spec["range"]
    r1, r2 = st.columns(2)
    lo = r1.number_input("Range low", value=float(lo_default), format="%.4f", key="sweep_lo")
    hi = r2.number_input("Range high", value=float(hi_default), format="%.4f", key="sweep_hi")
    if st.button("Run sweep", type="primary"):
        values = [lo + (hi - lo) * i / (n_steps - 1) for i in range(n_steps)] if n_steps > 1 else [lo]
        rows = []
        prog = st.progress(0.0)
        for i, v in enumerate(values):
            p, mth = _apply_override(base_params, methodology, sweep_param, v)
            mv = _run_for_metric(p, mth, target_label)
            rows.append({sweep_param: v, target_label: mv})
            prog.progress((i + 1) / len(values))
        prog.empty()
        sweep_df = pd.DataFrame(rows).set_index(sweep_param)
        st.session_state.sweep_result = {
            "df": sweep_df, "param": sweep_param, "target": target_label,
            "current": _current_value(base_params, methodology, sweep_param),
        }
    sr = st.session_state.get("sweep_result")
    if sr is not None:
        st.line_chart(sr["df"], height=300)
        cur = sr["current"]
        if cur is not None:
            st.caption(f"Current setting: **{sr['param']} = {cur:.4f}**. The curve shows {sr['target']} across the swept range.")
        show = sr["df"].copy()
        fmt = TARGET_METRICS[sr["target"]][2]
        show[sr["target"]] = show[sr["target"]].map(lambda x: _fmt(x, fmt))
        st.dataframe(show, use_container_width=True)

with tab_tornado:
    tc1, tc2 = st.columns([2, 1])
    tor_params = tc1.multiselect(
        "Parameters to test",
        list(PARAM_SPECS.keys()),
        default=["Sleeve gross fraction", "Total premium %/roll", "Cadence stress k",
                 "Regime gross (backwardation)", "Monetize giveback frac", "Borrow stress multiplier"],
        key="tor_params",
    )
    tor_target = tc2.selectbox("Target metric", list(TARGET_METRICS.keys()), key="tor_target")
    perturb = st.slider("Perturbation (± % of current value)", 5, 50, 20, 5, key="tor_perturb") / 100.0
    if st.button("Run tornado", type="primary") and tor_params:
        base_metric = _run_for_metric(base_params, methodology, tor_target)
        rows = []
        prog = st.progress(0.0)
        for i, label in enumerate(tor_params):
            cur = _current_value(base_params, methodology, label)
            if cur is None or cur == 0:
                lo_v, hi_v = PARAM_SPECS[label]["range"]
            else:
                lo_v, hi_v = cur * (1 - perturb), cur * (1 + perturb)
            p_lo, m_lo = _apply_override(base_params, methodology, label, lo_v)
            p_hi, m_hi = _apply_override(base_params, methodology, label, hi_v)
            v_lo = _run_for_metric(p_lo, m_lo, tor_target)
            v_hi = _run_for_metric(p_hi, m_hi, tor_target)
            rows.append({
                "parameter": label,
                "low": (v_lo - base_metric) if (v_lo is not None and base_metric is not None) else 0.0,
                "high": (v_hi - base_metric) if (v_hi is not None and base_metric is not None) else 0.0,
                "swing": abs((v_hi or 0) - (v_lo or 0)),
            })
            prog.progress((i + 1) / len(tor_params))
        prog.empty()
        tor_df = pd.DataFrame(rows).sort_values("swing", ascending=True).set_index("parameter")
        st.session_state.tornado_result = {"df": tor_df, "target": tor_target, "base": base_metric, "perturb": perturb}
    tr = st.session_state.get("tornado_result")
    if tr is not None:
        st.caption(
            f"Baseline {tr['target']} = **{_fmt(tr['base'], TARGET_METRICS[tr['target']][2])}**. "
            f"Bars show the change vs baseline when each parameter is moved ±{tr['perturb']*100:.0f}%. "
            f"Longest bars = most sensitive."
        )
        st.bar_chart(tr["df"][["low", "high"]], height=340)
        st.dataframe(tr["df"].round(4), use_container_width=True)

# ---------------------------------------------------------------------------
# Pinned comparisons (deltas vs first pinned baseline + crash payoffs)
# ---------------------------------------------------------------------------
if st.session_state.pinned:
    st.markdown("---")
    st.subheader("Pinned comparisons")
    rows = []
    for idx, pin in enumerate(st.session_state.pinned):
        pm = pin["result"]["metrics"]
        pc = pin["result"].get("crash", {})
        rows.append({
            "label": f"{pin['label']} #{idx+1}",
            "CAGR": pm.get("combined_CAGR"),
            "Vol": pm.get("combined_Vol"),
            "MaxDD": pm.get("combined_MaxDD"),
            "Sharpe": pm.get("combined_Sharpe"),
            "Calmar": pm.get("combined_Calmar"),
            "Realized$": pm.get("realized_$"),
            "Crash-30%": pc.get("crash_severe_-30%"),
        })
    cmp_df = pd.DataFrame(rows).set_index("label")
    disp = cmp_df.assign(
        CAGR=lambda d: (d["CAGR"] * 100).round(2),
        Vol=lambda d: (d["Vol"] * 100).round(2),
        MaxDD=lambda d: (d["MaxDD"] * 100).round(2),
        Sharpe=lambda d: d["Sharpe"].round(2),
        Calmar=lambda d: d["Calmar"].round(2),
        Realized_=lambda d: d["Realized$"].round(0),
    )
    st.dataframe(disp, use_container_width=True)
    if len(cmp_df) > 1:
        base_row = cmp_df.iloc[0]
        delta = cmp_df.subtract(base_row, axis=1)
        st.caption(f"Deltas vs baseline (**{cmp_df.index[0]}**): CAGR / MaxDD / Sharpe in absolute terms.")
        st.dataframe(
            delta.assign(
                CAGR=lambda d: (d["CAGR"] * 100).round(2),
                Vol=lambda d: (d["Vol"] * 100).round(2),
                MaxDD=lambda d: (d["MaxDD"] * 100).round(2),
                Sharpe=lambda d: d["Sharpe"].round(2),
            )[["CAGR", "Vol", "MaxDD", "Sharpe"]],
            use_container_width=True,
        )
    if st.button("Clear pinned"):
        st.session_state.pinned = []
        st.rerun()

with st.expander("Resolved config JSON"):
    st.json(result.get("config", {}))

with st.expander("Full metrics"):
    st.json(m)
