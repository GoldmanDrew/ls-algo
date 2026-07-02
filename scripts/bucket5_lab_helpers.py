"""Pure helpers for Bucket 5 lab sweeps / tornado (unit-testable, no Streamlit)."""

from __future__ import annotations

import copy
from typing import Any

TARGET_METRICS: dict[str, tuple[str, str, str, bool]] = {
    "CAGR": ("metrics", "combined_CAGR", "pct", True),
    "Vol": ("metrics", "combined_Vol", "pct", False),
    "Max drawdown": ("metrics", "combined_MaxDD", "pct", True),
    "Sharpe": ("metrics", "combined_Sharpe", "num", True),
    "Calmar": ("metrics", "combined_Calmar", "num", True),
    "Realized $ (harvested)": ("metrics", "realized_$", "usd", True),
    "Crash payoff -20%": ("crash", "crash_mild_-20%", "pct", True),
    "Crash payoff -30%": ("crash", "crash_severe_-30%", "pct", True),
    "Crash payoff -40%": ("crash", "crash_volmageddon_-40%", "pct", True),
}

PARAM_SPECS: dict[str, dict[str, Any]] = {
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


def get_path(d: dict, path: list[str]) -> Any:
    cur = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def set_path(d: dict, path: list[str], value: Any) -> None:
    cur = d
    for k in path[:-1]:
        cur = cur.setdefault(k, {})
    cur[path[-1]] = value


def apply_override(
    base_params: dict,
    methodology: dict,
    label: str,
    value: float,
) -> tuple[dict, dict]:
    spec = PARAM_SPECS[label]
    params = copy.deepcopy(base_params)
    meth = copy.deepcopy(methodology)
    if spec["loc"] == "methodology":
        set_path(meth, spec["path"], float(value))
    elif spec["loc"] == "total_premium":
        rungs = params.get("rungs") or []
        cur_total = sum(r.get("per_roll_frac", 0.0) for r in rungs) or 1e-9
        scale = float(value) / cur_total
        for r in rungs:
            r["per_roll_frac"] = r.get("per_roll_frac", 0.0) * scale
    else:
        set_path(params, spec["path"], float(value))
    return params, meth


def current_value(base_params: dict, methodology: dict, label: str) -> float | None:
    spec = PARAM_SPECS[label]
    if spec["loc"] == "methodology":
        return get_path(methodology, spec["path"])
    if spec["loc"] == "total_premium":
        return sum(r.get("per_roll_frac", 0.0) for r in (base_params.get("rungs") or []))
    return get_path(base_params, spec["path"])


def metric_value(result: dict, target_label: str) -> Any:
    src, key, _, _ = TARGET_METRICS[target_label]
    return (result.get(src) or {}).get(key)


def series_to_frame(series: dict, key: str) -> "pd.DataFrame | None":
    import pandas as pd

    rows = series.get(key) or []
    if not rows:
        return None
    df = pd.DataFrame(rows, columns=["date", key])
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")


def harvest_event_dates(series: dict, min_usd: float = 5000.0) -> list[str]:
    """Dates where cumulative realized harvest jumps materially."""
    import pandas as pd

    rc = series_to_frame(series, "realized_cum")
    if rc is None or len(rc) < 2:
        return []
    d = rc["realized_cum"].diff().fillna(0.0)
    hits = d[d >= min_usd].index
    return [ts.strftime("%Y-%m-%d") for ts in hits]


def monetize_event_dates(events: list[dict] | None, *, min_usd: float = 0.0) -> list[tuple[str, str]]:
    """Return (date, label) pairs from engine monetize_events."""
    if not events:
        return []
    out: list[tuple[str, str]] = []
    for ev in events:
        usd = float(ev.get("usd") or 0.0)
        if usd < min_usd:
            continue
        kind = str(ev.get("kind") or "harvest")
        out.append((str(ev.get("date")), kind.replace("_", " ")))
    return out


def apply_lab_preset_widget_state(blob: dict) -> dict[str, Any]:
    """Flatten a saved lab preset JSON into Streamlit widget session_state keys."""
    params = blob.get("params") or {}
    meth = blob.get("methodology") or {}
    mon = params.get("monetize") or {}
    reg = params.get("regime") or {}
    redeploy = params.get("redeploy") or {}
    hb = params.get("hedge_budget") or {}
    bs = params.get("backspread") or {}
    rungs = params.get("rungs") or []
    total_premium = sum(float(r.get("per_roll_frac", 0.0)) for r in rungs) or 0.024
    prem_sum = sum(float(r.get("per_roll_frac", 0.0)) for r in rungs) or total_premium
    def _rung_w(otm: float, default: float) -> float:
        frac = next((float(r.get("per_roll_frac", 0.0)) for r in rungs if abs(float(r.get("otm_pct", 0)) - otm) < 0.02), default * prem_sum)
        return frac / prem_sum if prem_sum > 0 else default
    w_near = _rung_w(0.10, 0.15)
    w_mid = _rung_w(0.25, 0.35)
    w_deep = _rung_w(0.35, 0.50)
    pt = mon.get("profit_tiers") or [[3.0, 0.34], [5.0, 0.50], [8.0, 1.0]]
    vt = mon.get("vix_tiers") or [[45.0, 0.50], [65.0, 1.0]]
    state: dict[str, Any] = {
        "lab_era": blob.get("era", "extended"),
        "lab_preset": blob.get("preset", "custom"),
        "sleeve_frac": float(params.get("sleeve_frac", 0.20)),
        "tbill_rate": float(params.get("tbill_rate", 0.043)),
        "uvix_slip": float(params.get("uvix_slip_bps", 5.0)),
        "fee_bps": float(params.get("fee_bps", 1.0)),
        "borrow_mode": meth.get("borrow_mode", "fixed"),
        "borrow_uvix": float(meth.get("borrow_uvix", 0.0284)) * 100.0,
        "borrow_svix": float(meth.get("borrow_svix", 0.0347)) * 100.0,
        "borrow_stress": float(meth.get("borrow_stress_mult", 1.0)),
        "hedge_kind": params.get("hedge_kind", "ladder"),
        "total_budget": float(total_premium) * 100.0,
        "w_deep": float(w_deep),
        "w_mid": float(w_mid),
        "dynamic_budget": hb.get("enabled", True),
        "hb_contango_mult": float(hb.get("contango_mult", 1.20)),
        "hb_stress_mult": float(hb.get("stress_mult", 0.85)),
        "hb_vix_lo": float(hb.get("vix_lo", 14.0)),
        "hb_vix_hi": float(hb.get("vix_hi", 28.0)),
        "hb_calm_boost": float(hb.get("vix_calm_boost", 1.10)),
        "hybrid_ladder_frac": float(params.get("hybrid_ladder_frac", 0.70)),
        "bs_otm_near": float(bs.get("otm_near", 0.12)),
        "bs_otm_far": float(bs.get("otm_far", 0.30)),
        "bs_far_ratio": int(bs.get("far_ratio", 3)),
        "bs_premium_frac": float(bs.get("premium_frac", 0.024)),
        "base_days": float(params.get("base_days", 14.0)),
        "cadence_k": float(params.get("cadence_k", 6.0)),
        "rho_contango": float(reg.get("rho_contango", 1.0)),
        "rho_backwardation": float(reg.get("rho_backwardation", 2.0)),
        "gross_contango": float(reg.get("gross_contango", 1.0)),
        "gross_backwardation": float(reg.get("gross_backwardation", 0.35)),
        "tier1_mult": float(pt[0][0]) if len(pt) > 0 else 3.0,
        "tier1_frac": float(pt[0][1]) if len(pt) > 0 else 0.34,
        "tier2_mult": float(pt[1][0]) if len(pt) > 1 else 5.0,
        "tier2_frac": float(pt[1][1]) if len(pt) > 1 else 0.50,
        "tier3_mult": float(pt[2][0]) if len(pt) > 2 else 8.0,
        "tier3_frac": float(pt[2][1]) if len(pt) > 2 else 1.0,
        "vix1_lvl": float(vt[0][0]) if len(vt) > 0 else 45.0,
        "vix1_frac": float(vt[0][1]) if len(vt) > 0 else 0.50,
        "vix2_lvl": float(vt[1][0]) if len(vt) > 1 else 65.0,
        "vix2_frac": float(vt[1][1]) if len(vt) > 1 else 1.0,
        "mon_giveback": float(mon.get("giveback_frac", 0.35)),
        "mon_giveback_min": float(mon.get("giveback_min_mult", 2.0)),
        "mon_bank": float(mon.get("bank_frac", 0.60)),
        "mon_runner": float(mon.get("runner_frac", 0.15)),
        "mon_runner_mult": float(mon.get("runner_mult", 12.0)),
        "mon_rearm": bool(mon.get("rearm", True)),
        "rd_contango": float(redeploy.get("sleeve_w_contango", 0.20)),
        "rd_backwardation": float(redeploy.get("sleeve_w_backwardation", 0.65)),
    }
    state["_w_near"] = float(w_near)
    return state


def run_2d_sweep(
    base_params: dict,
    methodology: dict,
    param_x: str,
    param_y: str,
    *,
    n_x: int = 5,
    n_y: int = 5,
    target_label: str = "CAGR",
    metric_fn,
) -> "pd.DataFrame":
    """Build a pivot table for two-parameter sensitivity."""
    import pandas as pd

    spec_x, spec_y = PARAM_SPECS[param_x], PARAM_SPECS[param_y]
    xs = [spec_x["range"][0] + (spec_x["range"][1] - spec_x["range"][0]) * i / (n_x - 1) for i in range(n_x)] if n_x > 1 else [spec_x["range"][0]]
    ys = [spec_y["range"][0] + (spec_y["range"][1] - spec_y["range"][0]) * j / (n_y - 1) for j in range(n_y)] if n_y > 1 else [spec_y["range"][0]]
    rows = []
    for xv in xs:
        for yv in ys:
            px, mx = apply_override(base_params, methodology, param_x, xv)
            py, my = apply_override(px, mx, param_y, yv)
            rows.append({param_x: xv, param_y: yv, target_label: metric_fn(py, my, target_label)})
    df = pd.DataFrame(rows)
    return df.pivot(index=param_y, columns=param_x, values=target_label)


def backwardation_mask(series: dict, r_lo: float = 0.88) -> list[tuple[str, str]]:
    """Return (start, end) date strings where VIX/VIX3M ratio is below r_lo (stress)."""
    import pandas as pd

    ratio = series_to_frame(series, "ratio")
    if ratio is None:
        return []
    stress = ratio["ratio"] < r_lo
    spans: list[tuple[str, str]] = []
    in_span = False
    start = None
    for dt, flag in stress.items():
        if flag and not in_span:
            in_span = True
            start = dt
        elif not flag and in_span:
            in_span = False
            spans.append((start.strftime("%Y-%m-%d"), dt.strftime("%Y-%m-%d")))
    if in_span and start is not None:
        spans.append((start.strftime("%Y-%m-%d"), ratio.index[-1].strftime("%Y-%m-%d")))
    return spans


def pack_lab_preset(
    *,
    era: str,
    preset: str,
    params: dict,
    methodology: dict,
    label: str = "",
) -> dict:
    return {
        "schema": "bucket5_lab_preset.v1",
        "label": label or preset,
        "era": era,
        "preset": preset,
        "params": params,
        "methodology": methodology,
    }
