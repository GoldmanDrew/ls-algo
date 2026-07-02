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
