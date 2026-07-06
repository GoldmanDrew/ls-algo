"""SPX scenario engine: horizon shock scaling and historical path integration."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .borrow_stress import (
    build_vix_cumulative_path,
    resolve_borrow_lift,
)
from .scenario_engine import (
    ScenarioLegResult,
    horizon_to_years,
    model_leg_return,
)
from .vix_scenario import HISTORICAL_VIX_SCENARIOS

HISTORICAL_SPX_SCENARIOS: tuple[dict[str, Any], ...] = (
    {
        "key": "aug_2015",
        "label": "Aug 2015",
        "spx_start": 0.0,
        "spx_peak": -0.10,
        "spx_end": -0.06,
        "peak_days": 30,
        "horizon_days": 252,
        "vix_template_key": "aug_2015_china",
    },
    {
        "key": "q4_2018",
        "label": "Q4 2018",
        "spx_start": 0.0,
        "spx_peak": -0.20,
        "spx_end": -0.14,
        "peak_days": 63,
        "horizon_days": 252,
        "vix_template_key": "feb_2018_xiv",
    },
    {
        "key": "mar_2020",
        "label": "Mar 2020 COVID",
        "spx_start": 0.0,
        "spx_peak": -0.34,
        "spx_end": 0.15,
        "peak_days": 21,
        "horizon_days": 252,
        "vix_template_key": "mar_2020_covid",
    },
    {
        "key": "bear_2022",
        "label": "2022 bear",
        "spx_start": 0.0,
        "spx_peak": -0.25,
        "spx_end": -0.18,
        "peak_days": 180,
        "horizon_days": 252,
        "vix_template_key": "sep_2022_inflation",
    },
    {
        "key": "recovery_2023",
        "label": "2023 rally",
        "spx_start": 0.0,
        "spx_peak": 0.24,
        "spx_end": 0.20,
        "peak_days": 200,
        "horizon_days": 252,
        "vix_template_key": None,
    },
)

PATH_STEPS_DEFAULT = 252
REALIZED_VOL_WINDOW = 21


def build_spx_cumulative_path(
    *,
    spx_start: float,
    spx_peak: float,
    spx_end: float,
    peak_days: int,
    horizon_days: int,
    n_steps: int = PATH_STEPS_DEFAULT,
) -> tuple[float, ...]:
    """Piecewise-linear cumulative SPX return path (0 → peak → end), daily by default."""
    n = max(int(n_steps), 2)
    h = max(int(horizon_days), 1)
    peak_d = max(min(int(peak_days), h), 1)
    out: list[float] = []
    for i in range(n + 1):
        day = h * i / n
        if day <= peak_d:
            frac = day / peak_d if peak_d > 0 else 1.0
            r = float(spx_start) + frac * (float(spx_peak) - float(spx_start))
        else:
            frac = (day - peak_d) / max(h - peak_d, 1)
            r = float(spx_peak) + frac * (float(spx_end) - float(spx_peak))
        out.append(r)
    return tuple(out)


def rolling_realized_vol_from_spx_path(
    spx_cumulative: tuple[float, ...],
    step_index: int,
    *,
    base_sigma: float,
    window: int = REALIZED_VOL_WINDOW,
) -> float:
    """Annualized realized vol from trailing SPX path step returns."""
    if step_index < 1:
        return max(base_sigma, 0.05)
    start = max(1, step_index - window + 1)
    rets: list[float] = []
    for j in range(start, step_index + 1):
        prev = float(spx_cumulative[j - 1])
        cur = float(spx_cumulative[j])
        denom = 1.0 + prev
        if abs(denom) < 1e-9:
            continue
        rets.append((cur - prev) / denom)
    if len(rets) < 2:
        return max(base_sigma, 0.05)
    mean = sum(rets) / len(rets)
    var = sum((r - mean) ** 2 for r in rets) / max(len(rets) - 1, 1)
    realized = math.sqrt(max(var, 0.0)) * math.sqrt(252.0)
    blend = 0.35 * realized + 0.65 * max(base_sigma, 0.05)
    return max(blend, 0.05)


def vix_template_for_spx_key(template_key: str | None) -> dict[str, Any] | None:
    if not template_key:
        return None
    for h in HISTORICAL_VIX_SCENARIOS:
        if h["key"] == template_key:
            return dict(h)
    return None


def build_vix_path_for_spx_scenario(spec: dict[str, Any], n_steps: int) -> tuple[float, ...] | None:
    tmpl = vix_template_for_spx_key(spec.get("vix_template_key"))
    if tmpl is None:
        return None
    return build_vix_cumulative_path(
        vix_start=float(tmpl["vix_start"]),
        vix_peak=float(tmpl["vix_peak"]),
        vix_end=float(tmpl["vix_end"]),
        peak_days=int(tmpl["peak_days"]),
        horizon_days=int(spec.get("horizon_days", 252)),
        n_steps=n_steps,
    )


@dataclass
class SpxScenarioSpec:
    key: str
    label: str
    spx_cumulative: tuple[float, ...]
    horizon_key: str = "12M"
    peak_days: int | None = None
    spx_peak_pct: float | None = None
    spx_end_pct: float | None = None
    vix_path: tuple[float, ...] | None = None
    vix_template_key: str | None = None
    borrow_lift: float = 1.0


def historical_scenario_specs(
    horizon_key: str = "12M",
    *,
    n_steps: int = PATH_STEPS_DEFAULT,
    borrow_stress_cfg: dict[str, Any] | None = None,
) -> list[SpxScenarioSpec]:
    specs: list[SpxScenarioSpec] = []
    for h in HISTORICAL_SPX_SCENARIOS:
        path = build_spx_cumulative_path(
            spx_start=float(h["spx_start"]),
            spx_peak=float(h["spx_peak"]),
            spx_end=float(h["spx_end"]),
            peak_days=int(h["peak_days"]),
            horizon_days=int(h["horizon_days"]),
            n_steps=n_steps,
        )
        vix_path = build_vix_path_for_spx_scenario(h, n_steps)
        tmpl_key = h.get("vix_template_key")
        default_lift = 1.0
        if tmpl_key:
            tmpl = vix_template_for_spx_key(str(tmpl_key))
            if tmpl:
                default_lift = float(tmpl.get("borrow_lift", 1.0))
        lift = resolve_borrow_lift(
            str(tmpl_key or h["key"]),
            default_lift,
            borrow_stress_cfg,
        )
        specs.append(
            SpxScenarioSpec(
                key=str(h["key"]),
                label=str(h["label"]),
                spx_cumulative=path,
                horizon_key=horizon_key,
                peak_days=int(h["peak_days"]),
                spx_peak_pct=float(h["spx_peak"]),
                spx_end_pct=float(h["spx_end"]),
                vix_path=vix_path,
                vix_template_key=str(tmpl_key) if tmpl_key else None,
                borrow_lift=lift,
            )
        )
    return specs


def integrate_leg_along_spx_path(
    leg: dict[str, Any],
    *,
    spx_cumulative: tuple[float, ...],
    horizon_years: float,
    underlying_return_per_step: list[float],
    underlying_sigma: float | None = None,
    zero_borrow: bool = False,
    vix_path: tuple[float, ...] | None = None,
    borrow_lift: float = 1.0,
    peak_days: int | None = None,
    borrow_stress_cfg: dict[str, Any] | None = None,
) -> ScenarioLegResult:
    """Daily compound leg P&L along SPX path with path vol and optional VIX borrow."""
    from .borrow_stress import borrow_tier_for_leg

    n = len(spx_cumulative)
    if n < 2:
        return ScenarioLegResult(
            ok=False,
            model="path",
            total_return=0.0,
            price_return=0.0,
            decay_return=0.0,
            borrow_return=0.0,
            distribution_return=0.0,
            error="path_too_short",
        )
    t_step = float(horizon_years) / (n - 1)
    base_sigma = float(underlying_sigma or 0.25)
    wealth = 1.0
    price_acc = 0.0
    decay_acc = 0.0
    borrow_acc = 0.0
    dist_acc = 0.0
    model = "path"
    borrow_base = float(leg.get("borrow_fee_annual") or 0.0)
    tier = borrow_tier_for_leg(leg, htb_threshold=float((borrow_stress_cfg or {}).get("htb_threshold_annual", 0.05)))
    cfg = borrow_stress_cfg or {}
    h_days = max(int(cfg.get("path_steps_per_year", 252)), 1)
    window = int(cfg.get("event_lift_peak_window_days", 45))
    peak_d = int(peak_days) if peak_days is not None else h_days // 4

    for i in range(1, n):
        u_ret = float(underlying_return_per_step[i - 1])
        sigma_step = rolling_realized_vol_from_spx_path(
            spx_cumulative, i, base_sigma=base_sigma
        )
        leg_use = dict(leg)
        if zero_borrow:
            leg_use["borrow_fee_annual"] = 0.0
        elif vix_path and len(vix_path) == n and borrow_base > 0:
            from .borrow_stress import borrow_rate_vix_stress, event_borrow_lift_at_day

            day = h_days * i / (n - 1)
            lift = event_borrow_lift_at_day(
                day, borrow_lift=borrow_lift, peak_days=peak_d, window_days=window
            )
            leg_use["borrow_fee_annual"] = borrow_rate_vix_stress(
                borrow_base,
                vix_pts=float(vix_path[i]),
                tier=tier,
                borrow_lift=lift,
                stress_cfg=cfg,
            )
        res = model_leg_return(
            leg=leg_use,
            underlying_return=u_ret,
            horizon_key="1M",
            vol_multiplier=1.0,
            underlying_sigma=sigma_step,
            horizon_years_override=t_step,
        )
        if not res.ok:
            continue
        model = res.model
        wealth *= 1.0 + float(res.total_return)
        price_acc += float(res.price_return)
        decay_acc += float(res.decay_return)
        borrow_acc += float(res.borrow_return)
        dist_acc += float(res.distribution_return)
    total = wealth - 1.0
    return ScenarioLegResult(
        ok=True,
        model=model,
        total_return=total,
        price_return=price_acc,
        decay_return=decay_acc,
        borrow_return=borrow_acc,
        distribution_return=dist_acc,
    )


def aggregate_path_scenario_pnl(
    legs: list[dict[str, Any]],
    *,
    spx_cumulative: tuple[float, ...],
    horizon_key: str,
    row: dict[str, Any],
    stress_cfg: dict[str, Any] | None,
    beta_spy_decomp: float | None,
    zero_borrow: bool = False,
    vix_path: tuple[float, ...] | None = None,
    borrow_lift: float = 1.0,
    peak_days: int | None = None,
    borrow_stress_cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Sum path-integrated leg P&L for one factor row."""
    from .spx_stress_beta import underlying_return_for_leg

    horizon_years = horizon_to_years(horizon_key)
    n = len(spx_cumulative)
    totals = {
        "beta_pnl_usd": 0.0,
        "decay_pnl_usd": 0.0,
        "borrow_pnl_usd": 0.0,
        "distribution_pnl_usd": 0.0,
        "total_pnl_usd": 0.0,
        "n_legs_modeled": 0,
        "n_legs_fallback": 0,
    }
    if n < 2:
        return totals
    for leg in legs:
        notional = float(leg.get("net_notional_usd") or 0.0)
        if abs(notional) < 1e-9:
            continue
        per_step_u_leg: list[float] = []
        for i in range(1, n):
            delta_spx = float(spx_cumulative[i] - spx_cumulative[i - 1])
            cum_spx = float(spx_cumulative[i])
            per_step_u_leg.append(
                underlying_return_for_leg(
                    row,
                    leg,
                    delta_spx,
                    delta_spx,
                    stress_cfg=stress_cfg,
                    beta_spy_decomp=beta_spy_decomp,
                    spx_cumulative_pct=cum_spx,
                )
            )
        res = integrate_leg_along_spx_path(
            leg,
            spx_cumulative=spx_cumulative,
            horizon_years=horizon_years,
            underlying_return_per_step=per_step_u_leg,
            underlying_sigma=row.get("sigma"),
            zero_borrow=zero_borrow,
            vix_path=vix_path,
            borrow_lift=borrow_lift,
            peak_days=peak_days,
            borrow_stress_cfg=borrow_stress_cfg,
        )
        if res.model == "beta_fallback":
            totals["n_legs_fallback"] += 1
        else:
            totals["n_legs_modeled"] += 1
        pnl_scale = abs(notional) if (notional < 0 and res.model == "yieldboost") else notional
        totals["beta_pnl_usd"] += pnl_scale * res.price_return
        totals["decay_pnl_usd"] += pnl_scale * res.decay_return
        totals["borrow_pnl_usd"] += pnl_scale * res.borrow_return
        totals["distribution_pnl_usd"] += pnl_scale * res.distribution_return
        totals["total_pnl_usd"] += pnl_scale * res.total_return
    return totals
