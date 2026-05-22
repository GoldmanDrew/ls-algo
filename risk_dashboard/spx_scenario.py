"""SPX scenario engine: horizon shock scaling and historical path integration."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

from .scenario_engine import (
    ScenarioLegResult,
    horizon_to_years,
    model_leg_return,
    scale_spx_shock_for_horizon,
)
from .spx_shock_config import HorizonShockMode

HISTORICAL_SPX_SCENARIOS: tuple[dict[str, Any], ...] = (
    {
        "key": "aug_2015",
        "label": "Aug 2015",
        "spx_start": 0.0,
        "spx_peak": -0.10,
        "spx_end": -0.06,
        "peak_days": 30,
        "horizon_days": 252,
    },
    {
        "key": "q4_2018",
        "label": "Q4 2018",
        "spx_start": 0.0,
        "spx_peak": -0.20,
        "spx_end": -0.14,
        "peak_days": 63,
        "horizon_days": 252,
    },
    {
        "key": "mar_2020",
        "label": "Mar 2020 COVID",
        "spx_start": 0.0,
        "spx_peak": -0.34,
        "spx_end": 0.15,
        "peak_days": 21,
        "horizon_days": 252,
    },
    {
        "key": "bear_2022",
        "label": "2022 bear",
        "spx_start": 0.0,
        "spx_peak": -0.25,
        "spx_end": -0.18,
        "peak_days": 180,
        "horizon_days": 252,
    },
    {
        "key": "recovery_2023",
        "label": "2023 rally",
        "spx_start": 0.0,
        "spx_peak": 0.24,
        "spx_end": 0.20,
        "peak_days": 200,
        "horizon_days": 252,
    },
)

PATH_STEPS_DEFAULT = 12


def build_spx_cumulative_path(
    *,
    spx_start: float,
    spx_peak: float,
    spx_end: float,
    peak_days: int,
    horizon_days: int,
    n_steps: int = PATH_STEPS_DEFAULT,
) -> tuple[float, ...]:
    """Piecewise-linear cumulative SPX return path (0 → peak → end)."""
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


@dataclass
class SpxScenarioSpec:
    key: str
    label: str
    spx_cumulative: tuple[float, ...]
    horizon_key: str = "12M"
    peak_days: int | None = None
    spx_peak_pct: float | None = None
    spx_end_pct: float | None = None


def historical_scenario_specs(
    horizon_key: str = "12M",
    *,
    n_steps: int = PATH_STEPS_DEFAULT,
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
        specs.append(
            SpxScenarioSpec(
                key=str(h["key"]),
                label=str(h["label"]),
                spx_cumulative=path,
                horizon_key=horizon_key,
                peak_days=int(h["peak_days"]),
                spx_peak_pct=float(h["spx_peak"]),
                spx_end_pct=float(h["spx_end"]),
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
) -> ScenarioLegResult:
    """Compound leg P&L along discrete SPX path steps (Phase 5)."""
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
    wealth = 1.0
    price_acc = 0.0
    decay_acc = 0.0
    borrow_acc = 0.0
    dist_acc = 0.0
    model = "path"
    for i in range(1, n):
        u_ret = float(underlying_return_per_step[i - 1])
        leg_use = leg
        if zero_borrow:
            leg_use = {**leg, "borrow_fee_annual": 0.0}
        res = model_leg_return(
            leg=leg_use,
            underlying_return=u_ret,
            horizon_key="1M",
            vol_multiplier=1.0,
            underlying_sigma=underlying_sigma,
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
            per_step_u_leg.append(
                underlying_return_for_leg(
                    row,
                    leg,
                    delta_spx,
                    delta_spx,
                    stress_cfg=stress_cfg,
                    beta_spy_decomp=beta_spy_decomp,
                )
            )
        res = integrate_leg_along_spx_path(
            leg,
            spx_cumulative=spx_cumulative,
            horizon_years=horizon_years,
            underlying_return_per_step=per_step_u_leg,
            underlying_sigma=row.get("sigma"),
            zero_borrow=zero_borrow,
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
