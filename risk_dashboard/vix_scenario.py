"""VIX scenario engine: path-integrated decay, historical analogs, borrow stress."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

from .variance_decomp import corr_lift, shocked_sigma_variance_decomp, vix_to_realized_spx_sigma
from .vol_vix_beta import MIN_SIGMA

ScenarioMode = Literal["sustained", "spike_revert"]

HISTORICAL_VIX_SCENARIOS: tuple[dict[str, Any], ...] = (
    {
        "key": "aug_2015_china",
        "label": "Aug 2015 China",
        "vix_start": 13.0,
        "vix_peak": 28.0,
        "vix_end": 18.0,
        "peak_days": 30,
        "corr_lift": 1.4,
        "borrow_lift": 1.2,
    },
    {
        "key": "feb_2018_xiv",
        "label": "Feb 2018 XIV",
        "vix_start": 12.0,
        "vix_peak": 37.0,
        "vix_end": 19.0,
        "peak_days": 14,
        "corr_lift": 1.6,
        "borrow_lift": 1.3,
    },
    {
        "key": "mar_2020_covid",
        "label": "Mar 2020 COVID",
        "vix_start": 14.0,
        "vix_peak": 82.0,
        "vix_end": 35.0,
        "peak_days": 45,
        "corr_lift": 1.9,
        "borrow_lift": 2.0,
    },
    {
        "key": "sep_2022_inflation",
        "label": "Sep 2022 inflation",
        "vix_start": 22.0,
        "vix_peak": 33.0,
        "vix_end": 25.0,
        "peak_days": 60,
        "corr_lift": 1.3,
        "borrow_lift": 1.1,
    },
    {
        "key": "aug_2024_yen_carry",
        "label": "Aug 2024 yen carry",
        "vix_start": 16.0,
        "vix_peak": 65.0,
        "vix_end": 22.0,
        "peak_days": 7,
        "corr_lift": 1.7,
        "borrow_lift": 1.5,
    },
)


def sigma_sq_path_integral(
    sigma_0: float,
    sigma_inf: float,
    *,
    kappa: float,
    horizon_years: float,
) -> float:
    """Closed-form ∫₀ᵀ σ²(t) dt for OU mean-reverting σ toward σ_∞."""
    t = max(float(horizon_years), 0.0)
    s0sq = float(sigma_0) ** 2
    sinfsq = float(sigma_inf) ** 2
    if t <= 0:
        return 0.0
    if kappa <= 0:
        return s0sq * t
    return sinfsq * t + (s0sq - sinfsq) * (1.0 - math.exp(-kappa * t)) / kappa


def effective_sigma_from_path_integral(integral: float, horizon_years: float) -> float:
    """σ_eff such that σ_eff² × T = ∫σ²(t)dt."""
    t = max(float(horizon_years), 1e-9)
    return max(MIN_SIGMA, math.sqrt(max(0.0, float(integral)) / t))


from .borrow_stress import borrow_rate_vix_stress as _borrow_rate_vix_stress_legacy


def borrow_rate_vix_stress(
    borrow_base: float,
    *,
    vix_pts: float,
    gamma_broad: float = 0.05,
    gamma_htb: float = 0.30,
    is_htb: bool = False,
    borrow_lift: float = 1.0,
) -> float:
    """Widen borrow cost in elevated VIX (legacy signature; delegates to borrow_stress)."""
    tier = "htb" if is_htb else "gc"
    return _borrow_rate_vix_stress_legacy(
        borrow_base,
        vix_pts=vix_pts,
        tier=tier,
        borrow_lift=borrow_lift,
        stress_cfg={
            "gamma_broad": gamma_broad,
            "gamma_htb": gamma_htb,
        },
    )


def select_regime_beta(
    beta_res: Any,
    *,
    vix_new_pts: float,
    vix_regime_split: float = 20.0,
) -> float:
    """Pick low- or high-VIX regime elasticity when available."""
    from .vol_vix_beta_v3 import DEFAULT_VOL_VIX_BETA

    if beta_res is None:
        return DEFAULT_VOL_VIX_BETA
    if isinstance(beta_res, dict):
        beta = beta_res.get("beta_vol_vix")
        beta_low = beta_res.get("beta_vol_vix_low")
        beta_high = beta_res.get("beta_vol_vix_high")
    else:
        beta = getattr(beta_res, "beta_vol_vix", None)
        beta_low = getattr(beta_res, "beta_vol_vix_low", None)
        beta_high = getattr(beta_res, "beta_vol_vix_high", None)
    if vix_new_pts > vix_regime_split and beta_high is not None:
        return float(beta_high)
    if vix_new_pts <= vix_regime_split and beta_low is not None:
        return float(beta_low)
    return float(beta) if beta is not None else DEFAULT_VOL_VIX_BETA


def resolve_shocked_sigma(
    *,
    sigma_base: float,
    underlying: str,
    vol_vix_pack: dict[str, Any],
    variance_decomp: dict[str, Any] | None,
    vix_current_pts: float,
    vix_new_pts: float,
    mode: ScenarioMode = "sustained",
    kappa: float = 5.0,
    vix_theta_pts: float = 18.0,
    horizon_years: float = 1.0,
    vrp_factor: float = 0.80,
    corr_lift_override: float | None = None,
) -> float:
    """Resolve effective scenario σ incorporating path integral and decomp."""
    betas = vol_vix_pack.get("betas") or {}
    beta_res = betas.get(underlying.upper())
    beta_vol = select_regime_beta(beta_res, vix_new_pts=vix_new_pts)

    decomp_rows = (variance_decomp or {}).get("rows") or {}
    decomp_row = decomp_rows.get(underlying.upper()) or {}
    beta_spy = decomp_row.get("beta_spy")
    use_decomp = bool(decomp_row.get("use_variance_decomp"))

    sigma_shock = shocked_sigma_variance_decomp(
        sigma_base,
        beta_spy=beta_spy,
        beta_vol_vix=beta_vol,
        vix_current_pts=vix_current_pts,
        vix_new_pts=vix_new_pts,
        vrp_factor=vrp_factor,
        corr_lift_override=corr_lift_override,
        use_decomp=use_decomp,
    )

    if mode == "sustained":
        return sigma_shock

    sigma_base_level = shocked_sigma_variance_decomp(
        sigma_base,
        beta_spy=beta_spy,
        beta_vol_vix=beta_vol,
        vix_current_pts=vix_current_pts,
        vix_new_pts=vix_current_pts,
        vrp_factor=vrp_factor,
        corr_lift_override=1.0,
        use_decomp=use_decomp,
    )
    sigma_inf = shocked_sigma_variance_decomp(
        sigma_base,
        beta_spy=beta_spy,
        beta_vol_vix=beta_vol,
        vix_current_pts=vix_current_pts,
        vix_new_pts=vix_theta_pts,
        vrp_factor=vrp_factor,
        corr_lift_override=1.0,
        use_decomp=use_decomp,
    )
    integral = sigma_sq_path_integral(
        sigma_shock, sigma_inf, kappa=kappa, horizon_years=horizon_years
    )
    return effective_sigma_from_path_integral(integral, horizon_years)


@dataclass
class VixScenarioSpec:
    key: str
    label: str
    vix_new_pts: float
    mode: ScenarioMode = "sustained"
    corr_lift: float | None = None
    borrow_lift: float = 1.0
    vix_peak_pts: float | None = None
    vix_end_pts: float | None = None
    vix_start_pts: float | None = None
    peak_days: int | None = None


def parallel_shock_specs(vix_shocks: tuple[int, ...]) -> list[VixScenarioSpec]:
    out: list[VixScenarioSpec] = []
    for pts in (0,) + tuple(sorted({int(x) for x in vix_shocks})):
        out.append(
            VixScenarioSpec(
                key=f"parallel_{pts:+d}",
                label=f"VIX {pts:+d} pts" if pts != 0 else "current",
                vix_new_pts=float("nan"),  # resolved at runtime from current + pts
                mode="sustained",
            )
        )
    return out


def historical_scenario_specs() -> list[VixScenarioSpec]:
    specs: list[VixScenarioSpec] = []
    for h in HISTORICAL_VIX_SCENARIOS:
        specs.append(
            VixScenarioSpec(
                key=str(h["key"]),
                label=str(h["label"]),
                vix_new_pts=float(h["vix_peak"]),
                mode="spike_revert",
                corr_lift=float(h["corr_lift"]),
                borrow_lift=float(h["borrow_lift"]),
                vix_peak_pts=float(h["vix_peak"]),
                vix_end_pts=float(h["vix_end"]),
                vix_start_pts=float(h["vix_start"]),
                peak_days=int(h["peak_days"]),
            )
        )
    return specs
