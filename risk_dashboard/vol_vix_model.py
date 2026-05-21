"""Unified vol→VIX model factory (v2 / v3) with full scenario pack."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml

from .variance_decomp import build_variance_decomp_pack
from .vol_vix_beta import compute_vol_vix_betas, leg_sigma_for_vix_shock
from .vol_vix_beta_v3 import compute_vol_vix_betas_v3
from .vix_scenario import (
    ScenarioMode,
    borrow_rate_vix_stress,
    historical_scenario_specs,
    resolve_shocked_sigma,
)

DEFAULT_CONFIG: dict[str, Any] = {
    "estimator": "v3_log_elasticity",
    "ewma_lambda": 0.94,
    "history_days": 504,
    "shrink_k": 30.0,
    "vix_mean_reversion_kappa": 5.0,
    "vix_long_run_theta_pts": 18.0,
    "vrp_realized_factor": 0.80,
    "scenario_mode_default": "sustained",
    "borrow_vix_gamma_broad": 0.05,
    "borrow_vix_gamma_htb": 0.30,
    "htb_borrow_rate_threshold_pct": 5.0,
}


def load_vol_vix_config(repo_root: Path | None = None) -> dict[str, Any]:
    cfg = dict(DEFAULT_CONFIG)
    if repo_root is None:
        return cfg
    path = repo_root / "config" / "strategy_config.yml"
    if not path.is_file():
        return cfg
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        block = raw.get("vol_vix_beta") or {}
        if isinstance(block, dict):
            cfg.update({k: v for k, v in block.items() if v is not None})
    except Exception:
        pass
    return cfg


def compute_vol_vix_pack(
    underlyings: list[str],
    *,
    cache_dir: Path | None = None,
    underlying_meta: Mapping[str, Mapping[str, Any]] | None = None,
    beta_results: Mapping[str, Mapping[str, Any]] | None = None,
    config: dict[str, Any] | None = None,
    yf_module: Any | None = None,
) -> dict[str, Any]:
    """Build vol→VIX pack with estimator dispatch and variance decomposition."""
    cfg = config or DEFAULT_CONFIG
    estimator = str(cfg.get("estimator") or "v3_log_elasticity").lower()

    if estimator in ("v2", "v2_diff_ols"):
        pack = compute_vol_vix_betas(
            underlyings,
            cache_dir=cache_dir,
            underlying_meta=underlying_meta,
            yf_module=yf_module,
        )
    else:
        pack = compute_vol_vix_betas_v3(
            underlyings,
            cache_dir=cache_dir,
            underlying_meta=underlying_meta,
            ewma_lambda=float(cfg.get("ewma_lambda", 0.94)),
            history_days=int(cfg.get("history_days", 504)),
            shrink_k=float(cfg.get("shrink_k", 30.0)),
            yf_module=yf_module,
        )

    vix_pts = float(pack.get("vix_current_pts") or 20.0)
    decomp = build_variance_decomp_pack(
        underlyings,
        vol_vix_pack=pack,
        beta_results=beta_results,
        vix_current_pts=vix_pts,
        vrp_factor=float(cfg.get("vrp_realized_factor", 0.80)),
    )
    pack["variance_decomp"] = decomp
    pack["config"] = cfg
    return pack


def leg_sigma_for_vix_scenario(
    leg: dict[str, Any],
    *,
    underlying: str,
    underlying_sigma: float | None,
    vol_vix_pack: dict[str, Any],
    vix_new_pts: float,
    vix_shock_pts: float = 0.0,
    mode: ScenarioMode | None = None,
    corr_lift_override: float | None = None,
    borrow_lift: float = 1.0,
) -> tuple[float | None, float | None]:
    """Return (sigma_effective, borrow_annual_stressed) for a leg."""
    from .scenario_engine import resolve_sigma_annual

    cfg = vol_vix_pack.get("config") or DEFAULT_CONFIG
    sigma_base, _ = resolve_sigma_annual(leg, underlying_sigma=underlying_sigma)
    if sigma_base is None:
        return None, None

    vix_current = float(vol_vix_pack.get("vix_current_pts") or 20.0)
    if not (vix_new_pts == vix_new_pts):  # NaN → parallel shock offset
        vix_new_pts = vix_current + float(vix_shock_pts)

    scenario_mode: ScenarioMode = mode or str(cfg.get("scenario_mode_default") or "sustained")  # type: ignore[assignment]
    horizon_years = 1.0

    if vol_vix_pack.get("estimator_version") == "v2_diff_ols":
        sigma = leg_sigma_for_vix_shock(
            leg,
            underlying=underlying,
            underlying_sigma=underlying_sigma,
            vol_vix_pack=vol_vix_pack,
            vix_shock_pts=vix_shock_pts if vix_shock_pts else (vix_new_pts - vix_current),
        )
    else:
        sigma = resolve_shocked_sigma(
            sigma_base=sigma_base,
            underlying=underlying,
            vol_vix_pack=vol_vix_pack,
            variance_decomp=vol_vix_pack.get("variance_decomp"),
            vix_current_pts=vix_current,
            vix_new_pts=vix_new_pts,
            mode=scenario_mode,
            kappa=float(cfg.get("vix_mean_reversion_kappa", 5.0)),
            vix_theta_pts=float(cfg.get("vix_long_run_theta_pts", 18.0)),
            horizon_years=horizon_years,
            vrp_factor=float(cfg.get("vrp_realized_factor", 0.80)),
            corr_lift_override=corr_lift_override,
        )

    borrow_base = float(leg.get("borrow_fee_annual") or 0.0)
    htb_thresh = float(cfg.get("htb_borrow_rate_threshold_pct", 5.0))
    is_htb = borrow_base * 100.0 >= htb_thresh if borrow_base > 0 else False
    borrow_stressed = borrow_rate_vix_stress(
        borrow_base,
        vix_pts=vix_new_pts,
        gamma_broad=float(cfg.get("borrow_vix_gamma_broad", 0.05)),
        gamma_htb=float(cfg.get("borrow_vix_gamma_htb", 0.30)),
        is_htb=is_htb,
        borrow_lift=borrow_lift,
    )
    return sigma, borrow_stressed


def beta_summary_dict(vol_vix_pack: dict[str, Any]) -> dict[str, dict[str, Any]]:
    betas = vol_vix_pack.get("betas") or {}
    out: dict[str, dict[str, Any]] = {}
    for sym, res in betas.items():
        if hasattr(res, "to_dict"):
            out[sym] = res.to_dict()
        elif isinstance(res, dict):
            out[sym] = res
    return out
