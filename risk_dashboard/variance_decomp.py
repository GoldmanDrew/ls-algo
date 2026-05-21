"""Variance decomposition: σ² = β_SPY² σ_SPX² + σ_idio² with correlation lift."""

from __future__ import annotations

import math
from typing import Any, Mapping

TRUSTED_BETA_SOURCES = frozenset({"computed", "shrunk", "curated_fallback"})


def corr_lift(vix_pts: float) -> float:
    """Cross-sectional correlation lift in stress regimes."""
    v = float(vix_pts)
    if v <= 18.0:
        return 1.0
    if v <= 35.0:
        return 1.0 + 0.05 * (v - 18.0)
    return 1.85


def vix_to_realized_spx_sigma(
    vix_pts: float,
    *,
    vrp_factor: float = 0.80,
) -> float:
    """Map VIX index level to realized SPX annualized vol (VRP-adjusted)."""
    return max(0.01, float(vrp_factor) * float(vix_pts) / 100.0)


def compute_idio_sigma(
    sigma_total: float,
    *,
    beta_spy: float | None,
    sigma_spx: float,
) -> tuple[float, float, float]:
    """Return (sigma_sys, sigma_idio, sigma_total_used)."""
    total = max(float(sigma_total), 0.01)
    if beta_spy is None or not math.isfinite(float(beta_spy)):
        return 0.0, total, total
    b = float(beta_spy)
    sys = abs(b) * float(sigma_spx)
    idio_sq = max(total * total - sys * sys, (0.25 * total) ** 2)
    return sys, math.sqrt(idio_sq), total


def shocked_sigma_variance_decomp(
    sigma_base: float,
    *,
    beta_spy: float | None,
    beta_vol_vix: float,
    vix_current_pts: float,
    vix_new_pts: float,
    vrp_factor: float = 0.80,
    corr_lift_override: float | None = None,
    use_decomp: bool = True,
) -> float:
    """Shock σ via variance decomposition when SPY β is trusted, else elasticity."""
    from .vol_vix_beta_v3 import shocked_sigma_multiplicative

    if not use_decomp or beta_spy is None:
        return shocked_sigma_multiplicative(
            sigma_base,
            beta_vol_vix=beta_vol_vix,
            vix_current_pts=vix_current_pts,
            vix_new_pts=vix_new_pts,
        )

    sigma_spx_now = vix_to_realized_spx_sigma(vix_current_pts, vrp_factor=vrp_factor)
    sigma_spx_new = vix_to_realized_spx_sigma(vix_new_pts, vrp_factor=vrp_factor)
    _, sigma_idio, _ = compute_idio_sigma(sigma_base, beta_spy=beta_spy, sigma_spx=sigma_spx_now)
    lift = corr_lift_override if corr_lift_override is not None else corr_lift(vix_new_pts)
    sigma_idio_new = sigma_idio * lift
    sys_new = abs(float(beta_spy)) * sigma_spx_new
    return max(0.05, math.sqrt(sys_new * sys_new + sigma_idio_new * sigma_idio_new))


def build_variance_decomp_pack(
    underlyings: list[str],
    *,
    vol_vix_pack: dict[str, Any],
    beta_results: Mapping[str, Mapping[str, Any]] | None = None,
    vix_current_pts: float,
    vrp_factor: float = 0.80,
) -> dict[str, Any]:
    """Per-name variance decomposition metadata."""
    betas = vol_vix_pack.get("betas") or {}
    sigma_spx = vix_to_realized_spx_sigma(vix_current_pts, vrp_factor=vrp_factor)
    rows: dict[str, dict[str, Any]] = {}
    n_decomp = 0
    for sym in underlyings:
        u = sym.upper()
        br = (beta_results or {}).get(u) or {}
        beta_spy = br.get("beta_to_spy")
        beta_source = br.get("provenance") or ""
        vol_res = betas.get(u)
        sigma_base = None
        if vol_res is not None:
            sigma_base = getattr(vol_res, "sigma_ewma", None) or (
                vol_res.get("sigma_ewma") if isinstance(vol_res, dict) else None
            )
        if sigma_base is None:
            sigma_base = br.get("regime_vol_pct")
            if sigma_base is not None:
                sigma_base = float(sigma_base) / 100.0
        if sigma_base is None:
            continue
        use_decomp = beta_source in TRUSTED_BETA_SOURCES and beta_spy is not None
        sys, idio, total = compute_idio_sigma(float(sigma_base), beta_spy=beta_spy, sigma_spx=sigma_spx)
        if use_decomp:
            n_decomp += 1
        rows[u] = {
            "underlying": u,
            "sigma_total": total,
            "sigma_sys": sys,
            "sigma_idio": idio,
            "beta_spy": beta_spy,
            "beta_spy_source": beta_source,
            "use_variance_decomp": use_decomp,
            "sigma_spx": sigma_spx,
        }
    return {
        "sigma_spx": sigma_spx,
        "vrp_factor": vrp_factor,
        "rows": rows,
        "n_decomp": n_decomp,
        "n_total": len(rows),
    }
