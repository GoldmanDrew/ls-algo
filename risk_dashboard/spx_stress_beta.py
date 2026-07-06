"""Stress-adjusted β for SPX down-shock scenarios."""

from __future__ import annotations

import math
from typing import Any, Mapping


def leg_beta_to_spy(
    row: Mapping[str, Any],
    leg: Mapping[str, Any],
) -> float | None:
    """Effective underlying β to SPY for a leg (Phase 4: screener Delta chain)."""
    beta_row = row.get("beta_to_spy")
    if beta_row is None or not math.isfinite(float(beta_row)):
        return None
    b = float(beta_row)
    delta = leg.get("beta_to_underlying")
    if delta is not None and math.isfinite(float(delta)):
        product_class = str(leg.get("product_class") or "").lower()
        if product_class in ("letf_long", "letf_inverse"):
            return b
    return b


def stress_beta_to_spy(
    beta_base: float,
    spx_shock_pct: float,
    *,
    stress_cfg: Mapping[str, Any] | None = None,
    beta_spy_decomp: float | None = None,
) -> float:
    """Widen β on large down SPX shocks; optional cap from variance-decomp β."""
    cfg = stress_cfg or {}
    if not cfg.get("enabled", True):
        return float(beta_base)
    b = float(beta_base)
    shock = float(spx_shock_pct)
    if cfg.get("use_cumulative_drawdown", True):
        shock = min(0.0, shock)
    threshold = float(cfg.get("down_threshold_pct", -0.05))
    delta = float(cfg.get("down_delta", 0.15))
    if shock < threshold:
        excess = max(0.0, abs(shock) - abs(threshold))
        b *= 1.0 + delta * excess / 0.10
    if cfg.get("use_variance_decomp_cap", True) and beta_spy_decomp is not None:
        cap_f = float(cfg.get("decomp_cap_factor", 1.0))
        cap = abs(float(beta_spy_decomp)) * cap_f
        if cap > 0 and abs(b) > cap:
            b = math.copysign(cap, b)
    return b


def underlying_return_for_leg(
    row: Mapping[str, Any],
    leg: Mapping[str, Any],
    spx_shock_terminal_pct: float,
    spx_shock_effective_pct: float,
    *,
    stress_cfg: Mapping[str, Any] | None = None,
    beta_spy_decomp: float | None = None,
    spx_cumulative_pct: float | None = None,
) -> float:
    """Map terminal + horizon-scaled SPX shocks to underlying return for one leg."""
    beta = leg_beta_to_spy(row, leg)
    if beta is None:
        return 0.0
    stress_input = (
        float(spx_cumulative_pct)
        if spx_cumulative_pct is not None
        else float(spx_shock_terminal_pct)
    )
    beta_eff = stress_beta_to_spy(
        beta,
        stress_input,
        stress_cfg=stress_cfg,
        beta_spy_decomp=beta_spy_decomp,
    )
    return beta_eff * float(spx_shock_effective_pct)


def leg_instant_price_return(
    leg: Mapping[str, Any],
    underlying_return: float,
) -> float:
    """First-order T+0 price return for a leg (no decay/borrow)."""
    product = str(leg.get("product_class") or "").lower()
    u = float(underlying_return)
    if product in ("letf_long", "letf_inverse"):
        k = float(leg.get("leverage_k") or leg.get("beta_to_underlying") or 1.0)
        return k * u
    if product in ("income_yieldboost", "income_put_spread", "scraped_income"):
        return u
    return u
