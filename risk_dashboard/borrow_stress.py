"""Borrow stress: tiered rates, VIX paths, path-integrated annualized borrow."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Literal

import yaml

BorrowTier = Literal["gc", "htb", "etf_short"]

ETF_SHORT_PRODUCT_CLASSES: frozenset[str] = frozenset(
    {
        "income_yieldboost",
        "income_put_spread",
        "scraped_income",
        "letf_long",
        "letf_inverse",
        "volatility_etp",
    }
)

DEFAULT_BORROW_STRESS_CONFIG: dict[str, Any] = {
    "gamma_broad": 0.05,
    "gamma_htb": 0.30,
    "gamma_etf": 0.02,
    "htb_threshold_annual": 0.05,
    "htb_cap_multiple": 2.0,
    "gc_cap_multiple": 1.5,
    "vix_stress_threshold_pts": 20.0,
    "event_lift_peak_window_days": 45,
    "path_steps_per_year": 252,
    "calibrated_lifts": {},
}


def load_borrow_stress_config(repo_root: Path | None = None) -> dict[str, Any]:
    cfg = {
        **DEFAULT_BORROW_STRESS_CONFIG,
        "calibrated_lifts": dict(DEFAULT_BORROW_STRESS_CONFIG["calibrated_lifts"]),
    }
    if repo_root is None:
        return cfg
    for rel in (
        "config/risk_dashboard_borrow_lifts.yml",
        "config/strategy_config.yml",
    ):
        path = repo_root / rel
        if not path.is_file():
            continue
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            block = raw.get("borrow_stress") or raw.get("risk_dashboard", {}).get("borrow_stress")
            if isinstance(block, dict):
                lifts = block.get("calibrated_lifts")
                if isinstance(lifts, dict):
                    cfg["calibrated_lifts"].update(lifts)
                for k, v in block.items():
                    if k != "calibrated_lifts" and v is not None:
                        cfg[k] = v
        except Exception:
            pass
    return cfg


def borrow_tier_for_leg(leg: dict[str, Any], *, htb_threshold: float = 0.05) -> BorrowTier:
    product = str(leg.get("product_class") or "").lower()
    if product in ETF_SHORT_PRODUCT_CLASSES:
        return "etf_short"
    base = float(leg.get("borrow_fee_annual") or 0.0)
    if base >= float(htb_threshold):
        return "htb"
    return "gc"


def borrow_rate_vix_stress(
    borrow_base: float,
    *,
    vix_pts: float,
    tier: BorrowTier = "gc",
    borrow_lift: float = 1.0,
    stress_cfg: dict[str, Any] | None = None,
) -> float:
    """Instantaneous stressed borrow rate (annualized fraction)."""
    cfg = stress_cfg or DEFAULT_BORROW_STRESS_CONFIG
    base = max(0.0, float(borrow_base))
    if tier == "htb":
        gamma = float(cfg.get("gamma_htb", 0.30))
        cap_m = float(cfg.get("htb_cap_multiple", 2.0))
    elif tier == "etf_short":
        gamma = float(cfg.get("gamma_etf", 0.02))
        cap_m = float(cfg.get("gc_cap_multiple", 1.5))
    else:
        gamma = float(cfg.get("gamma_broad", 0.05))
        cap_m = float(cfg.get("gc_cap_multiple", 1.5))
    vix_thr = float(cfg.get("vix_stress_threshold_pts", 20.0))
    bump = 1.0 + gamma * max(0.0, float(vix_pts) - vix_thr) / 10.0
    stressed = base * bump * float(borrow_lift)
    return min(stressed, base * cap_m * max(1.0, float(borrow_lift)))


def build_vix_cumulative_path(
    *,
    vix_start: float,
    vix_peak: float,
    vix_end: float,
    peak_days: int,
    horizon_days: int = 252,
    n_steps: int | None = None,
) -> tuple[float, ...]:
    """Piecewise-linear VIX path (start → peak → end), daily by default."""
    h = max(int(horizon_days), 1)
    n = int(n_steps) if n_steps is not None else h
    n = max(n, 2)
    peak_d = max(min(int(peak_days), h), 1)
    out: list[float] = []
    for i in range(n + 1):
        day = h * i / n
        if day <= peak_d:
            frac = day / peak_d if peak_d > 0 else 1.0
            v = float(vix_start) + frac * (float(vix_peak) - float(vix_start))
        else:
            frac = (day - peak_d) / max(h - peak_d, 1)
            v = float(vix_peak) + frac * (float(vix_end) - float(vix_peak))
        out.append(v)
    return tuple(out)


def event_borrow_lift_at_day(
    day: float,
    *,
    borrow_lift: float,
    peak_days: int,
    window_days: int = 45,
) -> float:
    """Apply narrative/calibrated lift only near the VIX peak window."""
    if borrow_lift <= 1.0 + 1e-9:
        return 1.0
    half = max(int(window_days), 1) / 2.0
    center = float(peak_days)
    if abs(float(day) - center) <= half:
        return float(borrow_lift)
    return 1.0


def effective_annual_borrow_from_vix_path(
    borrow_base: float,
    vix_path: tuple[float, ...],
    *,
    tier: BorrowTier = "gc",
    borrow_lift: float = 1.0,
    peak_days: int | None = None,
    stress_cfg: dict[str, Any] | None = None,
) -> float:
    """Time-average of path-wise stressed borrow → single annual rate for horizon."""
    cfg = stress_cfg or DEFAULT_BORROW_STRESS_CONFIG
    n = len(vix_path)
    if n < 2 or borrow_base <= 0:
        return max(0.0, float(borrow_base))
    h = max(int(cfg.get("path_steps_per_year", 252)), 1)
    window = int(cfg.get("event_lift_peak_window_days", 45))
    peak_d = int(peak_days) if peak_days is not None else h // 4
    daily_rates: list[float] = []
    for i in range(1, n):
        day = h * i / (n - 1)
        lift = event_borrow_lift_at_day(
            day, borrow_lift=borrow_lift, peak_days=peak_d, window_days=window
        )
        rate = borrow_rate_vix_stress(
            borrow_base,
            vix_pts=float(vix_path[i]),
            tier=tier,
            borrow_lift=lift,
            stress_cfg=cfg,
        )
        daily_rates.append(rate)
    return sum(daily_rates) / len(daily_rates) if daily_rates else float(borrow_base)


def resolve_borrow_lift(
    scenario_key: str,
    default_lift: float,
    stress_cfg: dict[str, Any] | None = None,
) -> float:
    cfg = stress_cfg or DEFAULT_BORROW_STRESS_CONFIG
    lifts = cfg.get("calibrated_lifts") or {}
    if scenario_key in lifts:
        entry = lifts[scenario_key]
        if isinstance(entry, dict):
            return float(entry.get("borrow_lift", default_lift))
        return float(entry)
    return float(default_lift)
