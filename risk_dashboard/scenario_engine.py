"""Portfolio scenario math aligned with etf-dashboard Scenarios tab.

Ports ``etf-dashboard/assets/scenario_returns.js`` and the YieldBOOST
``estimateIncomeStyleScenarioReturn`` path from ``index.html``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

try:
    from yieldboost_decay import (
        _DEFAULT_EXPENSE_RATIO_ANNUAL,
        _weekly_put_spread_loss,
    )
except ImportError:  # pragma: no cover
    _DEFAULT_EXPENSE_RATIO_ANNUAL = 0.0099

    def _weekly_put_spread_loss(  # type: ignore[misc]
        sigma_annual,
        *,
        underlying_return: float = 0.0,
        horizon_years: float = 1.0,
    ):
        raise ImportError("yieldboost_decay required for income scenarios")


DEFAULT_VOL_MULTIPLIERS: tuple[float, ...] = (0.6, 0.8, 1.0, 1.2, 1.4)
DEFAULT_SHOCK_MULTIPLIERS: tuple[float, ...] = (-3.0, -1.0, -1.0 / 3.0, 0.0, 1.0 / 3.0, 1.0, 3.0)

SLIDE_SCENARIO_HORIZONS: tuple[str, ...] = ("1M", "3M", "6M", "12M")

INCOME_PRODUCT_CLASSES: frozenset[str] = frozenset(
    {"income_yieldboost", "income_put_spread", "scraped_income"}
)


@dataclass(frozen=True)
class ScenarioLegResult:
    ok: bool
    model: str
    total_return: float
    price_return: float
    decay_return: float
    borrow_return: float
    distribution_return: float
    error: str | None = None


def horizon_to_years(horizon_key: str) -> float:
    k = str(horizon_key or "").upper()
    if k == "1M":
        return 1.0 / 12.0
    if k == "3M":
        return 3.0 / 12.0
    if k == "6M":
        return 6.0 / 12.0
    if k in ("12M", "1Y"):
        return 1.0
    raise ValueError(f"Unknown horizon key: {horizon_key!r}")


def build_vol_scenarios(
    base_vol_annual: float,
    multipliers: tuple[float, ...] = DEFAULT_VOL_MULTIPLIERS,
) -> list[dict[str, float]]:
    base = float(base_vol_annual)
    if not math.isfinite(base) or base <= 0:
        return []
    out: list[dict[str, float]] = []
    for m in multipliers:
        mult = float(m)
        if math.isfinite(mult) and mult > 0:
            out.append({"multiplier": mult, "sigma_annual": base * mult})
    return out


def build_shock_rows(
    best_vol_annual: float,
    horizon_years: float,
    multipliers: tuple[float, ...] = DEFAULT_SHOCK_MULTIPLIERS,
) -> list[dict[str, float]]:
    sigma = float(best_vol_annual)
    years = float(horizon_years)
    if not math.isfinite(sigma) or sigma <= 0 or not math.isfinite(years) or years <= 0:
        return []
    horizon_sigma = sigma * math.sqrt(years)
    rows: list[dict[str, float]] = []
    for m in multipliers:
        mm = float(m)
        if not math.isfinite(mm):
            continue
        rows.append(
            {
                "sigma_multiple": mm,
                "underlying_return": math.exp(mm * horizon_sigma) - 1.0,
            }
        )
    return rows


def estimate_etf_return(
    *,
    leverage: float,
    underlying_return: float,
    sigma_annual: float,
    horizon_years: float,
    annual_carry_drag: float = 0.0,
    is_short: bool = False,
    borrow_annual: float = 0.0,
    min_return: float = -0.9999,
) -> ScenarioLegResult:
    """LETF / inverse log-return model (``ScenarioReturns.estimateEtfReturn``)."""
    l_val = float(leverage)
    u_ret = float(underlying_return)
    sigma = float(sigma_annual)
    t_years = float(horizon_years)
    carry = float(annual_carry_drag) if math.isfinite(annual_carry_drag) else 0.0

    if (
        not math.isfinite(l_val)
        or not math.isfinite(u_ret)
        or not math.isfinite(sigma)
        or sigma < 0
        or not math.isfinite(t_years)
        or t_years <= 0
    ):
        return ScenarioLegResult(
            ok=False,
            model="letf",
            total_return=0.0,
            price_return=0.0,
            decay_return=0.0,
            borrow_return=0.0,
            distribution_return=0.0,
            error="Invalid LETF model inputs.",
        )

    one_plus_u = 1.0 + u_ret
    if not math.isfinite(one_plus_u) or one_plus_u <= 0:
        return ScenarioLegResult(
            ok=False,
            model="letf",
            total_return=0.0,
            price_return=0.0,
            decay_return=0.0,
            borrow_return=0.0,
            distribution_return=0.0,
            error="Invalid underlying return for log compounding.",
        )

    drag_log = 0.5 * l_val * (l_val - 1.0) * (sigma**2) * t_years
    underlying_log = math.log(one_plus_u)
    price_log = l_val * underlying_log
    price_return = math.exp(price_log) - 1.0
    after_drag_return = math.exp(price_log - drag_log) - 1.0
    decay_return = after_drag_return - price_return

    borrow_drag = 0.0
    if is_short and math.isfinite(borrow_annual) and borrow_annual > 0:
        borrow_drag = float(borrow_annual) * t_years

    etf_log = price_log - drag_log - (carry * t_years)
    raw = math.exp(etf_log) - 1.0
    if not math.isfinite(raw):
        return ScenarioLegResult(
            ok=False,
            model="letf",
            total_return=0.0,
            price_return=0.0,
            decay_return=0.0,
            borrow_return=0.0,
            distribution_return=0.0,
            error="Non-finite LETF model output.",
        )

    total_return = max(raw, min_return) + borrow_drag
    borrow_return = borrow_drag

    return ScenarioLegResult(
        ok=True,
        model="letf",
        total_return=total_return,
        price_return=price_return,
        decay_return=decay_return,
        borrow_return=borrow_return,
        distribution_return=0.0,
    )


def estimate_income_style_return(
    *,
    underlying_return: float,
    sigma_annual: float,
    annual_income_yield: float,
    horizon_years: float,
    annual_borrow_cost: float = 0.0,
    expense_ratio_annual: float = _DEFAULT_EXPENSE_RATIO_ANNUAL,
    is_short: bool = True,
) -> ScenarioLegResult | None:
    """YieldBOOST scenario return (``estimateIncomeStyleScenarioReturn``)."""
    d_annual = float(annual_income_yield)
    t = float(horizon_years)
    if not math.isfinite(d_annual) or d_annual < 0 or not math.isfinite(t) or t <= 0:
        return None

    sigma = float(sigma_annual)
    if not math.isfinite(sigma) or sigma <= 0:
        return None

    weekly_arr = _weekly_put_spread_loss(
        [sigma],
        underlying_return=float(underlying_return),
        horizon_years=t,
    )
    weekly_spread_loss = float(weekly_arr[0])
    if not math.isfinite(weekly_spread_loss):
        return None

    weeks = max(1, round(t * 52))
    weekly_expense = max(0.0, float(expense_ratio_annual)) / 52.0
    weekly_distribution = d_annual / 52.0
    q = max(0.0001, min(1.5, 1.0 - weekly_spread_loss - weekly_expense))
    nav_end_ratio = q**weeks
    nav_decay = 1.0 - nav_end_ratio
    geom_sum = weeks if abs(1.0 - q) < 1e-9 else (1.0 - (q**weeks)) / (1.0 - q)
    distributions_paid = weekly_distribution * geom_sum
    borrow_cost = (
        float(annual_borrow_cost) * t
        if math.isfinite(annual_borrow_cost) and annual_borrow_cost > 0
        else 0.0
    )

    zero_weekly = float(
        _weekly_put_spread_loss([sigma], underlying_return=0.0, horizon_years=t)[0]
    )
    q0 = max(0.0001, min(1.5, 1.0 - zero_weekly - weekly_expense))
    nav_decay0 = 1.0 - (q0**weeks)
    geom0 = weeks if abs(1.0 - q0) < 1e-9 else (1.0 - (q0**weeks)) / (1.0 - q0)
    dist0 = weekly_distribution * geom0
    borrow0 = borrow_cost

    long_total_return = -nav_decay + distributions_paid
    net_short_pnl = nav_decay - distributions_paid - borrow_cost

    if is_short:
        total_return = net_short_pnl
        distribution_return = -dist0
    else:
        total_return = long_total_return - borrow_cost
        distribution_return = dist0
    borrow_return = -borrow0

    if is_short:
        carry0 = nav_decay0 - dist0 - borrow0
        decay_return = nav_decay0
    else:
        carry0 = -nav_decay0 + dist0 - borrow0
        decay_return = -nav_decay0

    price_return = total_return - carry0

    return ScenarioLegResult(
        ok=True,
        model="yieldboost",
        total_return=total_return,
        price_return=price_return,
        decay_return=decay_return,
        borrow_return=borrow_return,
        distribution_return=distribution_return,
    )


def resolve_sigma_annual(
    leg: dict[str, Any],
    *,
    underlying_sigma: float | None = None,
) -> tuple[float | None, str]:
    """Best-estimate underlying σ at 1.0× (etf-dashboard forecast priority)."""
    for key, source in (
        ("forecast_vol_underlying_annual", "forecast_vol"),
        ("vol_underlying_annual", "vol_underlying"),
        ("vol_etf_annual", "vol_etf"),
    ):
        val = leg.get(key)
        if val is not None:
            sigma = float(val)
            if math.isfinite(sigma) and sigma > 0:
                return sigma, source
    if underlying_sigma is not None:
        sigma = float(underlying_sigma)
        if math.isfinite(sigma) and sigma > 0:
            return sigma, "regime_vol"
    return None, "missing"


def model_leg_return(
    *,
    leg: dict[str, Any],
    underlying_return: float,
    horizon_key: str,
    vol_multiplier: float = 1.0,
    underlying_sigma: float | None = None,
    sigma_annual_override: float | None = None,
) -> ScenarioLegResult:
    """Route a position leg to the appropriate scenario model."""
    t_years = horizon_to_years(horizon_key)
    sigma_base, _ = resolve_sigma_annual(leg, underlying_sigma=underlying_sigma)
    if sigma_base is None and sigma_annual_override is None:
        beta = leg.get("leverage_k") or leg.get("beta_to_spy") or 1.0
        linear = float(beta) * float(underlying_return)
        return ScenarioLegResult(
            ok=True,
            model="beta_fallback",
            total_return=linear,
            price_return=linear,
            decay_return=0.0,
            borrow_return=0.0,
            distribution_return=0.0,
            error="missing_sigma",
        )

    if sigma_annual_override is not None and math.isfinite(float(sigma_annual_override)):
        sigma = max(0.0, float(sigma_annual_override))
    else:
        sigma = float(sigma_base) * float(vol_multiplier)
    product_class = str(leg.get("product_class") or "").lower()
    is_short = float(leg.get("net_notional_usd") or 0.0) < 0
    borrow = float(leg.get("borrow_fee_annual") or 0.0)
    income_yield = leg.get("income_distributions_annual")

    if product_class in INCOME_PRODUCT_CLASSES and income_yield is not None:
        yb = estimate_income_style_return(
            underlying_return=float(underlying_return),
            sigma_annual=sigma,
            annual_income_yield=float(income_yield),
            horizon_years=t_years,
            annual_borrow_cost=borrow if is_short else 0.0,
            expense_ratio_annual=float(
                leg.get("expense_ratio_annual") or _DEFAULT_EXPENSE_RATIO_ANNUAL
            ),
            is_short=is_short,
        )
        if yb is not None:
            return yb

    leverage = float(leg.get("leverage_k") or leg.get("beta_to_underlying") or 1.0)
    if abs(leverage) <= 1.0001 and product_class not in ("letf_long", "letf_inverse"):
        passive_return = float(underlying_return)
        borrow_return = 0.0
        if is_short and math.isfinite(borrow) and borrow > 0:
            borrow_return = borrow * t_years
        return ScenarioLegResult(
            ok=True,
            model="passive",
            total_return=passive_return + borrow_return,
            price_return=passive_return,
            decay_return=0.0,
            borrow_return=borrow_return,
            distribution_return=0.0,
        )

    return estimate_etf_return(
        leverage=leverage,
        underlying_return=float(underlying_return),
        sigma_annual=sigma,
        horizon_years=t_years,
        is_short=is_short,
        borrow_annual=borrow,
    )


def aggregate_leg_scenario_pnl(
    legs: list[dict[str, Any]],
    *,
    underlying_return: float,
    horizon_key: str,
    vol_multiplier: float = 1.0,
    underlying_sigma: float | None = None,
    sigma_overrides: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Sum scenario P&L components across legs (USD, not %)."""
    totals = {
        "beta_pnl_usd": 0.0,
        "decay_pnl_usd": 0.0,
        "borrow_pnl_usd": 0.0,
        "distribution_pnl_usd": 0.0,
        "total_pnl_usd": 0.0,
    }
    n_modeled = 0
    n_fallback = 0
    models: set[str] = set()

    for leg in legs:
        notional = float(leg.get("net_notional_usd") or 0.0)
        if abs(notional) < 1e-9:
            continue
        sym = str(leg.get("symbol") or "").upper()
        sigma_override = (sigma_overrides or {}).get(sym)
        result = model_leg_return(
            leg=leg,
            underlying_return=underlying_return,
            horizon_key=horizon_key,
            vol_multiplier=vol_multiplier,
            underlying_sigma=underlying_sigma,
            sigma_annual_override=sigma_override,
        )
        models.add(result.model)
        if result.model == "beta_fallback":
            n_fallback += 1
        else:
            n_modeled += 1
        pnl_scale = abs(notional) if (notional < 0 and result.model == "yieldboost") else notional
        totals["beta_pnl_usd"] += pnl_scale * result.price_return
        totals["decay_pnl_usd"] += pnl_scale * result.decay_return
        totals["borrow_pnl_usd"] += pnl_scale * result.borrow_return
        totals["distribution_pnl_usd"] += pnl_scale * result.distribution_return
        totals["total_pnl_usd"] += pnl_scale * result.total_return

    totals["n_legs"] = len(legs)
    totals["n_legs_modeled"] = n_modeled
    totals["n_legs_fallback"] = n_fallback
    totals["models"] = sorted(models)
    return totals
