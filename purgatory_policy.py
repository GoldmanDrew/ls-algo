"""Shared reduce-only target policy for purgatory pair positions.

The policy is deliberately holdings-aware: a model target may reduce a pair,
including to zero, but the constrained two-leg target may never exceed the
pair's current gross exposure.  One hedge leg may grow only when the final
pair gross still declines.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

import math


EPSILON = 1e-9


@dataclass(frozen=True)
class PurgatoryConstraint:
    current_gross_usd: float
    desired_gross_usd: float
    allowed_gross_usd: float
    constrained_underlying_usd: float
    constrained_etf_usd: float
    blocked_add_usd: float
    reason: str

    def as_dict(self) -> dict[str, float | str]:
        return asdict(self)


def constrain_pair_targets(
    *,
    desired_underlying_usd: float,
    desired_etf_usd: float,
    current_underlying_usd: float,
    current_etf_usd: float,
) -> PurgatoryConstraint:
    """Scale desired pair targets so final gross cannot exceed current gross."""
    desired_under = float(desired_underlying_usd or 0.0)
    desired_etf = float(desired_etf_usd or 0.0)
    current_under = float(current_underlying_usd or 0.0)
    current_etf = float(current_etf_usd or 0.0)
    # Reduce to flat instead of flipping a held leg through zero.
    if current_under * desired_under < 0:
        desired_under = 0.0
    if current_etf * desired_etf < 0:
        desired_etf = 0.0

    current_gross = abs(current_under) + abs(current_etf)
    desired_gross = abs(desired_under) + abs(desired_etf)
    allowed_gross = min(current_gross, desired_gross)

    if current_gross <= EPSILON:
        constrained_under = 0.0
        constrained_etf = 0.0
        reason = "purgatory_flat_no_establish"
    elif desired_gross <= EPSILON:
        constrained_under = 0.0
        constrained_etf = 0.0
        reason = "purgatory_model_exit"
    elif desired_gross <= current_gross + EPSILON:
        constrained_under = desired_under
        constrained_etf = desired_etf
        reason = "purgatory_reduction_allowed"
    else:
        scale = current_gross / desired_gross
        constrained_under = desired_under * scale
        constrained_etf = desired_etf * scale
        reason = "purgatory_addition_clipped"

    constrained_gross = abs(constrained_under) + abs(constrained_etf)
    if constrained_gross > current_gross + 1e-6:
        raise ValueError(
            "reduce-only constraint increased pair gross: "
            f"current={current_gross:.6f} constrained={constrained_gross:.6f}"
        )

    return PurgatoryConstraint(
        current_gross_usd=current_gross,
        desired_gross_usd=desired_gross,
        allowed_gross_usd=constrained_gross,
        constrained_underlying_usd=constrained_under,
        constrained_etf_usd=constrained_etf,
        blocked_add_usd=max(0.0, desired_gross - constrained_gross),
        reason=reason,
    )


def allocate_shared_underlying(
    *,
    current_underlying_usd: float,
    etfs: Sequence[str],
    current_etf_usd: Mapping[str, float],
    desired_underlying_usd: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """Allocate a net shared-underlying position deterministically across pairs.

    Current ETF gross is the primary attribution key, matching the accounting
    convention.  Desired underlying gross is the fallback for flat ETF legs.
    """
    symbols = [str(s) for s in etfs]
    if not symbols:
        return {}
    desired = desired_underlying_usd or {}
    weights = {
        symbol: abs(float(current_etf_usd.get(symbol, 0.0) or 0.0))
        for symbol in symbols
    }
    if sum(weights.values()) <= EPSILON:
        weights = {
            symbol: abs(float(desired.get(symbol, 0.0) or 0.0))
            for symbol in symbols
        }
    total = sum(weights.values())
    if total <= EPSILON:
        weights = {symbol: 1.0 for symbol in symbols}
        total = float(len(symbols))
    current = float(current_underlying_usd or 0.0)
    return {symbol: current * weights[symbol] / total for symbol in symbols}


def projected_pair_gross_after_shares(
    *,
    underlying_shares: float,
    underlying_price: float,
    etf_shares: float,
    etf_price: float,
) -> float:
    values = (
        float(underlying_shares) * float(underlying_price),
        float(etf_shares) * float(etf_price),
    )
    if not all(math.isfinite(v) for v in values):
        return math.inf
    return abs(values[0]) + abs(values[1])


def is_reduce_only_projection(
    *,
    before_gross_usd: float,
    after_gross_usd: float,
    tolerance_usd: float = 1.0,
) -> bool:
    return float(after_gross_usd) <= float(before_gross_usd) + max(float(tolerance_usd), 0.0)


__all__ = [
    "PurgatoryConstraint",
    "allocate_shared_underlying",
    "constrain_pair_targets",
    "is_reduce_only_projection",
    "projected_pair_gross_after_shares",
]
