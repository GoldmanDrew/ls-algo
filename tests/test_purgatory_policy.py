from __future__ import annotations

import pytest

from purgatory_policy import (
    allocate_shared_underlying,
    constrain_pair_targets,
    is_reduce_only_projection,
    projected_pair_gross_after_shares,
)


def test_desired_reduction_passes_through() -> None:
    c = constrain_pair_targets(
        desired_underlying_usd=3_000,
        desired_etf_usd=-2_000,
        current_underlying_usd=6_000,
        current_etf_usd=-4_000,
    )
    assert c.allowed_gross_usd == pytest.approx(5_000)
    assert c.constrained_underlying_usd == pytest.approx(3_000)
    assert c.constrained_etf_usd == pytest.approx(-2_000)
    assert c.blocked_add_usd == pytest.approx(0)


def test_desired_addition_is_scaled_to_current_gross() -> None:
    c = constrain_pair_targets(
        desired_underlying_usd=9_000,
        desired_etf_usd=-6_000,
        current_underlying_usd=3_000,
        current_etf_usd=-2_000,
    )
    assert c.allowed_gross_usd == pytest.approx(5_000)
    assert abs(c.constrained_underlying_usd) + abs(c.constrained_etf_usd) == pytest.approx(5_000)
    assert c.blocked_add_usd == pytest.approx(10_000)


def test_zero_model_target_is_controlled_exit() -> None:
    c = constrain_pair_targets(
        desired_underlying_usd=0,
        desired_etf_usd=0,
        current_underlying_usd=-4_000,
        current_etf_usd=-3_000,
    )
    assert c.allowed_gross_usd == 0
    assert c.reason == "purgatory_model_exit"


def test_flat_pair_cannot_be_established() -> None:
    c = constrain_pair_targets(
        desired_underlying_usd=4_000,
        desired_etf_usd=-2_000,
        current_underlying_usd=0,
        current_etf_usd=0,
    )
    assert c.allowed_gross_usd == 0
    assert c.reason == "purgatory_flat_no_establish"


def test_sign_flip_reduces_leg_to_zero() -> None:
    c = constrain_pair_targets(
        desired_underlying_usd=-2_000,
        desired_etf_usd=1_000,
        current_underlying_usd=5_000,
        current_etf_usd=-2_000,
    )
    assert c.constrained_underlying_usd == 0
    assert c.constrained_etf_usd == 0


def test_shared_underlying_allocation_uses_etf_gross() -> None:
    alloc = allocate_shared_underlying(
        current_underlying_usd=9_000,
        etfs=["A", "B"],
        current_etf_usd={"A": -2_000, "B": -1_000},
    )
    assert alloc == pytest.approx({"A": 6_000, "B": 3_000})


def test_projection_helpers() -> None:
    gross = projected_pair_gross_after_shares(
        underlying_shares=10,
        underlying_price=100,
        etf_shares=-5,
        etf_price=200,
    )
    assert gross == pytest.approx(2_000)
    assert is_reduce_only_projection(before_gross_usd=2_001, after_gross_usd=gross)
    assert not is_reduce_only_projection(before_gross_usd=1_900, after_gross_usd=gross)
