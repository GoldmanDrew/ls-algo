from __future__ import annotations

import pytest

from scripts.bucket5_insurance_bt import production_config, reverse_solve_put_contracts


def test_production_b_doubles_prior_put_budget() -> None:
    cfg = production_config()
    assert sum(r.per_roll_frac for r in cfg.rungs) == pytest.approx(0.024)
    assert [r.quantity_multiplier for r in cfg.rungs] == [2, 2, 2]


def test_reverse_solver_respects_budget_and_integer_contracts() -> None:
    cfg = production_config()
    out = reverse_solve_put_contracts(
        equity_usd=1_200_000,
        spx_spot=6_000,
        atm_iv=0.20,
        rungs=cfg.rungs,
        hedge_budget=None,
        ratio=0.90,
        vix=20.0,
    )
    assert out["target_total_budget_usd"] == pytest.approx(57_600)
    assert out["target_total_contracts"] == 2 * out["baseline_total_contracts"]
    for row in out["rungs"]:
        unit = row["modeled_put_price"] * row["contract_multiplier"]
        assert isinstance(row["target_contracts"], int)
        assert row["target_contracts"] == 2 * row["baseline_contracts"]
        assert row["premium_used_usd"] == pytest.approx(row["target_contracts"] * unit)
