"""Contracts that keep purgatory inverse pairs inside the B4 cadence gate."""
from __future__ import annotations

import pandas as pd
import pytest

from generate_trade_plan import _execution_sleeves_for_screened
from rebalance_strategy import load_plan
from scripts.b4_plan_contract import validate_purgatory_inverse_sleeves


def test_execution_sleeve_classification_survives_purgatory() -> None:
    screened = pd.DataFrame(
        [
            {"ETF": "ASTN", "Underlying": "ASTS", "Delta": -1.0, "purgatory": True},
            {"ETF": "IREZ", "Underlying": "IREN", "Delta": -1.0, "purgatory": True},
            {"ETF": "MUZ", "Underlying": "MU", "Delta": -1.0, "purgatory": True},
            {"ETF": "SPCG", "Underlying": "SPCX", "Delta": -1.0, "purgatory": True},
            {
                "ETF": "UVIX",
                "Underlying": "SVIX",
                "Delta": -1.0,
                "Delta_product_class": "volatility_etp",
                "purgatory": True,
            },
            {
                "ETF": "COYY",
                "Underlying": "COIN",
                "Delta": 1.0,
                "is_yieldboost": True,
                "purgatory": True,
            },
        ]
    )
    sleeves = _execution_sleeves_for_screened(
        screened,
        flow_program_etfs=set(),
    )
    assert set(sleeves.iloc[:4]) == {"inverse_decay_bucket4"}
    assert sleeves.iloc[4] == "volatility_etp_bucket5"
    assert sleeves.iloc[5] == "yieldboost"


def test_plan_contract_rejects_blank_purgatory_inverse_sleeve() -> None:
    plan = pd.DataFrame(
        [
            {
                "ETF": "ASTN",
                "Underlying": "ASTS",
                "Delta": -1.0,
                "purgatory": True,
                "sleeve": "",
            }
        ]
    )
    with pytest.raises(RuntimeError, match="ASTN/ASTS"):
        validate_purgatory_inverse_sleeves(plan)


def test_load_plan_fails_closed_on_blank_purgatory_inverse_sleeve(tmp_path) -> None:
    path = tmp_path / "proposed_trades.csv"
    pd.DataFrame(
        [
            {
                "strategy_tag": "ls",
                "ETF": "ASTN",
                "Underlying": "ASTS",
                "Delta": -1.0,
                "purgatory": True,
                "sleeve": "",
                "long_usd": 0.0,
                "short_usd": 0.0,
                "execution_policy": "reduce_only",
            }
        ]
    ).to_csv(path, index=False)

    with pytest.raises(ValueError, match="Regenerate proposed_trades"):
        load_plan(path, "ls", flow_etfs=set())
