"""Regression tests for the refined product_class taxonomy.

The taxonomy in ``screener_v2_fields._product_class`` distinguishes:
  * ``income_yieldboost`` — weekly 95/88 put-spread on 2× ETF.
  * ``passive_low_beta``  — Bucket-2 low-β fund without an income overlay.
  * ``letf``              — standard 2×/3× LETF (β > 1.5).
  * ``inverse``           — β < 0.
  * ``income_put_spread`` — legacy 1× covered-call sleeve.
  * ``other_structured``  — fallback.

Dashboard semantics:
  * ``passive_low_beta`` → ``expected_decay_available = False`` so the screener
    will null out the expected/distributional decay columns and the dashboard
    falls back to the realized measure (renders "—" for Exp. decay).
  * ``income_yieldboost`` → ``expected_decay_available = True`` so the
    dashboard can substitute the put-spread NAV-decay model in the headline
    Exp. decay cell.
"""

from __future__ import annotations

import pandas as pd

from screener_v2_fields import (
    _expected_decay_available,
    _product_class,
    enrich_screener_v2_fields,
)


def test_product_class_yieldboost_wins_over_beta() -> None:
    # is_yieldboost=True forces ``income_yieldboost`` regardless of β/leverage.
    assert _product_class(2.0, 1.97, is_yieldboost=True) == "income_yieldboost"
    assert _product_class(1.0, 1.0, is_yieldboost=True) == "income_yieldboost"


def test_product_class_passive_low_beta_window() -> None:
    # 0 < β ≤ 1.5 with no yield-boost overlay → passive_low_beta.
    assert _product_class(None, 1.0) == "passive_low_beta"
    assert _product_class(None, 0.5) == "passive_low_beta"
    assert _product_class(None, 1.5) == "passive_low_beta"


def test_product_class_high_beta_letf() -> None:
    # β > 1.5 → standard LETF.
    assert _product_class(2.0, 1.95) == "letf"
    assert _product_class(3.0, 2.6) == "letf"


def test_product_class_inverse_when_negative_beta() -> None:
    assert _product_class(-2.0, -1.97) == "inverse"
    assert _product_class(None, -0.5) == "inverse"


def test_product_class_falls_back_when_beta_missing() -> None:
    # Leverage-only fallback path.
    assert _product_class(1.0, None) == "income_put_spread"
    assert _product_class(2.0, None) == "letf"
    assert _product_class(None, None) == "other_structured"


def test_expected_decay_available_policy() -> None:
    # The dashboard router uses this flag to decide whether to render "—".
    assert _expected_decay_available("letf") is True
    assert _expected_decay_available("inverse") is True
    assert _expected_decay_available("income_yieldboost") is True
    assert _expected_decay_available("income_put_spread") is True
    assert _expected_decay_available("volatility_etp") is True
    # Passive low-β: realized-only, so model-based expected decay is N/A.
    assert _expected_decay_available("passive_low_beta") is False
    assert _expected_decay_available("other_structured") is False


def test_enrich_emits_expected_decay_available_column() -> None:
    df = pd.DataFrame(
        [
            {  # passive low-β (β=1.0, no yield-boost)
                "ETF": "PASS", "Underlying": "QQQ", "Beta": 1.0,
                "Beta_n_obs": 0, "Leverage": 1.0,
                "borrow_current": 0.01, "is_yieldboost": False,
            },
            {  # YieldBOOST income strategy
                "ETF": "TQQY", "Underlying": "QQQ", "Beta": 0.95,
                "Beta_n_obs": 0, "Leverage": 2.0,
                "borrow_current": 0.05, "is_yieldboost": True,
            },
            {  # standard 2× LETF
                "ETF": "TQQQ", "Underlying": "QQQ", "Beta": 2.97,
                "Beta_n_obs": 0, "Leverage": 3.0,
                "borrow_current": 0.005, "is_yieldboost": False,
            },
            {  # 2× inverse
                "ETF": "SQQQ", "Underlying": "QQQ", "Beta": -2.97,
                "Beta_n_obs": 0, "Leverage": -3.0,
                "borrow_current": 0.012, "is_yieldboost": False,
            },
        ]
    )
    out = enrich_screener_v2_fields(df, tr_map={})
    classes = dict(zip(out["ETF"], out["product_class"]))
    avail = dict(zip(out["ETF"], out["expected_decay_available"]))
    assert classes["PASS"] == "passive_low_beta"
    assert classes["TQQY"] == "income_yieldboost"
    assert classes["TQQQ"] == "letf"
    assert classes["SQQQ"] == "inverse"
    assert avail["PASS"] is False
    assert avail["TQQY"] is True
    assert avail["TQQQ"] is True
    assert avail["SQQQ"] is True
