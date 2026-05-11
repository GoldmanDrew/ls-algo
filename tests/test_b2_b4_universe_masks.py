"""Tests for ``_b2_b4_universe_masks``: B2/B4 eligibility universe + flow exclusion.

The user-facing rule (``config/strategy_config.yml``):

* B2 ("yieldboost") = every screener row tagged ``is_yieldboost == True``
  UNION the YAML whitelist allow-list (additive override) - minus any ticker
  in ``flow_program.universe.shorts``.
* B4 (inverse decay) = every inverse, shortable row in the screener - minus
  any ticker in ``flow_program.universe.shorts`` (and minus per-name overrides
  like ``KOLD`` enforced separately by the caller).

These tests pin down the universe step alone; downstream mask combinations
(``positive_beta``, borrow caps, vol/edge gates, B4-specific exclusions) are
applied by the caller and tested elsewhere.
"""

from __future__ import annotations

import pandas as pd

from generate_trade_plan import _b2_b4_universe_masks


def _screener() -> pd.DataFrame:
    # Mix of: yieldboost names, an off-list research name, and a flow-program ETF
    # that also happens to be tagged ``is_yieldboost`` (e.g. TQQY / TSYY / NVYY /
    # AZYY / PLYY all sit in both lists today).
    return pd.DataFrame(
        {
            "ETF": [
                "MTYY",
                "AMYY",
                "TQQY",
                "AZYY",
                "CORD",
                "CORE1",
                "SDS",
                "DUST",
                "SQQQ",
                "QID",
                "SVXY",
            ],
            "is_yieldboost": [
                True, True, True, True,
                False, False, False, False, False, False, False,
            ],
        }
    )


WL_SET = {"MTYY", "CORD", "ASTN", "UVIX"}
FLOW_SET = {"SDS", "DUST", "SQQQ", "TQQY", "AZYY"}


def _idx_for(df: pd.DataFrame, sym: str) -> int:
    return df.index[df["ETF"] == sym][0]


def test_yieldboost_alone_promotes_to_b2_universe() -> None:
    df = _screener()
    is_yb, in_b2, _ = _b2_b4_universe_masks(
        df, wl_set=WL_SET, flow_program_etfs=FLOW_SET
    )
    # AMYY is yieldboost but not in the YAML whitelist.
    assert bool(in_b2.loc[_idx_for(df, "AMYY")]) is True
    # Pure core ETFs without YB tag stay outside B2.
    assert bool(in_b2.loc[_idx_for(df, "CORE1")]) is False
    assert bool(is_yb.loc[_idx_for(df, "AMYY")]) is True


def test_yaml_whitelist_remains_additive_override() -> None:
    df = _screener()
    _, in_b2, _ = _b2_b4_universe_masks(df, wl_set=WL_SET, flow_program_etfs=FLOW_SET)
    # CORD is NOT yieldboost, but the YAML whitelist still admits it.
    assert bool(in_b2.loc[_idx_for(df, "CORD")]) is True


def test_flow_program_tickers_get_marked_for_exclusion() -> None:
    df = _screener()
    _, in_b2, in_flow = _b2_b4_universe_masks(
        df, wl_set=WL_SET, flow_program_etfs=FLOW_SET
    )
    # TQQY and AZYY: in YB AND in flow; in_b2_universe is True (they ARE yieldboost),
    # but in_flow_program is True too - the caller ANDs ``~in_flow_program`` so they
    # won't actually be sized in B2. Pin both halves of that contract.
    for sym in ("TQQY", "AZYY"):
        idx = _idx_for(df, sym)
        assert bool(in_b2.loc[idx]) is True
        assert bool(in_flow.loc[idx]) is True
    # Pure flow-only inverse tickers: in_flow True, in_b2 False (not YB, not YAML).
    for sym in ("SDS", "DUST", "SQQQ"):
        idx = _idx_for(df, sym)
        assert bool(in_flow.loc[idx]) is True
        assert bool(in_b2.loc[idx]) is False


def test_non_flow_inverse_is_left_for_b4() -> None:
    df = _screener()
    _, _, in_flow = _b2_b4_universe_masks(
        df, wl_set=WL_SET, flow_program_etfs=FLOW_SET
    )
    # QID and SVXY are inverse-style but not in the flow set - caller then layers
    # ``negative_beta & inverse_shortable & ~in_flow_program`` to arrive at B4.
    assert bool(in_flow.loc[_idx_for(df, "QID")]) is False
    assert bool(in_flow.loc[_idx_for(df, "SVXY")]) is False


def test_missing_is_yieldboost_column_falls_back_to_yaml() -> None:
    df = pd.DataFrame({"ETF": ["MTYY", "AMYY", "FOO"]})
    is_yb, in_b2, in_flow = _b2_b4_universe_masks(
        df, wl_set={"MTYY"}, flow_program_etfs=set()
    )
    # No ``is_yieldboost`` column -> all is_yb False; B2 universe collapses to YAML whitelist.
    assert bool(is_yb.any()) is False
    assert bool(in_b2.loc[_idx_for(df, "MTYY")]) is True
    assert bool(in_b2.loc[_idx_for(df, "AMYY")]) is False
    assert bool(in_b2.loc[_idx_for(df, "FOO")]) is False
    assert bool(in_flow.any()) is False
