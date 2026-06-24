"""Tests for ``rebalance_strategy.load_plan``.

Stage A guarantees:

* ``load_plan`` returns a 3-tuple ``(plan_df, hedgeable_df, resize_df)``.
* ``hedgeable_df`` keeps its B1+YB-only filter (Phase 3 ``target_gross``
  math is sleeve-scoped and intentionally excludes B4).
* ``resize_df`` includes **all sleeves** (B1 + YB + B4), non-purgatory,
  non-flow — so Phase 2b can sum signed ``long_usd`` per Underlying and
  net B1 vs B4 positions on the same ticker.

These tests guard the contract surface; the netting math itself is
covered in ``test_phase2b_resize.py::TestB1B4Netting``.
"""

from __future__ import annotations

import pandas as pd
import pytest

from rebalance_strategy import load_plan


@pytest.fixture
def plan_csv(tmp_path):
    """A minimal proposed_trades.csv covering all four sleeves of interest."""
    rows = [
        # B1 (core_leveraged): long underlying / short LETF
        {"strategy_tag": "ls", "sleeve": "core_leveraged",
         "Underlying": "AAPL", "ETF": "AAPU",
         "long_usd": 10_000, "short_usd": -5_000, "purgatory": False},
        # YieldBoost (yieldboost): same shape as B1 for hedge math
        {"strategy_tag": "ls", "sleeve": "yieldboost",
         "Underlying": "TSLA", "ETF": "YBT",
         "long_usd":  4_000, "short_usd": -2_000, "purgatory": False},
        # B4 (inverse_decay_bucket4): short underlying / short inverse ETF
        {"strategy_tag": "ls", "sleeve": "inverse_decay_bucket4",
         "Underlying": "AAPL", "ETF": "AAPS",
         "long_usd": -3_000, "short_usd": -1_500, "purgatory": False},
        # Bucket 5 VOL ETP: same inverse-decay execution shape, separate sizing sleeve.
        {"strategy_tag": "ls", "sleeve": "volatility_etp_bucket5",
         "Underlying": "VIX", "ETF": "UVIX",
         "long_usd": -250, "short_usd": -250, "purgatory": False},
        # Purgatory row — must be excluded from BOTH hedgeable and resize.
        {"strategy_tag": "ls", "sleeve": "core_leveraged",
         "Underlying": "MSFT", "ETF": "MSFU",
         "long_usd":  2_000, "short_usd": -1_000, "purgatory": True},
        # Flow-program row — excluded from hedgeable AND resize via flow_etfs.
        {"strategy_tag": "ls", "sleeve": "flow_program",
         "Underlying": "GOOG", "ETF": "GOOGS",
         "long_usd":      0, "short_usd": -3_000, "purgatory": False},
        # Other strategy_tag — must be filtered out entirely.
        {"strategy_tag": "other", "sleeve": "core_leveraged",
         "Underlying": "NVDA", "ETF": "NVDL",
         "long_usd": 50_000, "short_usd": -25_000, "purgatory": False},
    ]
    df = pd.DataFrame(rows)
    p = tmp_path / "proposed_trades.csv"
    df.to_csv(p, index=False)
    return p


class TestLoadPlanContract:

    def test_returns_three_dataframes(self, plan_csv):
        plan, hedgeable, resize = load_plan(plan_csv, "ls", flow_etfs={"GOOGS"})
        assert isinstance(plan,      pd.DataFrame)
        assert isinstance(hedgeable, pd.DataFrame)
        assert isinstance(resize,    pd.DataFrame)

    def test_strategy_tag_filter_applied(self, plan_csv):
        plan, _, _ = load_plan(plan_csv, "ls", flow_etfs={"GOOGS"})
        # NVDA row has strategy_tag="other"; should be dropped.
        assert "NVDA" not in set(plan["Underlying"])

    def test_hedgeable_only_b1_yb_non_purgatory_non_flow(self, plan_csv):
        _, hedgeable, _ = load_plan(plan_csv, "ls", flow_etfs={"GOOGS"})
        sleeves = set(hedgeable["sleeve"])
        assert sleeves == {"core_leveraged", "yieldboost"}, (
            f"hedgeable_df must be B1+YB only (no B4); got {sleeves}"
        )
        # Purgatory row excluded.
        assert "MSFT" not in set(hedgeable["Underlying"])
        # Flow ETF excluded.
        assert "GOOGS" not in set(hedgeable["ETF"])
        # B4 excluded (Phase 3 hedge math is sleeve-scoped).
        assert "AAPS" not in set(hedgeable["ETF"])

    def test_resize_includes_b4_alongside_b1_yb(self, plan_csv):
        # Stage A core invariant: Phase 2b sees B4 rows.
        _, _, resize = load_plan(plan_csv, "ls", flow_etfs={"GOOGS"})
        sleeves = set(resize["sleeve"])
        assert sleeves == {
            "core_leveraged",
            "yieldboost",
            "inverse_decay_bucket4",
            "volatility_etp_bucket5",
        }, (
            f"resize_df must include all three live sleeves; got {sleeves}"
        )

    def test_resize_excludes_purgatory_and_flow(self, plan_csv):
        _, _, resize = load_plan(plan_csv, "ls", flow_etfs={"GOOGS"})
        # Purgatory row excluded.
        assert "MSFT" not in set(resize["Underlying"])
        # Flow ETF excluded.
        assert "GOOGS" not in set(resize["ETF"])

    def test_b1_and_b4_same_underlying_both_present_in_resize(self, plan_csv):
        # The whole point of Stage A: B1 long-AAPL and B4 short-AAPL must
        # coexist in the resize plan so build_resize_trades can sum them.
        _, _, resize = load_plan(plan_csv, "ls", flow_etfs={"GOOGS"})
        aapl_rows = resize[resize["Underlying"] == "AAPL"]
        sleeves   = set(aapl_rows["sleeve"])
        assert sleeves == {"core_leveraged", "inverse_decay_bucket4"}, (
            "Same-underlying B1+B4 rows must both survive into resize_df"
        )
        # Signed sum of long_usd over AAPL = 10_000 + (-3_000) = 7_000.
        assert float(aapl_rows["long_usd"].sum()) == pytest.approx(7_000.0)

    def test_empty_plan_returns_three_empties(self, tmp_path):
        p = tmp_path / "empty.csv"
        # Header-only CSV (no rows for our strategy_tag).
        pd.DataFrame(columns=[
            "strategy_tag", "sleeve", "Underlying", "ETF",
            "long_usd", "short_usd", "purgatory",
        ]).to_csv(p, index=False)
        plan, hedgeable, resize = load_plan(p, "ls", flow_etfs=set())
        assert plan.empty and hedgeable.empty and resize.empty

    def test_missing_plan_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_plan(tmp_path / "nope.csv", "ls", flow_etfs=set())
