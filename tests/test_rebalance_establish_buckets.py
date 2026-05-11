"""Tests for ``rebalance_strategy.build_establish_trades`` (Stage B).

Stage B contract:

* Output is a list of per-Underlying buckets, NOT a flat list of pairs.
* Each bucket carries:
    - ``underlying``    : str
    - ``net_long_usd``  : float (signed sum of long_usd; B1 +ve, B4 -ve)
    - ``etf_legs``      : list of {etf, short_usd, long_usd}
* Multiple plan rows for the same Underlying are merged into ONE bucket
  (no more ``seen_under`` short-circuit dropping the second row).
* B4 rows (``long_usd < 0``) qualify (the legacy
  ``long_usd <= 0 and short_usd <= 0`` filter was a bug — B4 has both
  values negative).
* Per-ETF and per-Underlying near-zero gates still apply.
* Purgatory + flow ETFs are excluded.

These tests exercise pure logic only — no IBKR / order routing.
"""

from __future__ import annotations

import pandas as pd
import pytest

from rebalance_strategy import build_establish_trades


def _plan(rows):
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Output shape
# ---------------------------------------------------------------------------

class TestBucketShape:

    def test_empty_plan_returns_empty_list(self):
        out = build_establish_trades(
            plan=pd.DataFrame(columns=["Underlying", "ETF", "long_usd", "short_usd"]),
            strat_pos={},
            prices={},
            purgatory_etfs=set(),
            flow_etfs=set(),
        )
        assert out == []

    def test_single_b1_pair_yields_one_bucket_one_leg(self):
        plan = _plan([
            {"Underlying": "AAPL", "ETF": "AAPU",
             "long_usd": 10_000, "short_usd": -5_000},
        ])
        out = build_establish_trades(
            plan=plan,
            strat_pos={"AAPL": 0, "AAPU": 0},
            prices={"AAPL": 100.0, "AAPU": 100.0},
            purgatory_etfs=set(), flow_etfs=set(),
        )
        assert len(out) == 1
        b = out[0]
        assert b["underlying"]   == "AAPL"
        assert b["net_long_usd"] == 10_000
        assert len(b["etf_legs"]) == 1
        assert b["etf_legs"][0]["etf"]       == "AAPU"
        assert b["etf_legs"][0]["short_usd"] == -5_000
        assert b["etf_legs"][0]["long_usd"]  == 10_000


# ---------------------------------------------------------------------------
# B1 + B4 same-underlying coexistence (the core Stage B fix)
# ---------------------------------------------------------------------------

class TestB1B4Coexistence:

    def test_b1_and_b4_same_underlying_merge_into_one_bucket(self):
        # B1 long $10k AAPL / short $5k AAPU
        # B4 short $4k AAPL / short $2k AAPS
        plan = _plan([
            {"Underlying": "AAPL", "ETF": "AAPU",
             "long_usd":  10_000, "short_usd": -5_000},
            {"Underlying": "AAPL", "ETF": "AAPS",
             "long_usd":  -4_000, "short_usd": -2_000},
        ])
        out = build_establish_trades(
            plan=plan,
            strat_pos={"AAPL": 0, "AAPU": 0, "AAPS": 0},
            prices={"AAPL": 100.0, "AAPU": 100.0, "AAPS": 100.0},
            purgatory_etfs=set(), flow_etfs=set(),
        )
        # Exactly ONE bucket for AAPL — legacy seen_under would have
        # produced one row only and dropped the other.
        assert len(out) == 1
        b = out[0]
        assert b["underlying"] == "AAPL"
        # Signed-net: 10_000 + (-4_000) = 6_000 long.
        assert b["net_long_usd"] == 6_000
        # Both ETF legs preserved.
        etfs = sorted(l["etf"] for l in b["etf_legs"])
        assert etfs == ["AAPS", "AAPU"]

    def test_b4_only_pair_qualifies(self):
        # Pre-Stage-B, B4 rows were silently dropped because
        # ``long_usd <= 0 and short_usd <= 0`` matched both negatives.
        plan = _plan([
            {"Underlying": "AAPL", "ETF": "AAPS",
             "long_usd": -10_000, "short_usd": -5_000},
        ])
        out = build_establish_trades(
            plan=plan,
            strat_pos={"AAPL": 0, "AAPS": 0},
            prices={"AAPL": 100.0, "AAPS": 100.0},
            purgatory_etfs=set(), flow_etfs=set(),
        )
        assert len(out) == 1
        b = out[0]
        assert b["underlying"]   == "AAPL"
        assert b["net_long_usd"] == -10_000   # signed: net SHORT target
        assert len(b["etf_legs"]) == 1

    def test_b1_b4_perfectly_offsetting_still_emits_etf_legs(self):
        # If B1 + B4 cancel on the underlying, we still want the ETF
        # SHORT legs to be opened.
        plan = _plan([
            {"Underlying": "AAPL", "ETF": "AAPU",
             "long_usd":  5_000, "short_usd": -2_500},
            {"Underlying": "AAPL", "ETF": "AAPS",
             "long_usd": -5_000, "short_usd": -2_500},
        ])
        out = build_establish_trades(
            plan=plan,
            strat_pos={"AAPL": 0, "AAPU": 0, "AAPS": 0},
            prices={"AAPL": 100.0, "AAPU": 100.0, "AAPS": 100.0},
            purgatory_etfs=set(), flow_etfs=set(),
        )
        assert len(out) == 1
        b = out[0]
        # Net underlying = 0 -> worker will skip the underlying leg, but
        # the bucket still carries the two ETF SHORT targets.
        assert b["net_long_usd"] == 0
        assert len(b["etf_legs"]) == 2

    def test_two_b1_letfs_same_underlying_merge(self):
        # Even pre-Stage-B users could have multiple LETFs on one
        # underlying (e.g. NVDL + NVDX). Make sure they all get their
        # legs opened in one bucket.
        plan = _plan([
            {"Underlying": "NVDA", "ETF": "NVDL",
             "long_usd": 4_000, "short_usd": -2_000},
            {"Underlying": "NVDA", "ETF": "NVDX",
             "long_usd": 6_000, "short_usd": -3_000},
        ])
        out = build_establish_trades(
            plan=plan,
            strat_pos={"NVDA": 0, "NVDL": 0, "NVDX": 0},
            prices={"NVDA": 100.0, "NVDL": 100.0, "NVDX": 100.0},
            purgatory_etfs=set(), flow_etfs=set(),
        )
        assert len(out) == 1
        b = out[0]
        assert b["net_long_usd"] == 10_000
        etfs = sorted(l["etf"] for l in b["etf_legs"])
        assert etfs == ["NVDL", "NVDX"]


# ---------------------------------------------------------------------------
# Eligibility gates
# ---------------------------------------------------------------------------

class TestEligibilityGates:

    def test_purgatory_etf_excluded(self):
        plan = _plan([
            {"Underlying": "AAPL", "ETF": "AAPU",
             "long_usd": 10_000, "short_usd": -5_000},
        ])
        out = build_establish_trades(
            plan=plan,
            strat_pos={"AAPL": 0, "AAPU": 0},
            prices={"AAPL": 100.0, "AAPU": 100.0},
            purgatory_etfs={"AAPU"}, flow_etfs=set(),
        )
        assert out == []

    def test_flow_etf_excluded(self):
        plan = _plan([
            {"Underlying": "AAPL", "ETF": "AAPU",
             "long_usd": 10_000, "short_usd": -5_000},
        ])
        out = build_establish_trades(
            plan=plan,
            strat_pos={"AAPL": 0, "AAPU": 0},
            prices={"AAPL": 100.0, "AAPU": 100.0},
            purgatory_etfs=set(), flow_etfs={"AAPU"},
        )
        assert out == []

    def test_b1_skipped_when_b4_purgatory_but_b4_skipped(self):
        # If only B4's ETF is in purgatory, B1 should still establish.
        plan = _plan([
            {"Underlying": "AAPL", "ETF": "AAPU",
             "long_usd": 10_000, "short_usd": -5_000},
            {"Underlying": "AAPL", "ETF": "AAPS",
             "long_usd":  -4_000, "short_usd": -2_000},
        ])
        out = build_establish_trades(
            plan=plan,
            strat_pos={"AAPL": 0, "AAPU": 0, "AAPS": 0},
            prices={"AAPL": 100.0, "AAPU": 100.0, "AAPS": 100.0},
            purgatory_etfs={"AAPS"}, flow_etfs=set(),
        )
        assert len(out) == 1
        b = out[0]
        # B4 row excluded -> net_long_usd is just B1's contribution.
        assert b["net_long_usd"] == 10_000
        assert {l["etf"] for l in b["etf_legs"]} == {"AAPU"}

    def test_zero_zero_row_skipped(self):
        # Real "zero" rows (no targets at all) — nothing to do.
        plan = _plan([
            {"Underlying": "AAPL", "ETF": "AAPU",
             "long_usd": 0, "short_usd": 0},
        ])
        out = build_establish_trades(
            plan=plan,
            strat_pos={"AAPL": 0, "AAPU": 0},
            prices={"AAPL": 100.0, "AAPU": 100.0},
            purgatory_etfs=set(), flow_etfs=set(),
        )
        assert out == []

    def test_etf_not_near_zero_skips_pair(self):
        # ETF leg already has $5k of position; threshold $100 -> skip.
        plan = _plan([
            {"Underlying": "AAPL", "ETF": "AAPU",
             "long_usd": 10_000, "short_usd": -5_000},
        ])
        out = build_establish_trades(
            plan=plan,
            strat_pos={"AAPL": 0, "AAPU": -50},  # AAPU at -$5k
            prices={"AAPL": 100.0, "AAPU": 100.0},
            purgatory_etfs=set(), flow_etfs=set(),
            establish_threshold_usd=100.0,
        )
        assert out == []

    def test_underlying_not_near_zero_skips_all_pairs(self):
        # Underlying already has $5k position -> no establish for ANY
        # pair on this Underlying (Phase 2b owns it).
        plan = _plan([
            {"Underlying": "AAPL", "ETF": "AAPU",
             "long_usd": 10_000, "short_usd": -5_000},
            {"Underlying": "AAPL", "ETF": "AAPS",
             "long_usd":  -4_000, "short_usd": -2_000},
        ])
        out = build_establish_trades(
            plan=plan,
            strat_pos={"AAPL": 50, "AAPU": 0, "AAPS": 0},
            prices={"AAPL": 100.0, "AAPU": 100.0, "AAPS": 100.0},
            purgatory_etfs=set(), flow_etfs=set(),
            establish_threshold_usd=100.0,
        )
        assert out == []

    def test_threshold_boundary_inclusive_below(self):
        # cur USD just under threshold -> still establish-eligible.
        plan = _plan([
            {"Underlying": "AAPL", "ETF": "AAPU",
             "long_usd": 10_000, "short_usd": -5_000},
        ])
        out = build_establish_trades(
            plan=plan,
            strat_pos={"AAPL": 0, "AAPU": 0.5},   # AAPU at $50, < $100
            prices={"AAPL": 100.0, "AAPU": 100.0},
            purgatory_etfs=set(), flow_etfs=set(),
            establish_threshold_usd=100.0,
        )
        assert len(out) == 1


# ---------------------------------------------------------------------------
# Multiple underlyings
# ---------------------------------------------------------------------------

class TestMultipleUnderlyings:

    def test_two_underlyings_two_buckets(self):
        plan = _plan([
            {"Underlying": "AAPL", "ETF": "AAPU",
             "long_usd": 10_000, "short_usd": -5_000},
            {"Underlying": "MSFT", "ETF": "MSFU",
             "long_usd":  8_000, "short_usd": -4_000},
        ])
        out = build_establish_trades(
            plan=plan,
            strat_pos={"AAPL": 0, "AAPU": 0, "MSFT": 0, "MSFU": 0},
            prices={"AAPL": 100.0, "AAPU": 100.0,
                    "MSFT": 100.0, "MSFU": 100.0},
            purgatory_etfs=set(), flow_etfs=set(),
        )
        assert len(out) == 2
        unders = {b["underlying"] for b in out}
        assert unders == {"AAPL", "MSFT"}

    def test_one_underlying_eligible_one_not(self):
        plan = _plan([
            {"Underlying": "AAPL", "ETF": "AAPU",
             "long_usd": 10_000, "short_usd": -5_000},
            {"Underlying": "MSFT", "ETF": "MSFU",
             "long_usd":  8_000, "short_usd": -4_000},
        ])
        out = build_establish_trades(
            plan=plan,
            strat_pos={"AAPL": 0, "AAPU": 0,
                       "MSFT": 50, "MSFU": 0},     # MSFT non-zero
            prices={"AAPL": 100.0, "AAPU": 100.0,
                    "MSFT": 100.0, "MSFU": 100.0},
            purgatory_etfs=set(), flow_etfs=set(),
            establish_threshold_usd=100.0,
        )
        assert len(out) == 1
        assert out[0]["underlying"] == "AAPL"


# ---------------------------------------------------------------------------
# Symbol normalization
# ---------------------------------------------------------------------------

class TestSymbolNormalization:

    def test_lowercase_symbols_normalised(self):
        plan = _plan([
            {"Underlying": "aapl", "ETF": "aapu",
             "long_usd": 10_000, "short_usd": -5_000},
        ])
        out = build_establish_trades(
            plan=plan,
            strat_pos={"AAPL": 0, "AAPU": 0},
            prices={"AAPL": 100.0, "AAPU": 100.0},
            purgatory_etfs=set(), flow_etfs=set(),
        )
        assert len(out) == 1
        assert out[0]["underlying"] == "AAPL"
        assert out[0]["etf_legs"][0]["etf"] == "AAPU"
