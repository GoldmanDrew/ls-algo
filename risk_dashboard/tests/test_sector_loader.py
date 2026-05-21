"""Tier-by-tier coverage for ``risk_dashboard.sector_loader.resolve_sector``."""

from __future__ import annotations

import pytest

from risk_dashboard.sector_loader import (
    HEURISTIC_PATTERNS,
    TIER_CONFIDENCE,
    batch_resolve,
    resolve_sector,
)


# ---------------------------------------------------------------------------
# Tier 1: hand-curated thematic override
# ---------------------------------------------------------------------------


def test_override_tier_wins_first():
    """QBTS is in OVERRIDE_SECTOR_MAP; nothing else should be consulted."""
    out = resolve_sector(
        "QBTS",
        screener_row={"sector": "noise-should-not-win"},
        vendor_info={"sector": "Technology", "industry": "Software"},
        use_override=True,
    )
    assert out["sector"] == "quantum"
    assert out["sector_source"] == "override"
    assert out["sector_confidence"] == TIER_CONFIDENCE["override"]


def test_override_handles_case_and_whitespace():
    out = resolve_sector(" qbts ", use_override=True)
    assert out["sector"] == "quantum"
    assert out["sector_source"] == "override"


# ---------------------------------------------------------------------------
# Tier 2: screener metadata
# ---------------------------------------------------------------------------


def test_override_off_by_default():
    out = resolve_sector("QBTS")
    assert out["sector_source"] != "override"
    assert out["sector"] == "other"
    out = resolve_sector(
        "NOT_IN_ANY_MAP",
        screener_row={"theme": "ai-infrastructure"},
    )
    assert out["sector"] == "ai-infrastructure"
    assert out["sector_source"] == "screener"


def test_screener_tier_prefers_underlying_sector_column_first():
    out = resolve_sector(
        "NEW_NAME",
        screener_row={"underlying_sector": "biotech", "theme": "should-not-win"},
    )
    assert out["sector"] == "biotech"
    assert out["sector_source"] == "screener"


def test_screener_tier_skips_placeholder_values():
    out = resolve_sector(
        "NEW_NAME",
        screener_row={"sector": "nan", "theme": "unknown"},
        vendor_info={"sector": "Technology"},
    )
    assert out["sector_source"] == "vendor"


# ---------------------------------------------------------------------------
# Tier 3: vendor sector/industry mapping
# ---------------------------------------------------------------------------


def test_vendor_industry_beats_vendor_sector():
    out = resolve_sector(
        "FAKE_SEMI",
        vendor_info={"sector": "Technology", "industry": "Semiconductors"},
    )
    assert out["sector"] == "semis"
    assert out["sector_source"] == "vendor"


def test_vendor_sector_used_when_industry_missing():
    out = resolve_sector(
        "FAKE_BIOTECH",
        vendor_info={"sector": "Healthcare"},
    )
    assert out["sector"] == "healthcare"
    assert out["sector_source"] == "vendor"


def test_vendor_unknown_sector_falls_through_to_heuristic():
    out = resolve_sector(
        "FAKE_DRONE",
        vendor_info={
            "sector": "Unknown Industry",
            "industry": "Unknown Industry",
            "longName": "Made Up Drone Holdings Inc",
        },
    )
    assert out["sector"] == "drones"
    assert out["sector_source"] == "heuristic"


# ---------------------------------------------------------------------------
# Tier 4: heuristic regex set
# ---------------------------------------------------------------------------


_HEURISTIC_CASES: list[tuple[str, str]] = [
    ("Acme Quantum Computing Inc", "quantum"),
    ("Pure Uranium Corp", "nuclear"),
    ("Small Modular Reactor Holdings", "nuclear"),
    ("New Bitcoin Strategy ETF", "crypto"),
    ("Bitcoin Miner Co", "crypto-equity"),
    ("Tactical Drone Defense", "drones"),
    ("Vertical eVTOL Aircraft Holdings", "evtol"),
    ("Orbital Launch Vehicle Corp", "space"),
    ("Lithium Battery Mining Co", "metals"),
    ("Hydrogen Fuel Cell Industries", "clean"),
    ("Neobank Holdings", "fintech"),
    ("Generic Software SaaS Inc", "software"),
    ("Biotech Therapeutics", "healthcare"),
    ("China Internet ETF", "china"),
]


@pytest.mark.parametrize("long_name,expected_sector", _HEURISTIC_CASES)
def test_heuristic_regex_set(long_name: str, expected_sector: str):
    out = resolve_sector(
        "FAKE_TICKER_" + expected_sector.upper(),
        vendor_info={"longName": long_name, "industry": "Unknown"},
    )
    assert out["sector"] == expected_sector, (
        f"longName={long_name!r} expected {expected_sector!r}, got {out['sector']!r}"
    )
    assert out["sector_source"] == "heuristic"


def test_every_heuristic_sector_has_at_least_one_pattern():
    for sector, patterns in HEURISTIC_PATTERNS.items():
        assert patterns, f"sector {sector!r} declared with no regex patterns"


# ---------------------------------------------------------------------------
# Tier 5: default
# ---------------------------------------------------------------------------


def test_unknown_symbol_with_no_metadata_falls_to_default():
    out = resolve_sector("ZZZZZ_NOT_REAL_42")
    assert out["sector"] == "other"
    assert out["sector_source"] == "default"
    assert out["sector_confidence"] == TIER_CONFIDENCE["default"]


def test_blank_symbol_returns_default():
    out = resolve_sector("")
    assert out["sector"] == "other"
    assert out["sector_source"] == "default"


# ---------------------------------------------------------------------------
# Batch helper
# ---------------------------------------------------------------------------


def test_batch_resolve_routes_per_symbol_inputs():
    out = batch_resolve(
        ["QBTS", "FAKE_SEMI", "ZZZ_BLANK"],
        screener_rows={"ZZZ_BLANK": {"sector": "consumer"}},
        vendor_info_by_symbol={"FAKE_SEMI": {"industry": "Semiconductors"}},
        use_override=True,
    )
    assert out["QBTS"]["sector_source"] == "override"
    assert out["FAKE_SEMI"]["sector"] == "semis"
    assert out["FAKE_SEMI"]["sector_source"] == "vendor"
    assert out["ZZZ_BLANK"]["sector"] == "consumer"
    assert out["ZZZ_BLANK"]["sector_source"] == "screener"


def test_batch_resolve_normalizes_keys_to_upper():
    out = batch_resolve(["qbts", " ionq "], use_override=True)
    assert "QBTS" in out
    assert "IONQ" in out
    assert out["QBTS"]["sector"] == "quantum"
    assert out["IONQ"]["sector"] == "quantum"
