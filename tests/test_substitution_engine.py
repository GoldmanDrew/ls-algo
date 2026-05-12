"""Tests for `substitution_engine.py`.

Covers:
* `SubstitutionConfig.from_dict` parsing & defaults
* `SubstitutionEngine.find_substitute`:
    - first-available from ordered list
    - exclusion of in-use substitutes (across active swaps)
    - screener-universe filter
    - borrow cap on short legs
    - returns None when pool absent / engine disabled
* `record_swap` persistence + idempotency
* `clear_swap`
* `active_substitutions` reconstructs from JSON state
* `maybe_swap_back` Stage 2 default no-op
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import pytest

from substitution_engine import (
    SubstitutionConfig,
    SubstitutionEngine,
    Swap,
)


# -----------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------

class TestSubstitutionConfig:

    def test_from_dict_defaults(self):
        cfg = SubstitutionConfig.from_dict({})
        assert cfg.enabled is True
        assert cfg.min_loss_usd_to_substitute == 500.0
        assert cfg.hold_substitute_days == 31
        assert cfg.underlyings == {}
        assert cfg.letfs == {}

    def test_from_dict_normalizes_symbols_uppercase(self):
        cfg = SubstitutionConfig.from_dict({
            "underlyings": {"ibit": ["fbtc", "bitb"]},
            "letfs": {"btc_2x": ["bitx", "bitu"]},
        })
        assert "IBIT" in cfg.underlyings
        assert cfg.underlyings["IBIT"] == ["FBTC", "BITB"]
        assert cfg.letfs["BTC_2X"] == ["BITX", "BITU"]

    def test_from_dict_none_yields_defaults(self):
        cfg = SubstitutionConfig.from_dict(None)
        assert cfg.enabled is True


# -----------------------------------------------------------------------
# Engine: find_substitute
# -----------------------------------------------------------------------

def _engine(tmp_path: Path, **kw) -> SubstitutionEngine:
    cfg_kw = {
        "enabled": True,
        "underlyings": {"IBIT": ["FBTC", "BITB", "ARKB"]},
        "letfs": {},
    }
    cfg_kw.update(kw.get("cfg_overrides", {}))
    cfg = SubstitutionConfig.from_dict(cfg_kw)
    return SubstitutionEngine(
        config=cfg,
        state_path=tmp_path / "active_substitutions.json",
        screener_borrow=kw.get("borrow") or {},
        screener_universe=kw.get("universe") or set(),
    )


class TestFindSubstitute:

    def test_returns_first_in_pool(self, tmp_path):
        eng = _engine(tmp_path)
        assert eng.find_substitute("IBIT") == "FBTC"

    def test_case_insensitive_lookup(self, tmp_path):
        eng = _engine(tmp_path)
        assert eng.find_substitute("ibit") == "FBTC"

    def test_returns_none_when_pool_absent(self, tmp_path):
        eng = _engine(tmp_path)
        assert eng.find_substitute("UNKNOWN") is None

    def test_returns_none_when_disabled(self, tmp_path):
        eng = _engine(tmp_path, cfg_overrides={"enabled": False})
        assert eng.find_substitute("IBIT") is None

    def test_skips_in_use_substitutes(self, tmp_path):
        # Use FBTC for IBIT, then ask for ETHA->[FBTC,FETH] -> should pick FETH
        eng = SubstitutionEngine(
            config=SubstitutionConfig.from_dict({
                "enabled": True,
                "underlyings": {
                    "IBIT": ["FBTC", "BITB"],
                    "ETHA": ["FBTC", "FETH"],
                },
            }),
            state_path=tmp_path / "active.json",
        )
        eng.record_swap(original="IBIT", substitute="FBTC", qty=100)
        # FBTC is now in-use -> ETHA should fall through to FETH
        assert eng.find_substitute("ETHA") == "FETH"

    def test_screener_universe_filter(self, tmp_path):
        # FBTC not in universe -> falls through to BITB
        eng = _engine(tmp_path, universe={"BITB", "ARKB", "IBIT"})
        assert eng.find_substitute("IBIT") == "BITB"

    def test_short_leg_borrow_cap(self, tmp_path):
        eng = SubstitutionEngine(
            config=SubstitutionConfig.from_dict({
                "enabled": True,
                "letfs": {"X": ["BITX", "BITU"]},
                "max_substitute_borrow_annual": 0.5,
            }),
            state_path=tmp_path / "a.json",
            screener_borrow={"BITX": 0.8, "BITU": 0.3},
        )
        # BITX borrow 80% > cap 50% -> falls through to BITU
        assert eng.find_substitute("X", leg="short") == "BITU"

    def test_excludes_self(self, tmp_path):
        eng = SubstitutionEngine(
            config=SubstitutionConfig.from_dict({
                "enabled": True,
                "underlyings": {"IBIT": ["IBIT", "FBTC"]},  # self-listed
            }),
            state_path=tmp_path / "a.json",
        )
        assert eng.find_substitute("IBIT") == "FBTC"


# -----------------------------------------------------------------------
# Engine: record_swap + persistence
# -----------------------------------------------------------------------

class TestRecordSwap:

    def test_record_writes_state_file(self, tmp_path):
        eng = _engine(tmp_path)
        sw = eng.record_swap(
            original="IBIT", substitute="FBTC", qty=100, leg="long",
            swap_in_date=date(2026, 5, 11),
        )
        assert isinstance(sw, Swap)
        assert sw.swap_back_due == date(2026, 5, 11) + timedelta(days=31)

        path = tmp_path / "active_substitutions.json"
        assert path.exists()
        data = json.loads(path.read_text())
        assert "IBIT" in data
        assert data["IBIT"]["substitute"] == "FBTC"
        assert data["IBIT"]["qty"] == 100

    def test_idempotent_overwrites_for_same_original(self, tmp_path):
        eng = _engine(tmp_path)
        eng.record_swap(original="IBIT", substitute="FBTC", qty=100)
        eng.record_swap(original="IBIT", substitute="BITB", qty=200)
        active = eng.active_substitutions()
        assert len(active) == 1
        assert active["IBIT"].substitute == "BITB"

    def test_no_swap_back_when_hold_zero(self, tmp_path):
        eng = SubstitutionEngine(
            config=SubstitutionConfig.from_dict({
                "enabled": True, "hold_substitute_days": 0,
                "underlyings": {"IBIT": ["FBTC"]},
            }),
            state_path=tmp_path / "a.json",
        )
        sw = eng.record_swap(original="IBIT", substitute="FBTC", qty=100)
        assert sw.swap_back_due is None

    def test_clear_swap_removes_entry(self, tmp_path):
        eng = _engine(tmp_path)
        eng.record_swap(original="IBIT", substitute="FBTC", qty=10)
        eng.clear_swap("IBIT")
        assert eng.active_substitutions() == {}
        assert eng.in_use_substitutes() == set()


# -----------------------------------------------------------------------
# State reload
# -----------------------------------------------------------------------

class TestStateReload:

    def test_loads_existing_state_on_init(self, tmp_path):
        path = tmp_path / "active.json"
        path.write_text(json.dumps({
            "IBIT": {
                "substitute": "FBTC",
                "swap_in_date": "2026-05-01",
                "qty": 50,
                "leg": "long",
                "swap_back_due": "2026-06-01",
            },
        }))
        eng = SubstitutionEngine(
            config=SubstitutionConfig.from_dict({
                "enabled": True,
                "underlyings": {"IBIT": ["FBTC", "BITB"]},
            }),
            state_path=path,
        )
        active = eng.active_substitutions()
        assert "IBIT" in active
        assert active["IBIT"].substitute == "FBTC"
        assert active["IBIT"].swap_in_date == date(2026, 5, 1)

    def test_corrupt_state_treated_as_empty(self, tmp_path):
        path = tmp_path / "active.json"
        path.write_text("not-valid-json")
        eng = _engine(tmp_path)
        # Just ensure no crash and active_substitutions empty
        # (engine state path differs from corrupt path; this verifies init OK)
        assert eng.active_substitutions() == {}

    def test_corrupt_state_at_engine_path(self, tmp_path):
        path = tmp_path / "active_substitutions.json"
        path.write_text("[invalid")
        eng = _engine(tmp_path)
        assert eng.active_substitutions() == {}
        # Recording still works
        eng.record_swap(original="IBIT", substitute="FBTC", qty=10)
        assert "IBIT" in eng.active_substitutions()


# -----------------------------------------------------------------------
# maybe_swap_back default no-op
# -----------------------------------------------------------------------

class TestMaybeSwapBack:

    def test_returns_empty_by_default(self, tmp_path):
        eng = _engine(tmp_path)
        eng.record_swap(original="IBIT", substitute="FBTC", qty=100)
        assert eng.maybe_swap_back() == []
