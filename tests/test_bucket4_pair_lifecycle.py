"""Tests for the Bucket 4 pair lifecycle demotion ladder (Phase 2)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.bucket4_pair_lifecycle import (  # noqa: E402
    LifecycleConfig,
    apply_lifecycle_to_b4,
    held_etfs,
    lifecycle_sets,
    update_lifecycle,
)

CFG = LifecycleConfig(enabled=True, recover_obs_days=2, reentry_cooldown_days=5)


def mk_monitor(rows: list[dict]) -> pd.DataFrame:
    base = {
        "flag_half": False, "flag_freeze": False, "flag_exit": False,
        "flag_vol_floor": False, "borrow_exceeds_decay": False,
    }
    return pd.DataFrame([{**base, **r} for r in rows])


def test_escalation_jumps_straight_to_exit():
    mon = mk_monitor([{"etf": "BEZ", "underlying": "BE", "flag_exit": True}])
    state, actions = update_lifecycle(mon, {}, CFG, "2026-06-01")
    assert state["BEZ|BE"]["status"] == "exit"
    assert state["BEZ|BE"]["exited_on"] == "2026-06-01"
    assert len(actions) == 1 and actions.iloc[0]["to"] == "exit"


def test_freeze_on_vol_floor():
    mon = mk_monitor([{"etf": "TBT", "underlying": "TLT", "flag_vol_floor": True}])
    state, _ = update_lifecycle(mon, {}, CFG, "2026-06-01")
    assert state["TBT|TLT"]["status"] == "freeze"
    assert state["TBT|TLT"]["reason"] == "vol<keep_floor"


def test_half_then_recovery_promotes_one_level():
    mon_bad = mk_monitor([{"etf": "MSDD", "underlying": "MSTR", "flag_half": True}])
    state, _ = update_lifecycle(mon_bad, {}, CFG, "2026-06-01")
    assert state["MSDD|MSTR"]["status"] == "half"

    mon_ok = mk_monitor([{"etf": "MSDD", "underlying": "MSTR"}])
    state, actions = update_lifecycle(mon_ok, state, CFG, "2026-06-02")
    assert state["MSDD|MSTR"]["status"] == "half"  # 1 clean day < recover_obs_days=2
    assert actions.empty
    state, actions = update_lifecycle(mon_ok, state, CFG, "2026-06-03")
    assert state["MSDD|MSTR"]["status"] == "normal"
    assert actions.iloc[0]["to"] == "normal"


def test_flagged_day_resets_clean_counter():
    mon_bad = mk_monitor([{"etf": "X", "underlying": "Y", "flag_freeze": True}])
    state, _ = update_lifecycle(mon_bad, {}, CFG, "2026-06-01")
    mon_ok = mk_monitor([{"etf": "X", "underlying": "Y"}])
    state, _ = update_lifecycle(mon_ok, state, CFG, "2026-06-02")
    assert state["X|Y"]["clean_days"] == 1
    # still flagged at same severity -> counter resets
    state, _ = update_lifecycle(mon_bad, state, CFG, "2026-06-03")
    assert state["X|Y"]["clean_days"] == 0
    assert state["X|Y"]["status"] == "freeze"


def test_exit_cooldown_then_cleared():
    mon = mk_monitor([{"etf": "BEZ", "underlying": "BE", "flag_exit": True}])
    state, _ = update_lifecycle(mon, {}, CFG, "2026-06-01")
    # during cooldown (next business day): stays exited even if flags clear
    mon_ok = mk_monitor([{"etf": "BEZ", "underlying": "BE"}])
    state, _ = update_lifecycle(mon_ok, state, CFG, "2026-06-02")
    assert state["BEZ|BE"]["status"] == "exit"
    # after cooldown elapses: cleared from state entirely
    state, actions = update_lifecycle(mon_ok, state, CFG, "2026-06-10")
    assert "BEZ|BE" not in state
    assert actions.iloc[0]["to"] == "(cleared)"


def test_lifecycle_sets_and_held():
    state = {
        "A|UA": {"status": "half"},
        "B|UB": {"status": "freeze"},
        "C|UC": {"status": "exit"},
        "D|UD": {"status": "normal"},
    }
    sets = lifecycle_sets(state)
    assert sets["half"] == {"A"} and sets["freeze"] == {"B"} and sets["exit"] == {"C"}
    assert held_etfs(state) == {"A", "B", "C", "D"}


def _b4_frame():
    return pd.DataFrame({
        "ETF": ["AAA", "BBB", "CCC", "DDD"],
        "Underlying": ["UA", "UB", "UC", "UD"],
        "purgatory": [False, False, False, False],
    })


def test_apply_exit_drops_row_and_renormalizes():
    state = {"CCC|UC": {"status": "exit"}}
    b4c, w, info = apply_lifecycle_to_b4(_b4_frame(), np.full(4, 0.25), state, CFG)
    assert info["n_exit"] == 1
    assert list(b4c["ETF"]) == ["AAA", "BBB", "DDD"]
    assert w == pytest.approx([1 / 3] * 3)


def test_apply_freeze_zero_weight_purgatory():
    state = {"BBB|UB": {"status": "freeze"}}
    b4c, w, info = apply_lifecycle_to_b4(_b4_frame(), np.full(4, 0.25), state, CFG)
    assert info["n_freeze"] == 1
    assert w[1] == 0.0
    assert bool(b4c.iloc[1]["purgatory"]) is True
    assert w.sum() == pytest.approx(1.0)


def test_apply_half_multiplies_then_renormalizes():
    state = {"AAA|UA": {"status": "half"}}
    b4c, w, info = apply_lifecycle_to_b4(_b4_frame(), np.full(4, 0.25), state, CFG)
    assert info["n_half"] == 1
    # 0.125 / (0.125 + 0.75)
    assert w[0] == pytest.approx(0.125 / 0.875)
    assert w.sum() == pytest.approx(1.0)


def test_apply_disabled_is_passthrough():
    cfg_off = LifecycleConfig(enabled=False)
    state = {"AAA|UA": {"status": "exit"}}
    b4c, w, info = apply_lifecycle_to_b4(_b4_frame(), np.full(4, 0.25), state, cfg_off)
    assert len(b4c) == 4 and info == {"n_exit": 0, "n_freeze": 0, "n_half": 0}


def test_from_rules_reads_yaml_block():
    rules = {"pair_lifecycle": {"enabled": True, "half_weight_mult": 0.4, "exit_ret_lt": -0.5}}
    cfg = LifecycleConfig.from_rules(rules)
    assert cfg.enabled and cfg.half_weight_mult == 0.4 and cfg.exit_ret_lt == -0.5
