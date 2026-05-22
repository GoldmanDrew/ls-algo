"""Tests for event-aware YieldBOOST put-spread decay."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from yieldboost_event_decay import (  # noqa: E402
    event_aware_decay_distribution,
    event_aware_weekly_loss,
    fair_spread_mid_from_sigma,
)
from yieldboost_decay import _weekly_put_spread_loss  # noqa: E402
import numpy as np  # noqa: E402


def test_event_jump_increases_weekly_loss():
    base = event_aware_weekly_loss(0.8, event_jump_underlying=0.0)
    with_jump = event_aware_weekly_loss(0.8, event_jump_underlying=-0.08)
    assert with_jump >= base


def test_fair_spread_mid_positive():
    mid = fair_spread_mid_from_sigma(0.9)
    assert mid is not None
    assert mid > 0


def test_event_aware_decay_worse_with_event_weeks():
    plain = event_aware_decay_distribution(0.3, 0.15, week_has_event=[False] * 52)
    eventy = event_aware_decay_distribution(
        0.3, 0.15,
        week_has_event=[True] + [False] * 51,
        event_jump_pool=[-0.06, -0.04],
    )
    assert plain is not None and eventy is not None
    assert eventy["p50"] >= plain["p50"]
    assert eventy["event_aware"] is True
