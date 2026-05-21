"""Tests for partial-fill preservation when execute_leg aborts mid-leg."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import execute_trade_plan as etp  # noqa: E402


class _FakeStatus:
    def __init__(self, filled: int) -> None:
        self.filled = filled


class _FakeTrade:
    def __init__(self, filled: int) -> None:
        self.orderStatus = _FakeStatus(filled)


def test_exec_result_on_exception_returns_partial_from_trade() -> None:
    trade = _FakeTrade(40)
    res = etp.exec_result_on_exception(
        qty=100,
        filled_total=10,
        last_trade=trade,
        exc=RuntimeError("disconnect"),
    )
    assert res.filled == 40
    assert res.status == "PARTIAL"
    assert "RuntimeError" in (res.error_msg or "")


def test_exec_result_on_exception_uses_max_of_totals() -> None:
    trade = _FakeTrade(20)
    res = etp.exec_result_on_exception(
        qty=100,
        filled_total=50,
        last_trade=trade,
        exc=TimeoutError("wait"),
    )
    assert res.filled == 50


def test_exec_result_on_exception_reraises_when_no_fill() -> None:
    with pytest.raises(ValueError, match="no fill"):
        etp.exec_result_on_exception(
            qty=100,
            filled_total=0,
            last_trade=None,
            exc=ValueError("no fill"),
        )
