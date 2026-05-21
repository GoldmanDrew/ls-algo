"""Unit tests for parallel establish helpers (clientIds, caps, aggregation)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import execute_trade_plan as etp  # noqa: E402
from execute_trade_plan import ExecResult  # noqa: E402
import rebalance_strategy as rs  # noqa: E402


@pytest.mark.parametrize(
    "worker_idx, leg_idx",
    [(0, 0), (0, 3), (10, 0), (10, 2), (24, 15)],
)
def test_establish_worker_and_leg_client_ids_disjoint(
    worker_idx: int, leg_idx: int
) -> None:
    base = 41
    worker_cid = rs._establish_worker_client_id(base, worker_idx)
    leg_cid = rs._establish_etf_leg_client_id(base, worker_idx, leg_idx)
    assert worker_cid != leg_cid


def test_establish_worker_client_id_no_collision_across_workers() -> None:
    base = 41
    ids = {rs._establish_worker_client_id(base, w) for w in range(30)}
    assert len(ids) == 30


def test_establish_leg_client_id_rejects_too_many_legs() -> None:
    with pytest.raises(ValueError, match="leg_idx"):
        rs._establish_etf_leg_client_id(41, 0, rs._ESTABLISH_ETF_LEGS_PER_BUCKET)


def test_establish_parallel_worker_cap_reserves_etf_slots() -> None:
    assert rs._establish_parallel_worker_cap(25) == min(
        25, etp.MAX_TWS_CLIENTS - rs._ESTABLISH_ETF_LEG_CONN_SLOTS - rs._ESTABLISH_TWS_HEADROOM
    )


def test_aggregate_establish_etf_short_outcomes() -> None:
    outcomes = [
        rs._EstablishEtfShortOutcome(
            etf="AAA", qty=100, px_etf=10.0, short_usd=-1000.0,
            blocked=True, why="ftp_avail0", res=None,
            fill_record={},
        ),
        rs._EstablishEtfShortOutcome(
            etf="BBB", qty=50, px_etf=20.0, short_usd=-500.0,
            blocked=False, why="", res=ExecResult(filled=25, trade=None, status="PARTIAL"),
            fill_record={},
        ),
        rs._EstablishEtfShortOutcome(
            etf="CCC", qty=30, px_etf=15.0, short_usd=-300.0,
            blocked=False, why="", res=None,
            fill_record={},
        ),
    ]
    req, filled, ok = rs._aggregate_establish_etf_short_outcomes(outcomes)
    assert req == 180
    assert filled == 25
    assert ok is True


def test_aggregate_all_blocked_no_success() -> None:
    outcomes = [
        rs._EstablishEtfShortOutcome(
            etf="X", qty=10, px_etf=1.0, short_usd=-10.0,
            blocked=True, why="ftp_avail0", res=None, fill_record={},
        ),
    ]
    req, filled, ok = rs._aggregate_establish_etf_short_outcomes(outcomes)
    assert req == 10
    assert filled == 0
    assert ok is False
