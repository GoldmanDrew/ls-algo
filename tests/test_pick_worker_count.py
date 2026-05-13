"""Unit tests for ``execute_trade_plan.pick_worker_count``.

Guards the worker-pool sizing contract used by Phase 1 (cleanup),
Phase 2 (establish), and Phase 3 (hedge / reconcile):

  workers = max(1, min(n_trades, parallel_n_cfg, hard_cap, MAX_TWS_CLIENTS))

In particular: a single-trade cleanup must open exactly one IB
connection, not the full ``parallel_n`` pool — the regression that
prompted introducing this helper.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import execute_trade_plan as etp  # noqa: E402


@pytest.mark.parametrize(
    "n_trades, parallel_n_cfg, hard_cap, expected",
    [
        # Single-trade phases must allocate exactly one worker, regardless
        # of the configured pool size — this is the cleanup regression.
        (1, 25, 25, 1),
        # 0 trades floors to 1 (caller already short-circuits no-op cases;
        # we just guarantee we never return 0).
        (0, 25, 25, 1),
        # Trade count below cfg/cap clamps to trade count.
        (3, 25, 25, 3),
        # Saturated phase scales up to the configured ceiling.
        (50, 25, 25, 25),
        # Lower of cfg vs hard_cap wins when both are below n_trades.
        (50, 10, 25, 10),
        (50, 25, 5, 5),
        # Very high trade volume cannot exceed the global TWS-clients
        # ceiling (currently 30).
        (1_000, 100, 100, etp.MAX_TWS_CLIENTS),
    ],
)
def test_pick_worker_count_clamps(
    n_trades: int, parallel_n_cfg: int, hard_cap: int, expected: int
) -> None:
    got = etp.pick_worker_count(
        n_trades=n_trades,
        parallel_n_cfg=parallel_n_cfg,
        hard_cap=hard_cap,
        label="TEST",
    )
    assert got == expected


def test_pick_worker_count_handles_invalid_inputs() -> None:
    """Negative / zero arguments must not produce <1 workers, and must
    not raise — the helper is invoked from hot paths during execution."""
    assert etp.pick_worker_count(
        n_trades=-1, parallel_n_cfg=25, hard_cap=25, label="TEST"
    ) == 1
    assert etp.pick_worker_count(
        n_trades=10, parallel_n_cfg=0, hard_cap=25, label="TEST"
    ) == 1
    assert etp.pick_worker_count(
        n_trades=10, parallel_n_cfg=25, hard_cap=0, label="TEST"
    ) == 1


def test_max_tws_clients_is_under_default_tws_limit() -> None:
    """TWS allows ~32 client IDs by default. Keep headroom for the
    coordinator (clientId=0), the cancel coordinator, and any
    operator-side tools — never let the helper exceed 30."""
    assert 1 <= etp.MAX_TWS_CLIENTS <= 30
