from __future__ import annotations

import numpy as np

from daily_screener import _parse_available_shares


def test_parse_available_shares_plain_integer() -> None:
    assert _parse_available_shares("100000") == 100000.0
    assert _parse_available_shares(2500) == 2500.0


def test_parse_available_shares_ibkr_cap_notation() -> None:
    assert _parse_available_shares(">10000000") == 10_000_000.0


def test_parse_available_shares_missing_or_invalid() -> None:
    assert np.isnan(_parse_available_shares(None))
    assert np.isnan(_parse_available_shares(""))
    assert np.isnan(_parse_available_shares("N/A"))
