from __future__ import annotations

from ibkr_accounting import _apply_signed_bucket_trade


def test_apply_signed_bucket_trade_long_partial_close() -> None:
    qty, cost, realized = _apply_signed_bucket_trade(100.0, 1000.0, -30.0, 12.0)
    assert qty == 70.0
    assert realized == 60.0
    assert abs(cost - 700.0) < 1e-6


def test_apply_signed_bucket_trade_short_cover() -> None:
    qty, cost, realized = _apply_signed_bucket_trade(-50.0, -500.0, 20.0, 8.0)
    assert qty == -30.0
    assert realized == 40.0
