"""Borrow history: exclude 0%% fee rows with no shares from weighted resample."""
import datetime as dt

import numpy as np

from screener_v2_fields import (
    _borrow_history_usable_currents_sorted,
    _weighted_borrow_values_probs,
)


def test_weighted_borrow_drops_zero_borrow_zero_shares():
    asof = dt.date(2025, 6, 1)
    hist = [
        {"date": "2025-01-01", "borrow_current": 0.0, "shares_available": 0},
        {"date": "2025-02-01", "borrow_current": 0.40, "shares_available": 50_000},
    ]
    vals, p = _weighted_borrow_values_probs(hist, asof, halflife_days=90.0)
    assert vals is not None and p is not None
    assert vals.size == 1
    assert float(vals[0]) == 0.40
    assert np.isclose(float(p.sum()), 1.0)


def test_weighted_borrow_keeps_zero_borrow_when_shares_positive():
    asof = dt.date(2025, 6, 1)
    hist = [
        {"date": "2025-03-01", "borrow_current": 0.0, "shares_available": 1_000_000},
    ]
    vals, p = _weighted_borrow_values_probs(hist, asof, halflife_days=90.0)
    assert vals is not None and float(vals[0]) == 0.0


def test_usable_currents_sorted_mean_excludes_placeholder():
    asof = dt.date(2025, 6, 1)
    hist = [
        {"date": "2025-01-01", "borrow_current": 0.0, "shares_available": 0},
        {"date": "2025-02-01", "borrow_current": 0.10, "shares_available": 10_000},
        {"date": "2025-03-01", "borrow_current": 0.30, "shares_available": 10_000},
    ]
    vals = _borrow_history_usable_currents_sorted(hist, asof)
    assert vals == [0.10, 0.30]
    assert abs(float(np.mean(vals)) - 0.20) < 1e-9


def test_weighted_borrow_none_when_only_placeholders():
    asof = dt.date(2025, 6, 1)
    hist = [
        {"date": "2025-01-01", "borrow_current": 0.0, "shares_available": 0},
        {"date": "2025-02-01", "borrow_current": 0.0, "shares_available": 0},
    ]
    assert _weighted_borrow_values_probs(hist, asof, halflife_days=90.0) is None
