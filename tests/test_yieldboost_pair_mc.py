"""Tests for yieldboost_pair_mc weekly path forward engine."""

from __future__ import annotations

import math
import os
import sys

import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from yieldboost_pair_mc import (
    simulate_weekly_compound_pair_pnl,
    stable_seed_from_symbol,
    yieldboost_pair_decay_distribution,
)


def test_quantile_ordering_log_axis():
    out = yieldboost_pair_decay_distribution(
        sigma_annual=0.674,
        beta=0.372,
        capture_ratio=0.80,
        borrow_annual=0.05,
        n_paths=5000,
        seed=stable_seed_from_symbol("MTYY"),
    )
    assert out is not None
    assert out["p10"] <= out["p50"] <= out["p90"]
    assert out["model"] == "yieldboost_weekly_compound_mc"
    assert out["axis"] == "log_continuous_annual"


def test_capture_ratio_lowers_p50():
    seed = stable_seed_from_symbol("MTYY")
    base = simulate_weekly_compound_pair_pnl(
        sigma_annual=0.674, beta=0.372, capture_ratio=0.0,
        borrow_annual=0.0, n_paths=5000, seed=seed,
    )
    high = simulate_weekly_compound_pair_pnl(
        sigma_annual=0.674, beta=0.372, capture_ratio=0.80,
        borrow_annual=0.0, n_paths=5000, seed=seed,
    )
    assert base is not None and high is not None
    assert base["p50_log"] > high["p50_log"]


def test_borrow_lowers_p50():
    seed = 42
    lo = simulate_weekly_compound_pair_pnl(
        sigma_annual=0.674, beta=0.372, capture_ratio=0.65,
        borrow_annual=0.0, n_paths=3000, seed=seed,
    )
    hi = simulate_weekly_compound_pair_pnl(
        sigma_annual=0.674, beta=0.372, capture_ratio=0.65,
        borrow_annual=0.05, n_paths=3000, seed=seed,
    )
    assert lo["p50_log"] > hi["p50_log"]
    assert hi["p50_log"] == pytest.approx(lo["p50_log"] - 0.05, abs=0.02)
