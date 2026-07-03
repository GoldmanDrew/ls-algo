"""Tests for B4 risk-sim universe weight resolution."""

from __future__ import annotations

import pandas as pd
import pytest

from scripts.bucket4_risk_sim_universe import (
    FORCE_INCLUDE_ETFS,
    locate_available,
    resolve_sim_gross,
)


def _row(**kwargs) -> pd.Series:
    base = {
        "gross_target_usd": 0.0,
        "optimal_gross_target_usd": 0.0,
        "shares_available": 0,
        "purgatory_no_locate": False,
        "exclude_no_shares": False,
        "net_edge_p50_annual": 0.5,
    }
    base.update(kwargs)
    return pd.Series(base)


def test_resolve_sim_gross_proposed_wins():
    g, src = resolve_sim_gross(_row(gross_target_usd=12_000), median_structural=10_000)
    assert g == 12_000
    assert src == "proposed"


def test_resolve_sim_gross_optimal_when_no_proposed_but_locate():
    g, src = resolve_sim_gross(
        _row(optimal_gross_target_usd=18_000, shares_available=500),
        median_structural=10_000,
    )
    assert g == 18_000
    assert src == "optimal"


def test_resolve_sim_gross_optimal_forced_for_force_include():
    g, src = resolve_sim_gross(
        _row(optimal_gross_target_usd=20_000, shares_available=0),
        median_structural=10_000,
        force_include=True,
    )
    assert g == 20_000
    assert src == "optimal_forced"


def test_resolve_sim_gross_structural_proxy_force_include_no_locate():
    g, src = resolve_sim_gross(
        _row(shares_available=0),
        median_structural=15_000,
        force_include=True,
    )
    assert g == 15_000
    assert src == "structural_proxy"


def test_resolve_sim_gross_screener_proxy_when_force_include_and_locate():
    g, src = resolve_sim_gross(
        _row(shares_available=500, pair_override_gross_mult=10.0, net_edge_p50_annual=0.5),
        median_structural=10_000,
        force_include=True,
    )
    assert g == 100_000  # 10k * 10 * (0.5/0.5)
    assert src == "screener_proxy"


def test_resolve_sim_gross_excluded_no_locate_no_force():
    g, src = resolve_sim_gross(_row(), median_structural=10_000)
    assert g == 0.0
    assert src == "excluded"


def test_locate_available_requires_shares():
    assert not locate_available(_row(shares_available=0))
    assert locate_available(_row(shares_available=100))
    assert not locate_available(_row(shares_available=100, purgatory_no_locate=True))


def test_load_risk_sim_universe_smoke():
    from scripts.bucket4_risk_sim_universe import load_risk_sim_universe

    df = load_risk_sim_universe("2026-07-02")
    if df.empty:
        pytest.skip("no run data for 2026-07-02")
    assert "sim_gross_usd" in df.columns
    assert "weight_source" in df.columns
    forced = df[df["ETF"].isin(FORCE_INCLUDE_ETFS)]
    if not forced.empty:
        assert (forced["sim_gross_usd"] > 0).all()
