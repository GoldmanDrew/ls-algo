"""Purgatory rows included in structural optimal sizing (generate_trade_plan)."""

from __future__ import annotations

import pandas as pd


def _purgatory_structural_mask(df: pd.DataFrame) -> pd.Series:
    """Mirror generate_trade_plan purgatory_structural logic."""
    purgatory_any = df["purgatory"].fillna(False).astype(bool)
    no_loc = df["purgatory_no_locate"].fillna(False).astype(bool)
    borrow = df["purgatory_borrow_band"].fillna(False).astype(bool)
    edge = df["purgatory_net_edge"].fillna(False).astype(bool)
    vol = df["purgatory_vol_ratio"].fillna(False).astype(bool)
    execution_block = (no_loc | borrow) & ~edge & ~vol
    return purgatory_any & ~execution_block


def test_no_locate_only_not_structural_excluded():
    df = pd.DataFrame([{
        "purgatory": True,
        "purgatory_no_locate": True,
        "purgatory_borrow_band": False,
        "purgatory_net_edge": False,
        "purgatory_vol_ratio": False,
    }])
    assert not bool(_purgatory_structural_mask(df).iloc[0])


def test_borrow_band_only_not_structural_excluded():
    df = pd.DataFrame([{
        "purgatory": True,
        "purgatory_no_locate": False,
        "purgatory_borrow_band": True,
        "purgatory_net_edge": False,
        "purgatory_vol_ratio": False,
    }])
    assert not bool(_purgatory_structural_mask(df).iloc[0])


def test_net_edge_purgatory_still_structural_excluded():
    df = pd.DataFrame([{
        "purgatory": True,
        "purgatory_no_locate": False,
        "purgatory_borrow_band": False,
        "purgatory_net_edge": True,
        "purgatory_vol_ratio": False,
    }])
    assert bool(_purgatory_structural_mask(df).iloc[0])
