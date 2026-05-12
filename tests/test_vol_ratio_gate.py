"""Tests for the structured vol-ratio gate (recompute_vol_ratio_gate)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import daily_screener as ds  # noqa: E402


def _make_df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_clean_pair_is_not_outlier() -> None:
    df = _make_df(
        [
            {"ETF": "AAPU", "vol_underlying_annual": 0.30, "vol_etf_annual": 0.60, "Beta": 2.0},
            {"ETF": "TQQQ", "vol_underlying_annual": 0.25, "vol_etf_annual": 0.75, "Beta": 3.0},
        ]
    )
    out = ds.recompute_vol_ratio_gate(df, screener_cfg=None)
    assert "vol_ratio_value" in out.columns
    assert "vol_ratio_outlier" in out.columns
    assert out["vol_ratio_outlier"].astype(bool).sum() == 0
    assert out.loc[0, "vol_ratio_value"] == pytest.approx(1.0)
    assert out.loc[1, "vol_ratio_value"] == pytest.approx(1.0)


def test_unhandled_split_outlier_is_flagged() -> None:
    """vol_etf=5.0 with vol_und=1.25 and beta=2 → ratio=2.0, well outside
    the [0.5, 1.5] default 2x gate. Models the BAIG-style failure mode
    where an unhandled reverse split inflates vol_etf far above |β|·σ_und.
    """
    df = _make_df(
        [
            {
                "ETF": "BAIG",
                "vol_underlying_annual": 1.25,
                "vol_etf_annual": 5.0,
                "Beta": 2.0,
            }
        ]
    )
    out = ds.recompute_vol_ratio_gate(df, screener_cfg=None)
    assert bool(out.loc[0, "vol_ratio_outlier"]) is True
    assert out.loc[0, "vol_ratio_value"] == pytest.approx(2.0, rel=1e-6)


def test_post_fix_baig_ratio_is_clean() -> None:
    """After the multi-source split pipeline corrects BAIG, vol_etf comes
    down from 3.19 to ~2.5, the ratio settles at 1.0, and the row should
    NOT be flagged for purgatory. Pins the verification-checklist item
    that the six split-affected tickers exit purgatory after the fix.
    """
    df = _make_df(
        [
            {
                "ETF": "BAIG",
                "vol_underlying_annual": 1.25,
                "vol_etf_annual": 2.50,
                "Beta": 2.0,
            }
        ]
    )
    out = ds.recompute_vol_ratio_gate(df, screener_cfg=None)
    assert bool(out.loc[0, "vol_ratio_outlier"]) is False
    assert out.loc[0, "vol_ratio_value"] == pytest.approx(1.0, rel=1e-6)


def test_yaml_gate_widening_relaxes_outlier_flag() -> None:
    """Custom YAML widening the gate to [0.3, 3.0] must accept the BAIG-style
    ratio=2.0 row that the default gate flags."""
    cfg = {
        "vol_ratio_gate": {
            "enabled": True,
            "by_abs_beta": {"2.0": {"min": 0.3, "max": 3.0}},
            "default": {"min": 0.3, "max": 3.0},
            "purgatory_on_outlier": True,
        }
    }
    df = _make_df(
        [
            {
                "ETF": "BAIG",
                "vol_underlying_annual": 1.25,
                "vol_etf_annual": 5.0,
                "Beta": 2.0,
            }
        ]
    )
    out = ds.recompute_vol_ratio_gate(df, screener_cfg=cfg)
    assert bool(out.loc[0, "vol_ratio_outlier"]) is False


def test_disabled_gate_suppresses_purgatory_effect_only() -> None:
    """``enabled: false`` populates the column but is wired in
    ``recompute_purgatory_by_bucket`` to skip the purgatory effect.
    """
    cfg = {"vol_ratio_gate": {"enabled": False}}
    df = _make_df(
        [
            {
                "ETF": "BAIG",
                "vol_underlying_annual": 1.25,
                "vol_etf_annual": 3.19,
                "Beta": 2.0,
            }
        ]
    )
    out = ds.recompute_vol_ratio_gate(df, screener_cfg=cfg)
    # Outlier flag still populated for diagnostics, but downstream gate is
    # disabled at recompute_purgatory_by_bucket.
    assert "vol_ratio_value" in out.columns


# pytest is imported lazily here to keep the file usable as a script.
import pytest  # noqa: E402
