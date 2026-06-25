"""Low-history σ relaxation in ``enrich_with_decay_and_vol`` (e.g. CBRZ/CBRS)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import daily_screener as ds  # noqa: E402


def _tr_series(n_returns: int, *, sigma_daily: float, seed: int) -> pd.Series:
    """Price index with ``n_returns`` daily log-return observations."""
    rng = np.random.default_rng(seed)
    shocks = rng.normal(0.0, sigma_daily, size=n_returns)
    idx = pd.bdate_range(end=pd.Timestamp("2026-06-24"), periods=n_returns + 1)
    return pd.Series(np.exp(np.cumsum(np.concatenate([[0.0], shocks]))), index=idx)


def test_effective_vol_min_days_relaxes_for_short_listings() -> None:
    assert ds._effective_vol_min_days(0, 30) == 30
    assert ds._effective_vol_min_days(9, 30) == 30
    assert ds._effective_vol_min_days(17, 30) == 17
    assert ds._effective_vol_min_days(40, 30) == 30


def test_enrich_with_decay_and_vol_populates_sigma_for_17_day_inverse() -> None:
    """Mirrors CBRZ/CBRS: 17 aligned days, β≈−2, min_delta_days=30."""
    n = 17
    und_tr = _tr_series(n, sigma_daily=0.025, seed=11)
    etf_tr = _tr_series(n, sigma_daily=0.05, seed=22)

    df = pd.DataFrame(
        [
            {
                "ETF": "CBRZ",
                "Underlying": "CBRS",
                "Delta": -2.0,
                "Delta_n_obs": n,
                "Leverage": -2.0,
                "borrow_current": 0.107197,
            }
        ]
    )
    tr_map = {"CBRZ": etf_tr, "CBRS": und_tr}
    expense_ratios = {"CBRZ": (0.0149, "tradr")}

    out = ds.enrich_with_decay_and_vol(
        df.copy(),
        tr_map,
        min_days=30,
        expense_ratios=expense_ratios,
        underlying_borrow_map={"CBRS": 0.013806},
    )
    row = out.iloc[0]

    assert pd.notna(row["vol_etf_annual"]) and float(row["vol_etf_annual"]) > 0
    assert pd.notna(row["vol_underlying_annual"]) and float(row["vol_underlying_annual"]) > 0
    assert pd.notna(row["expected_gross_decay_annual"]) and float(row["expected_gross_decay_annual"]) > 0

    src = json.loads(row["vol_underlying_source"])
    assert src["method"] != "no_data"
    assert src.get("n_probes", 0) >= 1 or src["method"] == "vol_shape_rv_fallback"
