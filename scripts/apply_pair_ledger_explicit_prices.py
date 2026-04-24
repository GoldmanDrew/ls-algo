"""
After ``pair_daily`` rows include ``underlying_price`` and ``etf_price`` (from the v15 engine),
use this to overwrite implied marks before Excel export.

Call from ``build_pair_ledger_frames`` (or equivalent) immediately after computing implied prices::

    up, ep = _implied_prices(d)
    up, ep = merge_implied_with_explicit(d, up, ep)
    d[\"underlying_price\"] = up
    d[\"etf_price\"] = ep
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def merge_implied_with_explicit(
    d: pd.DataFrame,
    implied_under: pd.Series,
    implied_etf: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    up = implied_under.copy()
    ep = implied_etf.copy()
    if "underlying_price" in d.columns:
        u0 = pd.to_numeric(d["underlying_price"], errors="coerce")
        mask = u0.notna() & np.isfinite(u0) & (u0 > 0)
        up = u0.where(mask, up)
    if "etf_price" in d.columns:
        e0 = pd.to_numeric(d["etf_price"], errors="coerce")
        mask = e0.notna() & np.isfinite(e0) & (e0 > 0)
        ep = e0.where(mask, ep)
    return up, ep
