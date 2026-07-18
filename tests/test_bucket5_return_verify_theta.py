"""Cache-dependent tests for the Theta option-mark replay."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.bucket5_return_verify import theta_cache_replay


CACHE = Path("data/cache/spx_options/theta")


@pytest.mark.skipif(not any(CACHE.glob("SPX_*_P.parquet")), reason="Theta cache unavailable")
def test_cached_theta_marks_produce_execution_replay_or_explicit_thin_skip():
    dates: set[pd.Timestamp] = set()
    for path in list(CACHE.glob("SPX_*_P.parquet"))[:8]:
        try:
            dates.update(pd.to_datetime(pd.read_parquet(path).index, errors="coerce").normalize())
        except Exception:
            continue
    dates = {d for d in dates if not pd.isna(d)}
    if not dates:
        pytest.skip("Theta cache has no readable dated quotes")
    idx = pd.DatetimeIndex(sorted(dates))
    # This test validates quote replay plumbing. Market-reality accuracy is
    # reported by the builder using actual cached SPX/VIX histories.
    spot = pd.Series(5000.0, index=idx)
    vix = pd.Series(20.0, index=idx)
    replay, summary = theta_cache_replay(CACHE, spot, vix, min_observations=1)
    assert summary["status"] in {"ok", "skip"}
    if summary["status"] == "ok":
        assert {"buy_ask", "sell_bid", "buy_stressed", "sell_stressed", "error"}.issubset(replay.columns)
        assert summary["observations"] == len(replay)
