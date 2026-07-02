"""Tests for bucket5_backtest_api."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.bucket5_backtest_api import (  # noqa: E402
    build_dashboard_payload,
    config_from_preset,
    list_presets,
    run_backtest,
)


def test_list_presets_nonempty():
    presets = list_presets()
    assert "B_production" in presets
    assert "F_dynamic_deep30" in presets


def test_config_from_preset_production():
    cfg = config_from_preset("B_production")
    assert cfg.sleeve_frac == 0.20
    assert cfg.hedge_budget is not None
    assert cfg.hedge_budget.enabled


def test_run_backtest_live_smoke():
    r = run_backtest(preset="B_production", era="live", max_series_points=50, use_cache=True)
    assert r["schema"] == "bucket5_backtest.v1"
    assert "combined_CAGR" in r["metrics"]
    assert "combined_equity" in r.get("series", {})
    assert r["meta"]["era"] == "live"
    assert len(r["series"]["combined_equity"]) <= 55
    if r.get("monetize_events"):
        assert isinstance(r["monetize_events"][0], dict)
        assert "kind" in r["monetize_events"][0]


def test_dashboard_payload_schema():
    payload = build_dashboard_payload(
        variants=[("B_live", "B_production", "live")],
    )
    assert payload["schema"] == "bucket5_backtest_panel.v1"
    assert "B_live" in payload["variants"]
    v = payload["variants"]["B_live"]
    assert "metrics" in v and "series" in v
    json.dumps(payload)  # serializable
