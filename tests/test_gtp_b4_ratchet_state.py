"""Tests for generate_trade_plan Bucket 4 ratchet-state I/O and cadence emitter."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

import generate_trade_plan as g


def test_ratchet_state_roundtrip(tmp_path: Path):
    p = tmp_path / "b4_inverse_ratchet_state.json"
    g._b4_write_ratchet_state(p, {"APLZ|APLD": 60_000.0, "NBIZ|NBIS": 12_345.67}, "2026-06-01")
    loaded = g._b4_load_ratchet_state(p)
    assert loaded["APLZ|APLD"] == pytest.approx(60_000.0)
    assert loaded["NBIZ|NBIS"] == pytest.approx(12_345.67)
    # payload is structured + atomic (no leftover tmp)
    raw = json.loads(p.read_text())
    assert raw["run_date"] == "2026-06-01"
    assert not (p.with_suffix(p.suffix + ".tmp")).exists()


def test_ratchet_state_path_from_config():
    rules = {"ratchet": {"state_json": "data/custom_b4_state.json"}}
    assert g._b4_ratchet_state_path(rules) == Path("data/custom_b4_state.json")
    assert g._b4_ratchet_state_path({}) == Path("data/b4_inverse_ratchet_state.json")


def test_load_missing_state_is_empty(tmp_path: Path):
    assert g._b4_load_ratchet_state(tmp_path / "nope.json") == {}


def test_emit_cadence_outputs(tmp_path: Path, monkeypatch):
    # redirect run_dir into tmp
    monkeypatch.setattr(g, "run_dir", lambda rd: tmp_path / rd)
    idx = pd.date_range("2026-01-01", periods=120, freq="B")
    state = SimpleNamespace(
        diagnostics={"cadence_by_underlying": {
            "APLD": {"interval_days": 2, "hedge_ratio": 0.56, "tr": 1.42, "vcr": 0.061,
                     "vcr_med": 0.040, "interval_explain": "denom=1.99 -> 2", "h_explain": "h=0.56"},
            "NBIS": {"interval_days": 5, "hedge_ratio": 0.55, "tr": 0.90, "vcr": 0.030,
                     "vcr_med": 0.045, "interval_explain": "denom=0.73 -> 5", "h_explain": "h=0.55"},
        }},
        hedge_by_underlying={"APLD": pd.Series(0.56, index=idx), "NBIS": pd.Series(0.55, index=idx)},
    )
    tgt = pd.DataFrame([{
        "ETF": "APLZ", "Underlying": "APLD", "inverse_etf_short_usd": 60_000.0,
        "inverse_short_solved_usd": 50_000.0, "underlying_short_usd": 60_000.0,
        "hedge_ratio": 0.56, "ratchet_binding": True, "ratchet_source": "held_position",
        "ratchet_explain": "floored to 60,000",
    }])
    g._emit_b4_cadence_outputs(state, tgt, "2026-06-01")
    base = tmp_path / "2026-06-01" / "b4_hedge_cadence"
    assert (base / "b4_cadence_explain.csv").is_file()
    txt = (base / "b4_cadence_explain.txt").read_text()
    assert "APLD: rebalance every ~2 trading day(s)" in txt
    assert "denom=1.99 -> 2" in txt
    assert (base / "b4_ratchet_targets.csv").is_file()
    # plots are best-effort (matplotlib present in this repo)
    assert (base / "b4_days_to_rebalance.png").is_file()
