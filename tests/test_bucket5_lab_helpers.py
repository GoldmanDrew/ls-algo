"""Tests for bucket5_lab_helpers (sweep / tornado plumbing)."""

from __future__ import annotations

from scripts.bucket5_lab_helpers import (
    apply_override,
    backwardation_mask,
    current_value,
    get_path,
    harvest_event_dates,
    metric_value,
    pack_lab_preset,
    set_path,
)


def test_get_set_path_roundtrip():
    d = {"regime": {"gross_backwardation": 0.35}}
    assert get_path(d, ["regime", "gross_backwardation"]) == 0.35
    set_path(d, ["monetize", "bank_frac"], 0.6)
    assert d["monetize"]["bank_frac"] == 0.6


def test_apply_override_sleeve():
    params = {"sleeve_frac": 0.20, "rungs": [{"otm_pct": 0.1, "per_roll_frac": 0.01}]}
    meth = {"borrow_stress_mult": 1.0}
    p2, m2 = apply_override(params, meth, "Sleeve gross fraction", 0.25)
    assert p2["sleeve_frac"] == 0.25
    assert m2["borrow_stress_mult"] == 1.0


def test_apply_override_total_premium_scales_rungs():
    params = {
        "rungs": [
            {"otm_pct": 0.10, "per_roll_frac": 0.008},
            {"otm_pct": 0.25, "per_roll_frac": 0.008},
            {"otm_pct": 0.35, "per_roll_frac": 0.008},
        ]
    }
    p2, _ = apply_override(params, {}, "Total premium %/roll", 0.030)
    total = sum(r["per_roll_frac"] for r in p2["rungs"])
    assert abs(total - 0.030) < 1e-9


def test_current_value_borrow_stress():
    params = {"sleeve_frac": 0.2}
    meth = {"borrow_stress_mult": 1.5}
    assert current_value(params, meth, "Borrow stress multiplier") == 1.5


def test_metric_value_from_result():
    r = {"metrics": {"combined_CAGR": 0.12}, "crash": {"crash_mild_-20%": 0.5}}
    assert metric_value(r, "CAGR") == 0.12
    assert metric_value(r, "Crash payoff -20%") == 0.5


def test_harvest_event_dates():
    series = {
        "realized_cum": [
            ["2024-01-02", 0.0],
            ["2024-01-03", 0.0],
            ["2024-01-04", 12000.0],
            ["2024-01-05", 12000.0],
        ]
    }
    hits = harvest_event_dates(series, min_usd=5000.0)
    assert hits == ["2024-01-04"]


def test_backwardation_mask():
    series = {
        "ratio": [
            ["2024-01-01", 0.95],
            ["2024-01-02", 0.85],
            ["2024-01-03", 0.84],
            ["2024-01-04", 0.92],
        ]
    }
    spans = backwardation_mask(series, r_lo=0.88)
    assert len(spans) == 1
    assert spans[0][0] == "2024-01-02"


def test_pack_lab_preset_schema():
    blob = pack_lab_preset(era="extended", preset="F_dynamic_deep30", params={"sleeve_frac": 0.2}, methodology={}, label="test")
    assert blob["schema"] == "bucket5_lab_preset.v1"
    assert blob["era"] == "extended"
