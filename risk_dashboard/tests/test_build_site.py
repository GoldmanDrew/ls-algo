"""Smoke tests for risk_dashboard.build_site history + delta helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from risk_dashboard.build_site import (
    _compute_deltas,
    _discover_accounting_run_dates,
    _extract_history_point,
    _load_history,
    _sanitize_for_json,
    _write_json,
)


def _write_snapshot(out_dir: Path, run_date: str, **overrides) -> None:
    payload = {
        "run_date": run_date,
        "book": {
            "nav_usd": overrides.get("nav_usd", 800_000),
            "gross_exposure_pct_nav": overrides.get("gross_pct_nav", 1.5),
            "net_exposure_pct_nav": overrides.get("net_pct_nav", -0.1),
            "pnl_today_pct_nav": overrides.get("pnl_pct_nav", 0.001),
        },
        "worst_shock": {
            "pnl_pct_nav": overrides.get("worst_shock_pct_nav", -0.04),
            "label": overrides.get("worst_label", "SPX -5% (delta-adj)"),
        },
        "factor_panel": {
            "totals": {"net_beta_to_spy": overrides.get("net_beta", -0.2)},
        },
        "concentration_panel": {
            "totals": {
                "top10_pct_nav": overrides.get("top10_pct_nav", 0.6),
                "hhi_underlying": overrides.get("hhi_underlying", 1800.0),
            }
        },
        "alert_rows": [
            {"status": "hard"} if overrides.get("hard_alerts", 1) > 0 else {"status": "warn"}
            for _ in range(overrides.get("n_alerts", 1))
        ],
    }
    (out_dir / f"{run_date}.json").write_text(json.dumps(payload), encoding="utf-8")


def test_extract_history_point_pulls_cockpit_keys(tmp_path: Path):
    _write_snapshot(tmp_path, "2026-05-15", gross_pct_nav=1.2, worst_shock_pct_nav=-0.03, n_alerts=2)
    payload = json.loads((tmp_path / "2026-05-15.json").read_text(encoding="utf-8"))
    point = _extract_history_point(payload)
    assert point["run_date"] == "2026-05-15"
    assert point["gross_pct_nav"] == pytest.approx(1.2)
    assert point["worst_shock_pct_nav"] == pytest.approx(-0.03)
    assert point["n_alerts"] == 2
    assert point["n_alerts_hard"] == 2  # default hard_alerts > 0


def test_load_history_skips_current_and_sorts(tmp_path: Path):
    _write_snapshot(tmp_path, "2026-05-14")
    _write_snapshot(tmp_path, "2026-05-15")
    _write_snapshot(tmp_path, "2026-05-18")
    history = _load_history(tmp_path, current_date="2026-05-18")
    assert [p["run_date"] for p in history] == ["2026-05-14", "2026-05-15"]


def test_discover_accounting_run_dates_requires_totals_json(tmp_path: Path):
    runs = tmp_path / "runs"
    for d in ("2026-05-15", "2026-05-18"):
        day = runs / d / "accounting"
        day.mkdir(parents=True)
        (day / "totals.json").write_text("{}", encoding="utf-8")
    (runs / "2026-05-19").mkdir()
    assert _discover_accounting_run_dates(runs) == ["2026-05-15", "2026-05-18"]


def test_compute_deltas_handles_missing_prior():
    current = {"gross_pct_nav": 1.5, "n_alerts_hard": 1}
    assert _compute_deltas(current, None) == {}
    prior = {"gross_pct_nav": 1.2, "n_alerts_hard": 0, "run_date": "2026-05-15"}
    deltas = _compute_deltas(current, prior)
    assert deltas["prior_run_date"] == "2026-05-15"
    assert deltas["delta_gross_pct_nav"] == pytest.approx(0.3)
    assert deltas["delta_n_alerts_hard"] == pytest.approx(1.0)


def test_sanitize_for_json_replaces_nan_with_null():
    import math

    assert _sanitize_for_json({"x": float("nan"), "y": 1.0}) == {"x": None, "y": 1.0}
    assert _sanitize_for_json([float("inf")]) == [None]
    assert _sanitize_for_json(math.nan) is None


def test_write_json_is_browser_parseable(tmp_path: Path):
    import math

    out = tmp_path / "snap.json"
    _write_json(out, {"proposed_gross_usd": float("nan"), "ok": 1})
    raw = out.read_text(encoding="utf-8")
    assert "NaN" not in raw
    parsed = json.loads(raw)
    assert parsed["proposed_gross_usd"] is None
    assert parsed["ok"] == 1
