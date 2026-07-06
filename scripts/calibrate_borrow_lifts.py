#!/usr/bin/env python3
"""Calibrate historical borrow_lift multipliers from borrow_history.json."""

from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import yaml

from risk_dashboard.vix_scenario import HISTORICAL_VIX_SCENARIOS
from risk_dashboard.borrow_stress import borrow_rate_vix_stress, load_borrow_stress_config

# Approximate event windows (calendar dates, US equity sessions).
EVENT_WINDOWS: dict[str, tuple[str, str]] = {
    "aug_2015_china": ("2015-08-20", "2015-09-30"),
    "feb_2018_xiv": ("2018-02-01", "2018-02-28"),
    "mar_2020_covid": ("2020-02-20", "2020-04-30"),
    "sep_2022_inflation": ("2022-08-15", "2022-10-15"),
    "aug_2024_yen_carry": ("2024-08-01", "2024-08-31"),
}


def _parse_date(s: str) -> datetime:
    return datetime.strptime(s[:10], "%Y-%m-%d")


def _obs_rate(row: dict[str, Any]) -> float | None:
    for key in ("borrow_fee", "borrowFee", "fee", "rate"):
        val = row.get(key)
        if val is None:
            continue
        try:
            r = float(val)
            if r > 1.5:
                r /= 100.0
            return max(0.0, r)
        except (TypeError, ValueError):
            continue
    return None


def _obs_date(row: dict[str, Any]) -> datetime | None:
    for key in ("date", "asof", "timestamp"):
        val = row.get(key)
        if not val:
            continue
        try:
            return _parse_date(str(val))
        except ValueError:
            continue
    return None


def _symbol_rates_in_window(
    history: list[dict[str, Any]],
    start: datetime,
    end: datetime,
) -> list[float]:
    out: list[float] = []
    for row in history:
        dt = _obs_date(row)
        if dt is None or dt < start or dt > end:
            continue
        rate = _obs_rate(row)
        if rate is not None:
            out.append(rate)
    return out


def calibrate_lifts(
    borrow_history: dict[str, list],
    *,
    symbols: list[str] | None = None,
    baseline_vix: float = 15.0,
) -> dict[str, Any]:
    cfg = load_borrow_stress_config()
    baseline_start = datetime(2024, 1, 1)
    baseline_end = datetime(2024, 12, 31)
    sym_list = symbols or sorted(borrow_history.keys())
    baseline_rates: list[float] = []
    for sym in sym_list:
        baseline_rates.extend(
            _symbol_rates_in_window(borrow_history.get(sym) or [], baseline_start, baseline_end)
        )
    baseline_median = statistics.median(baseline_rates) if baseline_rates else 0.05

    lifts: dict[str, Any] = {}
    for ev in HISTORICAL_VIX_SCENARIOS:
        key = str(ev["key"])
        window = EVENT_WINDOWS.get(key)
        if not window:
            lifts[key] = {
                "borrow_lift": float(ev["borrow_lift"]),
                "source": "catalog_default",
            }
            continue
        start, end = (_parse_date(window[0]), _parse_date(window[1]))
        event_rates: list[float] = []
        for sym in sym_list:
            event_rates.extend(
                _symbol_rates_in_window(borrow_history.get(sym) or [], start, end)
            )
        if len(event_rates) < 5 or baseline_median <= 0:
            lifts[key] = {
                "borrow_lift": float(ev["borrow_lift"]),
                "source": "insufficient_data",
                "n_obs": len(event_rates),
            }
            continue
        event_p95 = sorted(event_rates)[min(len(event_rates) - 1, int(len(event_rates) * 0.95))]
        peak_vix = float(ev["vix_peak"])
        mech = borrow_rate_vix_stress(
            baseline_median,
            vix_pts=peak_vix,
            tier="htb",
            borrow_lift=1.0,
            stress_cfg=cfg,
        )
        if mech <= 1e-9:
            lift = float(ev["borrow_lift"])
        else:
            lift = max(1.0, min(3.0, event_p95 / mech))
        lifts[key] = {
            "borrow_lift": round(lift, 3),
            "source": "borrow_history_p95",
            "n_obs": len(event_rates),
            "baseline_median": round(baseline_median, 4),
            "event_p95": round(event_p95, 4),
            "mechanical_at_peak": round(mech, 4),
            "catalog_default": float(ev["borrow_lift"]),
        }
    return lifts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--borrow-history",
        type=Path,
        default=None,
        help="Path to borrow_history.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("config/risk_dashboard_borrow_lifts.yml"),
    )
    parser.add_argument("--write", action="store_true", help="Write YAML output")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    bh_path = args.borrow_history
    if bh_path is None:
        for cand in (
            root / "data" / "borrow_history.json",
            root.parent / "etf-dashboard" / "data" / "borrow_history.json",
            root / "data" / "runs" / "2026-06-29" / "b4_borrow" / "borrow_history.json",
        ):
            if cand.is_file():
                bh_path = cand
                break
    if bh_path is None or not bh_path.is_file():
        print("No borrow_history.json found; writing catalog defaults only.")
        lifts = {
            str(ev["key"]): {"borrow_lift": float(ev["borrow_lift"]), "source": "catalog_default"}
            for ev in HISTORICAL_VIX_SCENARIOS
        }
    else:
        doc = json.loads(bh_path.read_text(encoding="utf-8"))
        raw = doc.get("symbols", doc)
        lifts = calibrate_lifts(raw)
        print(f"Calibrated from {bh_path} ({len(raw)} symbols)")

    payload = {
        "borrow_stress": {
            "calibrated_lifts": lifts,
            "calibrated_at": datetime.utcnow().strftime("%Y-%m-%d"),
            "borrow_history_path": str(bh_path) if bh_path else None,
        }
    }
    print(yaml.safe_dump(payload, sort_keys=False))
    if args.write:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
