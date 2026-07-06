"""Validate scenario carry forecasts against realized P&L history."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


def _read_pnl_history(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def compute_carry_validation(
    *,
    predicted_carry_pct_nav: float | None,
    repo_root: Path | None = None,
    pnl_history_path: Path | None = None,
    lookback_days: int = 63,
    mape_target: float = 0.25,
    ratio_warn: float = 2.0,
) -> dict[str, Any]:
    """Compare model 12M carry (as daily rate) to recent stock-sleeve P&L / capital."""
    root = repo_root or Path(__file__).resolve().parents[1]
    hist_path = pnl_history_path or root / "data" / "ledger" / "pnl_history.csv"
    rows = _read_pnl_history(hist_path)
    out: dict[str, Any] = {
        "available": False,
        "pnl_history_path": str(hist_path),
        "lookback_days": lookback_days,
        "mape_target": mape_target,
        "ratio_warn": ratio_warn,
        "tail_scenarios_trusted": True,
        "warnings": [],
    }
    if predicted_carry_pct_nav is None or not rows:
        out["reason"] = "missing_prediction_or_history"
        return out

    tail = rows[-lookback_days:] if len(rows) > lookback_days else rows
    pnl_vals: list[float] = []
    cap_vals: list[float] = []
    for r in tail:
        try:
            pnl_vals.append(float(r.get("pnl_stock_sleeves") or 0.0))
            cap_vals.append(float(r.get("net_capital_stock_sleeves") or 0.0))
        except (TypeError, ValueError):
            continue
    if not pnl_vals or not cap_vals or sum(abs(c) for c in cap_vals) < 1e-6:
        out["reason"] = "insufficient_history"
        return out

    avg_cap = sum(cap_vals) / len(cap_vals)
    realized_daily_pct = (sum(pnl_vals) / len(pnl_vals)) / avg_cap if avg_cap > 0 else None
    predicted_daily_pct = float(predicted_carry_pct_nav) / 252.0

    out["available"] = True
    out["realized_daily_pct_nav"] = realized_daily_pct
    out["predicted_daily_pct_nav"] = predicted_daily_pct
    out["realized_annualized_pct_nav"] = (
        realized_daily_pct * 252 if realized_daily_pct is not None else None
    )
    out["predicted_annualized_pct_nav"] = predicted_carry_pct_nav

    if realized_daily_pct is not None and abs(realized_daily_pct) > 1e-9:
        ratio = abs(predicted_daily_pct) / abs(realized_daily_pct)
        out["prediction_to_realized_ratio"] = ratio
        mape = abs(predicted_daily_pct - realized_daily_pct) / abs(realized_daily_pct)
        out["mape"] = mape
        out["mape_ok"] = mape <= mape_target
        if ratio > ratio_warn or ratio < 1.0 / ratio_warn:
            out["tail_scenarios_trusted"] = False
            out["warnings"].append(
                f"Model carry {predicted_carry_pct_nav:.1%}/yr vs realized "
                f"{out['realized_annualized_pct_nav']:.1%}/yr (ratio {ratio:.2f}). "
                "Tail VIX/SPX scenarios flagged for review."
            )
        if not out["mape_ok"]:
            out["warnings"].append(
                f"Carry MAPE {mape:.0%} exceeds target {mape_target:.0%}."
            )
    return out
