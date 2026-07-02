"""Precompute a Bucket 5 sensitivity grid for the static risk dashboard.

Nightly companion to ``build_bucket5_backtest_panel.py``. Sweeps sleeve fraction,
total hedge premium, and borrow stress around the F_dynamic_deep30 extended preset.

Run::

    python scripts/build_bucket5_sensitivity_grid.py
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.bucket5_backtest_api import config_from_preset, run_backtest, _dc_to_dict  # noqa: E402
from scripts.bucket5_insurance_bt import build_ladder  # noqa: E402

OUT = REPO / "risk_dashboard" / "data" / "bucket5_sensitivity.json"

DEFAULT_SLEEVE = [0.10, 0.15, 0.20, 0.25, 0.30]
DEFAULT_PREMIUM = [0.018, 0.024, 0.030]
DEFAULT_BORROW_STRESS = [1.0, 1.5, 2.0]


def _params_for(sleeve: float, premium: float) -> dict:
    base = _dc_to_dict(config_from_preset("F_dynamic_deep30"))
    rungs = build_ladder([(0.10, 0.15), (0.25, 0.35), (0.35, 0.50)], premium)
    base["sleeve_frac"] = sleeve
    base["rungs"] = [{"otm_pct": r.otm_pct, "per_roll_frac": r.per_roll_frac} for r in rungs]
    return base


def build_grid(
    *,
    era: str = "extended",
    sleeve_vals: list[float] | None = None,
    premium_vals: list[float] | None = None,
    borrow_stress_vals: list[float] | None = None,
) -> dict:
    sleeve_vals = sleeve_vals or DEFAULT_SLEEVE
    premium_vals = premium_vals or DEFAULT_PREMIUM
    borrow_stress_vals = borrow_stress_vals or DEFAULT_BORROW_STRESS
    points = []
    total = len(sleeve_vals) * len(premium_vals) * len(borrow_stress_vals)
    n = 0
    for sleeve in sleeve_vals:
        for prem in premium_vals:
            params = _params_for(sleeve, prem)
            for stress in borrow_stress_vals:
                n += 1
                meth = {"borrow_mode": "fixed", "borrow_stress_mult": stress}
                r = run_backtest(
                    params=params,
                    era=era,
                    methodology=meth,
                    include_series=True,
                    max_series_points=180,
                    use_cache=True,
                )
                m = r.get("metrics") or {}
                ser = r.get("series") or {}
                points.append({
                    "sleeve_frac": sleeve,
                    "total_premium": prem,
                    "borrow_stress_mult": stress,
                    "CAGR": m.get("combined_CAGR"),
                    "Vol": m.get("combined_Vol"),
                    "MaxDD": m.get("combined_MaxDD"),
                    "Sharpe": m.get("combined_Sharpe"),
                    "Calmar": m.get("combined_Calmar"),
                    "realized_usd": m.get("realized_$"),
                    "crash_mild": (r.get("crash") or {}).get("crash_mild_-20%"),
                    "crash_severe": (r.get("crash") or {}).get("crash_severe_-30%"),
                    "series": {
                        "combined_equity": ser.get("combined_equity"),
                        "drawdown": ser.get("drawdown"),
                    },
                })
                print(f"[b5-sens] {n}/{total} sleeve={sleeve:.0%} prem={prem:.1%} stress={stress:.1f}x")
    return {
        "schema": "bucket5_sensitivity.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "preset_base": "F_dynamic_deep30",
        "era": era,
        "axes": {
            "sleeve_frac": sleeve_vals,
            "total_premium": premium_vals,
            "borrow_stress_mult": borrow_stress_vals,
        },
        "points": points,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--era", default="extended", choices=["live", "extended"])
    ap.add_argument("--quick", action="store_true", help="3-point grid for CI smoke")
    args = ap.parse_args(argv)

    if args.quick:
        payload = build_grid(
            era=args.era,
            sleeve_vals=[0.15, 0.20, 0.25],
            premium_vals=[0.024, 0.030],
            borrow_stress_vals=[1.0, 2.0],
        )
    else:
        payload = build_grid(era=args.era)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[b5-sens] wrote {OUT} ({len(payload['points'])} points)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
