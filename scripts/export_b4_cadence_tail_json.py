"""Export a dashboard-ready B4 cadence tail-risk JSON from cadence_risk_opt.csv.

Produces ``risk_dashboard/data/bucket4_cadence_tail.json`` with the realized
performance + Monte-Carlo drawdown tail for the production cadence, the
risk-aware winner, and the full return-vs-tail frontier. This is the shape a
new ``bucket4_tail_panel`` could read in ``risk_dashboard/metrics.py``.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "notebooks/output/sizing_tilt_cadence_bt/cadence_risk_opt.csv"
OUT = REPO / "risk_dashboard/data/bucket4_cadence_tail.json"

PROD = "lin b12 k+2.25 m2.5"   # current production parametrization
WINNER_BY = "score"            # risk-aware objective column


def row_payload(r: pd.Series) -> dict:
    return {
        "variant": r.name,
        "family": r["family"],
        "cagr": round(float(r["cagr"]), 4),
        "vol": round(float(r["vol"]), 4),
        "sharpe": round(float(r["sharpe"]), 3),
        "hist_maxdd": round(float(r["maxdd"]), 4),
        "calmar": round(float(r["calmar"]), 3),
        "mean_n_rebal": round(float(r["mean_n_rebal"]), 1),
        "mc_drawdown_1y": {
            "block_bootstrap": {
                "median": round(float(r["boot_dd_med"]), 4),
                "p95": round(float(r["boot_dd_p95"]), 4),
                "p99": round(float(r["boot_dd_p99"]), 4),
                "p99_9": round(float(r["boot_dd_p999"]), 4),
                "P_gt_15": round(float(r["boot_P(dd>15)"]), 4),
                "P_gt_25": round(float(r["boot_P(dd>25)"]), 4),
                "P_gt_40": round(float(r["boot_P(dd>40)"]), 4),
            },
            "student_t": {
                "p95": round(float(r["t_dd_p95"]), 4),
                "p99": round(float(r["t_dd_p99"]), 4),
                "p99_9": round(float(r["t_dd_p999"]), 4),
            },
            "laplace": {
                "p95": round(float(r["lap_dd_p95"]), 4),
                "p99": round(float(r["lap_dd_p99"]), 4),
                "p99_9": round(float(r["lap_dd_p999"]), 4),
            },
        },
    }


def main() -> int:
    if not SRC.exists():
        print(f"missing {SRC}; run bucket4_cadence_risk_opt.py first", file=sys.stderr)
        return 1
    df = pd.read_csv(SRC).set_index("variant")
    winner = df.sort_values(WINNER_BY, ascending=False).index[0]

    frontier = [
        {"variant": idx, "family": r["family"], "cagr": round(float(r["cagr"]), 4),
         "boot_dd_p95": round(float(r["boot_dd_p95"]), 4),
         "P_gt_40": round(float(r["boot_P(dd>40)"]), 4)}
        for idx, r in df.iterrows()
    ]

    payload = {
        "schema": "bucket4_cadence_tail.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "horizon_days": 252,
        "note": ("Monte-Carlo 1y max-drawdown for the live B4 proposed book "
                 "(gross-weighted). Block bootstrap preserves vol clustering; "
                 "Student-t / Laplace are fat-tailed parametric fits."),
        "production": row_payload(df.loc[PROD]) if PROD in df.index else None,
        "recommended": row_payload(df.loc[winner]),
        "frontier": frontier,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2))
    print(f"wrote {OUT}")
    print(f"production={PROD}  recommended={winner}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
