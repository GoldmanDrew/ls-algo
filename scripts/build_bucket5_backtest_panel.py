"""Generate Bucket 5 insurance backtest JSON for risk dashboard + etf-dashboard."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.bucket5_backtest_api import build_dashboard_payload  # noqa: E402

OUT = REPO / "risk_dashboard" / "data" / "bucket5_backtest.json"
SENS_OUT = REPO / "risk_dashboard" / "data" / "bucket5_sensitivity.json"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-date", default=None)
    ap.add_argument("--copy-etf-dashboard", action="store_true")
    ap.add_argument("--with-sensitivity", action="store_true", help="Also build bucket5_sensitivity.json (45 runs)")
    ap.add_argument("--sensitivity-quick", action="store_true", help="Small sensitivity grid (12 runs)")
    args = ap.parse_args(argv)

    payload = build_dashboard_payload(run_date=args.run_date)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[b5-panel] wrote {OUT}")

    if args.with_sensitivity or args.sensitivity_quick:
        from scripts.build_bucket5_sensitivity_grid import build_grid  # noqa: E402

        sens = build_grid(
            era="extended",
            sleeve_vals=[0.15, 0.20, 0.25] if args.sensitivity_quick else None,
            premium_vals=[0.024, 0.030] if args.sensitivity_quick else None,
            borrow_stress_vals=[1.0, 2.0] if args.sensitivity_quick else None,
        )
        SENS_OUT.write_text(json.dumps(sens, indent=2), encoding="utf-8")
        payload["sensitivity_grid"] = sens
        OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[b5-panel] wrote {SENS_OUT} ({len(sens['points'])} points)")

    if args.copy_etf_dashboard:
        for root in (REPO.parent / "etf-dashboard", Path.home() / "Projects" / "quant" / "etf-dashboard"):
            dest = root / "data" / "bucket5_insurance_backtest.json"
            if root.is_dir():
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(OUT, dest)
                print(f"[b5-panel] copied -> {dest}")

    m = payload["variants"].get("F_extended", {}).get("metrics", {})
    if m:
        print(
            f"[b5-panel] F extended: CAGR={m.get('combined_CAGR', 0) * 100:.2f}% "
            f"MaxDD={m.get('combined_MaxDD', 0) * 100:.2f}%"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
