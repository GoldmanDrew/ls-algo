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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-date", default=None)
    ap.add_argument("--copy-etf-dashboard", action="store_true")
    args = ap.parse_args(argv)

    payload = build_dashboard_payload(run_date=args.run_date)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[b5-panel] wrote {OUT}")

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
