"""Export the authoritative production-actual Bucket 4 dashboard contract."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.b4_dashboard_contract import export_contract, validate_contract  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-dir", type=Path, default=REPO / "notebooks" / "output" / "production_actual_bt")
    ap.add_argument("--out-dir", type=Path, default=REPO / "risk_dashboard" / "data" / "bucket4_production_replay")
    args = ap.parse_args(argv)
    contract = export_contract(args.source_dir, args.out_dir, REPO)
    manifest = validate_contract(args.out_dir)
    print(json.dumps({
        "ok": True,
        "schema": manifest["schema"],
        "out_dir": str(args.out_dir.resolve()),
        "pairs": len(contract["pairs"]),
        "days": len(contract["book"]["dates"]),
        "source": manifest.get("source"),
        "reconciliation": manifest.get("reconciliation"),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

