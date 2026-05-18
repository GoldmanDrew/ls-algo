"""Build the risk-dashboard JSON snapshot consumed by the static site.

Usage (CI / local):

    python -m risk_dashboard.build_site \
        --run-date 2026-05-15 \
        --nav-usd 800000 \
        --runs-root data/runs \
        --out-dir risk_dashboard/data

This is the ONLY command run by ``.github/workflows/risk_dashboard.yml``
after the EOD pipeline has finished. It reads from
``data/runs/<RUN_DATE>/`` and writes:

* ``risk_dashboard/data/<RUN_DATE>.json`` -- the per-day snapshot.
* ``risk_dashboard/data/latest.json``     -- the same content, named so
  that the static site can fetch a stable URL via the GitHub Contents
  API. ``latest.json`` is the only file the SPA reads on load.
* ``risk_dashboard/data/index.json``      -- a small manifest of all
  available run dates so the SPA can render a date picker.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from .metrics import build_snapshot


def _discover_default_run_date(runs_root: Path) -> str:
    if not runs_root.is_dir():
        raise SystemExit(f"runs-root not found: {runs_root}")
    candidates = sorted(
        [p.name for p in runs_root.iterdir() if p.is_dir() and p.name[:4].isdigit()],
        reverse=True,
    )
    if not candidates:
        raise SystemExit(f"no run-date folders under {runs_root}")
    return candidates[0]


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _build_index(out_dir: Path) -> dict:
    runs = sorted(
        [
            p.stem
            for p in out_dir.glob("*.json")
            if p.stem not in {"latest", "index"} and p.stem[:4].isdigit()
        ],
        reverse=True,
    )
    return {
        "runs": runs,
        "latest": runs[0] if runs else None,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-date", default=os.getenv("RUN_DATE"), help="YYYY-MM-DD; defaults to latest under --runs-root")
    ap.add_argument("--runs-root", default="data/runs", help="root of run-date folders")
    ap.add_argument(
        "--nav-usd",
        type=float,
        default=float(os.getenv("MAGIS_NAV_USD", "800000")),
        help="Account NAV used as the denominator for %%-of-NAV metrics. "
        "Override via env MAGIS_NAV_USD or this flag.",
    )
    ap.add_argument(
        "--out-dir",
        default="risk_dashboard/data",
        help="where to write <run_date>.json + latest.json + index.json",
    )
    args = ap.parse_args(argv)

    runs_root = Path(args.runs_root).resolve()
    out_dir = Path(args.out_dir).resolve()

    run_date = args.run_date or _discover_default_run_date(runs_root)

    snap = build_snapshot(
        run_date=run_date,
        runs_root=runs_root,
        nav_usd=args.nav_usd,
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
    )
    payload = snap.to_dict()

    snap_path = out_dir / f"{run_date}.json"
    latest_path = out_dir / "latest.json"
    index_path = out_dir / "index.json"

    _write_json(snap_path, payload)
    _write_json(latest_path, payload)

    manifest = _build_index(out_dir)
    _write_json(index_path, manifest)

    print(f"[risk_dashboard] wrote {snap_path}")
    print(f"[risk_dashboard] wrote {latest_path}")
    print(f"[risk_dashboard] wrote {index_path} ({len(manifest['runs'])} runs)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
