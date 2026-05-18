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

HISTORY_MAX_RUNS = 60


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


def _extract_history_point(payload: dict) -> dict:
    book = payload.get("book") or {}
    worst = payload.get("worst_shock") or {}
    factor = (payload.get("factor_panel") or {}).get("totals") or {}
    conc = (payload.get("concentration_panel") or {}).get("totals") or {}
    alerts = payload.get("alert_rows") or []
    return {
        "run_date": payload.get("run_date"),
        "nav_usd": book.get("nav_usd"),
        "gross_pct_nav": book.get("gross_exposure_pct_nav"),
        "net_pct_nav": book.get("net_exposure_pct_nav"),
        "pnl_pct_nav": book.get("pnl_today_pct_nav"),
        "worst_shock_pct_nav": worst.get("pnl_pct_nav"),
        "worst_shock_label": worst.get("label"),
        "net_beta_to_spy": factor.get("net_beta_to_spy"),
        "top10_pct_nav": conc.get("top10_pct_nav"),
        "hhi_underlying": conc.get("hhi_underlying"),
        "n_alerts": len(alerts),
        "n_alerts_hard": sum(1 for a in alerts if a.get("status") == "hard"),
    }


def _load_history(out_dir: Path, current_date: str) -> list[dict]:
    snapshots: list[dict] = []
    for path in sorted(out_dir.glob("*.json")):
        stem = path.stem
        if stem in {"latest", "index"} or not stem[:4].isdigit():
            continue
        if stem == current_date:
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        snapshots.append(_extract_history_point(data))
    snapshots.sort(key=lambda r: r.get("run_date") or "")
    return snapshots[-HISTORY_MAX_RUNS:]


def _compute_deltas(current: dict, prior: dict | None) -> dict:
    if not prior:
        return {}
    deltas: dict = {"prior_run_date": prior.get("run_date")}
    for key in (
        "nav_usd",
        "gross_pct_nav",
        "net_pct_nav",
        "pnl_pct_nav",
        "worst_shock_pct_nav",
        "net_beta_to_spy",
        "top10_pct_nav",
        "hhi_underlying",
        "n_alerts",
        "n_alerts_hard",
    ):
        cur = current.get(key)
        pri = prior.get(key)
        if cur is None or pri is None:
            deltas[f"delta_{key}"] = None
            continue
        try:
            deltas[f"delta_{key}"] = float(cur) - float(pri)
        except (TypeError, ValueError):
            deltas[f"delta_{key}"] = None
    return deltas


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
    ap.add_argument(
        "--screener-csv",
        default="data/etf_screened_today.csv",
        help="optional screener CSV used for borrow-squeeze risk overlay",
    )
    args = ap.parse_args(argv)

    runs_root = Path(args.runs_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    screener_csv = Path(args.screener_csv).resolve() if args.screener_csv else None

    run_date = args.run_date or _discover_default_run_date(runs_root)

    snap = build_snapshot(
        run_date=run_date,
        runs_root=runs_root,
        nav_usd=args.nav_usd,
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        screener_csv=screener_csv,
    )
    payload = snap.to_dict()

    history = _load_history(out_dir, current_date=run_date)
    current_point = _extract_history_point(payload)
    payload["history"] = history + [current_point]
    payload["deltas"] = _compute_deltas(current_point, history[-1] if history else None)

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
