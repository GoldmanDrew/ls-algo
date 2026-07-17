"""Hard gates before publishing a risk-dashboard snapshot.

Usage::

    python scripts/verify_dashboard_snapshot.py --run-date 2026-06-30
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "data" / "runs"
SNAPSHOT_DIR = REPO / "risk_dashboard" / "data"

sys.path.insert(0, str(REPO))

from risk_dashboard.metrics import (  # noqa: E402
    BUCKET_KEYS,
    evaluate_exposure_reconciliation,
)
from scripts.run_data_contract import (  # noqa: E402
    discover_accounting_dates,
    is_config_nav_source,
    load_manifest,
)


class GateFailure(Exception):
    pass


def _fail(msg: str, errors: list[str]) -> None:
    errors.append(msg)


def verify_snapshot(
    run_date: str,
    *,
    runs_root: Path | None = None,
    snapshot_dir: Path | None = None,
    require_manifest: bool = True,
    require_config_nav: bool = True,
    require_fresh: bool = True,
) -> list[str]:
    runs_root = runs_root or RUNS
    snapshot_dir = snapshot_dir or SNAPSHOT_DIR
    errors: list[str] = []

    snap_path = snapshot_dir / f"{run_date}.json"
    latest_path = snapshot_dir / "latest.json"
    if not snap_path.is_file():
        _fail(f"missing snapshot {snap_path}", errors)
        return errors
    snap = json.loads(snap_path.read_text(encoding="utf-8"))

    if require_manifest:
        try:
            manifest = load_manifest(run_date, runs_root=runs_root)
        except FileNotFoundError:
            _fail(f"missing manifest.json for {run_date}", errors)
            manifest = {}
        else:
            if snap.get("run_date") != manifest.get("run_date"):
                _fail("snapshot run_date != manifest run_date", errors)

    acc = runs_root / run_date / "accounting"
    totals_path = acc / "totals.json"
    if not totals_path.is_file():
        _fail(f"missing {totals_path}", errors)
        return errors
    totals = json.loads(totals_path.read_text(encoding="utf-8"))

    dates = discover_accounting_dates(runs_root)
    if require_fresh and dates and run_date != dates[-1]:
        _fail(
            f"run_date {run_date} is not latest accounting ({dates[-1]})",
            errors,
        )

    fresh = snap.get("freshness") or {}
    if require_fresh and fresh.get("is_latest") is False:
        _fail("snapshot freshness.is_latest is false", errors)

    nav_source = snap.get("nav_source") or totals.get("nav_source")
    if require_config_nav and not is_config_nav_source(nav_source):
        _fail(f"NAV source is not config-derived: {nav_source!r}", errors)
    snap_nav = snap.get("nav_usd")
    totals_nav = totals.get("nav_usd")
    if snap_nav is not None and totals_nav is not None:
        try:
            if abs(float(snap_nav) - float(totals_nav)) > 0.01:
                _fail(
                    f"snapshot nav_usd {float(snap_nav):,.2f} != totals {float(totals_nav):,.2f}",
                    errors,
                )
        except (TypeError, ValueError):
            pass

    recon = evaluate_exposure_reconciliation(totals)
    if not recon.get("reconciles"):
        _fail(
            f"exposure reconciliation failed gross_diff={recon.get('gross_diff_pct'):.4%}",
            errors,
        )

    bucket_pnl = totals.get("bucket_pnl") or {}
    for bucket in BUCKET_KEYS:
        csv_path = acc / f"pnl_{bucket}.csv"
        if not csv_path.is_file():
            if bucket in bucket_pnl and float(bucket_pnl.get(bucket, 0) or 0) != 0:
                _fail(f"missing {csv_path.name} but bucket_pnl non-zero", errors)
            continue
        df = pd.read_csv(csv_path)
        if "total_pnl" not in df.columns:
            _fail(f"{csv_path.name} missing total_pnl column", errors)
            continue
        csv_sum = float(pd.to_numeric(df["total_pnl"], errors="coerce").fillna(0.0).sum())
        headline = float(bucket_pnl.get(bucket, 0.0) or 0.0)
        if abs(csv_sum - headline) > max(1.0, abs(headline) * 1e-6):
            _fail(
                f"{bucket}: csv sum {csv_sum:,.2f} != email headline {headline:,.2f}",
                errors,
            )
        snap_rows = (snap.get("buckets") or {}).get(bucket, {}).get("pnl_rows") or []
        snap_sum = sum(float(r.get("total_pnl", 0) or 0) for r in snap_rows)
        if abs(snap_sum - headline) > max(1.0, abs(headline) * 1e-6):
            _fail(
                f"{bucket}: snapshot sum {snap_sum:,.2f} != email headline {headline:,.2f}",
                errors,
            )

    dq = snap.get("data_quality") or {}
    if str(dq.get("status", "")).lower() == "hard":
        _fail("data_quality.status is hard", errors)

    # Hedged/unhedged lens: only gate when the panel is present and populated.
    hedged_panel = snap.get("hedged_pnl_panel") or {}
    if hedged_panel.get("available"):
        h = hedged_panel.get("hedged_ytd_usd")
        u = hedged_panel.get("unhedged_ytd_usd")
        t = hedged_panel.get("total_ytd_usd")
        if h is not None and u is not None and t is not None:
            gap = float(t) - (float(h) + float(u))
            if abs(gap) > 1.0:
                _fail(
                    f"hedged_pnl_panel does not tie out: hedged+unhedged differs from total by {gap:,.2f}",
                    errors,
                )

    exp_recon = snap.get("exposure_reconciliation") or {}
    if exp_recon and not exp_recon.get("reconciles", True):
        _fail("snapshot exposure_reconciliation.reconciles is false", errors)

    if latest_path.is_file():
        latest = None
        try:
            latest_raw = latest_path.read_text(encoding="utf-8")
            latest = json.loads(latest_raw)
            json.dumps(latest, allow_nan=False)
        except json.JSONDecodeError as exc:
            _fail(f"latest.json is not valid JSON: {exc}", errors)
        except (TypeError, ValueError) as exc:
            _fail(f"latest.json contains non-JSON floats (NaN/inf): {exc}", errors)
        if latest is not None and latest.get("run_date") != run_date:
            _fail(
                f"latest.json run_date {latest.get('run_date')} != built {run_date}",
                errors,
            )

    for key in (
        "freshness",
        "borrow_shock_panel",
        "drawdown_panel",
        "pnl_panel",
        "movers_panel",
        "display_sleeve_groups",
    ):
        if key not in snap:
            _fail(f"snapshot missing required key {key!r}", errors)

    book = snap.get("book") or {}
    if book.get("pnl_ytd_usd") is None and book.get("pnl_today_usd") is None:
        _fail("book missing pnl_ytd_usd", errors)

    try:
        json.dumps(snap, allow_nan=False)
    except (TypeError, ValueError) as exc:
        _fail(f"snapshot contains non-JSON floats (NaN/inf): {exc}", errors)

    return errors


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-date", required=True)
    ap.add_argument("--runs-root", default="data/runs")
    ap.add_argument("--snapshot-dir", default="risk_dashboard/data")
    ap.add_argument(
        "--allow-broker-nav",
        action="store_true",
        help="Skip requiring config:capital_usd NAV (legacy/dev only)",
    )
    ap.add_argument("--allow-stale", action="store_true")
    ap.add_argument("--no-manifest", action="store_true")
    args = ap.parse_args(argv)

    errors = verify_snapshot(
        args.run_date,
        runs_root=Path(args.runs_root).resolve(),
        snapshot_dir=Path(args.snapshot_dir).resolve(),
        require_manifest=not args.no_manifest,
        require_config_nav=not args.allow_broker_nav,
        require_fresh=not args.allow_stale,
    )
    if errors:
        print("VERIFY FAILED:", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        return 1
    print(f"verify ok: {args.run_date}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
