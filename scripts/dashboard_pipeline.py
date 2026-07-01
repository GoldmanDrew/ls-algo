#!/usr/bin/env python3
"""Build + verify dashboard snapshot (shared by EOD tail and risk_dashboard CI)."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _run(cmd: list[str], *, cwd: Path | None = None) -> None:
    print(f"[dashboard_pipeline] {' '.join(cmd)}")
    subprocess.check_call(cmd, cwd=cwd or REPO)


def resolve_screener_csv(run_date: str, runs_root: Path) -> Path:
    pinned = runs_root / run_date / "etf_screened_today.csv"
    if pinned.is_file():
        return pinned
    fallback = REPO / "data" / "etf_screened_today.csv"
    return fallback


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-date", required=True)
    ap.add_argument("--runs-root", default="data/runs")
    ap.add_argument("--out-dir", default="risk_dashboard/data")
    ap.add_argument("--skip-b4sim", action="store_true")
    ap.add_argument("--fail-if-stale", action="store_true")
    ap.add_argument("--allow-config-nav", action="store_true")
    ap.add_argument("--allow-stale", action="store_true")
    ap.add_argument("--skip-verify", action="store_true")
    ap.add_argument("--write-manifest", action="store_true")
    args = ap.parse_args(argv)

    runs_root = (REPO / args.runs_root).resolve()
    out_dir = (REPO / args.out_dir).resolve()
    run_date = args.run_date

    if args.write_manifest:
        _run(
            [
                sys.executable,
                str(REPO / "scripts" / "run_data_contract.py"),
                "--run-date",
                run_date,
                "--runs-root",
                str(runs_root),
            ]
        )

    if not args.skip_b4sim:
        _run(
            [
                sys.executable,
                "scripts/build_bucket4_risk_sim.py",
                "--run-date",
                run_date,
                "--n-mc",
                "8000",
            ]
        )

    screener = resolve_screener_csv(run_date, runs_root)
    build_cmd = [
        sys.executable,
        "-m",
        "risk_dashboard.build_site",
        "--run-date",
        run_date,
        "--runs-root",
        str(runs_root),
        "--out-dir",
        str(out_dir),
        "--screener-csv",
        str(screener),
    ]
    if args.fail_if_stale:
        build_cmd.append("--fail-if-stale")
    nav = os.getenv("MAGIS_NAV_USD", "").strip()
    if nav:
        build_cmd.extend(["--nav-usd", nav])
    _run(build_cmd)

    if not args.skip_verify:
        verify_cmd = [
            sys.executable,
            "scripts/verify_dashboard_snapshot.py",
            "--run-date",
            run_date,
            "--runs-root",
            str(runs_root),
            "--snapshot-dir",
            str(out_dir),
        ]
        if args.allow_config_nav:
            verify_cmd.append("--allow-config-nav")
        if args.allow_stale:
            verify_cmd.append("--allow-stale")
        _run(verify_cmd)

    print(f"[dashboard_pipeline] ok {run_date}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
