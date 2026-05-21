#!/usr/bin/env python3
"""Compare local accounting outputs against a known-good git baseline."""
from __future__ import annotations

import argparse
import subprocess
import sys
from io import StringIO
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE = "19de95e"


def _read_csv(ref: str, path: str) -> pd.DataFrame:
    if ref == "local":
        return pd.read_csv(PROJECT_ROOT / path)
    text = subprocess.check_output(
        ["git", "show", f"{ref}:{path}"],
        cwd=PROJECT_ROOT,
        text=True,
    )
    return pd.read_csv(StringIO(text))


def audit_run(run_date: str, baseline: str) -> list[str]:
    issues: list[str] = []
    b1_path = f"data/runs/{run_date}/accounting/pnl_bucket_1.csv"
    bb_path = f"data/runs/{run_date}/accounting/pnl_by_bucket.csv"
    sym_path = f"data/runs/{run_date}/accounting/pnl_by_symbol.csv"

    for path in (b1_path, bb_path, sym_path):
        if not (PROJECT_ROOT / path).exists():
            issues.append(f"{run_date}: missing {path}")
            return issues

    b1_local = _read_csv("local", b1_path)
    b1_base = _read_csv(baseline, b1_path)
    bb_local = _read_csv("local", bb_path)
    bb_base = _read_csv(baseline, bb_path)

    top_local = b1_local.sort_values("total_pnl", ascending=False).iloc[0]
    top_base = b1_base.sort_values("total_pnl", ascending=False).iloc[0]
    if top_local["underlying"] != top_base["underlying"]:
        issues.append(
            f"{run_date}: B1 #1 changed {top_base['underlying']} -> {top_local['underlying']}"
        )

    b1_delta = float(b1_local["total_pnl"].sum() - b1_base["total_pnl"].sum())
    if abs(b1_delta) > 500.0:
        issues.append(f"{run_date}: B1 underlying sum drift ${b1_delta:,.0f}")

    book_delta = float(bb_local["total_pnl"].sum() - bb_base["total_pnl"].sum())
    if abs(book_delta) > 1.0:
        issues.append(f"{run_date}: book total drift ${book_delta:,.2f}")

    sym_local = _read_csv("local", sym_path)
    sym_base = _read_csv(baseline, sym_path)
    for u in ("SMCI", "IONQ", "MSTR"):
        b2_local = sym_local[(sym_local.underlying == u) & (sym_local.bucket == "bucket_2")]
        b2_base = sym_base[(sym_base.underlying == u) & (sym_base.bucket == "bucket_2")]
        local_syms = set(b2_local["symbol"])
        base_syms = set(b2_base["symbol"])
        if local_syms != base_syms:
            issues.append(f"{run_date}: {u} B2 symbols changed {sorted(base_syms)} -> {sorted(local_syms)}")
        net_delta = float(b2_local["total_pnl"].sum() - b2_base["total_pnl"].sum())
        if abs(net_delta) > 500.0:
            issues.append(f"{run_date}: {u} B2 net drift ${net_delta:,.0f}")

    rcat_local = float(b1_local.loc[b1_local["underlying"] == "RCAT", "total_pnl"].iloc[0])
    rcat_base = float(b1_base.loc[b1_base["underlying"] == "RCAT", "total_pnl"].iloc[0])
    if abs(rcat_local - rcat_base) > 50.0:
        issues.append(f"{run_date}: RCAT B1 drift ${rcat_local - rcat_base:,.0f}")

    return issues


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline", default=DEFAULT_BASELINE)
    ap.add_argument("--dates", nargs="*", default=["2026-05-19", "2026-05-20", "2026-05-21"])
    args = ap.parse_args()

    all_issues: list[str] = []
    for d in args.dates:
        all_issues.extend(audit_run(d, args.baseline))

    if all_issues:
        print(f"FAIL vs {args.baseline}:")
        for item in all_issues:
            print(f"  - {item}")
        return 1

    print(f"OK: {', '.join(args.dates)} match baseline {args.baseline}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
