"""Run manifest, NAV persistence, and dashboard verification helpers.

Shared by EOD, ``build_site``, and CI gate scripts so the data plane has one
contract for what constitutes a complete, publishable accounting run.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "data" / "runs"
PNL_HISTORY = REPO / "data" / "ledger" / "pnl_history.csv"

CONFIG_NAV_SOURCES = frozenset({"config:capital_usd", "MAGIS_NAV_USD", "default", "cli:--nav-usd"})
BROKER_NAV_PREFIXES = ("totals.json", "flex", "flex_positions")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _nav_from_config() -> tuple[float, str]:
    path = REPO / "config" / "strategy_config.yml"
    try:
        cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        cap = ((cfg.get("strategy") or {}).get("capital_usd"))
        if cap is not None and float(cap) > 0:
            return float(cap), "config:capital_usd"
    except Exception:
        pass
    return 800_000.0, "default"


def resolve_broker_nav(flex_dir: Path) -> dict[str, Any] | None:
    """Broker-derived NAV from Flex (equity tags or position percentOfNAV)."""
    try:
        from risk_dashboard.flex_parser import parse_flex_nav

        return parse_flex_nav(flex_dir)
    except Exception:
        return None


def patch_totals_nav(run_date: str, *, runs_root: Path | None = None) -> dict[str, Any]:
    """Ensure ``totals.json`` carries strategy NAV (``config:capital_usd`` by default).

    Broker Flex equity is intentionally *not* used: %-of-NAV metrics align with the
    sizing capital in ``strategy_config.yml``, not inferred full-account equity.
    """
    runs_root = runs_root or RUNS
    totals_path = runs_root / run_date / "accounting" / "totals.json"
    if not totals_path.is_file():
        raise FileNotFoundError(totals_path)
    totals = json.loads(totals_path.read_text(encoding="utf-8"))
    nav_raw = os.getenv("MAGIS_NAV_USD", "").strip()
    if nav_raw:
        totals["nav_usd"] = float(nav_raw)
        totals["nav_source"] = "MAGIS_NAV_USD"
    else:
        nav, src = _nav_from_config()
        totals["nav_usd"] = nav
        totals["nav_source"] = src

    totals_path.write_text(json.dumps(totals, indent=2), encoding="utf-8")
    return totals


def pin_screener_to_run(run_date: str, *, runs_root: Path | None = None) -> Path | None:
    """Copy repo-root screener CSV into the run folder for reproducible builds."""
    runs_root = runs_root or RUNS
    src = REPO / "data" / "etf_screened_today.csv"
    if not src.is_file():
        return None
    dest = runs_root / run_date / "etf_screened_today.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(src.read_bytes())
    return dest


def write_run_manifest(
    run_date: str,
    *,
    runs_root: Path | None = None,
    workflow_run_id: str | None = None,
    git_sha: str | None = None,
) -> dict[str, Any]:
    """Write ``data/runs/<date>/manifest.json`` with checksums and lineage."""
    runs_root = runs_root or RUNS
    run_dir = runs_root / run_date
    accounting = run_dir / "accounting"
    flex_dir = run_dir / "ibkr_flex"
    totals_path = accounting / "totals.json"
    if not totals_path.is_file():
        raise FileNotFoundError(f"missing accounting totals for {run_date}")

    totals = patch_totals_nav(run_date, runs_root=runs_root)
    screener = pin_screener_to_run(run_date, runs_root=runs_root)

    checksums: dict[str, str] = {}
    for rel in (
        "accounting/totals.json",
        "accounting/pnl_by_bucket.csv",
        "accounting/pnl_bucket_1.csv",
        "accounting/pnl_bucket_2.csv",
        "accounting/pnl_bucket_3.csv",
        "accounting/pnl_bucket_4.csv",
        "accounting/pnl_bucket_5.csv",
        "ibkr_flex/flex_positions.xml",
        "etf_screened_today.csv",
    ):
        p = run_dir / rel
        if p.is_file():
            checksums[rel] = _sha256_file(p)

    screener_rel: str | None = None
    if screener is not None:
        try:
            screener_rel = str(screener.relative_to(REPO))
        except ValueError:
            screener_rel = str(screener)
    manifest: dict[str, Any] = {
        "run_date": run_date,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha or os.getenv("GITHUB_SHA"),
        "workflow_run_id": workflow_run_id or os.getenv("GITHUB_RUN_ID"),
        "nav_usd": totals.get("nav_usd"),
        "nav_source": totals.get("nav_source"),
        "screener_csv": screener_rel,
        "checksums": checksums,
        "accounting_files": sorted(checksums.keys()),
    }
    out = run_dir / "manifest.json"
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def load_manifest(run_date: str, *, runs_root: Path | None = None) -> dict[str, Any]:
    path = (runs_root or RUNS) / run_date / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def discover_accounting_dates(runs_root: Path | None = None) -> list[str]:
    root = runs_root or RUNS
    if not root.is_dir():
        return []
    out: list[str] = []
    for child in root.iterdir():
        if child.is_dir() and (child / "accounting" / "totals.json").is_file():
            out.append(child.name)
    return sorted(out)


def daily_pnl_from_history(
    run_date: str,
    *,
    pnl_history_csv: Path | None = None,
) -> tuple[float | None, str | None]:
    """Day-over-day cumulative PnL move from ``pnl_history.csv`` (authoritative)."""
    path = pnl_history_csv or PNL_HISTORY
    if not path.is_file():
        return None, None
    df = pd.read_csv(path, usecols=["date", "total_pnl"])
    df["date"] = df["date"].astype(str)
    df = df.sort_values("date")
    rows = df[df["date"] <= str(run_date)]
    if rows.empty:
        return None, None
    cur_idx = rows.index[-1]
    cur = float(rows.loc[cur_idx, "total_pnl"] or 0.0)
    prior_rows = df[df["date"] < str(run_date)]
    if prior_rows.empty:
        return None, None
    prior_date = str(prior_rows.iloc[-1]["date"])
    prior = float(prior_rows.iloc[-1]["total_pnl"] or 0.0)
    return cur - prior, prior_date


def is_broker_nav_source(source: str | None) -> bool:
    s = str(source or "")
    if not s or s in CONFIG_NAV_SOURCES:
        return False
    return s.startswith("flex") or s.startswith("totals.json")


def is_config_nav_source(source: str | None) -> bool:
    s = str(source or "")
    if not s:
        return False
    return s in CONFIG_NAV_SOURCES or s.startswith("config")


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-date", required=True)
    ap.add_argument("--runs-root", default="data/runs")
    args = ap.parse_args(argv)
    manifest = write_run_manifest(args.run_date, runs_root=Path(args.runs_root).resolve())
    print(f"[run_data_contract] wrote manifest for {args.run_date} nav={manifest.get('nav_usd')} source={manifest.get('nav_source')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
