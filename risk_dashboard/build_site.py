"""Build the risk-dashboard JSON snapshot consumed by the static site.

Usage (CI / local):

    python -m risk_dashboard.build_site \
        --run-date 2026-05-15 \
        --nav-usd 800000 \
        --runs-root data/runs \
        --out-dir risk_dashboard/data

After a full accounting restate, rebuild every dated snapshot (oldest first):

    python -m risk_dashboard.build_site --all-dates --runs-root data/runs --out-dir risk_dashboard/data

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

# Default NAV denominator falls back to the strategy capital base in
# ``config/strategy_config.yml`` (``strategy.capital_usd``) when totals.json /
# Flex equity are unavailable. This keeps the dashboard %-of-NAV figures aligned
# with the sizing capital the book is actually run against.
CONFIG_NAV_FALLBACK_USD = 800_000.0


def _nav_from_config(config_yml: Path | None = None) -> tuple[float, str]:
    """Return (nav_usd, source_label) read from strategy_config capital_usd."""
    path = config_yml or Path("config/strategy_config.yml")
    try:
        import yaml  # local import; pyyaml is already a project dep

        cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        cap = ((cfg.get("strategy") or {}).get("capital_usd"))
        if cap is not None and float(cap) > 0:
            return float(cap), "config:capital_usd"
    except Exception:
        pass
    return CONFIG_NAV_FALLBACK_USD, "default"

# Optional auxiliary panels merged into the snapshot if present. Each is a
# standalone JSON produced by a research/EOD generator (decoupled from the
# heavy accounting build) and keyed onto the payload under the same name.
AUX_PANELS = {
    "bucket4_risk_sim": "bucket4_risk_sim.json",
}


def _merge_aux_panels(payload: dict, *search_dirs: Path) -> None:
    """Attach auxiliary panel JSONs (e.g. the B4 risk simulator) to the payload.

    Looks for each file across ``search_dirs`` in order; first hit wins. Missing
    or unreadable files are skipped silently so the core build never fails on an
    optional panel.
    """
    for key, fname in AUX_PANELS.items():
        for d in search_dirs:
            fpath = d / fname
            if not fpath.is_file():
                continue
            try:
                payload[key] = json.loads(fpath.read_text(encoding="utf-8"))
            except Exception:
                pass
            break


def _discover_default_run_date(runs_root: Path) -> str:
    dates = _discover_accounting_run_dates(runs_root)
    if not dates:
        raise SystemExit(f"no run-date folders under {runs_root}")
    return dates[-1]


def _discover_accounting_run_dates(runs_root: Path) -> list[str]:
    """Run dates with accounting outputs, oldest first."""
    if not runs_root.is_dir():
        return []
    dates: list[str] = []
    for child in runs_root.iterdir():
        if not child.is_dir() or not child.name[:4].isdigit():
            continue
        if (child / "accounting" / "totals.json").is_file():
            dates.append(child.name)
    return sorted(dates)


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
        "pnl_cum_usd": book.get("pnl_today_usd"),
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
        "pnl_cum_usd",
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


def _attach_daily_pnl(payload: dict, deltas: dict, run_date: str) -> None:
    """Add day-over-day PnL to the book block.

    Primary: consecutive rows in ``pnl_history.csv`` (robust if a snapshot day
    was skipped). Fallback: delta vs the prior published snapshot.
    """
    book = payload.get("book") or {}
    nav = book.get("nav_usd") or 0.0
    daily: float | None = None
    prior_date: str | None = None
    source = "unavailable"
    try:
        from scripts.run_data_contract import daily_pnl_from_history

        daily, prior_date = daily_pnl_from_history(run_date)
        if daily is not None:
            source = "pnl_history.csv"
    except Exception:
        pass
    if daily is None:
        daily = deltas.get("delta_pnl_cum_usd")
        prior_date = deltas.get("prior_run_date")
        if daily is not None:
            source = "snapshot_delta"
    book["pnl_daily_usd"] = daily
    book["pnl_daily_pct_nav"] = (daily / nav) if (daily is not None and nav) else None
    book["pnl_daily_prior_run_date"] = prior_date
    book["pnl_daily_source"] = source
    payload["book"] = book


def _attach_lineage(payload: dict, run_date: str, runs_root: Path) -> None:
    try:
        from scripts.run_data_contract import load_manifest

        payload["manifest"] = load_manifest(run_date, runs_root=runs_root)
    except FileNotFoundError:
        payload["manifest"] = None
    payload.setdefault("data_quality", {})
    dq = payload["data_quality"]
    if isinstance(dq, dict):
        manifest = payload.get("manifest") or {}
        dq["lineage"] = {
            "manifest_present": manifest != {},
            "nav_source": manifest.get("nav_source") or payload.get("nav_source"),
            "git_sha": manifest.get("git_sha"),
            "workflow_run_id": manifest.get("workflow_run_id"),
            "checksum_count": len(manifest.get("checksums") or {}),
        }


def _attach_freshness(payload: dict, runs_root: Path, run_date: str) -> None:
    """Mark whether this snapshot's run_date is the latest accounting run."""
    latest = None
    dates = _discover_accounting_run_dates(runs_root)
    if dates:
        latest = dates[-1]
    age_days = None
    try:
        gen = payload.get("generated_at_utc") or ""
        gen_date = datetime.fromisoformat(gen.replace("Z", "+00:00")).date()
        rd = datetime.strptime(str(run_date), "%Y-%m-%d").date()
        age_days = (gen_date - rd).days
    except Exception:
        age_days = None
    payload["freshness"] = {
        "run_date": run_date,
        "latest_accounting_run_date": latest,
        "is_latest": (latest is None or run_date == latest),
        "data_age_days": age_days,
    }


def build_run_snapshot(
    *,
    run_date: str,
    runs_root: Path,
    out_dir: Path,
    nav_usd: float,
    screener_csv: Path | None,
    generated_at_utc: str | None = None,
    write_latest: bool = True,
    nav_source_hint: str = "MAGIS_NAV_USD",
) -> Path:
    """Build one dated snapshot (+ optional latest.json) from accounting outputs."""
    snap = build_snapshot(
        run_date=run_date,
        runs_root=runs_root,
        nav_usd=nav_usd,
        generated_at_utc=generated_at_utc or datetime.now(timezone.utc).isoformat(),
        screener_csv=screener_csv,
        nav_source_hint=nav_source_hint,
    )
    payload = snap.to_dict()

    # Optional standalone panels (B4 risk simulator, etc.). Prefer a per-run-date
    # copy under the run folder, else the shared dashboard data dir.
    _merge_aux_panels(payload, runs_root / run_date, out_dir, Path("risk_dashboard/data"))

    history = _load_history(out_dir, current_date=run_date)
    current_point = _extract_history_point(payload)
    payload["history"] = history + [current_point]
    payload["deltas"] = _compute_deltas(current_point, history[-1] if history else None)
    _attach_daily_pnl(payload, payload["deltas"], run_date)
    _attach_freshness(payload, runs_root, run_date)
    _attach_lineage(payload, run_date, runs_root)

    snap_path = out_dir / f"{run_date}.json"
    _write_json(snap_path, payload)
    if write_latest:
        _write_json(out_dir / "latest.json", payload)
    return snap_path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-date", default=os.getenv("RUN_DATE"), help="YYYY-MM-DD; defaults to latest under --runs-root")
    ap.add_argument("--runs-root", default="data/runs", help="root of run-date folders")
    ap.add_argument(
        "--nav-usd",
        type=float,
        default=None,
        help="Account NAV used as the denominator for %%-of-NAV metrics. "
        "Defaults to strategy.capital_usd from config/strategy_config.yml; "
        "override via env MAGIS_NAV_USD or this flag.",
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
    ap.add_argument(
        "--all-dates",
        action="store_true",
        help="Rebuild every run date with accounting/totals.json, oldest first "
        "(use after a full accounting restate).",
    )
    ap.add_argument(
        "--fail-if-stale",
        action="store_true",
        help="Exit non-zero if the built run_date is not the latest accounting run "
        "(CI guard against publishing a stale snapshot).",
    )
    args = ap.parse_args(argv)

    runs_root = Path(args.runs_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    screener_csv = Path(args.screener_csv).resolve() if args.screener_csv else None

    # NAV denominator precedence: explicit --nav-usd > env MAGIS_NAV_USD >
    # config strategy.capital_usd > hard default. build_snapshot still prefers
    # totals.json / Flex equity when present.
    if args.nav_usd is not None:
        nav_usd, nav_source_hint = float(args.nav_usd), "cli:--nav-usd"
    elif os.getenv("MAGIS_NAV_USD"):
        nav_usd, nav_source_hint = float(os.environ["MAGIS_NAV_USD"]), "MAGIS_NAV_USD"
    else:
        nav_usd, nav_source_hint = _nav_from_config()
    print(f"[risk_dashboard] NAV fallback = ${nav_usd:,.0f} (source hint: {nav_source_hint})")

    def _resolve_run_date_and_screener(rd: str) -> tuple[str, Path | None]:
        pinned = runs_root / rd / "etf_screened_today.csv"
        if pinned.is_file():
            return rd, pinned
        if screener_csv and screener_csv.is_file():
            return rd, screener_csv
        return rd, screener_csv

    if args.all_dates:
        run_dates = _discover_accounting_run_dates(runs_root)
        if not run_dates:
            raise SystemExit(f"no accounting runs under {runs_root}")
        print(f"[risk_dashboard] rebuilding {len(run_dates)} snapshot(s): {run_dates[0]} -> {run_dates[-1]}")
        for i, run_date in enumerate(run_dates, start=1):
            _, sc_path = _resolve_run_date_and_screener(run_date)
            snap_path = build_run_snapshot(
                run_date=run_date,
                runs_root=runs_root,
                out_dir=out_dir,
                nav_usd=nav_usd,
                screener_csv=sc_path,
                write_latest=(i == len(run_dates)),
                nav_source_hint=nav_source_hint,
            )
            print(f"[risk_dashboard] ({i}/{len(run_dates)}) wrote {snap_path}")
    else:
        run_date = args.run_date or _discover_default_run_date(runs_root)
        _, sc_path = _resolve_run_date_and_screener(run_date)
        snap_path = build_run_snapshot(
            run_date=run_date,
            runs_root=runs_root,
            out_dir=out_dir,
            nav_usd=nav_usd,
            screener_csv=sc_path,
            write_latest=True,
            nav_source_hint=nav_source_hint,
        )
        print(f"[risk_dashboard] wrote {snap_path}")
        print(f"[risk_dashboard] wrote {out_dir / 'latest.json'}")

        latest_acct = _discover_accounting_run_dates(runs_root)
        newest = latest_acct[-1] if latest_acct else None
        if newest and run_date != newest:
            msg = (
                f"[risk_dashboard] WARNING: built {run_date} but newest accounting "
                f"run is {newest} (snapshot is stale)."
            )
            print(msg)
            if args.fail_if_stale:
                raise SystemExit(msg)

    manifest = _build_index(out_dir)
    index_path = out_dir / "index.json"
    _write_json(index_path, manifest)
    print(f"[risk_dashboard] wrote {index_path} ({len(manifest['runs'])} runs)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
