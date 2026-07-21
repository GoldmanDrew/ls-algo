#!/usr/bin/env python3
"""Materialize missing data/runs/<date> screener/plan archives from git tip history.

Point-in-time production replay reads ``data/runs/<YYYY-MM-DD>/etf_screened_today.csv``
and ``proposed_trades.csv``. When those files are missing but the tip files
``data/etf_screened_today.csv`` / ``data/proposed_trades.csv`` were committed on
that calendar day, this script copies the git blob into the run directory.

Rules (hard):
  - Never invent a day with no commit touching the tip path.
  - Never copy the working-tree tip file into a past date (git blob only).
  - Never overwrite an existing archive file unless ``--force``.
  - Default is ``--dry-run`` (no writes).

Usage:
  python scripts/backfill_screened_history.py
  python scripts/backfill_screened_history.py --from 2026-04-27 --to 2026-06-25 --apply
  python scripts/backfill_screened_history.py --artifacts screened --apply --force
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Iterable

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

RUNS_DIR = REPO / "data" / "runs"

ARTIFACTS: dict[str, tuple[str, str]] = {
    # logical name -> (tip path relative to repo, filename under data/runs/<date>/)
    "screened": ("data/etf_screened_today.csv", "etf_screened_today.csv"),
    "plans": ("data/proposed_trades.csv", "proposed_trades.csv"),
}


@dataclass(frozen=True)
class ManifestRow:
    date: str
    artifact: str
    source_sha: str | None
    source_commit_date: str | None
    action: str  # wrote | would_write | skipped_exists | no_blob | skipped_out_of_range


def _run_git(args: list[str], *, cwd: Path = REPO) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"git {' '.join(args)} failed ({proc.returncode}): {proc.stderr.strip()}"
        )
    return proc.stdout


def list_tip_commits(tip_rel: str, *, cwd: Path = REPO) -> list[tuple[str, str]]:
    """Return [(sha, commit_iso)] newest-first for commits that touch tip_rel."""
    out = _run_git(["log", "--format=%H %cI", "--", tip_rel], cwd=cwd)
    rows: list[tuple[str, str]] = []
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        sha, ts = line.split(" ", 1)
        rows.append((sha, ts.strip()))
    return rows


def last_commit_by_day(commits: Iterable[tuple[str, str]]) -> dict[str, tuple[str, str]]:
    """Map YYYY-MM-DD -> (sha, commit_iso). Newest commit of the day wins."""
    by_day: dict[str, tuple[str, str]] = {}
    for sha, ts in commits:
        day = ts[:10]
        if day not in by_day:
            by_day[day] = (sha, ts)
    return by_day


def git_show_blob(sha: str, tip_rel: str, *, cwd: Path = REPO) -> bytes | None:
    proc = subprocess.run(
        ["git", "show", f"{sha}:{tip_rel}"],
        cwd=str(cwd),
        check=False,
        capture_output=True,
    )
    if proc.returncode != 0:
        return None
    return proc.stdout


def parse_artifacts(spec: str) -> list[str]:
    names = [p.strip().lower() for p in spec.split(",") if p.strip()]
    if not names:
        raise ValueError("artifacts list is empty")
    unknown = [n for n in names if n not in ARTIFACTS]
    if unknown:
        raise ValueError(f"unknown artifacts {unknown}; choose from {sorted(ARTIFACTS)}")
    return names


def materialize_archives(
    *,
    artifacts: list[str],
    date_from: date | None,
    date_to: date | None,
    apply: bool,
    force: bool,
    runs_dir: Path = RUNS_DIR,
    repo: Path = REPO,
) -> list[ManifestRow]:
    """Dry-run or apply git→runs materialization. Returns manifest rows."""
    rows: list[ManifestRow] = []
    for artifact in artifacts:
        tip_rel, dest_name = ARTIFACTS[artifact]
        by_day = last_commit_by_day(list_tip_commits(tip_rel, cwd=repo))
        for day in sorted(by_day):
            if date_from and day < date_from.isoformat():
                continue
            if date_to and day > date_to.isoformat():
                continue
            sha, commit_ts = by_day[day]
            dest = runs_dir / day / dest_name
            if dest.is_file() and not force:
                rows.append(
                    ManifestRow(
                        date=day,
                        artifact=artifact,
                        source_sha=sha,
                        source_commit_date=commit_ts,
                        action="skipped_exists",
                    )
                )
                continue
            blob = git_show_blob(sha, tip_rel, cwd=repo)
            if blob is None:
                rows.append(
                    ManifestRow(
                        date=day,
                        artifact=artifact,
                        source_sha=sha,
                        source_commit_date=commit_ts,
                        action="no_blob",
                    )
                )
                continue
            if apply:
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(blob)
                action = "wrote"
            else:
                action = "would_write"
            rows.append(
                ManifestRow(
                    date=day,
                    artifact=artifact,
                    source_sha=sha,
                    source_commit_date=commit_ts,
                    action=action,
                )
            )
    return rows


def summarize(rows: list[ManifestRow]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for r in rows:
        counts[r.action] = counts.get(r.action, 0) + 1
    return counts


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--artifacts",
        default="screened,plans",
        help="Comma-separated: screened,plans (default both)",
    )
    ap.add_argument("--from", dest="date_from", default=None, help="Inclusive YYYY-MM-DD")
    ap.add_argument("--to", dest="date_to", default=None, help="Inclusive YYYY-MM-DD")
    ap.add_argument(
        "--apply",
        action="store_true",
        help="Write files (default is dry-run)",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing archive files",
    )
    ap.add_argument(
        "--manifest-out",
        type=Path,
        default=None,
        help="Optional JSON path for the action manifest",
    )
    ap.add_argument(
        "--runs-dir",
        type=Path,
        default=RUNS_DIR,
        help="Override data/runs directory (tests)",
    )
    args = ap.parse_args(argv)

    artifacts = parse_artifacts(args.artifacts)
    d_from = date.fromisoformat(args.date_from) if args.date_from else None
    d_to = date.fromisoformat(args.date_to) if args.date_to else None
    if d_from and d_to and d_from > d_to:
        raise SystemExit("--from must be <= --to")

    rows = materialize_archives(
        artifacts=artifacts,
        date_from=d_from,
        date_to=d_to,
        apply=bool(args.apply),
        force=bool(args.force),
        runs_dir=Path(args.runs_dir),
        repo=REPO,
    )
    summary = summarize(rows)
    payload = {
        "ok": True,
        "dry_run": not bool(args.apply),
        "force": bool(args.force),
        "artifacts": artifacts,
        "from": args.date_from,
        "to": args.date_to,
        "summary": summary,
        "rows": [asdict(r) for r in rows],
        "note": (
            "Commit calendar date stamps the archive day; tip working tree is never copied. "
            "Aggregate B4 book production start remains 2026-02-27; pair inception research "
            "covers listing→plan gaps."
        ),
    }
    text = json.dumps(payload, indent=2) + "\n"
    if args.manifest_out:
        args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
        args.manifest_out.write_text(text, encoding="utf-8")
        print(f"wrote manifest {args.manifest_out}")
    print(json.dumps({"ok": True, "dry_run": payload["dry_run"], "summary": summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
