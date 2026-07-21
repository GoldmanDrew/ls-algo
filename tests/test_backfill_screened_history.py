"""Tests for scripts/backfill_screened_history.py."""
from __future__ import annotations

from datetime import date
from pathlib import Path
from unittest.mock import patch

from scripts.backfill_screened_history import materialize_archives, summarize


def test_dry_run_does_not_write(tmp_path: Path):
    runs = tmp_path / "runs"
    commits = {
        "2026-05-01": ("abc123", "2026-05-01T12:00:00-04:00"),
    }
    blob = b"ETF,Underlying\nQBTZ,QBTS\n"

    with (
        patch(
            "scripts.backfill_screened_history.list_tip_commits",
            return_value=[("abc123", "2026-05-01T12:00:00-04:00")],
        ),
        patch(
            "scripts.backfill_screened_history.git_show_blob",
            return_value=blob,
        ),
        patch(
            "scripts.backfill_screened_history.last_commit_by_day",
            return_value=commits,
        ),
    ):
        rows = materialize_archives(
            artifacts=["screened"],
            date_from=date(2026, 4, 27),
            date_to=date(2026, 6, 25),
            apply=False,
            force=False,
            runs_dir=runs,
            repo=tmp_path,
        )

    assert summarize(rows)["would_write"] == 1
    assert not (runs / "2026-05-01" / "etf_screened_today.csv").exists()


def test_apply_writes_missing_only(tmp_path: Path):
    runs = tmp_path / "runs"
    existing = runs / "2026-05-02" / "etf_screened_today.csv"
    existing.parent.mkdir(parents=True)
    existing.write_text("already\n", encoding="utf-8")

    by_day = {
        "2026-05-01": ("aaa", "2026-05-01T10:00:00Z"),
        "2026-05-02": ("bbb", "2026-05-02T10:00:00Z"),
    }
    blobs = {
        "aaa": b"new1\n",
        "bbb": b"new2\n",
    }

    def _show(sha: str, tip_rel: str, *, cwd: Path):
        return blobs.get(sha)

    with (
        patch(
            "scripts.backfill_screened_history.list_tip_commits",
            return_value=[],
        ),
        patch(
            "scripts.backfill_screened_history.last_commit_by_day",
            return_value=by_day,
        ),
        patch(
            "scripts.backfill_screened_history.git_show_blob",
            side_effect=_show,
        ),
    ):
        rows = materialize_archives(
            artifacts=["screened"],
            date_from=None,
            date_to=None,
            apply=True,
            force=False,
            runs_dir=runs,
            repo=tmp_path,
        )

    actions = {r.date: r.action for r in rows}
    assert actions["2026-05-01"] == "wrote"
    assert actions["2026-05-02"] == "skipped_exists"
    assert (runs / "2026-05-01" / "etf_screened_today.csv").read_bytes() == b"new1\n"
    assert existing.read_text(encoding="utf-8") == "already\n"


def test_force_overwrites(tmp_path: Path):
    runs = tmp_path / "runs"
    dest = runs / "2026-05-01" / "etf_screened_today.csv"
    dest.parent.mkdir(parents=True)
    dest.write_text("old\n", encoding="utf-8")

    with (
        patch(
            "scripts.backfill_screened_history.last_commit_by_day",
            return_value={"2026-05-01": ("ccc", "2026-05-01T10:00:00Z")},
        ),
        patch(
            "scripts.backfill_screened_history.list_tip_commits",
            return_value=[],
        ),
        patch(
            "scripts.backfill_screened_history.git_show_blob",
            return_value=b"fresh\n",
        ),
    ):
        rows = materialize_archives(
            artifacts=["screened"],
            date_from=date(2026, 5, 1),
            date_to=date(2026, 5, 1),
            apply=True,
            force=True,
            runs_dir=runs,
            repo=tmp_path,
        )

    assert rows[0].action == "wrote"
    assert dest.read_bytes() == b"fresh\n"


def test_no_blob_recorded(tmp_path: Path):
    with (
        patch(
            "scripts.backfill_screened_history.last_commit_by_day",
            return_value={"2026-05-01": ("ddd", "2026-05-01T10:00:00Z")},
        ),
        patch(
            "scripts.backfill_screened_history.list_tip_commits",
            return_value=[],
        ),
        patch(
            "scripts.backfill_screened_history.git_show_blob",
            return_value=None,
        ),
    ):
        rows = materialize_archives(
            artifacts=["screened"],
            date_from=None,
            date_to=None,
            apply=True,
            force=False,
            runs_dir=tmp_path / "runs",
            repo=tmp_path,
        )
    assert rows[0].action == "no_blob"
