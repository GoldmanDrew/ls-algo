"""US equity session calendar for scheduled EOD / dashboard pipelines.

GitHub cron is UTC-only; this module decides whether a scheduled run should
execute and which NY calendar date the run represents (same as EOD today).
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

NY = ZoneInfo("America/New_York")

# NYSE full-day closures (extend annually). Partial sessions treated as open.
NYSE_HOLIDAYS: frozenset[date] = frozenset(
    {
        date(2025, 1, 1),
        date(2025, 1, 20),
        date(2025, 2, 17),
        date(2025, 4, 18),
        date(2025, 5, 26),
        date(2025, 6, 19),
        date(2025, 7, 4),
        date(2025, 9, 1),
        date(2025, 11, 27),
        date(2025, 12, 25),
        date(2026, 1, 1),
        date(2026, 1, 19),
        date(2026, 2, 16),
        date(2026, 4, 3),
        date(2026, 5, 25),
        date(2026, 6, 19),
        date(2026, 7, 3),
        date(2026, 9, 7),
        date(2026, 11, 26),
        date(2026, 12, 25),
        date(2027, 1, 1),
        date(2027, 1, 18),
        date(2027, 2, 15),
        date(2027, 3, 26),
        date(2027, 5, 31),
        date(2027, 6, 18),
        date(2027, 7, 5),
        date(2027, 9, 6),
        date(2027, 11, 25),
        date(2027, 12, 24),
    }
)


def is_us_equity_session(d: date) -> bool:
    """True when ``d`` is a regular NYSE session day (Mon–Fri, not a full closure)."""
    if d.weekday() >= 5:
        return False
    return d not in NYSE_HOLIDAYS


def ny_today(when: datetime | None = None) -> date:
    when = when or datetime.now(NY)
    return when.date()


def scheduled_run_context(when: datetime | None = None) -> dict[str, str | bool]:
    """Context for a 06:00 UTC scheduled workflow kickoff."""
    when = when or datetime.now(NY)
    run_date = ny_today(when)
    return {
        "run_date": run_date.isoformat(),
        "should_run": is_us_equity_session(run_date),
        "ny_weekday": run_date.strftime("%A"),
    }


def prior_accounting_session(before: date, runs_root_dates: list[str] | None = None) -> date | None:
    """Previous US equity session on or before ``before`` (walks calendar)."""
    d = before - timedelta(days=1)
    for _ in range(14):
        if is_us_equity_session(d):
            return d
        d -= timedelta(days=1)
    if runs_root_dates:
        prior = [x for x in runs_root_dates if x < before.isoformat()]
        if prior:
            return date.fromisoformat(prior[-1])
    return None
