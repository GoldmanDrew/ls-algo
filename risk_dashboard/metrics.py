"""Risk metrics computed from accounting + Flex outputs.

Every metric here is mapped 1:1 to a row in ``Risk_Dashboard_Plan.md``
-- 4 (limits table). Each row carries its own threshold so the static
site can render a green / amber / red cell without re-implementing the
business logic.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from .flex_parser import (
    FlexBorrowFee,
    FlexPosition,
    parse_borrow_fee_details,
    parse_positions,
    summarize_borrow,
    summarize_positions,
)


# Limits live in code (not YAML) so a kill switch firing is auditable in
# the git history. Override at call-site only via an explicit kwarg.
DEFAULT_LIMITS: dict[str, dict[str, Any]] = {
    "gross_exposure_pct_nav": {"warn": 4.40, "hard": 5.00},
    "abs_net_beta": {"warn": 0.15, "hard": 0.20},
    "sleeve_drift_pp": {"warn": 10.0, "hard": 20.0},
    "es99_1d_pct_nav": {"warn": 0.012, "hard": 0.020},
    "borrow_apr_pct_b4": {"warn": 60.0, "hard": 90.0},
    "borrow_apr_pct_default": {"warn": 20.0, "hard": 30.0},
    "shares_outstanding_pct": {"warn": 15.0, "hard": 20.0},
    "aum_to_adv": {"warn": 0.30, "hard": 1.00},
    "premium_discount_zscore": {"warn": 3.0, "hard": 5.0},
}

SLEEVE_TARGET_WEIGHTS = {
    "bucket_1": 0.55,
    "bucket_2": 0.20,
    "bucket_3": None,
    "bucket_4": 0.25,
}


# ---------------------------------------------------------------------------
# Helpers


def _read_csv_or_empty(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _read_json_or_empty(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _classify(value: float | None, limits: dict[str, float], higher_is_worse: bool = True) -> str:
    """Return ``ok`` / ``warn`` / ``hard`` for a limit row."""
    if value is None:
        return "unknown"
    if higher_is_worse:
        if value >= limits["hard"]:
            return "hard"
        if value >= limits["warn"]:
            return "warn"
        return "ok"
    if value <= limits["hard"]:
        return "hard"
    if value <= limits["warn"]:
        return "warn"
    return "ok"


# ---------------------------------------------------------------------------
# Page 1 -- book summary


@dataclass
class BookSummary:
    nav_usd: float
    gross_notional_usd: float
    net_notional_usd: float
    long_notional_usd: float
    short_notional_usd: float
    gross_exposure_pct_nav: float
    net_exposure_pct_nav: float
    pnl_today_usd: float | None
    pnl_today_pct_nav: float | None
    sleeve_table: list[dict[str, Any]]
    breaches: list[dict[str, Any]]


def compute_book_summary(
    totals: dict[str, Any],
    pnl_by_bucket: pd.DataFrame,
    nav_usd: float,
    target_weights: dict[str, float | None] | None = None,
    limits: dict[str, dict[str, Any]] | None = None,
) -> BookSummary:
    limits = limits or DEFAULT_LIMITS
    target_weights = target_weights or SLEEVE_TARGET_WEIGHTS

    gross = float(totals.get("gross_exposure_total", 0.0) or 0.0)
    net = float(totals.get("net_exposure_total", 0.0) or 0.0)

    gross_pct = gross / nav_usd if nav_usd > 0 else 0.0
    net_pct = net / nav_usd if nav_usd > 0 else 0.0

    pnl_total = totals.get("total_pnl")
    pnl_pct = (pnl_total / nav_usd) if (nav_usd > 0 and pnl_total is not None) else None

    sleeve_table: list[dict[str, Any]] = []
    bucket_pnl = totals.get("bucket_pnl") or {}
    for bucket in ("bucket_1", "bucket_2", "bucket_3", "bucket_4"):
        gross_b = float(totals.get(f"gross_exposure_{bucket}", 0.0) or 0.0)
        net_b = float(totals.get(f"net_exposure_{bucket}", 0.0) or 0.0)
        actual_w = (gross_b / gross) if gross > 0 else 0.0
        target_w = target_weights.get(bucket)
        drift_pp = (
            (actual_w - target_w) * 100.0 if target_w is not None else None
        )
        drift_status = (
            _classify(abs(drift_pp), limits["sleeve_drift_pp"]) if drift_pp is not None else "ok"
        )
        sleeve_table.append(
            {
                "bucket": bucket,
                "gross_usd": gross_b,
                "net_usd": net_b,
                "actual_weight": actual_w,
                "target_weight": target_w,
                "drift_pp": drift_pp,
                "drift_status": drift_status,
                "pnl_usd": float(bucket_pnl.get(bucket, 0.0) or 0.0),
            }
        )

    breaches: list[dict[str, Any]] = []
    gross_status = _classify(gross_pct, limits["gross_exposure_pct_nav"])
    if gross_status != "ok":
        breaches.append(
            {
                "metric": "gross_exposure_pct_nav",
                "value": gross_pct,
                "status": gross_status,
                "limit": limits["gross_exposure_pct_nav"],
            }
        )

    return BookSummary(
        nav_usd=nav_usd,
        gross_notional_usd=gross,
        net_notional_usd=net,
        long_notional_usd=(gross + net) / 2.0,
        short_notional_usd=(net - gross) / 2.0,
        gross_exposure_pct_nav=gross_pct,
        net_exposure_pct_nav=net_pct,
        pnl_today_usd=pnl_total,
        pnl_today_pct_nav=pnl_pct,
        sleeve_table=sleeve_table,
        breaches=breaches,
    )


# ---------------------------------------------------------------------------
# Page 2/3 -- per-bucket detail


def compute_bucket_detail(
    bucket: str,
    pnl_csv: Path,
    net_exposure_csv: Path,
) -> dict[str, Any]:
    pnl = _read_csv_or_empty(pnl_csv)
    expo = _read_csv_or_empty(net_exposure_csv)

    rows: list[dict[str, Any]] = []
    if not pnl.empty:
        for _, row in pnl.iterrows():
            rows.append(
                {
                    "symbol": str(row.get("symbol", "")),
                    "description": str(row.get("description", "")),
                    "realized_pnl": float(row.get("realized_pnl", 0.0) or 0.0),
                    "unrealized_pnl": float(row.get("unrealized_pnl", 0.0) or 0.0),
                    "borrow_fees": float(row.get("borrow_fees", 0.0) or 0.0),
                    "short_credit_interest": float(row.get("short_credit_interest", 0.0) or 0.0),
                    "total_pnl": float(row.get("total_pnl", 0.0) or 0.0),
                }
            )
    rows.sort(key=lambda r: r["total_pnl"], reverse=True)

    expo_rows: list[dict[str, Any]] = []
    if not expo.empty:
        for _, row in expo.iterrows():
            expo_rows.append(
                {
                    "underlying": str(row.get("underlying", "")),
                    "symbols": str(row.get("symbols", "")),
                    "net_notional_usd": float(row.get("net_notional_usd", 0.0) or 0.0),
                    "gross_notional_usd": float(row.get("gross_notional_usd", 0.0) or 0.0),
                    "n_legs": int(row.get("n_legs", 0) or 0),
                }
            )
    expo_rows.sort(key=lambda r: r["gross_notional_usd"], reverse=True)

    return {
        "bucket": bucket,
        "pnl_rows": rows,
        "exposure_rows": expo_rows,
        "n_pnl_rows": len(rows),
        "n_exposure_rows": len(expo_rows),
        "winners": rows[:5],
        "losers": list(reversed(rows[-5:])) if len(rows) >= 5 else list(reversed(rows)),
    }


# ---------------------------------------------------------------------------
# Page 6 -- borrow / microstructure


def compute_borrow_panel(
    flex_borrow_xml: Path,
    flex_positions_xml: Path,
    limits: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    limits = limits or DEFAULT_LIMITS
    borrow_rows: list[FlexBorrowFee] = parse_borrow_fee_details(flex_borrow_xml)
    positions: list[FlexPosition] = parse_positions(flex_positions_xml)

    summary = summarize_borrow(borrow_rows)
    pos_summary = summarize_positions(positions)

    breaches: list[dict[str, Any]] = []
    for row in summary["names_over_30pct"]:
        status = _classify(row["fee_rate_pct"], limits["borrow_apr_pct_default"])
        if status != "ok":
            breaches.append({"metric": "borrow_apr", **row, "status": status})
    return {
        "borrow": summary,
        "positions": pos_summary,
        "breaches": breaches,
    }


# ---------------------------------------------------------------------------
# Top-level snapshot assembler


@dataclass
class RiskSnapshot:
    run_date: str
    generated_at_utc: str
    nav_usd: float
    book: BookSummary
    buckets: dict[str, dict[str, Any]] = field(default_factory=dict)
    borrow_panel: dict[str, Any] = field(default_factory=dict)
    universe_counts: dict[str, Any] = field(default_factory=dict)
    raw_totals: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_date": self.run_date,
            "generated_at_utc": self.generated_at_utc,
            "nav_usd": self.nav_usd,
            "book": {
                "nav_usd": self.book.nav_usd,
                "gross_notional_usd": self.book.gross_notional_usd,
                "net_notional_usd": self.book.net_notional_usd,
                "long_notional_usd": self.book.long_notional_usd,
                "short_notional_usd": self.book.short_notional_usd,
                "gross_exposure_pct_nav": self.book.gross_exposure_pct_nav,
                "net_exposure_pct_nav": self.book.net_exposure_pct_nav,
                "pnl_today_usd": self.book.pnl_today_usd,
                "pnl_today_pct_nav": self.book.pnl_today_pct_nav,
                "sleeve_table": self.book.sleeve_table,
                "breaches": self.book.breaches,
            },
            "buckets": self.buckets,
            "borrow_panel": self.borrow_panel,
            "universe_counts": self.universe_counts,
            "raw_totals": self.raw_totals,
            "limits": DEFAULT_LIMITS,
            "sleeve_target_weights": SLEEVE_TARGET_WEIGHTS,
        }


def build_snapshot(
    run_date: str,
    runs_root: Path,
    nav_usd: float,
    *,
    generated_at_utc: str,
    limits: dict[str, dict[str, Any]] | None = None,
) -> RiskSnapshot:
    """Build a full snapshot from a ``data/runs/<run_date>`` folder."""
    run_dir = runs_root / run_date
    accounting = run_dir / "accounting"
    flex = run_dir / "ibkr_flex"

    totals = _read_json_or_empty(accounting / "totals.json")
    pnl_by_bucket = _read_csv_or_empty(accounting / "pnl_by_bucket.csv")

    book = compute_book_summary(totals, pnl_by_bucket, nav_usd=nav_usd, limits=limits)

    buckets: dict[str, dict[str, Any]] = {}
    for bucket in ("bucket_1", "bucket_2", "bucket_3", "bucket_4"):
        buckets[bucket] = compute_bucket_detail(
            bucket=bucket,
            pnl_csv=accounting / f"pnl_{bucket}.csv",
            net_exposure_csv=accounting / f"net_exposure_{bucket}.csv",
        )

    borrow_panel = compute_borrow_panel(
        flex_borrow_xml=flex / "flex_borrow_fee_details.xml",
        flex_positions_xml=flex / "flex_positions.xml",
        limits=limits,
    )

    universe_counts = {
        "kept_symbols": totals.get("kept_symbols"),
        "kept_underlyings": totals.get("kept_underlyings"),
        "universe_allowed_etfs": totals.get("universe_allowed_etfs"),
        "universe_allowed_underlyings": totals.get("universe_allowed_underlyings"),
    }

    return RiskSnapshot(
        run_date=run_date,
        generated_at_utc=generated_at_utc,
        nav_usd=nav_usd,
        book=book,
        buckets=buckets,
        borrow_panel=borrow_panel,
        universe_counts=universe_counts,
        raw_totals=totals,
    )
