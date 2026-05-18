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
    "gross_exposure_pct_nav": {"warn": 2.25, "hard": 2.50},
    "abs_net_beta": {"warn": 0.15, "hard": 0.20},
    "sleeve_drift_pp": {"warn": 5.0, "hard": 10.0},
    "es99_1d_pct_nav": {"warn": 0.012, "hard": 0.020},
    "scenario_loss_pct_nav": {"warn": -0.05, "hard": -0.07},
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


def _clean_str(value: Any) -> str:
    if value is None:
        return ""
    if pd.isna(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none"} else text


def _first_nonblank(*values: Any) -> str:
    for value in values:
        text = _clean_str(value)
        if text:
            return text
    return ""


def _csv_columns(path: Path) -> list[str]:
    if not path.is_file():
        return []
    try:
        return list(pd.read_csv(path, nrows=0).columns)
    except Exception:
        return []


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


def _limit_row(
    metric: str,
    value: float | None,
    *,
    limits: dict[str, dict[str, Any]],
    higher_is_worse: bool = True,
    source: str,
    action: str,
) -> dict[str, Any]:
    limit = limits[metric]
    return {
        "metric": metric,
        "value": value,
        "status": _classify(value, limit, higher_is_worse=higher_is_worse),
        "limit": limit,
        "source": source,
        "action": action,
    }


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
                "source": "Risk_Dashboard_Plan.md §4.2",
                "action": "Reduce gross or explicitly approve temporary breach before adding risk.",
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
            underlying = _first_nonblank(row.get("underlying"), row.get("symbol"))
            symbols = _first_nonblank(row.get("symbols"), row.get("symbol"), underlying)
            symbol = _first_nonblank(row.get("symbol"), underlying, symbols)
            description = _first_nonblank(row.get("description"), symbols, underlying, symbol)
            display_name = _first_nonblank(underlying, symbol, symbols, description)
            rows.append(
                {
                    "symbol": symbol,
                    "underlying": underlying,
                    "symbols": symbols,
                    "description": description,
                    "display_name": display_name,
                    "primary_key": display_name,
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
            underlying = _first_nonblank(row.get("underlying"), row.get("symbol"))
            symbols = _first_nonblank(row.get("symbols"), row.get("symbol"), underlying)
            expo_rows.append(
                {
                    "underlying": underlying,
                    "symbols": symbols,
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
            breaches.append(
                {
                    "metric": "borrow_apr",
                    **row,
                    "status": status,
                    "limit": limits["borrow_apr_pct_default"],
                    "source": "Risk_Dashboard_Plan.md §4.2 / §5.2",
                    "action": "Review borrow economics; reduce or drop names above hard cap.",
                }
            )
    return {
        "borrow": summary,
        "positions": pos_summary,
        "breaches": breaches,
    }


# ---------------------------------------------------------------------------
# Data contract + scenario risk


def compute_data_quality(
    *,
    accounting_dir: Path,
    flex_dir: Path,
    buckets: dict[str, dict[str, Any]],
    totals: dict[str, Any] | None = None,
    run_date: str,
) -> dict[str, Any]:
    """Report source/schema issues that would make the dashboard misleading."""
    source_specs: list[dict[str, Any]] = [
        {"name": "totals", "path": accounting_dir / "totals.json", "type": "json"},
        {"name": "pnl_by_symbol", "path": accounting_dir / "pnl_by_symbol.csv", "required": ["symbol", "underlying", "bucket", "total_pnl"]},
        {"name": "pnl_by_underlying", "path": accounting_dir / "pnl_by_underlying.csv", "required": ["total_pnl"], "required_any": [["underlying", "symbol"], ["symbols", "description", "symbol"]]},
        {"name": "flex_positions", "path": flex_dir / "flex_positions.xml", "type": "xml"},
        {"name": "flex_borrow_fee_details", "path": flex_dir / "flex_borrow_fee_details.xml", "type": "xml"},
    ]
    for bucket in ("bucket_1", "bucket_2", "bucket_3", "bucket_4"):
        source_specs.extend(
            [
                {
                    "name": f"pnl_{bucket}",
                    "path": accounting_dir / f"pnl_{bucket}.csv",
                    "required": ["total_pnl"],
                    "required_any": [["underlying", "symbol"], ["symbols", "description", "symbol"]],
                },
                {
                    "name": f"net_exposure_{bucket}",
                    "path": accounting_dir / f"net_exposure_{bucket}.csv",
                    "required": ["net_notional_usd", "gross_notional_usd", "n_legs"],
                    "required_any": [["underlying", "symbol"], ["symbols", "description", "symbol"]],
                },
            ]
        )

    sources: list[dict[str, Any]] = []
    missing_sources = 0
    missing_columns = 0
    for spec in source_specs:
        path = spec["path"]
        exists = path.is_file()
        if not exists:
            missing_sources += 1
        columns = _csv_columns(path) if spec.get("type", "csv") == "csv" else []
        required = spec.get("required", [])
        missing = [c for c in required if c not in columns]
        for group in spec.get("required_any", []):
            if not any(c in columns for c in group):
                missing.append(" or ".join(group))
        missing_columns += len(missing)
        sources.append(
            {
                "name": spec["name"],
                "path": str(path),
                "exists": exists,
                "columns": columns,
                "missing_required_columns": missing,
            }
        )

    blank_render_fields: list[dict[str, Any]] = []
    row_counts: dict[str, Any] = {}
    for bucket, detail in buckets.items():
        row_counts[bucket] = {
            "pnl_rows": detail.get("n_pnl_rows", 0),
            "exposure_rows": detail.get("n_exposure_rows", 0),
        }
        for section in ("winners", "losers"):
            for idx, row in enumerate(detail.get(section, [])):
                for field_name in ("display_name", "description"):
                    if not _clean_str(row.get(field_name)):
                        blank_render_fields.append(
                            {"bucket": bucket, "section": section, "row": idx, "field": field_name}
                        )
        for idx, row in enumerate((detail.get("exposure_rows") or [])[:25]):
            for field_name in ("underlying", "symbols"):
                if not _clean_str(row.get(field_name)):
                    blank_render_fields.append(
                        {"bucket": bucket, "section": "exposure_rows", "row": idx, "field": field_name}
                    )

    reconciliations: list[dict[str, Any]] = []
    totals = totals or {}
    book_gross = float(totals.get("gross_exposure_total", 0.0) or 0.0)
    book_net = float(totals.get("net_exposure_total", 0.0) or 0.0)
    bucket_gross = sum(float(totals.get(f"gross_exposure_{b}", 0.0) or 0.0) for b in ("bucket_1", "bucket_2", "bucket_3", "bucket_4"))
    bucket_net = sum(float(totals.get(f"net_exposure_{b}", 0.0) or 0.0) for b in ("bucket_1", "bucket_2", "bucket_3", "bucket_4"))
    if book_gross:
        gross_diff_pct = abs(bucket_gross - book_gross) / abs(book_gross)
        reconciliations.append(
            {
                "name": "bucket_gross_vs_book_gross",
                "status": "ok" if gross_diff_pct <= 0.01 else "hard",
                "book_value": book_gross,
                "component_sum": bucket_gross,
                "diff_pct": gross_diff_pct,
            }
        )
    if book_gross:
        net_diff_pct = abs(bucket_net - book_net) / abs(book_gross)
        reconciliations.append(
            {
                "name": "bucket_net_vs_book_net",
                "status": "ok" if net_diff_pct <= 0.01 else "hard",
                "book_value": book_net,
                "component_sum": bucket_net,
                "diff_pct_of_gross": net_diff_pct,
            }
        )

    status = "ok"
    if missing_sources or missing_columns:
        status = "warn"
    if blank_render_fields or any(r["status"] == "hard" for r in reconciliations):
        status = "hard"

    return {
        "status": status,
        "run_date": run_date,
        "missing_source_count": missing_sources,
        "missing_required_column_count": missing_columns,
        "blank_render_field_count": len(blank_render_fields),
        "blank_render_fields": blank_render_fields,
        "reconciliations": reconciliations,
        "row_counts": row_counts,
        "sources": sources,
    }


def _scenario_from_contributors(
    *,
    scenario_id: str,
    label: str,
    description: str,
    contributors: list[dict[str, Any]],
    nav_usd: float,
    limits: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    contributors = sorted(contributors, key=lambda r: r["pnl_usd"])
    total = sum(float(r["pnl_usd"]) for r in contributors)
    pnl_pct = total / nav_usd if nav_usd > 0 else None
    bucket_pnl: dict[str, float] = {}
    for row in contributors:
        bucket = row.get("bucket", "unknown")
        bucket_pnl[bucket] = bucket_pnl.get(bucket, 0.0) + float(row["pnl_usd"])
    status = _classify(pnl_pct, limits["scenario_loss_pct_nav"], higher_is_worse=False)
    top = contributors[0] if contributors else None
    return {
        "id": scenario_id,
        "label": label,
        "description": description,
        "pnl_usd": total,
        "pnl_pct_nav": pnl_pct,
        "status": status,
        "limit": limits["scenario_loss_pct_nav"],
        "bucket_pnl": bucket_pnl,
        "top_contributor": top,
        "contributors": contributors[:25],
        "source": "Risk_Dashboard_Plan.md §4.2 stress test set",
        "action": "Review top contributors and cut/hedge if hard loss exceeds threshold.",
    }


def compute_scenario_panel(
    buckets: dict[str, dict[str, Any]],
    nav_usd: float,
    *,
    limits: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Compute deterministic first-pass shock P&L from current exposures."""
    limits = limits or DEFAULT_LIMITS

    def exposure_contributors(multiplier: float, shock_label: str) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for bucket, detail in buckets.items():
            for row in detail.get("exposure_rows", []):
                pnl = float(row.get("net_notional_usd", 0.0) or 0.0) * multiplier
                out.append(
                    {
                        "bucket": bucket,
                        "underlying": row.get("underlying", ""),
                        "symbols": row.get("symbols", ""),
                        "pnl_usd": pnl,
                        "driver": shock_label,
                        "net_notional_usd": row.get("net_notional_usd", 0.0),
                        "gross_notional_usd": row.get("gross_notional_usd", 0.0),
                    }
                )
        return out

    scenarios = [
        _scenario_from_contributors(
            scenario_id="market_down_3",
            label="Market -3%",
            description="Parallel underlying shock using current net notional by bucket.",
            contributors=exposure_contributors(-0.03, "underlying -3%"),
            nav_usd=nav_usd,
            limits=limits,
        ),
        _scenario_from_contributors(
            scenario_id="market_down_5",
            label="Market -5%",
            description="Parallel underlying shock using current net notional by bucket.",
            contributors=exposure_contributors(-0.05, "underlying -5%"),
            nav_usd=nav_usd,
            limits=limits,
        ),
        _scenario_from_contributors(
            scenario_id="market_down_10",
            label="Market -10%",
            description="Parallel underlying shock using current net notional by bucket.",
            contributors=exposure_contributors(-0.10, "underlying -10%"),
            nav_usd=nav_usd,
            limits=limits,
        ),
        _scenario_from_contributors(
            scenario_id="market_up_5",
            label="Market +5%",
            description="Parallel underlying shock using current net notional by bucket.",
            contributors=exposure_contributors(0.05, "underlying +5%"),
            nav_usd=nav_usd,
            limits=limits,
        ),
    ]

    vol_contrib: list[dict[str, Any]] = []
    for bucket, detail in buckets.items():
        haircut = -0.05 if bucket == "bucket_4" else -0.01 if bucket in {"bucket_1", "bucket_2", "book"} else 0.0
        for row in detail.get("exposure_rows", []):
            gross = float(row.get("gross_notional_usd", 0.0) or 0.0)
            if gross <= 0 or haircut == 0.0:
                continue
            vol_contrib.append(
                {
                    "bucket": bucket,
                    "underlying": row.get("underlying", ""),
                    "symbols": row.get("symbols", ""),
                    "pnl_usd": gross * haircut,
                    "driver": "vol spike gross haircut",
                    "net_notional_usd": row.get("net_notional_usd", 0.0),
                    "gross_notional_usd": gross,
                }
            )
    scenarios.append(
        _scenario_from_contributors(
            scenario_id="vol_spike",
            label="Vol Spike",
            description="Proxy stress: larger haircut on inverse sleeve gross and smaller haircut on short-heavy sleeves.",
            contributors=vol_contrib,
            nav_usd=nav_usd,
            limits=limits,
        )
    )

    borrow_contrib: list[dict[str, Any]] = []
    for bucket, detail in buckets.items():
        for row in detail.get("pnl_rows", []):
            borrow = float(row.get("borrow_fees", 0.0) or 0.0)
            if borrow >= 0:
                continue
            borrow_contrib.append(
                {
                    "bucket": bucket,
                    "underlying": row.get("display_name", row.get("underlying", "")),
                    "symbols": row.get("symbols", ""),
                    "pnl_usd": borrow,
                    "driver": "incremental borrow cost if APR doubles",
                    "net_notional_usd": None,
                    "gross_notional_usd": None,
                }
            )
    scenarios.append(
        _scenario_from_contributors(
            scenario_id="borrow_double",
            label="Borrow Doubles",
            description="Incremental cost equal to current borrow fees on names with borrow expense.",
            contributors=borrow_contrib,
            nav_usd=nav_usd,
            limits=limits,
        )
    )

    liq_contrib: list[dict[str, Any]] = []
    all_exposures: list[dict[str, Any]] = []
    for bucket, detail in buckets.items():
        for row in detail.get("exposure_rows", []):
            all_exposures.append({"bucket": bucket, **row})
    for row in sorted(all_exposures, key=lambda r: float(r.get("gross_notional_usd", 0.0) or 0.0), reverse=True)[:10]:
        gross = float(row.get("gross_notional_usd", 0.0) or 0.0)
        liq_contrib.append(
            {
                "bucket": row.get("bucket", ""),
                "underlying": row.get("underlying", ""),
                "symbols": row.get("symbols", ""),
                "pnl_usd": -0.02 * gross,
                "driver": "2% exit haircut on top gross exposures",
                "net_notional_usd": row.get("net_notional_usd", 0.0),
                "gross_notional_usd": gross,
            }
        )
    scenarios.append(
        _scenario_from_contributors(
            scenario_id="liquidity_exit",
            label="Liquidity Exit",
            description="2% exit haircut on the top 10 gross underlying exposures.",
            contributors=liq_contrib,
            nav_usd=nav_usd,
            limits=limits,
        )
    )

    worst = min(scenarios, key=lambda s: s["pnl_usd"]) if scenarios else None
    breaches = [s for s in scenarios if s["status"] != "ok"]
    return {
        "scenarios": scenarios,
        "worst_shock": worst,
        "breaches": breaches,
    }


def compute_alert_rows(
    *,
    book: BookSummary,
    scenario_panel: dict[str, Any],
    borrow_panel: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rows.extend(book.breaches)
    for scenario in scenario_panel.get("breaches", []):
        rows.append(
            {
                "metric": f"scenario:{scenario['id']}",
                "value": scenario["pnl_pct_nav"],
                "status": scenario["status"],
                "limit": scenario["limit"],
                "source": scenario["source"],
                "action": scenario["action"],
                "label": scenario["label"],
            }
        )
    rows.extend(borrow_panel.get("breaches", []))
    order = {"hard": 0, "warn": 1, "unknown": 2, "ok": 3}
    return sorted(rows, key=lambda r: order.get(r.get("status", "unknown"), 2))


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
    data_quality: dict[str, Any] = field(default_factory=dict)
    scenario_panel: dict[str, Any] = field(default_factory=dict)
    alert_rows: list[dict[str, Any]] = field(default_factory=list)
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
            "data_quality": self.data_quality,
            "scenario_panel": self.scenario_panel,
            "worst_shock": self.scenario_panel.get("worst_shock"),
            "top_risk_contributors": (
                (self.scenario_panel.get("worst_shock") or {}).get("contributors", [])
            ),
            "alert_rows": self.alert_rows,
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
    data_quality = compute_data_quality(
        accounting_dir=accounting,
        flex_dir=flex,
        buckets=buckets,
        totals=totals,
        run_date=run_date,
    )
    scenario_buckets = buckets
    if any(r.get("status") == "hard" for r in data_quality.get("reconciliations", [])):
        scenario_buckets = {
            "book": compute_bucket_detail(
                bucket="book",
                pnl_csv=accounting / "pnl_by_underlying.csv",
                net_exposure_csv=accounting / "net_exposure_by_underlying.csv",
            )
        }
    scenario_panel = compute_scenario_panel(
        buckets=scenario_buckets,
        nav_usd=nav_usd,
        limits=limits,
    )
    alert_rows = compute_alert_rows(
        book=book,
        scenario_panel=scenario_panel,
        borrow_panel=borrow_panel,
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
        data_quality=data_quality,
        scenario_panel=scenario_panel,
        alert_rows=alert_rows,
        universe_counts=universe_counts,
        raw_totals=totals,
    )
