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

from .factor_map import lookup_underlying
from .flex_parser import (
    FlexBorrowFee,
    FlexPosition,
    parse_borrow_fee_details,
    parse_positions,
    summarize_borrow,
    summarize_positions,
)

try:
    from .beta_loader import compute_betas  # noqa: F401
except Exception:  # pragma: no cover - optional dep / fallback
    compute_betas = None  # type: ignore[assignment]


TRADING_DAYS_PER_YEAR_DAYS: int = 252


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
    "single_name_gross_pct_nav": {"warn": 0.05, "hard": 0.10},
    "single_sector_gross_pct_nav": {"warn": 0.30, "hard": 0.50},
    "top10_gross_pct_nav": {"warn": 0.50, "hard": 0.75},
    "hhi_underlying": {"warn": 1500.0, "hard": 2500.0},
    "hhi_sector": {"warn": 2500.0, "hard": 4000.0},
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
    sleeve_attribution_available: bool = True
    sleeve_attribution_reason: str = ""


def _bucket_reconciles(totals: dict[str, Any]) -> tuple[bool, str]:
    """Return (sleeve_attribution_available, reason).

    Bucket gross/net components (b1 + b2 + b4) must sum to within 1% of
    the book aggregate. Bucket 3 is intentionally excluded because it is
    a beta-normalized hedge OVERLAY (sum of |beta| * notional for
    inverse / flow-hedge ETFs only), not a share-notional bucket -
    including it would double-count those legs against book gross.

    Mirrors the upstream accounting reconciliation gate in
    ``ibkr_accounting.py`` (see Phase G).
    """
    book_gross = float(totals.get("gross_exposure_total", 0.0) or 0.0)
    book_net = float(totals.get("net_exposure_total", 0.0) or 0.0)
    if book_gross <= 0:
        return True, ""
    bucket_gross = sum(
        float(totals.get(f"gross_exposure_{b}", 0.0) or 0.0)
        for b in ("bucket_1", "bucket_2", "bucket_4")
    )
    bucket_net = sum(
        float(totals.get(f"net_exposure_{b}", 0.0) or 0.0)
        for b in ("bucket_1", "bucket_2", "bucket_4")
    )
    gross_diff = abs(bucket_gross - book_gross) / abs(book_gross)
    net_diff = abs(bucket_net - book_net) / abs(book_gross)
    if gross_diff <= 0.01 and net_diff <= 0.01:
        return True, ""
    return False, (
        f"Bucket exposures do not reconcile to book aggregate "
        f"(gross diff {gross_diff:.1%}, net diff {net_diff:.1%}). "
        f"Showing sleeve P&L only; gross/net per sleeve suppressed."
    )


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

    sleeve_available, sleeve_reason = _bucket_reconciles(totals)

    sleeve_table: list[dict[str, Any]] = []
    bucket_pnl = totals.get("bucket_pnl") or {}
    for bucket in ("bucket_1", "bucket_2", "bucket_3", "bucket_4"):
        gross_b_raw = float(totals.get(f"gross_exposure_{bucket}", 0.0) or 0.0)
        net_b_raw = float(totals.get(f"net_exposure_{bucket}", 0.0) or 0.0)
        target_w = target_weights.get(bucket)
        if sleeve_available:
            gross_b = gross_b_raw
            net_b = net_b_raw
            actual_w = (gross_b / gross) if gross > 0 else 0.0
            drift_pp = ((actual_w - target_w) * 100.0) if target_w is not None else None
            drift_status = (
                _classify(abs(drift_pp), limits["sleeve_drift_pp"])
                if drift_pp is not None
                else "ok"
            )
        else:
            gross_b = None
            net_b = None
            actual_w = None
            drift_pp = None
            drift_status = "unknown"
        sleeve_table.append(
            {
                "bucket": bucket,
                "gross_usd": gross_b,
                "net_usd": net_b,
                "gross_usd_raw": gross_b_raw,
                "net_usd_raw": net_b_raw,
                "actual_weight": actual_w,
                "target_weight": target_w,
                "drift_pp": drift_pp,
                "drift_status": drift_status,
                "pnl_usd": float(bucket_pnl.get(bucket, 0.0) or 0.0),
                "attribution_available": sleeve_available,
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
        sleeve_attribution_available=sleeve_available,
        sleeve_attribution_reason=sleeve_reason,
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
    screener_csv: Path | None = None,
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

    # Borrow squeeze proxy: short position qty vs screener `shares_available`.
    squeeze_rows: list[dict[str, Any]] = []
    if screener_csv is not None and screener_csv.is_file():
        try:
            screener = pd.read_csv(
                screener_csv, usecols=["ETF", "shares_available", "borrow_fee_annual"]
            )
        except Exception:
            screener = pd.DataFrame()
        avail_map: dict[str, dict[str, Any]] = {}
        if not screener.empty:
            for _, r in screener.iterrows():
                key = _clean_str(r.get("ETF")).upper()
                if not key:
                    continue
                avail_map[key] = {
                    "shares_available": (
                        float(r.get("shares_available")) if pd.notna(r.get("shares_available")) else None
                    ),
                    "borrow_fee_annual": (
                        float(r.get("borrow_fee_annual")) if pd.notna(r.get("borrow_fee_annual")) else None
                    ),
                }
        for p in positions:
            qty = float(getattr(p, "qty", 0) or 0)
            if qty >= 0:
                continue
            symbol = (getattr(p, "symbol", "") or "").upper()
            meta = avail_map.get(symbol, {})
            shares_avail = meta.get("shares_available")
            short_qty = abs(qty)
            ratio = None
            status = "unknown"
            if shares_avail and shares_avail > 0:
                ratio = short_qty / shares_avail
                if ratio >= 0.5:
                    status = "hard"
                elif ratio >= 0.25:
                    status = "warn"
                else:
                    status = "ok"
            squeeze_rows.append(
                {
                    "symbol": symbol,
                    "short_qty": short_qty,
                    "shares_available": shares_avail,
                    "utilization": ratio,
                    "borrow_fee_annual": meta.get("borrow_fee_annual"),
                    "status": status,
                }
            )
        squeeze_rows.sort(
            key=lambda r: (-1.0 if r["utilization"] is None else r["utilization"]),
            reverse=True,
        )
        for r in squeeze_rows:
            if r["status"] == "hard":
                breaches.append(
                    {
                        "metric": f"borrow_squeeze:{r['symbol']}",
                        "value": r["utilization"],
                        "status": "hard",
                        "limit": {"warn": 0.25, "hard": 0.50},
                        "source": "shares_available vs short qty",
                        "action": f"Risk of buy-in on {r['symbol']}; reduce or lock borrow.",
                    }
                )
            elif r["status"] == "warn":
                breaches.append(
                    {
                        "metric": f"borrow_squeeze:{r['symbol']}",
                        "value": r["utilization"],
                        "status": "warn",
                        "limit": {"warn": 0.25, "hard": 0.50},
                        "source": "shares_available vs short qty",
                        "action": f"Monitor {r['symbol']} availability; consider hedge.",
                    }
                )

    return {
        "borrow": summary,
        "positions": pos_summary,
        "breaches": breaches,
        "squeeze_rows": squeeze_rows,
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
    # b1+b2+b4 only - bucket_3 is a beta-normalized overlay and excluded
    # to match the sleeve-attribution gate in ``_bucket_reconciles``.
    _reconcile_buckets = ("bucket_1", "bucket_2", "bucket_4")
    bucket_gross = sum(
        float(totals.get(f"gross_exposure_{b}", 0.0) or 0.0) for b in _reconcile_buckets
    )
    bucket_net = sum(
        float(totals.get(f"net_exposure_{b}", 0.0) or 0.0) for b in _reconcile_buckets
    )
    if book_gross:
        gross_diff_pct = abs(bucket_gross - book_gross) / abs(book_gross)
        reconciliations.append(
            {
                "name": "bucket_gross_vs_book_gross",
                "status": "ok" if gross_diff_pct <= 0.01 else "hard",
                "book_value": book_gross,
                "component_sum": bucket_gross,
                "diff_pct": gross_diff_pct,
                "components_included": list(_reconcile_buckets),
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
                "components_included": list(_reconcile_buckets),
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
    book_only_mode: bool = False,
    book_only_reason: str = "",
    factor_panel: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute deterministic first-pass shock P&L from current exposures.

    If ``factor_panel`` is supplied, an additional family of beta-adjusted
    SPX shocks is appended that uses curated underlying-to-SPY betas.
    """
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

    if factor_panel and factor_panel.get("available"):
        factor_rows = factor_panel.get("rows", [])
        for shock_pct, label_id, label in (
            (-0.03, "spx_beta_down_3", "SPX -3% (beta-adj)"),
            (-0.05, "spx_beta_down_5", "SPX -5% (beta-adj)"),
            (-0.10, "spx_beta_down_10", "SPX -10% (beta-adj)"),
            (0.05, "spx_beta_up_5", "SPX +5% (beta-adj)"),
        ):
            contributors: list[dict[str, Any]] = []
            for r in factor_rows:
                pnl = r["beta_weighted_net_usd"] * shock_pct
                if pnl == 0:
                    continue
                contributors.append(
                    {
                        "bucket": r.get("sector", "other"),
                        "underlying": r.get("underlying", ""),
                        "symbols": r.get("symbols", ""),
                        "pnl_usd": pnl,
                        "driver": f"SPX {shock_pct:+.0%} x beta {r['beta_to_spy']:+.2f}"
                                  + (" (default)" if r["beta_source"] == "default" else ""),
                        "net_notional_usd": r.get("net_notional_usd", 0.0),
                        "gross_notional_usd": r.get("gross_notional_usd", 0.0),
                        "beta_to_spy": r["beta_to_spy"],
                        "beta_source": r["beta_source"],
                    }
                )
            scenarios.append(
                _scenario_from_contributors(
                    scenario_id=label_id,
                    label=label,
                    description="Underlying-to-SPY beta x shock; coverage shown in factor panel.",
                    contributors=contributors,
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
        "book_only_mode": book_only_mode,
        "book_only_reason": book_only_reason,
    }


def _load_screener_vol_map(screener_csv: Path | None) -> dict[str, float]:
    """Underlying -> annualized realized vol (%). Falls back across
    ``vol_underlying_annual`` and ``vol_underlying_annual_legacy``."""
    if screener_csv is None or not screener_csv.is_file():
        return {}
    try:
        df = pd.read_csv(
            screener_csv,
            usecols=["Underlying", "vol_underlying_annual", "vol_underlying_annual_legacy"],
        )
    except Exception:
        return {}
    out: dict[str, float] = {}
    for _, r in df.iterrows():
        u = _clean_str(r.get("Underlying")).upper()
        if not u:
            continue
        v = r.get("vol_underlying_annual")
        if v is None or pd.isna(v):
            v = r.get("vol_underlying_annual_legacy")
        if v is None or pd.isna(v):
            continue
        try:
            out[u] = float(v) * 100.0
        except (TypeError, ValueError):
            continue
    return out


def compute_factor_panel(
    underlying_exposure_csv: Path,
    nav_usd: float,
    *,
    beta_results: dict[str, Any] | None = None,
    screener_vol_map: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Compute beta-weighted net exposure, sector groupings and top names.

    Inputs are book-level (sleeve-agnostic) so this works even when
    bucket reconciliation is broken. Beta source flags propagate so the
    UI can show coverage honestly.

    ``beta_results`` (Phase I): dict of ``{underlying_upper: BetaResult
    dict}`` from ``risk_dashboard.beta_loader.compute_betas``. When
    supplied, computed betas override the curated map and the panel
    also carries per-name ``beta_to_ndx`` / ``beta_to_rut`` /
    ``regime_vol_pct`` needed by the slide-risk and vol-shock panels.
    """
    df = _read_csv_or_empty(underlying_exposure_csv)
    if df.empty:
        return {
            "available": False,
            "reason": f"missing {underlying_exposure_csv.name}",
            "rows": [],
            "by_sector": [],
            "top_beta_long": [],
            "top_beta_short": [],
            "totals": {},
            "beta_provenance_counts": {},
        }

    rows: list[dict[str, Any]] = []
    for _, raw in df.iterrows():
        underlying = _first_nonblank(raw.get("underlying"), raw.get("symbol"))
        symbols = _first_nonblank(raw.get("symbols"), underlying)
        net = float(raw.get("net_notional_usd", 0.0) or 0.0)
        gross = float(raw.get("gross_notional_usd", 0.0) or 0.0)
        legs = int(raw.get("n_legs", 0) or 0)
        meta = lookup_underlying(underlying)
        sector = meta["sector"]
        sector_source = meta["sector_source"]
        beta_to_spy = meta["beta_to_spy"]
        beta_source = meta["beta_source"]
        beta_to_ndx: float | None = None
        beta_to_rut: float | None = None
        beta_se: float | None = None
        beta_n_obs: int | None = None
        beta_r2: float | None = None
        regime_vol_pct: float | None = None
        shrinkage_applied: bool | None = None
        if beta_results:
            br = beta_results.get((underlying or "").strip().upper())
            if br:
                provenance = br.get("provenance", "")
                if provenance in ("computed", "curated_fallback", "default_fallback"):
                    beta_source = provenance
                if br.get("beta_to_spy") is not None:
                    beta_to_spy = float(br["beta_to_spy"])
                beta_to_ndx = br.get("beta_to_ndx")
                beta_to_rut = br.get("beta_to_rut")
                beta_se = br.get("beta_se")
                beta_n_obs = br.get("n_obs")
                beta_r2 = br.get("r2")
                regime_vol_pct = br.get("regime_vol_pct")
                shrinkage_applied = bool(br.get("shrinkage_applied"))
        if regime_vol_pct is None and screener_vol_map:
            regime_vol_pct = screener_vol_map.get((underlying or "").strip().upper())
        beta_net = net * beta_to_spy
        beta_gross = gross * abs(beta_to_spy)
        rows.append(
            {
                "underlying": underlying,
                "symbols": symbols,
                "net_notional_usd": net,
                "gross_notional_usd": gross,
                "n_legs": legs,
                "sector": sector,
                "sector_source": sector_source,
                "beta_to_spy": beta_to_spy,
                "beta_to_ndx": beta_to_ndx,
                "beta_to_rut": beta_to_rut,
                "beta_se": beta_se,
                "beta_n_obs": beta_n_obs,
                "beta_r2": beta_r2,
                "regime_vol_pct": regime_vol_pct,
                "beta_source": beta_source,
                "shrinkage_applied": shrinkage_applied,
                "beta_weighted_net_usd": beta_net,
                "beta_weighted_gross_usd": beta_gross,
            }
        )

    total_net = sum(r["net_notional_usd"] for r in rows)
    total_gross = sum(r["gross_notional_usd"] for r in rows)
    total_beta_net = sum(r["beta_weighted_net_usd"] for r in rows)
    total_beta_gross = sum(r["beta_weighted_gross_usd"] for r in rows)
    trusted_sources = {"curated", "computed", "curated_fallback"}
    known_beta_gross = sum(
        r["gross_notional_usd"] for r in rows if r["beta_source"] in trusted_sources
    )
    coverage = (known_beta_gross / total_gross) if total_gross > 0 else 0.0
    provenance_counts: dict[str, int] = {}
    for r in rows:
        key = r["beta_source"] or "unknown"
        provenance_counts[key] = provenance_counts.get(key, 0) + 1

    sector_map: dict[str, dict[str, Any]] = {}
    for r in rows:
        s = r["sector"]
        bucket = sector_map.setdefault(
            s,
            {
                "sector": s,
                "n_names": 0,
                "net_notional_usd": 0.0,
                "gross_notional_usd": 0.0,
                "beta_weighted_net_usd": 0.0,
                "beta_weighted_gross_usd": 0.0,
            },
        )
        bucket["n_names"] += 1
        bucket["net_notional_usd"] += r["net_notional_usd"]
        bucket["gross_notional_usd"] += r["gross_notional_usd"]
        bucket["beta_weighted_net_usd"] += r["beta_weighted_net_usd"]
        bucket["beta_weighted_gross_usd"] += r["beta_weighted_gross_usd"]
    by_sector = sorted(
        sector_map.values(),
        key=lambda r: abs(r["beta_weighted_net_usd"]),
        reverse=True,
    )

    longs = sorted(
        [r for r in rows if r["beta_weighted_net_usd"] > 0],
        key=lambda r: r["beta_weighted_net_usd"],
        reverse=True,
    )[:12]
    shorts = sorted(
        [r for r in rows if r["beta_weighted_net_usd"] < 0],
        key=lambda r: r["beta_weighted_net_usd"],
    )[:12]

    totals = {
        "n_underlyings": len(rows),
        "net_notional_usd": total_net,
        "gross_notional_usd": total_gross,
        "beta_weighted_net_usd": total_beta_net,
        "beta_weighted_gross_usd": total_beta_gross,
        "net_beta_to_spy": (total_beta_net / nav_usd) if nav_usd > 0 else None,
        "gross_beta_to_spy": (total_beta_gross / nav_usd) if nav_usd > 0 else None,
        "beta_coverage_gross_pct": coverage,
    }

    return {
        "available": True,
        "rows": rows,
        "by_sector": by_sector,
        "top_beta_long": longs,
        "top_beta_short": shorts,
        "totals": totals,
        "beta_provenance_counts": provenance_counts,
    }


def compute_concentration_panel(
    factor_panel: dict[str, Any],
    nav_usd: float,
    *,
    limits: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Concentration metrics: HHI, top-N, single-name & sector caps."""
    limits = limits or DEFAULT_LIMITS
    if not factor_panel or not factor_panel.get("available"):
        return {
            "available": False,
            "reason": "factor panel unavailable",
            "top_names": [],
            "by_sector": [],
            "breaches": [],
        }
    rows = factor_panel.get("rows", [])
    sectors = factor_panel.get("by_sector", [])
    total_gross = sum(r["gross_notional_usd"] for r in rows) or 0.0
    if total_gross <= 0 or nav_usd <= 0:
        return {
            "available": False,
            "reason": "zero gross or nav",
            "top_names": [],
            "by_sector": [],
            "breaches": [],
        }

    ranked = sorted(rows, key=lambda r: r["gross_notional_usd"], reverse=True)
    top_names = []
    for r in ranked:
        pct_nav = r["gross_notional_usd"] / nav_usd
        share = r["gross_notional_usd"] / total_gross
        status = _classify(pct_nav, limits["single_name_gross_pct_nav"])
        top_names.append(
            {
                "underlying": r["underlying"],
                "sector": r["sector"],
                "gross_notional_usd": r["gross_notional_usd"],
                "net_notional_usd": r["net_notional_usd"],
                "pct_nav_gross": pct_nav,
                "pct_book_gross": share,
                "beta_to_spy": r["beta_to_spy"],
                "beta_source": r["beta_source"],
                "status": status,
                "limit": limits["single_name_gross_pct_nav"],
            }
        )

    sector_rows = []
    for s in sectors:
        sector_gross = s["gross_notional_usd"]
        pct_nav = sector_gross / nav_usd if nav_usd > 0 else 0.0
        share = sector_gross / total_gross if total_gross > 0 else 0.0
        status = _classify(pct_nav, limits["single_sector_gross_pct_nav"])
        sector_rows.append(
            {
                "sector": s["sector"],
                "n_names": s["n_names"],
                "gross_notional_usd": sector_gross,
                "net_notional_usd": s["net_notional_usd"],
                "pct_nav_gross": pct_nav,
                "pct_book_gross": share,
                "status": status,
                "limit": limits["single_sector_gross_pct_nav"],
            }
        )
    sector_rows.sort(key=lambda r: r["pct_nav_gross"], reverse=True)

    def hhi(shares: Iterable[float]) -> float:
        return float(sum((s * 100.0) ** 2 for s in shares))

    underlying_shares = [r["pct_book_gross"] for r in top_names]
    sector_shares = [r["pct_book_gross"] for r in sector_rows]
    hhi_underlying = hhi(underlying_shares)
    hhi_sector = hhi(sector_shares)

    top5 = sum(r["gross_notional_usd"] for r in ranked[:5]) / nav_usd
    top10 = sum(r["gross_notional_usd"] for r in ranked[:10]) / nav_usd

    breaches: list[dict[str, Any]] = []
    for r in top_names:
        if r["status"] != "ok":
            breaches.append(
                _limit_row(
                    metric=f"single_name:{r['underlying']}",
                    value=r["pct_nav_gross"],
                    limits={"single_name:" + r["underlying"]: limits["single_name_gross_pct_nav"]},
                    source="concentration cap (single name)",
                    action=f"Trim {r['underlying']} or hedge sector exposure.",
                )
            )
    for r in sector_rows:
        if r["status"] != "ok":
            breaches.append(
                _limit_row(
                    metric=f"sector:{r['sector']}",
                    value=r["pct_nav_gross"],
                    limits={"sector:" + r["sector"]: limits["single_sector_gross_pct_nav"]},
                    source="concentration cap (sector)",
                    action=f"Diversify away from {r['sector']} sleeve.",
                )
            )
    top10_status = _classify(top10, limits["top10_gross_pct_nav"])
    if top10_status != "ok":
        breaches.append(
            _limit_row(
                metric="top10_gross_pct_nav",
                value=top10,
                limits={"top10_gross_pct_nav": limits["top10_gross_pct_nav"]},
                source="concentration cap (top 10 names)",
                action="Spread gross across more names; current top-10 share dominates the book.",
            )
        )
    hhi_status_und = _classify(hhi_underlying, limits["hhi_underlying"])
    if hhi_status_und != "ok":
        breaches.append(
            _limit_row(
                metric="hhi_underlying",
                value=hhi_underlying,
                limits={"hhi_underlying": limits["hhi_underlying"]},
                source="concentration HHI (by underlying)",
                action="Increase number of names or reduce largest positions.",
            )
        )
    hhi_status_sec = _classify(hhi_sector, limits["hhi_sector"])
    if hhi_status_sec != "ok":
        breaches.append(
            _limit_row(
                metric="hhi_sector",
                value=hhi_sector,
                limits={"hhi_sector": limits["hhi_sector"]},
                source="concentration HHI (by sector)",
                action="Diversify sector exposure or hedge the dominant sector.",
            )
        )

    return {
        "available": True,
        "totals": {
            "n_underlyings": len(top_names),
            "n_sectors": len(sector_rows),
            "total_gross_usd": total_gross,
            "top5_pct_nav": top5,
            "top10_pct_nav": top10,
            "hhi_underlying": hhi_underlying,
            "hhi_sector": hhi_sector,
            "top10_status": top10_status,
            "hhi_underlying_status": hhi_status_und,
            "hhi_sector_status": hhi_status_sec,
        },
        "top_names": top_names[:25],
        "by_sector": sector_rows,
        "breaches": breaches,
    }


def compute_action_queue(
    *,
    book: BookSummary,
    factor_panel: dict[str, Any],
    concentration_panel: dict[str, Any],
    scenario_panel: dict[str, Any],
    borrow_panel: dict[str, Any],
    nav_usd: float,
    limits: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Turn raw alerts into a ranked, quantitative action queue.

    Each item carries a ``priority`` (0 = highest, hard breaches first),
    a short ``title``, a precise ``detail`` (USD or shares to move),
    and an optional ``hedge_hint`` (e.g. SPY notional to short).
    """
    limits = limits or DEFAULT_LIMITS
    actions: list[dict[str, Any]] = []

    # 1) Book-level gross breach -> tell the user how much gross to cut.
    if book.gross_exposure_pct_nav > limits["gross_exposure_pct_nav"]["warn"]:
        target_pct = limits["gross_exposure_pct_nav"]["warn"]
        cut_usd = (book.gross_exposure_pct_nav - target_pct) * nav_usd
        status = "hard" if book.gross_exposure_pct_nav >= limits["gross_exposure_pct_nav"]["hard"] else "warn"
        actions.append(
            {
                "priority": 0 if status == "hard" else 1,
                "status": status,
                "category": "gross_cap",
                "title": "Reduce gross exposure",
                "detail": f"Cut ~${cut_usd:,.0f} of gross to drop from {book.gross_exposure_pct_nav:.0%} to {target_pct:.0%} of NAV.",
                "source": "book gross cap (Risk_Dashboard_Plan.md §4.2)",
            }
        )

    # 2) Single-name and sector concentration breaches.
    for name in (concentration_panel or {}).get("top_names", []):
        if name["status"] == "ok":
            continue
        target_pct = limits["single_name_gross_pct_nav"]["warn"]
        trim_usd = (name["pct_nav_gross"] - target_pct) * nav_usd
        if trim_usd <= 0:
            continue
        actions.append(
            {
                "priority": 0 if name["status"] == "hard" else 1,
                "status": name["status"],
                "category": "single_name",
                "title": f"Trim {name['underlying']} ({name['sector']})",
                "detail": f"Reduce ~${trim_usd:,.0f} gross to bring {name['underlying']} below {target_pct:.0%} of NAV.",
                "source": "concentration cap (single name)",
            }
        )
    for sector in (concentration_panel or {}).get("by_sector", []):
        if sector["status"] == "ok":
            continue
        target_pct = limits["single_sector_gross_pct_nav"]["warn"]
        trim_usd = (sector["pct_nav_gross"] - target_pct) * nav_usd
        if trim_usd <= 0:
            continue
        actions.append(
            {
                "priority": 0 if sector["status"] == "hard" else 1,
                "status": sector["status"],
                "category": "sector",
                "title": f"De-risk {sector['sector']} sleeve",
                "detail": f"Reduce ~${trim_usd:,.0f} gross of {sector['sector']} (now {sector['pct_nav_gross']:.0%} of NAV).",
                "source": "concentration cap (sector)",
            }
        )

    # 3) Worst-shock action: SPY-beta hedge to bring scenario back under hard line.
    worst = (scenario_panel or {}).get("worst_shock") or {}
    factor_totals = (factor_panel or {}).get("totals") or {}
    if worst and worst.get("pnl_pct_nav") is not None:
        hard_limit = limits["scenario_loss_pct_nav"]["hard"]
        worst_pct = worst["pnl_pct_nav"]
        if worst_pct < hard_limit:
            excess_pct = hard_limit - worst_pct  # positive number we need to cover
            hedge_usd = excess_pct * nav_usd  # absolute USD loss to neutralize
            # If the worst case is a SPX shock, suggest SPY notional to short.
            shock_label = worst.get("label", "")
            spx_match = None
            for marker in ("-3%", "-5%", "-10%"):
                if marker in shock_label:
                    spx_match = float(marker.strip("%")) / 100.0
                    break
            hedge_hint = ""
            if spx_match is not None and spx_match != 0:
                spy_notional = abs(hedge_usd / spx_match)
                hedge_hint = (
                    f"Short ~${spy_notional:,.0f} SPY notional (1.0x) to "
                    f"absorb {-excess_pct:.1%} of NAV in this scenario."
                )
            elif factor_totals.get("net_beta_to_spy") is not None:
                hedge_hint = (
                    f"Trim net beta from {factor_totals['net_beta_to_spy']:.2f}x "
                    f"toward 0; ~${abs(hedge_usd):,.0f} hedge size."
                )
            actions.append(
                {
                    "priority": 0,
                    "status": "hard",
                    "category": "scenario",
                    "title": f"Hedge worst shock ({worst.get('label', 'scenario')})",
                    "detail": f"Current {worst_pct:.1%} of NAV vs hard limit {hard_limit:.1%}. Need ${abs(hedge_usd):,.0f} of coverage.",
                    "hedge_hint": hedge_hint,
                    "source": "worst-shock cap",
                }
            )

    # 4) Borrow squeeze actions (hard only -- warn left to alert table).
    for sq in (borrow_panel or {}).get("squeeze_rows", []):
        if sq.get("status") != "hard":
            continue
        target_qty = 0.5 * float(sq.get("shares_available") or 0)
        if target_qty <= 0:
            continue
        trim_qty = max(sq["short_qty"] - target_qty, 0)
        actions.append(
            {
                "priority": 0,
                "status": "hard",
                "category": "borrow_squeeze",
                "title": f"Reduce short on {sq['symbol']}",
                "detail": (
                    f"Utilization {sq['utilization']:.0%} of available borrow ("
                    f"{sq['short_qty']:,.0f} short vs {sq['shares_available']:,.0f} avail). "
                    f"Cut ~{trim_qty:,.0f} shares to fall below 50% utilization."
                ),
                "source": "shares_available vs short qty",
            }
        )

    actions.sort(key=lambda a: (a["priority"], a.get("category", "")))
    return actions


def compute_alert_rows(
    *,
    book: BookSummary,
    scenario_panel: dict[str, Any],
    borrow_panel: dict[str, Any],
    concentration_panel: dict[str, Any] | None = None,
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
    if concentration_panel:
        rows.extend(concentration_panel.get("breaches", []))
    order = {"hard": 0, "warn": 1, "unknown": 2, "ok": 3}
    return sorted(rows, key=lambda r: order.get(r.get("status", "unknown"), 2))


# ---------------------------------------------------------------------------
# Slide risk: SPX / NDX / RUT shock strips (Phase 1)


SLIDE_SHOCK_PCTS: tuple[float, ...] = (
    -0.20, -0.15, -0.10, -0.05, -0.03, -0.02, -0.01,
    0.01, 0.02, 0.03, 0.05, 0.10, 0.15, 0.20,
)
SLIDE_HORIZONS_DAYS: tuple[int, ...] = (0, 5, 20)
LETF_LEVERAGE_BY_PRODUCT_CLASS: dict[str, float] = {
    "letf_long": 2.0,
    "letf_inverse": -1.0,
    "covered_call_1x": 1.0,
    "scraped_income": 1.0,
    "income_yieldboost": 1.0,
    "volatility_etp": 1.0,
}


def _slide_index_label(idx_key: str) -> str:
    return {"spy": "SPX", "ndx": "NDX", "rut": "RUT"}.get(idx_key, idx_key.upper())


def _slide_status(pnl_pct: float | None, limits: dict[str, dict[str, Any]]) -> str:
    if pnl_pct is None:
        return "unknown"
    return _classify(pnl_pct, limits["scenario_loss_pct_nav"], higher_is_worse=False)


def _load_screener_etf_meta(screener_csv: Path | None) -> dict[str, dict[str, Any]]:
    """Per-ETF lookup: product class, leverage hint via Beta, vol_etf_annual,
    borrow_fee_annual. Keys are uppercase ETF symbols."""
    if screener_csv is None or not screener_csv.is_file():
        return {}
    try:
        df = pd.read_csv(
            screener_csv,
            usecols=[
                "ETF",
                "Underlying",
                "Beta",
                "Beta_product_class",
                "vol_etf_annual",
                "vol_underlying_annual",
                "borrow_fee_annual",
            ],
        )
    except Exception:
        return {}
    out: dict[str, dict[str, Any]] = {}
    for _, r in df.iterrows():
        etf = _clean_str(r.get("ETF")).upper()
        if not etf:
            continue
        out[etf] = {
            "underlying": _clean_str(r.get("Underlying")).upper(),
            "beta_to_underlying": (
                float(r.get("Beta")) if pd.notna(r.get("Beta")) else None
            ),
            "product_class": _clean_str(r.get("Beta_product_class")),
            "vol_etf_annual": (
                float(r.get("vol_etf_annual"))
                if pd.notna(r.get("vol_etf_annual"))
                else None
            ),
            "vol_underlying_annual": (
                float(r.get("vol_underlying_annual"))
                if pd.notna(r.get("vol_underlying_annual"))
                else None
            ),
            "borrow_fee_annual": (
                float(r.get("borrow_fee_annual"))
                if pd.notna(r.get("borrow_fee_annual"))
                else None
            ),
        }
    return out


def _leverage_from_product_class(product_class: str, beta: float | None = None) -> float:
    """Map screener Beta_product_class -> LETF leverage factor ``k``."""
    if beta is not None and abs(beta) > 0.25:
        if abs(beta - 2.0) < 0.25:
            return 2.0
        if abs(beta - 3.0) < 0.4:
            return 3.0
        if abs(beta + 1.0) < 0.25:
            return -1.0
        if abs(beta + 2.0) < 0.4:
            return -2.0
        if abs(beta + 3.0) < 0.5:
            return -3.0
    return LETF_LEVERAGE_BY_PRODUCT_CLASS.get(product_class, 1.0)


def _build_position_leg_map(
    flex_positions_xml: Path,
    etf_meta: dict[str, dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Return ``{underlying_upper: [leg_dict, ...]}`` from flex positions.

    Each leg: symbol, underlying, signed notional, product_class,
    leverage_k, vol_etf_annual (fraction), is_letf. Underlying
    resolution prefers screener ``Underlying`` (APLX -> APLD); falls
    back to flex ``underlyingSymbol`` then the symbol itself.
    """
    out: dict[str, list[dict[str, Any]]] = {}
    if not flex_positions_xml.is_file():
        return out
    positions = parse_positions(flex_positions_xml)
    for p in positions:
        sym = (p.symbol or "").upper()
        if not sym:
            continue
        screener_row = etf_meta.get(sym) or {}
        underlying = screener_row.get("underlying") or (p.underlying or "").upper() or sym
        product_class = screener_row.get("product_class") or ""
        beta_und = screener_row.get("beta_to_underlying")
        k = _leverage_from_product_class(product_class, beta_und)
        vol = screener_row.get("vol_etf_annual")
        is_letf = abs(k) > 1.0001 or product_class == "letf_inverse"
        leg = {
            "symbol": sym,
            "underlying": underlying,
            "net_notional_usd": float(getattr(p, "position_value", 0.0) or 0.0),
            "product_class": product_class,
            "leverage_k": float(k),
            "vol_etf_annual": float(vol) if vol is not None else None,
            "is_letf": is_letf,
        }
        out.setdefault(underlying, []).append(leg)
    return out


def compute_slide_risk_panel(
    *,
    factor_panel: dict[str, Any],
    nav_usd: float,
    screener_csv: Path | None = None,
    flex_positions_xml: Path | None = None,
    shocks: tuple[float, ...] = SLIDE_SHOCK_PCTS,
    horizons_days: tuple[int, ...] = SLIDE_HORIZONS_DAYS,
    limits: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Maven-style slide risk: beta-adjusted SPX / NDX / RUT shock strips.

    For each (index, shock_pct):
        * instantaneous P&L = sum_i (net_notional_i * beta_to_index_i * shock)
        * top contributor: largest |pnl|
    Plus a horizon overlay (T+5 / T+20) adding closed-form LETF vol drag:
        decay_pct(T) = -0.5 * k * (k - 1) * sigma^2 * T_years
    Decay is computed at the LEG level (not aggregated underlying) so a
    spot APLD leg doesn't get charged as a 2x LETF just because APLX is
    in the leg set.
    """
    limits = limits or DEFAULT_LIMITS
    rows = (factor_panel or {}).get("rows") or []
    if not rows or nav_usd <= 0:
        return {
            "available": False,
            "reason": "factor panel unavailable or NAV missing",
            "shocks_pct": list(shocks),
            "horizons_days": list(horizons_days),
            "indices": [],
        }

    etf_meta = _load_screener_etf_meta(screener_csv)
    leg_map: dict[str, list[dict[str, Any]]] = {}
    if flex_positions_xml is not None:
        leg_map = _build_position_leg_map(flex_positions_xml, etf_meta)

    enriched: list[dict[str, Any]] = []
    for r in rows:
        underlying = (r.get("underlying") or "").upper()
        symbols = r.get("symbols") or ""
        legs = leg_map.get(underlying, [])
        sigma_pct = r.get("regime_vol_pct")
        sigma = (float(sigma_pct) / 100.0) if sigma_pct is not None else None
        enriched.append(
            {
                "underlying": underlying,
                "symbols": symbols,
                "net_notional_usd": float(r.get("net_notional_usd", 0.0) or 0.0),
                "gross_notional_usd": float(r.get("gross_notional_usd", 0.0) or 0.0),
                "beta_to_spy": r.get("beta_to_spy"),
                "beta_to_ndx": r.get("beta_to_ndx"),
                "beta_to_rut": r.get("beta_to_rut"),
                "legs": legs,
                "is_letf": any(leg.get("is_letf") for leg in legs),
                "sigma": sigma,
            }
        )

    indices_out: list[dict[str, Any]] = []
    worst_shock: dict[str, Any] | None = None
    for idx_key in ("spy", "ndx", "rut"):
        beta_field = f"beta_to_{idx_key}"
        coverage_gross = 0.0
        total_gross = 0.0
        for e in enriched:
            total_gross += abs(e["net_notional_usd"])
            if e.get(beta_field) is not None:
                coverage_gross += abs(e["net_notional_usd"])

        shock_rows: list[dict[str, Any]] = []
        for shock_pct in shocks:
            per_name_pnl_t0: list[dict[str, Any]] = []
            for e in enriched:
                beta = e.get(beta_field)
                if beta is None:
                    continue
                expected_return = beta * shock_pct
                pnl_t0 = e["net_notional_usd"] * expected_return
                per_name_pnl_t0.append(
                    {
                        "underlying": e["underlying"],
                        "symbols": e["symbols"],
                        "net_notional_usd": e["net_notional_usd"],
                        "beta": beta,
                        "expected_return": expected_return,
                        "pnl_t0_usd": pnl_t0,
                    }
                )
            total_pnl_t0 = sum(p["pnl_t0_usd"] for p in per_name_pnl_t0)
            pnl_pct_t0 = total_pnl_t0 / nav_usd if nav_usd > 0 else None

            horizon_rows: list[dict[str, Any]] = []
            for t_days in horizons_days:
                if t_days <= 0:
                    horizon_rows.append(
                        {
                            "horizon_days": 0,
                            "total_pnl_usd": total_pnl_t0,
                            "total_pnl_pct_nav": pnl_pct_t0,
                            "decay_usd": 0.0,
                        }
                    )
                    continue
                t_years = t_days / TRADING_DAYS_PER_YEAR_DAYS
                decay_sum = 0.0
                for e in enriched:
                    for leg in e["legs"]:
                        if not leg.get("is_letf"):
                            continue
                        k = float(leg.get("leverage_k", 1.0))
                        if abs(k) <= 1.0001:
                            continue
                        sigma_etf = leg.get("vol_etf_annual")
                        if sigma_etf is None and e["sigma"] is not None:
                            sigma_etf = abs(k) * e["sigma"]
                        if sigma_etf is None or sigma_etf <= 0:
                            continue
                        decay_frac = -0.5 * k * (k - 1.0) * (sigma_etf ** 2) * t_years
                        decay_sum += leg["net_notional_usd"] * decay_frac
                total = total_pnl_t0 + decay_sum
                horizon_rows.append(
                    {
                        "horizon_days": t_days,
                        "total_pnl_usd": total,
                        "total_pnl_pct_nav": (total / nav_usd) if nav_usd > 0 else None,
                        "decay_usd": decay_sum,
                    }
                )

            sorted_by_pnl = sorted(per_name_pnl_t0, key=lambda p: p["pnl_t0_usd"])
            top_loss = sorted_by_pnl[0] if sorted_by_pnl else None
            top_gain = sorted_by_pnl[-1] if sorted_by_pnl else None
            row = {
                "shock_pct": shock_pct,
                "label": f"{_slide_index_label(idx_key)} {shock_pct:+.0%}",
                "pnl_usd": total_pnl_t0,
                "pnl_pct_nav": pnl_pct_t0,
                "net_delta_usd": total_pnl_t0,
                "net_delta_pct_nav": (total_pnl_t0 / nav_usd) if nav_usd > 0 else None,
                "status": _slide_status(pnl_pct_t0, limits),
                "horizons": horizon_rows,
                "top_loss": top_loss,
                "top_gain": top_gain,
                "n_contributors": len(per_name_pnl_t0),
            }
            shock_rows.append(row)
            if shock_pct < 0:
                cand_pnl = (
                    row.get("horizons", [{}])[-1].get("total_pnl_pct_nav") or pnl_pct_t0
                )
                if cand_pnl is not None and (
                    worst_shock is None
                    or (cand_pnl < (worst_shock.get("total_pnl_pct_nav") or 0.0))
                ):
                    worst_shock = {
                        "index": _slide_index_label(idx_key),
                        "label": row["label"],
                        "shock_pct": shock_pct,
                        "total_pnl_pct_nav": cand_pnl,
                        "horizon_days": horizons_days[-1],
                    }

        coverage_pct = (coverage_gross / total_gross) if total_gross > 0 else 0.0
        indices_out.append(
            {
                "index": _slide_index_label(idx_key),
                "key": idx_key,
                "shock_rows": shock_rows,
                "coverage_pct": coverage_pct,
                "n_names_covered": sum(1 for e in enriched if e.get(beta_field) is not None),
                "n_names_total": len(enriched),
            }
        )

    n_letf = sum(1 for e in enriched if e["is_letf"])
    has_vol = sum(1 for e in enriched if e["sigma"] is not None)
    return {
        "available": True,
        "shocks_pct": list(shocks),
        "horizons_days": list(horizons_days),
        "indices": indices_out,
        "worst_shock": worst_shock,
        "n_letf_names": n_letf,
        "n_names_with_vol": has_vol,
        "n_names_total": len(enriched),
    }


# ---------------------------------------------------------------------------
# Borrow shock sensitivity (Phase 3)


BORROW_ABS_SHOCKS_BP: tuple[int, ...] = (25, 50, 100, 200, 500)
BORROW_MULT_SHOCKS: tuple[float, ...] = (1.5, 2.0, 3.0, 5.0)


def compute_borrow_shock_panel(
    *,
    borrow_panel: dict[str, Any],
    flex_positions_xml: Path,
    nav_usd: float,
    screener_csv: Path | None = None,
    abs_shocks_bp: tuple[int, ...] = BORROW_ABS_SHOCKS_BP,
    mult_shocks: tuple[float, ...] = BORROW_MULT_SHOCKS,
    persistence_days: int = 30,
) -> dict[str, Any]:
    """Per-symbol borrow cost sensitivity to rate shocks.

    Inputs:
        * ``borrow_panel`` -- output of ``compute_borrow_panel``; supplies
          per-name current ``fee_rate_pct`` from flex SLBFee rows.
        * ``flex_positions_xml`` -- short notional per symbol.
        * ``screener_csv`` -- ``borrow_fee_annual`` baseline for synthetic
          / inverse-ETF shorts where flex reports 0.

    Output: absolute (bp) and multiplicative ladders showing aggregate
    annualized cost delta + per-name worst victims. The 30-day
    persistence column converts daily delta to a stressed MTD impact.
    """
    positions = parse_positions(flex_positions_xml)
    if not positions:
        return {
            "available": False,
            "reason": f"missing or empty {flex_positions_xml.name}",
            "abs_shocks_bp": list(abs_shocks_bp),
            "mult_shocks": list(mult_shocks),
            "abs_ladder": [],
            "mult_ladder": [],
            "names": [],
        }

    short_notional: dict[str, float] = {}
    for p in positions:
        if getattr(p, "position_value", 0.0) < 0:
            sym = (p.symbol or "").upper()
            short_notional[sym] = short_notional.get(sym, 0.0) + abs(p.position_value)

    fee_rate_map: dict[str, float] = {}
    borrow_summary = borrow_panel.get("borrow") or {}
    for sym, rate in (borrow_summary.get("fee_rate_by_symbol") or {}).items():
        if float(rate) > 0:
            fee_rate_map[sym.upper()] = float(rate)
    if screener_csv is not None and screener_csv.is_file():
        try:
            sdf = pd.read_csv(screener_csv, usecols=["ETF", "borrow_fee_annual"])
        except Exception:
            sdf = pd.DataFrame()
        for _, r in sdf.iterrows():
            sym = _clean_str(r.get("ETF")).upper()
            val = r.get("borrow_fee_annual")
            if not sym or val is None or pd.isna(val):
                continue
            try:
                rate_pct = float(val) * 100.0
            except (TypeError, ValueError):
                continue
            fee_rate_map.setdefault(sym, rate_pct)
    for row in borrow_panel.get("squeeze_rows") or []:
        sym = (row.get("symbol") or "").upper()
        if sym in fee_rate_map:
            continue
        ann = row.get("borrow_fee_annual")
        if ann is not None:
            fee_rate_map[sym] = float(ann) * 100.0

    name_rows: list[dict[str, Any]] = []
    for sym, short_n in short_notional.items():
        current_apr = fee_rate_map.get(sym, 0.0)
        current_annual_cost = short_n * (current_apr / 100.0)
        name_rows.append(
            {
                "symbol": sym,
                "short_notional_usd": short_n,
                "current_apr_pct": current_apr,
                "current_annual_cost_usd": current_annual_cost,
            }
        )
    name_rows.sort(key=lambda r: r["current_annual_cost_usd"], reverse=True)
    total_current_annual = sum(r["current_annual_cost_usd"] for r in name_rows)

    def _shock_ladder(shocks: Iterable[Any], *, kind: str) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for shock in shocks:
            if kind == "abs_bp":
                shifted_aprs = {
                    r["symbol"]: r["current_apr_pct"] + (shock / 100.0) for r in name_rows
                }
                label = f"+{int(shock)}bp"
            else:
                shifted_aprs = {
                    r["symbol"]: r["current_apr_pct"] * float(shock) for r in name_rows
                }
                label = f"{shock:g}x"
            per_name_delta = []
            for r in name_rows:
                new_apr = shifted_aprs[r["symbol"]]
                new_cost = r["short_notional_usd"] * (new_apr / 100.0)
                delta = -(new_cost - r["current_annual_cost_usd"])
                per_name_delta.append(
                    {
                        "symbol": r["symbol"],
                        "short_notional_usd": r["short_notional_usd"],
                        "current_apr_pct": r["current_apr_pct"],
                        "new_apr_pct": new_apr,
                        "annual_delta_usd": delta,
                        "daily_delta_usd": delta / TRADING_DAYS_PER_YEAR_DAYS,
                        "persistence_usd": (delta / TRADING_DAYS_PER_YEAR_DAYS)
                        * persistence_days,
                    }
                )
            per_name_delta.sort(key=lambda d: d["annual_delta_usd"])
            total_delta = sum(d["annual_delta_usd"] for d in per_name_delta)
            out.append(
                {
                    "shock": shock,
                    "kind": kind,
                    "label": label,
                    "annual_delta_usd": total_delta,
                    "annual_delta_pct_nav": (total_delta / nav_usd)
                    if nav_usd > 0
                    else None,
                    "persistence_delta_usd": (total_delta / TRADING_DAYS_PER_YEAR_DAYS)
                    * persistence_days,
                    "worst_victims": per_name_delta[:5],
                }
            )
        return out

    return {
        "available": True,
        "abs_shocks_bp": list(abs_shocks_bp),
        "mult_shocks": list(mult_shocks),
        "abs_ladder": _shock_ladder(abs_shocks_bp, kind="abs_bp"),
        "mult_ladder": _shock_ladder(mult_shocks, kind="mult"),
        "names": name_rows[:25],
        "n_short_symbols": len(name_rows),
        "current_annual_cost_usd": total_current_annual,
        "current_annual_cost_pct_nav": (total_current_annual / nav_usd)
        if nav_usd > 0
        else None,
        "persistence_days": persistence_days,
    }


# ---------------------------------------------------------------------------
# VIX / vol shock sensitivity (Phase 4)


VIX_ABS_SHOCKS_POINTS: tuple[int, ...] = (5, 10, 15, 20, 30)
VOL_REGIME_MULTIPLIERS: tuple[float, ...] = (1.25, 1.5, 2.0, 3.0)

VEGA_FRAC_PER_VOLPOINT_BY_PRODUCT_CLASS: dict[str, float] = {
    "income_yieldboost": -0.0025,
    "covered_call_1x": -0.0015,
    "scraped_income": -0.0020,
    "volatility_etp": +0.0150,
}


def compute_vol_shock_panel(
    *,
    factor_panel: dict[str, Any],
    nav_usd: float,
    screener_csv: Path | None = None,
    flex_positions_xml: Path | None = None,
    vix_shocks: tuple[int, ...] = VIX_ABS_SHOCKS_POINTS,
    vol_multipliers: tuple[float, ...] = VOL_REGIME_MULTIPLIERS,
    decay_horizon_days: int = 20,
) -> dict[str, Any]:
    """Two strips: VIX absolute moves (vega P&L) + realized-vol multipliers
    (LETF decay over T+``decay_horizon_days``).

    Vega: per-leg by Beta_product_class, signed by net_notional direction.
    Vol regime: per-leg LETF decay; sums signed leg decay so a short LETF
    position registers as a vol-spike *beneficiary*.
    """
    rows = (factor_panel or {}).get("rows") or []
    if not rows or nav_usd <= 0:
        return {
            "available": False,
            "reason": "factor panel unavailable or NAV missing",
            "vix_shocks_pts": list(vix_shocks),
            "vol_multipliers": list(vol_multipliers),
            "vix_ladder": [],
            "vol_ladder": [],
        }

    etf_meta = _load_screener_etf_meta(screener_csv)
    leg_map: dict[str, list[dict[str, Any]]] = {}
    if flex_positions_xml is not None:
        leg_map = _build_position_leg_map(flex_positions_xml, etf_meta)

    enriched: list[dict[str, Any]] = []
    for r in rows:
        underlying = (r.get("underlying") or "").upper()
        symbols = r.get("symbols") or ""
        legs = leg_map.get(underlying, [])
        vega_frac = 0.0
        chosen_class = ""
        for leg in legs:
            cls = leg.get("product_class") or ""
            v = VEGA_FRAC_PER_VOLPOINT_BY_PRODUCT_CLASS.get(cls, 0.0)
            if abs(v) > abs(vega_frac):
                vega_frac = v
                chosen_class = cls
        sigma_pct = r.get("regime_vol_pct")
        sigma = (float(sigma_pct) / 100.0) if sigma_pct is not None else None
        enriched.append(
            {
                "underlying": underlying,
                "symbols": symbols,
                "net_notional_usd": float(r.get("net_notional_usd", 0.0) or 0.0),
                "gross_notional_usd": float(r.get("gross_notional_usd", 0.0) or 0.0),
                "legs": legs,
                "is_letf": any(leg.get("is_letf") for leg in legs),
                "sigma": sigma,
                "vega_frac": vega_frac,
                "vega_product_class": chosen_class,
            }
        )

    vix_ladder: list[dict[str, Any]] = []
    for vix_shock in vix_shocks:
        per_name = []
        for e in enriched:
            if abs(e["vega_frac"]) < 1e-12:
                continue
            net = e["net_notional_usd"]
            sign = 1.0 if net >= 0 else -1.0
            pnl = sign * e["gross_notional_usd"] * e["vega_frac"] * vix_shock
            per_name.append(
                {
                    "underlying": e["underlying"],
                    "symbols": e["symbols"],
                    "net_notional_usd": net,
                    "gross_notional_usd": e["gross_notional_usd"],
                    "vega_product_class": e["vega_product_class"],
                    "vega_frac_per_volpoint": e["vega_frac"],
                    "pnl_usd": pnl,
                }
            )
        per_name.sort(key=lambda r: r["pnl_usd"])
        total = sum(r["pnl_usd"] for r in per_name)
        vix_ladder.append(
            {
                "vix_shock_pts": vix_shock,
                "label": f"VIX +{vix_shock}",
                "pnl_usd": total,
                "pnl_pct_nav": (total / nav_usd) if nav_usd > 0 else None,
                "n_contributors": len(per_name),
                "worst_victims": per_name[:5],
                "top_gains": list(reversed(per_name[-5:])),
            }
        )

    t_years = decay_horizon_days / TRADING_DAYS_PER_YEAR_DAYS
    vol_ladder: list[dict[str, Any]] = []
    for mult in vol_multipliers:
        per_name = []
        for e in enriched:
            decay_usd = 0.0
            worst_leg_k = 0.0
            sigma_used = None
            for leg in e["legs"]:
                if not leg.get("is_letf"):
                    continue
                k = float(leg.get("leverage_k", 1.0))
                if abs(k) <= 1.0001:
                    continue
                sigma_etf = leg.get("vol_etf_annual")
                if sigma_etf is None and e["sigma"] is not None:
                    sigma_etf = abs(k) * e["sigma"]
                if sigma_etf is None or sigma_etf <= 0:
                    continue
                new_sigma = sigma_etf * mult
                decay_frac = -0.5 * k * (k - 1.0) * (new_sigma ** 2) * t_years
                decay_usd += leg["net_notional_usd"] * decay_frac
                if abs(k) > abs(worst_leg_k):
                    worst_leg_k = k
                    sigma_used = sigma_etf
            if abs(decay_usd) < 1e-6:
                continue
            per_name.append(
                {
                    "underlying": e["underlying"],
                    "symbols": e["symbols"],
                    "leverage": worst_leg_k,
                    "current_sigma_pct": (sigma_used or 0.0) * 100.0,
                    "stressed_sigma_pct": (sigma_used or 0.0) * mult * 100.0,
                    "pnl_usd": decay_usd,
                }
            )
        per_name.sort(key=lambda r: r["pnl_usd"])
        total = sum(r["pnl_usd"] for r in per_name)
        vol_ladder.append(
            {
                "multiplier": mult,
                "label": f"{mult:g}x vol",
                "horizon_days": decay_horizon_days,
                "pnl_usd": total,
                "pnl_pct_nav": (total / nav_usd) if nav_usd > 0 else None,
                "n_contributors": len(per_name),
                "worst_victims": per_name[:5],
            }
        )

    return {
        "available": True,
        "vix_shocks_pts": list(vix_shocks),
        "vol_multipliers": list(vol_multipliers),
        "decay_horizon_days": decay_horizon_days,
        "vix_ladder": vix_ladder,
        "vol_ladder": vol_ladder,
        "n_vega_contributors": sum(1 for e in enriched if abs(e["vega_frac"]) > 0),
        "n_letf_decay_contributors": sum(
            1 for e in enriched if e["is_letf"] and e["sigma"] is not None
        ),
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
    data_quality: dict[str, Any] = field(default_factory=dict)
    scenario_panel: dict[str, Any] = field(default_factory=dict)
    factor_panel: dict[str, Any] = field(default_factory=dict)
    concentration_panel: dict[str, Any] = field(default_factory=dict)
    slide_risk_panel: dict[str, Any] = field(default_factory=dict)
    borrow_shock_panel: dict[str, Any] = field(default_factory=dict)
    vol_shock_panel: dict[str, Any] = field(default_factory=dict)
    action_queue: list[dict[str, Any]] = field(default_factory=list)
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
                "sleeve_attribution_available": self.book.sleeve_attribution_available,
                "sleeve_attribution_reason": self.book.sleeve_attribution_reason,
            },
            "buckets": self.buckets,
            "borrow_panel": self.borrow_panel,
            "data_quality": self.data_quality,
            "scenario_panel": self.scenario_panel,
            "factor_panel": self.factor_panel,
            "concentration_panel": self.concentration_panel,
            "slide_risk_panel": self.slide_risk_panel,
            "borrow_shock_panel": self.borrow_shock_panel,
            "vol_shock_panel": self.vol_shock_panel,
            "action_queue": self.action_queue,
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
    screener_csv: Path | None = None,
    enable_computed_betas: bool = True,
    beta_cache_dir: Path | None = None,
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
        screener_csv=screener_csv,
    )
    data_quality = compute_data_quality(
        accounting_dir=accounting,
        flex_dir=flex,
        buckets=buckets,
        totals=totals,
        run_date=run_date,
    )
    scenario_buckets = buckets
    book_only_mode = not book.sleeve_attribution_available
    if book_only_mode:
        scenario_buckets = {
            "book": compute_bucket_detail(
                bucket="book",
                pnl_csv=accounting / "pnl_by_underlying.csv",
                net_exposure_csv=accounting / "net_exposure_by_underlying.csv",
            )
        }
    beta_results_dicts: dict[str, dict[str, Any]] = {}
    if enable_computed_betas and compute_betas is not None:
        exposure_csv = accounting / "net_exposure_by_underlying.csv"
        underlyings: list[str] = []
        if exposure_csv.is_file():
            try:
                _df = pd.read_csv(exposure_csv, usecols=["underlying"])
                underlyings = [
                    str(u).strip().upper() for u in _df["underlying"].dropna().tolist()
                ]
            except Exception:
                underlyings = []
        if underlyings:
            try:
                _br = compute_betas(underlyings, cache_dir=beta_cache_dir)
                beta_results_dicts = {k: v.to_dict() for k, v in _br.items()}
            except Exception:
                beta_results_dicts = {}

    screener_vol_map = _load_screener_vol_map(screener_csv)
    factor_panel = compute_factor_panel(
        underlying_exposure_csv=accounting / "net_exposure_by_underlying.csv",
        nav_usd=nav_usd,
        beta_results=beta_results_dicts or None,
        screener_vol_map=screener_vol_map or None,
    )
    scenario_panel = compute_scenario_panel(
        buckets=scenario_buckets,
        nav_usd=nav_usd,
        limits=limits,
        book_only_mode=book_only_mode,
        book_only_reason=book.sleeve_attribution_reason,
        factor_panel=factor_panel,
    )
    concentration_panel = compute_concentration_panel(
        factor_panel=factor_panel,
        nav_usd=nav_usd,
        limits=limits,
    )
    slide_risk_panel = compute_slide_risk_panel(
        factor_panel=factor_panel,
        nav_usd=nav_usd,
        screener_csv=screener_csv,
        flex_positions_xml=flex / "flex_positions.xml",
        limits=limits,
    )
    borrow_shock_panel = compute_borrow_shock_panel(
        borrow_panel=borrow_panel,
        flex_positions_xml=flex / "flex_positions.xml",
        nav_usd=nav_usd,
        screener_csv=screener_csv,
    )
    vol_shock_panel = compute_vol_shock_panel(
        factor_panel=factor_panel,
        nav_usd=nav_usd,
        screener_csv=screener_csv,
        flex_positions_xml=flex / "flex_positions.xml",
    )
    alert_rows = compute_alert_rows(
        book=book,
        scenario_panel=scenario_panel,
        borrow_panel=borrow_panel,
        concentration_panel=concentration_panel,
    )
    action_queue = compute_action_queue(
        book=book,
        factor_panel=factor_panel,
        concentration_panel=concentration_panel,
        scenario_panel=scenario_panel,
        borrow_panel=borrow_panel,
        nav_usd=nav_usd,
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
        data_quality=data_quality,
        scenario_panel=scenario_panel,
        factor_panel=factor_panel,
        concentration_panel=concentration_panel,
        slide_risk_panel=slide_risk_panel,
        borrow_shock_panel=borrow_shock_panel,
        vol_shock_panel=vol_shock_panel,
        action_queue=action_queue,
        alert_rows=alert_rows,
        universe_counts=universe_counts,
        raw_totals=totals,
    )
