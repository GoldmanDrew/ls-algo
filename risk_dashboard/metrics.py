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


def compute_factor_panel(
    underlying_exposure_csv: Path,
    nav_usd: float,
) -> dict[str, Any]:
    """Compute beta-weighted net exposure, sector groupings and top names.

    Inputs are book-level (sleeve-agnostic) so this works even when
    bucket reconciliation is broken. Beta source flags propagate so the
    UI can show coverage honestly.
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
        }

    rows: list[dict[str, Any]] = []
    for _, raw in df.iterrows():
        underlying = _first_nonblank(raw.get("underlying"), raw.get("symbol"))
        symbols = _first_nonblank(raw.get("symbols"), underlying)
        net = float(raw.get("net_notional_usd", 0.0) or 0.0)
        gross = float(raw.get("gross_notional_usd", 0.0) or 0.0)
        legs = int(raw.get("n_legs", 0) or 0)
        meta = lookup_underlying(underlying)
        beta_net = net * meta["beta_to_spy"]
        beta_gross = gross * abs(meta["beta_to_spy"])
        rows.append(
            {
                "underlying": underlying,
                "symbols": symbols,
                "net_notional_usd": net,
                "gross_notional_usd": gross,
                "n_legs": legs,
                "sector": meta["sector"],
                "sector_source": meta["sector_source"],
                "beta_to_spy": meta["beta_to_spy"],
                "beta_source": meta["beta_source"],
                "beta_weighted_net_usd": beta_net,
                "beta_weighted_gross_usd": beta_gross,
            }
        )

    total_net = sum(r["net_notional_usd"] for r in rows)
    total_gross = sum(r["gross_notional_usd"] for r in rows)
    total_beta_net = sum(r["beta_weighted_net_usd"] for r in rows)
    total_beta_gross = sum(r["beta_weighted_gross_usd"] for r in rows)
    known_beta_gross = sum(
        r["gross_notional_usd"] for r in rows if r["beta_source"] == "curated"
    )
    coverage = (known_beta_gross / total_gross) if total_gross > 0 else 0.0

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
    factor_panel = compute_factor_panel(
        underlying_exposure_csv=accounting / "net_exposure_by_underlying.csv",
        nav_usd=nav_usd,
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
        action_queue=action_queue,
        alert_rows=alert_rows,
        universe_counts=universe_counts,
        raw_totals=totals,
    )
