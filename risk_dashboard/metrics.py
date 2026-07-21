"""Risk metrics computed from accounting + Flex outputs.

Every metric here is mapped 1:1 to a row in ``Risk_Dashboard_Plan.md``
-- 4 (limits table). Each row carries its own threshold so the static
site can render a green / amber / red cell without re-implementing the
business logic.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import yaml

from .factor_map import lookup_underlying
from .sector_loader import batch_resolve, resolve_sector
from .sector_vendor import fetch_vendor_info
from .flex_parser import (
    FlexBorrowFee,
    FlexPosition,
    parse_borrow_fee_details,
    parse_flex_nav,
    parse_positions,
    summarize_borrow,
    summarize_positions,
)
from .scenario_engine import (
    SLIDE_SCENARIO_HORIZONS,
    aggregate_leg_scenario_pnl,
    horizon_to_years,
    resolve_sigma_annual,
    scale_spx_shock_for_horizon,
)
from .borrow_stress import build_vix_cumulative_path, load_borrow_stress_config, resolve_borrow_lift
from .carry_validation import compute_carry_validation
from .spx_scenario import (
    HISTORICAL_SPX_SCENARIOS,
    PATH_STEPS_DEFAULT,
    aggregate_path_scenario_pnl,
    historical_scenario_specs as historical_spx_scenario_specs,
)
from .spx_shock_config import load_spx_shock_config
from .spx_stress_beta import leg_instant_price_return, underlying_return_for_leg
from .vol_vix_model import (
    beta_summary_dict,
    compute_vol_vix_pack,
    leg_sigma_for_vix_scenario,
    load_vol_vix_config,
)
from .vix_scenario import HISTORICAL_VIX_SCENARIOS, ScenarioMode, historical_scenario_specs

try:
    from .beta_loader import compute_betas, write_summary_cache  # noqa: F401
except Exception:  # pragma: no cover - optional dep / fallback
    compute_betas = None  # type: ignore[assignment]
    write_summary_cache = None  # type: ignore[assignment]

try:
    from ibkr_accounting import _filter_exposure_df
except ImportError:  # pragma: no cover
    _filter_exposure_df = None  # type: ignore[assignment]

try:
    from reporting_scope import (
        RECONCILE_EXPOSURE_BUCKETS,
        STOCK_SLEEVE_BUCKETS,
        load_blocked_exposure_keys as _scope_blocked_exposure_keys,
    )
except ImportError:  # pragma: no cover
    RECONCILE_EXPOSURE_BUCKETS = ("bucket_1", "bucket_2", "bucket_4")
    STOCK_SLEEVE_BUCKETS = ("bucket_1", "bucket_2", "bucket_4")
    _scope_blocked_exposure_keys = None  # type: ignore[assignment]

# Display-only stock sleeves matching the EOD email (B1+B2+B4+B5). Bucket 5 is a
# real stock sleeve (volatility ETP) but is intentionally excluded from
# ``RECONCILE_EXPOSURE_BUCKETS`` because ``gross_exposure_total`` in totals.json
# is B1+B2+B4 (+unbucketed); adding B5 to the reconcile set would break the gate
# that mirrors ``ibkr_accounting``. So B5 is shown but not reconciled.
DISPLAY_STOCK_SLEEVE_BUCKETS: tuple[str, ...] = tuple(STOCK_SLEEVE_BUCKETS) + ("bucket_5",)


TRADING_DAYS_PER_YEAR_DAYS: int = 252


def _load_blocked_exposure_keys(
    runs_root: Path,
    screener_csv: Path | None = None,
    run_date: str | None = None,
) -> set[str]:
    """Symbols and underlyings excluded from gross/net exposure metrics."""
    if _scope_blocked_exposure_keys is None:
        return set()
    project_root = runs_root.parent.parent
    return _scope_blocked_exposure_keys(
        screener_csv=screener_csv,
        run_date=run_date,
        runs_root=runs_root,
        project_root=project_root,
    )


# Fallback limits when strategy_config.yml is missing. Production uses
# ``load_risk_limits()`` which merges YAML borrow bands and sleeve caps.
DEFAULT_LIMITS: dict[str, dict[str, Any]] = {
    "gross_exposure_pct_nav": {"warn": 4.0, "hard": 4.5},
    "abs_net_beta": {"warn": 0.15, "hard": 0.20},
    "sleeve_drift_pp": {"warn": 5.0, "hard": 10.0},
    "es99_1d_pct_nav": {"warn": 0.012, "hard": 0.020},
    "scenario_loss_pct_nav": {"warn": -0.05, "hard": -0.07},
    "single_sector_gross_pct_book": {"warn": 0.30, "hard": 0.50},
    "top10_gross_pct_nav": {"warn": 0.50, "hard": 0.75},
    "hhi_underlying": {"warn": 1500.0, "hard": 2500.0},
    "hhi_sector": {"warn": 2500.0, "hard": 4000.0},
    "liquidity_short_vs_cap": {"warn": 0.80, "hard": 1.00},
}

BUCKET_SLEEVE_KEYS: dict[str, str] = {
    "bucket_1": "core_leveraged",
    "bucket_2": "yieldboost",
    "bucket_4": "inverse_decay_bucket4",
    "bucket_5": "volatility_etp_bucket5",
}

# Accounting buckets surfaced across the dashboard (sleeve table, bucket tabs,
# factor-by-bucket, data-quality scan). Bucket 5 (volatility ETP) is a real
# stock sleeve and must appear everywhere B1/B2/B4 do.
BUCKET_KEYS: tuple[str, ...] = (
    "bucket_1",
    "bucket_2",
    "bucket_3",
    "bucket_4",
    "bucket_5",
)
BUCKETS_WITHOUT_ROC: frozenset[str] = frozenset({"bucket_3"})
PNL_HISTORY_START_DATE = "2026-02-27"

BUCKET_LABELS: dict[str, str] = {
    "bucket_1": "Bucket 1 (core leveraged)",
    "bucket_2": "Bucket 2 (yield boost)",
    "bucket_3": "Bucket 3 (flow hedge overlay)",
    "bucket_4": "Bucket 4 (inverse / decay)",
    "bucket_5": "Bucket 5 (volatility ETP)",
    "unbucketed": "Unbucketed",
}

# Factor overlays (informative; excluded from additive sleeve sum).
FACTOR_OVERLAY_BUCKETS: tuple[str, ...] = ("bucket_3", "bucket_5")
# Ratio columns on bucket_exposure_detail.csv → additive B1/B2/B4.
_DETAIL_RATIO_COLS: tuple[tuple[str, str], ...] = (
    ("bucket_1", "_ratio_b1"),
    ("bucket_2", "_ratio_b2"),
    ("bucket_4", "_ratio_b4"),
)

DISPLAY_SLEEVE_GROUPS: dict[str, dict[str, Any]] = {
    "b1245": {
        "id": "b1245",
        "label": "Buckets 1, 2, 4, and 5",
        "buckets": DISPLAY_STOCK_SLEEVE_BUCKETS,
        "exposure_note": "Share-notional book (B1+B2+B4 ratio split + B5 vol ETP); excludes B3 overlay.",
    },
    "b3": {
        "id": "b3",
        "label": "Bucket 3",
        "buckets": ("bucket_3",),
        "exposure_note": "Delta-normalized flow hedge overlay (|β| × notional), not share gross.",
    },
}

DEFAULT_EXPOSURE_RECON_TOL_GROSS_PCT = 0.001
DEFAULT_EXPOSURE_RECON_TOL_NET_ABS_USD = 500.0


@dataclass
class RiskLimitsContext:
    """Limits and per-bucket thresholds from ``config/strategy_config.yml``."""

    limits: dict[str, dict[str, Any]]
    borrow_apr_pct_by_bucket: dict[str, dict[str, float]]
    underlying_gross_frac_by_bucket: dict[str, dict[str, float]]
    liquidity_cap_fracs: dict[str, float]
    sleeve_target_weights: dict[str, float | None] = field(default_factory=dict)
    book_target_gross_usd: float = 0.0
    sleeve_target_source: str = "fallback"


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _normalize_two_nonnegative_weights(a: float, b: float) -> tuple[float, float]:
    aa = max(0.0, float(a))
    bb = max(0.0, float(b))
    s = aa + bb
    if s <= 1e-18:
        return 1.0, 0.0
    return aa / s, bb / s


def compute_sleeve_target_weights(cfg: dict[str, Any]) -> tuple[dict[str, float | None], float]:
    """Mirror ``generate_trade_plan.py`` sleeve budget shares (fractions of total gross).

    Returns (bucket -> target weight, book_target_gross_usd).
    """
    strategy = cfg.get("strategy") or {}
    sleeves = (cfg.get("portfolio") or {}).get("sleeves") or {}
    capital_usd = float(strategy.get("capital_usd", 0.0) or 0.0)
    gross_leverage = float(strategy.get("gross_leverage", 0.0) or 0.0)
    book_target_gross_usd = capital_usd * gross_leverage

    core = sleeves.get("core_leveraged") or {}
    b4 = sleeves.get("inverse_decay_bucket4") or {}
    yb = sleeves.get("yieldboost") or {}

    b4_w = float(b4.get("target_weight", 0.0) or 0.0)
    b4_enabled = bool(b4.get("enabled", True))
    yb_enabled = bool(yb.get("enabled", True))
    core_stock_frac, yb_stock_frac = _normalize_two_nonnegative_weights(
        float(core.get("target_weight", 1.0) or 0.0),
        float(yb.get("target_weight", 0.0) or 0.0) if yb_enabled else 0.0,
    )
    stock_nominal_w = max(0.0, min(1.0, 1.0 - b4_w)) if b4_enabled else 1.0

    b4_rules = b4.get("rules") or {}
    vol_cfg = (
        b4_rules.get("volatility_etp_bucket5")
        or b4_rules.get("volatility_etp_bucket4")
        or {}
    )
    vol_enabled = bool(vol_cfg.get("enabled", False))
    if "target_weight" in vol_cfg:
        vol_etp_book_weight = _clamp01(float(vol_cfg.get("target_weight", 0.0) or 0.0))
    else:
        vol_etp_book_weight = _clamp01(b4_w * float(vol_cfg.get("share_of_b4_budget", 0.0) or 0.0))

    # Production uses explicit dollar ceilings when present: B2 and B4 are fixed,
    # B5 is independent, and B1 receives the residual. Keep the legacy normalized
    # weight path for configs that do not provide those ceilings.
    yb_fixed = float(yb.get("target_gross_usd", 0.0) or 0.0)
    b4_fixed = float(b4.get("target_gross_usd", 0.0) or 0.0)
    if book_target_gross_usd > 0 and (yb_fixed > 0 or b4_fixed > 0):
        b2_weight = _clamp01(yb_fixed / book_target_gross_usd) if yb_fixed > 0 else stock_nominal_w * yb_stock_frac
        b4_core_weight = _clamp01(b4_fixed / book_target_gross_usd) if b4_fixed > 0 else 0.0
        vol_weight = vol_etp_book_weight if (vol_enabled and vol_etp_book_weight > 0) else None
        b5_weight = float(vol_weight or 0.0)
        b1_weight = max(0.0, 1.0 - b2_weight - b4_core_weight - b5_weight)
        weights: dict[str, float | None] = {
            "bucket_1": b1_weight,
            "bucket_2": b2_weight,
            "bucket_3": None,
            "bucket_4": b4_core_weight if b4_core_weight > 0 else None,
            "bucket_5": vol_weight,
        }
        return weights, book_target_gross_usd

    b4_core_weight = max(0.0, b4_w - vol_etp_book_weight) if (b4_enabled and b4_w > 0) else 0.0
    vol_weight = vol_etp_book_weight if (b4_enabled and vol_enabled and vol_etp_book_weight > 0) else None

    weights: dict[str, float | None] = {
        "bucket_1": stock_nominal_w * core_stock_frac if stock_nominal_w > 0 else core_stock_frac,
        "bucket_2": stock_nominal_w * yb_stock_frac if (stock_nominal_w > 0 and yb_enabled) else 0.0,
        "bucket_3": None,
        "bucket_4": b4_core_weight if b4_core_weight > 0 else (b4_w if b4_enabled and b4_w > 0 else None),
        "bucket_5": vol_weight,
    }
    if not yb_enabled:
        weights["bucket_2"] = 0.0
    return weights, book_target_gross_usd


# Legacy fallback if strategy_config.yml is missing or unreadable.
FALLBACK_SLEEVE_TARGET_WEIGHTS: dict[str, float | None] = {
    "bucket_1": 0.55,
    "bucket_2": 0.20,
    "bucket_3": None,
    "bucket_4": 0.25,
    "bucket_5": None,
}

# Back-compat alias used by tests and snapshot metadata until YAML load runs.
SLEEVE_TARGET_WEIGHTS = FALLBACK_SLEEVE_TARGET_WEIGHTS


def _decimal_borrow_to_pct_limits(entry_cap: float, keep_cap: float) -> dict[str, float]:
    return {"warn": float(entry_cap) * 100.0, "hard": float(keep_cap) * 100.0}


def load_risk_limits(config_yml: Path | None = None) -> RiskLimitsContext:
    """Load dashboard limits from strategy_config (borrow bands, sleeve caps, liquidity)."""
    limits = {k: dict(v) for k, v in DEFAULT_LIMITS.items()}
    borrow_by_bucket: dict[str, dict[str, float]] = {
        "bucket_1": {"warn": 55.0, "hard": 75.0},
        "bucket_2": {"warn": 40.0, "hard": 50.0},
        "bucket_4": {"warn": 90.0, "hard": 120.0},
        "bucket_5": {"warn": 90.0, "hard": 120.0},
    }
    underlying_by_bucket: dict[str, dict[str, float]] = {
        "bucket_1": {"warn": 0.20, "hard": 0.20},
        "bucket_2": {"warn": 0.20, "hard": 0.20},
        "bucket_4": {"warn": 0.30, "hard": 0.30},
        "bucket_5": {"warn": 0.30, "hard": 0.30},
    }
    liquidity = {"shares_outstanding_use_frac": 0.35, "median_daily_volume_use_pct": 0.30}

    path = config_yml or Path("config/strategy_config.yml")
    cfg: dict[str, Any] = {}
    if path.is_file():
        try:
            cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except Exception:
            cfg = {}
        screener = (cfg.get("screener") or {}) if isinstance(cfg, dict) else {}
        per_bucket = (screener.get("per_bucket") or {}) if isinstance(screener, dict) else {}
        for bkt in ("bucket_1", "bucket_2", "bucket_4", "bucket_5"):
            row = per_bucket.get(bkt) or {}
            if row:
                borrow_by_bucket[bkt] = _decimal_borrow_to_pct_limits(
                    float(row.get("entry_borrow_cap", borrow_by_bucket[bkt]["warn"] / 100.0)),
                    float(row.get("keep_borrow_cap", borrow_by_bucket[bkt]["hard"] / 100.0)),
                )
        gross_caps = ((cfg.get("strategy") or {}).get("gross_sizing_caps") or {}) if isinstance(
            cfg, dict
        ) else {}
        if gross_caps:
            liquidity["shares_outstanding_use_frac"] = float(
                gross_caps.get("shares_outstanding_use_frac", liquidity["shares_outstanding_use_frac"])
            )
            liquidity["median_daily_volume_use_pct"] = float(
                gross_caps.get("median_daily_volume_use_pct", liquidity["median_daily_volume_use_pct"])
            )
        per_sleeve = (gross_caps.get("per_sleeve") or {}) if isinstance(gross_caps, dict) else {}
        for bkt, sleeve_key in BUCKET_SLEEVE_KEYS.items():
            sleeve_row = per_sleeve.get(sleeve_key) or {}
            cap = sleeve_row.get("max_underlying_weight")
            if cap is not None:
                frac = float(cap)
                underlying_by_bucket[bkt] = {"warn": frac, "hard": frac}

    return RiskLimitsContext(
        limits=limits,
        borrow_apr_pct_by_bucket=borrow_by_bucket,
        underlying_gross_frac_by_bucket=underlying_by_bucket,
        liquidity_cap_fracs=liquidity,
        **(_sleeve_target_fields_from_cfg(cfg)),
    )


def _sleeve_target_fields_from_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    if not cfg:
        return {
            "sleeve_target_weights": dict(FALLBACK_SLEEVE_TARGET_WEIGHTS),
            "book_target_gross_usd": 0.0,
            "sleeve_target_source": "fallback",
        }
    weights, book_gross = compute_sleeve_target_weights(cfg)
    return {
        "sleeve_target_weights": weights,
        "book_target_gross_usd": book_gross,
        "sleeve_target_source": "config:strategy_config.yml",
    }

BREACH_CATEGORY_LABELS: dict[str, str] = {
    "book_exposure": "Book exposure & leverage",
    "sleeve_allocation": "Sleeve allocation",
    "concentration": "Concentration limits",
    "market_risk": "Market risk (beta / shocks)",
    "borrow_cost": "Borrow cost (short ETF sleeve)",
    "borrow_availability": "Borrow availability / squeeze",
    "data_quality": "Data & reconciliation",
}


def _breach_category(metric: str, source: str = "") -> str:
    m = (metric or "").lower()
    src = (source or "").lower()
    if m.startswith("scenario:") or "shock" in m or "slide" in src:
        return "market_risk"
    if m.startswith("single_name:") or m.startswith("sector:") or m.startswith("top10") or m.startswith("hhi"):
        return "concentration"
    if m.startswith("borrow_squeeze"):
        return "borrow_availability"
    if "borrow" in m or "borrow" in src:
        return "borrow_cost"
    if "gross_exposure" in m or "net_exposure" in m or "sleeve" in m:
        return "book_exposure" if "sleeve" not in m else "sleeve_allocation"
    if "reconcil" in src or "data" in src:
        return "data_quality"
    return "book_exposure"


def _normalize_breach(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    cat = _breach_category(str(out.get("metric", "")), str(out.get("source", "")))
    out["category"] = cat
    out["category_label"] = BREACH_CATEGORY_LABELS.get(cat, cat)
    label = out.get("label")
    if not label:
        metric = str(out.get("metric", ""))
        if metric.startswith("single_name:"):
            label = metric.split(":", 1)[-1]
        elif metric.startswith("sector:"):
            label = metric.split(":", 1)[-1]
        elif metric.startswith("scenario:"):
            label = metric.split(":", 1)[-1]
        else:
            label = metric.replace("_", " ")
        out["label"] = label
    return out


def _squeeze_binding_cap(
    ratio_out: float | None,
    ratio_adv: float | None,
) -> str | None:
    """Return which liquidity cap binds (drives liquidity_utilization)."""
    candidates: list[tuple[str, float]] = []
    if ratio_out is not None:
        candidates.append(("shares_outstanding", float(ratio_out)))
    if ratio_adv is not None:
        candidates.append(("median_volume", float(ratio_adv)))
    if not candidates:
        return None
    return max(candidates, key=lambda x: x[1])[0]


def _squeeze_cap_labels(
    binding_cap: str | None,
    *,
    sout_frac: float,
    adv_frac: float,
    sh_out: float | None,
    med_vol: float | None,
    cap_shares_out: float | None,
    cap_adv: float | None,
) -> tuple[str | None, str | None]:
    """Human labels for binding and non-binding caps."""
    if binding_cap == "shares_outstanding":
        binding = (
            f"shares-out ({sout_frac:.0%} × {sh_out:,.0f} out = {cap_shares_out:,.0f} sh cap)"
            if cap_shares_out and sh_out
            else f"shares-out ({sout_frac:.0%} of shares outstanding)"
        )
        other = (
            f"median vol ({adv_frac:.0%} × {med_vol:,.0f} med = {cap_adv:,.0f} sh cap)"
            if cap_adv and med_vol
            else None
        )
        return binding, other
    if binding_cap == "median_volume":
        binding = (
            f"median vol ({adv_frac:.0%} × {med_vol:,.0f} 60d med = {cap_adv:,.0f} sh cap)"
            if cap_adv and med_vol
            else f"median vol ({adv_frac:.0%} of 60d median daily volume)"
        )
        other = (
            f"shares-out ({sout_frac:.0%} × {sh_out:,.0f} out = {cap_shares_out:,.0f} sh cap)"
            if cap_shares_out and sh_out
            else None
        )
        return binding, other
    return None, None


def _squeeze_breach_copy(
    sym: str,
    *,
    short_qty: float,
    binding_cap: str | None,
    binding_label: str | None,
    other_label: str | None,
    ratio_out: float | None,
    ratio_adv: float | None,
    cap_shares_out: float | None,
    cap_adv: float | None,
    ratio_peak: float,
    status: str,
) -> tuple[str, str]:
    """Source + action text for squeeze alert rows."""
    qty_txt = f"{short_qty:,.0f} sh short"
    bind_txt = binding_label or "liquidity cap"
    source_parts = [qty_txt, f"Binding: {bind_txt} ({ratio_peak:.0%} of cap)"]
    if other_label and ratio_out is not None and ratio_adv is not None:
        other_ratio = ratio_adv if binding_cap == "shares_outstanding" else ratio_out
        source_parts.append(f"Other: {other_label} ({other_ratio:.0%} of cap)")

    cap_qty = cap_adv if binding_cap == "median_volume" else cap_shares_out
    if cap_qty and cap_qty > 0:
        over_sh = max(short_qty - cap_qty, 0.0)
        trim_warn = max(short_qty - cap_qty * 0.8, 0.0)
        if status == "hard":
            action = (
                f"Cut ~{trim_warn:,.0f} sh on {sym} to reach warn band "
                f"({over_sh:,.0f} sh over {bind_txt})."
            )
        else:
            action = (
                f"Trim ~{trim_warn:,.0f} sh on {sym} toward warn band "
                f"({qty_txt} vs {cap_qty:,.0f} sh {bind_txt})."
            )
    else:
        action = f"Reduce {sym} short; position exceeds sizing liquidity cap."
    return " · ".join(source_parts), action


def _action_dedupe_key(action: dict[str, Any]) -> str:
    return "|".join(
        [
            str(action.get("category", "")),
            str(action.get("title", "")),
            str(action.get("sleeve", "")),
        ]
    )


def _alert_dedupe_key(row: dict[str, Any]) -> str:
    return "|".join(
        [
            str(row.get("category", "")),
            str(row.get("metric", "")),
            str(row.get("label", "")),
        ]
    )


def _load_screener_etf_symbols(screener_csv: Path | None) -> set[str]:
    symbols: set[str] = set()
    if screener_csv is None or not screener_csv.is_file():
        return symbols
    try:
        df = pd.read_csv(screener_csv, usecols=["ETF"])
        for val in df["ETF"].dropna().tolist():
            sym = _clean_str(val).upper()
            if sym:
                symbols.add(sym)
    except Exception:
        pass
    return symbols


def _load_short_position_symbols(flex_positions_xml: Path | None) -> set[str]:
    if flex_positions_xml is None or not flex_positions_xml.is_file():
        return set()
    shorts: set[str] = set()
    for p in parse_positions(flex_positions_xml):
        if float(getattr(p, "position_value", 0) or 0) < 0:
            sym = (getattr(p, "symbol", "") or "").strip().upper()
            if sym:
                shorts.add(sym)
    return shorts


def _borrow_watchlist_symbols(
    *,
    screener_csv: Path | None,
    flex_positions_xml: Path | None,
) -> set[str]:
    """Universe for borrow panels: held shorts ∪ ETFs in ``etf_screened_today``."""
    return _load_screener_etf_symbols(screener_csv) | _load_short_position_symbols(
        flex_positions_xml
    )


def _load_screener_borrow_meta(screener_csv: Path | None) -> dict[str, dict[str, Any]]:
    """Per-ETF fields from ``etf_screened_today.csv`` (borrow + bucket assignment)."""
    meta: dict[str, dict[str, Any]] = {}
    if screener_csv is None or not screener_csv.is_file():
        return meta
    try:
        screener = pd.read_csv(screener_csv)
    except Exception:
        return meta
    if "ETF" not in screener.columns:
        return meta
    for _, r in screener.iterrows():
        key = _clean_str(r.get("ETF")).upper()
        if not key:
            continue
        row: dict[str, Any] = {}
        if "bucket" in screener.columns and pd.notna(r.get("bucket")):
            row["bucket"] = _clean_str(r.get("bucket"))
        if "borrow_price_ref" in screener.columns and pd.notna(r.get("borrow_price_ref")):
            row["borrow_price_ref"] = float(r.get("borrow_price_ref"))
        for col in ("borrow_current", "borrow_fee_annual", "borrow_net_annual"):
            if col in screener.columns and pd.notna(r.get(col)):
                row[col] = float(r.get(col))
        meta[key] = row
    return meta


def _etf_dashboard_data_paths(repo_root: Path, cfg: dict[str, Any]) -> list[Path]:
    """Candidate etf-dashboard data files (sibling repo or vendored copy under data/)."""
    repo_root = repo_root.resolve()
    paths_cfg = (cfg.get("paths") or {}) if isinstance(cfg, dict) else {}
    out: list[Path] = []
    for key in ("etf_metrics_daily_csv", "etf_metrics_latest_json"):
        configured = paths_cfg.get(key)
        if configured:
            out.append(repo_root / str(configured))
    out.extend(
        [
            repo_root.parent / "etf-dashboard" / "data" / "etf_metrics_daily.csv",
            repo_root.parent / "etf-dashboard" / "data" / "etf_metrics_latest.json",
            repo_root / "etf-dashboard" / "data" / "etf_metrics_daily.csv",
            repo_root / "etf-dashboard" / "data" / "etf_metrics_latest.json",
            repo_root / "data" / "etf_metrics_daily.csv",
            repo_root / "data" / "etf_metrics_latest.json",
        ]
    )
    seen: set[str] = set()
    unique: list[Path] = []
    for p in out:
        key = str(p.resolve()) if p.exists() else str(p)
        if key in seen:
            continue
        seen.add(key)
        unique.append(p)
    return unique


def _load_shares_outstanding_from_etf_metrics(repo_root: Path, cfg: dict[str, Any]) -> dict[str, float]:
    """Latest shares_outstanding per ticker from etf-dashboard metrics ingest."""
    for path in _etf_dashboard_data_paths(repo_root, cfg):
        if path.suffix.lower() == ".json" and path.is_file():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            rows = payload.get("rows") or []
            out: dict[str, float] = {}
            for row in rows:
                sym = _clean_str(row.get("ticker") or row.get("ETF")).upper()
                sh = row.get("shares_outstanding")
                if sym and sh is not None and float(sh) > 0:
                    out[sym] = float(sh)
            if out:
                return out
        if path.suffix.lower() == ".csv" and path.is_file():
            try:
                df = pd.read_csv(path, usecols=["date", "ticker", "shares_outstanding"])
            except Exception:
                try:
                    df = pd.read_csv(path)
                except Exception:
                    continue
            if "ticker" not in df.columns or "shares_outstanding" not in df.columns:
                continue
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.dropna(subset=["date", "ticker"])
            df["shares_outstanding"] = pd.to_numeric(df["shares_outstanding"], errors="coerce")
            latest = (
                df.sort_values("date")
                .groupby(df["ticker"].astype(str).str.upper(), as_index=False)
                .tail(1)
            )
            out = {
                _clean_str(r["ticker"]).upper(): float(r["shares_outstanding"])
                for _, r in latest.iterrows()
                if pd.notna(r["shares_outstanding"]) and float(r["shares_outstanding"]) > 0
            }
            if out:
                return out
    return {}


def _load_median_volume_from_etf_metrics(
    repo_root: Path, cfg: dict[str, Any], *, lookback_days: int = 60
) -> dict[str, float]:
    """Median daily ``shares_traded`` over recent history (etf-dashboard metrics)."""
    for path in _etf_dashboard_data_paths(repo_root, cfg):
        if path.suffix.lower() != ".csv" or not path.is_file():
            continue
        try:
            df = pd.read_csv(path, usecols=["date", "ticker", "shares_traded"])
        except Exception:
            try:
                df = pd.read_csv(path)
            except Exception:
                continue
        if "ticker" not in df.columns or "shares_traded" not in df.columns:
            continue
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["shares_traded"] = pd.to_numeric(df["shares_traded"], errors="coerce")
        df = df.dropna(subset=["date", "ticker"])
        df = df[df["shares_traded"] > 0]
        if df.empty:
            continue
        max_date = df["date"].max()
        cutoff = max_date - pd.Timedelta(days=int(lookback_days) * 2)
        df = df[df["date"] >= cutoff]
        med = (
            df.groupby(df["ticker"].astype(str).str.upper())["shares_traded"]
            .median()
            .dropna()
        )
        return {str(k).upper(): float(v) for k, v in med.items() if float(v) > 0}
    return {}


def _load_shares_outstanding_map(repo_root: Path, cfg: dict[str, Any]) -> dict[str, float]:
    repo_root = repo_root.resolve()
    try:
        from generate_trade_plan import load_shares_outstanding_map

        paths_cfg = dict((cfg.get("paths") or {}))
        out, src = load_shares_outstanding_map(paths_cfg)
        if out:
            return out
    except Exception:
        pass
    paths_cfg = (cfg.get("paths") or {}) if isinstance(cfg, dict) else {}
    configured = paths_cfg.get("etf_shares_outstanding_csv")
    candidates: list[Path] = []
    if configured:
        candidates.append((repo_root / str(configured)).resolve())
    candidates.extend(
        [
            repo_root.parent / "etf-dashboard" / "data" / "etf_shares_outstanding.csv",
            repo_root / "data" / "etf_shares_outstanding.csv",
        ]
    )
    for path in candidates:
        if not path.is_file():
            continue
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        sym_col = next((c for c in df.columns if c.upper() in ("ETF", "SYMBOL", "TICKER")), None)
        sh_col = next(
            (c for c in df.columns if "shares" in c.lower() and "out" in c.lower()),
            None,
        )
        if sym_col is None or sh_col is None:
            continue
        out_map: dict[str, float] = {}
        for _, r in df[[sym_col, sh_col]].dropna().iterrows():
            sym = _clean_str(r[sym_col]).upper()
            if sym:
                out_map[sym] = float(r[sh_col])
        if out_map:
            return out_map
    return _load_shares_outstanding_from_etf_metrics(repo_root, cfg)


def _load_median_daily_volume_map(repo_root: Path, cfg: dict[str, Any]) -> dict[str, float]:
    repo_root = repo_root.resolve()
    paths_cfg = (cfg.get("paths") or {}) if isinstance(cfg, dict) else {}
    configured = paths_cfg.get("etf_median_daily_volume_csv")
    candidates: list[Path] = []
    if configured:
        candidates.append((repo_root / str(configured)).resolve())
    candidates.extend(
        [
            repo_root.parent / "etf-dashboard" / "data" / "etf_median_daily_volume.csv",
            repo_root.parent / "etf-dashboard" / "data" / "etf_adv.csv",
            repo_root / "data" / "etf_median_daily_volume.csv",
        ]
    )
    vol_cols = (
        "median_daily_volume_shares",
        "median_volume_shares_60d",
        "adv_median_shares",
        "median_volume_shares",
    )
    for path in candidates:
        if not path.is_file():
            continue
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        sym_col = next((c for c in df.columns if c.upper() in ("ETF", "SYMBOL", "TICKER")), None)
        vol_col = next((c for c in vol_cols if c in df.columns), None)
        if sym_col is None or vol_col is None:
            continue
        out: dict[str, float] = {}
        for _, r in df[[sym_col, vol_col]].dropna().iterrows():
            sym = _clean_str(r[sym_col]).upper()
            if sym:
                out[sym] = float(r[vol_col])
        if out:
            return out
    return _load_median_volume_from_etf_metrics(repo_root, cfg)


def _symbol_bucket(sym: str, screener_meta: dict[str, dict[str, Any]]) -> str:
    row = screener_meta.get(sym.upper()) or {}
    bkt = _clean_str(row.get("bucket"))
    if bkt in BUCKET_SLEEVE_KEYS:
        return bkt
    return "bucket_4"


def _screener_borrow_decimal(screener_meta: dict[str, dict[str, Any]], symbol: str) -> tuple[float | None, str]:
    """Match etf-dashboard ``_pick_borrow_fee_only``: fee-only annual rate as a decimal."""
    row = screener_meta.get(symbol.upper()) or {}
    for key in ("borrow_current", "borrow_fee_annual", "borrow_net_annual"):
        val = row.get(key)
        if val is None:
            continue
        try:
            return float(val), key
        except (TypeError, ValueError):
            continue
    return None, ""


def _screener_borrow_rate(symbol: str, screener_meta: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Screener borrow rate for dashboard cost math (decimal annual → ``borrow_rate_pct``)."""
    dec, source = _screener_borrow_decimal(screener_meta, symbol)
    if dec is None:
        return {
            "borrow_rate_pct": None,
            "borrow_rate_decimal": None,
            "borrow_rate_known": False,
            "borrow_rate_source": None,
            "current_apr_pct": None,
            "fee_rate_pct": None,
        }
    pct = float(dec) * 100.0
    return {
        "borrow_rate_pct": pct,
        "borrow_rate_decimal": dec,
        "borrow_rate_known": True,
        "borrow_rate_source": source or "screener",
        "current_apr_pct": pct,
        "fee_rate_pct": pct,
    }


def _load_bucket4_symbols(
    net_exposure_bucket4_csv: Path | None,
    screener_csv: Path | None = None,
) -> set[str]:
    """ETF symbols in the short / inverse sleeve (bucket 4)."""
    symbols: set[str] = set()
    if net_exposure_bucket4_csv is not None and net_exposure_bucket4_csv.is_file():
        try:
            df = pd.read_csv(net_exposure_bucket4_csv)
            for _, row in df.iterrows():
                sym_field = _first_nonblank(row.get("symbols"), row.get("symbol"))
                for part in str(sym_field).replace(";", ",").split(","):
                    s = part.strip().upper()
                    if s:
                        symbols.add(s)
        except Exception:
            pass
    if screener_csv is not None and screener_csv.is_file():
        try:
            sdf = pd.read_csv(screener_csv, usecols=["ETF", "product_class"])
            for _, row in sdf.iterrows():
                cls = _clean_str(row.get("product_class")).lower()
                if cls in ("letf_inverse", "inverse", "scraped_inverse"):
                    sym = _clean_str(row.get("ETF")).upper()
                    if sym:
                        symbols.add(sym)
        except Exception:
            pass
    return symbols


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


def _safe_num(value: Any) -> float | None:
    """Float or None; NaN/inf become None so snapshots stay valid JSON."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


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


def exposure_reconciliation_tolerances(totals: dict[str, Any]) -> tuple[float, float]:
    """Gross % and net $ tolerances mirrored from ``ibkr_accounting`` totals.json."""
    gross_pct = float(
        totals.get("exposure_reconciliation_tol_gross_pct", DEFAULT_EXPOSURE_RECON_TOL_GROSS_PCT)
        or DEFAULT_EXPOSURE_RECON_TOL_GROSS_PCT
    )
    net_abs = float(
        totals.get("exposure_reconciliation_tol_net_abs_usd", DEFAULT_EXPOSURE_RECON_TOL_NET_ABS_USD)
        or DEFAULT_EXPOSURE_RECON_TOL_NET_ABS_USD
    )
    return gross_pct, net_abs


def bucket_exposure_component_sums(totals: dict[str, Any]) -> tuple[float, float]:
    """Sum B1+B2+B4 gross/net; both include unbucketed spot (matches accounting gate)."""
    bucket_gross = sum(
        float(totals.get(f"gross_exposure_{b}", 0.0) or 0.0) for b in RECONCILE_EXPOSURE_BUCKETS
    )
    bucket_net = sum(
        float(totals.get(f"net_exposure_{b}", 0.0) or 0.0) for b in RECONCILE_EXPOSURE_BUCKETS
    )
    bucket_gross += float(totals.get("gross_exposure_unbucketed", 0.0) or 0.0)
    bucket_net += float(totals.get("net_exposure_unbucketed", 0.0) or 0.0)
    return bucket_gross, bucket_net


def evaluate_exposure_reconciliation(totals: dict[str, Any]) -> dict[str, Any]:
    """Structured reconciliation result for dashboard gates and CI parity tests."""
    book_gross = float(totals.get("gross_exposure_total", 0.0) or 0.0)
    book_net = float(totals.get("net_exposure_total", 0.0) or 0.0)
    bucket_gross, bucket_net = bucket_exposure_component_sums(totals)
    tol_gross_pct, tol_net_abs = exposure_reconciliation_tolerances(totals)
    if abs(book_gross) > 1e-6:
        gross_diff_pct = abs(bucket_gross - book_gross) / abs(book_gross)
    else:
        gross_diff_pct = 0.0
    net_diff_abs = abs(bucket_net - book_net)
    gross_ok = gross_diff_pct <= tol_gross_pct
    net_ok = net_diff_abs <= tol_net_abs
    return {
        "book_gross_usd": book_gross,
        "book_net_usd": book_net,
        "bucket_gross_usd": bucket_gross,
        "bucket_net_usd": bucket_net,
        "gross_diff_pct": gross_diff_pct,
        "net_diff_abs_usd": net_diff_abs,
        "tol_gross_pct": tol_gross_pct,
        "tol_net_abs_usd": tol_net_abs,
        "components_included": list(RECONCILE_EXPOSURE_BUCKETS) + ["net_exposure_unbucketed"],
        "reconciles": gross_ok and net_ok,
    }


def _bucket_reconciles(totals: dict[str, Any]) -> tuple[bool, str]:
    """Return (sleeve_attribution_available, reason).

    Bucket gross/net components (b1 + b2 + b4 + unbucketed net) must sum to
    the book aggregate within accounting tolerances. Bucket 3 is intentionally
    excluded because it is a delta-normalized hedge OVERLAY, not share-notional.

    Mirrors the upstream accounting reconciliation gate in ``ibkr_accounting.py``.
    """
    recon = evaluate_exposure_reconciliation(totals)
    book_gross = recon["book_gross_usd"]
    if book_gross <= 0:
        return True, ""
    if recon["reconciles"]:
        return True, ""
    gross_diff = recon["gross_diff_pct"]
    net_diff_abs = recon["net_diff_abs_usd"]
    return False, (
        f"Bucket exposures do not reconcile to book aggregate "
        f"(gross diff {gross_diff:.1%} vs tol {recon['tol_gross_pct']:.2%}, "
        f"net abs diff ${net_diff_abs:,.0f} vs tol ${recon['tol_net_abs_usd']:,.0f}). "
        f"Showing sleeve P&L only; gross/net per sleeve suppressed."
    )


def compute_book_summary(
    totals: dict[str, Any],
    pnl_by_bucket: pd.DataFrame,
    nav_usd: float,
    target_weights: dict[str, float | None] | None = None,
    limits: dict[str, dict[str, Any]] | None = None,
    book_target_gross_usd: float | None = None,
) -> BookSummary:
    limits = limits or DEFAULT_LIMITS
    target_weights = target_weights or SLEEVE_TARGET_WEIGHTS

    gross = float(totals.get("gross_exposure_total", 0.0) or 0.0)
    net = float(totals.get("net_exposure_total", 0.0) or 0.0)
    weight_denominator = (
        float(book_target_gross_usd)
        if book_target_gross_usd is not None and book_target_gross_usd > 0
        else gross
    )

    gross_pct = gross / nav_usd if nav_usd > 0 else 0.0
    net_pct = net / nav_usd if nav_usd > 0 else 0.0

    pnl_total = totals.get("total_pnl")
    pnl_pct = (pnl_total / nav_usd) if (nav_usd > 0 and pnl_total is not None) else None

    sleeve_available, sleeve_reason = _bucket_reconciles(totals)

    sleeve_table: list[dict[str, Any]] = []
    bucket_pnl = totals.get("bucket_pnl") or {}
    for bucket in BUCKET_KEYS:
        gross_b_raw = float(totals.get(f"gross_exposure_{bucket}", 0.0) or 0.0)
        net_b_raw = float(totals.get(f"net_exposure_{bucket}", 0.0) or 0.0)
        target_w = target_weights.get(bucket)
        target_gross_usd = (
            (target_w * weight_denominator) if (target_w is not None and weight_denominator > 0) else None
        )
        bucket_label = BUCKET_LABELS.get(bucket, bucket.replace("_", " ").title())
        if sleeve_available:
            gross_b = gross_b_raw
            net_b = net_b_raw
            actual_w = (gross_b / weight_denominator) if weight_denominator > 0 else 0.0
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
                "bucket_label": bucket_label,
                "gross_usd": gross_b,
                "net_usd": net_b,
                "gross_usd_raw": gross_b_raw,
                "net_usd_raw": net_b_raw,
                "target_gross_usd": target_gross_usd,
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
            _normalize_breach(
                {
                    "metric": "gross_exposure_pct_nav",
                    "label": "Gross exposure % NAV",
                    "value": gross_pct,
                    "status": gross_status,
                    "limit": limits["gross_exposure_pct_nav"],
                    "source": "Risk limits — book gross cap",
                    "action": "Reduce gross or explicitly approve temporary breach before adding risk.",
                }
            )
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


def bucket_exposure_header(bucket: str, totals: dict[str, Any] | None) -> dict[str, Any]:
    """Authoritative sleeve exposure from totals.json (ratio-split for B4)."""
    totals = totals or {}
    if bucket == "bucket_4":
        return {
            "attribution_net_usd": float(totals.get("net_exposure_bucket_4", 0.0) or 0.0),
            "attribution_gross_usd": float(totals.get("gross_exposure_bucket_4", 0.0) or 0.0),
            "pair_view_net_usd": float(totals.get("net_exposure_bucket_4_pair", 0.0) or 0.0),
            "pair_view_gross_usd": float(totals.get("gross_exposure_bucket_4_pair", 0.0) or 0.0),
            "source": "totals.json ratio-split (sleeve table); pair CSV is detail-only",
        }
    return {
        "attribution_net_usd": float(totals.get(f"net_exposure_{bucket}", 0.0) or 0.0),
        "attribution_gross_usd": float(totals.get(f"gross_exposure_{bucket}", 0.0) or 0.0),
        "source": "totals.json",
    }


def compute_bucket_detail(
    bucket: str,
    pnl_csv: Path,
    net_exposure_csv: Path,
    *,
    blocked_exposure_keys: set[str] | None = None,
    exposure_detail_csv: Path | None = None,
    totals: dict[str, Any] | None = None,
) -> dict[str, Any]:
    pnl = _read_csv_or_empty(pnl_csv)
    expo = _read_csv_or_empty(net_exposure_csv)
    if blocked_exposure_keys and not expo.empty and _filter_exposure_df is not None:
        underlying_col = "underlying" if "underlying" in expo.columns else "symbol"
        expo = _filter_exposure_df(expo, blocked_exposure_keys, underlying_col=underlying_col)

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

    exposure_leg_rows: list[dict[str, Any]] = []
    if exposure_detail_csv is not None and exposure_detail_csv.is_file():
        leg_df = _read_csv_or_empty(exposure_detail_csv)
        if not leg_df.empty:
            for _, row in leg_df.iterrows():
                underlying = _first_nonblank(row.get("underlying"), row.get("symbol"))
                symbol = _first_nonblank(row.get("symbol"), underlying)
                leg_type = str(row.get("leg_type", "") or "")
                exposure_leg_rows.append(
                    {
                        "underlying": underlying,
                        "symbol": symbol,
                        "leg_type": leg_type,
                        "net_notional_usd": float(row.get("net_notional_usd", 0.0) or 0.0),
                        "gross_notional_usd": float(row.get("gross_notional_usd", 0.0) or 0.0),
                    }
                )
            exposure_leg_rows.sort(
                key=lambda r: (str(r.get("underlying", "")), str(r.get("leg_type", "")))
            )

    return {
        "bucket": bucket,
        "bucket_label": BUCKET_LABELS.get(bucket, bucket),
        "exposure_header": bucket_exposure_header(bucket, totals),
        "pnl_rows": rows,
        "exposure_rows": expo_rows,
        "exposure_leg_rows": exposure_leg_rows,
        "n_pnl_rows": len(rows),
        "n_exposure_rows": len(expo_rows),
        "n_exposure_leg_rows": len(exposure_leg_rows),
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
    *,
    run_date: str | None = None,
    limits_ctx: RiskLimitsContext | None = None,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    ctx = limits_ctx or load_risk_limits()
    limits = limits or ctx.limits
    repo_root = repo_root or Path(".")
    cfg: dict[str, Any] = {}
    cfg_path = repo_root / "config" / "strategy_config.yml"
    if cfg_path.is_file():
        try:
            cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        except Exception:
            cfg = {}

    borrow_rows: list[FlexBorrowFee] = parse_borrow_fee_details(flex_borrow_xml, run_date=run_date)
    positions: list[FlexPosition] = parse_positions(flex_positions_xml)

    watchlist = _borrow_watchlist_symbols(
        screener_csv=screener_csv,
        flex_positions_xml=flex_positions_xml,
    )
    screener_meta = _load_screener_borrow_meta(screener_csv)
    shares_out_map = _load_shares_outstanding_map(repo_root, cfg)
    median_vol_map = _load_median_daily_volume_map(repo_root, cfg)
    sout_frac = float(ctx.liquidity_cap_fracs.get("shares_outstanding_use_frac", 0.35))
    adv_frac = float(ctx.liquidity_cap_fracs.get("median_daily_volume_use_pct", 0.30))
    liq_limits = limits.get(
        "liquidity_short_vs_cap", DEFAULT_LIMITS["liquidity_short_vs_cap"]
    )

    summary = summarize_borrow(borrow_rows)
    pos_summary = summarize_positions(positions)

    short_notional: dict[str, float] = {}
    short_qty_by_sym: dict[str, float] = {}
    for p in positions:
        if float(getattr(p, "position_value", 0) or 0) < 0:
            sym = (getattr(p, "symbol", "") or "").upper()
            if watchlist and sym not in watchlist:
                continue
            short_notional[sym] = short_notional.get(sym, 0.0) + abs(p.position_value)
            short_qty_by_sym[sym] = short_qty_by_sym.get(sym, 0.0) + abs(
                float(getattr(p, "quantity", 0) or 0)
            )

    short_etf_rows: list[dict[str, Any]] = []
    expensive_borrow: list[dict[str, Any]] = []
    breaches: list[dict[str, Any]] = []

    for sym, notional in short_notional.items():
        rate = _screener_borrow_rate(sym, screener_meta)
        bkt = _symbol_bucket(sym, screener_meta)
        bkt_limits = ctx.borrow_apr_pct_by_bucket.get(bkt, ctx.borrow_apr_pct_by_bucket["bucket_4"])
        eff = float(rate["borrow_rate_pct"] or 0.0)
        short_etf_rows.append(
            {
                "symbol": sym,
                "bucket": bkt,
                "short_notional_usd": notional,
                **rate,
                "implied_annual_cost_usd": notional * (eff / 100.0)
                if rate["borrow_rate_known"]
                else None,
                "borrow_limit_pct": bkt_limits,
            }
        )
        if rate["borrow_rate_known"]:
            br_status = _classify(eff, bkt_limits)
            if br_status != "ok":
                expensive_borrow.append({"symbol": sym, "bucket": bkt, **rate})
                breaches.append(
                    _normalize_breach(
                        {
                            "metric": f"borrow_apr:{sym}",
                            "label": sym,
                            "value": eff,
                            **rate,
                            "status": br_status,
                            "limit": bkt_limits,
                            "source": (
                                f"screener borrow vs {bkt} "
                                f"(entry {bkt_limits['warn']:.0f}%, keep {bkt_limits['hard']:.0f}%)"
                            ),
                            "action": "Review borrow economics; reduce or drop names above keep cap.",
                            "sleeve": bkt,
                        }
                    )
                )

    short_etf_rows.sort(
        key=lambda r: (r["implied_annual_cost_usd"] is not None, r["implied_annual_cost_usd"] or 0.0),
        reverse=True,
    )
    expensive_borrow.sort(key=lambda d: -(float(d.get("borrow_rate_pct") or 0.0)))

    squeeze_rows: list[dict[str, Any]] = []
    for sym, notional in short_notional.items():
        rate = _screener_borrow_rate(sym, screener_meta)
        bkt = _symbol_bucket(sym, screener_meta)
        short_qty = short_qty_by_sym.get(sym)
        sh_out = shares_out_map.get(sym)
        med_vol = median_vol_map.get(sym)
        cap_shares_out = (sh_out * sout_frac) if sh_out and sh_out > 0 else None
        cap_adv = (med_vol * adv_frac) if med_vol and med_vol > 0 else None
        ratio_out = (
            (short_qty / cap_shares_out) if (short_qty is not None and cap_shares_out) else None
        )
        ratio_adv = (short_qty / cap_adv) if (short_qty is not None and cap_adv) else None
        ratios = [r for r in (ratio_out, ratio_adv) if r is not None]
        ratio_peak = max(ratios) if ratios else None
        binding_cap = _squeeze_binding_cap(ratio_out, ratio_adv)
        binding_label, other_cap_label = _squeeze_cap_labels(
            binding_cap,
            sout_frac=sout_frac,
            adv_frac=adv_frac,
            sh_out=sh_out,
            med_vol=med_vol,
            cap_shares_out=cap_shares_out,
            cap_adv=cap_adv,
        )
        status = _classify(ratio_peak, liq_limits) if ratio_peak is not None else "unknown"
        squeeze_rows.append(
            {
                "symbol": sym,
                "bucket": bkt,
                "short_qty": short_qty,
                "short_notional_usd": notional,
                "shares_outstanding": sh_out,
                "median_daily_volume_shares": med_vol,
                "cap_shares_out_shares": cap_shares_out,
                "cap_median_vol_shares": cap_adv,
                "short_vs_shares_out_cap": ratio_out,
                "short_vs_adv_cap": ratio_adv,
                "binding_cap": binding_cap,
                "binding_cap_label": binding_label,
                "other_cap_label": other_cap_label,
                "liquidity_utilization": ratio_peak,
                "status": status,
                **rate,
            }
        )
        if status in ("warn", "hard") and short_qty is not None and ratio_peak is not None:
            source, action = _squeeze_breach_copy(
                sym,
                short_qty=float(short_qty),
                binding_cap=binding_cap,
                binding_label=binding_label,
                other_label=other_cap_label,
                ratio_out=ratio_out,
                ratio_adv=ratio_adv,
                cap_shares_out=cap_shares_out,
                cap_adv=cap_adv,
                ratio_peak=float(ratio_peak),
                status=status,
            )
            breaches.append(
                _normalize_breach(
                    {
                        "metric": f"borrow_squeeze:{sym}",
                        "label": sym,
                        "value": ratio_peak,
                        "status": status,
                        "limit": liq_limits,
                        "source": source,
                        "action": action,
                        "sleeve": bkt,
                        "short_qty": short_qty,
                        "binding_cap": binding_cap,
                        "binding_cap_label": binding_label,
                        "other_cap_label": other_cap_label,
                        "cap_shares_out_shares": cap_shares_out,
                        "cap_median_vol_shares": cap_adv,
                        "short_vs_shares_out_cap": ratio_out,
                        "short_vs_adv_cap": ratio_adv,
                    }
                )
            )

    squeeze_rows.sort(
        key=lambda r: (-1.0 if r["liquidity_utilization"] is None else r["liquidity_utilization"]),
        reverse=True,
    )

    return {
        "borrow": {
            **summary,
            "names_over_30pct": expensive_borrow,
            "names_over_30pct_flex": summary.get("names_over_30pct", []),
        },
        "positions": pos_summary,
        "breaches": breaches,
        "squeeze_rows": squeeze_rows,
        "short_etf_rows": short_etf_rows,
        "watchlist_n_symbols": len(watchlist),
        "n_short_etfs": len(short_etf_rows),
        "borrow_limits_by_bucket": ctx.borrow_apr_pct_by_bucket,
        "liquidity_cap_fracs": ctx.liquidity_cap_fracs,
        "liquidity_data": {
            "n_shares_outstanding": len(shares_out_map),
            "n_median_volume": len(median_vol_map),
            "n_short_with_liquidity_util": sum(
                1 for r in squeeze_rows if r.get("liquidity_utilization") is not None
            ),
        },
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
    for bucket in BUCKET_KEYS:
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
    recon = evaluate_exposure_reconciliation(totals)
    if recon["book_gross_usd"]:
        reconciliations.append(
            {
                "name": "bucket_gross_vs_book_gross",
                "status": "ok" if recon["gross_diff_pct"] <= recon["tol_gross_pct"] else "hard",
                "book_value": recon["book_gross_usd"],
                "component_sum": recon["bucket_gross_usd"],
                "diff_pct": recon["gross_diff_pct"],
                "tol_pct": recon["tol_gross_pct"],
                "components_included": list(RECONCILE_EXPOSURE_BUCKETS),
            }
        )
        reconciliations.append(
            {
                "name": "bucket_net_vs_book_net",
                "status": "ok" if recon["net_diff_abs_usd"] <= recon["tol_net_abs_usd"] else "hard",
                "book_value": recon["book_net_usd"],
                "component_sum": recon["bucket_net_usd"],
                "diff_abs_usd": recon["net_diff_abs_usd"],
                "tol_abs_usd": recon["tol_net_abs_usd"],
                "components_included": recon["components_included"],
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

    If ``factor_panel`` is supplied, an additional family of delta-adjusted
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
            (-0.03, "spx_beta_down_3", "SPX -3% (delta-adj)"),
            (-0.05, "spx_beta_down_5", "SPX -5% (delta-adj)"),
            (-0.10, "spx_beta_down_10", "SPX -10% (delta-adj)"),
            (0.05, "spx_beta_up_5", "SPX +5% (delta-adj)"),
        ):
            contributors: list[dict[str, Any]] = []
            for r in factor_rows:
                beta_net = r.get("beta_weighted_net_usd", r.get("delta_weighted_net_usd", 0.0))
                pnl = beta_net * shock_pct
                if pnl == 0:
                    continue
                beta_src = r.get("beta_source", r.get("delta_source", ""))
                contributors.append(
                    {
                        "bucket": r.get("sector", "other"),
                        "underlying": r.get("underlying", ""),
                        "symbols": r.get("symbols", ""),
                        "pnl_usd": pnl,
                        "driver": f"SPX {shock_pct:+.0%} x beta {r['beta_to_spy']:+.2f}"
                                  + (" (default)" if beta_src == "default" else ""),
                        "net_notional_usd": r.get("net_notional_usd", 0.0),
                        "gross_notional_usd": r.get("gross_notional_usd", 0.0),
                        "beta_to_spy": r["beta_to_spy"],
                        "beta_source": beta_src,
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


def _load_screener_underlying_rows(
    screener_csv: Path | None,
) -> dict[str, dict[str, Any]]:
    """Underlying -> screener metadata (instrument class + optional economic sector)."""
    if screener_csv is None or not screener_csv.is_file():
        return {}
    try:
        df = pd.read_csv(screener_csv, low_memory=False)
    except Exception:
        return {}
    if "Underlying" not in df.columns:
        return {}
    out: dict[str, dict[str, Any]] = {}
    for _, r in df.iterrows():
        u = _clean_str(r.get("Underlying")).upper()
        if not u or u in out:
            continue
        row: dict[str, Any] = {}
        pc = _clean_str(r.get("product_class")) or _clean_str(r.get("Delta_product_class"))
        if pc:
            row["instrument_class"] = pc.lower()
            row["product_class"] = pc.lower()
        for col in ("underlying_sector", "sector", "theme", "underlying_theme"):
            val = _clean_str(r.get(col))
            if val and val.lower() not in {"nan", "none", "other", "unknown", ""}:
                row.setdefault("underlying_sector", val.lower())
                break
        out[u] = row
    return out


def _resolve_underlying_betas(
    underlying: str | None,
    *,
    beta_results: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return SPY/QQQ/IWM/BTC betas and source. OLS loader only — not curated map."""
    from .factor_map import DEFAULT_SINGLE_NAME_BETA

    beta_to_spy = float(DEFAULT_SINGLE_NAME_BETA)
    beta_to_qqq: float | None = None
    beta_to_iwm: float | None = None
    beta_to_btc: float | None = None
    beta_source = "default_fallback"
    if beta_results:
        br = beta_results.get((underlying or "").strip().upper())
        if br:
            provenance = br.get("provenance", "")
            if provenance in (
                "computed",
                "shrunk",
                "curated_fallback",
                "default_fallback",
            ):
                beta_source = provenance
            if br.get("beta_to_spy") is not None:
                beta_to_spy = float(br["beta_to_spy"])
            if br.get("beta_to_ndx") is not None:
                beta_to_qqq = float(br["beta_to_ndx"])
            if br.get("beta_to_rut") is not None:
                beta_to_iwm = float(br["beta_to_rut"])
            if br.get("beta_to_btc") is not None:
                beta_to_btc = float(br["beta_to_btc"])
    return {
        "beta_to_spy": beta_to_spy,
        "beta_to_qqq": beta_to_qqq,
        "beta_to_iwm": beta_to_iwm,
        "beta_to_btc": beta_to_btc,
        "beta_source": beta_source,
    }


def _write_sector_audit_csv(rows: list[dict[str, Any]], path: Path) -> None:
    """Persist per-name sector attribution for drift review."""
    if not rows:
        return
    cols = [
        "underlying",
        "sector",
        "sector_source",
        "sector_confidence",
        "instrument_class",
        "beta_to_spy",
        "beta_source",
        "net_notional_usd",
        "gross_notional_usd",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            w.writerow([r.get(c, "") for c in cols])


def _empty_factor_bucket_row(
    bucket: str,
    nav_usd: float,
    *,
    additive: bool,
    role: str,
    attribution_mode: str,
) -> dict[str, Any]:
    return {
        "bucket": bucket,
        "bucket_label": BUCKET_LABELS.get(bucket, bucket),
        "n_names": 0,
        "net_notional_usd": 0.0,
        "gross_notional_usd": 0.0,
        "beta_weighted_net_usd": 0.0,
        "beta_weighted_gross_usd": 0.0,
        "beta_weighted_net_qqq_usd": 0.0,
        "beta_weighted_net_iwm_usd": 0.0,
        "beta_weighted_net_btc_usd": 0.0,
        "net_beta_to_spy": 0.0 if nav_usd > 0 else None,
        "net_beta_to_qqq": 0.0 if nav_usd > 0 else None,
        "net_beta_to_iwm": 0.0 if nav_usd > 0 else None,
        "net_beta_to_btc": 0.0 if nav_usd > 0 else None,
        "gross_beta_to_spy": 0.0 if nav_usd > 0 else None,
        "implied_avg_beta": None,
        "top_beta_names": [],
        "additive": additive,
        "role": role,
        "attribution_mode": attribution_mode,
    }


def _factor_bucket_row_from_name_nets(
    bucket: str,
    name_nets: dict[str, dict[str, Any]],
    nav_usd: float,
    *,
    beta_results: dict[str, Any] | None,
    additive: bool,
    role: str,
    attribution_mode: str,
) -> dict[str, Any]:
    """Build one factor-by-bucket row from per-underlying net/gross maps."""
    if not name_nets:
        return _empty_factor_bucket_row(
            bucket,
            nav_usd,
            additive=additive,
            role=role,
            attribution_mode=attribution_mode,
        )

    name_rows: list[dict[str, Any]] = []
    total_net = 0.0
    total_gross = 0.0
    total_beta_net = 0.0
    total_beta_gross = 0.0
    total_beta_net_qqq = 0.0
    total_beta_net_iwm = 0.0
    total_beta_net_btc = 0.0
    for underlying, vals in name_nets.items():
        net = float(vals.get("net_notional_usd", 0.0) or 0.0)
        gross = float(vals.get("gross_notional_usd", 0.0) or 0.0)
        symbols = _first_nonblank(vals.get("symbols"), underlying)
        betas = _resolve_underlying_betas(underlying, beta_results=beta_results)
        beta_to_spy = betas["beta_to_spy"]
        beta_to_qqq = betas["beta_to_qqq"]
        beta_to_iwm = betas["beta_to_iwm"]
        beta_to_btc = betas["beta_to_btc"]
        beta_source = betas["beta_source"]
        beta_net = net * beta_to_spy
        beta_gross = gross * abs(beta_to_spy)
        beta_net_qqq = (net * beta_to_qqq) if beta_to_qqq is not None else 0.0
        beta_net_iwm = (net * beta_to_iwm) if beta_to_iwm is not None else 0.0
        beta_net_btc = (net * beta_to_btc) if beta_to_btc is not None else 0.0
        total_net += net
        total_gross += gross
        total_beta_net += beta_net
        total_beta_gross += beta_gross
        total_beta_net_qqq += beta_net_qqq
        total_beta_net_iwm += beta_net_iwm
        total_beta_net_btc += beta_net_btc
        if abs(beta_net) > 1e-6:
            name_rows.append(
                {
                    "underlying": underlying,
                    "symbols": symbols,
                    "net_notional_usd": net,
                    "beta_to_spy": beta_to_spy,
                    "beta_to_qqq": beta_to_qqq,
                    "beta_to_iwm": beta_to_iwm,
                    "beta_to_btc": beta_to_btc,
                    "beta_source": beta_source,
                    "beta_weighted_net_usd": beta_net,
                }
            )

    top_names = sorted(
        name_rows,
        key=lambda r: abs(r["beta_weighted_net_usd"]),
        reverse=True,
    )[:5]
    implied_avg = (total_beta_net / total_net) if abs(total_net) > 1e-6 else None
    return {
        "bucket": bucket,
        "bucket_label": BUCKET_LABELS.get(bucket, bucket),
        "n_names": len(name_nets),
        "net_notional_usd": total_net,
        "gross_notional_usd": total_gross,
        "beta_weighted_net_usd": total_beta_net,
        "beta_weighted_gross_usd": total_beta_gross,
        "beta_weighted_net_qqq_usd": total_beta_net_qqq,
        "beta_weighted_net_iwm_usd": total_beta_net_iwm,
        "beta_weighted_net_btc_usd": total_beta_net_btc,
        "net_beta_to_spy": (total_beta_net / nav_usd) if nav_usd > 0 else None,
        "net_beta_to_qqq": (total_beta_net_qqq / nav_usd) if nav_usd > 0 else None,
        "net_beta_to_iwm": (total_beta_net_iwm / nav_usd) if nav_usd > 0 else None,
        "net_beta_to_btc": (total_beta_net_btc / nav_usd) if nav_usd > 0 else None,
        "gross_beta_to_spy": (total_beta_gross / nav_usd) if nav_usd > 0 else None,
        "implied_avg_beta": implied_avg,
        "top_beta_names": top_names,
        "additive": additive,
        "role": role,
        "attribution_mode": attribution_mode,
    }


def _name_nets_from_exposure_df(df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Collapse an exposure CSV into per-underlying net/gross maps."""
    name_nets: dict[str, dict[str, Any]] = {}
    if df is None or df.empty:
        return name_nets
    for _, raw in df.iterrows():
        underlying = _first_nonblank(raw.get("underlying"), raw.get("symbol"))
        if not underlying:
            continue
        key = underlying
        net = float(raw.get("net_notional_usd", 0.0) or 0.0)
        gross = float(raw.get("gross_notional_usd", 0.0) or 0.0)
        symbols = _first_nonblank(raw.get("symbols"), raw.get("symbol"), underlying)
        slot = name_nets.get(key)
        if slot is None:
            name_nets[key] = {
                "net_notional_usd": net,
                "gross_notional_usd": gross,
                "symbols": symbols,
            }
        else:
            slot["net_notional_usd"] = float(slot["net_notional_usd"]) + net
            slot["gross_notional_usd"] = float(slot["gross_notional_usd"]) + gross
            prev = str(slot.get("symbols") or "")
            if symbols and symbols not in prev.split(", "):
                slot["symbols"] = ", ".join(
                    sorted({*prev.split(", "), symbols} - {""})
                )
    return name_nets


def _name_nets_from_detail_ratio(
    detail: pd.DataFrame,
    ratio_col: str,
    *,
    blocked_exposure_keys: set[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Ratio-scale ``bucket_exposure_detail`` legs into per-underlying nets."""
    name_nets: dict[str, dict[str, Any]] = {}
    if detail is None or detail.empty or ratio_col not in detail.columns:
        return name_nets
    blocked_upper = (
        {k.strip().upper() for k in blocked_exposure_keys}
        if blocked_exposure_keys
        else None
    )
    for _, raw in detail.iterrows():
        ratio = float(raw.get(ratio_col, 0.0) or 0.0)
        if abs(ratio) <= 1e-12:
            continue
        underlying = _first_nonblank(raw.get("underlying"), raw.get("symbol"))
        if not underlying:
            continue
        if blocked_upper and underlying.strip().upper() in blocked_upper:
            continue
        net = float(raw.get("net_notional_usd", 0.0) or 0.0) * ratio
        gross = float(raw.get("gross_notional_usd", 0.0) or 0.0) * ratio
        symbols = _first_nonblank(raw.get("symbol"), underlying)
        slot = name_nets.get(underlying)
        if slot is None:
            name_nets[underlying] = {
                "net_notional_usd": net,
                "gross_notional_usd": gross,
                "symbols": symbols,
            }
        else:
            slot["net_notional_usd"] = float(slot["net_notional_usd"]) + net
            slot["gross_notional_usd"] = float(slot["gross_notional_usd"]) + gross
            prev = str(slot.get("symbols") or "")
            if symbols and symbols not in prev.split(", "):
                slot["symbols"] = ", ".join(
                    sorted({*prev.split(", "), symbols} - {""})
                )
    return name_nets


def _load_totals_json(accounting_dir: Path) -> dict[str, Any]:
    path = accounting_dir / "totals.json"
    if not path.is_file():
        return {}
    try:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _factor_row_from_exposure_csv(
    accounting_dir: Path,
    bucket: str,
    nav_usd: float,
    *,
    beta_results: dict[str, Any] | None,
    blocked_exposure_keys: set[str] | None,
    additive: bool,
    role: str,
    attribution_mode: str,
) -> dict[str, Any]:
    path = accounting_dir / f"net_exposure_{bucket}.csv"
    df = _read_csv_or_empty(path)
    if blocked_exposure_keys and not df.empty and _filter_exposure_df is not None:
        df = _filter_exposure_df(df, blocked_exposure_keys)
    return _factor_bucket_row_from_name_nets(
        bucket,
        _name_nets_from_exposure_df(df),
        nav_usd,
        beta_results=beta_results,
        additive=additive,
        role=role,
        attribution_mode=attribution_mode,
    )


def compute_factor_by_bucket(
    accounting_dir: Path,
    nav_usd: float,
    *,
    beta_results: dict[str, Any] | None = None,
    blocked_exposure_keys: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Per-bucket beta-weighted net exposure and net beta to SPY/QQQ/IWM/BTC.

    Additive sleeves (B1/B2/B4 + unbucketed) come from ratio-scaled
    ``bucket_exposure_detail.csv`` so they partition the book. B3 and B5 are
    overlay views (informative, excluded from the additive sum).
    """
    if accounting_dir is None or not accounting_dir.is_dir():
        return []

    detail = _read_csv_or_empty(accounting_dir / "bucket_exposure_detail.csv")
    has_detail = not detail.empty and all(
        col in detail.columns for _, col in _DETAIL_RATIO_COLS
    )
    attribution_mode = "detail_ratio" if has_detail else "legacy_csv"

    bucket_rows: list[dict[str, Any]] = []

    if has_detail:
        for bucket, ratio_col in _DETAIL_RATIO_COLS:
            name_nets = _name_nets_from_detail_ratio(
                detail,
                ratio_col,
                blocked_exposure_keys=blocked_exposure_keys,
            )
            bucket_rows.append(
                _factor_bucket_row_from_name_nets(
                    bucket,
                    name_nets,
                    nav_usd,
                    beta_results=beta_results,
                    additive=True,
                    role="sleeve",
                    attribution_mode=attribution_mode,
                )
            )
        unbuck_df = _read_csv_or_empty(accounting_dir / "net_exposure_unbucketed.csv")
        if blocked_exposure_keys and not unbuck_df.empty and _filter_exposure_df is not None:
            unbuck_df = _filter_exposure_df(unbuck_df, blocked_exposure_keys)
        bucket_rows.append(
            _factor_bucket_row_from_name_nets(
                "unbucketed",
                _name_nets_from_exposure_df(unbuck_df),
                nav_usd,
                beta_results=beta_results,
                additive=True,
                role="unbucketed",
                attribution_mode=attribution_mode,
            )
        )
    else:
        # Legacy: B1/B2 sleeve CSVs; B4 ratio-split total from totals.json when present.
        for bucket in ("bucket_1", "bucket_2"):
            bucket_rows.append(
                _factor_row_from_exposure_csv(
                    accounting_dir,
                    bucket,
                    nav_usd,
                    beta_results=beta_results,
                    blocked_exposure_keys=blocked_exposure_keys,
                    additive=True,
                    role="sleeve",
                    attribution_mode=attribution_mode,
                )
            )
        totals = _load_totals_json(accounting_dir)
        b4_net = float(totals.get("net_exposure_bucket_4", 0.0) or 0.0)
        b4_gross = float(totals.get("gross_exposure_bucket_4", 0.0) or 0.0)
        if abs(b4_net) > 1e-9 or abs(b4_gross) > 1e-9:
            # Single aggregate row — no per-name split without detail.
            bucket_rows.append(
                _factor_bucket_row_from_name_nets(
                    "bucket_4",
                    {
                        "_B4_RATIO_TOTAL": {
                            "net_notional_usd": b4_net,
                            "gross_notional_usd": b4_gross,
                            "symbols": "ratio_split_total",
                        }
                    },
                    nav_usd,
                    beta_results={
                        **(beta_results or {}),
                        "_B4_RATIO_TOTAL": {
                            "provenance": "default_fallback",
                            "beta_to_spy": 1.0,
                        },
                    },
                    additive=True,
                    role="sleeve",
                    attribution_mode=attribution_mode,
                )
            )
        else:
            bucket_rows.append(
                _factor_row_from_exposure_csv(
                    accounting_dir,
                    "bucket_4",
                    nav_usd,
                    beta_results=beta_results,
                    blocked_exposure_keys=blocked_exposure_keys,
                    additive=True,
                    role="sleeve",
                    attribution_mode=attribution_mode,
                )
            )
        unbuck_df = _read_csv_or_empty(accounting_dir / "net_exposure_unbucketed.csv")
        if blocked_exposure_keys and not unbuck_df.empty and _filter_exposure_df is not None:
            unbuck_df = _filter_exposure_df(unbuck_df, blocked_exposure_keys)
        bucket_rows.append(
            _factor_bucket_row_from_name_nets(
                "unbucketed",
                _name_nets_from_exposure_df(unbuck_df),
                nav_usd,
                beta_results=beta_results,
                additive=True,
                role="unbucketed",
                attribution_mode=attribution_mode,
            )
        )

    for bucket in FACTOR_OVERLAY_BUCKETS:
        bucket_rows.append(
            _factor_row_from_exposure_csv(
                accounting_dir,
                bucket,
                nav_usd,
                beta_results=beta_results,
                blocked_exposure_keys=blocked_exposure_keys,
                additive=False,
                role="overlay",
                attribution_mode=attribution_mode,
            )
        )

    additive_beta_net = sum(
        float(r["beta_weighted_net_usd"]) for r in bucket_rows if r.get("additive")
    )
    for row in bucket_rows:
        if not row.get("additive"):
            row["pct_of_total_beta_net"] = None
        elif additive_beta_net and abs(additive_beta_net) > 1e-6:
            row["pct_of_total_beta_net"] = (
                float(row["beta_weighted_net_usd"]) / additive_beta_net
            )
        else:
            row["pct_of_total_beta_net"] = None

    return bucket_rows


def compute_factor_panel(
    underlying_exposure_csv: Path,
    nav_usd: float,
    *,
    beta_results: dict[str, Any] | None = None,
    screener_vol_map: dict[str, float] | None = None,
    screener_underlying_rows: dict[str, dict[str, Any]] | None = None,
    vendor_info_by_symbol: dict[str, dict[str, Any]] | None = None,
    blocked_exposure_keys: set[str] | None = None,
) -> dict[str, Any]:
    """Compute beta-weighted net exposure, sector groupings and top names.

    Inputs are book-level (sleeve-agnostic) so this works even when
    bucket reconciliation is broken. Beta source flags propagate so the
    UI can show coverage honestly.

    ``beta_results`` (Phase I): dict of ``{underlying_upper: DeltaResult
    dict}`` from ``risk_dashboard.beta_loader.compute_betas``. When
    supplied, computed betas override the curated map and the panel
    also carries per-name ``beta_to_ndx`` / ``beta_to_rut`` /
    ``regime_vol_pct`` needed by the slide-risk and vol-shock panels.
    """
    df = _read_csv_or_empty(underlying_exposure_csv)
    if blocked_exposure_keys and not df.empty and _filter_exposure_df is not None:
        df = _filter_exposure_df(df, blocked_exposure_keys)
    if df.empty:
        return {
            "available": False,
            "reason": f"missing {underlying_exposure_csv.name}",
            "rows": [],
            "by_sector": [],
            "top_beta_long": [],
            "top_beta_short": [],
            "top_btc_beta_long": [],
            "top_btc_beta_short": [],
            "by_bucket": [],
            "totals": {},
            "beta_provenance_counts": {},
            "sector_provenance_counts": {},
        }

    rows: list[dict[str, Any]] = []
    for _, raw in df.iterrows():
        underlying = _first_nonblank(raw.get("underlying"), raw.get("symbol"))
        symbols = _first_nonblank(raw.get("symbols"), underlying)
        net = float(raw.get("net_notional_usd", 0.0) or 0.0)
        gross = float(raw.get("gross_notional_usd", 0.0) or 0.0)
        legs = int(raw.get("n_legs", 0) or 0)
        # Economic sector: override map + screener theme + vendor/heuristic.
        # Beta stays on OLS loader (``beta_results``) — not curated map.
        u_key = (underlying or "").strip().upper()
        screener_row = (screener_underlying_rows or {}).get(u_key)
        sector_meta = resolve_sector(
            underlying,
            screener_row=screener_row,
            vendor_info=(vendor_info_by_symbol or {}).get(u_key),
            use_override=True,
        )
        sector = sector_meta["sector"]
        sector_source = sector_meta["sector_source"]
        sector_confidence: float | None = sector_meta.get("sector_confidence")
        instrument_class: str | None = None
        if screener_row:
            instrument_class = screener_row.get("instrument_class") or screener_row.get(
                "product_class"
            )
        betas = _resolve_underlying_betas(underlying, beta_results=beta_results)
        beta_to_spy = betas["beta_to_spy"]
        beta_to_qqq = betas["beta_to_qqq"]
        beta_to_iwm = betas["beta_to_iwm"]
        beta_to_btc = betas["beta_to_btc"]
        beta_source = betas["beta_source"]
        beta_to_spy_raw: float | None = None
        beta_se: float | None = None
        beta_n_obs: int | None = None
        beta_n_eff: int | None = None
        beta_r2: float | None = None
        regime_vol_pct: float | None = None
        shrinkage_applied: bool | None = None
        shrinkage_weight: float | None = None
        prior_used_spy: float | None = None
        prior_source: str | None = None
        if beta_results:
            br = beta_results.get((underlying or "").strip().upper())
            if br:
                provenance = br.get("provenance", "")
                if provenance in (
                    "computed",
                    "shrunk",
                    "curated_fallback",
                    "default_fallback",
                ):
                    beta_source = provenance
                if br.get("beta_to_spy") is not None:
                    beta_to_spy = float(br["beta_to_spy"])
                if br.get("beta_to_ndx") is not None:
                    beta_to_qqq = float(br["beta_to_ndx"])
                if br.get("beta_to_rut") is not None:
                    beta_to_iwm = float(br["beta_to_rut"])
                if br.get("beta_to_btc") is not None:
                    beta_to_btc = float(br["beta_to_btc"])
                beta_to_spy_raw = br.get("beta_to_spy_raw")
                beta_se = br.get("beta_se")
                beta_n_obs = br.get("n_obs")
                beta_n_eff = br.get("n_eff")
                beta_r2 = br.get("r2")
                regime_vol_pct = br.get("regime_vol_pct")
                shrinkage_applied = bool(br.get("shrinkage_applied"))
                shrinkage_weight = br.get("shrinkage_weight")
                prior_used_spy = br.get("prior_used_spy")
                prior_source = br.get("prior_source")
        if regime_vol_pct is None and screener_vol_map:
            regime_vol_pct = screener_vol_map.get((underlying or "").strip().upper())
        beta_net = net * beta_to_spy
        beta_gross = gross * abs(beta_to_spy)
        beta_net_qqq = (net * beta_to_qqq) if beta_to_qqq is not None else None
        beta_net_iwm = (net * beta_to_iwm) if beta_to_iwm is not None else None
        beta_net_btc = (net * beta_to_btc) if beta_to_btc is not None else None
        rows.append(
            {
                "underlying": underlying,
                "symbols": symbols,
                "net_notional_usd": net,
                "gross_notional_usd": gross,
                "n_legs": legs,
                "sector": sector,
                "sector_source": sector_source,
                "sector_confidence": sector_confidence,
                "instrument_class": instrument_class,
                "beta_to_spy": beta_to_spy,
                "beta_to_qqq": beta_to_qqq,
                "beta_to_iwm": beta_to_iwm,
                "beta_to_btc": beta_to_btc,
                "beta_to_spy_raw": beta_to_spy_raw,
                "beta_se": beta_se,
                "beta_n_obs": beta_n_obs,
                "beta_n_eff": beta_n_eff,
                "beta_r2": beta_r2,
                "regime_vol_pct": regime_vol_pct,
                "beta_source": beta_source,
                "shrinkage_applied": shrinkage_applied,
                "shrinkage_weight": shrinkage_weight,
                "prior_used_spy": prior_used_spy,
                "prior_source": prior_source,
                "beta_weighted_net_usd": beta_net,
                "beta_weighted_gross_usd": beta_gross,
                "beta_weighted_net_qqq_usd": beta_net_qqq,
                "beta_weighted_net_iwm_usd": beta_net_iwm,
                "beta_weighted_net_btc_usd": beta_net_btc,
            }
        )

    total_net = sum(r["net_notional_usd"] for r in rows)
    total_gross = sum(r["gross_notional_usd"] for r in rows)
    total_beta_net = sum(r["beta_weighted_net_usd"] for r in rows)
    total_beta_gross = sum(r["beta_weighted_gross_usd"] for r in rows)
    qqq_rows = [r for r in rows if r.get("beta_weighted_net_qqq_usd") is not None]
    iwm_rows = [r for r in rows if r.get("beta_weighted_net_iwm_usd") is not None]
    btc_rows = [r for r in rows if r.get("beta_weighted_net_btc_usd") is not None]
    total_beta_net_qqq = sum(r["beta_weighted_net_qqq_usd"] for r in qqq_rows)
    total_beta_net_iwm = sum(r["beta_weighted_net_iwm_usd"] for r in iwm_rows)
    total_beta_net_btc = sum(r["beta_weighted_net_btc_usd"] for r in btc_rows)
    known_btc_gross = sum(
        r["gross_notional_usd"] for r in btc_rows if r.get("beta_to_btc") is not None
    )
    btc_coverage = (known_btc_gross / total_gross) if total_gross > 0 else 0.0
    trusted_sources = {"curated", "computed", "shrunk", "curated_fallback"}
    known_beta_gross = sum(
        r["gross_notional_usd"] for r in rows if r["beta_source"] in trusted_sources
    )
    coverage = (known_beta_gross / total_gross) if total_gross > 0 else 0.0
    provenance_counts: dict[str, int] = {}
    sector_provenance_counts: dict[str, int] = {}
    for r in rows:
        key = r["beta_source"] or "unknown"
        provenance_counts[key] = provenance_counts.get(key, 0) + 1
        skey = r.get("sector_source") or "unknown"
        sector_provenance_counts[skey] = sector_provenance_counts.get(skey, 0) + 1

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
    if total_gross > 0:
        for s in by_sector:
            share = s["gross_notional_usd"] / total_gross
            s["pct_book_gross_raw"] = share
            s["pct_book_gross"] = share
        sector_gross_sum = sum(s["gross_notional_usd"] for s in by_sector)
        if sector_gross_sum > 0 and abs(sector_gross_sum - total_gross) > 1e-6:
            scale = total_gross / sector_gross_sum
            for s in by_sector:
                s["pct_book_gross"] = (s["gross_notional_usd"] * scale) / total_gross

    longs = sorted(
        [r for r in rows if r["beta_weighted_net_usd"] > 0],
        key=lambda r: r["beta_weighted_net_usd"],
        reverse=True,
    )[:12]
    shorts = sorted(
        [r for r in rows if r["beta_weighted_net_usd"] < 0],
        key=lambda r: r["beta_weighted_net_usd"],
    )[:12]

    btc_longs = sorted(
        [r for r in btc_rows if r["beta_weighted_net_btc_usd"] > 0],
        key=lambda r: r["beta_weighted_net_btc_usd"],
        reverse=True,
    )[:12]
    btc_shorts = sorted(
        [r for r in btc_rows if r["beta_weighted_net_btc_usd"] < 0],
        key=lambda r: r["beta_weighted_net_btc_usd"],
    )[:12]

    totals = {
        "n_underlyings": len(rows),
        "net_notional_usd": total_net,
        "gross_notional_usd": total_gross,
        "beta_weighted_net_usd": total_beta_net,
        "beta_weighted_gross_usd": total_beta_gross,
        "beta_weighted_net_qqq_usd": total_beta_net_qqq if qqq_rows else None,
        "beta_weighted_net_iwm_usd": total_beta_net_iwm if iwm_rows else None,
        "beta_weighted_net_btc_usd": total_beta_net_btc if btc_rows else None,
        "net_beta_to_spy": (total_beta_net / nav_usd) if nav_usd > 0 else None,
        "net_beta_to_qqq": (total_beta_net_qqq / nav_usd) if nav_usd > 0 and qqq_rows else None,
        "net_beta_to_iwm": (total_beta_net_iwm / nav_usd) if nav_usd > 0 and iwm_rows else None,
        "gross_beta_to_spy": (total_beta_gross / nav_usd) if nav_usd > 0 else None,
        "net_beta_to_btc": (total_beta_net_btc / nav_usd) if nav_usd > 0 and btc_rows else None,
        "beta_coverage_gross_pct": coverage,
        "btc_beta_coverage_gross_pct": btc_coverage,
        "n_btc_beta_names": len(btc_rows),
    }

    return {
        "available": True,
        "rows": rows,
        "by_sector": by_sector,
        "top_beta_long": longs,
        "top_beta_short": shorts,
        "top_btc_beta_long": btc_longs,
        "top_btc_beta_short": btc_shorts,
        "by_bucket": [],
        "totals": totals,
        "beta_provenance_counts": provenance_counts,
        "sector_provenance_counts": sector_provenance_counts,
    }


def _load_bucket_underlying_rows(
    accounting_dir: Path | None,
    *,
    blocked_exposure_keys: set[str] | None = None,
) -> list[dict[str, Any]]:
    if accounting_dir is None or not accounting_dir.is_dir():
        return []
    out: list[dict[str, Any]] = []
    for bucket in BUCKET_SLEEVE_KEYS:
        path = accounting_dir / f"net_exposure_{bucket}.csv"
        df = _read_csv_or_empty(path)
        if blocked_exposure_keys and not df.empty and _filter_exposure_df is not None:
            df = _filter_exposure_df(df, blocked_exposure_keys)
        if df.empty:
            continue
        for _, raw in df.iterrows():
            underlying = _first_nonblank(raw.get("underlying"), raw.get("symbol"))
            gross = float(raw.get("gross_notional_usd", 0.0) or 0.0)
            if gross <= 0:
                continue
            out.append(
                {
                    "bucket": bucket,
                    "underlying": underlying,
                    "symbols": _first_nonblank(raw.get("symbols"), underlying),
                    "gross_notional_usd": gross,
                    "net_notional_usd": float(raw.get("net_notional_usd", 0.0) or 0.0),
                    "n_legs": int(raw.get("n_legs", 0) or 0),
                }
            )
    return out


def compute_concentration_panel(
    factor_panel: dict[str, Any],
    nav_usd: float,
    *,
    limits: dict[str, dict[str, Any]] | None = None,
    limits_ctx: RiskLimitsContext | None = None,
    accounting_dir: Path | None = None,
    blocked_exposure_keys: set[str] | None = None,
) -> dict[str, Any]:
    """Concentration metrics: HHI, top-N, per-bucket single-name & sector caps."""
    ctx = limits_ctx or load_risk_limits()
    limits = limits or ctx.limits
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

    sector_lookup = {r["underlying"]: r.get("sector") for r in rows}
    bucket_rows = _load_bucket_underlying_rows(
        accounting_dir, blocked_exposure_keys=blocked_exposure_keys
    )
    if not bucket_rows:
        bucket_rows = [
            {
                "bucket": "bucket_1",
                "underlying": r["underlying"],
                "symbols": r.get("symbols"),
                "gross_notional_usd": r["gross_notional_usd"],
                "net_notional_usd": r["net_notional_usd"],
                "n_legs": r.get("n_legs", 0),
            }
            for r in rows
        ]

    top_names: list[dict[str, Any]] = []
    for br in bucket_rows:
        bkt = br["bucket"]
        underlying = br["underlying"]
        gross = br["gross_notional_usd"]
        pct_nav = gross / nav_usd
        bkt_limits = ctx.underlying_gross_frac_by_bucket.get(
            bkt, ctx.underlying_gross_frac_by_bucket["bucket_4"]
        )
        status = _classify(pct_nav, bkt_limits)
        warn_pct = bkt_limits["warn"]
        trim_to_warn_usd = max((pct_nav - warn_pct) * nav_usd, 0.0)
        top_names.append(
            {
                "underlying": underlying,
                "bucket": bkt,
                "sector": sector_lookup.get(underlying),
                "gross_notional_usd": gross,
                "net_notional_usd": br["net_notional_usd"],
                "pct_nav_gross": pct_nav,
                "pct_book_gross": gross / total_gross if total_gross > 0 else 0.0,
                "status": status,
                "limit": bkt_limits,
                "trim_to_warn_usd": trim_to_warn_usd,
            }
        )
    top_names.sort(
        key=lambda r: (0 if r["status"] == "ok" else 1, -r["trim_to_warn_usd"], -r["pct_nav_gross"])
    )

    sector_cap = limits["single_sector_gross_pct_book"]
    sector_rows = []
    for s in sectors:
        sector_gross = s["gross_notional_usd"]
        pct_nav = sector_gross / nav_usd if nav_usd > 0 else 0.0
        share = float(s.get("pct_book_gross", 0.0) or 0.0)
        if share <= 0 and total_gross > 0:
            share = sector_gross / total_gross
        status = _classify(share, sector_cap)
        sector_rows.append(
            {
                "sector": s["sector"],
                "n_names": s["n_names"],
                "gross_notional_usd": sector_gross,
                "net_notional_usd": s["net_notional_usd"],
                "pct_nav_gross": pct_nav,
                "pct_book_gross": share,
                "status": status,
                "limit": sector_cap,
            }
        )
    sector_rows.sort(key=lambda r: r["pct_book_gross"], reverse=True)

    def hhi(shares: Iterable[float]) -> float:
        return float(sum((s * 100.0) ** 2 for s in shares))

    ranked = sorted(rows, key=lambda r: r["gross_notional_usd"], reverse=True)
    book_underlying_shares = [
        r["gross_notional_usd"] / total_gross for r in ranked if total_gross > 0
    ]
    sector_shares = [r["pct_book_gross"] for r in sector_rows]
    hhi_underlying = hhi(book_underlying_shares)
    hhi_sector = hhi(sector_shares)

    top5 = sum(r["gross_notional_usd"] for r in ranked[:5]) / nav_usd
    top10 = sum(r["gross_notional_usd"] for r in ranked[:10]) / nav_usd

    breaches: list[dict[str, Any]] = []
    for r in top_names:
        if r["status"] != "ok":
            metric_key = f"single_name:{r['bucket']}:{r['underlying']}"
            breaches.append(
                _limit_row(
                    metric=metric_key,
                    value=r["pct_nav_gross"],
                    limits={metric_key: r["limit"]},
                    source=f"per-bucket cap ({r['bucket']})",
                    action=(
                        f"Trim {r['underlying']} in {r['bucket']} "
                        f"(now {r['pct_nav_gross']:.0%} of NAV vs {r['limit']['warn']:.0%} warn)."
                    ),
                )
            )
    for r in sector_rows:
        if r["status"] != "ok":
            breaches.append(
                _limit_row(
                    metric=f"sector:{r['sector']}",
                    value=r["pct_book_gross"],
                    limits={"sector:" + r["sector"]: sector_cap},
                    source="concentration cap (sector share of book gross)",
                    action=f"Diversify away from {r['sector']} ({r['pct_book_gross']:.0%} of book gross).",
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


ACTION_QUEUE_CAP: int = 5


def compute_action_queue(
    *,
    book: BookSummary,
    factor_panel: dict[str, Any],
    concentration_panel: dict[str, Any],
    slide_risk_panel: dict[str, Any],
    borrow_panel: dict[str, Any],
    nav_usd: float,
    limits: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Ranked action queue with breach categories and sleeve attribution."""
    limits = limits or DEFAULT_LIMITS
    actions: list[dict[str, Any]] = []

    def _append(action: dict[str, Any]) -> None:
        cat = _breach_category(
            str(action.get("category", "")),
            str(action.get("source", "")),
        )
        action["breach_category"] = cat
        action["breach_category_label"] = BREACH_CATEGORY_LABELS.get(cat, cat)
        actions.append(action)

    if book.gross_exposure_pct_nav > limits["gross_exposure_pct_nav"]["warn"]:
        target_pct = limits["gross_exposure_pct_nav"]["warn"]
        cut_usd = (book.gross_exposure_pct_nav - target_pct) * nav_usd
        status = "hard" if book.gross_exposure_pct_nav >= limits["gross_exposure_pct_nav"]["hard"] else "warn"
        _append(
            {
                "priority": 0 if status == "hard" else 1,
                "status": status,
                "category": "gross_exposure_pct_nav",
                "sleeve": "book",
                "title": "Reduce gross exposure",
                "detail": f"Cut ~${cut_usd:,.0f} of gross to drop from {book.gross_exposure_pct_nav:.0%} to {target_pct:.0%} of NAV.",
                "source": "book gross cap",
            }
        )

    for name in (concentration_panel or {}).get("top_names", []):
        if name["status"] == "ok":
            continue
        target_pct = (name.get("limit") or {}).get("warn", 0.0)
        trim_usd = (name["pct_nav_gross"] - target_pct) * nav_usd
        if trim_usd <= 0:
            continue
        bkt = name.get("bucket") or "book"
        _append(
            {
                "priority": 0 if name["status"] == "hard" else 1,
                "status": name["status"],
                "category": f"single_name:{bkt}:{name['underlying']}",
                "sleeve": bkt,
                "title": f"Trim {name['underlying']} ({bkt})",
                "detail": (
                    f"Reduce ~${trim_usd:,.0f} gross in {bkt} to bring "
                    f"{name['underlying']} below {target_pct:.0%} of NAV."
                ),
                "source": f"per-bucket cap ({bkt})",
            }
        )
    for sector in (concentration_panel or {}).get("by_sector", []):
        if sector["status"] == "ok":
            continue
        target_share = limits["single_sector_gross_pct_book"]["warn"]
        trim_usd = (sector["pct_book_gross"] - target_share) * (
            (concentration_panel or {}).get("totals", {}).get("total_gross_usd") or nav_usd
        )
        if trim_usd <= 0:
            continue
        _append(
            {
                "priority": 0 if sector["status"] == "hard" else 1,
                "status": sector["status"],
                "category": f"sector:{sector['sector']}",
                "sleeve": sector["sector"],
                "title": f"De-risk {sector['sector']} sector",
                "detail": (
                    f"Reduce ~${trim_usd:,.0f} book gross in {sector['sector']} "
                    f"(now {sector['pct_book_gross']:.0%} of book vs {target_share:.0%} warn)."
                ),
                "source": "concentration cap (sector share of book gross)",
            }
        )

    worst = (slide_risk_panel or {}).get("worst_shock") or {}
    factor_totals = (factor_panel or {}).get("totals") or {}
    worst_pct = worst.get("pnl_pct_nav")
    if worst_pct is None:
        worst_pct = worst.get("total_pnl_pct_nav")
    if worst_pct is not None:
        hard_limit = limits["scenario_loss_pct_nav"]["hard"]
        if worst_pct < hard_limit:
            excess_pct = hard_limit - worst_pct
            hedge_usd = excess_pct * nav_usd
            scenario_label = worst.get("scenario") or worst.get("label", "slide risk")
            shock_label = str(scenario_label)
            spx_match = worst.get("shock_pct")
            if spx_match is None:
                for marker in ("-20%", "-15%", "-10%", "-5%", "-3%"):
                    if marker in shock_label:
                        spx_match = float(marker.strip("%")) / 100.0
                        break
            hedge_hint = ""
            if spx_match is not None and spx_match != 0:
                spy_notional = abs(hedge_usd / spx_match)
                hedge_hint = (
                    f"Short ~${spy_notional:,.0f} SPY notional (1.0x) to "
                    f"absorb {-excess_pct:.1%} of NAV under {scenario_label}."
                )
            elif factor_totals.get("net_beta_to_spy") is not None:
                hedge_hint = (
                    f"Trim net beta from {factor_totals['net_beta_to_spy']:.2f}x "
                    f"toward 0; ~${abs(hedge_usd):,.0f} hedge size."
                )
            _append(
                {
                    "priority": 0,
                    "status": "hard",
                    "category": f"slide:{scenario_label}",
                    "sleeve": "book",
                    "title": f"Hedge worst slide ({scenario_label})",
                    "detail": f"Scenario {worst_pct:.1%} of NAV vs hard limit {hard_limit:.1%}. Need ${abs(hedge_usd):,.0f} of coverage.",
                    "hedge_hint": hedge_hint,
                    "source": "slide risk worst shock",
                }
            )

    for sq in (borrow_panel or {}).get("squeeze_rows", []):
        if sq.get("status") != "hard":
            continue
        util = sq.get("liquidity_utilization")
        short_qty = sq.get("short_qty")
        if util is None or short_qty is None:
            continue
        binding = sq.get("binding_cap")
        cap_qty = (
            sq.get("cap_median_vol_shares")
            if binding == "median_volume"
            else sq.get("cap_shares_out_shares")
        )
        if cap_qty is None or cap_qty <= 0:
            cap_qty = short_qty / util if util > 0 else None
        trim_qty = max(short_qty - (cap_qty or 0) * 0.8, 0) if cap_qty else 0
        bind_txt = sq.get("binding_cap_label") or binding or "liquidity cap"
        sym = str(sq.get("symbol") or "")
        over_sh = max(float(short_qty) - float(cap_qty or 0), 0.0) if cap_qty else 0.0
        ratio_out = sq.get("short_vs_shares_out_cap")
        ratio_adv = sq.get("short_vs_adv_cap")
        other_label = sq.get("other_cap_label")
        other_ratio = (
            ratio_out if binding == "median_volume" else ratio_adv
        )
        detail_parts = [
            f"{short_qty:,.0f} sh short",
            f"Binding cap: {bind_txt} ({util:.0%} utilized)",
        ]
        if other_label and other_ratio is not None:
            detail_parts.append(f"Other cap: {other_label} ({other_ratio:.0%} utilized)")
        detail_parts.append(
            f"Cut ~{trim_qty:,.0f} sh to warn band ({over_sh:,.0f} sh over binding cap)."
        )
        _append(
            {
                "priority": 0,
                "status": "hard",
                "category": f"borrow_squeeze:{sym}",
                "sleeve": sq.get("bucket") or "bucket_4",
                "title": f"Reduce short on {sym}",
                "detail": " · ".join(detail_parts),
                "source": "short qty vs shares-out / median-volume caps",
                "short_qty": short_qty,
                "binding_cap": binding,
                "binding_cap_label": bind_txt,
                "other_cap_label": other_label,
                "cap_shares_out_shares": sq.get("cap_shares_out_shares"),
                "cap_median_vol_shares": sq.get("cap_median_vol_shares"),
                "short_vs_shares_out_cap": ratio_out,
                "short_vs_adv_cap": ratio_adv,
                "liquidity_utilization": util,
                "trim_qty": trim_qty,
                "over_cap_shares": over_sh,
            }
        )

    actions.sort(key=lambda a: (a["priority"], a.get("category", "")))
    return {
        "items": actions,
        "total": len(actions),
        "cap": ACTION_QUEUE_CAP,
    }


def compute_alert_rows(
    *,
    book: BookSummary,
    slide_risk_panel: dict[str, Any],
    borrow_panel: dict[str, Any],
    concentration_panel: dict[str, Any] | None = None,
    action_queue: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rows.extend(_normalize_breach(b) for b in book.breaches)
    for breach in (slide_risk_panel or {}).get("breaches", []):
        rows.append(_normalize_breach(breach))
    for breach in borrow_panel.get("breaches", []):
        rows.append(_normalize_breach(breach))
    if concentration_panel:
        for breach in concentration_panel.get("breaches", []):
            rows.append(_normalize_breach(breach))

    action_keys = {
        _action_dedupe_key(a)
        for a in (action_queue or {}).get("items", [])
    }
    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        key = _alert_dedupe_key(row)
        if key in seen:
            continue
        seen.add(key)
        if _action_dedupe_key(
            {
                "category": row.get("metric", ""),
                "title": row.get("label", ""),
                "sleeve": row.get("sleeve", ""),
            }
        ) in action_keys:
            continue
        deduped.append(row)
    order = {"hard": 0, "warn": 1, "unknown": 2, "ok": 3}
    return sorted(deduped, key=lambda r: order.get(r.get("status", "unknown"), 2))


# ---------------------------------------------------------------------------
# Slide risk: SPX / NDX / RUT shock strips (Phase 1)


SLIDE_SHOCK_PCTS: tuple[float, ...] = (
    -0.20, -0.15, -0.10, -0.05, -0.03, -0.02, -0.01,
    0.0,
    0.01, 0.02, 0.03, 0.05, 0.10, 0.15, 0.20,
)
SLIDE_HORIZONS_DAYS: tuple[int, ...] = (0, 5, 20)  # legacy; slide panel uses SLIDE_SCENARIO_HORIZONS
VIX_ABS_SHOCKS_POINTS: tuple[int, ...] = (-10, -5, 2, 5, 10, 15, 20, 25, 30)
VIX_DECAY_HORIZON: str = "12M"
VEGA_FRAC_PER_VOLPOINT_BY_PRODUCT_CLASS: dict[str, float] = {
    "income_yieldboost": -0.0025,
    "covered_call_1x": -0.0015,
    "scraped_income": -0.0020,
    "volatility_etp": +0.0150,
}
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
        df = pd.read_csv(screener_csv)
    except Exception:
        return {}
    if "ETF" not in df.columns:
        return {}
    out: dict[str, dict[str, Any]] = {}
    for _, r in df.iterrows():
        etf = _clean_str(r.get("ETF")).upper()
        if not etf:
            continue
        out[etf] = {
            "underlying": _clean_str(r.get("Underlying")).upper(),
            "beta_to_underlying": (
                float(r.get("Delta")) if pd.notna(r.get("Delta")) else None
            ),
            "product_class": _clean_str(r.get("Delta_product_class")),
            "vol_etf_annual": (
                float(r.get("vol_etf_annual"))
                if "vol_etf_annual" in df.columns and pd.notna(r.get("vol_etf_annual"))
                else None
            ),
            "vol_underlying_annual": (
                float(r.get("vol_underlying_annual"))
                if "vol_underlying_annual" in df.columns and pd.notna(r.get("vol_underlying_annual"))
                else None
            ),
            "borrow_fee_annual": (
                float(r.get("borrow_fee_annual"))
                if "borrow_fee_annual" in df.columns and pd.notna(r.get("borrow_fee_annual"))
                else None
            ),
            "income_distributions_annual": (
                float(r.get("income_distributions_annual"))
                if "income_distributions_annual" in df.columns
                and pd.notna(r.get("income_distributions_annual"))
                else None
            ),
            "expense_ratio_annual": (
                float(r.get("expense_ratio_annual"))
                if "expense_ratio_annual" in df.columns and pd.notna(r.get("expense_ratio_annual"))
                else None
            ),
        }
    return out


def _leverage_from_product_class(product_class: str, beta: float | None = None) -> float:
    """Map screener Delta_product_class -> LETF leverage factor ``k``."""
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
            "vol_underlying_annual": screener_row.get("vol_underlying_annual"),
            "borrow_fee_annual": screener_row.get("borrow_fee_annual"),
            "income_distributions_annual": screener_row.get("income_distributions_annual"),
            "expense_ratio_annual": screener_row.get("expense_ratio_annual"),
            "beta_to_underlying": beta_und,
            "is_letf": is_letf,
        }
        out.setdefault(underlying, []).append(leg)
    return out


def _slide_enriched_factor_rows(
    factor_panel: dict[str, Any],
    *,
    screener_csv: Path | None,
    flex_positions_xml: Path | None,
) -> list[dict[str, Any]]:
    rows = (factor_panel or {}).get("rows") or []
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
        vega_frac = 0.0
        vega_product_class = ""
        for leg in legs:
            cls = leg.get("product_class") or ""
            v = VEGA_FRAC_PER_VOLPOINT_BY_PRODUCT_CLASS.get(cls, 0.0)
            if abs(v) > abs(vega_frac):
                vega_frac = v
                vega_product_class = cls
        enriched.append(
            {
                "underlying": underlying,
                "symbols": symbols,
                "net_notional_usd": float(r.get("net_notional_usd", 0.0) or 0.0),
                "gross_notional_usd": float(r.get("gross_notional_usd", 0.0) or 0.0),
                "beta_to_spy": r.get("beta_to_spy"),
                "legs": legs,
                "is_letf": any(leg.get("is_letf") for leg in legs),
                "sigma": sigma,
                "vega_frac": vega_frac,
                "vega_product_class": vega_product_class,
            }
        )
    return enriched


def _slide_scenario_legs_for_row(
    row: dict[str, Any],
    *,
    etf_meta: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Flatten factor row + flex legs into scenario-model legs."""
    legs = row.get("legs") or []
    if legs:
        return legs
    underlying = (row.get("underlying") or "").upper()
    symbols = [s.strip().upper() for s in (row.get("symbols") or "").split(",") if s.strip()]
    sym = symbols[0] if symbols else underlying
    meta = etf_meta.get(sym) or {}
    product_class = meta.get("product_class") or ""
    beta_und = meta.get("beta_to_underlying")
    k = _leverage_from_product_class(product_class, beta_und)
    return [
        {
            "symbol": sym,
            "underlying": underlying,
            "net_notional_usd": float(row.get("net_notional_usd") or 0.0),
            "product_class": product_class,
            "leverage_k": float(k),
            "vol_etf_annual": meta.get("vol_etf_annual"),
            "vol_underlying_annual": meta.get("vol_underlying_annual"),
            "borrow_fee_annual": meta.get("borrow_fee_annual"),
            "income_distributions_annual": meta.get("income_distributions_annual"),
            "expense_ratio_annual": meta.get("expense_ratio_annual"),
            "beta_to_underlying": beta_und,
            "beta_to_spy": row.get("beta_to_spy"),
            "is_letf": abs(k) > 1.0001 or product_class == "letf_inverse",
        }
    ]


def _vix_shock_leg_overrides(
    legs: list[dict[str, Any]],
    *,
    underlying: str,
    underlying_sigma: float | None,
    vol_vix_pack: dict[str, Any],
    vix_shock_pts: float = 0.0,
    vix_new_pts: float | None = None,
    mode: ScenarioMode | None = None,
    corr_lift_override: float | None = None,
    borrow_lift_override: float | None = None,
    vix_path: tuple[float, ...] | None = None,
    peak_days: int | None = None,
    borrow_stress_cfg: dict[str, Any] | None = None,
) -> tuple[dict[str, float], dict[str, float]]:
    sigma_out: dict[str, float] = {}
    borrow_out: dict[str, float] = {}
    vix_current = float(vol_vix_pack.get("vix_current_pts") or 20.0)
    v_new = vix_new_pts if vix_new_pts is not None else vix_current + float(vix_shock_pts)
    borrow_lift = float(borrow_lift_override) if borrow_lift_override is not None else 1.0
    for leg in legs:
        sym = str(leg.get("symbol") or "").upper()
        if not sym:
            continue
        sigma, borrow_stressed = leg_sigma_for_vix_scenario(
            leg,
            underlying=underlying,
            underlying_sigma=underlying_sigma,
            vol_vix_pack=vol_vix_pack,
            vix_new_pts=v_new,
            vix_shock_pts=vix_shock_pts,
            mode=mode,
            corr_lift_override=corr_lift_override,
            borrow_lift=borrow_lift,
            vix_path=vix_path,
            peak_days=peak_days,
            borrow_stress_cfg=borrow_stress_cfg,
        )
        if sigma is not None:
            sigma_out[sym] = float(sigma)
        if borrow_stressed is not None:
            borrow_out[sym] = float(borrow_stressed)
    return sigma_out, borrow_out


def _sigma_overrides_for_vix_shock(
    legs: list[dict[str, Any]],
    *,
    underlying: str,
    underlying_sigma: float | None,
    vol_vix_pack: dict[str, Any],
    vix_shock_pts: float = 0.0,
    vix_new_pts: float | None = None,
    mode: ScenarioMode | None = None,
    corr_lift_override: float | None = None,
    borrow_lift_override: float | None = None,
) -> dict[str, float]:
    sigma_out, _ = _vix_shock_leg_overrides(
        legs,
        underlying=underlying,
        underlying_sigma=underlying_sigma,
        vol_vix_pack=vol_vix_pack,
        vix_shock_pts=vix_shock_pts,
        vix_new_pts=vix_new_pts,
        mode=mode,
        corr_lift_override=corr_lift_override,
        borrow_lift_override=borrow_lift_override,
    )
    return sigma_out


def _beta_spy_decomp_for_underlying(
    underlying: str,
    *,
    variance_decomp: dict[str, Any] | None,
) -> float | None:
    """SPY β from variance decomposition when trusted."""
    rows = (variance_decomp or {}).get("rows") or {}
    row = rows.get(str(underlying).upper()) or {}
    if row.get("use_variance_decomp") and row.get("beta_spy") is not None:
        return float(row["beta_spy"])
    return None


def _effective_spx_shock(
    shock_pct: float,
    horizon_key: str,
    *,
    horizon_shock_mode: str = "rms",
    shock_scale_override: float | None = None,
) -> float:
    if shock_scale_override is not None:
        return float(shock_pct) * float(shock_scale_override)
    return scale_spx_shock_for_horizon(
        shock_pct, horizon_key, mode=horizon_shock_mode  # type: ignore[arg-type]
    )


def _slide_horizon_scenario_totals(
    enriched: list[dict[str, Any]],
    *,
    etf_meta: dict[str, dict[str, Any]],
    shock_pct: float,
    horizon_key: str,
    vol_multiplier: float = 1.0,
    require_beta: bool = True,
    vol_vix_pack: dict[str, Any] | None = None,
    vix_shock_pts: float = 0.0,
    vix_new_pts: float | None = None,
    vix_scenario_mode: ScenarioMode | None = None,
    corr_lift_override: float | None = None,
    borrow_lift_override: float | None = None,
    zero_borrow: bool = False,
    spx_shock_cfg: dict[str, Any] | None = None,
    horizon_shock_mode: str = "rms",
    shock_scale_override: float | None = None,
    variance_decomp: dict[str, Any] | None = None,
    per_leg_beta: bool = True,
    vix_path: tuple[float, ...] | None = None,
    vix_peak_days: int | None = None,
    borrow_stress_cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    totals = {
        "beta_pnl_usd": 0.0,
        "decay_pnl_usd": 0.0,
        "borrow_pnl_usd": 0.0,
        "distribution_pnl_usd": 0.0,
        "total_pnl_usd": 0.0,
        "n_legs_modeled": 0,
        "n_legs_fallback": 0,
    }
    sigma_samples: list[float] = []
    stress_cfg = (spx_shock_cfg or {}).get("stress_beta") or {}
    spx_effective = _effective_spx_shock(
        shock_pct,
        horizon_key,
        horizon_shock_mode=horizon_shock_mode,
        shock_scale_override=shock_scale_override,
    )
    for e in enriched:
        if require_beta and e.get("beta_to_spy") is None:
            continue
        underlying = str(e.get("underlying") or "").upper()
        beta_decomp = _beta_spy_decomp_for_underlying(
            underlying, variance_decomp=variance_decomp
        )
        legs = _slide_scenario_legs_for_row(e, etf_meta=etf_meta)
        sigma_overrides: dict[str, float] | None = None
        borrow_overrides: dict[str, float] | None = None
        if vol_vix_pack:
            sigma_overrides, borrow_overrides = _vix_shock_leg_overrides(
                legs,
                underlying=underlying,
                underlying_sigma=e.get("sigma"),
                vol_vix_pack=vol_vix_pack,
                vix_shock_pts=vix_shock_pts,
                vix_new_pts=vix_new_pts,
                mode=vix_scenario_mode,
                corr_lift_override=corr_lift_override,
                borrow_lift_override=borrow_lift_override,
                vix_path=vix_path,
                peak_days=vix_peak_days,
                borrow_stress_cfg=borrow_stress_cfg,
            )
            if not sigma_overrides:
                sigma_overrides = None
            if zero_borrow or not borrow_overrides:
                borrow_overrides = None
        for leg in legs:
            if require_beta:
                if per_leg_beta and spx_shock_cfg is not None:
                    underlying_return = underlying_return_for_leg(
                        e,
                        leg,
                        float(shock_pct),
                        spx_effective,
                        stress_cfg=stress_cfg,
                        beta_spy_decomp=beta_decomp,
                    )
                else:
                    beta = float(e.get("beta_to_spy") or 0.0)
                    underlying_return = beta * spx_effective
            else:
                underlying_return = 0.0
            agg = aggregate_leg_scenario_pnl(
                [leg],
                underlying_return=underlying_return,
                horizon_key=horizon_key,
                vol_multiplier=vol_multiplier,
                underlying_sigma=e.get("sigma"),
                sigma_overrides=sigma_overrides,
                borrow_overrides=borrow_overrides,
                zero_borrow=zero_borrow,
            )
            for key in (
                "beta_pnl_usd",
                "decay_pnl_usd",
                "borrow_pnl_usd",
                "distribution_pnl_usd",
                "total_pnl_usd",
            ):
                totals[key] += float(agg.get(key) or 0.0)
            totals["n_legs_modeled"] += int(agg.get("n_legs_modeled") or 0)
            totals["n_legs_fallback"] += int(agg.get("n_legs_fallback") or 0)
            sym = str(leg.get("symbol") or "").upper()
            if sigma_overrides and sym in sigma_overrides:
                sigma_samples.append(sigma_overrides[sym])
                continue
            sigma, _ = resolve_sigma_annual(leg, underlying_sigma=e.get("sigma"))
            if sigma is not None:
                sigma_samples.append(sigma * vol_multiplier)
    totals["sigma_annual_median"] = (
        float(sorted(sigma_samples)[len(sigma_samples) // 2]) if sigma_samples else None
    )
    totals["horizon_key"] = horizon_key
    totals["horizon_years"] = horizon_to_years(horizon_key)
    totals["vol_multiplier"] = vol_multiplier
    totals["vix_shock_pts"] = vix_shock_pts
    totals["spx_shock_effective_pct"] = spx_effective
    totals["horizon_shock_mode"] = horizon_shock_mode
    return totals


def _compute_t0_per_leg_pnl(
    enriched: list[dict[str, Any]],
    *,
    shock_pct: float,
    etf_meta: dict[str, Any],
    spx_shock_cfg: dict[str, Any],
    variance_decomp: dict[str, Any] | None,
) -> tuple[float, list[dict[str, Any]]]:
    """Per-leg instantaneous beta P&L (no decay/borrow)."""
    stress_cfg = spx_shock_cfg.get("stress_beta") or {}
    per_name: list[dict[str, Any]] = []
    total = 0.0
    for e in enriched:
        if e.get("beta_to_spy") is None:
            continue
        beta_decomp = _beta_spy_decomp_for_underlying(
            str(e.get("underlying") or ""),
            variance_decomp=variance_decomp,
        )
        legs = _slide_scenario_legs_for_row(e, etf_meta=etf_meta)
        row_pnl = 0.0
        for leg in legs:
            notional = float(leg.get("net_notional_usd") or 0.0)
            if abs(notional) < 1e-9:
                continue
            cum = float(shock_pct) if float(shock_pct) < 0 else 0.0
            u_ret = underlying_return_for_leg(
                e,
                leg,
                float(shock_pct),
                float(shock_pct),
                stress_cfg=stress_cfg,
                beta_spy_decomp=beta_decomp,
                spx_cumulative_pct=cum,
            )
            price_ret = leg_instant_price_return(leg, u_ret)
            product = str(leg.get("product_class") or "").lower()
            if notional < 0 and product in (
                "income_yieldboost",
                "income_put_spread",
                "scraped_income",
            ):
                scale = abs(notional)
            else:
                scale = notional
            row_pnl += scale * price_ret
        total += row_pnl
        per_name.append(
            {
                "underlying": e["underlying"],
                "symbols": e["symbols"],
                "net_notional_usd": e["net_notional_usd"],
                "beta": e.get("beta_to_spy"),
                "expected_return": row_pnl / e["net_notional_usd"]
                if abs(e["net_notional_usd"]) > 1e-9
                else 0.0,
                "pnl_t0_usd": row_pnl,
            }
        )
    return total, per_name


def _build_historical_spx_scenarios(
    enriched: list[dict[str, Any]],
    *,
    etf_meta: dict[str, dict[str, Any]],
    nav_usd: float,
    spx_shock_cfg: dict[str, Any],
    variance_decomp: dict[str, Any] | None,
    horizon_key: str = "12M",
    zero_borrow: bool = False,
    borrow_stress_cfg: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Phase 5: daily path-integrated historical SPX analogs with VIX borrow stress."""
    if not spx_shock_cfg.get("path_scenarios_enabled", True):
        return []
    stress_cfg = spx_shock_cfg.get("stress_beta") or {}
    n_steps = int(spx_shock_cfg.get("path_steps") or PATH_STEPS_DEFAULT)
    path_borrow = bool(spx_shock_cfg.get("path_borrow_stress_enabled", True))
    bcfg = borrow_stress_cfg or load_borrow_stress_config()
    out: list[dict[str, Any]] = []
    for spec in historical_spx_scenario_specs(
        horizon_key=horizon_key,
        n_steps=n_steps,
        borrow_stress_cfg=bcfg,
    ):
        totals = {
            "beta_pnl_usd": 0.0,
            "decay_pnl_usd": 0.0,
            "borrow_pnl_usd": 0.0,
            "distribution_pnl_usd": 0.0,
            "total_pnl_usd": 0.0,
        }
        for e in enriched:
            if e.get("beta_to_spy") is None:
                continue
            legs = _slide_scenario_legs_for_row(e, etf_meta=etf_meta)
            beta_decomp = _beta_spy_decomp_for_underlying(
                str(e.get("underlying") or ""),
                variance_decomp=variance_decomp,
            )
            agg = aggregate_path_scenario_pnl(
                legs,
                spx_cumulative=spec.spx_cumulative,
                horizon_key=horizon_key,
                row=e,
                stress_cfg=stress_cfg,
                beta_spy_decomp=beta_decomp,
                zero_borrow=zero_borrow or not path_borrow,
                vix_path=spec.vix_path if path_borrow else None,
                borrow_lift=spec.borrow_lift,
                peak_days=spec.peak_days,
                borrow_stress_cfg=bcfg,
            )
            for k in totals:
                totals[k] += float(agg.get(k) or 0.0)
        out.append(
            {
                "scenario_key": spec.key,
                "label": spec.label,
                "horizon_key": horizon_key,
                "scenario_mode": "spx_path",
                "spx_peak_pct": spec.spx_peak_pct,
                "spx_end_pct": spec.spx_end_pct,
                "peak_days": spec.peak_days,
                "path_steps": n_steps,
                "vix_template_key": spec.vix_template_key,
                "beta_pnl_pct_nav": totals["beta_pnl_usd"] / nav_usd if nav_usd > 0 else None,
                "decay_pnl_pct_nav": totals["decay_pnl_usd"] / nav_usd if nav_usd > 0 else None,
                "borrow_pnl_pct_nav": totals["borrow_pnl_usd"] / nav_usd if nav_usd > 0 else None,
                "total_pnl_pct_nav": totals["total_pnl_usd"] / nav_usd if nav_usd > 0 else None,
                "total_pnl_usd": totals["total_pnl_usd"],
                "beta_pnl_usd": totals["beta_pnl_usd"],
                "decay_pnl_usd": totals["decay_pnl_usd"],
                "borrow_pnl_usd": totals["borrow_pnl_usd"],
            }
        )
    baseline = next((c for c in out if c.get("scenario_key") == "aug_2015"), out[0] if out else None)
    if baseline and nav_usd > 0:
        base_decay = float(baseline.get("decay_pnl_usd") or 0.0)
        for cell in out:
            cell["delta_vs_rms_12m_pct_nav"] = (
                float(cell["decay_pnl_usd"]) - base_decay
            ) / nav_usd
    return out


def _run_vix_scenario_cell(
    enriched: list[dict[str, Any]],
    *,
    etf_meta: dict[str, dict[str, Any]],
    nav_usd: float,
    vol_vix_pack: dict[str, Any],
    horizon_key: str,
    label: str,
    vix_new_pts: float,
    vix_shock_pts: float = 0.0,
    mode: ScenarioMode | None = None,
    corr_lift_override: float | None = None,
    borrow_lift_override: float | None = None,
    scenario_key: str = "",
    vix_path: tuple[float, ...] | None = None,
    vix_peak_days: int | None = None,
    borrow_stress_cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    totals = _slide_horizon_scenario_totals(
        enriched,
        etf_meta=etf_meta,
        shock_pct=0.0,
        horizon_key=horizon_key,
        require_beta=False,
        vol_vix_pack=vol_vix_pack,
        vix_shock_pts=vix_shock_pts,
        vix_new_pts=vix_new_pts,
        vix_scenario_mode=mode,
        corr_lift_override=corr_lift_override,
        borrow_lift_override=borrow_lift_override,
        zero_borrow=False,
        vix_path=vix_path,
        vix_peak_days=vix_peak_days,
        borrow_stress_cfg=borrow_stress_cfg,
    )
    return {
        "scenario_key": scenario_key,
        "label": label,
        "vix_shock_pts": vix_shock_pts,
        "vix_level_pts": vix_new_pts,
        "scenario_mode": mode or (vol_vix_pack.get("config") or {}).get("scenario_mode_default", "sustained"),
        "sigma_annual_median": totals.get("sigma_annual_median"),
        "beta_pnl_pct_nav": totals["beta_pnl_usd"] / nav_usd if nav_usd > 0 else None,
        "decay_pnl_pct_nav": totals["decay_pnl_usd"] / nav_usd if nav_usd > 0 else None,
        "borrow_pnl_pct_nav": totals["borrow_pnl_usd"] / nav_usd if nav_usd > 0 else None,
        "total_pnl_pct_nav": totals["total_pnl_usd"] / nav_usd if nav_usd > 0 else None,
        "total_pnl_usd": totals["total_pnl_usd"],
        "decay_pnl_usd": totals["decay_pnl_usd"],
        "borrow_pnl_usd": totals["borrow_pnl_usd"],
        "borrow_lift_override": borrow_lift_override,
    }


def _build_vix_decay_matrix(
    enriched: list[dict[str, Any]],
    *,
    etf_meta: dict[str, dict[str, Any]],
    nav_usd: float,
    vol_vix_pack: dict[str, Any],
    vix_shocks: tuple[int, ...],
    horizon_key: str = VIX_DECAY_HORIZON,
    borrow_stress_cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """12M expected book carry at SPX 0% under VIX-shocked forecast vol."""
    vix_pts = float(vol_vix_pack.get("vix_current_pts") or 20.0)
    cfg = vol_vix_pack.get("config") or {}
    bcfg = borrow_stress_cfg or load_borrow_stress_config()
    sustained_mode: ScenarioMode = "sustained"
    spike_mode: ScenarioMode = "spike_revert"

    def _finalize_cells(raw_cells: list[dict[str, Any]]) -> list[dict[str, Any]]:
        baseline = next(
            (c["total_pnl_usd"] for c in raw_cells if c.get("vix_shock_pts") == 0),
            None,
        )
        out: list[dict[str, Any]] = []
        for c in raw_cells:
            delta_pct = None
            if baseline is not None and nav_usd > 0:
                if c.get("vix_shock_pts") == 0:
                    delta_pct = 0.0
                else:
                    delta_pct = (float(c["total_pnl_usd"]) - float(baseline)) / nav_usd
            out.append({**c, "delta_vs_current_pct_nav": delta_pct})
        return out

    cells_sustained: list[dict[str, Any]] = []
    shock_list = (0,) + tuple(sorted({int(x) for x in vix_shocks}))
    for pts in shock_list:
        vix_new = vix_pts + float(pts)
        if pts == 0:
            label = f"Current VIX ({vix_pts:.1f})"
        else:
            label = f"VIX {pts:+d} pts"
        cells_sustained.append(
            _run_vix_scenario_cell(
                enriched,
                etf_meta=etf_meta,
                nav_usd=nav_usd,
                vol_vix_pack=vol_vix_pack,
                horizon_key=horizon_key,
                label=label,
                vix_new_pts=vix_new,
                vix_shock_pts=float(pts),
                mode=sustained_mode,
                scenario_key=f"sustained_{pts:+d}",
            )
        )
    cells_sustained = _finalize_cells(cells_sustained)

    cells_spike: list[dict[str, Any]] = []
    for pts in shock_list:
        if pts == 0:
            continue
        vix_new = vix_pts + float(pts)
        cells_spike.append(
            _run_vix_scenario_cell(
                enriched,
                etf_meta=etf_meta,
                nav_usd=nav_usd,
                vol_vix_pack=vol_vix_pack,
                horizon_key=horizon_key,
                label=f"VIX {pts:+d} pts (spike & revert)",
                vix_new_pts=vix_new,
                vix_shock_pts=float(pts),
                mode=spike_mode,
                scenario_key=f"spike_{pts:+d}",
            )
        )
    if cells_sustained:
        baseline = cells_sustained[0]
        cells_spike.insert(
            0,
            {**baseline, "label": f"Current VIX ({vix_pts:.1f})", "scenario_mode": spike_mode},
        )
    cells_spike = _finalize_cells(cells_spike)

    historical: list[dict[str, Any]] = []
    for spec in historical_scenario_specs():
        peak = float(spec.vix_peak_pts or spec.vix_new_pts)
        lift = resolve_borrow_lift(spec.key, float(spec.borrow_lift), bcfg)
        vix_path: tuple[float, ...] | None = None
        if spec.vix_start_pts is not None and spec.vix_end_pts is not None and spec.peak_days:
            vix_path = build_vix_cumulative_path(
                vix_start=float(spec.vix_start_pts),
                vix_peak=peak,
                vix_end=float(spec.vix_end_pts),
                peak_days=int(spec.peak_days),
                horizon_days=252,
                n_steps=int(bcfg.get("path_steps_per_year", 252)),
            )
        cell = _run_vix_scenario_cell(
            enriched,
            etf_meta=etf_meta,
            nav_usd=nav_usd,
            vol_vix_pack=vol_vix_pack,
            horizon_key=horizon_key,
            label=spec.label,
            vix_new_pts=peak,
            mode=spike_mode,
            corr_lift_override=spec.corr_lift,
            borrow_lift_override=lift,
            scenario_key=spec.key,
            vix_path=vix_path,
            vix_peak_days=spec.peak_days,
            borrow_stress_cfg=bcfg,
        )
        cell["vix_peak_pts"] = peak
        cell["vix_end_pts"] = spec.vix_end_pts
        cell["peak_days"] = spec.peak_days
        cell["corr_lift"] = spec.corr_lift
        cell["borrow_lift"] = lift
        if cells_sustained and nav_usd > 0:
            cell["delta_vs_current_pct_nav"] = (
                float(cell["total_pnl_usd"]) - float(cells_sustained[0]["total_pnl_usd"])
            ) / nav_usd
        historical.append(cell)

    per_name: list[dict[str, Any]] = []
    vix_plus_20 = vix_pts + 20.0
    for e in enriched:
        underlying = str(e.get("underlying") or "").upper()
        if not underlying:
            continue
        legs = _slide_scenario_legs_for_row(e, etf_meta=etf_meta)
        sigma_base = e.get("sigma")
        betas = vol_vix_pack.get("betas") or {}
        beta_res = betas.get(underlying)
        beta_vol = None
        if beta_res is not None:
            beta_vol = getattr(beta_res, "beta_vol_vix", None) or (
                beta_res.get("beta_vol_vix") if isinstance(beta_res, dict) else None
            )
        sigma_shocked = None
        if sigma_base is not None:
            sigmas = _sigma_overrides_for_vix_shock(
                legs,
                underlying=underlying,
                underlying_sigma=sigma_base,
                vol_vix_pack=vol_vix_pack,
                vix_new_pts=vix_plus_20,
                mode=sustained_mode,
            )
            if sigmas:
                sigma_shocked = float(sorted(sigmas.values())[len(sigmas) // 2])
        totals = _slide_horizon_scenario_totals(
            [e],
            etf_meta=etf_meta,
            shock_pct=0.0,
            horizon_key=horizon_key,
            require_beta=False,
            vol_vix_pack=vol_vix_pack,
            vix_new_pts=vix_plus_20,
            vix_scenario_mode=sustained_mode,
            zero_borrow=False,
        )
        per_name.append(
            {
                "underlying": underlying,
                "symbols": e.get("symbols"),
                "beta_vol_vix": beta_vol,
                "sigma_base": sigma_base,
                "sigma_shocked_plus_20": sigma_shocked,
                "decay_pnl_usd": totals["decay_pnl_usd"],
                "borrow_pnl_usd": totals["borrow_pnl_usd"],
                "total_pnl_usd": totals["total_pnl_usd"],
                "net_notional_usd": e.get("net_notional_usd"),
            }
        )
    per_name.sort(key=lambda r: abs(float(r.get("total_pnl_usd") or 0.0)), reverse=True)

    term = vol_vix_pack.get("term_structure") or {}
    current_cell = next((c for c in cells_sustained if c.get("vix_shock_pts") == 0), cells_sustained[0] if cells_sustained else None)
    headline = {
        "sigma_book_median": (current_cell or {}).get("sigma_annual_median"),
        "vix_spot_pts": vix_pts,
        "vix9d_pts": term.get("vix9d_pts"),
        "vix3m_pts": term.get("vix3m_pts"),
        "vvix_pts": term.get("vvix_pts"),
        "term_structure": term.get("term_structure"),
        "decay_12m_pct_nav": cells_sustained[0]["decay_pnl_pct_nav"] if cells_sustained else None,
        "borrow_12m_pct_nav": cells_sustained[0]["borrow_pnl_pct_nav"] if cells_sustained else None,
        "carry_12m_pct_nav": cells_sustained[0]["total_pnl_pct_nav"] if cells_sustained else None,
    }

    beta_summary = beta_summary_dict(vol_vix_pack)
    decomp = vol_vix_pack.get("variance_decomp") or {}
    est_ver = vol_vix_pack.get("estimator_version", "unknown")
    return {
        "spx_shock_pct": 0.0,
        "horizon_key": horizon_key,
        "horizon_label": f"{horizon_key} expected carry (SPX 0%)",
        "description": (
            "Expected net book carry over the next 12 months at SPX 0%, "
            "with forecast vol adjusted by vol elasticity (v3 log-log), "
            "variance decomposition, optional spike-revert path integral, "
            "and VIX-stressed borrow on short legs."
        ),
        "vix_current_pts": vix_pts,
        "vix_current": vol_vix_pack.get("vix_current"),
        "cells": cells_sustained,
        "cells_spike_revert": cells_spike,
        "historical_scenarios": historical,
        "headline": headline,
        "per_name_contributions": per_name[:50],
        "vol_vix_betas": beta_summary,
        "variance_decomp_summary": {
            "n_decomp": decomp.get("n_decomp"),
            "n_total": decomp.get("n_total"),
            "sigma_spx": decomp.get("sigma_spx"),
            "vrp_factor": decomp.get("vrp_factor"),
        },
        "n_vol_betas_computed": vol_vix_pack.get("n_computed"),
        "n_vol_betas_shrunk": vol_vix_pack.get("n_shrunk"),
        "vol_vix_estimator_version": est_ver,
        "ewma_lambda": vol_vix_pack.get("ewma_lambda"),
        "scenario_mode_default": cfg.get("scenario_mode_default", "sustained"),
        "historical_catalog": list(HISTORICAL_VIX_SCENARIOS),
    }


def _pnl_concentration_summary(
    per_name: list[dict[str, Any]],
    total_pnl_usd: float,
    *,
    pnl_key: str = "pnl_t0_usd",
    name_key: str = "underlying",
    top_n: int = 5,
) -> dict[str, Any]:
    """Rank names by |P&L| and report how much of the scenario the top N explain."""
    if not per_name:
        return {
            "top_contributors": [],
            "top_n": top_n,
            "top_n_share_of_scenario": None,
            "diversified": True,
        }
    denom = sum(abs(float(p.get(pnl_key) or 0.0)) for p in per_name)
    if denom <= 1e-9:
        return {
            "top_contributors": [],
            "top_n": top_n,
            "top_n_share_of_scenario": None,
            "diversified": True,
        }
    ranked = sorted(
        per_name,
        key=lambda p: abs(float(p.get(pnl_key) or 0.0)),
        reverse=True,
    )
    top = ranked[:top_n]
    top_sum = sum(abs(float(p.get(pnl_key) or 0.0)) for p in top)
    share = top_sum / denom
    contributors: list[dict[str, Any]] = []
    for p in top:
        pnl = float(p.get(pnl_key) or 0.0)
        contributors.append(
            {
                "underlying": p.get(name_key) or p.get("symbol"),
                "symbols": p.get("symbols"),
                "pnl_usd": pnl,
                "pct_of_scenario_abs": abs(pnl) / denom,
                "pct_of_scenario_signed": (pnl / total_pnl_usd)
                if abs(total_pnl_usd) > 1e-9
                else None,
            }
        )
    return {
        "top_contributors": contributors,
        "top_n": top_n,
        "top_n_share_of_scenario": share,
        "diversified": share < 0.70,
    }


def _update_worst_slide(
    worst_shock: dict[str, Any] | None,
    *,
    index_label: str,
    row: dict[str, Any],
    shock_pct: float | None,
    horizons_days: tuple[int, ...],
    nav_usd: float,
) -> dict[str, Any] | None:
    if shock_pct is not None and shock_pct >= 0:
        return worst_shock
    pnl_pct_t0 = row.get("pnl_pct_nav")
    cand_pnl = pnl_pct_t0
    if cand_pnl is None:
        return worst_shock
    if worst_shock is None or cand_pnl < (worst_shock.get("pnl_pct_nav") or 0.0):
        conc = row.get("concentration") or {}
        top_contributors = conc.get("top_contributors") or []
        scenario = row["label"]
        lead = top_contributors[0] if top_contributors else (row.get("top_loss") or {})
        return {
            "index": index_label,
            "scenario": scenario,
            "label": scenario,
            "shock_pct": shock_pct,
            "pnl_usd": row.get("pnl_usd"),
            "pnl_pct_nav": cand_pnl,
            "total_pnl_pct_nav": cand_pnl,
            "horizon_days": 0,
            "top_contributors": top_contributors,
            "top5_share_of_scenario": conc.get("top_n_share_of_scenario"),
            "top_contributor": {
                "underlying": lead.get("underlying"),
                "symbols": lead.get("symbols"),
                "pnl_usd": lead.get("pnl_usd"),
                "pct_of_scenario_pnl": lead.get("pct_of_scenario_abs"),
            }
            if lead
            else None,
        }
    return worst_shock


def compute_slide_risk_panel(
    *,
    factor_panel: dict[str, Any],
    nav_usd: float,
    screener_csv: Path | None = None,
    flex_positions_xml: Path | None = None,
    shocks: tuple[float, ...] = SLIDE_SHOCK_PCTS,
    scenario_horizons: tuple[str, ...] = SLIDE_SCENARIO_HORIZONS,
    vix_shocks: tuple[int, ...] = VIX_ABS_SHOCKS_POINTS,
    limits: dict[str, dict[str, Any]] | None = None,
    beta_cache_dir: Path | None = None,
    vol_vix_pack: dict[str, Any] | None = None,
    beta_results: dict[str, dict[str, Any]] | None = None,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Slide risk strips for SPX (beta-adjusted) and VIX (vega + vol-beta decay)."""
    limits = limits or DEFAULT_LIMITS
    rows = (factor_panel or {}).get("rows") or []
    if not rows or nav_usd <= 0:
        return {
            "available": False,
            "reason": "factor panel unavailable or NAV missing",
            "shocks_pct": list(shocks),
            "scenario_horizons": list(scenario_horizons),
            "horizons_days": [],
            "vix_shocks_pts": list(vix_shocks),
            "indices": [],
            "breaches": [],
        }

    etf_meta = _load_screener_etf_meta(screener_csv)
    enriched = _slide_enriched_factor_rows(
        factor_panel,
        screener_csv=screener_csv,
        flex_positions_xml=flex_positions_xml,
    )
    if vol_vix_pack is None:
        underlyings = [str(e.get("underlying") or "").upper() for e in enriched]
        vol_meta: dict[str, dict[str, str]] = {}
        for e in enriched:
            u = str(e.get("underlying") or "").upper()
            if not u:
                continue
            sec = lookup_underlying(u).get("sector") or "other"
            vol_meta[u] = {
                "product_class": str(e.get("vega_product_class") or ""),
                "sector": str(sec),
            }
        vol_cfg = load_vol_vix_config(repo_root)
        try:
            vol_vix_pack = compute_vol_vix_pack(
                underlyings,
                cache_dir=beta_cache_dir,
                underlying_meta=vol_meta,
                beta_results=beta_results,
                config=vol_cfg,
            )
        except Exception:
            vol_vix_pack = {
                "vix_current": None,
                "vix_current_pts": None,
                "betas": {},
                "n_computed": 0,
                "n_total": 0,
                "estimator_version": "error",
            }

    spx_shock_cfg = load_spx_shock_config(repo_root)
    borrow_stress_cfg = load_borrow_stress_config(repo_root)
    horizon_shock_mode = str(spx_shock_cfg.get("horizon_shock_mode") or "rms")
    variance_decomp = (vol_vix_pack or {}).get("variance_decomp")
    factor_totals = (factor_panel or {}).get("totals") or {}

    indices_out: list[dict[str, Any]] = []
    breaches: list[dict[str, Any]] = []
    worst_shock: dict[str, Any] | None = None

    # --- SPX strip (beta to SPY) ---
    coverage_gross = sum(
        abs(e["net_notional_usd"]) for e in enriched if e.get("beta_to_spy") is not None
    )
    total_gross = sum(abs(e["net_notional_usd"]) for e in enriched)
    spx_shock_rows: list[dict[str, Any]] = []
    for shock_pct in shocks:
        total_pnl_t0, per_name_pnl_t0 = _compute_t0_per_leg_pnl(
            enriched,
            shock_pct=float(shock_pct),
            etf_meta=etf_meta,
            spx_shock_cfg=spx_shock_cfg,
            variance_decomp=variance_decomp,
        )
        pnl_pct_t0 = total_pnl_t0 / nav_usd if nav_usd > 0 else None

        horizon_rows: list[dict[str, Any]] = [
            {
                "horizon_key": "T+0",
                "horizon_days": 0,
                "total_pnl_usd": total_pnl_t0,
                "total_pnl_pct_nav": pnl_pct_t0,
                "beta_pnl_usd": total_pnl_t0,
                "decay_pnl_usd": 0.0,
                "borrow_pnl_usd": 0.0,
                "distribution_pnl_usd": 0.0,
                "decay_usd": 0.0,
            }
        ]
        for horizon_key in scenario_horizons:
            totals = _slide_horizon_scenario_totals(
                enriched,
                etf_meta=etf_meta,
                shock_pct=float(shock_pct),
                horizon_key=horizon_key,
                vol_multiplier=1.0,
                spx_shock_cfg=spx_shock_cfg,
                horizon_shock_mode=horizon_shock_mode,
                variance_decomp=variance_decomp,
                per_leg_beta=True,
            )
            total = float(totals["total_pnl_usd"])
            horizon_rows.append(
                {
                    "horizon_key": horizon_key,
                    "horizon_label": f"{horizon_key} ({horizon_shock_mode})",
                    "horizon_years": totals["horizon_years"],
                    "vol_multiplier": 1.0,
                    "sigma_annual_median": totals.get("sigma_annual_median"),
                    "spx_shock_effective_pct": totals.get("spx_shock_effective_pct"),
                    "horizon_shock_mode": horizon_shock_mode,
                    "total_pnl_usd": total,
                    "total_pnl_pct_nav": (total / nav_usd) if nav_usd > 0 else None,
                    "beta_pnl_usd": totals["beta_pnl_usd"],
                    "decay_pnl_usd": totals["decay_pnl_usd"],
                    "borrow_pnl_usd": totals["borrow_pnl_usd"],
                    "distribution_pnl_usd": totals["distribution_pnl_usd"],
                    "decay_usd": totals["decay_pnl_usd"],
                    "n_legs_modeled": totals["n_legs_modeled"],
                    "n_legs_fallback": totals["n_legs_fallback"],
                }
            )
        if spx_shock_cfg.get("show_terminal_12m_row") and "12M" in scenario_horizons:
            totals_term = _slide_horizon_scenario_totals(
                enriched,
                etf_meta=etf_meta,
                shock_pct=float(shock_pct),
                horizon_key="12M",
                vol_multiplier=1.0,
                spx_shock_cfg=spx_shock_cfg,
                horizon_shock_mode="terminal",
                shock_scale_override=1.0,
                variance_decomp=variance_decomp,
                per_leg_beta=True,
            )
            total_term = float(totals_term["total_pnl_usd"])
            horizon_rows.append(
                {
                    "horizon_key": "12M-terminal",
                    "horizon_label": "12M terminal ΔSPX",
                    "horizon_years": totals_term["horizon_years"],
                    "vol_multiplier": 1.0,
                    "sigma_annual_median": totals_term.get("sigma_annual_median"),
                    "spx_shock_effective_pct": float(shock_pct),
                    "horizon_shock_mode": "terminal",
                    "total_pnl_usd": total_term,
                    "total_pnl_pct_nav": (total_term / nav_usd) if nav_usd > 0 else None,
                    "beta_pnl_usd": totals_term["beta_pnl_usd"],
                    "decay_pnl_usd": totals_term["decay_pnl_usd"],
                    "borrow_pnl_usd": totals_term["borrow_pnl_usd"],
                    "distribution_pnl_usd": totals_term["distribution_pnl_usd"],
                    "decay_usd": totals_term["decay_pnl_usd"],
                    "n_legs_modeled": totals_term["n_legs_modeled"],
                    "n_legs_fallback": totals_term["n_legs_fallback"],
                }
            )

        sorted_by_pnl = sorted(per_name_pnl_t0, key=lambda p: p["pnl_t0_usd"])
        top_loss = sorted_by_pnl[0] if sorted_by_pnl else None
        top_gain = sorted_by_pnl[-1] if sorted_by_pnl else None
        concentration = _pnl_concentration_summary(per_name_pnl_t0, total_pnl_t0, top_n=5)
        status = _slide_status(pnl_pct_t0, limits)
        row = {
            "shock_pct": shock_pct,
            "label": f"SPX {shock_pct:+.0%}",
            "pnl_usd": total_pnl_t0,
            "pnl_pct_nav": pnl_pct_t0,
            "status": status,
            "horizons": horizon_rows,
            "top_loss": top_loss,
            "top_gain": top_gain,
            "n_contributors": len(per_name_pnl_t0),
            "concentration": concentration,
        }
        spx_shock_rows.append(row)
        if status != "ok" and shock_pct < 0:
            breaches.append(
                _normalize_breach(
                    {
                        "metric": f"slide:spx_{shock_pct:+.0%}",
                        "label": row["label"],
                        "value": pnl_pct_t0,
                        "status": status,
                        "limit": limits["scenario_loss_pct_nav"],
                        "source": "slide risk — SPX shock",
                        "action": "Reduce beta or hedge index exposure.",
                    }
                )
            )
        worst_shock = _update_worst_slide(
            worst_shock,
            index_label="SPX",
            row=row,
            shock_pct=shock_pct,
            horizons_days=(0,),
            nav_usd=nav_usd,
        )

    decay_reference: dict[str, Any] = {}
    for horizon_key in scenario_horizons:
        totals = _slide_horizon_scenario_totals(
            enriched,
            etf_meta=etf_meta,
            shock_pct=0.0,
            horizon_key=horizon_key,
            vol_multiplier=1.0,
            spx_shock_cfg=spx_shock_cfg,
            horizon_shock_mode=horizon_shock_mode,
            variance_decomp=variance_decomp,
            per_leg_beta=True,
        )
        decay_reference[horizon_key] = {
            "horizon_years": totals["horizon_years"],
            "sigma_annual_median": totals.get("sigma_annual_median"),
            "beta_pnl_pct_nav": totals["beta_pnl_usd"] / nav_usd if nav_usd > 0 else None,
            "decay_pnl_pct_nav": totals["decay_pnl_usd"] / nav_usd if nav_usd > 0 else None,
            "borrow_pnl_pct_nav": totals["borrow_pnl_usd"] / nav_usd if nav_usd > 0 else None,
            "distribution_pnl_pct_nav": totals["distribution_pnl_usd"] / nav_usd
            if nav_usd > 0
            else None,
            "total_pnl_pct_nav": totals["total_pnl_usd"] / nav_usd if nav_usd > 0 else None,
            "beta_pnl_usd": totals["beta_pnl_usd"],
            "decay_pnl_usd": totals["decay_pnl_usd"],
            "borrow_pnl_usd": totals["borrow_pnl_usd"],
            "total_pnl_usd": totals["total_pnl_usd"],
        }

    binding_shock: dict[str, Any] | None = None
    down_rows = [r for r in spx_shock_rows if float(r.get("shock_pct") or 0.0) < 0]
    if down_rows:
        binding_shock = min(down_rows, key=lambda r: float(r.get("pnl_pct_nav") or 0.0))
    historical_spx = _build_historical_spx_scenarios(
        enriched,
        etf_meta=etf_meta,
        nav_usd=nav_usd,
        spx_shock_cfg=spx_shock_cfg,
        variance_decomp=variance_decomp,
        horizon_key="12M",
        zero_borrow=False,
        borrow_stress_cfg=borrow_stress_cfg,
    )
    indices_out.append(
        {
            "index": "SPX",
            "key": "spy",
            "strip_type": "equity_pct",
            "shock_rows": spx_shock_rows,
            "binding_shock": binding_shock,
            "binding_concentration": (binding_shock or {}).get("concentration"),
            "decay_reference": decay_reference,
            "coverage_pct": (coverage_gross / total_gross) if total_gross > 0 else 0.0,
            "n_names_covered": sum(1 for e in enriched if e.get("beta_to_spy") is not None),
            "n_names_total": len(enriched),
            "net_beta_to_spy": factor_totals.get("net_beta_to_spy"),
            "gross_beta_to_spy": factor_totals.get("gross_beta_to_spy"),
            "horizon_shock_mode": horizon_shock_mode,
            "spx_shock_config": spx_shock_cfg,
            "historical_spx_scenarios": historical_spx,
            "historical_spx_catalog": list(HISTORICAL_SPX_SCENARIOS),
            "description": (
                "T+0: per-leg instantaneous β×ΔSPX (linear, no decay/borrow). "
                f"1M–12M: horizon-scaled equity shock ({horizon_shock_mode}) + LETF decay/borrow. "
                "12M-terminal row uses full labeled ΔSPX. "
                "Historical rows: daily SPX paths, path-realized vol, VIX-linked borrow stress."
            ),
        }
    )

    # --- VIX strip (12M decay vs VIX shocks) ---
    vix_decay_matrix = _build_vix_decay_matrix(
        enriched,
        etf_meta=etf_meta,
        nav_usd=nav_usd,
        vol_vix_pack=vol_vix_pack,
        vix_shocks=vix_shocks,
        borrow_stress_cfg=borrow_stress_cfg,
    )
    baseline_carry = None
    if vix_decay_matrix.get("headline"):
        baseline_carry = vix_decay_matrix["headline"].get("carry_12m_pct_nav")
    carry_validation = compute_carry_validation(
        predicted_carry_pct_nav=baseline_carry,
        repo_root=repo_root,
    )
    if not carry_validation.get("tail_scenarios_trusted", True):
        vix_decay_matrix["tail_scenarios_trusted"] = False
        vix_decay_matrix["carry_validation"] = carry_validation
    else:
        vix_decay_matrix["tail_scenarios_trusted"] = True
        vix_decay_matrix["carry_validation"] = carry_validation

    indices_out.append(
        {
            "index": "VIX",
            "key": "vix",
            "strip_type": "vix_decay",
            "vix_decay_matrix": vix_decay_matrix,
            "n_vol_betas_computed": vix_decay_matrix.get("n_vol_betas_computed"),
        }
    )

    n_letf = sum(1 for e in enriched if e["is_letf"])
    has_vol = sum(1 for e in enriched if e["sigma"] is not None)
    return {
        "available": True,
        "shocks_pct": list(shocks),
        "scenario_horizons": list(scenario_horizons),
        "horizons_days": [],
        "vix_shocks_pts": list(vix_shocks),
        "indices": indices_out,
        "worst_shock": worst_shock,
        "breaches": breaches,
        "n_letf_names": n_letf,
        "n_names_with_vol": has_vol,
        "n_names_total": len(enriched),
        "model": "etf_dashboard_scenarios_v4_daily_paths",
        "spx_shock_config": spx_shock_cfg,
        "borrow_stress_config": borrow_stress_cfg,
        "carry_validation": carry_validation,
        "horizon_shock_mode": horizon_shock_mode,
    }


# ---------------------------------------------------------------------------
# VIX / vol shock sensitivity (Phase 4) — legacy panel; folded into slide risk

VOL_REGIME_MULTIPLIERS: tuple[float, ...] = (1.25, 1.5, 2.0, 3.0)


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

    Vega: per-leg by Delta_product_class, signed by net_notional direction.
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
# NAV, capital, and consolidated sleeve groups


def resolve_nav_usd(
    totals: dict[str, Any],
    *,
    cli_fallback: float,
    flex_dir: Path | None = None,
    cli_source: str = "MAGIS_NAV_USD",
) -> tuple[float, str]:
    """Pick NAV denominator: persisted totals → CLI/config fallback (strategy capital).

    Flex broker equity is not used so %-of-NAV matches ``strategy.capital_usd``.
    ``cli_source`` labels the fallback (e.g. ``config:capital_usd``).
    """
    _ = flex_dir  # kept for call-site compatibility; broker NAV is not used
    nav = totals.get("nav_usd")
    if nav is not None:
        try:
            nav_f = float(nav)
            if nav_f > 0:
                return nav_f, str(totals.get("nav_source") or "totals.json")
        except (TypeError, ValueError):
            pass
    return float(cli_fallback), cli_source


def compute_display_sleeve_groups(
    totals: dict[str, Any],
    *,
    nav_usd: float,
    sleeve_available: bool,
) -> list[dict[str, Any]]:
    """EOD-aligned B124 vs B3 display groups (PnL from totals bucket_pnl)."""
    bucket_pnl = totals.get("bucket_pnl") or {}
    groups: list[dict[str, Any]] = []
    for spec in DISPLAY_SLEEVE_GROUPS.values():
        buckets = spec["buckets"]
        gross: float | None
        net: float | None
        if sleeve_available:
            gross = sum(float(totals.get(f"gross_exposure_{b}", 0.0) or 0.0) for b in buckets)
            net = sum(float(totals.get(f"net_exposure_{b}", 0.0) or 0.0) for b in buckets)
            if spec["id"] == "b1245":
                net += float(totals.get("net_exposure_unbucketed", 0.0) or 0.0)
        else:
            gross = None
            net = None
        pnl = sum(float(bucket_pnl.get(b, 0.0) or 0.0) for b in buckets)
        groups.append(
            {
                "id": spec["id"],
                "label": spec["label"],
                "buckets": list(buckets),
                "gross_usd": gross,
                "net_usd": net,
                "pnl_usd": pnl,
                "pnl_pct_nav": (pnl / nav_usd) if nav_usd > 0 else None,
                "gross_pct_nav": (gross / nav_usd) if (nav_usd > 0 and gross is not None) else None,
                "exposure_note": spec.get("exposure_note", ""),
            }
        )
    return groups


def _safe_return_ratio(numerator: float, denominator: float) -> float | None:
    if abs(denominator) <= 1e-12:
        return None
    return numerator / denominator


def _roc_on_net_capital(pnl: float, net_cap: float, *, bucket: str | None = None) -> float | None:
    if bucket is not None and bucket in BUCKETS_WITHOUT_ROC:
        return None
    if net_cap <= 0:
        return None
    return _safe_return_ratio(pnl, net_cap)


def _load_pnl_history_for_returns(pnl_history_csv: Path | None) -> pd.DataFrame:
    if pnl_history_csv is None or not pnl_history_csv.is_file():
        return pd.DataFrame()
    try:
        hist = pd.read_csv(pnl_history_csv)
        if "date" not in hist.columns:
            return pd.DataFrame()
        hist["date"] = pd.to_datetime(hist["date"], errors="coerce")
        return hist[hist["date"] >= pd.to_datetime(PNL_HISTORY_START_DATE)].copy()
    except Exception:
        return pd.DataFrame()


def _average_bucket_capital(history: pd.DataFrame) -> dict[str, float]:
    """Time-average per-bucket capital columns (same convention as EOD email)."""
    out: dict[str, float] = {}
    for bucket in BUCKET_KEYS:
        for metric in ("net_capital", "gross_capital", "margin_req"):
            col = f"{metric}_{bucket}"
            out[col] = 0.0
            if history is None or history.empty or col not in history.columns:
                continue
            vals = pd.to_numeric(history[col], errors="coerce").dropna()
            if not vals.empty:
                out[col] = float(vals.mean())
    for group in ("stock_sleeves", "bucket_3"):
        for metric in ("net_capital", "gross_capital", "margin_req"):
            col = f"{metric}_{group}"
            out[col] = 0.0
            if history is None or history.empty or col not in history.columns:
                continue
            vals = pd.to_numeric(history[col], errors="coerce").dropna()
            if not vals.empty:
                out[col] = float(vals.mean())
    return out


def compute_bucket_return_rows(
    bucket_pnl: dict[str, Any],
    capital_avg: dict[str, float],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for bucket in BUCKET_KEYS:
        pnl = float(bucket_pnl.get(bucket, 0.0) or 0.0)
        net_cap = float(capital_avg.get(f"net_capital_{bucket}", 0.0) or 0.0)
        gross_cap = float(capital_avg.get(f"gross_capital_{bucket}", 0.0) or 0.0)
        margin = float(capital_avg.get(f"margin_req_{bucket}", 0.0) or 0.0)
        rows.append(
            {
                "id": bucket,
                "label": BUCKET_LABELS.get(bucket, bucket),
                "pnl_usd": pnl,
                "avg_net_capital_usd": net_cap,
                "avg_gross_capital_usd": gross_cap,
                "avg_margin_req_usd": margin,
                "roc_on_net_capital": _roc_on_net_capital(pnl, net_cap, bucket=bucket),
                "rog_on_gross_capital": _safe_return_ratio(pnl, gross_cap),
                "rom_on_margin_req": _safe_return_ratio(pnl, margin),
            }
        )
    return rows


def compute_capital_panel(
    totals: dict[str, Any],
    nav_usd: float,
    *,
    pnl_history_csv: Path | None = None,
) -> dict[str, Any]:
    """Per-bucket deployed capital and return metrics for the dashboard."""
    _ = nav_usd  # reserved for future %-of-NAV columns
    snap = totals.get("capital_snapshot")
    if not isinstance(snap, dict) or not snap:
        return {
            "available": False,
            "reason": "capital_snapshot missing from totals.json (run EOD after upgrade)",
            "bucket_return_rows": [],
        }

    bucket_pnl = totals.get("bucket_pnl") or {}
    capital_avg = _average_bucket_capital(_load_pnl_history_for_returns(pnl_history_csv))
    return {
        "available": True,
        "bucket_return_rows": compute_bucket_return_rows(bucket_pnl, capital_avg),
        "return_denominator_note": (
            f"ROC / ROG / ROM denominators = average per-day capital since {PNL_HISTORY_START_DATE} "
            "(same as EOD email). ROC omitted when avg net capital is not positive."
        ),
        "source": "totals.json capital_snapshot",
    }


def compute_bucket_sleeve_rows(
    sleeve_table: list[dict[str, Any]],
    capital_panel: dict[str, Any],
    totals: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Unified per-bucket sleeve row: exposure allocation + deployed capital + returns."""
    totals = totals or {}
    snap = totals.get("capital_snapshot")
    snap_dict = snap if isinstance(snap, dict) else {}
    capital_available = bool(capital_panel.get("available"))
    returns_by_bucket = {
        str(r.get("id", "")): r for r in capital_panel.get("bucket_return_rows") or []
    }
    sleeves_by_bucket = {str(r.get("bucket", "")): r for r in sleeve_table}

    rows: list[dict[str, Any]] = []
    for bucket in BUCKET_KEYS:
        sleeve = sleeves_by_bucket.get(bucket, {})
        ret = returns_by_bucket.get(bucket, {})
        if capital_available and snap_dict:
            net_cap = float(snap_dict.get(f"net_capital_{bucket}", 0.0) or 0.0)
            gross_cap = float(snap_dict.get(f"gross_capital_{bucket}", 0.0) or 0.0)
            margin = float(snap_dict.get(f"margin_req_{bucket}", 0.0) or 0.0)
        else:
            net_cap = gross_cap = margin = None

        gross_usd = sleeve.get("gross_usd")
        net_usd = sleeve.get("net_usd")
        if bucket == "bucket_4" and sleeve.get("attribution_available", True):
            pair_gross = float(totals.get("gross_exposure_bucket_4_pair", 0.0) or 0.0)
            pair_net = float(totals.get("net_exposure_bucket_4_pair", 0.0) or 0.0)
            if abs(pair_gross) > 1e-6 or abs(pair_net) > 1e-6:
                gross_usd = pair_gross
                net_usd = pair_net

        rows.append(
            {
                "bucket": bucket,
                "bucket_label": sleeve.get("bucket_label") or BUCKET_LABELS.get(bucket, bucket),
                "exposure_gross_usd": gross_usd,
                "exposure_net_usd": net_usd,
                "target_gross_usd": sleeve.get("target_gross_usd"),
                "actual_weight": sleeve.get("actual_weight"),
                "target_weight": sleeve.get("target_weight"),
                "drift_pp": sleeve.get("drift_pp"),
                "drift_status": sleeve.get("drift_status", "unknown"),
                "pnl_usd": ret.get("pnl_usd", sleeve.get("pnl_usd")),
                "net_capital_usd": net_cap,
                "gross_capital_usd": gross_cap,
                "margin_req_usd": margin,
                "avg_net_capital_usd": ret.get("avg_net_capital_usd"),
                "avg_gross_capital_usd": ret.get("avg_gross_capital_usd"),
                "avg_margin_req_usd": ret.get("avg_margin_req_usd"),
                "roc_on_net_capital": ret.get("roc_on_net_capital"),
                "rog_on_gross_capital": ret.get("rog_on_gross_capital"),
                "rom_on_margin_req": ret.get("rom_on_margin_req"),
                "attribution_available": sleeve.get("attribution_available", True),
            }
        )

    exposure_note = (
        "Exposure gross/net = delta-normalized (β × notional). "
        "Deployed capital = signed MV on etf_screened_today (screener universe). "
        "B3 is a flow overlay. B4 uses pair-view gross/net (short underlying + inverse ETF legs); "
        "ratio-split net reconciles to the book total."
    )
    return {
        "rows": rows,
        "capital_available": capital_available,
        "capital_reason": capital_panel.get("reason"),
        "return_denominator_note": capital_panel.get("return_denominator_note", ""),
        "exposure_note": exposure_note,
        "source": capital_panel.get("source", "totals.json"),
    }


# ---------------------------------------------------------------------------
# Borrow-rate shock sensitivity (short ETF sleeve)

# Relative borrow-rate shocks applied to every short ETF leg. Borrow APR is the
# single most volatile cost line for the inverse-ETF sleeves, so we show the
# incremental *annualized* carry drag (and its bite on NAV) under each shock.
BORROW_SHOCK_MULTIPLIERS: tuple[float, ...] = (1.25, 1.5, 2.0)
BORROW_SHOCK_ABS_BPS: tuple[float, ...] = (1000.0, 2500.0)  # +10pp, +25pp absolute APR


def compute_borrow_shock_panel(
    borrow_panel: dict[str, Any] | None,
    *,
    nav_usd: float,
) -> dict[str, Any]:
    """Annualized borrow-carry drag on the held short-ETF book under rate shocks.

    Uses the already-computed ``borrow_panel.short_etf_rows`` (short notional +
    current borrow APR). For each shock we recompute carry = Σ short_notional ×
    shocked_apr and report the *incremental* cost vs the current bill.
    """
    rows = list((borrow_panel or {}).get("short_etf_rows") or [])
    legs: list[dict[str, Any]] = []
    base_cost = 0.0
    base_notional = 0.0
    for r in rows:
        notion = abs(float(r.get("short_notional_usd") or 0.0))
        rate_pct = r.get("borrow_rate_pct")
        if notion <= 0 or rate_pct is None:
            continue
        rate = float(rate_pct) / 100.0
        legs.append({"symbol": r.get("symbol"), "notional": notion, "rate": rate})
        base_cost += notion * rate
        base_notional += notion

    available = bool(legs)
    scenarios: list[dict[str, Any]] = []

    def _scenario(label: str, shocked_cost: float) -> dict[str, Any]:
        incr = shocked_cost - base_cost
        return {
            "label": label,
            "annual_cost_usd": shocked_cost,
            "incremental_cost_usd": incr,
            "incremental_pct_nav": (incr / nav_usd) if nav_usd > 0 else None,
        }

    for mult in BORROW_SHOCK_MULTIPLIERS:
        shocked = sum(leg["notional"] * leg["rate"] * mult for leg in legs)
        scenarios.append(_scenario(f"x{mult:g} borrow APR", shocked))
    for bps in BORROW_SHOCK_ABS_BPS:
        add = bps / 10000.0
        shocked = sum(leg["notional"] * (leg["rate"] + add) for leg in legs)
        scenarios.append(_scenario(f"+{bps / 100:.0f}pp APR", shocked))

    return {
        "available": available,
        "reason": "" if available else "No short-ETF rows with borrow rates in borrow panel.",
        "short_notional_usd": base_notional,
        "current_annual_cost_usd": base_cost,
        "current_pct_nav": (base_cost / nav_usd) if nav_usd > 0 else None,
        "n_short_etfs": len(legs),
        "scenarios": scenarios,
        "note": (
            "Annualized carry = Σ |short notional| × borrow APR on held short ETFs. "
            "Shocks are applied to every leg's current APR; incremental = shocked − current."
        ),
    }


# ---------------------------------------------------------------------------
# Book drawdown (cumulative-PnL equity curve)


def compute_drawdown_panel(
    pnl_history_csv: Path,
    *,
    nav_usd: float,
    run_date: str | None = None,
) -> dict[str, Any]:
    """Max drawdown of the strategy cumulative-PnL curve (capital base = NAV).

    ``pnl_history.csv`` carries the restated *cumulative* (YTD) PnL per date in
    ``total_pnl``; equity = NAV + cumulative PnL. Filters to dates <= run_date so
    no future-dated rows leak into a historical snapshot.
    """
    if not pnl_history_csv.is_file():
        return {"available": False, "reason": f"missing {pnl_history_csv.name}"}
    try:
        df = pd.read_csv(pnl_history_csv, usecols=["date", "total_pnl"])
    except Exception as exc:  # pragma: no cover - defensive
        return {"available": False, "reason": f"unreadable pnl_history.csv: {exc}"}
    if df.empty:
        return {"available": False, "reason": "pnl_history.csv is empty"}
    df["date"] = df["date"].astype(str)
    if run_date:
        df = df[df["date"] <= str(run_date)]
    df = df.sort_values("date")
    if df.empty:
        return {"available": False, "reason": "no pnl_history rows on/before run_date"}

    base = float(nav_usd) if nav_usd and nav_usd > 0 else 0.0
    points: list[dict[str, Any]] = []
    running_peak = float("-inf")
    max_dd_usd = 0.0
    max_dd_pct = 0.0
    max_dd_date = None
    peak_equity = base
    for _, r in df.iterrows():
        cum = float(r["total_pnl"] or 0.0)
        equity = base + cum
        running_peak = max(running_peak, equity)
        dd = equity - running_peak  # <= 0
        dd_pct = (dd / running_peak) if running_peak > 0 else 0.0
        if dd < max_dd_usd:
            max_dd_usd = dd
            max_dd_pct = dd_pct
            max_dd_date = str(r["date"])
        peak_equity = running_peak
        points.append(
            {
                "date": str(r["date"]),
                "cum_pnl_usd": cum,
                "equity_usd": equity,
                "drawdown_usd": dd,
                "drawdown_pct": dd_pct,
            }
        )

    last = points[-1]
    cur_equity = last["equity_usd"]
    cur_dd_usd = cur_equity - peak_equity
    return {
        "available": True,
        "base_nav_usd": base,
        "current_cum_pnl_usd": last["cum_pnl_usd"],
        "current_equity_usd": cur_equity,
        "peak_equity_usd": peak_equity,
        "current_drawdown_usd": cur_dd_usd,
        "current_drawdown_pct": (cur_dd_usd / peak_equity) if peak_equity > 0 else None,
        "max_drawdown_usd": max_dd_usd,
        "max_drawdown_pct": max_dd_pct,
        "max_drawdown_date": max_dd_date,
        "n_points": len(points),
        "curve": points[-90:],
    }


# ---------------------------------------------------------------------------
# P&L attribution (daily / weekly deltas from pnl_history.csv)


PNL_PANEL_BUCKET_KEYS: list[tuple[str, str]] = [
    ("bucket_1", "pnl_bucket_1"),
    ("bucket_2", "pnl_bucket_2"),
    ("bucket_3", "pnl_bucket_3"),
    ("bucket_4", "pnl_bucket_4"),
    ("bucket_5", "pnl_bucket_5"),
    ("stock_sleeves", "pnl_stock_sleeves"),
]

PNL_PANEL_DAILY_LOOKBACK = 365
PNL_PANEL_WEEKLY_LOOKBACK = 52


def _pnl_panel_bucket_labels() -> dict[str, str]:
    labels = {k: BUCKET_LABELS.get(k, k) for k, _ in PNL_PANEL_BUCKET_KEYS}
    labels["stock_sleeves"] = "Stock sleeves (B1+B2+B4+B5 rollup)"
    return labels


PNL_RECON_BUCKET_KEYS: tuple[str, ...] = (
    "bucket_1",
    "bucket_2",
    "bucket_3",
    "bucket_4",
    "bucket_5",
)

PNL_RECON_TOL_USD = 100.0

COMPONENT_ATTRIBUTION_COLS: list[tuple[str, str]] = [
    ("long_realized_pnl", "Long realized"),
    ("long_unrealized_pnl", "Long unrealized"),
    ("short_realized_pnl", "Short realized"),
    ("short_unrealized_pnl", "Short unrealized"),
    ("borrow_fees", "Borrow fees"),
    ("short_credit_interest", "Short credit interest"),
    ("other_fees", "Other fees"),
    ("dividends", "Dividends"),
    ("withholding_tax", "Withholding tax"),
    ("pil_dividends", "PIL dividends"),
    ("bond_interest", "Bond interest"),
]


def _pnl_empty_buckets() -> dict[str, float]:
    return {k: 0.0 for k, _ in PNL_PANEL_BUCKET_KEYS}


def _pnl_sum_daily_buckets(
    daily_rows: list[dict[str, Any]],
    pred,
) -> tuple[float, dict[str, float]]:
    book = 0.0
    buckets = _pnl_empty_buckets()
    for row in daily_rows:
        if not pred(row):
            continue
        book += float(row.get("daily_usd") or 0.0)
        for key, val in (row.get("buckets") or {}).items():
            buckets[key] = buckets.get(key, 0.0) + float(val or 0.0)
    return book, buckets


def _pnl_period_payload(
    book_usd: float,
    buckets: dict[str, float],
    *,
    nav_usd: float,
) -> dict[str, Any]:
    recon_keys = PNL_RECON_BUCKET_KEYS
    bucket_sum = sum(float(buckets.get(k) or 0.0) for k in recon_keys)
    delta = book_usd - bucket_sum
    pct_book: dict[str, float | None] = {}
    for key in buckets:
        val = float(buckets.get(key) or 0.0)
        pct_book[key] = (val / book_usd) if abs(book_usd) > 1e-6 else None
    return {
        "book_usd": book_usd,
        "book_pct_nav": (book_usd / nav_usd) if nav_usd > 0 else None,
        "buckets": buckets,
        "bucket_pct_book": pct_book,
        "bucket_sum_usd": bucket_sum,
        "recon_delta_usd": delta,
        "recon_ok": abs(delta) <= PNL_RECON_TOL_USD,
    }


def _pnl_build_periods(
    daily_rows: list[dict[str, Any]],
    weekly_rows: list[dict[str, Any]],
    *,
    run_dt,
    nav_usd: float,
    ytd_usd: float,
    df: pd.DataFrame,
) -> dict[str, dict[str, Any]]:
    from datetime import datetime as _dt

    def _parse(d: str) -> _dt:
        return _dt.strptime(d, "%Y-%m-%d")

    week_start = run_dt.date().fromordinal(run_dt.toordinal() - run_dt.weekday())
    month_start = run_dt.replace(day=1).date()
    prior_week_ref = (run_dt.date().fromordinal(run_dt.toordinal() - 7)).isocalendar()
    prior_month = (run_dt.replace(day=1) - pd.Timedelta(days=1)).strftime("%Y-%m")

    today_book, today_buckets = _pnl_sum_daily_buckets(
        daily_rows, lambda r: r["date"] == daily_rows[-1]["date"]
    )
    wtd_book, wtd_buckets = _pnl_sum_daily_buckets(
        daily_rows, lambda r: _parse(r["date"]).date() >= week_start
    )
    mtd_book, mtd_buckets = _pnl_sum_daily_buckets(
        daily_rows, lambda r: _parse(r["date"]).date() >= month_start
    )
    prior_week_book, prior_week_buckets = _pnl_sum_daily_buckets(
        daily_rows,
        lambda r: _parse(r["date"]).isocalendar()[:2] == prior_week_ref[:2],
    )
    prior_month_book, prior_month_buckets = _pnl_sum_daily_buckets(
        daily_rows, lambda r: r["date"].startswith(prior_month)
    )

    ytd_buckets = _pnl_empty_buckets()
    if not df.empty:
        first = df.iloc[0]
        last = df.iloc[-1]
        for key, col in PNL_PANEL_BUCKET_KEYS:
            ytd_buckets[key] = float(last.get(col) or 0.0) - float(first.get(col) or 0.0)

    last_week = weekly_rows[-1] if weekly_rows else None
    prior_week_row = weekly_rows[-2] if len(weekly_rows) >= 2 else None

    periods: dict[str, dict[str, Any]] = {
        "today": _pnl_period_payload(today_book, today_buckets, nav_usd=nav_usd),
        "wtd": _pnl_period_payload(wtd_book, wtd_buckets, nav_usd=nav_usd),
        "mtd": _pnl_period_payload(mtd_book, mtd_buckets, nav_usd=nav_usd),
        "ytd": _pnl_period_payload(ytd_usd, ytd_buckets, nav_usd=nav_usd),
        "prior_week": _pnl_period_payload(prior_week_book, prior_week_buckets, nav_usd=nav_usd),
        "prior_month": _pnl_period_payload(prior_month_book, prior_month_buckets, nav_usd=nav_usd),
    }
    if last_week:
        periods["current_week"] = _pnl_period_payload(
            float(last_week.get("daily_usd") or 0.0),
            dict(last_week.get("buckets") or {}),
            nav_usd=nav_usd,
        )
    if prior_week_row:
        periods["last_week"] = _pnl_period_payload(
            float(prior_week_row.get("daily_usd") or 0.0),
            dict(prior_week_row.get("buckets") or {}),
            nav_usd=nav_usd,
        )
    return periods


def _pnl_rolling_stats(daily_rows: list[dict[str, Any]], *, nav_usd: float) -> dict[str, Any]:
    if not daily_rows:
        return {}
    vals = [float(r.get("daily_usd") or 0.0) for r in daily_rows]
    wins = sum(1 for v in vals if v > 0)
    losses = sum(1 for v in vals if v < 0)
    best = max(vals)
    worst = min(vals)
    mean = sum(vals) / len(vals)
    var = sum((v - mean) ** 2 for v in vals) / max(len(vals) - 1, 1)
    stdev = var ** 0.5
    return {
        "n_days": len(vals),
        "win_days": wins,
        "loss_days": losses,
        "win_rate": wins / len(vals) if vals else None,
        "best_day_usd": best,
        "worst_day_usd": worst,
        "avg_daily_usd": mean,
        "daily_stdev_usd": stdev,
        "sharpe_like": (mean / stdev) if stdev > 1e-6 else None,
        "lookback_pct_nav": (sum(vals) / nav_usd) if nav_usd > 0 else None,
    }


def compute_pnl_panel(
    pnl_history_csv: Path,
    *,
    nav_usd: float,
    run_date: str | None = None,
    daily_lookback: int = PNL_PANEL_DAILY_LOOKBACK,
    weekly_lookback: int = PNL_PANEL_WEEKLY_LOOKBACK,
) -> dict[str, Any]:
    """Day-over-day and week-over-week P&L from ``pnl_history.csv``.

    ``total_pnl`` and per-bucket columns are cumulative (YTD); daily moves are
    consecutive-row diffs. Rows are filtered to ``date <= run_date``.
    """
    if not pnl_history_csv.is_file():
        return {"available": False, "reason": f"missing {pnl_history_csv.name}"}

    usecols = ["date", "total_pnl"] + [col for _, col in PNL_PANEL_BUCKET_KEYS]
    try:
        df = pd.read_csv(pnl_history_csv, usecols=usecols)
    except Exception as exc:  # pragma: no cover - defensive
        return {"available": False, "reason": f"unreadable pnl_history.csv: {exc}"}
    if df.empty:
        return {"available": False, "reason": "pnl_history.csv is empty"}

    df["date"] = df["date"].astype(str)
    if run_date:
        df = df[df["date"] <= str(run_date)]
    df = df.sort_values("date").reset_index(drop=True)
    if df.empty:
        return {"available": False, "reason": "no pnl_history rows on/before run_date"}

    nav = float(nav_usd) if nav_usd and nav_usd > 0 else 0.0

    daily_rows: list[dict[str, Any]] = []
    for i, row in df.iterrows():
        if i == 0:
            continue
        prior = df.iloc[i - 1]
        daily_usd = float(row["total_pnl"] or 0.0) - float(prior["total_pnl"] or 0.0)
        buckets: dict[str, float] = {}
        for key, col in PNL_PANEL_BUCKET_KEYS:
            cur_b = float(row.get(col) or 0.0)
            pri_b = float(prior.get(col) or 0.0)
            buckets[key] = cur_b - pri_b
        daily_rows.append(
            {
                "date": str(row["date"]),
                "prior_date": str(prior["date"]),
                "daily_usd": daily_usd,
                "daily_pct_nav": (daily_usd / nav) if nav > 0 else None,
                "cum_usd": float(row["total_pnl"] or 0.0),
                "buckets": buckets,
            }
        )

    if not daily_rows:
        return {"available": False, "reason": "need at least two pnl_history rows"}

    last_daily = daily_rows[-1]
    last_row = df.iloc[-1]
    ytd_usd = float(last_row["total_pnl"] or 0.0)

    from datetime import datetime as _dt

    def _parse(d: str) -> _dt:
        return _dt.strptime(d, "%Y-%m-%d")

    run_dt = _parse(str(run_date or last_daily["date"]))
    week_start = run_dt.date().fromordinal(run_dt.toordinal() - run_dt.weekday())
    month_start = run_dt.replace(day=1).date()

    wtd_usd = sum(
        r["daily_usd"]
        for r in daily_rows
        if _parse(r["date"]).date() >= week_start
    )
    mtd_usd = sum(
        r["daily_usd"]
        for r in daily_rows
        if _parse(r["date"]).date() >= month_start
    )

    prior_week_ref = (run_dt.date().fromordinal(run_dt.toordinal() - 7)).isocalendar()
    prior_week_usd = sum(
        r["daily_usd"]
        for r in daily_rows
        if _parse(r["date"]).isocalendar()[:2] == prior_week_ref[:2]
    )

    prior_month = (run_dt.replace(day=1) - pd.Timedelta(days=1)).strftime("%Y-%m")
    prior_mtd_usd = sum(
        r["daily_usd"]
        for r in daily_rows
        if r["date"].startswith(prior_month)
    )

    weekly_map: dict[tuple[int, int], dict[str, Any]] = {}
    for r in daily_rows:
        iso = _parse(r["date"]).isocalendar()
        wk_key = (iso[0], iso[1])
        slot = weekly_map.setdefault(
            wk_key,
            {
                "week_label": f"{iso[0]}-W{iso[1]:02d}",
                "week_end": r["date"],
                "week_start": r["date"],
                "daily_usd": 0.0,
                "n_days": 0,
                "buckets": {k: 0.0 for k, _ in PNL_PANEL_BUCKET_KEYS},
            },
        )
        slot["daily_usd"] += r["daily_usd"]
        slot["n_days"] += 1
        slot["week_end"] = r["date"]
        if r["date"] < slot["week_start"]:
            slot["week_start"] = r["date"]
        for bk, val in (r.get("buckets") or {}).items():
            slot["buckets"][bk] = slot["buckets"].get(bk, 0.0) + float(val or 0.0)

    weekly_rows = []
    for wk_key in sorted(weekly_map.keys()):
        slot = weekly_map[wk_key]
        weekly_rows.append(
            {
                "week_label": slot["week_label"],
                "week_start": slot["week_start"],
                "week_end": slot["week_end"],
                "daily_usd": slot["daily_usd"],
                "daily_pct_nav": (slot["daily_usd"] / nav) if nav > 0 else None,
                "n_days": slot["n_days"],
                "buckets": slot["buckets"],
            }
        )

    periods = _pnl_build_periods(
        daily_rows,
        weekly_rows,
        run_dt=run_dt,
        nav_usd=nav,
        ytd_usd=ytd_usd,
        df=df,
    )
    rolling = _pnl_rolling_stats(daily_rows[-daily_lookback:], nav_usd=nav)
    periods_by_date: dict[str, dict[str, Any]] = {}
    for row in daily_rows:
        periods_by_date[str(row["date"])] = _pnl_period_payload(
            float(row.get("daily_usd") or 0.0),
            dict(row.get("buckets") or {}),
            nav_usd=nav,
        )

    return {
        "available": True,
        "source": "pnl_history.csv",
        "run_date": str(run_date or last_daily["date"]),
        "latest_date": str(last_daily["date"]),
        "available_dates": [str(r["date"]) for r in daily_rows],
        "bucket_labels": _pnl_panel_bucket_labels(),
        "recon_bucket_keys": list(PNL_RECON_BUCKET_KEYS),
        "recon_tol_usd": PNL_RECON_TOL_USD,
        "summary": {
            "daily_usd": last_daily["daily_usd"],
            "daily_pct_nav": last_daily["daily_pct_nav"],
            "prior_date": last_daily["prior_date"],
            "wtd_usd": wtd_usd,
            "wtd_pct_nav": (wtd_usd / nav) if nav > 0 else None,
            "mtd_usd": mtd_usd,
            "mtd_pct_nav": (mtd_usd / nav) if nav > 0 else None,
            "ytd_usd": ytd_usd,
            "ytd_pct_nav": (ytd_usd / nav) if nav > 0 else None,
            "prior_week_usd": prior_week_usd,
            "prior_week_pct_nav": (prior_week_usd / nav) if nav > 0 else None,
            "prior_month_usd": prior_mtd_usd,
            "prior_month_pct_nav": (prior_mtd_usd / nav) if nav > 0 else None,
        },
        "periods": periods,
        "periods_by_date": periods_by_date,
        "rolling": rolling,
        "daily_all": daily_rows,
        "daily": daily_rows[-daily_lookback:],
        "weekly": weekly_rows[-weekly_lookback:],
        "n_daily_rows": len(daily_rows),
        "n_weekly_rows": len(weekly_rows),
    }


# ---------------------------------------------------------------------------
# Hedged vs unhedged PnL lens (additional view on top of bucket accounting)

HEDGED_PNL_SERIES_LOOKBACK = 90
HEDGED_PNL_SPLIT_JSON = "hedged_pnl_split.json"
HEDGED_PNL_B4_PAIR_CSV = "hedged_pnl_b4_by_pair.csv"


def compute_hedged_pnl_panel(
    accounting_dir: Path,
    *,
    hedged_history_csv: Path,
    nav_usd: float,
    run_date: str | None = None,
    series_lookback: int = HEDGED_PNL_SERIES_LOOKBACK,
) -> dict[str, Any]:
    """Hedged vs unhedged PnL lens from ``hedged_pnl.py`` artifacts.

    Hedged = B1 + B2 + the matched slice of each B4 pair (short underlying
    offsetting the short inverse ETF up to the realized book hedge ratio);
    unhedged = B3 + B5 + the B4 slice above each pair's hedge ratio. YTD values
    are daily-accumulated in ``data/ledger/hedged_pnl_history.csv`` so hedge
    ratio drift is respected. Purely a lens: bucket PnL is unchanged and
    hedged + unhedged ties to the bucket-sum total.
    """
    split = _read_json_or_empty(accounting_dir / HEDGED_PNL_SPLIT_JSON)
    hist = _read_csv_or_empty(hedged_history_csv)
    if hist.empty and not split:
        return {
            "available": False,
            "reason": f"missing {HEDGED_PNL_SPLIT_JSON} and {hedged_history_csv.name}",
        }

    series: list[dict[str, Any]] = []
    last_row: dict[str, Any] | None = None
    if not hist.empty and {"date", "hedged_pnl", "unhedged_pnl"}.issubset(hist.columns):
        hist = hist.copy()
        hist["date"] = hist["date"].astype(str)
        if run_date:
            hist = hist[hist["date"] <= str(run_date)]
        hist = hist.sort_values("date").reset_index(drop=True)
        for _, r in hist.tail(series_lookback).iterrows():
            series.append(
                {
                    "date": str(r["date"]),
                    "hedged_pnl": _safe_num(r.get("hedged_pnl")),
                    "unhedged_pnl": _safe_num(r.get("unhedged_pnl")),
                    "hedged_daily": _safe_num(r.get("hedged_daily")),
                    "unhedged_daily": _safe_num(r.get("unhedged_daily")),
                    "total_pnl": _safe_num(r.get("total_pnl")),
                }
            )
        if not hist.empty:
            last_row = hist.iloc[-1].to_dict()

    # Headline numbers: prefer the run's split artifact; fall back to the ledger.
    if split and (not run_date or str(split.get("run_date")) == str(run_date)):
        hedged_ytd = _safe_num(split.get("hedged_pnl_ytd"))
        unhedged_ytd = _safe_num(split.get("unhedged_pnl_ytd"))
        hedged_daily = _safe_num(split.get("hedged_daily"))
        unhedged_daily = _safe_num(split.get("unhedged_daily"))
        total_ytd = _safe_num(split.get("total_pnl_ytd"))
        total_daily = _safe_num(split.get("total_daily"))
        components = dict(split.get("components") or {})
        reconciliation = dict(split.get("reconciliation") or {})
        source = HEDGED_PNL_SPLIT_JSON
    elif last_row is not None:
        hedged_ytd = _safe_num(last_row.get("hedged_pnl"))
        unhedged_ytd = _safe_num(last_row.get("unhedged_pnl"))
        hedged_daily = _safe_num(last_row.get("hedged_daily"))
        unhedged_daily = _safe_num(last_row.get("unhedged_daily"))
        total_ytd = _safe_num(last_row.get("total_pnl"))
        total_daily = (
            (hedged_daily + unhedged_daily)
            if hedged_daily is not None and unhedged_daily is not None
            else None
        )
        components = {
            k: _safe_num(last_row.get(k))
            for k in ("b12_daily", "b3_daily", "b5_daily", "b4_hedged_daily", "b4_unhedged_daily")
        }
        reconciliation = {"residual_daily": _safe_num(last_row.get("recon_residual"))}
        source = hedged_history_csv.name
    else:
        return {"available": False, "reason": "no hedged split rows on/before run_date"}

    ytd_gap = None
    if total_ytd is not None and hedged_ytd is not None and unhedged_ytd is not None:
        ytd_gap = total_ytd - (hedged_ytd + unhedged_ytd)
    reconciliation.setdefault("ytd_gap_vs_total", ytd_gap)
    reconciliation["ties_out"] = ytd_gap is not None and abs(ytd_gap) <= 1.0

    b4_pair_rows: list[dict[str, Any]] = []
    pair_df = _read_csv_or_empty(accounting_dir / HEDGED_PNL_B4_PAIR_CSV)
    if not pair_df.empty:
        for _, r in pair_df.iterrows():
            b4_pair_rows.append(
                {
                    "leg_type": _clean_str(r.get("leg_type")),
                    "symbol": _clean_str(r.get("symbol")),
                    "underlying": _clean_str(r.get("underlying")),
                    "pair": _clean_str(r.get("pair")),
                    "gross_usd": _safe_num(r.get("gross_usd")),
                    "matched_usd": _safe_num(r.get("matched_usd")),
                    "f_hedged": _safe_num(r.get("f_hedged")),
                    "f_source": _clean_str(r.get("f_source")),
                    "daily_pnl": _safe_num(r.get("daily_pnl")),
                    "hedged_daily": _safe_num(r.get("hedged_daily")),
                    "unhedged_daily": _safe_num(r.get("unhedged_daily")),
                    "ytd_pnl": _safe_num(r.get("ytd_pnl")),
                }
            )

    nav = float(nav_usd) if nav_usd and nav_usd > 0 else 0.0
    return {
        "available": True,
        "source": source,
        "run_date": str(run_date) if run_date else None,
        "hedged_ytd_usd": hedged_ytd,
        "unhedged_ytd_usd": unhedged_ytd,
        "hedged_daily_usd": hedged_daily,
        "unhedged_daily_usd": unhedged_daily,
        "total_ytd_usd": total_ytd,
        "total_daily_usd": total_daily,
        "hedged_ytd_pct_nav": (hedged_ytd / nav) if nav > 0 and hedged_ytd is not None else None,
        "unhedged_ytd_pct_nav": (unhedged_ytd / nav) if nav > 0 and unhedged_ytd is not None else None,
        "components": {k: _safe_num(v) for k, v in components.items()},
        "reconciliation": {
            k: (_safe_num(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else v)
            for k, v in reconciliation.items()
        },
        "series": series,
        "b4_pair_rows": b4_pair_rows,
        "definitions": {
            "hedged": "B1 + B2 + B4 matched (short underlying offsets short inverse ETF up to realized book hedge ratio)",
            "unhedged": "B3 + B5 + B4 unmatched (slice above each pair's hedge ratio)",
        },
    }


# ---------------------------------------------------------------------------
# Shared-underlying map (names that appear in more than one accounting bucket)


def compute_shared_underlying_panel(
    accounting_dir: Path,
    *,
    blocked_exposure_keys: set[str] | None = None,
) -> dict[str, Any]:
    """Underlyings whose exposure spans multiple buckets (shared spot lines).

    Reads ``net_exposure_<bucket>.csv`` for each bucket and groups by underlying.
    Names present in >1 bucket are flagged: their broker spot line is shared and
    nets across sleeves, which is exactly why B4 attribution slices a larger line.
    """
    by_underlying: dict[str, dict[str, Any]] = {}
    for bucket in BUCKET_KEYS:
        path = accounting_dir / f"net_exposure_{bucket}.csv"
        df = _read_csv_or_empty(path)
        if df.empty:
            continue
        if blocked_exposure_keys and _filter_exposure_df is not None:
            df = _filter_exposure_df(df, blocked_exposure_keys)
        for _, raw in df.iterrows():
            u = _clean_str(_first_nonblank(raw.get("underlying"), raw.get("symbol")))
            if not u:
                continue
            net = float(raw.get("net_notional_usd", 0.0) or 0.0)
            gross = float(raw.get("gross_notional_usd", 0.0) or 0.0)
            slot = by_underlying.setdefault(
                u, {"underlying": u, "buckets": {}, "net_usd": 0.0, "gross_usd": 0.0}
            )
            slot["buckets"][bucket] = {"net_usd": net, "gross_usd": gross}
            slot["net_usd"] += net
            slot["gross_usd"] += gross

    book_net_by_u: dict[str, float] = {}
    book_df = _read_csv_or_empty(accounting_dir / "net_exposure_by_underlying.csv")
    if not book_df.empty:
        for _, raw in book_df.iterrows():
            u = _clean_str(_first_nonblank(raw.get("underlying"), raw.get("symbol")))
            if not u:
                continue
            book_net_by_u[u] = float(raw.get("net_notional_usd", 0.0) or 0.0)

    recon_tol_usd = 1000.0

    def _bucket_net(buckets: dict[str, dict[str, float]], bucket: str) -> float | None:
        if bucket not in buckets:
            return None
        return float(buckets[bucket].get("net_usd", 0.0) or 0.0)

    shared: list[dict[str, Any]] = []
    for v in by_underlying.values():
        if len(v["buckets"]) <= 1:
            continue
        buckets = v["buckets"]
        book_net = book_net_by_u.get(v["underlying"])
        sleeve_sum = float(v["net_usd"])
        recon_diff = (sleeve_sum - book_net) if book_net is not None else None
        shared.append(
            {
                "underlying": v["underlying"],
                "buckets": sorted(buckets.keys()),
                "bucket_detail": buckets,
                "net_b1_usd": _bucket_net(buckets, "bucket_1"),
                "net_b2_usd": _bucket_net(buckets, "bucket_2"),
                "net_b4_usd": _bucket_net(buckets, "bucket_4"),
                "net_usd": sleeve_sum,
                "gross_usd": v["gross_usd"],
                "book_net_usd": book_net,
                "recon_diff_usd": recon_diff,
                "recon_ok": (
                    recon_diff is not None and abs(recon_diff) <= recon_tol_usd
                ),
            }
        )
    shared.sort(key=lambda r: abs(r["gross_usd"]), reverse=True)
    n_recon_fail = sum(1 for r in shared if r.get("book_net_usd") is not None and not r.get("recon_ok"))
    return {
        "available": True,
        "n_underlyings": len(by_underlying),
        "n_shared": len(shared),
        "n_recon_fail": n_recon_fail,
        "recon_tol_usd": recon_tol_usd,
        "rows": shared[:40],
        "note": (
            "Shared broker spot lines are split across sleeves. "
            "B1 + B2 + B4 net (per bucket CSV) should match book net on the same underlying; "
            "Δ flags attribution drift vs net_exposure_by_underlying.csv."
        ),
    }


# ---------------------------------------------------------------------------
# Top movers (largest cumulative PnL contributors by underlying)


def compute_movers_panel(
    accounting_dir: Path,
    *,
    top_n: int = 8,
) -> dict[str, Any]:
    """Largest positive/negative cumulative PnL contributors at the book level."""
    path = accounting_dir / "pnl_by_underlying.csv"
    df = _read_csv_or_empty(path)
    if df.empty or "total_pnl" not in df.columns:
        return {"available": False, "reason": f"missing/empty {path.name}"}
    rows: list[dict[str, Any]] = []
    for _, r in df.iterrows():
        u = _clean_str(_first_nonblank(r.get("underlying"), r.get("symbol")))
        if not u:
            continue
        rows.append(
            {
                "underlying": u,
                "symbols": _clean_str(r.get("symbols")),
                "total_pnl": float(r.get("total_pnl", 0.0) or 0.0),
            }
        )
    winners = sorted([r for r in rows if r["total_pnl"] > 0], key=lambda r: r["total_pnl"], reverse=True)
    losers = sorted([r for r in rows if r["total_pnl"] < 0], key=lambda r: r["total_pnl"])
    return {
        "available": True,
        "winners": winners[:top_n],
        "losers": losers[:top_n],
        "note": "Cumulative (YTD) PnL by underlying from pnl_by_underlying.csv.",
    }


def compute_bucket_drawdown_panel(
    pnl_history_csv: Path,
    *,
    nav_usd: float,
    run_date: str | None = None,
) -> dict[str, Any]:
    """Per-bucket max drawdown on cumulative bucket PnL (excludes stock_sleeves rollup)."""
    if not pnl_history_csv.is_file():
        return {"available": False, "reason": f"missing {pnl_history_csv.name}"}
    bucket_cols = [(k, c) for k, c in PNL_PANEL_BUCKET_KEYS if k in PNL_RECON_BUCKET_KEYS]
    usecols = ["date"] + [c for _, c in bucket_cols]
    try:
        df = pd.read_csv(pnl_history_csv, usecols=usecols)
    except Exception as exc:  # pragma: no cover
        return {"available": False, "reason": f"unreadable pnl_history.csv: {exc}"}
    if df.empty:
        return {"available": False, "reason": "pnl_history.csv is empty"}
    df["date"] = df["date"].astype(str)
    if run_date:
        df = df[df["date"] <= str(run_date)]
    df = df.sort_values("date")
    rows: list[dict[str, Any]] = []
    for key, col in bucket_cols:
        peak = float("-inf")
        max_dd = 0.0
        max_dd_date = None
        cur_cum = 0.0
        for _, r in df.iterrows():
            cum = float(r.get(col) or 0.0)
            cur_cum = cum
            peak = max(peak, cum)
            dd = cum - peak
            if dd < max_dd:
                max_dd = dd
                max_dd_date = str(r["date"])
        rows.append(
            {
                "bucket": key,
                "bucket_label": BUCKET_LABELS.get(key, key),
                "cum_pnl_usd": cur_cum,
                "max_drawdown_usd": max_dd,
                "max_drawdown_date": max_dd_date,
            }
        )
    rows.sort(key=lambda r: abs(r["max_drawdown_usd"]), reverse=True)
    return {
        "available": True,
        "rows": rows,
        "note": "Bucket drawdown on cumulative YTD bucket PnL (not NAV-scaled).",
    }


def _movers_from_cum_history(
    hist_df: pd.DataFrame,
    *,
    end_date: str,
    start_date: str | None,
    top_n: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    end_rows = hist_df[hist_df["date"] == end_date]
    if end_rows.empty:
        return [], []
    if start_date:
        start_rows = hist_df[hist_df["date"] == start_date]
    else:
        start_rows = pd.DataFrame()
    start_map: dict[tuple[str, str], float] = {}
    if not start_rows.empty:
        for _, r in start_rows.iterrows():
            start_map[(str(r["bucket"]), str(r["underlying"]))] = float(r["cum_pnl_usd"] or 0.0)
    rows: list[dict[str, Any]] = []
    for _, r in end_rows.iterrows():
        key = (str(r["bucket"]), str(r["underlying"]))
        end_val = float(r["cum_pnl_usd"] or 0.0)
        start_val = start_map.get(key, 0.0)
        delta = end_val - start_val
        if abs(delta) < 0.5:
            continue
        rows.append(
            {
                "underlying": str(r["underlying"]),
                "symbols": _clean_str(r.get("symbols")),
                "total_pnl": delta,
                "cum_pnl_usd": end_val,
            }
        )
    winners = sorted([r for r in rows if r["total_pnl"] > 0], key=lambda r: r["total_pnl"], reverse=True)
    losers = sorted([r for r in rows if r["total_pnl"] < 0], key=lambda r: r["total_pnl"])
    return winners[:top_n], losers[:top_n]


def _aggregate_book_daily_movers(
    by_bucket: dict[str, Any],
    *,
    top_n: int,
) -> dict[str, list[dict[str, Any]]]:
    """Sum per-underlying daily PnL across buckets (shared lines may double-count)."""
    by_u: dict[str, dict[str, Any]] = {}
    for slot in by_bucket.values():
        for r in (slot.get("winners") or []) + (slot.get("losers") or []):
            u = _clean_str(r.get("underlying"))
            if not u:
                continue
            cur = by_u.setdefault(
                u,
                {
                    "underlying": u,
                    "symbols": _clean_str(r.get("symbols")),
                    "total_pnl": 0.0,
                },
            )
            cur["total_pnl"] = float(cur["total_pnl"]) + float(r.get("total_pnl") or 0.0)
            if not cur.get("symbols") and r.get("symbols"):
                cur["symbols"] = _clean_str(r.get("symbols"))
    rows = list(by_u.values())
    winners = sorted([r for r in rows if r["total_pnl"] > 0], key=lambda r: r["total_pnl"], reverse=True)
    losers = sorted([r for r in rows if r["total_pnl"] < 0], key=lambda r: r["total_pnl"])
    return {"winners": winners[:top_n], "losers": losers[:top_n]}


def _resolve_period_dates(
    daily_rows: list[dict[str, Any]],
    run_date: str,
    period: str,
) -> tuple[str, str | None]:
    from datetime import datetime as _dt

    if not daily_rows:
        return run_date, None

    def _parse(d: str) -> _dt:
        return _dt.strptime(d, "%Y-%m-%d")

    run_dt = _parse(run_date)
    dates = [r["date"] for r in daily_rows]
    if period == "today":
        if len(dates) < 2:
            return run_date, None
        return run_date, daily_rows[-1]["prior_date"]
    if period == "wtd":
        week_start = run_dt.date().fromordinal(run_dt.toordinal() - run_dt.weekday())
        prior = None
        for d in dates:
            if _parse(d).date() < week_start:
                prior = d
        return run_date, prior
    if period == "mtd":
        month_start = run_dt.replace(day=1).date()
        prior = None
        for d in dates:
            if _parse(d).date() < month_start:
                prior = d
        return run_date, prior
    if period == "ytd":
        if not daily_rows:
            return run_date, None
        return run_date, daily_rows[0].get("prior_date") or daily_rows[0]["date"]
    if period in ("prior_week", "last_week"):
        prior_week_ref = (run_dt.date().fromordinal(run_dt.toordinal() - 7)).isocalendar()
        week_dates = [d for d in dates if _parse(d).isocalendar()[:2] == prior_week_ref[:2]]
        if not week_dates:
            return run_date, None
        start = week_dates[0]
        prior = None
        for d in dates:
            if d < start:
                prior = d
        return week_dates[-1], prior
    if period == "prior_month":
        prior_month = (run_dt.replace(day=1) - pd.Timedelta(days=1)).strftime("%Y-%m")
        month_dates = [d for d in dates if d.startswith(prior_month)]
        if not month_dates:
            return run_date, None
        start = month_dates[0]
        prior = None
        for d in dates:
            if d < start:
                prior = d
        return month_dates[-1], prior
    return run_date, None


def compute_bucket_movers_panel(
    accounting_dir: Path,
    *,
    bucket_underlying_history_csv: Path | None = None,
    pnl_history_csv: Path | None = None,
    run_date: str,
    top_n: int = 8,
    periods: tuple[str, ...] = ("today", "wtd", "mtd", "ytd", "prior_week", "prior_month"),
) -> dict[str, Any]:
    """Per-bucket winners/losers for selected periods (history) plus YTD snapshot."""
    snapshot: dict[str, dict[str, Any]] = {}
    for bucket in BUCKET_KEYS:
        pnl_csv = accounting_dir / f"pnl_{bucket}.csv"
        detail = compute_bucket_detail(
            bucket=bucket,
            pnl_csv=pnl_csv,
            net_exposure_csv=accounting_dir / f"net_exposure_{bucket}.csv",
        )
        winners = [
            {
                "underlying": _clean_str(r.get("underlying")),
                "symbols": _clean_str(r.get("symbols")),
                "total_pnl": float(r.get("total_pnl") or 0.0),
            }
            for r in (detail.get("winners") or [])
            if float(r.get("total_pnl") or 0.0) > 0
        ]
        losers = [
            {
                "underlying": _clean_str(r.get("underlying")),
                "symbols": _clean_str(r.get("symbols")),
                "total_pnl": float(r.get("total_pnl") or 0.0),
            }
            for r in (detail.get("losers") or [])
            if float(r.get("total_pnl") or 0.0) < 0
        ]
        snapshot[bucket] = {
            "winners": winners[:top_n],
            "losers": losers[:top_n],
        }

    by_period: dict[str, dict[str, Any]] = {}
    hist_available = (
        bucket_underlying_history_csv is not None
        and bucket_underlying_history_csv.is_file()
    )
    daily_rows: list[dict[str, Any]] = []
    if pnl_history_csv is not None and pnl_history_csv.is_file():
        pnl_tmp = compute_pnl_panel(
            pnl_history_csv,
            nav_usd=1.0,
            run_date=run_date,
            daily_lookback=10_000,
            weekly_lookback=520,
        )
        if pnl_tmp.get("available"):
            daily_rows = pnl_tmp.get("daily_all") or pnl_tmp.get("daily") or []

    by_date: dict[str, dict[str, Any]] = {}
    book_by_date: dict[str, dict[str, Any]] = {}
    if hist_available and daily_rows:
        hist = pd.read_csv(bucket_underlying_history_csv)
        hist["date"] = hist["date"].astype(str)
        hist = hist[hist["date"] <= str(run_date)]
        for period in periods:
            end_date, start_date = _resolve_period_dates(daily_rows, run_date, period)
            period_buckets: dict[str, Any] = {}
            for bucket in BUCKET_KEYS:
                sub = hist[hist["bucket"] == bucket]
                if sub.empty:
                    period_buckets[bucket] = {"winners": [], "losers": []}
                    continue
                w, l = _movers_from_cum_history(
                    sub, end_date=end_date, start_date=start_date, top_n=top_n
                )
                period_buckets[bucket] = {"winners": w, "losers": l}
            by_period[period] = {
                "end_date": end_date,
                "start_date": start_date,
                "by_bucket": period_buckets,
            }
        for row in daily_rows:
            d = str(row["date"])
            start_date = row.get("prior_date")
            period_buckets = {}
            for bucket in BUCKET_KEYS:
                sub = hist[hist["bucket"] == bucket]
                if sub.empty:
                    period_buckets[bucket] = {"winners": [], "losers": []}
                    continue
                w, l = _movers_from_cum_history(
                    sub, end_date=d, start_date=start_date, top_n=top_n
                )
                period_buckets[bucket] = {"winners": w, "losers": l}
            by_date[d] = {
                "end_date": d,
                "start_date": start_date,
                "by_bucket": period_buckets,
            }
            book_by_date[d] = _aggregate_book_daily_movers(period_buckets, top_n=top_n)

    return {
        "available": True,
        "snapshot_ytd": snapshot,
        "by_period": by_period,
        "by_date": by_date,
        "book_by_date": book_by_date,
        "history_available": bool(by_period or by_date),
        "available_dates": [str(r["date"]) for r in daily_rows],
        "note": (
            "Period movers from pnl_bucket_underlying_history.csv diffs; "
            "snapshot_ytd from latest pnl_<bucket>.csv."
        ),
    }


def compute_component_attribution_panel(
    attribution_history_csv: Path,
    *,
    nav_usd: float,
    run_date: str | None = None,
    periods: tuple[str, ...] = ("today", "wtd", "mtd", "ytd", "prior_week", "prior_month"),
) -> dict[str, Any]:
    """Book-level component PnL deltas from pnl_attribution_history.csv."""
    if not attribution_history_csv.is_file():
        return {"available": False, "reason": f"missing {attribution_history_csv.name}"}
    try:
        df = pd.read_csv(attribution_history_csv)
    except Exception as exc:  # pragma: no cover
        return {"available": False, "reason": f"unreadable attribution history: {exc}"}
    if df.empty or "date" not in df.columns:
        return {"available": False, "reason": "attribution history empty"}
    df["date"] = df["date"].astype(str)
    if run_date:
        df = df[df["date"] <= str(run_date)]
    df = df.sort_values("date").reset_index(drop=True)
    if len(df) < 2:
        return {"available": False, "reason": "need at least two attribution rows"}

    daily_rows: list[dict[str, Any]] = []
    for i, row in df.iterrows():
        if i == 0:
            continue
        prior = df.iloc[i - 1]
        slot: dict[str, Any] = {"date": str(row["date"]), "prior_date": str(prior["date"])}
        for col, _label in COMPONENT_ATTRIBUTION_COLS:
            slot[col] = float(row.get(col) or 0.0) - float(prior.get(col) or 0.0)
        slot["strategy_total_pnl"] = float(row.get("strategy_total_pnl") or 0.0) - float(
            prior.get("strategy_total_pnl") or 0.0
        )
        daily_rows.append(slot)

    nav = float(nav_usd) if nav_usd and nav_usd > 0 else 0.0
    run_dt_str = str(run_date or daily_rows[-1]["date"])
    from datetime import datetime as _dt

    run_dt = _dt.strptime(run_dt_str, "%Y-%m-%d")
    week_start = run_dt.date().fromordinal(run_dt.toordinal() - run_dt.weekday())
    month_start = run_dt.replace(day=1).date()
    prior_week_ref = (run_dt.date().fromordinal(run_dt.toordinal() - 7)).isocalendar()
    prior_month = (run_dt.replace(day=1) - pd.Timedelta(days=1)).strftime("%Y-%m")

    def _sum_period(pred) -> dict[str, float]:
        out = {col: 0.0 for col, _ in COMPONENT_ATTRIBUTION_COLS}
        out["strategy_total_pnl"] = 0.0
        for row in daily_rows:
            if not pred(row):
                continue
            for col, _ in COMPONENT_ATTRIBUTION_COLS:
                out[col] += float(row.get(col) or 0.0)
            out["strategy_total_pnl"] += float(row.get("strategy_total_pnl") or 0.0)
        return out

    def _payload(totals: dict[str, float]) -> dict[str, Any]:
        components = []
        for col, label in COMPONENT_ATTRIBUTION_COLS:
            val = float(totals.get(col) or 0.0)
            if abs(val) < 0.5:
                continue
            components.append(
                {
                    "key": col,
                    "label": label,
                    "usd": val,
                    "pct_book": (val / totals["strategy_total_pnl"])
                    if abs(totals["strategy_total_pnl"]) > 1e-6
                    else None,
                    "pct_nav": (val / nav) if nav > 0 else None,
                }
            )
        components.sort(key=lambda r: abs(r["usd"]), reverse=True)
        return {
            "total_usd": totals["strategy_total_pnl"],
            "total_pct_nav": (totals["strategy_total_pnl"] / nav) if nav > 0 else None,
            "components": components,
        }

    period_map = {
        "today": lambda r: r["date"] == daily_rows[-1]["date"],
        "wtd": lambda r: _dt.strptime(r["date"], "%Y-%m-%d").date() >= week_start,
        "mtd": lambda r: _dt.strptime(r["date"], "%Y-%m-%d").date() >= month_start,
        "ytd": lambda r: True,
        "prior_week": lambda r: _dt.strptime(r["date"], "%Y-%m-%d").isocalendar()[:2]
        == prior_week_ref[:2],
        "prior_month": lambda r: r["date"].startswith(prior_month),
    }
    out_periods: dict[str, Any] = {}
    for key in periods:
        pred = period_map.get(key)
        if pred is None:
            continue
        out_periods[key] = _payload(_sum_period(pred))

    by_date: dict[str, Any] = {}
    for row in daily_rows:
        totals = {col: float(row.get(col) or 0.0) for col, _ in COMPONENT_ATTRIBUTION_COLS}
        totals["strategy_total_pnl"] = float(row.get("strategy_total_pnl") or 0.0)
        by_date[str(row["date"])] = _payload(totals)

    return {
        "available": True,
        "source": "pnl_attribution_history.csv",
        "periods": out_periods,
        "by_date": by_date,
        "available_dates": [str(r["date"]) for r in daily_rows],
        "component_labels": {col: label for col, label in COMPONENT_ATTRIBUTION_COLS},
    }


DIVIDEND_CATEGORY_KEYS: tuple[str, ...] = ("dividends", "withholding_tax", "pil_dividends")
DIVIDEND_CATEGORY_LABELS: dict[str, str] = {
    "dividends": "Dividends",
    "withholding_tax": "Withholding tax",
    "pil_dividends": "PIL dividends",
}


def _dividend_daily_from_attribution(attribution_csv: Path, run_date: str | None) -> dict[str, dict[str, float]]:
    """Daily dividend-component deltas keyed by ISO date."""
    out: dict[str, dict[str, float]] = {}
    if not attribution_csv.is_file():
        return out
    try:
        df = pd.read_csv(attribution_csv)
    except Exception:
        return out
    if df.empty or "date" not in df.columns:
        return out
    df["date"] = df["date"].astype(str)
    if run_date:
        df = df[df["date"] <= str(run_date)]
    df = df.sort_values("date").reset_index(drop=True)
    for i in range(1, len(df)):
        row = df.iloc[i]
        prior = df.iloc[i - 1]
        slot = {}
        for col in DIVIDEND_CATEGORY_KEYS:
            slot[col] = float(row.get(col) or 0.0) - float(prior.get(col) or 0.0)
        slot["net_usd"] = sum(slot.values())
        out[str(row["date"])] = slot
    return out


def _attribution_lags_recent_cash(
    att: dict[str, float],
    cash_pil_by_date: dict[str, float],
    cash_net_by_date: dict[str, float],
    date: str,
    *,
    lookback_days: int = 4,
    session_lookback_days: int = 2,
    material_cash_pil_usd: float = 500.0,
) -> bool:
    """True when an attribution delta likely repeats Flex cash booked recently."""
    att_pil = float(att.get("pil_dividends") or 0.0)
    att_net = float(att.get("net_usd") or 0.0)
    if abs(att_pil) < 0.5 and abs(att_net) < 0.5:
        return False
    dt = pd.Timestamp(date)
    ref = max(abs(att_pil), abs(att_net), 1.0)
    tol = max(150.0, 0.02 * ref)
    for offset in range(1, lookback_days + 1):
        prior = (dt - pd.Timedelta(days=offset)).strftime("%Y-%m-%d")
        cash_pil = float(cash_pil_by_date.get(prior) or 0.0)
        cash_net = float(cash_net_by_date.get(prior) or 0.0)
        if abs(cash_pil) < 0.5 and abs(cash_net) < 0.5:
            continue
        if abs(att_pil - cash_pil) <= tol or abs(att_net - cash_net) <= tol:
            return True
        if att_pil * cash_pil > 0 and abs(abs(att_pil) - abs(cash_pil)) <= tol:
            return True
        if att_net * cash_net > 0 and abs(abs(att_net) - abs(cash_net)) <= tol:
            return True
    # Same ex-div cycle: material Flex cash PIL 1–2 sessions ago, attribution steps later
    # with a different magnitude (accounting vs report-date cash).
    if att_pil < -0.5:
        for offset in range(1, session_lookback_days + 1):
            prior = (dt - pd.Timedelta(days=offset)).strftime("%Y-%m-%d")
            cash_pil = float(cash_pil_by_date.get(prior) or 0.0)
            if cash_pil <= -material_cash_pil_usd:
                return True
    return False


def compute_dividend_panel(
    *,
    dividend_cash_history_csv: Path,
    attribution_history_csv: Path,
    accounting_dir: Path,
    runs_root: Path,
    run_date: str,
    nav_usd: float,
    screener_csv: Path | None = None,
    plan_csv: Path | None = None,
    horizon_days: int = 7,
    warn_usd: float = 500.0,
) -> dict[str, Any]:
    """Daily dividend / PIL / withholding detail, sparkline, and ex-div warnings."""
    from ibkr_accounting import (
        load_yieldboost_etf_syms,
        parse_change_in_dividend_accruals,
        parse_open_positions,
    )

    try:
        from daily_screener import YIELDBOOST_BUCKET2_PAIRS
    except Exception:
        YIELDBOOST_BUCKET2_PAIRS = []

    att_daily = _dividend_daily_from_attribution(attribution_history_csv, run_date)

    cash_rows: list[dict[str, Any]] = []
    if dividend_cash_history_csv.is_file():
        try:
            ddf = pd.read_csv(dividend_cash_history_csv)
            if not ddf.empty and "date" in ddf.columns:
                ddf["date"] = ddf["date"].astype(str)
                ddf = ddf[ddf["date"] <= str(run_date)]
                for _, r in ddf.iterrows():
                    cash_rows.append(
                        {
                            "date": str(r.get("date") or ""),
                            "symbol": str(r.get("symbol") or ""),
                            "underlying": str(r.get("underlying") or ""),
                            "bucket": str(r.get("bucket") or ""),
                            "pair": str(r.get("pair") or ""),
                            "type": str(r.get("type") or ""),
                            "category": str(r.get("category") or ""),
                            "amount_usd": float(r.get("amount_usd") or 0.0),
                            "ex_date": str(r.get("ex_date") or ""),
                            "description": str(r.get("description") or ""),
                        }
                    )
        except Exception as exc:
            return {"available": False, "reason": f"unreadable dividend history: {exc}"}

    cash_pil_by_date: dict[str, float] = {}
    cash_net_by_date: dict[str, float] = {}
    for r in cash_rows:
        d = str(r.get("date") or "")
        if not d:
            continue
        cat = str(r.get("category") or "")
        amt = float(r.get("amount_usd") or 0.0)
        cash_net_by_date[d] = cash_net_by_date.get(d, 0.0) + amt
        if cat == "pil_dividends":
            cash_pil_by_date[d] = cash_pil_by_date.get(d, 0.0) + amt

    by_date: dict[str, Any] = {}
    dates = sorted({r["date"] for r in cash_rows if r.get("date")} | set(att_daily.keys()))
    for d in dates:
        day_rows = [r for r in cash_rows if r["date"] == d]
        by_bucket: dict[str, float] = {}
        by_cat: dict[str, float] = {k: 0.0 for k in DIVIDEND_CATEGORY_KEYS}
        for r in day_rows:
            cat = str(r.get("category") or "")
            amt = float(r.get("amount_usd") or 0.0)
            if cat in by_cat:
                by_cat[cat] += amt
            b = str(r.get("bucket") or "") or "unassigned"
            by_bucket[b] = by_bucket.get(b, 0.0) + amt
        att = att_daily.get(d, {k: 0.0 for k in DIVIDEND_CATEGORY_KEYS})
        att_net = float(att.get("net_usd") or sum(att.get(k, 0.0) for k in DIVIDEND_CATEGORY_KEYS))
        cash_net = sum(by_cat.values())
        lag_deduped = False
        source = "none"
        if day_rows:
            net_usd = cash_net
            dividends_usd = by_cat["dividends"]
            withholding_usd = by_cat["withholding_tax"]
            pil_usd = by_cat["pil_dividends"]
            source = "cash"
        elif _attribution_lags_recent_cash(att, cash_pil_by_date, cash_net_by_date, d):
            net_usd = 0.0
            dividends_usd = 0.0
            withholding_usd = 0.0
            pil_usd = 0.0
            lag_deduped = True
            source = "attribution_suppressed"
        else:
            net_usd = att_net
            dividends_usd = float(att.get("dividends") or 0.0)
            withholding_usd = float(att.get("withholding_tax") or 0.0)
            pil_usd = float(att.get("pil_dividends") or 0.0)
            source = "attribution"
        by_date[d] = {
            "net_usd": net_usd,
            "dividends_usd": dividends_usd,
            "withholding_usd": withholding_usd,
            "pil_usd": pil_usd,
            "attribution_net_usd": att_net,
            "source": source,
            "lag_deduped": lag_deduped,
            "rows": sorted(day_rows, key=lambda r: abs(float(r.get("amount_usd") or 0.0)), reverse=True),
            "by_bucket": dict(sorted(by_bucket.items(), key=lambda kv: abs(kv[1]), reverse=True)),
        }

    sparkline = [
        {
            "date": d,
            "net_usd": float(by_date[d]["net_usd"]),
            "pil_usd": float(by_date[d]["pil_usd"]),
            "dividends_usd": float(by_date[d]["dividends_usd"]),
        }
        for d in dates
    ]

    # Expected PIL / ex-div warnings (short ETF legs)
    warnings: list[dict[str, Any]] = []
    run_dt = pd.Timestamp(run_date)
    horizon_end = run_dt + pd.Timedelta(days=horizon_days)
    cash_xml = runs_root / run_date / "ibkr_flex" / "flex_cash.xml"
    flex_pos = runs_root / run_date / "ibkr_flex" / "flex_positions.xml"
    yb_etfs = set(load_yieldboost_etf_syms(screener_csv or Path(), plan_csv))
    yb_etfs |= {str(etf) for etf, _ in YIELDBOOST_BUCKET2_PAIRS}

    short_qty: dict[str, float] = {}
    if flex_pos.is_file():
        pos = parse_open_positions(flex_pos)
        if not pos.empty and "symbol" in pos.columns and "position" in pos.columns:
            for _, p in pos.iterrows():
                sym = str(p.get("symbol") or "")
                qty = float(p.get("position") or 0.0)
                if qty < -1e-9:
                    short_qty[sym] = min(short_qty.get(sym, 0.0), qty)

    if cash_xml.is_file():
        accr = parse_change_in_dividend_accruals(cash_xml)
        if not accr.empty:
            accr = accr[accr["code"].astype(str).str.upper() == "PO"].copy()
            for _, a in accr.iterrows():
                ex_raw = str(a.get("exDate") or "")
                if len(ex_raw) != 8:
                    continue
                ex_dt = pd.Timestamp(f"{ex_raw[:4]}-{ex_raw[4:6]}-{ex_raw[6:8]}")
                if ex_dt <= run_dt or ex_dt > horizon_end:
                    continue
                sym = str(a.get("symbol") or "")
                sq = float(short_qty.get(sym, 0.0))
                if sq >= -1e-9:
                    continue
                gross_rate = float(a.get("grossRate") or 0.0)
                fx = float(a.get("fxRateToBase") or 1.0)
                est_pil = -abs(gross_rate * sq * fx)
                if abs(est_pil) < 1.0:
                    continue
                pay_raw = str(a.get("payDate") or "")
                pay_iso = (
                    f"{pay_raw[:4]}-{pay_raw[4:6]}-{pay_raw[6:8]}" if len(pay_raw) == 8 else ""
                )
                warnings.append(
                    {
                        "symbol": sym,
                        "underlying": str(a.get("underlyingSymbol") or sym),
                        "ex_date": ex_dt.strftime("%Y-%m-%d"),
                        "pay_date": pay_iso,
                        "short_qty": sq,
                        "gross_rate": gross_rate,
                        "estimated_pil_usd": est_pil,
                        "is_yieldboost": sym in yb_etfs,
                        "reason": "Short position through ex-div (estimated PIL)",
                    }
                )

    warnings.sort(key=lambda w: w["estimated_pil_usd"])
    week_total = sum(float(w["estimated_pil_usd"]) for w in warnings)

    sym_meta: dict[str, dict[str, str]] = {}
    pnl_sym = accounting_dir / "pnl_by_symbol.csv"
    if pnl_sym.is_file():
        try:
            sdf = pd.read_csv(pnl_sym)
            for _, r in sdf.iterrows():
                sym = str(r.get("symbol") or "")
                if sym:
                    sym_meta[sym] = {
                        "underlying": str(r.get("underlying") or sym),
                        "bucket": str(r.get("bucket") or ""),
                        "pair": str(r.get("pair") or ""),
                    }
        except Exception:
            pass
    for w in warnings:
        info = sym_meta.get(w["symbol"], {})
        w["bucket"] = info.get("bucket", "")
        w["pair"] = info.get("pair", "")

    nav = float(nav_usd) if nav_usd and nav_usd > 0 else 0.0
    return {
        "available": True,
        "source": "dividend_cash_history.csv + pnl_attribution_history.csv",
        "by_date": by_date,
        "sparkline": sparkline,
        "available_dates": dates,
        "category_labels": DIVIDEND_CATEGORY_LABELS,
        "expected_pil": {
            "available": True,
            "horizon_days": horizon_days,
            "threshold_usd": warn_usd,
            "week_total_usd": week_total,
            "week_total_pct_nav": (week_total / nav) if nav > 0 else None,
            "warn": abs(week_total) >= warn_usd,
            "warnings": warnings,
        },
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
    factor_panel: dict[str, Any] = field(default_factory=dict)
    concentration_panel: dict[str, Any] = field(default_factory=dict)
    slide_risk_panel: dict[str, Any] = field(default_factory=dict)
    action_queue: dict[str, Any] = field(default_factory=dict)
    alert_rows: list[dict[str, Any]] = field(default_factory=list)
    universe_counts: dict[str, Any] = field(default_factory=dict)
    raw_totals: dict[str, Any] = field(default_factory=dict)
    nav_source: str = "MAGIS_NAV_USD"
    bucket_sleeve_panel: dict[str, Any] = field(default_factory=dict)
    exposure_reconciliation: dict[str, Any] = field(default_factory=dict)
    borrow_shock_panel: dict[str, Any] = field(default_factory=dict)
    drawdown_panel: dict[str, Any] = field(default_factory=dict)
    pnl_panel: dict[str, Any] = field(default_factory=dict)
    hedged_pnl_panel: dict[str, Any] = field(default_factory=dict)
    shared_underlying_panel: dict[str, Any] = field(default_factory=dict)
    movers_panel: dict[str, Any] = field(default_factory=dict)
    bucket_movers_panel: dict[str, Any] = field(default_factory=dict)
    component_attribution_panel: dict[str, Any] = field(default_factory=dict)
    dividend_panel: dict[str, Any] = field(default_factory=dict)
    bucket_drawdown_panel: dict[str, Any] = field(default_factory=dict)
    display_sleeve_groups: list[dict[str, Any]] = field(default_factory=list)
    capital_panel: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_date": self.run_date,
            "generated_at_utc": self.generated_at_utc,
            "nav_usd": self.nav_usd,
            "nav_source": self.nav_source,
            "book": {
                "nav_usd": self.book.nav_usd,
                "gross_notional_usd": self.book.gross_notional_usd,
                "net_notional_usd": self.book.net_notional_usd,
                "long_notional_usd": self.book.long_notional_usd,
                "short_notional_usd": self.book.short_notional_usd,
                "gross_exposure_pct_nav": self.book.gross_exposure_pct_nav,
                "net_exposure_pct_nav": self.book.net_exposure_pct_nav,
                # ``pnl_today_*`` is a legacy field name that has always carried the
                # strategy *cumulative* (YTD) PnL, not a single-day move. The
                # ``pnl_ytd_*`` aliases make that explicit; ``pnl_daily_*`` (added
                # post-hoc in build_site from the prior snapshot) carries the true
                # day-over-day change.
                "pnl_today_usd": self.book.pnl_today_usd,
                "pnl_today_pct_nav": self.book.pnl_today_pct_nav,
                "pnl_ytd_usd": self.book.pnl_today_usd,
                "pnl_ytd_pct_nav": self.book.pnl_today_pct_nav,
                "sleeve_table": self.book.sleeve_table,
                "breaches": self.book.breaches,
                "sleeve_attribution_available": self.book.sleeve_attribution_available,
                "sleeve_attribution_reason": self.book.sleeve_attribution_reason,
            },
            "buckets": self.buckets,
            "borrow_panel": self.borrow_panel,
            "data_quality": self.data_quality,
            "factor_panel": self.factor_panel,
            "concentration_panel": self.concentration_panel,
            "slide_risk_panel": self.slide_risk_panel,
            "action_queue": self.action_queue,
            "worst_shock": (self.slide_risk_panel or {}).get("worst_shock"),
            "top_risk_contributors": [
                c
                for c in [
                    (self.slide_risk_panel or {}).get("worst_shock", {}).get("top_contributor")
                ]
                if c
            ],
            "alert_rows": self.alert_rows,
            "universe_counts": self.universe_counts,
            "raw_totals": self.raw_totals,
            "bucket_sleeve_panel": self.bucket_sleeve_panel,
            "exposure_reconciliation": self.exposure_reconciliation,
            "borrow_shock_panel": self.borrow_shock_panel,
            "drawdown_panel": self.drawdown_panel,
            "pnl_panel": self.pnl_panel,
            "hedged_pnl_panel": self.hedged_pnl_panel,
            "shared_underlying_panel": self.shared_underlying_panel,
            "movers_panel": self.movers_panel,
            "bucket_movers_panel": self.bucket_movers_panel,
            "component_attribution_panel": self.component_attribution_panel,
            "dividend_panel": self.dividend_panel,
            "bucket_drawdown_panel": self.bucket_drawdown_panel,
            "display_sleeve_groups": self.display_sleeve_groups,
            "capital_panel": self.capital_panel,
            "limits": getattr(self, "_limits", DEFAULT_LIMITS),
            "borrow_limits_by_bucket": getattr(self, "_borrow_limits_by_bucket", {}),
            "underlying_gross_frac_by_bucket": getattr(
                self, "_underlying_gross_frac_by_bucket", {}
            ),
            "liquidity_cap_fracs": getattr(self, "_liquidity_cap_fracs", {}),
            "sleeve_target_weights": getattr(
                self, "_sleeve_target_weights", SLEEVE_TARGET_WEIGHTS
            ),
            "book_target_gross_usd": getattr(self, "_book_target_gross_usd", None),
            "sleeve_target_source": getattr(self, "_sleeve_target_source", "fallback"),
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
    nav_source_hint: str = "MAGIS_NAV_USD",
) -> RiskSnapshot:
    """Build a full snapshot from a ``data/runs/<run_date>`` folder."""
    repo_root = runs_root
    for candidate in (runs_root, runs_root.parent, runs_root.parent.parent):
        if (candidate / "config" / "strategy_config.yml").is_file():
            repo_root = candidate
            break
    limits_ctx = load_risk_limits(repo_root / "config" / "strategy_config.yml")
    limits = limits or limits_ctx.limits

    run_dir = runs_root / run_date
    accounting = run_dir / "accounting"
    flex = run_dir / "ibkr_flex"

    totals = _read_json_or_empty(accounting / "totals.json")
    pnl_by_bucket = _read_csv_or_empty(accounting / "pnl_by_bucket.csv")
    blocked_exposure_keys = _load_blocked_exposure_keys(
        runs_root, screener_csv, run_date=run_date
    )

    cli_nav_usd = nav_usd
    nav_usd, nav_source = resolve_nav_usd(
        totals,
        cli_fallback=cli_nav_usd,
        flex_dir=flex if flex.is_dir() else None,
        cli_source=nav_source_hint,
    )
    book = compute_book_summary(
        totals,
        pnl_by_bucket,
        nav_usd=nav_usd,
        limits=limits,
        target_weights=limits_ctx.sleeve_target_weights,
        book_target_gross_usd=limits_ctx.book_target_gross_usd or None,
    )
    capital_panel = compute_capital_panel(
        totals,
        nav_usd=nav_usd,
        pnl_history_csv=repo_root / "data" / "ledger" / "pnl_history.csv",
    )
    bucket_sleeve_panel = compute_bucket_sleeve_rows(
        book.sleeve_table,
        capital_panel,
        totals,
    )
    exposure_reconciliation = evaluate_exposure_reconciliation(totals)

    buckets: dict[str, dict[str, Any]] = {}
    for bucket in BUCKET_KEYS:
        detail_csv = (
            accounting / "net_exposure_bucket_4_detail.csv"
            if bucket == "bucket_4"
            else None
        )
        buckets[bucket] = compute_bucket_detail(
            bucket=bucket,
            pnl_csv=accounting / f"pnl_{bucket}.csv",
            net_exposure_csv=accounting / f"net_exposure_{bucket}.csv",
            blocked_exposure_keys=blocked_exposure_keys,
            exposure_detail_csv=detail_csv,
            totals=totals,
        )

    borrow_panel = compute_borrow_panel(
        flex_borrow_xml=flex / "flex_borrow_fee_details.xml",
        flex_positions_xml=flex / "flex_positions.xml",
        limits=limits,
        screener_csv=screener_csv,
        run_date=run_date,
        limits_ctx=limits_ctx,
        repo_root=repo_root,
    )
    data_quality = compute_data_quality(
        accounting_dir=accounting,
        flex_dir=flex,
        buckets=buckets,
        totals=totals,
        run_date=run_date,
    )
    beta_results_dicts: dict[str, dict[str, Any]] = {}
    if enable_computed_betas and compute_betas is not None:
        exposure_csv = accounting / "net_exposure_by_underlying.csv"
        underlyings: list[str] = []
        if exposure_csv.is_file():
            try:
                _df = pd.read_csv(exposure_csv, usecols=["underlying"])
                if blocked_exposure_keys:
                    _df = _df[~_df["underlying"].astype(str).isin(blocked_exposure_keys)]
                underlyings = [
                    str(u).strip().upper() for u in _df["underlying"].dropna().tolist()
                ]
            except Exception:
                underlyings = []
        if underlyings:
            try:
                _br = compute_betas(
                    underlyings,
                    cache_dir=beta_cache_dir,
                )
                beta_results_dicts = {k: v.to_dict() for k, v in _br.items()}
                if write_summary_cache is not None and _br:
                    write_summary_cache(
                        _br,
                        snapshot_date=run_date,
                        path=repo_root / "data" / "cache" / "beta_summary.json",
                    )
            except Exception:
                beta_results_dicts = {}

    screener_vol_map = _load_screener_vol_map(screener_csv)
    screener_underlying_rows = _load_screener_underlying_rows(screener_csv)
    underlyings_for_vendor: list[str] = []
    exposure_csv = accounting / "net_exposure_by_underlying.csv"
    if exposure_csv.is_file():
        try:
            _udf = pd.read_csv(exposure_csv, usecols=["underlying"])
            if blocked_exposure_keys:
                _udf = _udf[~_udf["underlying"].astype(str).isin(blocked_exposure_keys)]
            underlyings_for_vendor = [
                str(u).strip().upper() for u in _udf["underlying"].dropna().tolist()
            ]
        except Exception:
            underlyings_for_vendor = []
    vendor_info_by_symbol: dict[str, dict[str, Any]] = {}
    if underlyings_for_vendor:
        try:
            vendor_info_by_symbol = fetch_vendor_info(
                underlyings_for_vendor,
                cache_path=repo_root / "data" / "cache" / "sector_vendor.json",
            )
        except Exception:
            vendor_info_by_symbol = {}
    factor_panel = compute_factor_panel(
        underlying_exposure_csv=accounting / "net_exposure_by_underlying.csv",
        nav_usd=nav_usd,
        beta_results=beta_results_dicts or None,
        screener_vol_map=screener_vol_map or None,
        screener_underlying_rows=screener_underlying_rows or None,
        vendor_info_by_symbol=vendor_info_by_symbol or None,
        blocked_exposure_keys=blocked_exposure_keys,
    )
    if factor_panel.get("available") and factor_panel.get("rows"):
        _write_sector_audit_csv(
            factor_panel["rows"],
            accounting / "sector_audit.csv",
        )
    if factor_panel.get("available"):
        by_bucket = compute_factor_by_bucket(
            accounting,
            nav_usd,
            beta_results=beta_results_dicts or None,
            blocked_exposure_keys=blocked_exposure_keys,
        )
        factor_panel["by_bucket"] = by_bucket
        portfolio_beta_net = float(
            (factor_panel.get("totals") or {}).get("beta_weighted_net_usd") or 0.0
        )
        # Additive sleeves only (B1/B2/B4 + unbucketed). Overlays (B3/B5) excluded.
        additive_rows = [r for r in by_bucket if r.get("additive")]
        overlay_rows = [r for r in by_bucket if not r.get("additive")]
        bucket_beta_sum = sum(float(r["beta_weighted_net_usd"]) for r in additive_rows)
        overlay_beta_sum = sum(float(r["beta_weighted_net_usd"]) for r in overlay_rows)
        additive_net = sum(float(r["net_notional_usd"]) for r in additive_rows)
        additive_beta_qqq = sum(
            float(r.get("beta_weighted_net_qqq_usd") or 0.0) for r in additive_rows
        )
        additive_beta_iwm = sum(
            float(r.get("beta_weighted_net_iwm_usd") or 0.0) for r in additive_rows
        )
        additive_beta_btc = sum(
            float(r.get("beta_weighted_net_btc_usd") or 0.0) for r in additive_rows
        )
        for row in by_bucket:
            if portfolio_beta_net and abs(portfolio_beta_net) > 1e-6:
                row["pct_of_portfolio_beta_net"] = (
                    float(row["beta_weighted_net_usd"]) / portfolio_beta_net
                    if row.get("additive")
                    else None
                )
            else:
                row["pct_of_portfolio_beta_net"] = None
            if row.get("additive") and bucket_beta_sum and abs(bucket_beta_sum) > 1e-6:
                row["pct_of_bucket_sum_beta_net"] = (
                    float(row["beta_weighted_net_usd"]) / bucket_beta_sum
                )
            else:
                row["pct_of_bucket_sum_beta_net"] = None
        totals_block = factor_panel.setdefault("totals", {})
        totals_block["by_bucket_beta_weighted_net_usd"] = bucket_beta_sum
        totals_block["by_bucket_beta_weighted_net_qqq_usd"] = additive_beta_qqq
        totals_block["by_bucket_beta_weighted_net_iwm_usd"] = additive_beta_iwm
        totals_block["by_bucket_beta_weighted_net_btc_usd"] = additive_beta_btc
        totals_block["by_bucket_net_notional_usd"] = additive_net
        totals_block["by_bucket_net_beta_to_spy"] = (
            (bucket_beta_sum / nav_usd) if nav_usd > 0 else None
        )
        totals_block["by_bucket_net_beta_to_qqq"] = (
            (additive_beta_qqq / nav_usd) if nav_usd > 0 else None
        )
        totals_block["by_bucket_net_beta_to_iwm"] = (
            (additive_beta_iwm / nav_usd) if nav_usd > 0 else None
        )
        totals_block["by_bucket_net_beta_to_btc"] = (
            (additive_beta_btc / nav_usd) if nav_usd > 0 else None
        )
        totals_block["by_bucket_overlay_beta_weighted_net_usd"] = overlay_beta_sum
        totals_block["by_bucket_attribution_mode"] = (
            (by_bucket[0].get("attribution_mode") if by_bucket else None)
        )
        totals_block["by_bucket_reconciles"] = (
            abs(bucket_beta_sum - portfolio_beta_net)
            < max(1.0, abs(portfolio_beta_net) * 0.02)
        )
    concentration_panel = compute_concentration_panel(
        factor_panel=factor_panel,
        nav_usd=nav_usd,
        limits=limits,
        limits_ctx=limits_ctx,
        accounting_dir=accounting,
        blocked_exposure_keys=blocked_exposure_keys,
    )
    slide_risk_panel = compute_slide_risk_panel(
        factor_panel=factor_panel,
        nav_usd=nav_usd,
        screener_csv=screener_csv,
        flex_positions_xml=flex / "flex_positions.xml",
        limits=limits,
        beta_cache_dir=beta_cache_dir,
        beta_results=beta_results_dicts or None,
        repo_root=repo_root,
    )
    display_sleeve_groups = compute_display_sleeve_groups(
        totals,
        nav_usd=nav_usd,
        sleeve_available=book.sleeve_attribution_available,
    )
    borrow_shock_panel = compute_borrow_shock_panel(borrow_panel, nav_usd=nav_usd)
    drawdown_panel = compute_drawdown_panel(
        repo_root / "data" / "ledger" / "pnl_history.csv",
        nav_usd=nav_usd,
        run_date=run_date,
    )
    pnl_panel = compute_pnl_panel(
        repo_root / "data" / "ledger" / "pnl_history.csv",
        nav_usd=nav_usd,
        run_date=run_date,
    )
    hedged_pnl_panel = compute_hedged_pnl_panel(
        accounting,
        hedged_history_csv=repo_root / "data" / "ledger" / "hedged_pnl_history.csv",
        nav_usd=nav_usd,
        run_date=run_date,
    )
    bucket_underlying_history = repo_root / "data" / "ledger" / "pnl_bucket_underlying_history.csv"
    bucket_movers_panel = compute_bucket_movers_panel(
        accounting,
        bucket_underlying_history_csv=bucket_underlying_history,
        pnl_history_csv=repo_root / "data" / "ledger" / "pnl_history.csv",
        run_date=run_date,
    )
    component_attribution_panel = compute_component_attribution_panel(
        repo_root / "data" / "ledger" / "pnl_attribution_history.csv",
        nav_usd=nav_usd,
        run_date=run_date,
    )
    dividend_panel = compute_dividend_panel(
        dividend_cash_history_csv=repo_root / "data" / "ledger" / "dividend_cash_history.csv",
        attribution_history_csv=repo_root / "data" / "ledger" / "pnl_attribution_history.csv",
        accounting_dir=accounting,
        runs_root=runs_root,
        run_date=run_date,
        nav_usd=nav_usd,
        screener_csv=screener_csv,
        plan_csv=run_dir / "proposed_trades.csv",
    )
    bucket_drawdown_panel = compute_bucket_drawdown_panel(
        repo_root / "data" / "ledger" / "pnl_history.csv",
        nav_usd=nav_usd,
        run_date=run_date,
    )
    shared_underlying_panel = compute_shared_underlying_panel(
        accounting, blocked_exposure_keys=blocked_exposure_keys
    )
    movers_panel = compute_movers_panel(accounting)
    action_queue = compute_action_queue(
        book=book,
        factor_panel=factor_panel,
        concentration_panel=concentration_panel,
        slide_risk_panel=slide_risk_panel,
        borrow_panel=borrow_panel,
        nav_usd=nav_usd,
        limits=limits,
    )
    alert_rows = compute_alert_rows(
        book=book,
        slide_risk_panel=slide_risk_panel,
        borrow_panel=borrow_panel,
        concentration_panel=concentration_panel,
        action_queue=action_queue,
    )

    universe_counts = {
        "kept_symbols": totals.get("kept_symbols"),
        "kept_underlyings": totals.get("kept_underlyings"),
        "universe_allowed_etfs": totals.get("universe_allowed_etfs"),
        "universe_allowed_underlyings": totals.get("universe_allowed_underlyings"),
    }

    snap = RiskSnapshot(
        run_date=run_date,
        generated_at_utc=generated_at_utc,
        nav_usd=nav_usd,
        nav_source=nav_source,
        book=book,
        buckets=buckets,
        borrow_panel=borrow_panel,
        data_quality=data_quality,
        factor_panel=factor_panel,
        concentration_panel=concentration_panel,
        slide_risk_panel=slide_risk_panel,
        action_queue=action_queue,
        alert_rows=alert_rows,
        universe_counts=universe_counts,
        raw_totals=totals,
        bucket_sleeve_panel=bucket_sleeve_panel,
        exposure_reconciliation=exposure_reconciliation,
        borrow_shock_panel=borrow_shock_panel,
        drawdown_panel=drawdown_panel,
        pnl_panel=pnl_panel,
        hedged_pnl_panel=hedged_pnl_panel,
        shared_underlying_panel=shared_underlying_panel,
        movers_panel=movers_panel,
        bucket_movers_panel=bucket_movers_panel,
        component_attribution_panel=component_attribution_panel,
        dividend_panel=dividend_panel,
        bucket_drawdown_panel=bucket_drawdown_panel,
        display_sleeve_groups=display_sleeve_groups,
        capital_panel=capital_panel,
    )
    snap._limits = limits_ctx.limits
    snap._borrow_limits_by_bucket = limits_ctx.borrow_apr_pct_by_bucket
    snap._underlying_gross_frac_by_bucket = limits_ctx.underlying_gross_frac_by_bucket
    snap._liquidity_cap_fracs = limits_ctx.liquidity_cap_fracs
    snap._sleeve_target_weights = limits_ctx.sleeve_target_weights
    snap._book_target_gross_usd = limits_ctx.book_target_gross_usd
    snap._sleeve_target_source = limits_ctx.sleeve_target_source
    return snap
