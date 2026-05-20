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
import yaml

from .factor_map import lookup_underlying
from .flex_parser import (
    FlexBorrowFee,
    FlexPosition,
    parse_borrow_fee_details,
    parse_positions,
    summarize_borrow,
    summarize_positions,
)
from .scenario_engine import (
    SLIDE_SCENARIO_HORIZONS,
    aggregate_leg_scenario_pnl,
    horizon_to_years,
    model_leg_return,
    resolve_sigma_annual,
)

try:
    from .beta_loader import compute_betas  # noqa: F401
except Exception:  # pragma: no cover - optional dep / fallback
    compute_betas = None  # type: ignore[assignment]

try:
    from ibkr_accounting import (
        SUPPLEMENTAL_ETF_MAP,
        expand_blacklist,
        load_blacklist,
        load_etf_to_under_map,
        _filter_exposure_df,
    )
except ImportError:  # pragma: no cover
    SUPPLEMENTAL_ETF_MAP = {}
    expand_blacklist = None  # type: ignore[assignment]
    load_blacklist = None  # type: ignore[assignment]
    load_etf_to_under_map = None  # type: ignore[assignment]
    _filter_exposure_df = None  # type: ignore[assignment]


TRADING_DAYS_PER_YEAR_DAYS: int = 252


def _load_blocked_exposure_keys(
    runs_root: Path,
    screener_csv: Path | None = None,
) -> set[str]:
    """Symbols and underlyings excluded from gross/net exposure metrics."""
    if expand_blacklist is None or load_blacklist is None or load_etf_to_under_map is None:
        return set()
    project_root = runs_root.parent.parent
    config_yml = project_root / "config" / "strategy_config.yml"
    if not config_yml.exists():
        return set()
    screener_path = screener_csv or (project_root / "data" / "etf_screened_today.csv")
    etf_to_under = load_etf_to_under_map(screener_path) if screener_path.is_file() else {}
    for e_sym, u_sym in SUPPLEMENTAL_ETF_MAP.items():
        etf_to_under.setdefault(e_sym, u_sym)
    blacklist = load_blacklist(config_yml)
    blocked_symbols, blocked_underlyings = expand_blacklist(blacklist, etf_to_under)
    return blocked_symbols | blocked_underlyings


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
}


@dataclass
class RiskLimitsContext:
    """Limits and per-bucket thresholds from ``config/strategy_config.yml``."""

    limits: dict[str, dict[str, Any]]
    borrow_apr_pct_by_bucket: dict[str, dict[str, float]]
    underlying_gross_frac_by_bucket: dict[str, dict[str, float]]
    liquidity_cap_fracs: dict[str, float]


def _decimal_borrow_to_pct_limits(entry_cap: float, keep_cap: float) -> dict[str, float]:
    return {"warn": float(entry_cap) * 100.0, "hard": float(keep_cap) * 100.0}


def load_risk_limits(config_yml: Path | None = None) -> RiskLimitsContext:
    """Load dashboard limits from strategy_config (borrow bands, sleeve caps, liquidity)."""
    limits = {k: dict(v) for k, v in DEFAULT_LIMITS.items()}
    borrow_by_bucket: dict[str, dict[str, float]] = {
        "bucket_1": {"warn": 55.0, "hard": 75.0},
        "bucket_2": {"warn": 40.0, "hard": 50.0},
        "bucket_4": {"warn": 90.0, "hard": 120.0},
    }
    underlying_by_bucket: dict[str, dict[str, float]] = {
        "bucket_1": {"warn": 0.20, "hard": 0.20},
        "bucket_2": {"warn": 0.20, "hard": 0.20},
        "bucket_4": {"warn": 0.30, "hard": 0.30},
    }
    liquidity = {"shares_outstanding_use_frac": 0.35, "median_daily_volume_use_pct": 0.30}

    path = config_yml or Path("config/strategy_config.yml")
    if path.is_file():
        try:
            cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except Exception:
            cfg = {}
        screener = (cfg.get("screener") or {}) if isinstance(cfg, dict) else {}
        per_bucket = (screener.get("per_bucket") or {}) if isinstance(screener, dict) else {}
        for bkt in ("bucket_1", "bucket_2", "bucket_4"):
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
    )

SLEEVE_TARGET_WEIGHTS = {
    "bucket_1": 0.55,
    "bucket_2": 0.20,
    "bucket_3": None,
    "bucket_4": 0.25,
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
    a delta-normalized hedge OVERLAY (sum of |beta| * notional for
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
        target_gross_usd = (target_w * gross) if (target_w is not None and gross > 0) else None
        bucket_label = (
            "Bucket 3 (flow hedge overlay)"
            if bucket == "bucket_3"
            else bucket.replace("_", " ").title()
        )
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


def compute_bucket_detail(
    bucket: str,
    pnl_csv: Path,
    net_exposure_csv: Path,
    *,
    blocked_exposure_keys: set[str] | None = None,
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
    # b1+b2+b4 only - bucket_3 is a delta-normalized overlay and excluded
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


def compute_factor_panel(
    underlying_exposure_csv: Path,
    nav_usd: float,
    *,
    beta_results: dict[str, Any] | None = None,
    screener_vol_map: dict[str, float] | None = None,
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
        beta_to_qqq: float | None = None
        beta_to_iwm: float | None = None
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
                if br.get("beta_to_ndx") is not None:
                    beta_to_qqq = float(br["beta_to_ndx"])
                if br.get("beta_to_rut") is not None:
                    beta_to_iwm = float(br["beta_to_rut"])
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
                "beta_to_qqq": beta_to_qqq,
                "beta_to_iwm": beta_to_iwm,
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
    0.01, 0.02, 0.03, 0.05, 0.10, 0.15, 0.20,
)
SLIDE_HORIZONS_DAYS: tuple[int, ...] = (0, 5, 20)  # legacy; slide panel uses SLIDE_SCENARIO_HORIZONS
VIX_ABS_SHOCKS_POINTS: tuple[int, ...] = (-10, -5, 2, 5, 10, 15, 20, 25, 30)
VOL_REGIME_MULTIPLIERS: tuple[float, ...] = (1.25, 1.5, 2.0, 3.0)
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


def _slide_horizon_scenario_totals(
    enriched: list[dict[str, Any]],
    *,
    etf_meta: dict[str, dict[str, Any]],
    shock_pct: float,
    horizon_key: str,
    vol_multiplier: float = 1.0,
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
    for e in enriched:
        beta = e.get("beta_to_spy")
        if beta is None:
            continue
        underlying_return = float(shock_pct) * float(beta)
        legs = _slide_scenario_legs_for_row(e, etf_meta=etf_meta)
        agg = aggregate_leg_scenario_pnl(
            legs,
            underlying_return=underlying_return,
            horizon_key=horizon_key,
            vol_multiplier=vol_multiplier,
            underlying_sigma=e.get("sigma"),
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
        for leg in legs:
            sigma, _ = resolve_sigma_annual(leg, underlying_sigma=e.get("sigma"))
            if sigma is not None:
                sigma_samples.append(sigma * vol_multiplier)
    totals["sigma_annual_median"] = (
        float(sorted(sigma_samples)[len(sigma_samples) // 2]) if sigma_samples else None
    )
    totals["horizon_key"] = horizon_key
    totals["horizon_years"] = horizon_to_years(horizon_key)
    totals["vol_multiplier"] = vol_multiplier
    return totals


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


def _borrow_cost_concentration(
    victims: list[dict[str, Any]],
    *,
    top_n: int = 3,
) -> dict[str, Any]:
    """Share of incremental borrow cost explained by the largest names."""
    positive = [
        v
        for v in victims
        if float(v.get("annual_cost_delta_usd") or v.get("annual_delta_usd") or 0.0) > 0
    ]
    if not positive:
        return {"top_victims": [], "top_n": top_n, "top_n_share": None, "diversified": True}
    denom = sum(
        float(v.get("annual_cost_delta_usd") or v.get("annual_delta_usd") or 0.0) for v in positive
    )
    ranked = sorted(
        positive,
        key=lambda v: float(v.get("annual_cost_delta_usd") or v.get("annual_delta_usd") or 0.0),
        reverse=True,
    )
    top = ranked[:top_n]
    top_sum = sum(
        float(v.get("annual_cost_delta_usd") or v.get("annual_delta_usd") or 0.0) for v in top
    )
    share = top_sum / denom if denom > 0 else None
    out_victims: list[dict[str, Any]] = []
    for v in top:
        delta = float(v.get("annual_cost_delta_usd") or v.get("annual_delta_usd") or 0.0)
        out_victims.append(
            {
                "symbol": v.get("symbol"),
                "annual_cost_delta_usd": delta,
                "pct_of_shock": (delta / denom) if denom > 0 else None,
                "borrow_rate_pct": v.get("borrow_rate_pct") or v.get("current_apr_pct"),
                "stressed_borrow_rate_pct": v.get("stressed_borrow_rate_pct")
                or v.get("new_apr_pct"),
            }
        )
    return {
        "top_victims": out_victims,
        "top_n": top_n,
        "top_n_share": share,
        "diversified": share is not None and share < 0.70,
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
    vol_multipliers: tuple[float, ...] = VOL_REGIME_MULTIPLIERS,
    limits: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Slide risk strips for SPX (beta-adjusted) and VIX (vega + vol regime)."""
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
        per_name_pnl_t0: list[dict[str, Any]] = []
        for e in enriched:
            beta = e.get("beta_to_spy")
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
            )
            total = float(totals["total_pnl_usd"])
            horizon_rows.append(
                {
                    "horizon_key": horizon_key,
                    "horizon_years": totals["horizon_years"],
                    "vol_multiplier": 1.0,
                    "sigma_annual_median": totals.get("sigma_annual_median"),
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
        }
    )

    # --- VIX strip (vega P&L + vol regime decay) ---
    vix_shock_rows: list[dict[str, Any]] = []
    for vix_pts in vix_shocks:
        per_name = []
        for e in enriched:
            if abs(e["vega_frac"]) < 1e-12:
                continue
            net = e["net_notional_usd"]
            sign = 1.0 if net >= 0 else -1.0
            pnl = sign * e["gross_notional_usd"] * e["vega_frac"] * vix_pts
            per_name.append(
                {
                    "underlying": e["underlying"],
                    "symbols": e["symbols"],
                    "pnl_t0_usd": pnl,
                    "vega_product_class": e["vega_product_class"],
                }
            )
        per_name.sort(key=lambda p: p["pnl_t0_usd"])
        total = sum(p["pnl_t0_usd"] for p in per_name)
        pnl_pct = (total / nav_usd) if nav_usd > 0 else None
        sign_label = "+" if vix_pts >= 0 else ""
        vix_row = {
            "vix_shock_pts": vix_pts,
            "label": f"VIX {sign_label}{vix_pts} pts",
            "pnl_usd": total,
            "pnl_pct_nav": pnl_pct,
            "status": _slide_status(pnl_pct, limits),
            "top_loss": per_name[0] if per_name else None,
            "top_gain": per_name[-1] if per_name else None,
            "n_contributors": len(per_name),
        }
        vix_shock_rows.append(vix_row)
        if vix_row["status"] != "ok" and vix_pts > 0:
            breaches.append(
                _normalize_breach(
                    {
                        "metric": f"slide:vix_{vix_pts:+d}pts",
                        "label": vix_row["label"],
                        "value": pnl_pct,
                        "status": vix_row["status"],
                        "limit": limits["scenario_loss_pct_nav"],
                        "source": "slide risk — VIX shock",
                        "action": "Review short-vol / yieldboost exposure.",
                    }
                )
            )

    vol_regime_rows: list[dict[str, Any]] = []
    for mult in vol_multipliers:
        per_name = []
        for e in enriched:
            decay_usd = 0.0
            legs = _slide_scenario_legs_for_row(e, etf_meta=etf_meta)
            for leg in legs:
                if not leg.get("is_letf"):
                    continue
                k = float(leg.get("leverage_k", 1.0))
                if abs(k) <= 1.0001:
                    continue
                sigma, _ = resolve_sigma_annual(leg, underlying_sigma=e.get("sigma"))
                if sigma is None or sigma <= 0:
                    continue
                result = model_leg_return(
                    leg=leg,
                    underlying_return=0.0,
                    horizon_key=scenario_horizons[-1],
                    vol_multiplier=float(mult),
                    underlying_sigma=e.get("sigma"),
                )
                decay_usd += leg["net_notional_usd"] * result.decay_return
            if abs(decay_usd) < 1e-6:
                continue
            per_name.append(
                {"underlying": e["underlying"], "symbols": e["symbols"], "pnl_t0_usd": decay_usd}
            )
        per_name.sort(key=lambda p: p["pnl_t0_usd"])
        total = sum(p["pnl_t0_usd"] for p in per_name)
        pnl_pct = (total / nav_usd) if nav_usd > 0 else None
        decay_conc = _pnl_concentration_summary(per_name, total, top_n=3)
        vol_regime_rows.append(
            {
                "multiplier": mult,
                "label": f"{mult:g}x forecast vol ({scenario_horizons[-1]})",
                "pnl_usd": total,
                "pnl_pct_nav": pnl_pct,
                "status": _slide_status(pnl_pct, limits),
                "top_loss": per_name[0] if per_name else None,
                "n_contributors": len(per_name),
                "decay_concentration": decay_conc,
            }
        )

    indices_out.append(
        {
            "index": "VIX",
            "key": "vix",
            "strip_type": "vix_pts",
            "shock_rows": vix_shock_rows,
            "vol_regime_rows": vol_regime_rows,
            "n_vega_contributors": sum(1 for e in enriched if abs(e["vega_frac"]) > 0),
            "n_letf_decay_contributors": sum(
                1 for e in enriched if e["is_letf"] and e["sigma"] is not None
            ),
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
        "model": "etf_dashboard_scenarios_v1",
    }


# ---------------------------------------------------------------------------
# Borrow shock sensitivity (Phase 3)


BORROW_ABS_SHOCKS_BP: tuple[int, ...] = (25, 50, 100, 200, 500)
BORROW_MULT_SHOCKS: tuple[float, ...] = (1.5, 2.0, 3.0, 5.0)
BORROW_SHOCK_FOCUS_BP: tuple[int, ...] = (50,)
BORROW_SHOCK_FOCUS_MULT: tuple[float, ...] = (2.0,)
ACTION_QUEUE_CAP: int = 5


def compute_borrow_shock_panel(
    *,
    borrow_panel: dict[str, Any],
    flex_positions_xml: Path,
    nav_usd: float,
    screener_csv: Path | None = None,
    watchlist_symbols: set[str] | None = None,
    abs_shocks_bp: tuple[int, ...] = BORROW_ABS_SHOCKS_BP,
    mult_shocks: tuple[float, ...] = BORROW_MULT_SHOCKS,
    persistence_days: int = 30,
) -> dict[str, Any]:
    """Per-symbol borrow cost sensitivity to rate shocks.

    Inputs:
        * ``flex_positions_xml`` -- short notional per symbol.
        * ``screener_csv`` -- borrow rate (``borrow_current`` / ``borrow_fee_annual``),
          same convention as etf-dashboard.

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

    screener_meta = _load_screener_borrow_meta(screener_csv)
    watch = {s.upper() for s in (watchlist_symbols or set())}

    name_rows: list[dict[str, Any]] = []
    for sym, short_n in short_notional.items():
        if watch and sym not in watch:
            continue
        rate = _screener_borrow_rate(sym, screener_meta)
        eff = float(rate["borrow_rate_pct"] or 0.0)
        current_annual_cost = short_n * (eff / 100.0) if rate["borrow_rate_known"] else 0.0
        name_rows.append(
            {
                "symbol": sym,
                "short_notional_usd": short_n,
                "current_annual_cost_usd": current_annual_cost,
                **rate,
            }
        )
    name_rows.sort(key=lambda r: r["current_annual_cost_usd"], reverse=True)
    total_current_annual = sum(r["current_annual_cost_usd"] for r in name_rows)

    def _shock_ladder(shocks: Iterable[Any], *, kind: str) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for shock in shocks:
            if kind == "abs_bp":
                shifted_aprs = {
                    r["symbol"]: float(r["borrow_rate_pct"] or 0.0) + (shock / 100.0)
                    for r in name_rows
                }
                label = f"+{int(shock)}bp"
            else:
                shifted_aprs = {
                    r["symbol"]: float(r["borrow_rate_pct"] or 0.0) * float(shock)
                    for r in name_rows
                }
                label = f"{shock:g}x"
            per_name_delta = []
            for r in name_rows:
                new_apr = shifted_aprs[r["symbol"]]
                new_cost = r["short_notional_usd"] * (new_apr / 100.0)
                cost_delta_usd = new_cost - r["current_annual_cost_usd"]
                per_name_delta.append(
                    {
                        "symbol": r["symbol"],
                        "short_notional_usd": r["short_notional_usd"],
                        "borrow_rate_pct": r.get("borrow_rate_pct"),
                        "stressed_borrow_rate_pct": new_apr,
                        "stressed_apr_pct": new_apr,
                        "current_apr_pct": r.get("borrow_rate_pct"),
                        "new_apr_pct": new_apr,
                        "annual_cost_delta_usd": cost_delta_usd,
                        "annual_delta_usd": cost_delta_usd,
                        "daily_cost_delta_usd": cost_delta_usd / TRADING_DAYS_PER_YEAR_DAYS,
                        "daily_delta_usd": cost_delta_usd / TRADING_DAYS_PER_YEAR_DAYS,
                        "persistence_usd": (cost_delta_usd / TRADING_DAYS_PER_YEAR_DAYS)
                        * persistence_days,
                    }
                )
            per_name_delta.sort(key=lambda d: d["annual_cost_delta_usd"], reverse=True)
            total_delta = sum(d["annual_cost_delta_usd"] for d in per_name_delta)
            victim_conc = _borrow_cost_concentration(per_name_delta, top_n=3)
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
                    "victim_concentration": victim_conc,
                    "is_focus": False,
                }
            )
        return out

    focus_abs = [s for s in abs_shocks_bp if s in BORROW_SHOCK_FOCUS_BP] or list(BORROW_SHOCK_FOCUS_BP)
    focus_mult = [s for s in mult_shocks if s in BORROW_SHOCK_FOCUS_MULT] or list(BORROW_SHOCK_FOCUS_MULT)
    abs_ladder = _shock_ladder(abs_shocks_bp, kind="abs_bp")
    mult_ladder = _shock_ladder(mult_shocks, kind="mult")
    for row in abs_ladder:
        if row.get("shock") in focus_abs:
            row["is_focus"] = True
    for row in mult_ladder:
        if row.get("shock") in focus_mult:
            row["is_focus"] = True

    current_conc = _borrow_cost_concentration(
        [
            {
                "symbol": r["symbol"],
                "annual_cost_delta_usd": r["current_annual_cost_usd"],
            }
            for r in name_rows
            if float(r.get("current_annual_cost_usd") or 0.0) > 0
        ],
        top_n=3,
    )

    def _tile_from_ladder(ladder: list[dict[str, Any]], label_match: str) -> dict[str, Any] | None:
        for row in ladder:
            if row.get("label") == label_match:
                return row
        return None

    focus_50 = _tile_from_ladder(abs_ladder, "+50bp")
    focus_2x = _tile_from_ladder(mult_ladder, "2x")

    return {
        "available": True,
        "sleeve": "short_etf",
        "watchlist_n_symbols": len(watch),
        "abs_shocks_bp": list(abs_shocks_bp),
        "mult_shocks": list(mult_shocks),
        "abs_ladder": abs_ladder,
        "mult_ladder": mult_ladder,
        "focus_abs_ladder": [r for r in abs_ladder if r.get("is_focus")],
        "focus_mult_ladder": [r for r in mult_ladder if r.get("is_focus")],
        "names": name_rows[:25],
        "n_short_symbols": len(name_rows),
        "current_annual_cost_usd": total_current_annual,
        "current_annual_cost_pct_nav": (total_current_annual / nav_usd)
        if nav_usd > 0
        else None,
        "persistence_days": persistence_days,
        "current_borrow_concentration": current_conc,
        "summary_tiles": {
            "focus_abs_50bp": focus_50,
            "focus_mult_2x": focus_2x,
            "top3_share_current_borrow": current_conc.get("top_n_share"),
        },
    }


# ---------------------------------------------------------------------------
# VIX / vol shock sensitivity (Phase 4) — legacy panel; folded into slide risk


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
    borrow_shock_panel: dict[str, Any] = field(default_factory=dict)
    action_queue: dict[str, Any] = field(default_factory=dict)
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
            "factor_panel": self.factor_panel,
            "concentration_panel": self.concentration_panel,
            "slide_risk_panel": self.slide_risk_panel,
            "borrow_shock_panel": self.borrow_shock_panel,
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
            "limits": getattr(self, "_limits", DEFAULT_LIMITS),
            "borrow_limits_by_bucket": getattr(self, "_borrow_limits_by_bucket", {}),
            "underlying_gross_frac_by_bucket": getattr(
                self, "_underlying_gross_frac_by_bucket", {}
            ),
            "liquidity_cap_fracs": getattr(self, "_liquidity_cap_fracs", {}),
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
    blocked_exposure_keys = _load_blocked_exposure_keys(runs_root, screener_csv)

    book = compute_book_summary(totals, pnl_by_bucket, nav_usd=nav_usd, limits=limits)

    buckets: dict[str, dict[str, Any]] = {}
    for bucket in ("bucket_1", "bucket_2", "bucket_3", "bucket_4"):
        buckets[bucket] = compute_bucket_detail(
            bucket=bucket,
            pnl_csv=accounting / f"pnl_{bucket}.csv",
            net_exposure_csv=accounting / f"net_exposure_{bucket}.csv",
            blocked_exposure_keys=blocked_exposure_keys,
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
    borrow_watchlist = _borrow_watchlist_symbols(
        screener_csv=screener_csv,
        flex_positions_xml=flex / "flex_positions.xml",
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
        blocked_exposure_keys=blocked_exposure_keys,
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
    )
    borrow_shock_panel = compute_borrow_shock_panel(
        borrow_panel=borrow_panel,
        flex_positions_xml=flex / "flex_positions.xml",
        nav_usd=nav_usd,
        screener_csv=screener_csv,
        watchlist_symbols=borrow_watchlist,
    )
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
        book=book,
        buckets=buckets,
        borrow_panel=borrow_panel,
        data_quality=data_quality,
        factor_panel=factor_panel,
        concentration_panel=concentration_panel,
        slide_risk_panel=slide_risk_panel,
        borrow_shock_panel=borrow_shock_panel,
        action_queue=action_queue,
        alert_rows=alert_rows,
        universe_counts=universe_counts,
        raw_totals=totals,
    )
    snap._limits = limits_ctx.limits
    snap._borrow_limits_by_bucket = limits_ctx.borrow_apr_pct_by_bucket
    snap._underlying_gross_frac_by_bucket = limits_ctx.underlying_gross_frac_by_bucket
    snap._liquidity_cap_fracs = limits_ctx.liquidity_cap_fracs
    return snap
