"""Shared screener universe and exposure blacklist helpers.

Used by EOD email, IBKR accounting consumers, and the risk dashboard so
universe / blacklist scope stays aligned across reporting surfaces.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ibkr_accounting import (
    SUPPLEMENTAL_ETF_MAP,
    canonical_symbol,
    expand_blacklist,
    load_blacklist,
    load_etf_to_under_map,
)

RECONCILE_EXPOSURE_BUCKETS: tuple[str, ...] = ("bucket_1", "bucket_2", "bucket_4")
STOCK_SLEEVE_BUCKETS: tuple[str, ...] = ("bucket_1", "bucket_2", "bucket_4")


def screened_etf_and_underlying_sets(screened: pd.DataFrame) -> tuple[set[str], set[str]]:
    """Return (ETF tickers, Underlying tickers) from etf_screened_today.csv."""
    if screened is None or screened.empty:
        return set(), set()
    cols = {c.lower(): c for c in screened.columns}
    etf_col = cols.get("etf")
    under_col = next(
        (cols[k] for k in ("underlying", "underlyingsymbol", "underlying_symbol", "root") if k in cols),
        None,
    )
    etfs: set[str] = set()
    unders: set[str] = set()
    if etf_col:
        etfs.update(
            canonical_symbol(str(x))
            for x in screened[etf_col].dropna()
            if str(x).strip()
        )
    if under_col:
        unders.update(
            canonical_symbol(str(x))
            for x in screened[under_col].dropna()
            if str(x).strip()
        )
    return {s for s in etfs if s}, {s for s in unders if s}


def screened_universe_symbols(screened: pd.DataFrame) -> set[str]:
    """ETF and underlying tickers from etf_screened_today.csv (strategy universe)."""
    etfs, unders = screened_etf_and_underlying_sets(screened)
    return etfs | unders


def resolve_screened_csv(
    *,
    run_date: str | None = None,
    runs_root: Path | None = None,
    project_root: Path | None = None,
    screener_csv: Path | None = None,
) -> Path | None:
    """Best-effort path to the screener CSV for a run."""
    if screener_csv is not None and screener_csv.is_file():
        return screener_csv
    root = project_root or Path(".")
    runs = runs_root or (root / "data" / "runs")
    if run_date:
        dated = runs / run_date / "etf_screened_today.csv"
        if dated.is_file():
            return dated
    latest = root / "data" / "etf_screened_today.csv"
    return latest if latest.is_file() else None


def load_screened_for_run(
    run_date: str,
    *,
    runs_root: Path | None = None,
    project_root: Path | None = None,
) -> pd.DataFrame:
    path = resolve_screened_csv(run_date=run_date, runs_root=runs_root, project_root=project_root)
    if path is None:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def load_blocked_exposure_keys(
    *,
    config_yml: Path | None = None,
    screener_csv: Path | None = None,
    run_date: str | None = None,
    runs_root: Path | None = None,
    project_root: Path | None = None,
) -> set[str]:
    """Symbols and underlyings excluded from gross/net exposure metrics."""
    root = project_root or Path(".")
    cfg_path = config_yml or (root / "config" / "strategy_config.yml")
    if not cfg_path.is_file():
        return set()
    screened_path = resolve_screened_csv(
        run_date=run_date,
        runs_root=runs_root,
        project_root=root,
        screener_csv=screener_csv,
    )
    etf_to_under = load_etf_to_under_map(screened_path) if screened_path else {}
    for e_sym, u_sym in SUPPLEMENTAL_ETF_MAP.items():
        etf_to_under.setdefault(e_sym, u_sym)
    blacklist = load_blacklist(cfg_path)
    blocked_symbols, blocked_underlyings = expand_blacklist(blacklist, etf_to_under)
    return blocked_symbols | blocked_underlyings


def load_blocked_exposure_sets(
    run_date: str,
    *,
    runs_root: Path | None = None,
    project_root: Path | None = None,
) -> tuple[set[str], set[str]]:
    """Return (blocked_symbols, blocked_underlyings) for exposure metrics."""
    root = project_root or Path(".")
    cfg_path = root / "config" / "strategy_config.yml"
    screened_path = resolve_screened_csv(
        run_date=run_date, runs_root=runs_root, project_root=root
    )
    etf_to_under = load_etf_to_under_map(screened_path) if screened_path else {}
    for e_sym, u_sym in SUPPLEMENTAL_ETF_MAP.items():
        etf_to_under.setdefault(e_sym, u_sym)
    blacklist = load_blacklist(cfg_path) if cfg_path.exists() else set()
    return expand_blacklist(blacklist, etf_to_under)
