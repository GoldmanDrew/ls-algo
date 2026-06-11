"""Split-aware realized gross decay from sibling etf-dashboard metrics store."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd

from vol_shape import resolve_etf_metrics_daily_path


def _etf_dashboard_scripts_dir() -> Path | None:
    root = Path(__file__).resolve().parent
    candidates = [
        root.parent / "etf-dashboard" / "scripts",
        Path(r"C:/Users/werdn/Documents/Investing/etf-dashboard/scripts"),
    ]
    for p in candidates:
        if (p / "realized_gross_decay.py").exists():
            return p
    return None


def load_split_aware_gross_decay_map(
    universe_etfs: set[str],
    *,
    metrics_daily_path: str | Path | None = None,
    beta_by_etf: dict[str, float] | None = None,
    min_obs: int = 40,
) -> dict[str, dict[str, Any]]:
    """Return per-ETF split-aware gross decay from etf_metrics_daily when available."""
    scripts = _etf_dashboard_scripts_dir()
    metrics = resolve_etf_metrics_daily_path(metrics_daily_path)
    if scripts is None or metrics is None:
        return {}

    metrics_path = Path(metrics)
    parquet = metrics_path.with_suffix(".parquet")
    csv_path = metrics_path if metrics_path.suffix.lower() == ".csv" else metrics_path.with_suffix(".csv")

    def _parquet_readable() -> bool:
        try:
            import pyarrow  # noqa: F401
            return True
        except ImportError:
            try:
                import fastparquet  # noqa: F401
                return True
            except ImportError:
                return False

    if parquet.exists() and _parquet_readable():
        source = parquet
    elif csv_path.exists():
        source = csv_path
    elif metrics_path.exists():
        source = metrics_path
    else:
        return {}

    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))

    from realized_gross_decay import load_gross_decay_from_metrics  # noqa: WPS433

    corp = scripts.parent / "data" / "corporate_actions.json"
    return load_gross_decay_from_metrics(
        source,
        {str(x or "").strip().upper() for x in universe_etfs if str(x or "").strip()},
        corp_actions_path=corp if corp.exists() else None,
        beta_by_symbol=beta_by_etf,
        min_obs=min_obs,
    )


def overlay_gross_decay_from_metrics(
    df: pd.DataFrame,
    *,
    metrics_daily_path: str | Path | None = None,
    min_obs: int = 40,
) -> pd.DataFrame:
    """Replace ``gross_decay_annual`` with split-aware metrics TR when computed."""
    if df.empty or "ETF" not in df.columns or "gross_decay_annual" not in df.columns:
        return df

    etfs = df["ETF"].dropna().astype(str).str.upper()
    beta_map: dict[str, float] = {}
    if "Delta" in df.columns:
        for _, row in df.iterrows():
            sym = str(row.get("ETF") or "").strip().upper()
            try:
                beta = float(row.get("Delta"))
            except (TypeError, ValueError):
                continue
            if sym and pd.notna(beta):
                beta_map[sym] = beta

    decay_map = load_split_aware_gross_decay_map(
        set(etfs),
        metrics_daily_path=metrics_daily_path,
        beta_by_etf=beta_map,
        min_obs=min_obs,
    )
    if not decay_map:
        return df

    out = df.copy()
    n = 0
    for i, row in out.iterrows():
        sym = str(row.get("ETF") or "").strip().upper()
        hit = decay_map.get(sym)
        if not hit:
            continue
        out.at[i, "gross_decay_annual"] = hit["gross_decay_annual"]
        n += 1
    if n:
        print(f"[DECAY][metrics-tr] overlay split-aware gross_decay_annual for {n} ETF(s)")
    return out
