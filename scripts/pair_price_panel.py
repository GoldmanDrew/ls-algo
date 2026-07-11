"""Shared ETF/underlying price panels with flex split adjustment.

Used by production actual, research backtests, and (via copy/import) dashboard
builders so reverse splits do not invent +400% daily returns.
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
MIN_PRICE_PANEL_DAYS = 40


def _norm_sym(x: object) -> str:
    return str(x).strip().upper().replace(".", "-")


def split_events_by_symbol(
    *,
    flex_csv: Path | None = None,
    repo: Path | None = None,
) -> dict[str, list]:
    """Load flex split events keyed by panel ticker form."""
    try:
        from splits import load_flex_splits_csv
    except ImportError:
        return {}

    root = repo or REPO
    flex = flex_csv or (root / "data" / "splits_from_flex.csv")
    if not flex.is_file():
        return {}
    events = load_flex_splits_csv(flex)
    by_sym: dict[str, list] = {}
    for ev in events:
        raw = str(getattr(ev, "symbol", "") or "").strip().upper()
        if not raw:
            continue
        # Prefer clean tickers (COYY) over Flex id-prefixed rows (20260601…COYY).
        if len(raw) > 6 and any(ch.isdigit() for ch in raw[:8]):
            tail = "".join(ch for ch in raw if ch.isalpha() or ch in ".-")
            if 2 <= len(tail) <= 6:
                raw = tail
            else:
                continue
        sym = _norm_sym(raw)
        by_sym.setdefault(sym, []).append(ev)
    out: dict[str, list] = {}
    for sym, evs in by_sym.items():
        seen: set[tuple] = set()
        keep = []
        for ev in sorted(evs, key=lambda e: pd.Timestamp(e.ex_date)):
            key = (pd.Timestamp(ev.ex_date).normalize(), float(ev.factor))
            if key in seen:
                continue
            seen.add(key)
            keep.append(ev)
        out[sym] = keep
    return out


def apply_flex_splits_to_series(
    prices: pd.Series,
    symbol: str,
    *,
    split_map: Mapping[str, list] | None = None,
    residual_heuristic: bool = True,
    residual_jump_threshold: float = 0.5,
) -> pd.Series:
    """Return a copy of ``prices`` with flex (+ optional residual heuristic) splits applied."""
    try:
        from splits import apply_split_events, detect_heuristic_splits, repair_split_craters
    except ImportError:
        return prices.copy()

    sym = _norm_sym(symbol)
    smap = split_map if split_map is not None else split_events_by_symbol()
    a = prices.sort_index().copy()
    a = a[~a.index.duplicated(keep="last")]
    # Always stitch multi-day reverse-split garbage before Flex ×N.
    a = repair_split_craters(a, sym_label=sym)
    evs = list(smap.get(sym, []))
    if evs:
        a, _ = apply_split_events(a, evs, sym_label=sym)
    else:
        a = repair_split_craters(a, sym_label=sym)
    if residual_heuristic:
        r_chk = a.pct_change().abs()
        if len(r_chk) and float(r_chk.max()) > residual_jump_threshold:
            try:
                extra = list(detect_heuristic_splits(a, sym_label=sym))
                if extra:
                    a, _ = apply_split_events(a, extra, sym_label=sym)
                else:
                    a = repair_split_craters(a, sym_label=sym)
            except Exception:
                a = repair_split_craters(a, sym_label=sym)
    return a


def frames_from_metrics(
    md: pd.DataFrame,
    *,
    min_days: int = MIN_PRICE_PANEL_DAYS,
    etf_col: str = "etf_adj_close",
    und_col: str = "underlying_adj_close",
    ticker_col: str = "ticker",
    date_col: str = "date",
    apply_splits: bool = True,
    split_map: Mapping[str, list] | None = None,
) -> dict[str, pd.DataFrame]:
    """Build ``{ETF: DataFrame(a_px, b_px)}`` from a metrics table."""
    need = [date_col, ticker_col, etf_col, und_col]
    missing = [c for c in need if c not in md.columns]
    if missing:
        raise ValueError(f"metrics missing columns: {missing}")

    frame = md[need].copy()
    frame[ticker_col] = frame[ticker_col].map(_norm_sym)
    frame[date_col] = pd.to_datetime(frame[date_col], errors="coerce").dt.normalize()
    frame = frame.dropna(subset=[date_col]).sort_values([ticker_col, date_col])
    smap = split_map if split_map is not None else (split_events_by_symbol() if apply_splits else {})

    out: dict[str, pd.DataFrame] = {}
    for etf, g in frame.groupby(ticker_col):
        g = g.dropna(subset=[etf_col, und_col])
        if len(g) < min_days:
            continue
        idx = pd.DatetimeIndex(g[date_col])
        a = pd.Series(g[etf_col].to_numpy(dtype=float), index=idx)
        b = pd.Series(g[und_col].to_numpy(dtype=float), index=idx)
        a = a[~a.index.duplicated(keep="last")]
        b = b[~b.index.duplicated(keep="last")].reindex(a.index)
        if apply_splits:
            a = apply_flex_splits_to_series(a, str(etf), split_map=smap)
        df = pd.DataFrame(
            {"a_px": a.to_numpy(dtype=float), "b_px": b.to_numpy(dtype=float)},
            index=a.index,
        ).dropna()
        if len(df) < min_days:
            continue
        out[str(etf)] = df
    return out


def load_run_price_panel(
    run_date: str,
    *,
    repo: Path | None = None,
    min_days: int = MIN_PRICE_PANEL_DAYS,
) -> dict[str, pd.DataFrame]:
    """Load split-adjusted panels from ``data/runs/<date>/model_inputs/etf_metrics_daily.parquet``."""
    root = repo or REPO
    pq = root / f"data/runs/{run_date}/model_inputs/etf_metrics_daily.parquet"
    md = pd.read_parquet(pq, columns=["date", "ticker", "etf_adj_close", "underlying_adj_close"])
    return frames_from_metrics(md, min_days=min_days)
