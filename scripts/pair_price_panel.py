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
    """Return a copy of ``prices`` with flex + overrides (+ residual heuristic).

    Order matches the historical panel path so residual heuristics see the
    post-flex series (not raw Yahoo), which matters for BAIG/BMNG-style
    windows. Operator overrides from ``data/splits_overrides.csv`` are applied
    after flex / legacy manuals so COYY / SNDU residuals still land even when
    flex self-heals on an earlier crater date.
    """
    try:
        from splits import (
            apply_split_events,
            detect_heuristic_splits,
            load_legacy_manual_overrides,
            load_splits_overrides_csv,
            repair_split_craters,
        )
    except ImportError:
        return prices.copy()

    sym = _norm_sym(symbol)
    smap = split_map if split_map is not None else split_events_by_symbol()
    a = prices.sort_index().copy()
    a = a[~a.index.duplicated(keep="last")]
    # Always stitch multi-day reverse-split garbage before Flex ×N.
    a = repair_split_craters(a, sym_label=sym)

    flex_evs = list(smap.get(sym, []))
    legacy = [
        e
        for e in load_legacy_manual_overrides()
        if str(getattr(e, "symbol", "") or "").strip().upper().replace(".", "-") == sym
    ]
    stage1 = flex_evs + legacy
    if stage1:
        a, _ = apply_split_events(a, stage1, sym_label=sym)
    else:
        a = repair_split_craters(a, sym_label=sym)

    overrides_path = REPO / "data" / "splits_overrides.csv"
    if overrides_path.is_file():
        ov = [
            e
            for e in load_splits_overrides_csv(overrides_path)
            if str(getattr(e, "symbol", "") or "").strip().upper().replace(".", "-")
            == sym
        ]
        if ov:
            a, _ = apply_split_events(a, ov, sym_label=sym)

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
    a = apply_price_patches(a, sym)
    return a


def load_price_patches(*, repo: Path | None = None) -> dict[str, pd.Series]:
    """``{SYMBOL: Series(date -> close)}`` from ``data/price_patches.csv``."""
    root = repo or REPO
    path = root / "data" / "price_patches.csv"
    if not path.is_file():
        return {}
    try:
        df = pd.read_csv(path)
    except Exception:
        return {}
    if df.empty or "symbol" not in df.columns or "date" not in df.columns or "close" not in df.columns:
        return {}
    out: dict[str, pd.Series] = {}
    df["symbol"] = df["symbol"].map(_norm_sym)
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["symbol", "date", "close"])
    for sym, g in df.groupby("symbol"):
        s = pd.Series(g["close"].to_numpy(dtype=float), index=pd.DatetimeIndex(g["date"]))
        out[str(sym)] = s[~s.index.duplicated(keep="last")].sort_index()
    return out


def apply_price_patches(prices: pd.Series, symbol: str, *, repo: Path | None = None) -> pd.Series:
    """Overwrite closes on patched dates (operator / Yahoo referee)."""
    patches = load_price_patches(repo=repo).get(_norm_sym(symbol))
    if patches is None or patches.empty:
        return prices
    out = prices.sort_index().copy()
    out.index = pd.DatetimeIndex(pd.to_datetime(out.index)).normalize()
    for dt, px in patches.items():
        ts = pd.Timestamp(dt).normalize()
        if ts in out.index or True:
            out.loc[ts] = float(px)
    return out[~out.index.duplicated(keep="last")].sort_index()


def referee_replace_false_prints(
    a: pd.Series,
    symbol: str,
    *,
    jump_thresh: float = 0.5,
    scale_band: float = 0.40,
) -> pd.Series:
    """Replace metrics prints that disagree with Yahoo on phantom jumps / scale.

    Corroboration: if Yahoo move is also large, keep metrics (real event).
    """
    a = pd.to_numeric(a, errors="coerce").dropna().sort_index()
    if a.empty:
        return a
    start = pd.Timestamp(a.index.min()) - pd.Timedelta(days=5)
    end = pd.Timestamp(a.index.max())
    y = _yahoo_close(symbol, start, end)
    if y.empty:
        return a
    j = pd.DataFrame({"m": a, "y": y}).dropna()
    j = j[(j["m"] > 0) & (j["y"] > 0)]
    if len(j) < 3:
        return a
    out = a.copy()
    m_ret = j["m"].pct_change()
    y_ret = j["y"].pct_change()
    for dt in j.index:
        mr = float(m_ret.loc[dt]) if dt in m_ret.index and pd.notna(m_ret.loc[dt]) else 0.0
        yr = float(y_ret.loc[dt]) if dt in y_ret.index and pd.notna(y_ret.loc[dt]) else 0.0
        ratio = float(j.loc[dt, "m"] / j.loc[dt, "y"])
        phantom = abs(mr) > jump_thresh and abs(yr) < max(0.10, 0.25 * abs(mr))
        scale = abs(ratio - 1.0) > scale_band
        if phantom or scale:
            out.loc[pd.Timestamp(dt).normalize()] = float(j.loc[dt, "y"])
    return out[~out.index.duplicated(keep="last")].sort_index()


def load_delistings(*, repo: Path | None = None) -> dict[str, pd.Timestamp]:
    """``{SYMBOL: last_trade_date}`` from ``data/delistings.csv``."""
    root = repo or REPO
    path = root / "data" / "delistings.csv"
    if not path.is_file():
        return {}
    try:
        df = pd.read_csv(path)
    except Exception:
        return {}
    if df.empty or "symbol" not in df.columns or "last_trade_date" not in df.columns:
        return {}
    out: dict[str, pd.Timestamp] = {}
    for _, r in df.iterrows():
        sym = _norm_sym(r.get("symbol"))
        dt = pd.to_datetime(r.get("last_trade_date"), errors="coerce")
        if sym and pd.notna(dt):
            out[sym] = pd.Timestamp(dt).normalize()
    return out


def apply_delist_cutoff(
    panel: dict[str, pd.DataFrame],
    *,
    delist_map: Mapping[str, pd.Timestamp] | None = None,
    repo: Path | None = None,
) -> dict[str, pd.DataFrame]:
    """Drop panel rows after each ETF's last_trade_date (no post-delist marks)."""
    dmap = dict(delist_map or load_delistings(repo=repo))
    if not dmap:
        return panel
    out: dict[str, pd.DataFrame] = {}
    for etf, df in panel.items():
        last = dmap.get(_norm_sym(etf))
        if last is None or df is None or df.empty:
            out[etf] = df
            continue
        frame = df.copy()
        frame.index = pd.DatetimeIndex(pd.to_datetime(frame.index)).normalize()
        out[etf] = frame.loc[frame.index <= last]
    return out


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
    underlying_by_etf: Mapping[str, str] | None = None,
    required_etfs: set[str] | None = None,
    min_days_by_etf: Mapping[str, int] | None = None,
    repo: Path | None = None,
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
    und_map = {_norm_sym(k): _norm_sym(v) for k, v in (underlying_by_etf or {}).items()}
    required = {_norm_sym(x) for x in (required_etfs or set())}
    per_etf_min = {_norm_sym(k): max(int(v), 2) for k, v in (min_days_by_etf or {}).items()}
    root = repo or REPO

    out: dict[str, pd.DataFrame] = {}
    for etf, g in frame.groupby(ticker_col):
        if required and etf not in required:
            continue
        required_days = per_etf_min.get(etf, min_days)
        # Keep ETF prints even when underlying is briefly NaN (common near the
        # metrics vendor cutoff). Forward-fill underlying for short gaps so the
        # sim calendar matches the book through the run end; long und holes are
        # still NaN and get no return that day.
        g = g.dropna(subset=[etf_col])
        if len(g) < required_days:
            continue
        idx = pd.DatetimeIndex(g[date_col])
        a = pd.Series(g[etf_col].to_numpy(dtype=float), index=idx)
        b = pd.Series(pd.to_numeric(g[und_col], errors="coerce").to_numpy(dtype=float), index=idx)
        a = a[~a.index.duplicated(keep="last")]
        b = b[~b.index.duplicated(keep="last")].reindex(a.index)
        b = b.ffill(limit=5)
        # Drop leading und-NaN (no prior print to ffill from).
        both = pd.DataFrame({"a_px": a, "b_px": b}).dropna(subset=["a_px"])
        both = both.dropna(subset=["b_px"])
        if len(both) < required_days:
            continue
        a = both["a_px"]
        b = both["b_px"]
        if apply_splits:
            a = apply_flex_splits_to_series(a, str(etf), split_map=smap)
            if root != REPO:
                a = apply_price_patches(a, str(etf), repo=root)
        else:
            a = apply_price_patches(a, str(etf), repo=root)
        # Underlying patches (Yahoo referee) — ETF-only patches miss CVNA/IREN/ONDS-class bugs.
        und = und_map.get(_norm_sym(etf), "")
        if und:
            b = apply_price_patches(b, und, repo=root)
        # Live Yahoo referee is opt-in (slow). Bulk repairs live in
        # data/price_patches.csv from ``python -m scripts.price_integrity_audit --write-patches``.
        df = pd.DataFrame(
            {"a_px": a.to_numpy(dtype=float), "b_px": b.reindex(a.index).to_numpy(dtype=float)},
            index=a.index,
        ).dropna()
        if len(df) < required_days:
            continue
        out[str(etf)] = df
    return out


def apply_panel_leg_patches(
    panel: dict[str, pd.DataFrame],
    *,
    underlying_by_etf: Mapping[str, str] | None = None,
    repo: Path | None = None,
) -> dict[str, pd.DataFrame]:
    """Re-apply ``data/price_patches.csv`` to ETF (a_px) and underlying (b_px) legs."""
    und_map = {_norm_sym(k): _norm_sym(v) for k, v in (underlying_by_etf or {}).items()}
    root = repo or REPO
    out: dict[str, pd.DataFrame] = {}
    for etf, df in panel.items():
        if df is None or df.empty or "a_px" not in df.columns:
            out[etf] = df
            continue
        frame = df.copy()
        frame.index = pd.DatetimeIndex(pd.to_datetime(frame.index)).normalize()
        a = apply_price_patches(frame["a_px"], str(etf), repo=root)
        und = und_map.get(_norm_sym(etf), "")
        if und and "b_px" in frame.columns:
            b = apply_price_patches(frame["b_px"], und, repo=root)
        else:
            b = frame["b_px"] if "b_px" in frame.columns else pd.Series(dtype=float)
        joined = pd.DataFrame({"a_px": a, "b_px": b}).dropna().sort_index()
        out[etf] = joined if not joined.empty else frame
    return out


def _yahoo_close(symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    """Adjusted close from cache/Yahoo; empty Series on failure."""
    try:
        from scripts.bucket4_price_loading import load_single_close

        s = load_single_close(
            _norm_sym(symbol).replace("-", "."),
            start=str(start.date()),
            end=str((end + pd.Timedelta(days=1)).date()),
        )
        if s is None or len(s) == 0:
            return pd.Series(dtype=float)
        s = pd.to_numeric(s, errors="coerce")
        s.index = pd.DatetimeIndex(pd.to_datetime(s.index)).tz_localize(None).normalize()
        return s[~s.index.duplicated(keep="last")].sort_index().dropna()
    except Exception:
        return pd.Series(dtype=float)


def _align_extend(base: pd.Series, ext: pd.Series) -> pd.Series:
    """Append ``ext`` after ``base``, level-matched at the join so returns are continuous."""
    base = pd.to_numeric(base, errors="coerce").dropna().sort_index()
    ext = pd.to_numeric(ext, errors="coerce").dropna().sort_index()
    if base.empty:
        return ext
    if ext.empty:
        return base
    last = pd.Timestamp(base.index.max())
    tail = ext.loc[ext.index > last]
    if tail.empty:
        return base
    # Prefer an overlapping anchor day; else scale first tail print to last base.
    overlap = ext.index.intersection(base.index)
    if len(overlap):
        anchor = pd.Timestamp(overlap.max())
        b0 = float(base.loc[anchor])
        e0 = float(ext.loc[anchor])
        if e0 > 0 and b0 > 0:
            tail = tail * (b0 / e0)
    else:
        e0 = float(tail.iloc[0])
        b0 = float(base.iloc[-1])
        if e0 > 0 and b0 > 0:
            # Scale so first extended print matches last metrics level.
            # (Return on the join day will be ~0; subsequent days are Yahoo returns.)
            tail = tail * (b0 / e0)
            # Drop the first extended bar if it only restates the last close.
            if abs(float(tail.iloc[0]) / b0 - 1.0) < 1e-6:
                tail = tail.iloc[1:]
    if tail.empty:
        return base
    out = pd.concat([base, tail])
    return out[~out.index.duplicated(keep="last")].sort_index()


def _beta_etf_vs_und(a: pd.Series, b: pd.Series, *, lookback: int = 40) -> float:
    """OLS daily beta of ETF vs und over the last ``lookback`` overlapping days."""
    both = pd.DataFrame({"a": a, "b": b}).dropna().sort_index()
    if len(both) < 5:
        return 0.0
    rets = both.pct_change().dropna().iloc[-int(lookback) :]
    if len(rets) < 5:
        return 0.0
    var = float((rets["b"] ** 2).sum())
    if var <= 0:
        return 0.0
    beta = float((rets["a"] * rets["b"]).sum() / var)
    return max(-5.0, min(5.0, beta))


def _synthesize_etf_from_und(a: pd.Series, b: pd.Series) -> pd.Series:
    """Roll ETF levels forward on und dates past the last ETF print using recent beta.

    Needed when Yahoo stops quoting the ETF (or metrics drop it) while the
    underlying keeps trading — otherwise an inner join freezes the book.
    """
    a = pd.to_numeric(a, errors="coerce").dropna().sort_index()
    b = pd.to_numeric(b, errors="coerce").dropna().sort_index()
    if a.empty or b.empty:
        return a
    last_a = pd.Timestamp(a.index.max())
    b_tail = b.loc[b.index > last_a]
    if b_tail.empty:
        return a
    beta = _beta_etf_vs_und(a, b)
    b_hist = b.loc[b.index <= last_a]
    if b_hist.empty:
        return a
    prev_b = float(b_hist.iloc[-1])
    prev_a = float(a.iloc[-1])
    syn_idx: list[pd.Timestamp] = []
    syn_vals: list[float] = []
    for dt, bx in b_tail.items():
        bx = float(bx)
        if prev_b > 0 and bx > 0:
            r_b = bx / prev_b - 1.0
            prev_a = prev_a * (1.0 + beta * r_b)
            syn_idx.append(pd.Timestamp(dt))
            syn_vals.append(prev_a)
        prev_b = bx
    if not syn_idx:
        filled = a.reindex(a.index.union(b_tail.index)).ffill()
        return filled.dropna()
    syn = pd.Series(syn_vals, index=pd.DatetimeIndex(syn_idx))
    out = pd.concat([a, syn])
    return out[~out.index.duplicated(keep="last")].sort_index()

def underlying_map_from_run(
    run_date: str,
    *,
    repo: Path | None = None,
) -> dict[str, str]:
    """Best-effort ETF→underlying map from screened / proposed trades for ``run_date``."""
    root = repo or REPO
    out: dict[str, str] = {}
    for rel in (
        f"data/runs/{run_date}/etf_screened_today.csv",
        f"data/runs/{run_date}/proposed_trades.csv",
        "notebooks/output/production_actual_bt/pair_stats.csv",
    ):
        p = root / rel
        if not p.is_file():
            continue
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        etf_col = "ETF" if "ETF" in df.columns else ("ticker" if "ticker" in df.columns else None)
        und_col = "Underlying" if "Underlying" in df.columns else None
        if etf_col is None or und_col is None:
            continue
        for _, r in df.iterrows():
            etf = _norm_sym(r.get(etf_col))
            und = _norm_sym(r.get(und_col))
            if etf and und and etf not in out:
                out[etf] = und
    return out


def extend_price_panel_yahoo(
    panel: dict[str, pd.DataFrame],
    *,
    underlying_by_etf: Mapping[str, str] | None = None,
    end: str | pd.Timestamp | None = None,
    min_gap_days: int = 2,
) -> dict[str, pd.DataFrame]:
    """Extend truncated metrics panels with Yahoo so the sim marks through ``end``.

    Metrics often go NaN mid-window (e.g. CLSZ und missing after mid-June). Without
    this, share-held books freeze (0 price PnL) while diagnostic charts still move
    on Yahoo. Level-match at the join so we do not invent a one-day jump.
    """
    if not panel:
        return panel
    end_ts = pd.Timestamp(end).normalize() if end is not None else pd.Timestamp.today().normalize()
    und_map = { _norm_sym(k): _norm_sym(v) for k, v in (underlying_by_etf or {}).items() }
    out: dict[str, pd.DataFrame] = {}
    for etf, df in panel.items():
        if df is None or df.empty or "a_px" not in df.columns or "b_px" not in df.columns:
            out[etf] = df
            continue
        frame = df.copy()
        frame.index = pd.DatetimeIndex(pd.to_datetime(frame.index)).normalize()
        frame = frame[~frame.index.duplicated(keep="last")].sort_index()
        last = pd.Timestamp(frame.index.max())
        if (end_ts - last).days < int(min_gap_days):
            out[etf] = frame
            continue
        start = last - pd.Timedelta(days=14)
        a = _align_extend(
            frame["a_px"],
            _yahoo_close(etf, start, end_ts),
        )
        und = und_map.get(_norm_sym(etf), "")
        b_src = _yahoo_close(und, start, end_ts) if und else pd.Series(dtype=float)
        b = _align_extend(frame["b_px"], b_src) if len(b_src) else frame["b_px"].copy()
        # If und extends past ETF quotes, synthesize ETF from und returns (beta).
        if len(b.dropna()) and (
            a.dropna().empty or pd.Timestamp(b.dropna().index.max()) > pd.Timestamp(a.dropna().index.max())
        ):
            a = _synthesize_etf_from_und(a, b)
        joined = pd.DataFrame({"a_px": a, "b_px": b}).dropna().sort_index()
        if len(joined) < len(frame):
            # Extension failed; keep original.
            out[etf] = frame
        else:
            out[etf] = joined
    return out


def load_run_price_panel(
    run_date: str,
    *,
    repo: Path | None = None,
    min_days: int = MIN_PRICE_PANEL_DAYS,
    extend_yahoo: bool = True,
    extend_to: str | pd.Timestamp | None = None,
    underlying_by_etf: Mapping[str, str] | None = None,
    required_etfs: set[str] | None = None,
    min_days_by_etf: Mapping[str, int] | None = None,
    apply_delist_cut: bool = True,
) -> dict[str, pd.DataFrame]:
    """Load split-adjusted panels from ``data/runs/<date>/model_inputs/etf_metrics_daily.parquet``.

    Applies flex/overrides/patches/Yahoo-referee, optional Yahoo tail extend, then
    truncates at ``data/delistings.csv`` last_trade_date.
    """
    root = repo or REPO
    pq = root / f"data/runs/{run_date}/model_inputs/etf_metrics_daily.parquet"
    md = pd.read_parquet(pq, columns=["date", "ticker", "etf_adj_close", "underlying_adj_close"])
    und_map = dict(underlying_by_etf or {})
    if not und_map:
        und_map = underlying_map_from_run(run_date, repo=root)
    panel = frames_from_metrics(
        md,
        min_days=min_days,
        underlying_by_etf=und_map,
        required_etfs=required_etfs,
        min_days_by_etf=min_days_by_etf,
        repo=root,
    )
    if extend_yahoo:
        end = extend_to if extend_to is not None else run_date
        try:
            end_ts = max(pd.Timestamp(end), pd.Timestamp(run_date))
        except Exception:
            end_ts = pd.Timestamp(run_date)
        panel = extend_price_panel_yahoo(panel, underlying_by_etf=und_map, end=end_ts)
        # Re-apply patches after Yahoo extend so und/ETF referee rows stay sticky.
        panel = apply_panel_leg_patches(panel, underlying_by_etf=und_map, repo=root)
    else:
        panel = apply_panel_leg_patches(panel, underlying_by_etf=und_map, repo=root)
    if apply_delist_cut:
        panel = apply_delist_cutoff(panel, repo=root)
    return panel
