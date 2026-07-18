"""B4 path diagnostics for the production-actual backtest notebook.

Builds EOD-style multipanel charts from notebook artifacts
(``pair_daily_pnl.csv`` + price panel + production cadence knobs).
Optionally tops up missing underlying prints via Yahoo for *plotting only*.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.bucket4_hedge_cadence import build_h_series, build_rebal_dates
from scripts.bucket4_vol_shape_signals import get_pair_signal
from scripts.sizing_tilt_cadence_bt import knobs_from_yaml, make_knobs

B4_SLEEVE = "inverse_decay_bucket4"
B5_SLEEVE = "volatility_etp_bucket5"
REPO = Path(__file__).resolve().parents[1]


def _norm(x: object) -> str:
    return str(x).strip().upper().replace(".", "-")


def select_extreme_b4_pairs(
    pair_stats: pd.DataFrame,
    *,
    n_top: int = 3,
    n_bottom: int = 2,
    sleeves: Sequence[str] = (B4_SLEEVE,),
    mode: str = "extremes",
) -> pd.DataFrame:
    if pair_stats is None or pair_stats.empty:
        return pd.DataFrame()
    df = pair_stats.copy()
    if "sleeve" not in df.columns or "pnl_usd" not in df.columns:
        return pd.DataFrame()
    df = df[df["sleeve"].astype(str).isin(set(sleeves))].copy()
    if df.empty:
        return df
    df["ETF"] = df["ETF"].map(_norm)
    df["Underlying"] = df["Underlying"].map(_norm)
    df = df.sort_values("pnl_usd", ascending=False)
    mode_n = str(mode or "extremes").strip().lower()
    if mode_n in {"all", "full", "*"}:
        return df.reset_index(drop=True)
    top = df.head(max(0, int(n_top)))
    bot = df.tail(max(0, int(n_bottom)))
    out = pd.concat([top, bot], axis=0)
    return out[~out.index.duplicated(keep="first")].reset_index(drop=True)


def _days_to_rebal_series(
    calendar: pd.DatetimeIndex,
    rebal_dates: pd.DatetimeIndex,
) -> pd.Series:
    cal = pd.DatetimeIndex(calendar).sort_values()
    rb = pd.DatetimeIndex(rebal_dates).sort_values()
    if len(cal) == 0:
        return pd.Series(dtype=float)
    if len(rb) == 0:
        return pd.Series(np.nan, index=cal)
    out = pd.Series(index=cal, dtype=float)
    j = 0
    for d in cal:
        while j < len(rb) and rb[j] < d:
            j += 1
        out.loc[d] = float("nan") if j >= len(rb) else float((rb[j] - d).days)
    return out


def _yahoo_close(symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    """Best-effort Yahoo close for chart fill; empty on failure."""
    try:
        from scripts.bucket4_price_loading import load_single_close

        s = load_single_close(symbol, start=str(start.date()), end=str(end.date()))
        if s is not None and len(s):
            s = pd.to_numeric(s, errors="coerce")
            s.index = pd.DatetimeIndex(pd.to_datetime(s.index)).normalize()
            return s[~s.index.duplicated(keep="last")].sort_index()
    except Exception:
        pass
    try:
        import yfinance as yf
    except ImportError:
        return pd.Series(dtype=float)
    try:
        raw = yf.download(
            symbol,
            start=str(start.date()),
            end=str((end + pd.Timedelta(days=1)).date()),
            progress=False,
            auto_adjust=True,
        )
        if raw is None or raw.empty:
            return pd.Series(dtype=float)
        col = "Close" if "Close" in raw.columns else raw.columns[0]
        s = pd.to_numeric(raw[col], errors="coerce")
        s.index = pd.DatetimeIndex(pd.to_datetime(s.index)).tz_localize(None).normalize()
        return s[~s.index.duplicated(keep="last")].sort_index().rename(symbol)
    except Exception:
        return pd.Series(dtype=float)


def _extend_underlying(
    und_px: pd.Series,
    *,
    und_sym: str,
    book_index: pd.DatetimeIndex,
    fill_yahoo: bool,
) -> tuple[pd.Series, str, pd.Timestamp | None]:
    """Return und series covering book window; note panel_end and fill source."""
    s = pd.to_numeric(und_px, errors="coerce").copy()
    if not isinstance(s.index, pd.DatetimeIndex):
        s.index = pd.to_datetime(s.index, errors="coerce")
    s.index = pd.DatetimeIndex(s.index).normalize()
    s = s[~s.index.duplicated(keep="last")].sort_index()
    finite = s.dropna()
    panel_end = pd.Timestamp(finite.index.max()) if len(finite) else None
    source = "metrics"
    need_end = pd.Timestamp(book_index.max()) if len(book_index) else panel_end
    need_start = pd.Timestamp(book_index.min()) if len(book_index) else (panel_end or pd.Timestamp("2025-01-01"))
    if fill_yahoo and need_end is not None and (
        panel_end is None or panel_end < need_end - pd.Timedelta(days=2) or s.reindex(book_index).isna().mean() > 0.05
    ):
        y = _yahoo_close(und_sym, need_start - pd.Timedelta(days=90), need_end)
        if len(y):
            # Prefer metrics where finite; fill holes / extend with Yahoo.
            merged = s.combine_first(y)
            if panel_end is not None:
                # After panel_end, allow Yahoo exclusively.
                post = y.loc[y.index > panel_end]
                merged = merged.combine_first(post)
            s = merged
            source = "metrics+yahoo" if panel_end is not None else "yahoo"
    # Light ffill only for interior 1–3 day holes (holidays), not long tails.
    s = s.reindex(s.index.union(book_index)).sort_index()
    s = s.ffill(limit=3)
    return s, source, panel_end


def _warmup_bdays(knobs: Any, cal_full: pd.DatetimeIndex, sig: pd.DataFrame) -> int:
    """Warmup on the *full* panel calendar from YAML ``warmup_bdays`` (default 65)."""
    del knobs, sig  # knobs used by callers; signal-based clamp not needed on full cal
    cfg = 65
    try:
        from strategy_config import load_config

        raw = load_config(REPO / "config" / "strategy_config.yml")
        opt2 = (
            ((raw.get("portfolio") or {}).get("sleeves") or {})
            .get("inverse_decay_bucket4", {})
            .get("rules", {})
            .get("bucket4_weekly_opt2", {})
            or {}
        )
        cfg = int(opt2.get("warmup_bdays", 65) or 65)
    except Exception:
        pass
    return max(0, min(cfg, max(0, len(cal_full) - 5)))


def build_pair_path_bundle(
    *,
    etf: str,
    underlying: str,
    pair_daily: pd.DataFrame,
    prices: pd.DataFrame | None,
    start: str | pd.Timestamp | None = None,
    fill_yahoo: bool = True,
    plan_h: float | None = None,
) -> dict[str, Any]:
    """Join book path + model hedge / cadence for one pair."""
    etf_n, und_n = _norm(etf), _norm(underlying)
    daily = pair_daily.copy()
    if "ETF" in daily.columns:
        daily = daily[daily["ETF"].map(_norm) == etf_n].copy()
    if daily.empty:
        return {"ok": False, "reason": "no_pair_daily", "etf": etf_n, "underlying": und_n}

    daily["date"] = pd.to_datetime(daily["date"], errors="coerce").dt.normalize()
    daily = daily.dropna(subset=["date"]).sort_values("date")
    if start is not None:
        daily = daily[daily["date"] >= pd.Timestamp(start)]
    if daily.empty:
        return {"ok": False, "reason": "empty_after_start", "etf": etf_n, "underlying": und_n}

    book = daily.set_index("date")
    book_idx = pd.DatetimeIndex(book.index)
    book_h = pd.to_numeric(book.get("hedge_ratio"), errors="coerce")
    cum_pnl = pd.to_numeric(book.get("cum_pnl"), errors="coerce")
    if cum_pnl.isna().all() and "daily_pnl" in book.columns:
        cum_pnl = pd.to_numeric(book["daily_pnl"], errors="coerce").fillna(0.0).cumsum()
    etf_usd = pd.to_numeric(book.get("etf_usd"), errors="coerce")
    und_usd = pd.to_numeric(book.get("underlying_usd"), errors="coerce")
    book_rebals = book_idx[
        pd.to_numeric(book.get("is_rebalance"), errors="coerce").fillna(0).astype(int) > 0
    ]

    model_h = pd.Series(dtype=float)
    model_h_valid = pd.Series(dtype=bool)
    model_rebals = pd.DatetimeIndex([])
    days_to = pd.Series(dtype=float)
    tr = pd.Series(dtype=float)
    und_px_book = pd.Series(dtype=float)
    etf_px_book = pd.Series(dtype=float)
    reason = ""
    price_source = "none"
    panel_end: pd.Timestamp | None = None
    signal_ok_mask = pd.Series(False, index=book_idx)

    if prices is None or prices.empty:
        reason = "no_price_panel"
    else:
        px = prices.copy()
        if not isinstance(px.index, pd.DatetimeIndex):
            px.index = pd.to_datetime(px.index, errors="coerce")
        px = px[~px.index.isna()].sort_index()
        px.index = pd.DatetimeIndex(px.index).normalize()
        und_raw = pd.to_numeric(px.get("b_px"), errors="coerce")
        etf_raw = pd.to_numeric(px.get("a_px"), errors="coerce")
        und_full, price_source, panel_end = _extend_underlying(
            und_raw, und_sym=und_n, book_index=book_idx, fill_yahoo=fill_yahoo
        )
        # Cadence on FULL underlying history calendar (warmup), not book-only.
        cal_full = pd.DatetimeIndex(und_full.dropna().index).sort_values().unique()
        if start is not None:
            # Keep pre-start warmup bars.
            pass
        try:
            blk = knobs_from_yaml()
            knobs = make_knobs(blk)
            sig_full = get_pair_signal(
                etf_n,
                und_n,
                cal_full,
                history={},
                underlying_prices=und_full,
                window=60,
                lookahead_shift=1,
            )
            model_h_full = build_h_series(sig_full, cal_full, knobs=knobs)
            warmup = _warmup_bdays(knobs, cal_full, sig_full)
            model_rebals_full, _diag = build_rebal_dates(
                sig_full, cal_full, knobs=knobs, warmup_bdays=warmup
            )
            # Align to book window.
            model_h = model_h_full.reindex(book_idx).ffill()
            model_rebals = pd.DatetimeIndex(
                [d for d in model_rebals_full if book_idx.min() <= d <= book_idx.max()]
            )
            days_to = _days_to_rebal_series(book_idx, model_rebals_full)
            col = str(getattr(knobs, "cadence_signal_col", "tr_est") or "tr_est")
            if sig_full is not None and not sig_full.empty:
                if col not in sig_full.columns:
                    col = "tr" if "tr" in sig_full.columns else None
                if col:
                    tr = pd.to_numeric(sig_full[col], errors="coerce").reindex(book_idx)
                    signal_ok_mask = tr.notna()
            model_h_valid = signal_ok_mask.reindex(book_idx).fillna(False)
            # Where signal missing, blank model_h for plotting (keep separate fallback series).
            und_px_book = und_full.reindex(book_idx)
            etf_px_book = etf_raw.reindex(book_idx).ffill(limit=3)
        except Exception as exc:  # noqa: BLE001
            reason = f"cadence_failed:{type(exc).__name__}:{exc}"
            und_px_book = und_full.reindex(book_idx)
            etf_px_book = etf_raw.reindex(book_idx) if len(etf_raw) else etf_px_book

    risk: dict[str, float] = {}
    if cum_pnl.notna().sum() >= 5:
        rets = cum_pnl.diff().fillna(0.0)
        vol = float(rets.std(ddof=0) * np.sqrt(252.0)) if len(rets) > 1 else float("nan")
        peak = cum_pnl.cummax()
        dd = (
            float(((cum_pnl - peak) / peak.abs().clip(lower=1.0)).min())
            if len(cum_pnl)
            else float("nan")
        )
        risk = {
            "end_pnl": float(cum_pnl.iloc[-1]),
            "vol_proxy": vol,
            "max_dd_proxy": dd,
            "n_days": int(len(cum_pnl)),
        }

    tr_cov = float(signal_ok_mask.mean()) if len(signal_ok_mask) else 0.0
    px_cov = float(und_px_book.notna().mean()) if len(und_px_book) else 0.0

    return {
        "ok": True,
        "reason": reason,
        "etf": etf_n,
        "underlying": und_n,
        "book_h": book_h,
        "model_h": model_h,
        "model_h_valid": model_h_valid,
        "plan_h": float(plan_h) if plan_h is not None and np.isfinite(plan_h) else float("nan"),
        "cum_pnl": cum_pnl,
        "etf_usd": etf_usd,
        "und_usd": und_usd,
        "book_rebals": pd.DatetimeIndex(book_rebals),
        "model_rebals": pd.DatetimeIndex(model_rebals),
        "days_to_rebal": days_to,
        "tr": tr,
        "und_px": und_px_book,
        "etf_px": etf_px_book,
        "signal_ok": signal_ok_mask,
        "price_source": price_source,
        "panel_end": panel_end,
        "tr_coverage": tr_cov,
        "price_coverage": px_cov,
        "risk": risk,
    }


def _mark_rebals(ax, dates: pd.DatetimeIndex, *, color: str, label: str, ls: str = "--") -> None:
    if dates is None or len(dates) == 0:
        return
    labeled = False
    for d in dates:
        ax.axvline(
            d,
            color=color,
            alpha=0.35,
            lw=0.9,
            ls=ls,
            label=(label if not labeled else None),
        )
        labeled = True


def _shade_mask(ax, index: pd.DatetimeIndex, mask: pd.Series, *, color: str, alpha: float = 0.12) -> None:
    if mask is None or len(mask) == 0 or not mask.any():
        return
    m = mask.reindex(index).fillna(False).astype(bool)
    if not m.any():
        return
    # Shade contiguous True regions.
    vals = m.to_numpy()
    idx = list(index)
    start = None
    for i, flag in enumerate(vals):
        if flag and start is None:
            start = idx[i]
        if (not flag or i == len(vals) - 1) and start is not None:
            end = idx[i] if flag and i == len(vals) - 1 else idx[i]
            ax.axvspan(start, end, color=color, alpha=alpha, lw=0)
            start = None


def plot_b4_pair_path(
    bundle: dict[str, Any],
    *,
    sleeve_label: str = "B4",
    figsize: tuple[float, float] = (12.5, 10.8),
) -> plt.Figure | None:
    if not bundle.get("ok"):
        return None

    etf, und = bundle["etf"], bundle["underlying"]
    cum = bundle["cum_pnl"]
    etf_usd, und_usd = bundle["etf_usd"], bundle["und_usd"]
    book_h = bundle["book_h"]
    model_h = bundle["model_h"]
    model_ok = bundle.get("model_h_valid")
    days_to = bundle["days_to_rebal"]
    tr = bundle["tr"]
    und_px = bundle["und_px"]
    risk = bundle.get("risk") or {}
    signal_ok = bundle.get("signal_ok")
    panel_end = bundle.get("panel_end")

    fig, axes = plt.subplots(
        5,
        1,
        figsize=figsize,
        sharex=True,
        gridspec_kw={"height_ratios": [1.15, 1.0, 1.15, 0.9, 1.05]},
    )

    # 1) Cum PnL
    ax = axes[0]
    ax.plot(cum.index, cum.values, color="#1f77b4", lw=1.6, label="sim cum PnL")
    _mark_rebals(ax, bundle["book_rebals"], color="#111827", label="book rebal (W-FRI)", ls="-")
    _mark_rebals(ax, bundle["model_rebals"], color="#f59e0b", label="model cadence", ls="--")
    ax.axhline(0.0, color="#9ca3af", lw=0.8)
    ax.set_ylabel("PnL ($)")
    ax.set_title(f"{sleeve_label} {etf}/{und} — path diagnostics (sim book)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    hdr = (
        f"end PnL ${risk.get('end_pnl', float('nan')):,.0f}  |  "
        f"vol≈{risk.get('vol_proxy', float('nan')):,.0f}  |  "
        f"maxDD≈{risk.get('max_dd_proxy', float('nan')):.1%}  |  "
        f"prices {bundle.get('price_coverage', 0):.0%}  |  "
        f"TR {bundle.get('tr_coverage', 0):.0%}  |  "
        f"src={bundle.get('price_source', '')}  |  "
        f"model rebals={len(bundle.get('model_rebals') if bundle.get('model_rebals') is not None else [])}  "
        f"book rebals={len(bundle.get('book_rebals') if bundle.get('book_rebals') is not None else [])}"
    )
    if bundle.get("reason"):
        hdr += f"  |  note={bundle['reason']}"
    ax.text(0.01, 0.02, hdr, transform=ax.transAxes, fontsize=7.5, color="#334155", family="monospace")

    # 2) Gross
    ax = axes[1]
    ax.plot(etf_usd.index, etf_usd.abs().values, color="#c44e52", lw=1.3, label="|ETF|")
    ax.plot(und_usd.index, und_usd.abs().values, color="#4c72b0", lw=1.3, label="|underlying|")
    _mark_rebals(ax, bundle["book_rebals"], color="#111827", label="", ls="-")
    ax.set_ylabel("Gross ($)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)

    # 3) Hedge
    ax = axes[2]
    if len(book_h):
        ax.plot(book_h.index, book_h.values, color="#1f77b4", lw=1.5, label="book h")
    if len(model_h):
        if model_ok is not None and len(model_ok):
            mh = model_h.where(model_ok.reindex(model_h.index).fillna(False))
            ax.plot(mh.index, mh.values, color="#d62728", lw=1.4, label="model h (signal ok)")
            # Fallback h_mid region in gray
            fb = model_h.where(~model_ok.reindex(model_h.index).fillna(False))
            if fb.notna().any():
                ax.plot(fb.index, fb.values, color="#9ca3af", lw=1.0, ls=":", label="model h (no signal / h_mid)")
        else:
            ax.plot(model_h.index, model_h.values, color="#d62728", lw=1.4, label="model h")
    plan_h = bundle.get("plan_h")
    if plan_h is not None and np.isfinite(plan_h):
        ax.axhline(float(plan_h), color="#7e57c2", lw=1.0, ls="-.", label=f"plan h={float(plan_h):.3f}")
    if signal_ok is not None:
        _shade_mask(ax, book_h.index, ~signal_ok.astype(bool), color="#f59e0b", alpha=0.10)
    _mark_rebals(ax, bundle["book_rebals"], color="#111827", label="", ls="-")
    _mark_rebals(ax, bundle["model_rebals"], color="#f59e0b", label="", ls="--")
    ax.set_ylabel("h")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)

    # 4) Days + TR
    ax = axes[3]
    if len(days_to):
        ax.step(days_to.index, days_to.values, where="post", color="#7f7f7f", lw=1.2, label="days→model rebal")
    ax.set_ylabel("days")
    ax2 = ax.twinx()
    if len(tr):
        ax2.plot(tr.index, tr.values, color="#2ca02c", alpha=0.8, lw=1.1, label="cadence signal")
        ax2.set_ylabel("TR / signal", color="#2ca02c")
    if signal_ok is not None:
        _shade_mask(ax, days_to.index if len(days_to) else tr.index, ~signal_ok.astype(bool), color="#f59e0b", alpha=0.10)
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    if lines or lines2:
        ax.legend(lines + lines2, labels + labels2, loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)

    # 5) Underlying price
    ax = axes[4]
    if len(und_px) and und_px.notna().any():
        ax.plot(und_px.index, und_px.values, color="#4c72b0", lw=1.3, label=f"{und} ({bundle.get('price_source')})")
    if panel_end is not None:
        ax.axvline(panel_end, color="#b91c1c", lw=1.2, ls=":", label=f"metrics panel ends {panel_end.date()}")
        # Shade post-panel if using yahoo extension
        if str(bundle.get("price_source", "")).endswith("yahoo"):
            ax.axvspan(panel_end, und_px.index.max(), color="#b91c1c", alpha=0.06, lw=0)
    miss = und_px.isna() if len(und_px) else pd.Series(dtype=bool)
    if len(miss) and miss.any():
        _shade_mask(ax, und_px.index, miss, color="#64748b", alpha=0.15)
    _mark_rebals(ax, bundle["book_rebals"], color="#111827", label="book rebal", ls="-")
    _mark_rebals(ax, bundle["model_rebals"], color="#f59e0b", label="model cadence", ls="--")
    ax.set_ylabel("Price")
    ax.set_xlabel("Date")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    fig.tight_layout()
    return fig


def plot_b4_path_gallery(
    *,
    pair_stats: pd.DataFrame,
    pair_daily: pd.DataFrame,
    panel: dict[str, pd.DataFrame],
    start: str | pd.Timestamp | None = None,
    n_top: int = 3,
    n_bottom: int = 2,
    sleeves: Sequence[str] = (B4_SLEEVE,),
    fill_yahoo: bool = True,
    mode: str = "extremes",
    pdf_path: Path | str | None = None,
    show: bool = True,
) -> list[dict[str, Any]]:
    picks = select_extreme_b4_pairs(
        pair_stats, n_top=n_top, n_bottom=n_bottom, sleeves=sleeves, mode=mode
    )
    meta: list[dict[str, Any]] = []
    if picks.empty:
        return meta

    daily = pair_daily.copy()
    daily["ETF"] = daily["ETF"].map(_norm)
    daily["date"] = pd.to_datetime(daily["date"], errors="coerce").dt.normalize()

    pdf = None
    if pdf_path is not None:
        from matplotlib.backends.backend_pdf import PdfPages

        pdf = PdfPages(str(pdf_path))

    try:
        for _, row in picks.iterrows():
            etf = _norm(row["ETF"])
            und = _norm(row.get("Underlying", ""))
            sleeve = str(row.get("sleeve", B4_SLEEVE))
            label = "B4" if sleeve == B4_SLEEVE else ("B5" if sleeve == B5_SLEEVE else sleeve)
            plan_h = pd.to_numeric(row.get("hedge_ratio"), errors="coerce")
            px = panel.get(etf)
            bundle = build_pair_path_bundle(
                etf=etf,
                underlying=und,
                pair_daily=daily[daily["ETF"] == etf],
                prices=px,
                start=start,
                fill_yahoo=fill_yahoo,
                plan_h=float(plan_h) if pd.notna(plan_h) else None,
            )
            fig = plot_b4_pair_path(bundle, sleeve_label=label)
            meta.append(
                {
                    "ETF": etf,
                    "Underlying": und,
                    "sleeve": sleeve,
                    "pnl_usd": float(pd.to_numeric(row.get("pnl_usd"), errors="coerce") or np.nan),
                    "ok": bool(bundle.get("ok")),
                    "reason": bundle.get("reason", ""),
                    "price_source": bundle.get("price_source", ""),
                    "tr_coverage": bundle.get("tr_coverage", 0.0),
                    "price_coverage": bundle.get("price_coverage", 0.0),
                    "n_model_rebals": int(
                        len(bundle["model_rebals"]) if bundle.get("model_rebals") is not None else 0
                    ),
                    "plotted": fig is not None,
                }
            )
            if fig is not None:
                if pdf is not None:
                    pdf.savefig(fig)
                if show:
                    try:
                        plt.show()
                    except Exception:
                        pass
                plt.close(fig)
    finally:
        if pdf is not None:
            pdf.close()
    return meta
