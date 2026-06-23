"""Per-pair Bucket 4 PnL, hedge, and rebalance charts for the EOD email.

The EOD email calls :func:`make_b4_pair_pnl_hedge_chart`.  It is intentionally
fail-soft: bad/missing optional data skips the chart instead of blocking the
email.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

DEFAULT_METRICS_CSV = REPO.parent / "Levered ETFs" / "etf-dashboard" / "data" / "etf_metrics_daily.csv"
DEFAULT_VOL_SHAPE_JSON = REPO.parent / "Levered ETFs" / "etf-dashboard" / "data" / "vol_shape_history.json"
MODEL_START = "2025-10-07"
SIGNAL_WINDOW = 45


def _norm(x: object) -> str:
    return str(x).strip().upper().replace(".", "-")


def _pair_key(etf: object, underlying: object) -> str:
    return f"{_norm(etf)}|{_norm(underlying)}"


def _scalar_float(value: object, default: float = 0.0) -> float:
    try:
        out = pd.to_numeric(value, errors="coerce")
        if pd.isna(out):
            return default
        return float(out)
    except Exception:
        return default


def _has_conflict_markers(path: Path) -> bool:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return False
    return any(marker in text for marker in ("<<<<<<<", "=======", ">>>>>>>"))


def _resolve_proposed_trades_path(run_date: str, runs_root: Path) -> Path | None:
    dated = runs_root / run_date / "proposed_trades.csv"
    if dated.is_file():
        return dated
    latest = REPO / "data" / "proposed_trades.csv"
    return latest if latest.is_file() else None


def load_active_b4_pairs_from_proposed(
    run_date: str,
    *,
    runs_root: Path,
    min_gross_usd: float = 0.0,
) -> pd.DataFrame:
    """Active B4 pairs from proposed trades, normalized and de-duped.

    Raises ``ValueError`` for malformed inputs so the top-level chart builder can
    fail soft while unit tests can assert the exact behavior.
    """
    path = _resolve_proposed_trades_path(run_date, runs_root)
    if path is None:
        raise ValueError("proposed_trades.csv not found")
    if _has_conflict_markers(path):
        raise ValueError(f"{path} contains merge-conflict markers")
    df = pd.read_csv(path)
    required = {"ETF", "Underlying", "sleeve", "gross_target_usd"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")

    d = df.copy()
    d["etf"] = d["ETF"].map(_norm)
    d["underlying"] = d["Underlying"].map(_norm)
    d["pair"] = [_pair_key(e, u) for e, u in zip(d["etf"], d["underlying"])]
    d["gross_target_usd"] = pd.to_numeric(d["gross_target_usd"], errors="coerce").fillna(0.0)
    sleeve = d["sleeve"].astype(str).str.strip().str.lower()
    d = d[sleeve.eq("inverse_decay_bucket4") & d["gross_target_usd"].gt(float(min_gross_usd))].copy()
    if d.empty:
        return pd.DataFrame(columns=["etf", "underlying", "pair", "gross_target_usd", "delta", "borrow_current"])
    d["delta"] = pd.to_numeric(d.get("Delta", np.nan), errors="coerce")
    d["borrow_current"] = pd.to_numeric(d.get("borrow_current", np.nan), errors="coerce")
    d = d.sort_values("gross_target_usd", ascending=False)
    d = d.drop_duplicates(subset=["etf", "underlying"], keep="first")
    cols = ["etf", "underlying", "pair", "gross_target_usd", "delta", "borrow_current"]
    return d[cols].sort_values("gross_target_usd", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Accounting history
# ---------------------------------------------------------------------------
def list_run_dates(runs_root: Path) -> list[str]:
    if not runs_root.is_dir():
        return []
    out = []
    for d in sorted(runs_root.iterdir()):
        if d.is_dir() and (d / "accounting" / "pnl_bucket_4_by_pair.csv").is_file():
            out.append(d.name)
    return out


def load_b4_pair_leg_history(runs_root: Path) -> pd.DataFrame:
    """Long frame per (date, pair): cumulative pair / ETF-leg / underlying-leg PnL."""
    rows: list[dict] = []
    for ds in list_run_dates(runs_root):
        acct = runs_root / ds / "accounting"
        try:
            by_pair = pd.read_csv(acct / "pnl_bucket_4_by_pair.csv")
            by_sym = pd.read_csv(acct / "pnl_bucket_4_by_symbol.csv")
        except Exception:
            continue
        if by_pair.empty or not {"etf", "underlying", "total_pnl"}.issubset(by_pair.columns):
            continue
        by_pair["etf"] = by_pair["etf"].map(_norm)
        by_pair["underlying"] = by_pair["underlying"].map(_norm)
        if not by_sym.empty and {"symbol", "total_pnl"}.issubset(by_sym.columns):
            sym_pnl = by_sym.assign(symbol=by_sym["symbol"].map(_norm)).set_index("symbol")["total_pnl"].astype(float)
        else:
            sym_pnl = pd.Series(dtype=float)
        for _, r in by_pair.iterrows():
            pair_total = _scalar_float(r.get("total_pnl"), np.nan)
            etf_leg = float(sym_pnl.get(r["etf"], np.nan))
            rows.append(
                {
                    "date": pd.Timestamp(ds),
                    "pair": _pair_key(r["etf"], r["underlying"]),
                    "etf": r["etf"],
                    "underlying": r["underlying"],
                    "delta": float(pd.to_numeric(r.get("delta", np.nan), errors="coerce")),
                    "pair_pnl_cum": pair_total,
                    "etf_leg_pnl_cum": etf_leg,
                    "und_leg_pnl_cum": pair_total - etf_leg if np.isfinite(etf_leg) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def load_pair_gross_and_realized_h(runs_root: Path, run_date: str) -> pd.DataFrame:
    """Per pair on run date: ETF gross and realized hedge ratio from accounting detail."""
    p = runs_root / run_date / "accounting" / "net_exposure_bucket_4_detail.csv"
    by_pair_p = runs_root / run_date / "accounting" / "pnl_bucket_4_by_pair.csv"
    if not (p.is_file() and by_pair_p.is_file()):
        return pd.DataFrame()
    det = pd.read_csv(p)
    needed = {"underlying", "symbol", "leg_type", "gross_notional_usd"}
    if det.empty or not needed.issubset(det.columns):
        return pd.DataFrame()
    det["underlying"] = det["underlying"].map(_norm)
    det["symbol"] = det["symbol"].map(_norm)
    det["gross_notional_usd"] = pd.to_numeric(det["gross_notional_usd"], errors="coerce").fillna(0.0)
    etf_rows = det[det["leg_type"].astype(str).str.lower() == "etf"]
    und_rows = det[det["leg_type"].astype(str).str.lower() == "underlying"]
    und_gross = und_rows.groupby("underlying")["gross_notional_usd"].sum()

    deltas = pd.read_csv(by_pair_p)
    if deltas.empty or "etf" not in deltas.columns:
        return pd.DataFrame()
    deltas["etf"] = deltas["etf"].map(_norm)
    delta_by_etf = pd.to_numeric(deltas.set_index("etf").get("delta", pd.Series(dtype=float)), errors="coerce").abs()

    rows: list[dict] = []
    for u, grp in etf_rows.groupby("underlying"):
        tot_etf = float(grp["gross_notional_usd"].sum())
        ug = float(und_gross.get(u, 0.0))
        for _, r in grp.iterrows():
            eg = float(r["gross_notional_usd"])
            share = eg / tot_etf if tot_etf > 0 else 0.0
            beta = float(delta_by_etf.get(r["symbol"], np.nan))
            h_real = (ug * share) / (beta * eg) if eg > 0 and np.isfinite(beta) and beta > 0 else np.nan
            rows.append(
                {
                    "date": pd.Timestamp(run_date),
                    "etf": r["symbol"],
                    "underlying": u,
                    "pair": _pair_key(r["symbol"], u),
                    "etf_gross_usd": eg,
                    "realized_h": h_real,
                }
            )
    return pd.DataFrame(rows)


def load_book_h_history(runs_root: Path) -> pd.DataFrame:
    rows = []
    for ds in list_run_dates(runs_root):
        sub = load_pair_gross_and_realized_h(runs_root, ds)
        if not sub.empty:
            rows.append(sub)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


# ---------------------------------------------------------------------------
# Actual trade markers
# ---------------------------------------------------------------------------
def load_actual_trade_markers(runs_root: Path, active_pairs: pd.DataFrame) -> pd.DataFrame:
    """Map Flex trade events to active B4 pairs.

    ETF trades attach only to that pair. Underlying trades attach to every active
    pair sharing that underlying and are flagged as shared/ambiguous.
    """
    cols = ["date", "pair", "symbol", "marker_type", "quantity", "orderReference"]
    if active_pairs.empty:
        return pd.DataFrame(columns=cols)
    etf_to_pair = {r["etf"]: r["pair"] for _, r in active_pairs.iterrows()}
    und_to_pairs: dict[str, list[str]] = {}
    for _, r in active_pairs.iterrows():
        und_to_pairs.setdefault(str(r["underlying"]), []).append(str(r["pair"]))

    try:
        from ibkr_accounting import parse_trade_events
    except Exception:
        return pd.DataFrame(columns=cols)

    rows: list[dict] = []
    if not runs_root.is_dir():
        return pd.DataFrame(columns=cols)
    for d in sorted(runs_root.iterdir()):
        if not d.is_dir():
            continue
        xml = d / "ibkr_flex" / "flex_trades.xml"
        if not xml.is_file():
            continue
        try:
            tr = parse_trade_events(xml)
        except Exception:
            continue
        if tr.empty or "symbol" not in tr.columns:
            continue
        for _, r in tr.iterrows():
            sym = _norm(r.get("symbol", ""))
            dt = pd.to_datetime(r.get("dateTime", ""), errors="coerce")
            if pd.isna(dt):
                dt = pd.Timestamp(d.name)
            dt = pd.Timestamp(dt).normalize()
            base = {
                "date": dt,
                "symbol": sym,
                "quantity": _scalar_float(r.get("quantity", 0.0), 0.0),
                "orderReference": str(r.get("orderReference", "") or ""),
            }
            if sym in etf_to_pair:
                rows.append({**base, "pair": etf_to_pair[sym], "marker_type": "actual_etf_trade"})
            if sym in und_to_pairs:
                for pair in und_to_pairs[sym]:
                    rows.append({**base, "pair": pair, "marker_type": "actual_underlying_trade_shared"})
    return pd.DataFrame(rows, columns=cols)


# ---------------------------------------------------------------------------
# Model/backtest data
# ---------------------------------------------------------------------------
@dataclass
class ModelPairResult:
    bt: pd.DataFrame
    h_signal: pd.Series
    close: pd.Series


def _load_knobs_and_costs() -> tuple[object, dict, float, float, str]:
    from scripts.bucket4_hedge_cadence import load_policy_from_config
    from strategy_config import load_config

    cfg = load_config()
    knobs, tilts, _src = load_policy_from_config(cfg)
    rules = (
        cfg.get("portfolio", {})
        .get("sleeves", {})
        .get("inverse_decay_bucket4", {})
        .get("rules", {})
    )
    opt2 = rules.get("bucket4_weekly_opt2") or {}
    fee_bps = float(opt2.get("fee_bps", 1.0))
    slip_bps = float(opt2.get("slippage_bps", 20.0))
    tag = (
        f"model h v7+v9 h_mid={knobs.h_mid:.2f} k_z={knobs.k_z:.2f} "
        f"clip[{knobs.h_min:.2f},{knobs.h_max:.2f}] cadence base={knobs.base_days:.0f}d cap={knobs.max_interval}d"
    )
    return knobs, tilts, fee_bps, slip_bps, tag


def _empty_model_results() -> tuple[dict[str, ModelPairResult], str]:
    return {}, "model unavailable"


def build_model_pair_results(
    active_pairs: pd.DataFrame,
    *,
    metrics_csv: Path,
    vol_shape_json: Path,
    start: str,
) -> tuple[dict[str, ModelPairResult], str]:
    """Run model backtests for active pairs. Missing data returns an empty map."""
    if active_pairs.empty or not metrics_csv.is_file():
        return _empty_model_results()
    try:
        from scripts.bucket4_base_days_frequency_sweep import build_prices
        from scripts.bucket4_dynamic_bt import run_bucket4_backtest_dynamic_h
        from scripts.bucket4_hedge_cadence import build_h_series, build_xsec_z_panel
        from scripts.bucket4_phase345_backtest import load_metrics_filtered
        from scripts.bucket4_vol_shape_signals import get_pair_signal, load_vol_shape_history, policy_continuous_interval
    except Exception:
        return _empty_model_results()

    try:
        knobs, tilts, fee_bps, slip_bps, tag = _load_knobs_and_costs()
        metrics = load_metrics_filtered(metrics_csv, set(active_pairs["etf"].astype(str)))
        vs_hist = load_vol_shape_history(vol_shape_json) if vol_shape_json.is_file() else {}
    except Exception:
        return _empty_model_results()

    price_by_pair: dict[str, pd.DataFrame] = {}
    close_by_und: dict[str, pd.Series] = {}
    for _, r in active_pairs.iterrows():
        prices = build_prices(metrics, str(r["etf"]), pd.Timestamp(start))
        if prices is None or prices.empty:
            continue
        pair = str(r["pair"])
        price_by_pair[pair] = prices
        close_by_und[str(r["underlying"])] = prices["b_px"].astype(float)

    xsec = None
    if close_by_und and float(getattr(knobs, "k_z", 0.0)) != 0.0:
        try:
            xsec = build_xsec_z_panel(pd.DataFrame(close_by_und))
        except Exception:
            xsec = None

    out: dict[str, ModelPairResult] = {}
    for _, r in active_pairs.iterrows():
        pair = str(r["pair"])
        prices = price_by_pair.get(pair)
        if prices is None or prices.empty:
            continue
        etf = str(r["etf"])
        und = str(r["underlying"])
        beta = abs(_scalar_float(r.get("delta", np.nan), np.nan))
        if not np.isfinite(beta) or beta <= 0:
            beta = 2.0
        borrow = _scalar_float(r.get("borrow_current", 0.0), 0.0)
        gross = _scalar_float(r.get("gross_target_usd", 0.0), 0.0)
        if gross <= 0:
            continue
        try:
            sig = get_pair_signal(
                etf,
                und,
                prices.index,
                history=vs_hist,
                underlying_prices=prices["b_px"],
                window=SIGNAL_WINDOW,
                lookahead_shift=1,
                prefer_underlying_recompute=True,
                norm_sym=_norm,
            )
            if xsec is not None and und in xsec.columns:
                sig = sig.copy()
                sig["xsec_z"] = xsec[und].reindex(sig.index)
            tilt = tilts.get(etf) or tilts.get(und)
            h = build_h_series(sig, pd.DatetimeIndex(prices.index), knobs=knobs, name_tilt=tilt)
            sched, _ = policy_continuous_interval(
                pd.DatetimeIndex(prices.index),
                sig,
                base_days=float(knobs.base_days),
                k_tr=float(knobs.k_tr),
                m_vcr=float(knobs.m_vcr),
                min_interval=int(knobs.min_interval),
                max_interval=int(knobs.max_interval),
            )
            sched = pd.DatetimeIndex(sched).intersection(prices.index)
            if len(sched) == 0:
                sched = pd.DatetimeIndex([prices.index[0]])
            bt = run_bucket4_backtest_dynamic_h(
                prices,
                h,
                sched,
                initial_capital=gross,
                beta_a=-beta,
                beta_b=1.0,
                borrow_a_annual=max(0.0, borrow),
                fee_bps=fee_bps,
                slippage_bps=slip_bps,
                opt2_h_base=float(getattr(knobs, "h_mid", 0.45)),
            )
            out[pair] = ModelPairResult(bt=bt, h_signal=h, close=prices["b_px"].astype(float))
        except Exception:
            continue
    return out, tag


# ---------------------------------------------------------------------------
# Charting
# ---------------------------------------------------------------------------
def _plot_marker_lines(
    ax,
    model_dates: pd.DatetimeIndex | list[pd.Timestamp] | None,
    actual_etf_dates: pd.Series | list[pd.Timestamp] | None,
    actual_und_dates: pd.Series | list[pd.Timestamp] | None,
) -> None:
    first_model = True
    model_list = [] if model_dates is None else list(model_dates)
    etf_list = [] if actual_etf_dates is None else list(actual_etf_dates)
    und_list = [] if actual_und_dates is None else list(actual_und_dates)
    for dt in model_list:
        ax.axvline(pd.Timestamp(dt), color="#1f77b4", lw=0.7, alpha=0.32, ls="--",
                   label="model rebalance" if first_model else None)
        first_model = False
    first_etf = True
    for dt in etf_list:
        ax.axvline(pd.Timestamp(dt), color="#d62728", lw=0.9, alpha=0.55, ls="-",
                   label="actual ETF trade" if first_etf else None)
        first_etf = False
    first_und = True
    for dt in und_list:
        ax.axvline(pd.Timestamp(dt), color="#9467bd", lw=0.8, alpha=0.55, ls=":",
                   label="actual underlying trade (shared)" if first_und else None)
        first_und = False


def _safe_last(series: pd.Series) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna()
    return float(s.iloc[-1]) if len(s) else np.nan


def _plot_pair_page(
    pdf: PdfPages,
    row: pd.Series,
    *,
    run_date: str,
    hist: pd.DataFrame,
    book_h: pd.DataFrame,
    trades: pd.DataFrame,
    model: ModelPairResult | None,
    model_tag: str,
) -> dict:
    etf, und, pair = str(row["etf"]), str(row["underlying"]), str(row["pair"])
    pair_hist = hist[hist["pair"].eq(pair)].sort_values("date")
    pair_book = book_h[book_h["pair"].eq(pair)].sort_values("date") if not book_h.empty else pd.DataFrame()
    pair_trades = trades[trades["pair"].eq(pair)].copy() if not trades.empty else pd.DataFrame()
    etf_trade_dates = (
        pair_trades.loc[pair_trades["marker_type"].eq("actual_etf_trade"), "date"].drop_duplicates()
        if not pair_trades.empty else []
    )
    und_trade_dates = (
        pair_trades.loc[pair_trades["marker_type"].eq("actual_underlying_trade_shared"), "date"].drop_duplicates()
        if not pair_trades.empty else []
    )
    model_reb_dates = pd.DatetimeIndex([])
    if model is not None and "rebalance" in model.bt.columns:
        model_reb_dates = pd.DatetimeIndex(model.bt.index[model.bt["rebalance"].astype(bool)])

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=False, constrained_layout=True)
    fig.suptitle(
        f"Bucket 4 {etf}/{und} | proposed gross ${float(row['gross_target_usd']):,.0f} | run {run_date}",
        fontsize=12,
    )

    ax = axes[0]
    if not pair_hist.empty:
        ax.plot(pair_hist["date"], pair_hist["pair_pnl_cum"], color="#1f77b4", lw=1.7, label="actual pair total")
        ax.plot(pair_hist["date"], pair_hist["etf_leg_pnl_cum"], color="#ff7f0e", lw=1.0, label=f"actual {etf} leg")
        ax.plot(pair_hist["date"], pair_hist["und_leg_pnl_cum"], color="#2ca02c", lw=1.0, label=f"actual {und} leg")
    else:
        ax.text(0.5, 0.5, "no actual accounting PnL history", ha="center", va="center", transform=ax.transAxes)
    _plot_marker_lines(ax, model_reb_dates, etf_trade_dates, und_trade_dates)
    ax.axhline(0, color="#888", lw=0.6)
    ax.set_title("Actual accounting cumulative PnL", loc="left", fontsize=10)
    ax.set_ylabel("$")
    ax.grid(True, ls="--", alpha=0.3)
    ax.legend(loc="best", fontsize=7)

    ax = axes[1]
    model_pnl_last = np.nan
    if model is not None and not model.bt.empty:
        model_pnl = model.bt["equity"].astype(float) - float(row["gross_target_usd"])
        model_pnl_last = _safe_last(model_pnl)
        ax.plot(model.bt.index, model_pnl.values, color="#0f766e", lw=1.5, label="model/backtest cumulative PnL")
        if "rebalance_skipped_below_drift" in model.bt.columns:
            skipped = model.bt.index[model.bt["rebalance_skipped_below_drift"].astype(bool)]
            for dt in skipped:
                ax.axvline(dt, color="#aaaaaa", lw=0.5, alpha=0.3, ls=":")
    else:
        ax.text(0.5, 0.5, "no model/backtest data", ha="center", va="center", transform=ax.transAxes)
    _plot_marker_lines(ax, model_reb_dates, etf_trade_dates, und_trade_dates)
    ax.axhline(0, color="#888", lw=0.6)
    ax.set_title(f"Model/backtest PnL scaled to proposed gross ({model_tag})", loc="left", fontsize=10)
    ax.set_ylabel("$")
    ax.grid(True, ls="--", alpha=0.3)
    ax.legend(loc="best", fontsize=7)

    ax = axes[2]
    current_model_h = np.nan
    if model is not None and not model.h_signal.empty:
        h = model.h_signal.dropna()
        current_model_h = _safe_last(h[h.index <= pd.Timestamp(run_date)]) if len(h) else np.nan
        ax.plot(h.index, h.values, color="#d62728", lw=1.4, label="model h")
    if not pair_book.empty:
        ax.plot(pair_book["date"], pair_book["realized_h"], color="#9467bd", lw=1.1, marker="o", ms=2.5,
                label="book/realized h")
    if np.isfinite(current_model_h):
        ax.scatter([pd.Timestamp(run_date)], [current_model_h], color="#d62728", s=36, zorder=5)
    _plot_marker_lines(ax, model_reb_dates, etf_trade_dates, und_trade_dates)
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Hedge ratio over time", loc="left", fontsize=10)
    ax.set_ylabel("h")
    ax.grid(True, ls="--", alpha=0.3)
    ax.legend(loc="best", fontsize=7)

    ax = axes[3]
    if model is not None and not model.close.empty:
        ax.plot(model.close.index, model.close.values, color="#444444", lw=1.1, label=f"{und} adj close")
    else:
        ax.text(0.5, 0.5, "no underlying price data", ha="center", va="center", transform=ax.transAxes)
    _plot_marker_lines(ax, model_reb_dates, etf_trade_dates, und_trade_dates)
    ax.set_title("Underlying price and rebalance/trade markers", loc="left", fontsize=10)
    ax.set_ylabel("price")
    ax.grid(True, ls="--", alpha=0.3)
    ax.legend(loc="best", fontsize=7)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    latest_actual = np.nan
    if not pair_hist.empty:
        latest_actual = _safe_last(pair_hist["pair_pnl_cum"])
    current_book_h = np.nan
    if not pair_book.empty:
        current_book_h = _safe_last(pair_book.loc[pair_book["date"] <= pd.Timestamp(run_date), "realized_h"])
    return {
        "pair": pair,
        "etf": etf,
        "underlying": und,
        "gross_target_usd": float(row["gross_target_usd"]),
        "actual_pair_pnl_cum": latest_actual,
        "model_pair_pnl_cum": model_pnl_last,
        "current_model_h": current_model_h,
        "current_book_h": current_book_h,
        "model_rebalance_count": int(len(model_reb_dates)),
        "actual_etf_trade_count": int(len(etf_trade_dates)),
        "actual_underlying_trade_count": int(len(und_trade_dates)),
    }


def make_b4_pair_pnl_hedge_chart(
    run_date: str,
    *,
    runs_root: Path,
    out_dir: Path,
    metrics_csv: Path = DEFAULT_METRICS_CSV,
    vol_shape_json: Path = DEFAULT_VOL_SHAPE_JSON,
    max_pairs: int = 0,
) -> tuple[Path | None, Path | None]:
    """Build the multipage per-pair PDF and companion summary CSV."""
    try:
        return _make_chart_inner(
            run_date,
            runs_root=runs_root,
            out_dir=out_dir,
            metrics_csv=metrics_csv,
            vol_shape_json=vol_shape_json,
            max_pairs=max_pairs,
        )
    except Exception as exc:
        print(f"[B4-pair-charts] skipped: {type(exc).__name__}: {exc}")
        return None, None


def _make_chart_inner(
    run_date: str,
    *,
    runs_root: Path,
    out_dir: Path,
    metrics_csv: Path,
    vol_shape_json: Path,
    max_pairs: int,
) -> tuple[Path | None, Path | None]:
    active = load_active_b4_pairs_from_proposed(run_date, runs_root=runs_root)
    if active.empty:
        print("[B4-pair-charts] no active B4 pairs in proposed trades")
        return None, None
    max_n = int(max_pairs or 0)
    if max_n > 0:
        active = active.head(max_n).copy()

    hist = load_b4_pair_leg_history(runs_root)
    book_h = load_book_h_history(runs_root)
    trades = load_actual_trade_markers(runs_root, active)
    model_results, model_tag = build_model_pair_results(
        active,
        metrics_csv=metrics_csv,
        vol_shape_json=vol_shape_json,
        start=MODEL_START,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / f"b4_pair_pnl_hedge_{run_date}.pdf"
    csv_path = out_dir / f"b4_pair_pnl_hedge_summary_{run_date}.csv"
    summary_rows: list[dict] = []
    with PdfPages(pdf_path) as pdf:
        for _, row in active.iterrows():
            summary_rows.append(
                _plot_pair_page(
                    pdf,
                    row,
                    run_date=run_date,
                    hist=hist,
                    book_h=book_h,
                    trades=trades,
                    model=model_results.get(str(row["pair"])),
                    model_tag=model_tag,
                )
            )
    pd.DataFrame(summary_rows).to_csv(csv_path, index=False)
    print(f"[B4-pair-charts] wrote {pdf_path.name} ({len(summary_rows)} pairs) + {csv_path.name}")
    return pdf_path, csv_path


def _knob_tag() -> str:
    try:
        _, _, _, _, tag = _load_knobs_and_costs()
        return tag
    except Exception:
        return "model h: config unavailable"
