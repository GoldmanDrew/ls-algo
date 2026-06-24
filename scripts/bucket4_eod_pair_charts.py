"""Per-pair Bucket 4 PnL, hedge, and rebalance charts for the EOD email.

The EOD email calls :func:`make_b4_pair_pnl_hedge_chart`.  It is intentionally
fail-soft: bad/missing optional data skips the chart instead of blocking the
email.
"""
from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
ACTIVE_B4_SLEEVES = frozenset({"inverse_decay_bucket4", "volatility_etp_bucket5"})
MIN_BACKTEST_PRICE_ROWS = 20
TRADING_DAYS = 252.0


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
    d = d[sleeve.isin(ACTIVE_B4_SLEEVES) & d["gross_target_usd"].gt(float(min_gross_usd))].copy()
    if d.empty:
        return pd.DataFrame(
            columns=["etf", "underlying", "pair", "gross_target_usd", "delta", "borrow_current", "sleeve", "sizing_tr_fwd"]
        )
    d["delta"] = pd.to_numeric(d.get("Delta", np.nan), errors="coerce")
    d["borrow_current"] = pd.to_numeric(d.get("borrow_current", np.nan), errors="coerce")
    d = d.sort_values("gross_target_usd", ascending=False)
    d = d.drop_duplicates(subset=["etf", "underlying"], keep="first")
    d["sleeve"] = sleeve.reindex(d.index).fillna("")
    d["sizing_tr_fwd"] = pd.to_numeric(d.get("und_trend_ratio_fwd_60d", np.nan), errors="coerce")
    cols = ["etf", "underlying", "pair", "gross_target_usd", "delta", "borrow_current", "sleeve", "sizing_tr_fwd"]
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
_XML_ATTR_RE = re.compile(r'([A-Za-z_:][\w:.-]*)="([^"]*)"')


def _trade_attrs_for_symbols(xml: Path, symbols: set[str]) -> list[dict[str, str]]:
    """Lightweight line scan for relevant Flex Trade rows.

    The full accounting parser is intentionally comprehensive and can be slow on
    many large historical Flex files. Charts only need marker dates/quantities
    for active symbols, so scan Trade lines and parse attributes for matches.
    """
    if not symbols:
        return []
    wanted = {_norm(s) for s in symbols if _norm(s)}
    rows: list[dict[str, str]] = []
    try:
        with xml.open("r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                if "<Trade" not in line:
                    continue
                attrs = {k: v for k, v in _XML_ATTR_RE.findall(line)}
                sym = _norm(attrs.get("symbol", ""))
                if sym in wanted:
                    rows.append(attrs)
    except Exception:
        return []
    return rows


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
    relevant_symbols = set(etf_to_pair) | set(und_to_pairs)

    rows: list[dict] = []
    if not runs_root.is_dir():
        return pd.DataFrame(columns=cols)
    for d in sorted(runs_root.iterdir()):
        if not d.is_dir():
            continue
        xml = d / "ibkr_flex" / "flex_trades.xml"
        if not xml.is_file():
            continue
        for r in _trade_attrs_for_symbols(xml, relevant_symbols):
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
                "tradeID": str(r.get("tradeID", "") or ""),
                "ibExecID": str(r.get("ibExecID", "") or ""),
            }
            if sym in etf_to_pair:
                rows.append({**base, "pair": etf_to_pair[sym], "marker_type": "actual_etf_trade"})
            if sym in und_to_pairs:
                for pair in und_to_pairs[sym]:
                    rows.append({**base, "pair": pair, "marker_type": "actual_underlying_trade_shared"})
    if not rows:
        return pd.DataFrame(columns=cols)
    out = pd.DataFrame(rows)
    dedupe_cols = [
        c
        for c in ("tradeID", "ibExecID", "date", "pair", "symbol", "marker_type", "quantity", "orderReference")
        if c in out.columns
    ]
    out = out.drop_duplicates(subset=dedupe_cols, keep="first")
    return out[cols].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Model/backtest data
# ---------------------------------------------------------------------------
@dataclass
class ModelPairResult:
    bt: pd.DataFrame
    h_signal: pd.Series
    close: pd.Series
    cadence_diag: pd.DataFrame
    status: str
    missing_reason: str
    signal_source: str
    metrics_first_date: pd.Timestamp | None
    metrics_last_date: pd.Timestamp | None
    price_rows: int
    config_tag: str


@dataclass(frozen=True)
class B4BacktestConfig:
    knobs: Any
    tilts: dict[str, Any]
    source: str
    history_start: str
    warmup_bdays: int
    fee_bps: float
    slippage_bps: float
    borrow_multiplier: float
    tag: str


def _load_b4_backtest_config() -> B4BacktestConfig:
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
    policy = opt2.get("hedge_cadence_policy") or rules.get("hedge_cadence_policy") or {}
    fee_bps = float(opt2.get("fee_bps", 1.0))
    slip_bps = float(opt2.get("slippage_bps", 20.0))
    borrow_multiplier = float(opt2.get("borrow_multiplier", 1.0))
    tag = (
        f"model h v7+v9 h_mid={knobs.h_mid:.2f} k_z={knobs.k_z:.2f} "
        f"clip[{knobs.h_min:.2f},{knobs.h_max:.2f}] cadence {knobs.cadence_signal_col} "
        f"base={knobs.base_days:.0f}d cap={knobs.max_interval}d"
    )
    return B4BacktestConfig(
        knobs=knobs,
        tilts=tilts,
        source=str(policy.get("source", _src) or _src),
        history_start=str(opt2.get("history_start", MODEL_START) or MODEL_START),
        warmup_bdays=int(opt2.get("warmup_bdays", 0) or 0),
        fee_bps=fee_bps,
        slippage_bps=slip_bps,
        borrow_multiplier=borrow_multiplier,
        tag=tag,
    )


def _empty_model_results() -> tuple[dict[str, ModelPairResult], str]:
    return {}, "model unavailable"


def _resolve_model_path(path: Path, env_key: str) -> Path:
    raw = os.environ.get(env_key, "").strip()
    if raw:
        return Path(raw)
    return Path(path)


def _build_prices_flexible(metrics: pd.DataFrame, etf: str, start: pd.Timestamp) -> tuple[pd.DataFrame | None, str, int]:
    sub = metrics[metrics["ticker"].astype(str).str.upper().eq(_norm(etf))].copy()
    if sub.empty:
        return None, "ticker_missing_from_metrics", 0
    sub["date"] = pd.to_datetime(sub["date"], errors="coerce")
    for c in ("close_price", "nav", "etf_adj_close", "underlying_adj_close"):
        if c in sub.columns:
            sub[c] = pd.to_numeric(sub[c], errors="coerce")
    if "etf_px" not in sub.columns:
        etf_px = sub.get("etf_adj_close", pd.Series(np.nan, index=sub.index)).where(
            sub.get("etf_adj_close", pd.Series(np.nan, index=sub.index)) > 0
        )
        etf_px = etf_px.fillna(sub.get("close_price", pd.Series(np.nan, index=sub.index)).where(
            sub.get("close_price", pd.Series(np.nan, index=sub.index)) > 0
        ))
        etf_px = etf_px.fillna(sub.get("nav", pd.Series(np.nan, index=sub.index)).where(
            sub.get("nav", pd.Series(np.nan, index=sub.index)) > 0
        ))
        sub["etf_px"] = etf_px
    sub = sub.dropna(subset=["date", "etf_px", "underlying_adj_close"])
    sub = sub.drop_duplicates("date").sort_values("date")
    sub = sub[sub["date"] >= start]
    sub = sub[(sub["etf_px"] > 0) & (sub["underlying_adj_close"] > 0)]
    if len(sub) < MIN_BACKTEST_PRICE_ROWS:
        return None, f"insufficient_price_history:{len(sub)}<{MIN_BACKTEST_PRICE_ROWS}", int(len(sub))
    prices = pd.DataFrame(
        {"a_px": sub["etf_px"].to_numpy(), "b_px": sub["underlying_adj_close"].to_numpy()},
        index=pd.DatetimeIndex(sub["date"]).normalize(),
    )
    return prices, "ok", int(len(prices))


def _add_backtest_derived_columns(bt: pd.DataFrame, initial_capital: float) -> pd.DataFrame:
    out = bt.copy()
    out["a_mv"] = out["a_shares"].astype(float) * out["a_px"].astype(float)
    out["b_mv"] = out["b_shares"].astype(float) * out["b_px"].astype(float)
    out["etf_gross"] = out["a_mv"].abs()
    out["underlying_gross"] = out["b_mv"].abs()
    out["total_gross"] = out["etf_gross"] + out["underlying_gross"]
    prev_a_sh = out["a_shares"].astype(float).shift(1).fillna(0.0)
    prev_b_sh = out["b_shares"].astype(float).shift(1).fillna(0.0)
    prev_a_px = out["a_px"].astype(float).shift(1).fillna(out["a_px"].astype(float))
    prev_b_px = out["b_px"].astype(float).shift(1).fillna(out["b_px"].astype(float))
    out["etf_leg_pnl_daily"] = prev_a_sh * (out["a_px"].astype(float) - prev_a_px)
    out["underlying_leg_pnl_daily"] = prev_b_sh * (out["b_px"].astype(float) - prev_b_px)
    out["etf_leg_pnl_cum"] = out["etf_leg_pnl_daily"].cumsum()
    out["underlying_leg_pnl_cum"] = out["underlying_leg_pnl_daily"].cumsum()
    out["borrow_cost_cum"] = pd.to_numeric(out.get("borrow_cost", 0.0), errors="coerce").fillna(0.0).cumsum()
    out["tcost_cum"] = pd.to_numeric(out.get("rebalance_fee", 0.0), errors="coerce").fillna(0.0).cumsum()
    out["short_proceeds_credit_cum"] = pd.to_numeric(out.get("short_proceeds_credit", 0.0), errors="coerce").fillna(0.0).cumsum()
    out["net_pnl"] = out["equity"].astype(float) - float(initial_capital)
    return out


def compute_risk_metrics(equity: pd.Series, *, risk_free_rate: float = 0.0) -> dict[str, float]:
    eq = pd.to_numeric(equity, errors="coerce").dropna()
    out = {
        "cagr": np.nan,
        "vol_annual": np.nan,
        "sharpe": np.nan,
        "max_drawdown": np.nan,
        "daily_hit_rate": np.nan,
        "best_day": np.nan,
        "worst_day": np.nan,
        "obs_days": float(len(eq)),
    }
    if len(eq) < 2:
        return out
    ret = eq.pct_change().dropna()
    if not ret.empty:
        out["vol_annual"] = float(ret.std(ddof=1) * np.sqrt(TRADING_DAYS)) if len(ret) > 1 else np.nan
        mean_ann = float(ret.mean() * TRADING_DAYS)
        out["sharpe"] = (mean_ann - float(risk_free_rate)) / out["vol_annual"] if np.isfinite(out["vol_annual"]) and out["vol_annual"] > 0 else np.nan
        out["daily_hit_rate"] = float((ret > 0).mean())
        out["best_day"] = float(ret.max())
        out["worst_day"] = float(ret.min())
    dd = eq / eq.cummax() - 1.0
    out["max_drawdown"] = float(dd.min()) if len(dd) else np.nan
    if len(eq) >= 20 and eq.iloc[0] > 0 and eq.iloc[-1] > 0:
        years = max(len(eq) / TRADING_DAYS, 1.0 / TRADING_DAYS)
        out["cagr"] = float((eq.iloc[-1] / eq.iloc[0]) ** (1.0 / years) - 1.0)
    return out


def build_model_pair_results(
    active_pairs: pd.DataFrame,
    *,
    metrics_csv: Path | None,
    vol_shape_json: Path | None,
    start: str,
) -> tuple[dict[str, ModelPairResult], str]:
    """Run config-exact model backtests for active pairs with per-pair diagnostics."""
    if active_pairs.empty:
        return _empty_model_results()
    try:
        from scripts.bucket4_dynamic_bt import run_bucket4_backtest_dynamic_h
        from scripts.bucket4_hedge_cadence import build_h_series, build_rebal_dates, build_xsec_z_panel
        from scripts.bucket4_phase345_backtest import load_metrics_filtered
        from scripts.bucket4_vol_shape_signals import get_pair_signal, load_vol_shape_history
    except Exception:
        return _empty_model_results()

    try:
        cfg = _load_b4_backtest_config()
    except Exception:
        return _empty_model_results()
    model_start = pd.Timestamp(cfg.history_start or start)

    def _status_result(row: pd.Series, status: str, reason: str, *, first=None, last=None, nrows: int = 0) -> ModelPairResult:
        return ModelPairResult(
            bt=pd.DataFrame(),
            h_signal=pd.Series(dtype=float),
            close=pd.Series(dtype=float),
            cadence_diag=pd.DataFrame(),
            status=status,
            missing_reason=reason,
            signal_source="missing",
            metrics_first_date=first,
            metrics_last_date=last,
            price_rows=int(nrows),
            config_tag=cfg.tag,
        )

    metrics_csv = _resolve_model_path(Path(metrics_csv) if metrics_csv is not None else DEFAULT_METRICS_CSV, "EOD_B4_METRICS_CSV")
    vol_shape_json = _resolve_model_path(Path(vol_shape_json) if vol_shape_json is not None else DEFAULT_VOL_SHAPE_JSON, "EOD_B4_VOL_SHAPE_JSON")
    if not metrics_csv.is_file():
        return {
            str(r["pair"]): _status_result(r, "missing_metrics_file", str(metrics_csv))
            for _, r in active_pairs.iterrows()
        }, cfg.tag

    try:
        metrics = load_metrics_filtered(metrics_csv, set(active_pairs["etf"].astype(str)))
        metrics["ticker"] = metrics["ticker"].astype(str).map(_norm)
        metrics["date"] = pd.to_datetime(metrics["date"], errors="coerce")
        vs_hist = load_vol_shape_history(vol_shape_json) if vol_shape_json.is_file() else {}
    except Exception as exc:
        return {
            str(r["pair"]): _status_result(r, "metrics_load_failed", f"{type(exc).__name__}: {exc}")
            for _, r in active_pairs.iterrows()
        }, cfg.tag

    price_by_pair: dict[str, pd.DataFrame] = {}
    close_by_und: dict[str, pd.Series] = {}
    price_status: dict[str, tuple[str, int]] = {}
    metric_ranges: dict[str, tuple[pd.Timestamp | None, pd.Timestamp | None]] = {}
    for _, r in active_pairs.iterrows():
        pair = str(r["pair"])
        etf = str(r["etf"])
        sub = metrics[metrics["ticker"].eq(_norm(etf))]
        metric_ranges[pair] = (
            sub["date"].min() if not sub.empty else None,
            sub["date"].max() if not sub.empty else None,
        )
        prices, reason, nrows = _build_prices_flexible(metrics, etf, model_start)
        price_status[pair] = (reason, nrows)
        if prices is None or prices.empty:
            continue
        price_by_pair[pair] = prices
        close_by_und[str(r["underlying"])] = prices["b_px"].astype(float)

    xsec = None
    if close_by_und and float(getattr(cfg.knobs, "k_z", 0.0)) != 0.0:
        try:
            xsec = build_xsec_z_panel(pd.DataFrame(close_by_und))
        except Exception:
            xsec = None

    out: dict[str, ModelPairResult] = {}
    for _, r in active_pairs.iterrows():
        pair = str(r["pair"])
        first_dt, last_dt = metric_ranges.get(pair, (None, None))
        prices = price_by_pair.get(pair)
        if prices is None or prices.empty:
            reason, nrows = price_status.get(pair, ("missing_price_data", 0))
            out[pair] = _status_result(r, reason, reason, first=first_dt, last=last_dt, nrows=nrows)
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
            tilt = cfg.tilts.get(etf) or cfg.tilts.get(und)
            h = build_h_series(sig, pd.DatetimeIndex(prices.index), knobs=cfg.knobs, name_tilt=tilt)
            warmup = min(int(cfg.warmup_bdays), max(0, len(prices.index) - 1))
            sched, cadence_diag = build_rebal_dates(
                sig,
                pd.DatetimeIndex(prices.index),
                knobs=cfg.knobs,
                name_tilt=tilt,
                warmup_bdays=warmup,
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
                borrow_a_annual=max(0.0, borrow) * float(cfg.borrow_multiplier),
                fee_bps=cfg.fee_bps,
                slippage_bps=cfg.slippage_bps,
                opt2_h_base=float(getattr(cfg.knobs, "h_mid", 0.45)),
                force_rebalance_after_days=int(getattr(cfg.knobs, "max_interval", 0) or 0),
            )
            bt = _add_backtest_derived_columns(bt, gross)
            status = "ok"
            try:
                run_ts = pd.Timestamp(os.environ.get("RUN_DATE") or pd.Timestamp.today()).normalize()
                if last_dt is not None and pd.notna(last_dt) and pd.Timestamp(last_dt).normalize() < run_ts - pd.Timedelta(days=5):
                    status = "ok_stale_metrics"
            except Exception:
                pass
            out[pair] = ModelPairResult(
                bt=bt,
                h_signal=h,
                close=prices["b_px"].astype(float),
                cadence_diag=cadence_diag,
                status=status,
                missing_reason="",
                signal_source=str(sig.attrs.get("signal_source", "")),
                metrics_first_date=first_dt,
                metrics_last_date=last_dt,
                price_rows=int(len(prices)),
                config_tag=cfg.tag,
            )
        except Exception as exc:
            out[pair] = _status_result(
                r,
                "backtest_failed",
                f"{type(exc).__name__}: {exc}",
                first=first_dt,
                last=last_dt,
                nrows=int(len(prices)),
            )
    return out, cfg.tag


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

    fig, axes = plt.subplots(5, 1, figsize=(13, 12), sharex=False, constrained_layout=True)
    fig.suptitle(
        f"Bucket 4/5 {etf}/{und} | {row.get('sleeve', '')} | proposed gross ${float(row['gross_target_usd']):,.0f} | run {run_date}",
        fontsize=12,
    )
    risk_free = _scalar_float(os.environ.get("EOD_B4_RISK_FREE_RATE", 0.0), 0.0)
    risk = compute_risk_metrics(model.bt["equity"], risk_free_rate=risk_free) if model is not None and not model.bt.empty else {}
    latest_model_pnl = _safe_last(model.bt["net_pnl"]) if model is not None and not model.bt.empty and "net_pnl" in model.bt else np.nan
    latest_borrow = _safe_last(model.bt["borrow_cost_cum"]) if model is not None and not model.bt.empty and "borrow_cost_cum" in model.bt else np.nan
    latest_tcost = _safe_last(model.bt["tcost_cum"]) if model is not None and not model.bt.empty and "tcost_cum" in model.bt else np.nan
    status = model.status if model is not None else "model_missing"
    reason = model.missing_reason if model is not None else ""
    fig.text(
        0.01,
        0.965,
        (
            f"model={status}"
            + (f" ({reason})" if reason else "")
            + f" | sizing fwd TR={_scalar_float(row.get('sizing_tr_fwd'), np.nan):.3f}"
            + (f" | metrics through {pd.Timestamp(model.metrics_last_date).date()}" if model is not None and model.metrics_last_date is not None and pd.notna(model.metrics_last_date) else "")
        ),
        fontsize=8,
        color="#333333",
    )
    fig.text(
        0.01,
        0.945,
        (
            f"Model PnL ${latest_model_pnl:,.0f} | Borrow ${latest_borrow:,.0f} | T-costs ${latest_tcost:,.0f} | "
            f"CAGR {_scalar_float(risk.get('cagr'), np.nan):.1%} | Vol {_scalar_float(risk.get('vol_annual'), np.nan):.1%} | "
            f"Sharpe {_scalar_float(risk.get('sharpe'), np.nan):.2f} | Max DD {_scalar_float(risk.get('max_drawdown'), np.nan):.1%}"
        ),
        fontsize=8,
        color="#333333",
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
        model_pnl = model.bt["net_pnl"].astype(float)
        model_pnl_last = _safe_last(model_pnl)
        ax.plot(model.bt.index, model_pnl.values, color="#06b6d4", lw=1.7, label="net model PnL")
        ax.plot(model.bt.index, model.bt["etf_leg_pnl_cum"], color="#10b981", lw=1.0, label="ETF leg MTM")
        ax.plot(model.bt.index, model.bt["underlying_leg_pnl_cum"], color="#8b5cf6", lw=1.0, label="underlying leg MTM")
        ax.plot(model.bt.index, -model.bt["borrow_cost_cum"], color="#ef4444", lw=1.0, ls="--", label="borrow cost drag")
        ax.plot(model.bt.index, -model.bt["tcost_cum"], color="#f59e0b", lw=1.0, ls="--", label="T-cost drag")
        if "rebalance_skipped_below_drift" in model.bt.columns:
            skipped = model.bt.index[model.bt["rebalance_skipped_below_drift"].astype(bool)]
            for dt in skipped:
                ax.axvline(dt, color="#aaaaaa", lw=0.5, alpha=0.3, ls=":")
    else:
        msg = "no model/backtest data"
        if model is not None and model.status:
            msg = f"{model.status}: {model.missing_reason or 'no backtest'}"
        ax.text(0.5, 0.5, msg, ha="center", va="center", transform=ax.transAxes, wrap=True)
    _plot_marker_lines(ax, model_reb_dates, etf_trade_dates, und_trade_dates)
    ax.axhline(0, color="#888", lw=0.6)
    ax.set_title(f"Model/backtest PnL breakdown scaled to proposed gross ({model_tag})", loc="left", fontsize=10)
    ax.set_ylabel("$")
    ax.grid(True, ls="--", alpha=0.3)
    ax.legend(loc="best", fontsize=7)

    ax = axes[2]
    if model is not None and not model.bt.empty:
        ax.plot(model.bt.index, model.bt["etf_gross"], color="#10b981", lw=1.2, label=f"{etf} short gross")
        ax.plot(model.bt.index, model.bt["underlying_gross"], color="#8b5cf6", lw=1.2, label=f"{und} short gross")
        ax.plot(model.bt.index, model.bt["total_gross"], color="#06b6d4", lw=1.0, ls="--", label="total gross")
        beta_abs = abs(_scalar_float(row.get("delta"), np.nan))
        if np.isfinite(beta_abs):
            ax.plot(model.bt.index, model.bt["etf_gross"] * beta_abs, color="#0891b2", lw=0.9, ls=":", label="beta-adjusted ETF gross")
    else:
        ax.text(0.5, 0.5, "no model gross exposure data", ha="center", va="center", transform=ax.transAxes)
    _plot_marker_lines(ax, model_reb_dates, etf_trade_dates, und_trade_dates)
    ax.set_title("Gross leg exposure", loc="left", fontsize=10)
    ax.set_ylabel("$")
    ax.grid(True, ls="--", alpha=0.3)
    ax.legend(loc="best", fontsize=7)

    ax = axes[3]
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

    ax = axes[4]
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
    risk = compute_risk_metrics(model.bt["equity"]) if model is not None and not model.bt.empty else {}
    latest_borrow = _safe_last(model.bt["borrow_cost_cum"]) if model is not None and not model.bt.empty and "borrow_cost_cum" in model.bt else np.nan
    latest_tcost = _safe_last(model.bt["tcost_cum"]) if model is not None and not model.bt.empty and "tcost_cum" in model.bt else np.nan
    latest_etf_gross = _safe_last(model.bt["etf_gross"]) if model is not None and not model.bt.empty and "etf_gross" in model.bt else np.nan
    latest_und_gross = _safe_last(model.bt["underlying_gross"]) if model is not None and not model.bt.empty and "underlying_gross" in model.bt else np.nan
    latest_total_gross = _safe_last(model.bt["total_gross"]) if model is not None and not model.bt.empty and "total_gross" in model.bt else np.nan
    return {
        "pair": pair,
        "etf": etf,
        "underlying": und,
        "sleeve": str(row.get("sleeve", "")),
        "gross_target_usd": float(row["gross_target_usd"]),
        "sizing_tr_fwd": _scalar_float(row.get("sizing_tr_fwd"), np.nan),
        "model_status": model.status if model is not None else "missing",
        "model_missing_reason": model.missing_reason if model is not None else "",
        "model_config": model.config_tag if model is not None else model_tag,
        "signal_source": model.signal_source if model is not None else "",
        "metrics_first_date": model.metrics_first_date if model is not None else None,
        "metrics_last_date": model.metrics_last_date if model is not None else None,
        "backtest_rows": len(model.bt) if model is not None and model.bt is not None else 0,
        "price_rows": model.price_rows if model is not None else 0,
        "actual_pair_pnl_cum": latest_actual,
        "model_pair_pnl_cum": model_pnl_last,
        "borrow_cost_cum": latest_borrow,
        "transaction_cost_cum": latest_tcost,
        "current_etf_gross": latest_etf_gross,
        "current_underlying_gross": latest_und_gross,
        "current_total_gross": latest_total_gross,
        "current_model_h": current_model_h,
        "current_book_h": current_book_h,
        "cagr": risk.get("cagr", np.nan),
        "vol_annual": risk.get("vol_annual", np.nan),
        "sharpe": risk.get("sharpe", np.nan),
        "max_drawdown": risk.get("max_drawdown", np.nan),
        "daily_hit_rate": risk.get("daily_hit_rate", np.nan),
        "model_rebalance_count": int(len(model_reb_dates)),
        "actual_etf_trade_count": int(len(etf_trade_dates)),
        "actual_underlying_trade_count": int(len(und_trade_dates)),
    }


def make_b4_pair_pnl_hedge_chart(
    run_date: str,
    *,
    runs_root: Path,
    out_dir: Path,
    metrics_csv: Path | None = DEFAULT_METRICS_CSV,
    vol_shape_json: Path | None = DEFAULT_VOL_SHAPE_JSON,
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
    metrics_csv: Path | None,
    vol_shape_json: Path | None,
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
        return _load_b4_backtest_config().tag
    except Exception:
        return "model h: config unavailable"
