"""Per-pair Bucket 4 PnL, hedge, and rebalance charts for the EOD email.

The EOD email calls :func:`make_b4_pair_pnl_hedge_chart`.  It is intentionally
fail-soft: bad/missing optional data skips the chart instead of blocking the
email.
"""
from __future__ import annotations

import os
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.request import urlopen

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

DEFAULT_METRICS_CSV = REPO.parent / "Levered ETFs" / "etf-dashboard" / "data" / "etf_metrics_daily.csv"
DEFAULT_VOL_SHAPE_JSON = REPO.parent / "Levered ETFs" / "etf-dashboard" / "data" / "vol_shape_history.json"
DASHBOARD_DATA_DIRS = (
    REPO.parent / "etf-dashboard" / "data",
    REPO.parent / "Levered ETFs" / "etf-dashboard" / "data",
)
DASHBOARD_RAW_BASES = (
    "https://raw.githubusercontent.com/GoldmanDrew/etf-dashboard/main/data",
    "https://raw.githubusercontent.com/magis-capital-partners/etf-dashboard/main/data",
)
METRICS_FILENAMES = ("etf_metrics_daily.parquet", "etf_metrics_daily.csv")
VOL_SHAPE_FILENAMES = ("vol_shape_history.json",)
MODEL_START = "2025-10-07"
SIGNAL_WINDOW = 45
ACTIVE_B4_SLEEVES = frozenset({"inverse_decay_bucket4", "volatility_etp_bucket5"})
MIN_BACKTEST_PRICE_ROWS = 20
TRADING_DAYS = 252.0
# Minimum fraction of actual-history dates that must carry a per-leg PnL value
# (from ``pnl_bucket_4_by_symbol.csv``) before the leg-decomposition lines are
# drawn. Below this they are sparse fragments that don't sum to the pair total.
LEG_COVERAGE_MIN = 0.6


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


@dataclass(frozen=True)
class B4ModelInputPaths:
    metrics: Path
    vol_shape: Path | None
    metrics_source: str
    vol_shape_source: str
    diagnostics: tuple[str, ...] = ()


def _env_path(name: str) -> Path | None:
    raw = os.environ.get(name, "").strip()
    return Path(raw) if raw else None


def _env_raw_bases() -> tuple[str, ...]:
    raw = os.environ.get("EOD_B4_DASHBOARD_RAW_BASES", "").strip()
    if not raw:
        raw = os.environ.get("EOD_B4_DASHBOARD_RAW_BASE", "").strip()
    if not raw:
        return DASHBOARD_RAW_BASES
    return tuple(x.strip().rstrip("/") for x in raw.split(";") if x.strip())


def _allow_github_download() -> bool:
    return os.environ.get("EOD_B4_ALLOW_GITHUB_DOWNLOAD", "1").strip().lower() not in (
        "0",
        "false",
        "no",
    )


def _copy_input_to_run(path: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / path.name
    if path.resolve() != dest.resolve():
        shutil.copy2(path, dest)
    return dest


def _download_dashboard_file(
    filename: str,
    out_dir: Path,
    *,
    raw_bases: tuple[str, ...],
    timeout_s: float = 45.0,
) -> tuple[Path | None, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / filename
    errors: list[str] = []
    for base in raw_bases:
        url = f"{base.rstrip('/')}/{filename}"
        try:
            with urlopen(url, timeout=timeout_s) as resp, dest.open("wb") as fh:
                shutil.copyfileobj(resp, fh)
            if dest.stat().st_size > 0:
                return dest, f"github:{url}"
        except Exception as exc:  # noqa: BLE001 - diagnostics only
            errors.append(f"{url}: {type(exc).__name__}: {exc}")
            try:
                if dest.exists() and dest.stat().st_size == 0:
                    dest.unlink()
            except OSError:
                pass
    return None, "; ".join(errors)


def _candidate_dashboard_dirs(extra: tuple[Path, ...] = ()) -> tuple[Path, ...]:
    seen: set[Path] = set()
    out: list[Path] = []
    for d in (*extra, *DASHBOARD_DATA_DIRS):
        p = Path(d)
        if p in seen:
            continue
        seen.add(p)
        out.append(p)
    return tuple(out)


def _first_existing(base_dirs: tuple[Path, ...], filenames: tuple[str, ...]) -> Path | None:
    for d in base_dirs:
        for name in filenames:
            p = d / name
            if p.is_file():
                return p
    return None


def resolve_b4_model_inputs(
    run_date: str,
    *,
    runs_root: Path,
    metrics_path: Path | None = None,
    vol_shape_path: Path | None = None,
    dashboard_data_dirs: tuple[Path, ...] = (),
    allow_download: bool | None = None,
    snapshot: bool = True,
) -> B4ModelInputPaths:
    """Resolve and optionally snapshot ETF-dashboard inputs for B4 chart backtests.

    Priority:
      1. run-local ``model_inputs`` snapshots,
      2. explicit/env paths,
      3. local sibling ``etf-dashboard`` checkouts,
      4. GitHub raw download into the run-local snapshot directory.
    """
    run_input_dir = runs_root / run_date / "model_inputs"
    diagnostics: list[str] = []
    allow_download = _allow_github_download() if allow_download is None else bool(allow_download)

    env_metrics = _env_path("EOD_B4_METRICS_CSV")
    env_vol = _env_path("EOD_B4_VOL_SHAPE_JSON")
    explicit_metrics = Path(metrics_path) if metrics_path is not None else env_metrics
    explicit_vol = Path(vol_shape_path) if vol_shape_path is not None else env_vol

    metric_files = METRICS_FILENAMES
    run_metrics = _first_existing((run_input_dir,), metric_files)
    metrics_source = ""
    metrics = run_metrics
    if metrics is not None:
        metrics_source = f"run-local:{metrics}"
    elif explicit_metrics is not None:
        if explicit_metrics.is_file():
            metrics = _copy_input_to_run(explicit_metrics, run_input_dir) if snapshot else explicit_metrics
            metrics_source = f"explicit:{explicit_metrics}"
        else:
            metrics = explicit_metrics
            metrics_source = f"missing-explicit:{explicit_metrics}"
            diagnostics.append(f"explicit metrics path missing: {explicit_metrics}")
    else:
        local = _first_existing(_candidate_dashboard_dirs(dashboard_data_dirs), metric_files)
        if local is not None:
            metrics = _copy_input_to_run(local, run_input_dir) if snapshot else local
            metrics_source = f"local-dashboard:{local}"
        elif allow_download:
            downloaded, source = _download_dashboard_file(
                "etf_metrics_daily.parquet",
                run_input_dir,
                raw_bases=_env_raw_bases(),
            )
            if downloaded is None:
                downloaded, source = _download_dashboard_file(
                    "etf_metrics_daily.csv",
                    run_input_dir,
                    raw_bases=_env_raw_bases(),
                    timeout_s=120.0,
                )
            if downloaded is not None:
                metrics = downloaded
                metrics_source = source
            else:
                metrics = run_input_dir / "etf_metrics_daily.parquet"
                metrics_source = f"download_failed:{source}"
                diagnostics.append(f"metrics download failed: {source}")
        else:
            metrics = run_input_dir / "etf_metrics_daily.parquet"
            metrics_source = "unresolved"
            diagnostics.append("metrics unresolved and GitHub download disabled")

    run_vol = _first_existing((run_input_dir,), VOL_SHAPE_FILENAMES)
    vol_source = ""
    vol_shape = run_vol
    if vol_shape is not None:
        vol_source = f"run-local:{vol_shape}"
    elif explicit_vol is not None:
        if explicit_vol.is_file():
            vol_shape = _copy_input_to_run(explicit_vol, run_input_dir) if snapshot else explicit_vol
            vol_source = f"explicit:{explicit_vol}"
        else:
            vol_shape = explicit_vol
            vol_source = f"missing-explicit:{explicit_vol}"
            diagnostics.append(f"explicit vol-shape path missing: {explicit_vol}")
    else:
        local_vol = _first_existing(_candidate_dashboard_dirs(dashboard_data_dirs), VOL_SHAPE_FILENAMES)
        if local_vol is not None:
            vol_shape = _copy_input_to_run(local_vol, run_input_dir) if snapshot else local_vol
            vol_source = f"local-dashboard:{local_vol}"
        elif allow_download:
            downloaded, source = _download_dashboard_file(
                "vol_shape_history.json",
                run_input_dir,
                raw_bases=_env_raw_bases(),
            )
            if downloaded is not None:
                vol_shape = downloaded
                vol_source = source
            else:
                vol_shape = None
                vol_source = f"download_failed:{source}"
                diagnostics.append(f"vol-shape download failed: {source}")
        else:
            vol_shape = None
            vol_source = "unresolved"
            diagnostics.append("vol-shape unresolved and GitHub download disabled")

    return B4ModelInputPaths(
        metrics=Path(metrics),
        vol_shape=Path(vol_shape) if vol_shape is not None else None,
        metrics_source=metrics_source,
        vol_shape_source=vol_source,
        diagnostics=tuple(diagnostics),
    )


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
    # Operator pair-override audit columns (config/pair_overrides.yml). Absent on
    # older proposed_trades.csv -> default to no-override values.
    if "pair_override_gross_mult" in d.columns:
        d["pair_override_gross_mult"] = pd.to_numeric(d["pair_override_gross_mult"], errors="coerce").fillna(1.0)
    else:
        d["pair_override_gross_mult"] = 1.0
    if "pair_override_hedge_add" in d.columns:
        d["pair_override_hedge_add"] = pd.to_numeric(d["pair_override_hedge_add"], errors="coerce").fillna(0.0)
    else:
        d["pair_override_hedge_add"] = 0.0
    if "pair_override_note" in d.columns:
        d["pair_override_note"] = d["pair_override_note"].fillna("").astype(str)
    else:
        d["pair_override_note"] = ""
    if "b4_opt2_hedge_ratio" in d.columns:
        d["applied_hedge_ratio"] = pd.to_numeric(d["b4_opt2_hedge_ratio"], errors="coerce")
    else:
        d["applied_hedge_ratio"] = np.nan
    cols = [
        "etf", "underlying", "pair", "gross_target_usd", "delta", "borrow_current", "sleeve", "sizing_tr_fwd",
        "pair_override_gross_mult", "pair_override_hedge_add", "pair_override_note", "applied_hedge_ratio",
    ]
    return d[cols].sort_values("gross_target_usd", ascending=False).reset_index(drop=True)


def load_ratchet_targets(runs_root: Path, run_date: str) -> dict[str, dict]:
    """Per-pair grow-only ratchet / continuous-trim audit for the run.

    Returns a dict keyed by ``ETF|UNDERLYING`` with the creep ratio, dollar gap
    above the freshly solved target, the trim fraction (lambda) applied this run,
    and the dollar amount being trimmed. Fail-soft: returns ``{}`` when the
    ``b4_ratchet_targets.csv`` produced by ``generate_trade_plan`` is absent.
    """
    p = runs_root / run_date / "b4_hedge_cadence" / "b4_ratchet_targets.csv"
    if not p.is_file():
        return {}
    try:
        df = pd.read_csv(p)
    except Exception:
        return {}
    if df.empty or not {"ETF", "Underlying"}.issubset(df.columns):
        return {}
    out: dict[str, dict] = {}
    for _, r in df.iterrows():
        key = _pair_key(r["ETF"], r["Underlying"])
        out[key] = {
            "inverse_solved_usd": _scalar_float(r.get("inverse_short_solved_usd"), np.nan),
            "inverse_target_usd": _scalar_float(r.get("inverse_etf_short_usd"), np.nan),
            "ratchet_floor_usd": _scalar_float(r.get("ratchet_floor_usd"), np.nan),
            "creep_ratio": _scalar_float(r.get("ratchet_creep_ratio"), np.nan),
            "gap_usd": _scalar_float(r.get("ratchet_gap_usd"), np.nan),
            "trim_lambda": _scalar_float(r.get("ratchet_trim_lambda"), np.nan),
            "trim_usd": _scalar_float(r.get("ratchet_trim_usd"), np.nan),
            "released": bool(r.get("ratchet_released", False)),
            "source": str(r.get("ratchet_source", "") or ""),
            "forward_edge_annual": _scalar_float(r.get("forward_edge_annual"), np.nan),
        }
    return out


def load_shared_underlyings(runs_root: Path, run_date: str, active: pd.DataFrame) -> set[str]:
    """Underlyings whose B4 short shares a netted broker line with B1/B2/B5.

    Two independent signals are unioned:
      * the same underlying ticker carries a B1/B2/B5 claim in
        ``accounting/pnl_by_symbol.csv`` (one physical spot line, many sleeves);
      * the underlying backs more than one active B4 pair.
    For these names the per-pair "underlying leg" PnL is an *attributed slice*,
    not an independently observed series, so the chart should say so.
    """
    shared: set[str] = set()
    if not active.empty and "underlying" in active.columns:
        counts = active["underlying"].map(_norm).value_counts()
        shared |= set(counts[counts > 1].index)
    p = runs_root / run_date / "accounting" / "pnl_by_symbol.csv"
    if p.is_file():
        try:
            sym = pd.read_csv(p)
        except Exception:
            sym = pd.DataFrame()
        if not sym.empty and {"symbol", "bucket"}.issubset(sym.columns):
            sym["symbol"] = sym["symbol"].map(_norm)
            non_b4 = sym[sym["bucket"].astype(str).isin(["bucket_1", "bucket_2", "bucket_5"])].copy()
            # Only treat a non-B4 claim as material (avoid flagging splitter dust).
            if "total_pnl" in non_b4.columns:
                _tp = pd.to_numeric(non_b4["total_pnl"], errors="coerce").abs()
                non_b4 = non_b4[_tp > 50.0]
            b4_unds = set(active["underlying"].map(_norm)) if not active.empty else set()
            shared |= (set(non_b4["symbol"]) & b4_unds)
    return shared


def _apply_date_axis(ax: "plt.Axes") -> None:
    """Consistent, adaptive date ticks (days on short ranges, months/years on long).

    ``ConciseDateFormatter`` avoids the duplicate ``2026-06 2026-06`` labels that a
    fixed ``%Y-%m`` formatter produces when a pair only has a few weeks of history.
    """
    try:
        loc = mdates.AutoDateLocator(minticks=4, maxticks=8)
        ax.xaxis.set_major_locator(loc)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))
        for lbl in ax.get_xticklabels():
            lbl.set_rotation(0)
            lbl.set_fontsize(7)
    except Exception:
        pass


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


def _reconcile_actual_endpoints(
    actual_by_pair: dict[str, float],
    *,
    runs_root: Path,
    run_date: str,
    tol: float = 1.0,
) -> list[str]:
    """Warn when a charted actual-PnL endpoint diverges from the run's accounting.

    The chart's last "actual pair total" point must equal the ``total_pnl`` the
    EOD email reports for the same run date (``pnl_bucket_4_by_pair.csv``). Any
    divergence means the stitched history and the email disagree — log it so the
    mismatch is caught instead of silently shipping a wrong-looking graph.
    """
    mismatches: list[str] = []
    p = runs_root / run_date / "accounting" / "pnl_bucket_4_by_pair.csv"
    if not p.is_file():
        return mismatches
    try:
        ap = pd.read_csv(p)
    except Exception:
        return mismatches
    if ap.empty or not {"etf", "underlying", "total_pnl"}.issubset(ap.columns):
        return mismatches
    latest = {
        _pair_key(e, u): _scalar_float(t, np.nan)
        for e, u, t in zip(ap["etf"], ap["underlying"], ap["total_pnl"])
    }
    for pair, charted in actual_by_pair.items():
        acct = latest.get(pair, np.nan)
        if np.isfinite(acct) and np.isfinite(charted) and abs(acct - charted) > tol:
            msg = f"{pair}: chart={charted:,.0f} vs accounting={acct:,.0f}"
            mismatches.append(msg)
            print(f"[B4-pair-charts] WARN actual-PnL endpoint mismatch {msg}")
    return mismatches


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
    metrics_source: str = ""
    vol_shape_source: str = ""
    diagnostics: str = ""


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


def _load_metrics_filtered_any(path: Path, tickers: set[str]) -> pd.DataFrame:
    usecols = ["date", "ticker", "close_price", "nav", "etf_adj_close", "underlying_adj_close"]
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path, columns=usecols)
        df["ticker"] = df["ticker"].map(_norm)
        df = df[df["ticker"].isin({_norm(t) for t in tickers})].copy()
    else:
        from scripts.bucket4_phase345_backtest import load_metrics_filtered

        df = load_metrics_filtered(path, {_norm(t) for t in tickers})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for c in ("close_price", "nav", "etf_adj_close", "underlying_adj_close"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if "etf_px" not in df.columns:
        etf = df["etf_adj_close"].where(df["etf_adj_close"] > 0)
        etf = etf.fillna(df["close_price"].where(df["close_price"] > 0))
        etf = etf.fillna(df["nav"].where(df["nav"] > 0))
        df["etf_px"] = etf
    return df


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
    metrics_source: str = "",
    vol_shape_source: str = "",
    diagnostics: tuple[str, ...] = (),
) -> tuple[dict[str, ModelPairResult], str]:
    """Run config-exact model backtests for active pairs with per-pair diagnostics."""
    if active_pairs.empty:
        return _empty_model_results()
    try:
        from scripts.bucket4_dynamic_bt import run_bucket4_backtest_dynamic_h
        from scripts.bucket4_hedge_cadence import build_h_series, build_rebal_dates, build_xsec_z_panel
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
            metrics_source=metrics_source,
            vol_shape_source=vol_shape_source,
            diagnostics=" | ".join(diagnostics),
        )

    metrics_csv = _resolve_model_path(Path(metrics_csv) if metrics_csv is not None else DEFAULT_METRICS_CSV, "EOD_B4_METRICS_CSV")
    vol_shape_json = Path(vol_shape_json) if vol_shape_json is not None else None
    if not metrics_csv.is_file():
        return {
            str(r["pair"]): _status_result(r, "missing_metrics_file", str(metrics_csv))
            for _, r in active_pairs.iterrows()
        }, cfg.tag

    try:
        metrics = _load_metrics_filtered_any(metrics_csv, set(active_pairs["etf"].astype(str)))
        metrics["ticker"] = metrics["ticker"].astype(str).map(_norm)
        metrics["date"] = pd.to_datetime(metrics["date"], errors="coerce")
        vs_hist = load_vol_shape_history(vol_shape_json) if vol_shape_json is not None and vol_shape_json.is_file() else {}
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
                metrics_source=metrics_source,
                vol_shape_source=vol_shape_source,
                diagnostics=" | ".join(diagnostics),
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


def _trim_leading_flat_history(
    df: pd.DataFrame,
    value_cols: tuple[str, ...] = ("pair_pnl_cum",),
    eps: float = 1e-6,
) -> pd.DataFrame:
    """Drop leading run-date rows where the pair had no economic PnL yet.

    Historical ``pnl_bucket_4_by_pair.csv`` snapshots emit a ``0.0`` row for
    every run since a pair entered the registry — often months before any
    position existed. Plotting those flat-zero months stretches the x-axis and
    compresses the real PnL into a sliver on the right, which is the main reason
    the chart looks nothing like the emailed figure. Keep a single zero anchor
    immediately before the first economically meaningful row so the curve still
    visibly starts from 0.
    """
    if df.empty:
        return df
    d = df.sort_values("date").reset_index(drop=True)
    mag: pd.Series | None = None
    for c in value_cols:
        if c in d.columns:
            col = pd.to_numeric(d[c], errors="coerce").abs().fillna(0.0)
            mag = col if mag is None else mag.add(col, fill_value=0.0)
    if mag is None:
        return d
    nz = mag > eps
    if not bool(nz.any()):
        return d.tail(1).reset_index(drop=True)
    first = int(nz.to_numpy().argmax())
    start = max(0, first - 1)
    return d.iloc[start:].reset_index(drop=True)


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
    ratchet_by_pair: dict[str, dict] | None = None,
    shared_unds: set[str] | None = None,
) -> dict:
    etf, und, pair = str(row["etf"]), str(row["underlying"]), str(row["pair"])
    pair_hist = (
        hist[hist["pair"].eq(pair)].sort_values("date")
        if not hist.empty and "pair" in hist.columns
        else pd.DataFrame()
    )
    if not pair_hist.empty:
        pair_hist = _trim_leading_flat_history(pair_hist)
    pair_book = (
        book_h[book_h["pair"].eq(pair)].sort_values("date")
        if not book_h.empty and "pair" in book_h.columns
        else pd.DataFrame()
    )
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

    ratchet = (ratchet_by_pair or {}).get(pair, {})
    is_shared = und in (shared_unds or set())

    model_ok = model is not None and not model.bt.empty
    risk_free = _scalar_float(os.environ.get("EOD_B4_RISK_FREE_RATE", 0.0), 0.0)
    risk = compute_risk_metrics(model.bt["equity"], risk_free_rate=risk_free) if model_ok else {}
    model_pnl_last = _safe_last(model.bt["net_pnl"]) if model_ok and "net_pnl" in model.bt else np.nan
    latest_model_pnl = model_pnl_last
    latest_borrow = _safe_last(model.bt["borrow_cost_cum"]) if model_ok and "borrow_cost_cum" in model.bt else np.nan
    latest_tcost = _safe_last(model.bt["tcost_cum"]) if model_ok and "tcost_cum" in model.bt else np.nan
    status = model.status if model is not None else "model_missing"
    reason = model.missing_reason if model is not None else ""
    latest_actual_pnl = _safe_last(pair_hist["pair_pnl_cum"]) if not pair_hist.empty else np.nan
    current_model_h = np.nan
    if model is not None and not model.h_signal.empty:
        _h = model.h_signal.dropna()
        current_model_h = _safe_last(_h[_h.index <= pd.Timestamp(run_date)]) if len(_h) else np.nan

    _ovr_gm = _scalar_float(row.get("pair_override_gross_mult", 1.0), 1.0)
    _ovr_ha = _scalar_float(row.get("pair_override_hedge_add", 0.0), 0.0)
    _ovr_note = str(row.get("pair_override_note", "") or "")
    _ovr_active = (abs(_ovr_gm - 1.0) > 1e-9) or (abs(_ovr_ha) > 1e-9)

    # ---- panel drawing closures (each takes a single Axes) ------------------
    def draw_actual(ax: "plt.Axes") -> None:
        span_lo = span_hi = None
        if not pair_hist.empty:
            span_lo = pd.Timestamp(pair_hist["date"].min())
            span_hi = pd.Timestamp(pair_hist["date"].max())
            ax.plot(pair_hist["date"], pair_hist["pair_pnl_cum"], color="#1f77b4", lw=1.7, label="actual pair total")
            # The per-leg split comes from pnl_bucket_4_by_symbol.csv, which only
            # recently began carrying these symbols. Draw the leg lines only when
            # they cover enough of the window to be meaningful; otherwise they are
            # misleading sparse fragments that don't visually sum to the pair total.
            etf_cov = float(pd.to_numeric(pair_hist["etf_leg_pnl_cum"], errors="coerce").notna().mean())
            if etf_cov >= LEG_COVERAGE_MIN:
                ax.plot(pair_hist["date"], pair_hist["etf_leg_pnl_cum"], color="#ff7f0e", lw=1.0, label=f"actual {etf} leg")
                ax.plot(pair_hist["date"], pair_hist["und_leg_pnl_cum"], color="#2ca02c", lw=1.0, label=f"actual {und} leg")
            else:
                ax.text(
                    0.005, 0.90,
                    f"leg split unavailable for early history (coverage {etf_cov:.0%})",
                    transform=ax.transAxes, fontsize=6.5, color="#9ca3af", va="top", style="italic",
                )
        else:
            ax.text(0.5, 0.5, "no actual accounting PnL history", ha="center", va="center", transform=ax.transAxes)
        if is_shared:
            ax.text(
                0.005, 0.97,
                f"\u26a0 {und} is a SHARED underlying (netted with B1/B2/B5) — "
                f"the {und}-leg line is an attributed slice, not an independently observed PnL",
                transform=ax.transAxes, fontsize=6.5, color="#b45309", va="top", style="italic",
            )
        # Only overlay markers that fall inside the actual-accounting window. The
        # model backtest starts months earlier, so its rebalance markers are
        # omitted here (they belong on the model panels) and the trade markers are
        # clipped — otherwise they stretch this panel's x-axis and distort the
        # curve relative to the emailed figure.
        if span_lo is not None:
            def _clip(dates) -> list:
                seq = list(dates) if dates is not None else []
                return [x for x in seq if span_lo <= pd.Timestamp(x) <= span_hi]

            _plot_marker_lines(ax, None, _clip(etf_trade_dates), _clip(und_trade_dates))
            ax.set_xlim(span_lo - pd.Timedelta(days=1), span_hi + pd.Timedelta(days=1))
        ax.axhline(0, color="#888", lw=0.6)
        ax.set_title("Actual accounting cumulative PnL", loc="left", fontsize=10)
        ax.set_ylabel("$")
        ax.grid(True, ls="--", alpha=0.3)
        ax.legend(loc="best", fontsize=7)
        _apply_date_axis(ax)

    def draw_model_pnl(ax: "plt.Axes") -> None:
        if model_ok:
            model_pnl = model.bt["net_pnl"].astype(float)
            ax.plot(model.bt.index, model_pnl.values, color="#06b6d4", lw=1.7, label="net model PnL")
            ax.plot(model.bt.index, model.bt["etf_leg_pnl_cum"], color="#10b981", lw=1.0, label="ETF leg MTM")
            ax.plot(model.bt.index, model.bt["underlying_leg_pnl_cum"], color="#8b5cf6", lw=1.0, label="underlying leg MTM")
            ax.plot(model.bt.index, -model.bt["borrow_cost_cum"], color="#ef4444", lw=1.0, ls="--", label="borrow cost drag")
            ax.plot(model.bt.index, -model.bt["tcost_cum"], color="#f59e0b", lw=1.0, ls="--", label="T-cost drag")
            if "rebalance_skipped_below_drift" in model.bt.columns:
                for dt in model.bt.index[model.bt["rebalance_skipped_below_drift"].astype(bool)]:
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
        _apply_date_axis(ax)

    def draw_gross(ax: "plt.Axes") -> None:
        if model_ok:
            ax.plot(model.bt.index, model.bt["etf_gross"], color="#10b981", lw=1.2, label=f"{etf} short gross")
            ax.plot(model.bt.index, model.bt["underlying_gross"], color="#8b5cf6", lw=1.2, label=f"{und} short gross")
            ax.plot(model.bt.index, model.bt["total_gross"], color="#06b6d4", lw=1.0, ls="--", label="total gross")
            beta_abs = abs(_scalar_float(row.get("delta"), np.nan))
            if np.isfinite(beta_abs):
                ax.plot(model.bt.index, model.bt["etf_gross"] * beta_abs, color="#0891b2", lw=0.9, ls=":", label="beta-adjusted ETF gross")
        else:
            ax.text(0.5, 0.5, "no model gross exposure data", ha="center", va="center", transform=ax.transAxes)
        # Ratchet / continuous-trim overlays: visualize the inverse-ETF gap being unwound.
        _solved = _scalar_float(ratchet.get("inverse_solved_usd"), np.nan)
        _floor = _scalar_float(ratchet.get("ratchet_floor_usd"), np.nan)
        _target = _scalar_float(ratchet.get("inverse_target_usd"), np.nan)
        if np.isfinite(_solved):
            ax.axhline(_solved, color="#16a34a", lw=1.1, ls=":", label=f"inverse solved ${_solved:,.0f}")
        if np.isfinite(_floor) and _floor > _solved + 1e-6:
            ax.axhline(_floor, color="#b91c1c", lw=1.1, ls="--", label=f"ratchet floor (pinned) ${_floor:,.0f}")
            ax.axhspan(min(_solved, _floor), max(_solved, _floor), color="#fca5a5", alpha=0.12)
        if np.isfinite(_target) and abs(_target - _floor) > 1e-6:
            ax.axhline(_target, color="#2563eb", lw=1.3, label=f"proposed inverse target ${_target:,.0f}")
        _plot_marker_lines(ax, model_reb_dates, etf_trade_dates, und_trade_dates)
        ax.set_title("Gross leg exposure (+ ratchet target / floor)", loc="left", fontsize=10)
        ax.set_ylabel("$")
        ax.grid(True, ls="--", alpha=0.3)
        ax.legend(loc="best", fontsize=6.5)
        _apply_date_axis(ax)

    def draw_hedge(ax: "plt.Axes") -> None:
        if model is not None and not model.h_signal.empty:
            _h = model.h_signal.dropna()
            ax.plot(_h.index, _h.values, color="#d62728", lw=1.4, label="model h")
        if not pair_book.empty:
            ax.plot(pair_book["date"], pair_book["realized_h"], color="#9467bd", lw=1.1, marker="o", ms=2.5,
                    label="book/realized h")
        if np.isfinite(current_model_h):
            ax.scatter([pd.Timestamp(run_date)], [current_model_h], color="#d62728", s=36, zorder=5)
        if _ovr_active and abs(_ovr_ha) > 1e-9:
            _applied_h = _scalar_float(row.get("applied_hedge_ratio", np.nan), np.nan)
            if np.isfinite(_applied_h):
                ax.axhline(_applied_h, color="#b91c1c", lw=1.4, ls="--",
                           label=f"applied h (override {_ovr_ha:+g})")
        # Annotate the trim state: hedge ratio is preserved while gross de-grosses.
        _lam = _scalar_float(ratchet.get("trim_lambda"), np.nan)
        if np.isfinite(_lam) and _lam > 1e-9:
            ax.text(
                0.005, 0.04,
                f"continuous trim active: \u03bb={_lam:.2f} ({ratchet.get('source', '')}) — "
                f"h held, gross trimmed ${_scalar_float(ratchet.get('trim_usd'), 0.0):,.0f}",
                transform=ax.transAxes, fontsize=6.5, color="#15803d", va="bottom",
            )
        _plot_marker_lines(ax, model_reb_dates, etf_trade_dates, und_trade_dates)
        ax.set_ylim(0.0, 1.05)
        ax.set_title("Hedge ratio over time", loc="left", fontsize=10)
        ax.set_ylabel("h")
        ax.grid(True, ls="--", alpha=0.3)
        ax.legend(loc="best", fontsize=7)
        _apply_date_axis(ax)

    def draw_price(ax: "plt.Axes") -> None:
        if model is not None and not model.close.empty:
            ax.plot(model.close.index, model.close.values, color="#444444", lw=1.1, label=f"{und} adj close")
        else:
            ax.text(0.5, 0.5, "no underlying price data", ha="center", va="center", transform=ax.transAxes)
        _plot_marker_lines(ax, model_reb_dates, etf_trade_dates, und_trade_dates)
        ax.set_title("Underlying price and rebalance/trade markers", loc="left", fontsize=10)
        ax.set_ylabel("price")
        ax.grid(True, ls="--", alpha=0.3)
        ax.legend(loc="best", fontsize=7)
        _apply_date_axis(ax)

    def draw_status(ax: "plt.Axes") -> None:
        ax.axis("off")
        lines = [
            f"Model backtest unavailable: {status}" + (f" ({reason})" if reason else ""),
            f"price rows={model.price_rows if model is not None else 0} "
            f"(need >= {MIN_BACKTEST_PRICE_ROWS})",
            "",
            f"proposed gross ${float(row['gross_target_usd']):,.0f} | sleeve {row.get('sleeve', '')}",
            f"latest actual pair PnL ${_safe_last(pair_hist['pair_pnl_cum']) if not pair_hist.empty else float('nan'):,.0f}"
            if not pair_hist.empty else "no actual accounting PnL history yet",
        ]
        if ratchet:
            lines.append(
                f"ratchet: creep {_scalar_float(ratchet.get('creep_ratio'), np.nan):.2f}x | "
                f"gap ${_scalar_float(ratchet.get('gap_usd'), 0.0):,.0f} | "
                f"\u03bb {_scalar_float(ratchet.get('trim_lambda'), 0.0):.2f} | "
                f"trimming ${_scalar_float(ratchet.get('trim_usd'), 0.0):,.0f}/reb"
            )
        ax.text(0.02, 0.95, "\n".join(lines), transform=ax.transAxes, va="top", ha="left",
                fontsize=9, color="#334155", family="monospace")

    # Adaptive layout: skip empty model panels for short-history pairs (P2).
    if model_ok:
        panels = [draw_actual, draw_model_pnl, draw_gross, draw_hedge, draw_price]
    else:
        panels = [draw_actual]
        if not pair_book.empty:
            panels.append(draw_hedge)
        if model is not None and not model.close.empty:
            panels.append(draw_price)
        panels.append(draw_status)

    header_h = 1.05
    panel_h = 2.15
    height_ratios = [header_h] + [panel_h] * len(panels)
    fig, all_axes = plt.subplots(
        len(panels) + 1, 1,
        figsize=(13, header_h * 1.3 + panel_h * 1.3 * len(panels)),
        squeeze=False, constrained_layout=True,
        gridspec_kw={"height_ratios": height_ratios},
    )
    all_axes = all_axes[:, 0]

    # ---- header panel (all run/model/ratchet metadata, no axes overlap) -----
    hax = all_axes[0]
    hax.axis("off")
    _ovr_txt = ""
    if _ovr_active:
        _ovr_txt = f"  |  OVERRIDE gross\u00d7{_ovr_gm:g} h{_ovr_ha:+g}" + (f" ({_ovr_note})" if _ovr_note else "")
    hax.text(
        0.0, 1.0,
        f"Bucket 4/5 {etf}/{und}  |  {row.get('sleeve', '')}  |  proposed gross "
        f"${float(row['gross_target_usd']):,.0f}  |  run {run_date}{_ovr_txt}",
        transform=hax.transAxes, va="top", ha="left", fontsize=13, fontweight="bold",
        color=("#b91c1c" if _ovr_active else "black"),
    )
    _metrics_through = (
        f"  |  metrics through {pd.Timestamp(model.metrics_last_date).date()}"
        if model is not None and model.metrics_last_date is not None and pd.notna(model.metrics_last_date)
        else ""
    )
    hax.text(
        0.0, 0.66,
        f"model={status}" + (f" ({reason})" if reason else "")
        + f"  |  sizing fwd TR={_scalar_float(row.get('sizing_tr_fwd'), np.nan):.3f}{_metrics_through}",
        transform=hax.transAxes, va="top", ha="left", fontsize=8.5, color="#333333",
    )
    _actual_txt = f"${latest_actual_pnl:,.0f}" if np.isfinite(latest_actual_pnl) else "n/a"
    hax.text(
        0.0, 0.40,
        f"Actual PnL (accounting/email) {_actual_txt}  |  Model PnL ${latest_model_pnl:,.0f}  |  "
        f"Borrow ${latest_borrow:,.0f}  |  "
        f"T-costs ${latest_tcost:,.0f}  |  CAGR {_scalar_float(risk.get('cagr'), np.nan):.1%}  |  "
        f"Vol {_scalar_float(risk.get('vol_annual'), np.nan):.1%}  |  "
        f"Sharpe {_scalar_float(risk.get('sharpe'), np.nan):.2f}  |  "
        f"Max DD {_scalar_float(risk.get('max_drawdown'), np.nan):.1%}",
        transform=hax.transAxes, va="top", ha="left", fontsize=8.5, color="#333333",
    )
    if ratchet:
        _rel = bool(ratchet.get("released", False))
        hax.text(
            0.0, 0.14,
            f"ratchet: creep {_scalar_float(ratchet.get('creep_ratio'), np.nan):.2f}x  |  "
            f"gap ${_scalar_float(ratchet.get('gap_usd'), 0.0):,.0f}  |  "
            f"\u03bb {_scalar_float(ratchet.get('trim_lambda'), 0.0):.2f}  |  "
            f"trimming ${_scalar_float(ratchet.get('trim_usd'), 0.0):,.0f}/reb  |  "
            f"released={_rel}  |  src={ratchet.get('source', '')}  |  "
            f"fwd_edge {_scalar_float(ratchet.get('forward_edge_annual'), np.nan):.3f}",
            transform=hax.transAxes, va="top", ha="left", fontsize=8.5,
            color=("#15803d" if _rel else "#6b7280"), fontweight=("bold" if _rel else "normal"),
        )
    else:
        hax.text(
            0.0, 0.14,
            "ratchet: (no b4_ratchet_targets.csv for this run)",
            transform=hax.transAxes, va="top", ha="left", fontsize=8.5, color="#9ca3af",
        )

    for ax, draw in zip(all_axes[1:], panels):
        draw(ax)

    pdf.savefig(fig)
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
        "pair_override_gross_mult": _scalar_float(row.get("pair_override_gross_mult", 1.0), 1.0),
        "pair_override_hedge_add": _scalar_float(row.get("pair_override_hedge_add", 0.0), 0.0),
        "pair_override_note": str(row.get("pair_override_note", "") or ""),
        "applied_hedge_ratio": _scalar_float(row.get("applied_hedge_ratio", np.nan), np.nan),
        "sizing_tr_fwd": _scalar_float(row.get("sizing_tr_fwd"), np.nan),
        "model_status": model.status if model is not None else "missing",
        "model_missing_reason": model.missing_reason if model is not None else "",
        "model_config": model.config_tag if model is not None else model_tag,
        "metrics_source": model.metrics_source if model is not None else "",
        "vol_shape_source": model.vol_shape_source if model is not None else "",
        "model_input_diagnostics": model.diagnostics if model is not None else "",
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
        "shared_underlying": bool(is_shared),
        "ratchet_creep_ratio": _scalar_float(ratchet.get("creep_ratio"), np.nan) if ratchet else np.nan,
        "ratchet_gap_usd": _scalar_float(ratchet.get("gap_usd"), np.nan) if ratchet else np.nan,
        "ratchet_trim_lambda": _scalar_float(ratchet.get("trim_lambda"), np.nan) if ratchet else np.nan,
        "ratchet_trim_usd": _scalar_float(ratchet.get("trim_usd"), np.nan) if ratchet else np.nan,
        "ratchet_released": bool(ratchet.get("released", False)) if ratchet else False,
        "ratchet_source": str(ratchet.get("source", "")) if ratchet else "",
    }


def _plot_summary_page(
    pdf: PdfPages,
    active: pd.DataFrame,
    *,
    run_date: str,
    model_tag: str,
    ratchet_by_pair: dict[str, dict],
    shared_unds: set[str],
    model_results: dict[str, "ModelPairResult"],
    actual_by_pair: dict[str, float] | None = None,
) -> None:
    """Index / overview page: one row per pair with sizing, ratchet, and risk."""
    actual_by_pair = actual_by_pair or {}
    headers = ["Pair", "Sleeve", "Gross $", "Actual $", "Creep x", "lambda", "Trim $",
               "h model", "Sharpe", "MaxDD", "Shared", "Status"]
    cells: list[list[str]] = []
    risk_free = _scalar_float(os.environ.get("EOD_B4_RISK_FREE_RATE", 0.0), 0.0)
    for _, row in active.iterrows():
        pair = str(row["pair"])
        rt = ratchet_by_pair.get(pair, {})
        model = model_results.get(pair)
        model_ok = model is not None and not model.bt.empty
        risk = compute_risk_metrics(model.bt["equity"], risk_free_rate=risk_free) if model_ok else {}
        hm = np.nan
        if model is not None and not model.h_signal.empty:
            _h = model.h_signal.dropna()
            hm = _safe_last(_h[_h.index <= pd.Timestamp(run_date)]) if len(_h) else np.nan
        status = model.status if model is not None else "missing"
        if status and status != "ok":
            status = "insuf_price" if str(status).startswith("insufficient_price") else str(status)[:12]
        sleeve = str(row.get("sleeve", "")).replace("_bucket4", " b4").replace("_bucket5", " b5")
        _act = _scalar_float(actual_by_pair.get(pair, np.nan), np.nan)
        cells.append([
            f"{row['etf']}/{row['underlying']}",
            sleeve,
            f"{float(row['gross_target_usd']):,.0f}",
            f"{_act:,.0f}" if np.isfinite(_act) else "-",
            f"{_scalar_float(rt.get('creep_ratio'), np.nan):.2f}" if rt else "-",
            f"{_scalar_float(rt.get('trim_lambda'), np.nan):.2f}" if rt else "-",
            f"{_scalar_float(rt.get('trim_usd'), 0.0):,.0f}" if rt else "-",
            f"{hm:.2f}" if np.isfinite(hm) else "-",
            f"{_scalar_float(risk.get('sharpe'), np.nan):.2f}" if risk else "-",
            f"{_scalar_float(risk.get('max_drawdown'), np.nan):.0%}" if risk else "-",
            "yes" if str(row["underlying"]) in shared_unds else "",
            status,
        ])

    n = len(cells)
    fig, ax = plt.subplots(figsize=(13, max(3.4, 2.0 + 0.34 * n)), constrained_layout=True)
    ax.axis("off")
    # Title + subtitle drawn in axes space above the table so they never collide.
    ax.text(
        0.0, 1.09,
        f"Bucket 4/5 per-pair overview  |  run {run_date}  |  {n} pair(s)",
        transform=ax.transAxes, va="bottom", ha="left", fontsize=13, fontweight="bold",
    )
    ax.text(
        0.0, 1.035,
        f"{model_tag}   |   pages follow, one per pair (sorted by proposed gross). "
        "Actual $ = accounting cumulative PnL (matches the EOD email). "
        "Shared = underlying netted with B1/B2/B5 (attributed slice).",
        transform=ax.transAxes, va="bottom", ha="left", fontsize=7.5, color="#555555",
    )
    tbl = ax.table(cellText=cells, colLabels=headers, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.0, 1.35)
    for c in range(len(headers)):
        hdr = tbl[0, c]
        hdr.set_facecolor("#1f2937")
        hdr.set_text_props(color="white", fontweight="bold")
    for r in range(1, n + 1):
        rt = ratchet_by_pair.get(str(active.iloc[r - 1]["pair"]), {})
        released = bool(rt.get("released", False)) if rt else False
        base = "#ecfdf5" if released else ("#f9fafb" if r % 2 else "#ffffff")
        for c in range(len(headers)):
            tbl[r, c].set_facecolor(base)
            tbl[r, c].set_edgecolor("#e5e7eb")
    pdf.savefig(fig)
    plt.close(fig)


def make_b4_pair_pnl_hedge_chart(
    run_date: str,
    *,
    runs_root: Path,
    out_dir: Path,
    metrics_csv: Path | None = None,
    vol_shape_json: Path | None = None,
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
    ratchet_by_pair = load_ratchet_targets(runs_root, run_date)
    shared_unds = load_shared_underlyings(runs_root, run_date, active)
    if shared_unds:
        print(f"[B4-pair-charts] shared underlyings (attributed slice): {sorted(shared_unds)}")
    model_inputs = resolve_b4_model_inputs(
        run_date,
        runs_root=runs_root,
        metrics_path=metrics_csv,
        vol_shape_path=vol_shape_json,
    )
    if model_inputs.diagnostics:
        print("[B4-pair-charts] model input diagnostics: " + " | ".join(model_inputs.diagnostics))
    print(
        "[B4-pair-charts] model inputs: "
        f"metrics={model_inputs.metrics.name} ({model_inputs.metrics_source}); "
        f"vol_shape={model_inputs.vol_shape.name if model_inputs.vol_shape is not None else 'none'} "
        f"({model_inputs.vol_shape_source})"
    )
    model_results, model_tag = build_model_pair_results(
        active,
        metrics_csv=model_inputs.metrics,
        vol_shape_json=model_inputs.vol_shape,
        start=MODEL_START,
        metrics_source=model_inputs.metrics_source,
        vol_shape_source=model_inputs.vol_shape_source,
        diagnostics=model_inputs.diagnostics,
    )

    # Latest charted endpoint per pair = the figure the EOD email reports. Build it
    # once for the overview table and to reconcile against the run's accounting.
    actual_by_pair: dict[str, float] = {}
    if not hist.empty and "pair" in hist.columns:
        for _p, _g in hist.groupby("pair"):
            actual_by_pair[str(_p)] = _safe_last(_g.sort_values("date")["pair_pnl_cum"])
    _reconcile_actual_endpoints(actual_by_pair, runs_root=runs_root, run_date=run_date)

    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / f"b4_pair_pnl_hedge_{run_date}.pdf"
    csv_path = out_dir / f"b4_pair_pnl_hedge_summary_{run_date}.csv"
    summary_rows: list[dict] = []
    with PdfPages(pdf_path) as pdf:
        _plot_summary_page(
            pdf, active,
            run_date=run_date, model_tag=model_tag,
            ratchet_by_pair=ratchet_by_pair, shared_unds=shared_unds,
            model_results=model_results,
            actual_by_pair=actual_by_pair,
        )
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
                    ratchet_by_pair=ratchet_by_pair,
                    shared_unds=shared_unds,
                )
            )
        meta = pdf.infodict()
        meta["Title"] = f"Bucket 4/5 per-pair PnL + hedge ({run_date})"
        meta["Subject"] = "B4/B5 pair PnL, hedge ratio, gross exposure, ratchet/continuous-trim audit"
        meta["Author"] = "ls-algo EOD"
    pd.DataFrame(summary_rows).to_csv(csv_path, index=False)
    print(f"[B4-pair-charts] wrote {pdf_path.name} ({len(summary_rows)} pairs) + {csv_path.name}")
    return pdf_path, csv_path


def _knob_tag() -> str:
    try:
        return _load_b4_backtest_config().tag
    except Exception:
        return "model h: config unavailable"
