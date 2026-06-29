"""Vol-shape trend-ratio signal-quality study (Section A).

Research-only. Does NOT modify production config or data. Writes diagnostics to
``notebooks/output/vol_shape_signal_study/``.

Questions answered
------------------
A2  Does the forward-looking trend ratio (``tr_est`` = ``trend_ratio_fwd``)
    predict forward-realized trend ratio AND forward decay capture better than
    the raw trend ratio (``und_trend_ratio_60d``)?  Measured by Spearman IC,
    time-series IC (per name, across time) and cross-sectional IC (per date,
    across names -- the form the production sizing tilt actually consumes).

A1  Does a 20d + 60d horizon blend of ``tr_est`` improve predictive IC versus
    the 60d-only signal currently in production?

A3  If we re-fit the evidence-blend weights inside
    ``trend_regime_estimator_from_returns`` to maximize *out-of-sample*
    predictive IC, how much lift is there over the hand-set production weights,
    and is it stable across time?

Offline data sources (no network required)
------------------------------------------
* ``data/cache/beta_history/<SYM>.csv`` -> underlying total-return proxy
  (adjusted close). Drives the forward-TR / forward-RV prediction study.
* ``data/runs/<date>/all_pairs_with_deltas.csv`` -> ETF -> (Underlying, Delta).
* ``data/runs/<date>/model_inputs/etf_metrics_daily.parquet`` -> aligned
  ``etf_adj_close`` / ``underlying_adj_close`` per ETF. Drives the production-
  faithful forward decay-capture study.

Run
---
    python scripts/vol_shape_signal_study.py
    python scripts/vol_shape_signal_study.py --max-names 60   # faster smoke run
"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# Reuse the exact production helpers so the estimator replica cannot drift.
from vol_shape import (  # noqa: E402
    TRADING_DAYS,
    _clip,
    _clip01,
    _regime_persistence,
    _trend_r2,
    trend_regime_estimator_from_returns,
)

# Production evidence-blend weights (vol_shape.trend_regime_estimator_from_returns).
# Order: tr_z, eff_z, cons_z, r2_z, persistence_z, jump_penalty.
COMP_NAMES = ["tr_z", "eff_z", "cons_z", "r2_z", "persistence_z", "jump_penalty"]
PROD_WEIGHTS = np.array([0.40, 0.22, 0.20, 0.20, 0.22, -0.35], dtype=float)

MIN_XSEC_NAMES = 8       # min names per date for a cross-sectional IC point
MIN_TS_OBS = 60          # min time-series obs per name for a time-series IC point
MIN_POOLED = 200         # min obs for a pooled IC


# ---------------------------------------------------------------------------
# Vol-shape primitives (mirror vol_shape.py exactly; self-checked below)
# ---------------------------------------------------------------------------
def _tr_vcr_rv(tail: np.ndarray) -> tuple[float, float, float]:
    """Return (raw trend_ratio, vcr, annualized daily RV) for a return window.

    ``tail`` length must be a positive multiple of 5 (weekly blocks).
    """
    n = tail.size
    sq = tail * tail
    sum_sq = float(np.sum(sq))
    if not np.isfinite(sum_sq) or sum_sq <= 0:
        return np.nan, np.nan, np.nan
    rv_daily = float(np.sqrt((sum_sq / n) * TRADING_DAYS))
    weekly = tail.reshape(n // 5, 5).sum(axis=1)
    rv_weekly = float(np.sqrt(np.mean(weekly ** 2) * (TRADING_DAYS / 5.0)))
    tr = rv_weekly / rv_daily if rv_daily > 0 else np.nan
    vcr = float(np.max(sq) / sum_sq)
    return tr, vcr, rv_daily


def regime_components(tail: np.ndarray, trend_ratio: float, vcr: float) -> dict[str, float]:
    """Component z-scores feeding the production evidence blend.

    Faithful copy of the z-score construction in
    ``vol_shape.trend_regime_estimator_from_returns`` (lines ~157-198) so we can
    recombine components with arbitrary weights for A3.
    """
    a = np.asarray(tail, dtype=float)
    a = a[np.isfinite(a)]
    n = int(a.size)
    if n < 5:
        return {}
    sq = a * a
    sum_sq = float(np.sum(sq))
    if not np.isfinite(sum_sq) or sum_sq <= 0:
        return {}

    tr = float(trend_ratio) if np.isfinite(trend_ratio) else 1.0
    vc = float(vcr) if np.isfinite(vcr) else float(np.max(sq) / sum_sq)
    efficiency = _clip01(abs(float(np.sum(a))) / np.sqrt(sum_sq * n))
    signs = np.sign(a[np.abs(a) > 1e-12])
    consistency = _clip01(abs(float(np.mean(signs)))) if signs.size else 0.0
    r2 = _trend_r2(a)
    persistence = _regime_persistence(a)

    random_eff = float(np.sqrt(2.0 / np.pi) / np.sqrt(n))
    random_consistency = random_eff
    sample_conf = _clip01(np.sqrt(n / 60.0))
    jump_floor = max(0.15, 3.0 / n)
    jump_penalty = _clip01((vc - jump_floor) / 0.35)

    tr_shrunk = 1.0 + (tr - 1.0) * sample_conf * (1.0 - 0.55 * jump_penalty)
    tr_z = _clip((tr_shrunk - 1.0) / 0.18, -3.0, 3.0)
    eff_z = _clip((efficiency - 1.5 * random_eff) / 0.25, -3.0, 3.0)
    cons_z = _clip((consistency - 1.5 * random_consistency) / 0.25, -3.0, 3.0)
    r2_z = _clip((r2 - 0.35) / 0.25, -3.0, 3.0)
    persistence_z = _clip((persistence - 0.50) / 0.25, -3.0, 3.0)
    confidence = _clip01(sample_conf * (1.0 - 0.45 * jump_penalty))

    return {
        "tr_z": tr_z,
        "eff_z": eff_z,
        "cons_z": cons_z,
        "r2_z": r2_z,
        "persistence_z": persistence_z,
        "jump_penalty": jump_penalty,
        "confidence": confidence,
    }


def evidence_to_tr_est(evidence: np.ndarray, confidence: np.ndarray) -> np.ndarray:
    """Map an evidence score to the bounded forward TR exactly like production."""
    scaled = evidence * (0.50 + 0.50 * confidence)
    return 1.0 + 0.22 * np.tanh(scaled / 2.0)  # rank-equivalent to the clipped value


def _self_check() -> None:
    """Assert the component replica reproduces the production tr_est."""
    rng = np.random.default_rng(0)
    for _ in range(5):
        tail = rng.normal(0, 0.02, size=60)
        tr, vcr, _ = _tr_vcr_rv(tail)
        comp = regime_components(tail, tr, vcr)
        ev = float(np.dot(PROD_WEIGHTS, [comp[c] for c in COMP_NAMES]))
        tr_est_replica = float(_clip(evidence_to_tr_est(np.array([ev]), np.array([comp["confidence"]]))[0], 0.72, 1.28))
        prod = trend_regime_estimator_from_returns(tail, trend_ratio=tr, vcr=vcr)["trend_ratio_fwd"]
        assert abs(tr_est_replica - float(prod)) < 1e-9, (tr_est_replica, prod)


# ---------------------------------------------------------------------------
# Price loading
# ---------------------------------------------------------------------------
def _norm_sym(x: str) -> str:
    return str(x).strip().upper().replace(".", "-")


def load_beta_history(cache_dir: Path, max_names: int | None) -> dict[str, pd.Series]:
    """Load underlying adjusted-close series from the beta_history cache."""
    out: dict[str, pd.Series] = {}
    files = sorted(cache_dir.glob("*.csv"))
    for f in files:
        sym = f.stem
        try:
            df = pd.read_csv(f)
            if "close" not in df.columns or "date" not in df.columns:
                continue
            idx = pd.to_datetime(df["date"], utc=True, errors="coerce").dt.tz_localize(None).dt.normalize()
            s = pd.Series(pd.to_numeric(df["close"], errors="coerce").to_numpy(), index=idx)
            s = s[~s.index.isna()].dropna()
            s = s[~s.index.duplicated(keep="last")].sort_index()
            if len(s) >= 200:
                out[sym] = s
        except Exception as exc:  # pragma: no cover - defensive
            print(f"[study] skip {sym}: {exc}")
        if max_names and len(out) >= max_names:
            break
    return out


# ---------------------------------------------------------------------------
# Per-name rolling signals + forward targets (integer-indexed -> no leakage)
# ---------------------------------------------------------------------------
def build_name_frame(
    name: str,
    prices: pd.Series,
    windows: tuple[int, ...] = (20, 60),
    horizons: tuple[int, ...] = (20, 60),
    want_components: bool = True,
) -> pd.DataFrame | None:
    """Tidy per-name frame keyed by the signal *as-of* date.

    Each row's signals use returns up to and including the as-of date; forward
    targets use returns strictly after it (no look-ahead).
    """
    s = pd.to_numeric(prices, errors="coerce").dropna().sort_index()
    log_r = np.log(s / s.shift(1)).replace([np.inf, -np.inf], np.nan).dropna()
    r = log_r.to_numpy(dtype=float)
    dates = log_r.index
    n = r.size
    wmax = max(windows)
    hmax = max(horizons)
    if n < wmax + hmax + 5:
        return None

    rows: list[dict] = []
    # as-of return index k: needs >= wmax returns ending at k, and forward room.
    for k in range(wmax - 1, n):
        row: dict = {"name": name, "date": dates[k]}
        ok = False
        for w in windows:
            tail = r[k - w + 1 : k + 1]
            if tail.size != w:
                continue
            tr, vcr, _ = _tr_vcr_rv(tail)
            row[f"tr_raw{w}"] = tr
            row[f"vcr{w}"] = vcr
            # Compute components once and derive tr_est from them (avoids the
            # heavyweight estimator's extra outputs). Equals production tr_est
            # to <1e-9 -- see _self_check().
            comp = regime_components(tail, tr, vcr)
            if comp:
                ev = float(np.dot(PROD_WEIGHTS, [comp[c] for c in COMP_NAMES]))
                row[f"tr_est{w}"] = _clip(
                    float(evidence_to_tr_est(np.array([ev]), np.array([comp["confidence"]]))[0]),
                    0.72, 1.28,
                )
                if w == 60 and want_components:
                    for c in COMP_NAMES + ["confidence"]:
                        row[c] = comp.get(c, np.nan)
            else:
                row[f"tr_est{w}"] = np.nan
            ok = True
        for h in horizons:
            fwd = r[k + 1 : k + 1 + h]
            if fwd.size == h:
                ftr, _, frv = _tr_vcr_rv(fwd)
                row[f"fwd_tr{h}"] = ftr
                row[f"fwd_rv{h}"] = frv
            else:
                row[f"fwd_tr{h}"] = np.nan
                row[f"fwd_rv{h}"] = np.nan
        if ok:
            rows.append(row)
    if not rows:
        return None
    return pd.DataFrame(rows)


def build_pool(prices: dict[str, pd.Series], **kw) -> pd.DataFrame:
    frames = []
    for i, (name, s) in enumerate(prices.items(), 1):
        fr = build_name_frame(name, s, **kw)
        if fr is not None:
            frames.append(fr)
        if i % 25 == 0:
            print(f"[study] rolling signals: {i}/{len(prices)} names")
    pool = pd.concat(frames, ignore_index=True)
    pool["date"] = pd.to_datetime(pool["date"])
    return pool


# ---------------------------------------------------------------------------
# IC machinery (Spearman = Pearson on ranks)
# ---------------------------------------------------------------------------
def _spearman(x: np.ndarray, y: np.ndarray, min_n: int) -> tuple[float, int]:
    m = np.isfinite(x) & np.isfinite(y)
    nn = int(m.sum())
    if nn < min_n:
        return np.nan, nn
    xr = pd.Series(x[m]).rank().to_numpy()
    yr = pd.Series(y[m]).rank().to_numpy()
    if np.std(xr) == 0 or np.std(yr) == 0:
        return np.nan, nn
    return float(np.corrcoef(xr, yr)[0, 1]), nn


def pooled_ic(pool: pd.DataFrame, sig: str, tgt: str) -> tuple[float, int]:
    return _spearman(pool[sig].to_numpy(dtype=float), pool[tgt].to_numpy(dtype=float), MIN_POOLED)


def timeseries_ic(pool: pd.DataFrame, sig: str, tgt: str) -> dict:
    rhos = []
    for _, g in pool.groupby("name"):
        rho, nn = _spearman(g[sig].to_numpy(dtype=float), g[tgt].to_numpy(dtype=float), MIN_TS_OBS)
        if np.isfinite(rho):
            rhos.append(rho)
    a = np.asarray(rhos, dtype=float)
    if a.size == 0:
        return {"mean": np.nan, "median": np.nan, "share_pos": np.nan, "tstat": np.nan, "n": 0}
    se = float(np.std(a, ddof=1) / np.sqrt(a.size)) if a.size > 1 else np.nan
    return {
        "mean": float(np.mean(a)),
        "median": float(np.median(a)),
        "share_pos": float(np.mean(a > 0)),
        "tstat": float(np.mean(a) / se) if se and se > 0 else np.nan,
        "n": int(a.size),
    }


def crosssectional_ic(pool: pd.DataFrame, sig: str, tgt: str) -> dict:
    rhos = []
    for _, g in pool.groupby("date"):
        rho, nn = _spearman(g[sig].to_numpy(dtype=float), g[tgt].to_numpy(dtype=float), MIN_XSEC_NAMES)
        if np.isfinite(rho):
            rhos.append(rho)
    a = np.asarray(rhos, dtype=float)
    if a.size == 0:
        return {"mean": np.nan, "median": np.nan, "share_pos": np.nan, "tstat": np.nan, "n": 0}
    se = float(np.std(a, ddof=1) / np.sqrt(a.size)) if a.size > 1 else np.nan
    return {
        "mean": float(np.mean(a)),
        "median": float(np.median(a)),
        "share_pos": float(np.mean(a > 0)),
        "tstat": float(np.mean(a) / se) if se and se > 0 else np.nan,
        "n": int(a.size),
    }


# ---------------------------------------------------------------------------
# A2: signal-vs-signal IC comparison
# ---------------------------------------------------------------------------
def run_a2(pool: pd.DataFrame, outdir: Path) -> pd.DataFrame:
    targets = ["fwd_tr60", "fwd_tr20", "fwd_rv60"]
    signals = ["tr_raw60", "tr_est60", "vcr60",
               "eff_z", "cons_z", "r2_z", "persistence_z"]
    recs = []
    for tgt in targets:
        for sig in signals:
            if sig not in pool.columns:
                continue
            p_ic, p_n = pooled_ic(pool, sig, tgt)
            ts = timeseries_ic(pool, sig, tgt)
            xs = crosssectional_ic(pool, sig, tgt)
            recs.append({
                "target": tgt, "signal": sig,
                "pooled_ic": p_ic, "pooled_n": p_n,
                "ts_ic_mean": ts["mean"], "ts_ic_tstat": ts["tstat"],
                "ts_share_pos": ts["share_pos"], "ts_names": ts["n"],
                "xs_ic_mean": xs["mean"], "xs_ic_tstat": xs["tstat"],
                "xs_share_pos": xs["share_pos"], "xs_dates": xs["n"],
            })
    res = pd.DataFrame(recs)
    res.to_csv(outdir / "a2_ic_table.csv", index=False)
    return res


# ---------------------------------------------------------------------------
# A1: horizon blend
# ---------------------------------------------------------------------------
def run_a1(pool: pd.DataFrame, outdir: Path, base: str = "tr_est", target: str = "fwd_tr60") -> pd.DataFrame:
    sub = pool.dropna(subset=[f"{base}20", f"{base}60", target]).copy()
    recs = []
    for w in np.round(np.arange(0.0, 1.0001, 0.1), 2):
        blended = w * sub[f"{base}20"].to_numpy(dtype=float) + (1 - w) * sub[f"{base}60"].to_numpy(dtype=float)
        tmp = sub.assign(_blend=blended)
        xs = crosssectional_ic(tmp, "_blend", target)
        ts = timeseries_ic(tmp, "_blend", target)
        p_ic, _ = pooled_ic(tmp, "_blend", target)
        recs.append({
            "w20": w, "pooled_ic": p_ic,
            "xs_ic_mean": xs["mean"], "xs_ic_tstat": xs["tstat"],
            "ts_ic_mean": ts["mean"], "ts_ic_tstat": ts["tstat"],
        })
    res = pd.DataFrame(recs)
    res.to_csv(outdir / f"a1_horizon_blend_{base}_{target}.csv", index=False)
    return res


# ---------------------------------------------------------------------------
# A3: re-fit evidence weights with a temporal train/test split
# ---------------------------------------------------------------------------
def _evidence(df: pd.DataFrame, weights: np.ndarray) -> np.ndarray:
    X = df[COMP_NAMES].to_numpy(dtype=float)
    return X @ weights


def _xsec_rank_target(df: pd.DataFrame, target: str) -> pd.Series:
    """Within-date percentile rank of the forward target (the thing to predict)."""
    return df.groupby("date")[target].rank(pct=True)


def run_a3(pool: pd.DataFrame, outdir: Path, target: str = "fwd_tr60", n_folds: int = 4) -> dict:
    cols = COMP_NAMES + ["date", target]
    df = pool.dropna(subset=cols).copy()
    df["_yrank"] = _xsec_rank_target(df, target)
    df = df.dropna(subset=["_yrank"])
    df = df.sort_values("date").reset_index(drop=True)

    # Ordered unique dates -> temporal folds (expanding-window walk-forward).
    udates = np.array(sorted(df["date"].unique()))
    fold_edges = np.linspace(0, len(udates), n_folds + 1, dtype=int)

    def fit_weights(train: pd.DataFrame) -> np.ndarray:
        X = np.column_stack([train[COMP_NAMES].to_numpy(dtype=float), np.ones(len(train))])
        y = train["_yrank"].to_numpy(dtype=float)
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        return beta[:-1]  # drop intercept (irrelevant for cross-sectional rank)

    wf_rows = []
    refit_ic, prodlin_ic, full_ic = [], [], []
    for fi in range(1, n_folds):  # expanding window: train [0:edge], test next block
        tr_dates = set(udates[: fold_edges[fi]])
        te_dates = set(udates[fold_edges[fi]: fold_edges[fi + 1]])
        if not te_dates:
            continue
        train = df[df["date"].isin(tr_dates)]
        test = df[df["date"].isin(te_dates)]
        if len(train) < 500 or len(test) < 200:
            continue
        w_fit = fit_weights(train)

        te = test.copy()
        te["_ev_refit"] = _evidence(te, w_fit)
        te["_ev_prod"] = _evidence(te, PROD_WEIGHTS)
        ic_refit = crosssectional_ic(te.assign(s=te["_ev_refit"]), "s", target)["mean"]
        ic_prod = crosssectional_ic(te.assign(s=te["_ev_prod"]), "s", target)["mean"]
        ic_full = crosssectional_ic(te, "tr_est60", target)["mean"] if "tr_est60" in te else np.nan
        refit_ic.append(ic_refit)
        prodlin_ic.append(ic_prod)
        full_ic.append(ic_full)
        wf_rows.append({
            "fold": fi,
            "train_to": str(pd.Timestamp(udates[fold_edges[fi] - 1]).date()),
            "test_from": str(pd.Timestamp(udates[fold_edges[fi]]).date()),
            "test_to": str(pd.Timestamp(udates[fold_edges[fi + 1] - 1]).date()),
            "n_train": len(train), "n_test": len(test),
            "ic_prod_linear": ic_prod, "ic_refit": ic_refit, "ic_full_tr_est": ic_full,
            **{f"w_{c}": w_fit[j] for j, c in enumerate(COMP_NAMES)},
        })

    wf = pd.DataFrame(wf_rows)
    wf.to_csv(outdir / f"a3_walkforward_{target}.csv", index=False)

    # Final reference fit on the full sample for reporting the suggested weights.
    w_full = fit_weights(df)
    # Normalize to the same L1 magnitude as production for human comparison.
    scale = float(np.sum(np.abs(PROD_WEIGHTS)) / max(np.sum(np.abs(w_full)), 1e-9))
    w_full_norm = w_full * scale
    weights_tbl = pd.DataFrame({
        "component": COMP_NAMES,
        "prod_weight": PROD_WEIGHTS,
        "refit_weight_raw": w_full,
        "refit_weight_norm": w_full_norm,
    })
    weights_tbl.to_csv(outdir / f"a3_weights_{target}.csv", index=False)

    return {
        "oos_ic_prod_linear": float(np.nanmean(prodlin_ic)) if prodlin_ic else np.nan,
        "oos_ic_refit": float(np.nanmean(refit_ic)) if refit_ic else np.nan,
        "oos_ic_full_tr_est": float(np.nanmean(full_ic)) if full_ic else np.nan,
        "weights_tbl": weights_tbl,
        "walkforward": wf,
    }


# ---------------------------------------------------------------------------
# Decay-capture study (production-faithful, joint-metrics basis)
# ---------------------------------------------------------------------------
def run_decay_study(parquet: Path, pairs_csv: Path, outdir: Path, horizon: int = 60,
                    min_obs: int = 260) -> pd.DataFrame | None:
    if not parquet.exists() or not pairs_csv.exists():
        print(f"[study] decay study skipped (missing {parquet.name} or {pairs_csv.name})")
        return None
    pairs = pd.read_csv(pairs_csv)
    pairs["ETF"] = pairs["ETF"].map(_norm_sym)
    beta_map = dict(zip(pairs["ETF"], pd.to_numeric(pairs["Delta"], errors="coerce")))

    cols = ["date", "ticker", "etf_adj_close", "underlying_adj_close"]
    md = pd.read_parquet(parquet, columns=cols)
    md["ticker"] = md["ticker"].map(_norm_sym)
    md = md[md["ticker"].isin(beta_map.keys())].copy()
    md["date"] = pd.to_datetime(md["date"], errors="coerce").dt.normalize()
    md = md.dropna(subset=["date"]).sort_values(["ticker", "date"])

    frames = []
    for etf, g in md.groupby("ticker"):
        beta = beta_map.get(etf)
        if beta is None or not np.isfinite(beta):
            continue
        g = g.dropna(subset=["etf_adj_close", "underlying_adj_close"])
        if len(g) < min_obs:
            continue
        und = pd.Series(g["underlying_adj_close"].to_numpy(dtype=float), index=g["date"])
        etfp = pd.Series(g["etf_adj_close"].to_numpy(dtype=float), index=g["date"])
        r_und = np.log(und / und.shift(1)).to_numpy()
        r_etf = np.log(etfp / etfp.shift(1)).to_numpy()
        drag = beta * r_und - r_etf  # >0 favors the ETF short (realized decay)
        rr = np.log(und / und.shift(1)).replace([np.inf, -np.inf], np.nan).to_numpy()
        dates = und.index
        n = len(dates)
        recs = []
        for k in range(60, n):  # as-of date index k (need 60 trailing returns)
            tail = rr[k - 59: k + 1]
            if np.isnan(tail).any():
                continue
            tr, vcr, _ = _tr_vcr_rv(tail)
            est = trend_regime_estimator_from_returns(tail, trend_ratio=tr, vcr=vcr)
            fwd = drag[k + 1: k + 1 + horizon]
            if fwd.size != horizon or np.isnan(fwd).any():
                continue
            recs.append({
                "etf": etf, "date": dates[k],
                "tr_raw60": tr, "tr_est60": est["trend_ratio_fwd"],
                "fwd_decay_annual": float(np.mean(fwd) * TRADING_DAYS),
            })
        if recs:
            frames.append(pd.DataFrame(recs))
    if not frames:
        print("[study] decay study: no usable pairs")
        return None
    pool = pd.concat(frames, ignore_index=True)
    pool["date"] = pd.to_datetime(pool["date"])

    recs = []
    for sig in ["tr_raw60", "tr_est60"]:
        p_ic, p_n = pooled_ic(pool, sig, "fwd_decay_annual")
        xs = crosssectional_ic(pool, sig, "fwd_decay_annual")
        ts = timeseries_ic(pool.rename(columns={"etf": "name"}), sig, "fwd_decay_annual")
        recs.append({
            "signal": sig, "pooled_ic": p_ic, "pooled_n": p_n,
            "xs_ic_mean": xs["mean"], "xs_ic_tstat": xs["tstat"], "xs_dates": xs["n"],
            "ts_ic_mean": ts["mean"], "ts_ic_tstat": ts["tstat"], "ts_names": ts["n"],
        })
    res = pd.DataFrame(recs)
    res.to_csv(outdir / "decay_capture_ic.csv", index=False)
    pool.to_parquet(outdir / "decay_pool.parquet")
    return res


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def _fmt(v, nd=4):
    return "nan" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:+.{nd}f}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-dir", type=Path, default=REPO / "data/cache/beta_history")
    ap.add_argument("--run-date", default="2026-06-25")
    ap.add_argument("--max-names", type=int, default=None, help="limit underlyings (smoke test)")
    ap.add_argument("--outdir", type=Path, default=REPO / "notebooks/output/vol_shape_signal_study")
    ap.add_argument("--skip-decay", action="store_true")
    args = ap.parse_args(argv)

    _self_check()
    args.outdir.mkdir(parents=True, exist_ok=True)
    print(f"[study] output -> {args.outdir}")

    prices = load_beta_history(args.cache_dir, args.max_names)
    print(f"[study] loaded {len(prices)} underlying series")
    pool = build_pool(prices)
    print(f"[study] pooled obs: {len(pool):,}  names: {pool['name'].nunique()}  "
          f"dates: {pool['date'].min().date()}..{pool['date'].max().date()}")
    pool.to_parquet(args.outdir / "signal_pool.parquet")

    a2 = run_a2(pool, args.outdir)
    a1_est = run_a1(pool, args.outdir, base="tr_est", target="fwd_tr60")
    a1_raw = run_a1(pool, args.outdir, base="tr_raw", target="fwd_tr60")
    a3 = run_a3(pool, args.outdir, target="fwd_tr60")

    decay = None
    if not args.skip_decay:
        parquet = REPO / f"data/runs/{args.run_date}/model_inputs/etf_metrics_daily.parquet"
        pairs_csv = REPO / f"data/runs/{args.run_date}/all_pairs_with_deltas.csv"
        decay = run_decay_study(parquet, pairs_csv, args.outdir)

    _write_markdown(args.outdir, pool, a2, a1_est, a1_raw, a3, decay)
    print(f"[study] DONE. See {args.outdir / 'REPORT.md'}")
    return 0


def _write_markdown(outdir, pool, a2, a1_est, a1_raw, a3, decay):
    lines: list[str] = []
    L = lines.append
    L("# Vol-shape trend-ratio signal-quality study (Section A)\n")
    L(f"- Underlyings: **{pool['name'].nunique()}**, pooled obs: **{len(pool):,}**, "
      f"window: 2021-06..2026 (per-name history).")
    L("- IC = Spearman rank correlation. `xs` = cross-sectional (per date, across names; "
      "this is the form the sizing tilt consumes). `ts` = time-series (per name, across time; "
      "the form the cadence engine consumes).\n")

    L("## A2 - Does forward `tr_est` predict better than raw TR?\n")
    L("| target | signal | pooled IC | xs IC (t) | ts IC (t) |")
    L("|---|---|---:|---:|---:|")
    for _, r in a2.iterrows():
        L(f"| {r['target']} | {r['signal']} | {_fmt(r['pooled_ic'])} | "
          f"{_fmt(r['xs_ic_mean'])} ({_fmt(r['xs_ic_tstat'],1)}) | "
          f"{_fmt(r['ts_ic_mean'])} ({_fmt(r['ts_ic_tstat'],1)}) |")
    L("")

    L("## A1 - Horizon blend  w*tr_est20 + (1-w)*tr_est60  vs fwd_tr60\n")
    L("| w20 | pooled IC | xs IC (t) | ts IC (t) |")
    L("|---:|---:|---:|---:|")
    for _, r in a1_est.iterrows():
        L(f"| {r['w20']:.1f} | {_fmt(r['pooled_ic'])} | {_fmt(r['xs_ic_mean'])} ({_fmt(r['xs_ic_tstat'],1)}) "
          f"| {_fmt(r['ts_ic_mean'])} ({_fmt(r['ts_ic_tstat'],1)}) |")
    L("\n(`a1_horizon_blend_tr_raw_fwd_tr60.csv` has the raw-TR blend for comparison.)\n")

    L("## A3 - Re-fit evidence weights (walk-forward, target fwd_tr60)\n")
    L(f"- Out-of-sample mean xs IC, production linear weights: **{_fmt(a3['oos_ic_prod_linear'])}**")
    L(f"- Out-of-sample mean xs IC, refit weights: **{_fmt(a3['oos_ic_refit'])}**")
    L(f"- Out-of-sample mean xs IC, full production tr_est: **{_fmt(a3['oos_ic_full_tr_est'])}**\n")
    L("Suggested weights (full-sample fit, L1-normalized to production scale):\n")
    L("| component | prod | refit (norm) |")
    L("|---|---:|---:|")
    for _, r in a3["weights_tbl"].iterrows():
        L(f"| {r['component']} | {_fmt(r['prod_weight'],2)} | {_fmt(r['refit_weight_norm'],3)} |")
    L("")

    if decay is not None:
        L("## Decay-capture target (production joint-metrics basis)\n")
        L("Sign note: a *negative* IC supports the thesis (lower TR -> more forward decay).\n")
        L("| signal | pooled IC | xs IC (t) | ts IC (t) |")
        L("|---|---:|---:|---:|")
        for _, r in decay.iterrows():
            L(f"| {r['signal']} | {_fmt(r['pooled_ic'])} | {_fmt(r['xs_ic_mean'])} ({_fmt(r['xs_ic_tstat'],1)}) "
              f"| {_fmt(r['ts_ic_mean'])} ({_fmt(r['ts_ic_tstat'],1)}) |")
        L("")

    (outdir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
