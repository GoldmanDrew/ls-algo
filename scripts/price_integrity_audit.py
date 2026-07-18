"""Audit metrics ETF closes vs a Yahoo referee; flag phantom jumps.

Usage
-----
    python -m scripts.price_integrity_audit --run-date 2026-07-10
    python -m scripts.price_integrity_audit --run-date 2026-07-10 --held-from notebooks/output/production_actual_bt/pair_daily_pnl.csv
    python -m scripts.price_integrity_audit --write-patches data/price_patches.csv --min-jump 0.5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _norm(x: object) -> str:
    return str(x).strip().upper().replace(".", "-")


def _yahoo_close(symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    try:
        from scripts.bucket4_price_loading import load_single_close

        s = load_single_close(
            _norm(symbol).replace("-", "."),
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


def load_metrics_etf_closes(run_date: str, *, repo: Path | None = None) -> pd.DataFrame:
    root = repo or REPO
    pq = root / f"data/runs/{run_date}/model_inputs/etf_metrics_daily.parquet"
    md = pd.read_parquet(pq, columns=["date", "ticker", "etf_adj_close", "underlying_adj_close"])
    md["date"] = pd.to_datetime(md["date"], errors="coerce").dt.normalize()
    md["ticker"] = md["ticker"].map(_norm)
    md["etf_adj_close"] = pd.to_numeric(md["etf_adj_close"], errors="coerce")
    md["underlying_adj_close"] = pd.to_numeric(md["underlying_adj_close"], errors="coerce")
    return md.dropna(subset=["date", "ticker"])


def audit_ticker(
    md: pd.DataFrame,
    etf: str,
    *,
    start_floor: str = "2026-02-01",
    jump_thresh: float = 0.5,
    yahoo_band: float = 0.15,
) -> dict:
    x = md[md["ticker"] == _norm(etf)].sort_values("date").reset_index(drop=True)
    out = {
        "etf": _norm(etf),
        "status": "ok",
        "n": int(len(x)),
        "n_big50": 0,
        "worst_abs_ret": np.nan,
        "worst_date": "",
        "yahoo_ov": 0,
        "yahoo_gt15pct": 0,
        "max_abs_ratio_m1": np.nan,
        "phantom_days": 0,
    }
    if x.empty:
        out["status"] = "missing_metrics"
        return out
    a = x["etf_adj_close"]
    ret = a.pct_change()
    absret = ret.abs()
    out["n_big50"] = int((absret > jump_thresh).fillna(False).sum())
    if absret.notna().any():
        wi = int(absret.idxmax())
        out["worst_abs_ret"] = float(absret.iloc[wi])
        out["worst_date"] = str(pd.Timestamp(x.loc[wi, "date"]).date())
    start = max(pd.Timestamp(x["date"].min()), pd.Timestamp(start_floor))
    end = pd.Timestamp(x["date"].max())
    y = _yahoo_close(etf, start, end)
    if y.empty:
        return out
    m = pd.Series(a.values, index=pd.DatetimeIndex(pd.to_datetime(x["date"])).normalize())
    j = pd.DataFrame({"m": m, "y": y}).dropna()
    j = j[(j["m"] > 0) & (j["y"] > 0)]
    out["yahoo_ov"] = int(len(j))
    if j.empty:
        return out
    ratio = j["m"] / j["y"]
    out["yahoo_gt15pct"] = int(((ratio - 1).abs() > yahoo_band).sum())
    out["max_abs_ratio_m1"] = float((ratio - 1).abs().max())
    # Phantom: metrics day return > jump while Yahoo |r| < 0.2 * metrics |r| (or Yahoo |r|<10%).
    phantom = 0
    m_ret = j["m"].pct_change()
    y_ret = j["y"].pct_change()
    for dt in j.index[1:]:
        mr = float(m_ret.loc[dt]) if np.isfinite(m_ret.loc[dt]) else 0.0
        yr = float(y_ret.loc[dt]) if np.isfinite(y_ret.loc[dt]) else 0.0
        if abs(mr) > jump_thresh and abs(yr) < max(0.10, 0.25 * abs(mr)):
            phantom += 1
    out["phantom_days"] = int(phantom)
    return out


def audit_universe(
    run_date: str,
    tickers: Iterable[str],
    *,
    repo: Path | None = None,
) -> pd.DataFrame:
    md = load_metrics_etf_closes(run_date, repo=repo)
    rows = [audit_ticker(md, t) for t in sorted({_norm(t) for t in tickers if _norm(t)})]
    return pd.DataFrame(rows)


def audit_sim_panel_ticker(
    panel_a: pd.Series,
    etf: str,
    *,
    jump_thresh: float = 0.5,
    scale_band: float = 0.20,
) -> dict:
    """Compare post-split/patch panel ETF closes to Yahoo (what the sim actually marks)."""
    out = {
        "etf": _norm(etf),
        "n_panel": int(len(panel_a.dropna())),
        "n_big50": 0,
        "worst_abs_ret": np.nan,
        "worst_date": "",
        "yahoo_ov": 0,
        "scale_bad_days": 0,
        "phantom_days": 0,
        "max_abs_ratio_m1": np.nan,
        "scale_mode": "",
        "issue": "",
    }
    a = pd.to_numeric(panel_a, errors="coerce").dropna().sort_index()
    if a.empty:
        out["issue"] = "empty_panel"
        return out
    ret = a.pct_change()
    absret = ret.abs()
    out["n_big50"] = int((absret > jump_thresh).fillna(False).sum())
    if absret.notna().any():
        wi = absret.idxmax()
        out["worst_abs_ret"] = float(absret.loc[wi])
        out["worst_date"] = str(pd.Timestamp(wi).date())
    y = _yahoo_close(etf, pd.Timestamp(a.index.min()) - pd.Timedelta(days=5), pd.Timestamp(a.index.max()))
    if y.empty:
        out["issue"] = "no_yahoo"
        return out
    j = pd.DataFrame({"m": a, "y": y}).dropna()
    j = j[(j["m"] > 0) & (j["y"] > 0)]
    out["yahoo_ov"] = int(len(j))
    if len(j) < 3:
        out["issue"] = "short_overlap"
        return out
    ratio = j["m"] / j["y"]
    out["max_abs_ratio_m1"] = float((ratio - 1).abs().max())
    out["scale_bad_days"] = int(((ratio - 1).abs() > scale_band).sum())
    # Detect stuck fractional scale (1/2, 1/3, 1/4, 1/5, 1/6, 1/8) then snap to 1.0
    med = float(ratio.median())
    for den in (2, 3, 4, 5, 6, 8, 10):
        if abs(med - 1.0 / den) < 0.05:
            out["scale_mode"] = f"1/{den}"
            break
    if not out["scale_mode"] and abs(med - 1.0) > scale_band:
        out["scale_mode"] = f"med={med:.3f}"
    phantom = 0
    m_ret = j["m"].pct_change()
    y_ret = j["y"].pct_change()
    for dt in j.index[1:]:
        mr = float(m_ret.loc[dt]) if np.isfinite(m_ret.loc[dt]) else 0.0
        yr = float(y_ret.loc[dt]) if np.isfinite(y_ret.loc[dt]) else 0.0
        if abs(mr) > jump_thresh and abs(yr) < max(0.10, 0.25 * abs(mr)):
            phantom += 1
    out["phantom_days"] = int(phantom)
    if out["phantom_days"] or out["scale_bad_days"] >= 3 or out["n_big50"] >= 1:
        bits = []
        if out["phantom_days"]:
            bits.append(f"phantom={out['phantom_days']}")
        if out["scale_bad_days"]:
            bits.append(f"scale_bad={out['scale_bad_days']}")
        if out["scale_mode"]:
            bits.append(f"mode={out['scale_mode']}")
        if out["n_big50"]:
            bits.append(f"jumps={out['n_big50']}@{out['worst_date']}")
        out["issue"] = ";".join(bits)
    return out


def audit_sim_panel(
    run_date: str,
    tickers: Iterable[str],
    *,
    repo: Path | None = None,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Build the sim price panel and audit each ticker vs Yahoo."""
    from scripts.pair_price_panel import frames_from_metrics, underlying_map_from_run

    root = repo or REPO
    md = load_metrics_etf_closes(run_date, repo=root)
    # Restrict metrics to requested tickers for speed.
    want = sorted({_norm(t) for t in tickers if _norm(t)})
    md = md[md["ticker"].isin(want)].copy()
    und_map = underlying_map_from_run(run_date, repo=root)
    panel = frames_from_metrics(
        md.rename(columns={"ticker": "ticker"}),
        min_days=20,
        ticker_col="ticker",
        date_col="date",
        etf_col="etf_adj_close",
        und_col="underlying_adj_close",
        underlying_by_etf=und_map,
        repo=root,
    )
    rows = []
    for etf in want:
        px = panel.get(etf)
        if px is None or px.empty:
            rows.append({"etf": etf, "issue": "not_in_panel", "n_panel": 0, "phantom_days": 0,
                         "scale_bad_days": 0, "n_big50": 0, "worst_abs_ret": np.nan,
                         "worst_date": "", "yahoo_ov": 0, "max_abs_ratio_m1": np.nan, "scale_mode": ""})
            continue
        rows.append(audit_sim_panel_ticker(px["a_px"], etf))
    return pd.DataFrame(rows), panel


def _expand_und_map(run_date: str, *, repo: Path | None = None) -> dict[str, str]:
    from scripts.pair_price_panel import underlying_map_from_run

    root = repo or REPO
    und_map = underlying_map_from_run(run_date, repo=root)
    # Broader book history (all pairs that traded in the sim).
    for rel in (
        "notebooks/output/production_actual_bt/pair_stats.csv",
        "notebooks/output/production_actual_bt/pair_daily_pnl.csv",
    ):
        p = root / rel
        if not p.is_file():
            continue
        try:
            df = pd.read_csv(p, usecols=lambda c: c in {"ETF", "Underlying"})
        except Exception:
            continue
        if "ETF" not in df.columns or "Underlying" not in df.columns:
            continue
        for _, r in df.iterrows():
            etf, und = _norm(r["ETF"]), _norm(r["Underlying"])
            if etf and und and etf not in und_map:
                und_map[etf] = und
    return und_map


def audit_underlyings(
    run_date: str,
    *,
    repo: Path | None = None,
    start_floor: str = "2026-02-01",
) -> pd.DataFrame:
    """Audit raw metrics underlying closes vs Yahoo (CVNA/IREN/ONDS-class bugs).

    Important: the same underlying can have *different* vendor series across ETF
    rows (CVNY clean, CVNX 5× then snap). Audit every ETF's und leg and keep the
    worst score per underlying.
    """
    root = repo or REPO
    md = load_metrics_etf_closes(run_date, repo=root)
    und_map = _expand_und_map(run_date, repo=root)
    by_und: dict[str, list[str]] = {}
    for etf, und in und_map.items():
        by_und.setdefault(und, []).append(etf)

    rows: list[dict] = []
    for und, etfs in sorted(by_und.items()):
        worst: dict | None = None
        for etf in sorted(etfs):
            x = md[md["ticker"] == etf].sort_values("date")
            if x.empty or x["underlying_adj_close"].notna().sum() < 20:
                continue
            series = pd.Series(
                pd.to_numeric(x["underlying_adj_close"], errors="coerce").to_numpy(dtype=float),
                index=pd.DatetimeIndex(pd.to_datetime(x["date"])).normalize(),
            )
            series = series.dropna()
            series = series[~series.index.duplicated(keep="last")].sort_index()
            # Slightly tighter scale band for stocks (vendor 5× / date shifts).
            row = audit_sim_panel_ticker(series, und, scale_band=0.15, jump_thresh=0.35)
            row["underlying"] = und
            row["src_etf"] = etf
            row.pop("etf", None)
            # Extra: return disagreement without requiring |m_ret|>jump_thresh.
            y = _yahoo_close(
                und,
                pd.Timestamp(series.index.min()) - pd.Timedelta(days=5),
                pd.Timestamp(series.index.max()),
            )
            disagree = 0
            if not y.empty:
                j = pd.DataFrame({"m": series, "y": y}).dropna()
                j = j[(j["m"] > 0) & (j["y"] > 0)]
                if len(j) >= 3:
                    mr = j["m"].pct_change()
                    yr = j["y"].pct_change()
                    for dt in j.index[1:]:
                        mrv = float(mr.loc[dt]) if np.isfinite(mr.loc[dt]) else 0.0
                        yrv = float(yr.loc[dt]) if np.isfinite(yr.loc[dt]) else 0.0
                        if abs(mrv) >= 0.08 and abs(mrv - yrv) >= 0.12:
                            disagree += 1
            row["ret_disagree_days"] = int(disagree)
            if disagree and not row.get("issue"):
                row["issue"] = f"ret_disagree={disagree}"
            elif disagree and row.get("issue"):
                row["issue"] = f"{row['issue']};ret_disagree={disagree}"
            if worst is None or _und_severity_score(row) > _und_severity_score(worst):
                worst = row
        if worst is None:
            rows.append(
                {
                    "underlying": und,
                    "src_etf": etfs[0] if etfs else "",
                    "issue": "no_metrics_und",
                    "n_panel": 0,
                    "phantom_days": 0,
                    "scale_bad_days": 0,
                    "n_big50": 0,
                    "ret_disagree_days": 0,
                    "worst_abs_ret": np.nan,
                    "worst_date": "",
                    "yahoo_ov": 0,
                    "max_abs_ratio_m1": np.nan,
                    "scale_mode": "",
                }
            )
        else:
            rows.append(worst)
    return pd.DataFrame(rows)


def _und_severity_score(row: dict) -> float:
    return float(
        1000 * int(row.get("phantom_days") or 0)
        + 10 * int(row.get("scale_bad_days") or 0)
        + 50 * int(row.get("ret_disagree_days") or 0)
        + int(row.get("n_big50") or 0)
        + 100 * float(row.get("max_abs_ratio_m1") or 0)
    )


def und_suspects_mask(df: pd.DataFrame) -> pd.Series:
    skip = {"", "no_metrics_und", "no_yahoo", "short_overlap", "empty_panel", "not_in_panel"}
    issue = df.get("issue", "").astype(str)
    return (
        (df.get("phantom_days", 0).fillna(0) >= 1)
        | (df.get("scale_bad_days", 0).fillna(0) >= 3)
        | (df.get("ret_disagree_days", 0).fillna(0) >= 2)
        | (df.get("n_big50", 0).fillna(0) >= 1)
        | (df.get("max_abs_ratio_m1", 0).fillna(0) > 0.40)
    ) & ~issue.isin(skip)


def build_yahoo_full_patches_for_tickers(
    tickers: Iterable[str],
    *,
    start: str = "2026-02-01",
    end: str = "2026-07-11",
    note: str = "sim_panel_scale_or_phantom",
) -> pd.DataFrame:
    """Overwrite entire Yahoo window for broken tickers (kills 1/N then snap bugs)."""
    rows: list[dict] = []
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    for etf in sorted({_norm(t) for t in tickers if _norm(t)}):
        y = _yahoo_close(etf, start_ts, end_ts)
        for dt, px in y.items():
            rows.append(
                {
                    "symbol": etf,
                    "date": str(pd.Timestamp(dt).date()),
                    "close": float(px),
                    "source": "yahoo_referee",
                    "note": note,
                }
            )
    return pd.DataFrame(rows)


def panel_suspects_mask(df: pd.DataFrame) -> pd.Series:
    return (
        (df.get("phantom_days", 0).fillna(0) >= 1)
        | (df.get("scale_bad_days", 0).fillna(0) >= 3)
        | (df.get("n_big50", 0).fillna(0) >= 1)
        | df.get("issue", "").astype(str).str.len().gt(0)
    ) & ~df.get("issue", "").astype(str).isin(["", "empty_panel", "no_yahoo", "short_overlap", "not_in_panel"])


def suspects_mask(df: pd.DataFrame) -> pd.Series:
    return (
        (df["n_big50"].fillna(0) >= 1)
        | (df["yahoo_gt15pct"].fillna(0) >= 5)
        | (df["max_abs_ratio_m1"].fillna(0) > 0.8)
        | (df["phantom_days"].fillna(0) >= 1)
    )


def build_referee_patches(
    run_date: str,
    tickers: Iterable[str],
    *,
    repo: Path | None = None,
    jump_thresh: float = 0.5,
    start_floor: str = "2026-02-01",
) -> pd.DataFrame:
    """For phantom jump days, emit Yahoo closes as patches."""
    root = repo or REPO
    md = load_metrics_etf_closes(run_date, repo=root)
    rows: list[dict] = []
    for etf in sorted({_norm(t) for t in tickers if _norm(t)}):
        x = md[md["ticker"] == etf].sort_values("date")
        if x.empty:
            continue
        start = max(pd.Timestamp(x["date"].min()), pd.Timestamp(start_floor))
        end = pd.Timestamp(x["date"].max())
        y = _yahoo_close(etf, start, end)
        if y.empty:
            continue
        m = pd.Series(
            pd.to_numeric(x["etf_adj_close"], errors="coerce").values,
            index=pd.DatetimeIndex(pd.to_datetime(x["date"])).normalize(),
        )
        j = pd.DataFrame({"m": m, "y": y}).dropna()
        j = j[(j["m"] > 0) & (j["y"] > 0)]
        if len(j) < 3:
            continue
        m_ret = j["m"].pct_change()
        y_ret = j["y"].pct_change()
        # Also patch sustained scale errors: |m/y-1| > 40% for a day.
        for dt in j.index:
            mr = float(m_ret.loc[dt]) if dt in m_ret.index and np.isfinite(m_ret.loc[dt]) else 0.0
            yr = float(y_ret.loc[dt]) if dt in y_ret.index and np.isfinite(y_ret.loc[dt]) else 0.0
            ratio = float(j.loc[dt, "m"] / j.loc[dt, "y"])
            phantom = abs(mr) > jump_thresh and abs(yr) < max(0.10, 0.25 * abs(mr))
            scale = abs(ratio - 1.0) > 0.40
            if phantom or scale:
                rows.append(
                    {
                        "symbol": etf,
                        "date": str(pd.Timestamp(dt).date()),
                        "close": float(j.loc[dt, "y"]),
                        "source": "yahoo_referee",
                        "note": (
                            f"phantom_r={mr:.3f}_vs_y={yr:.3f}"
                            if phantom
                            else f"scale_ratio={ratio:.3f}"
                        ),
                    }
                )
    return pd.DataFrame(rows)


def merge_patches(existing: Path, new: pd.DataFrame) -> pd.DataFrame:
    cols = ["symbol", "date", "close", "source", "note"]
    if existing.is_file():
        old = pd.read_csv(existing)
        for c in cols:
            if c not in old.columns:
                old[c] = "" if c != "close" else np.nan
        old = old[cols]
    else:
        old = pd.DataFrame(columns=cols)
    if new is None or new.empty:
        return old
    both = pd.concat([old, new[cols]], ignore_index=True)
    both["symbol"] = both["symbol"].map(_norm)
    both["date"] = pd.to_datetime(both["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    both["close"] = pd.to_numeric(both["close"], errors="coerce")
    both = both.dropna(subset=["symbol", "date", "close"])
    # Prefer newer rows (last wins) for same symbol/date.
    both = both.drop_duplicates(subset=["symbol", "date"], keep="last")
    return both.sort_values(["symbol", "date"]).reset_index(drop=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-date", default="2026-07-10")
    ap.add_argument("--held-from", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=REPO / "notebooks/output/production_actual_bt")
    ap.add_argument("--write-patches", type=Path, default=None)
    ap.add_argument("--min-jump", type=float, default=0.5)
    ap.add_argument(
        "--sim-panel",
        action="store_true",
        help="Audit post-split/patch sim panel vs Yahoo (catches SNXX/NEBX-class bugs).",
    )
    ap.add_argument(
        "--fix-panel-breaks",
        action="store_true",
        help="With --sim-panel: write full Yahoo patches for all panel suspects.",
    )
    ap.add_argument(
        "--underlying",
        action="store_true",
        help="Audit metrics underlying closes vs Yahoo (CVNA/IREN/ONDS-class).",
    )
    ap.add_argument(
        "--fix-und-breaks",
        action="store_true",
        help="With --underlying: write full Yahoo patches for suspect underlyings.",
    )
    args = ap.parse_args(argv)

    if args.held_from and args.held_from.is_file():
        held = pd.read_csv(args.held_from)
        col = "ETF" if "ETF" in held.columns else "ticker"
        tickers = held[col].astype(str).tolist()
    else:
        md = load_metrics_etf_closes(args.run_date)
        tickers = md["ticker"].unique().tolist()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.underlying:
        df = audit_underlyings(args.run_date)
        sus = df.loc[und_suspects_mask(df)].sort_values(
            ["phantom_days", "scale_bad_days", "n_big50"], ascending=False
        )
        df.to_csv(args.out_dir / "_price_integrity_underlyings_all.csv", index=False)
        sus.to_csv(args.out_dir / "_price_integrity_underlyings_suspects.csv", index=False)
        print(f"underlyings audited {len(df)}; suspects {len(sus)}")
        if not sus.empty:
            cols = [c for c in [
                "underlying", "src_etf", "phantom_days", "scale_bad_days", "n_big50",
                "worst_abs_ret", "worst_date", "scale_mode", "issue",
            ] if c in sus.columns]
            print(sus[cols].head(40).to_string(index=False))
        if args.fix_und_breaks and not sus.empty:
            patches = build_yahoo_full_patches_for_tickers(
                sus["underlying"].tolist(),
                note="underlying_scale_or_phantom",
            )
            dest = args.write_patches or (REPO / "data" / "price_patches.csv")
            merged = merge_patches(dest, patches)
            dest.parent.mkdir(parents=True, exist_ok=True)
            merged.to_csv(dest, index=False)
            print(f"wrote {len(merged)} patches ({len(sus)} underlyings) -> {dest}")
        return 0

    if args.sim_panel:
        df, _panel = audit_sim_panel(args.run_date, tickers)
        sus = df.loc[panel_suspects_mask(df)].sort_values(
            ["phantom_days", "scale_bad_days", "n_big50"], ascending=False
        )
        df.to_csv(args.out_dir / "_price_integrity_sim_panel_all.csv", index=False)
        sus.to_csv(args.out_dir / "_price_integrity_sim_panel_suspects.csv", index=False)
        print(f"sim-panel audited {len(df)}; suspects {len(sus)}")
        if not sus.empty:
            print(sus.head(40).to_string(index=False))
        if args.fix_panel_breaks and not sus.empty:
            patches = build_yahoo_full_patches_for_tickers(
                sus["etf"].tolist(),
                note="sim_panel_scale_or_phantom",
            )
            dest = args.write_patches or (REPO / "data" / "price_patches.csv")
            merged = merge_patches(dest, patches)
            dest.parent.mkdir(parents=True, exist_ok=True)
            merged.to_csv(dest, index=False)
            print(f"wrote {len(merged)} patches ({len(sus)} tickers) -> {dest}")
        return 0

    df = audit_universe(args.run_date, tickers)
    sus = df.loc[suspects_mask(df)].sort_values(
        ["phantom_days", "yahoo_gt15pct", "n_big50"], ascending=False
    )
    df.to_csv(args.out_dir / "_price_integrity_all_etfs.csv", index=False)
    sus.to_csv(args.out_dir / "_price_integrity_suspects.csv", index=False)
    print(f"audited {len(df)} tickers; suspects {len(sus)}; phantoms {(df['phantom_days']>0).sum()}")

    if args.write_patches is not None:
        patches = build_referee_patches(
            args.run_date,
            sus["etf"].tolist() if not sus.empty else tickers,
            jump_thresh=args.min_jump,
        )
        merged = merge_patches(args.write_patches, patches)
        args.write_patches.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(args.write_patches, index=False)
        print(f"wrote {len(merged)} patches -> {args.write_patches}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
