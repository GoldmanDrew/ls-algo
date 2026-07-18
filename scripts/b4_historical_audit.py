"""Historical B4 audit series for the production-actual notebook.

Derives time series from cached ``plans/*.csv`` plus any archived
``b4_crash_budget.csv`` / ``b4_sizing_waterfall.csv`` under ``data/runs``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
B4_SLEEVE = "inverse_decay_bucket4"


def _norm(x: object) -> str:
    return str(x).strip().upper().replace(".", "-")


def _is_b4(df: pd.DataFrame) -> pd.Series:
    if "sleeve" in df.columns:
        s = df["sleeve"].astype(str).str.strip().str.lower()
        m = s.eq(B4_SLEEVE) | s.str.contains("inverse_decay", na=False)
        if bool(m.any()):
            return m
    if "Delta" in df.columns:
        return pd.to_numeric(df["Delta"], errors="coerce") < 0
    return pd.Series(False, index=df.index)


def load_b4_plan_history(plans_dir: Path | str) -> pd.DataFrame:
    """One row per (date, ETF) from cached prod plans."""
    root = Path(plans_dir)
    rows: list[dict[str, Any]] = []
    if not root.is_dir():
        return pd.DataFrame()
    for path in sorted(root.glob("*.csv")):
        try:
            d = pd.Timestamp(path.stem)
        except Exception:
            continue
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if df.empty:
            continue
        m = _is_b4(df)
        sub = df.loc[m].copy()
        if sub.empty:
            continue
        sub["ETF"] = sub["ETF"].map(_norm) if "ETF" in sub.columns else ""
        sub["Underlying"] = sub["Underlying"].map(_norm) if "Underlying" in sub.columns else ""
        long_u = pd.to_numeric(sub.get("long_usd"), errors="coerce").fillna(0.0).abs()
        short_u = pd.to_numeric(sub.get("short_usd"), errors="coerce").fillna(0.0).abs()
        if "gross_target_usd" in sub.columns:
            gross = pd.to_numeric(sub["gross_target_usd"], errors="coerce")
        else:
            gross = long_u + short_u
        if "hedge_ratio" in sub.columns:
            hedge = pd.to_numeric(sub["hedge_ratio"], errors="coerce")
        else:
            hedge = long_u / short_u.replace(0.0, np.nan)
        if not isinstance(gross, pd.Series):
            gross = pd.Series(gross, index=sub.index)
        if not isinstance(hedge, pd.Series):
            hedge = pd.Series(hedge, index=sub.index)
        if hedge.isna().all():
            hedge = long_u / short_u.replace(0.0, np.nan)
        if gross.isna().all():
            gross = long_u + short_u
        for i, r in sub.iterrows():
            g_i = float(gross.loc[i]) if pd.notna(gross.loc[i]) else 0.0
            h_i = float(hedge.loc[i]) if pd.notna(hedge.loc[i]) else float("nan")
            rows.append(
                {
                    "date": d.normalize(),
                    "ETF": _norm(r.get("ETF")),
                    "Underlying": _norm(r.get("Underlying")),
                    "gross_target_usd": g_i,
                    "hedge_ratio": h_i,
                    "Delta": float(pd.to_numeric(r.get("Delta"), errors="coerce") or np.nan),
                }
            )
    return pd.DataFrame(rows)


def summarize_b4_plan_history(hist: pd.DataFrame) -> pd.DataFrame:
    """Daily sleeve aggregates from plan history."""
    if hist is None or hist.empty:
        return pd.DataFrame()
    g = (
        hist.groupby("date", as_index=False)
        .agg(
            n_pairs=("ETF", "nunique"),
            gross_b4=("gross_target_usd", "sum"),
            median_hedge=("hedge_ratio", "median"),
            mean_abs_delta=("Delta", lambda s: float(np.nanmean(np.abs(pd.to_numeric(s, errors="coerce"))))),
        )
        .sort_values("date")
    )
    return g


def load_archived_crash_waterfall(runs_dir: Path | str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stack any ``b4_crash_budget.csv`` / ``b4_sizing_waterfall.csv`` under runs."""
    root = Path(runs_dir) if runs_dir is not None else REPO / "data" / "runs"
    crash_rows: list[pd.DataFrame] = []
    wf_rows: list[pd.DataFrame] = []
    if not root.is_dir():
        return pd.DataFrame(), pd.DataFrame()
    for run in sorted(root.iterdir()):
        if not run.is_dir():
            continue
        try:
            d = pd.Timestamp(run.name).normalize()
        except Exception:
            continue
        for rel in ("b4_crash_budget.csv", "b4_hedge_cadence/b4_crash_budget.csv"):
            p = run / rel
            if p.is_file():
                try:
                    df = pd.read_csv(p)
                    df["run_date"] = d
                    crash_rows.append(df)
                except Exception:
                    pass
                break
        for rel in ("b4_sizing_waterfall.csv", "b4_hedge_cadence/b4_sizing_waterfall.csv"):
            p = run / rel
            if p.is_file():
                try:
                    df = pd.read_csv(p)
                    df["run_date"] = d
                    wf_rows.append(df)
                except Exception:
                    pass
                break
    crash = pd.concat(crash_rows, ignore_index=True) if crash_rows else pd.DataFrame()
    wf = pd.concat(wf_rows, ignore_index=True) if wf_rows else pd.DataFrame()
    return crash, wf


def plot_b4_plan_history(summary: pd.DataFrame) -> plt.Figure | None:
    if summary is None or summary.empty:
        return None
    fig, axes = plt.subplots(3, 1, figsize=(11.5, 7.5), sharex=True)
    ax = axes[0]
    ax.plot(summary["date"], summary["gross_b4"], color="#c44e52", lw=1.5)
    ax.set_ylabel("B4 plan gross ($)")
    ax.set_title("Historical B4 from cached plans (gross / n pairs / median h)")
    ax.grid(True, alpha=0.3)
    ax = axes[1]
    ax.bar(summary["date"], summary["n_pairs"], width=1.6, color="#4c72b0", alpha=0.8)
    ax.set_ylabel("n B4 pairs")
    ax.grid(True, axis="y", alpha=0.3)
    ax = axes[2]
    ax.plot(summary["date"], summary["median_hedge"], color="#8172b2", lw=1.4, label="median plan h")
    ax.set_ylabel("hedge h")
    ax.set_xlabel("plan date")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_crash_waterfall_snapshots(crash: pd.DataFrame, wf: pd.DataFrame) -> list[plt.Figure]:
    """Plot per-run_date crash mult + waterfall stages when archives exist."""
    figs: list[plt.Figure] = []
    if crash is not None and not crash.empty and "crash_budget_mult" in crash.columns:
        for d, g in crash.groupby("run_date"):
            top = g.sort_values("crash_budget_mult").head(12)
            fig, ax = plt.subplots(figsize=(10, 4.2))
            labels = top["ETF"].astype(str) + "/" + top["Underlying"].astype(str)
            ax.barh(labels, top["crash_budget_mult"], color="#c44e52")
            ax.set_xlabel("crash_budget_mult")
            ax.set_title(f"B4 crash-budget snapshot — {pd.Timestamp(d).date()}")
            ax.grid(True, axis="x", alpha=0.3)
            fig.tight_layout()
            figs.append(fig)
    if wf is not None and not wf.empty:
        cols = [c for c in ("weight_solved", "weight_capped", "weight_smoothed") if c in wf.columns]
        if cols:
            for d, g in wf.groupby("run_date"):
                top = g.nlargest(10, cols[-1])
                fig, ax = plt.subplots(figsize=(10, 4.5))
                y = np.arange(len(top))
                width = 0.25
                for i, col in enumerate(cols):
                    vals = pd.to_numeric(top[col], errors="coerce").fillna(0.0).to_numpy()
                    ax.barh(y + (i - 1) * width, vals, height=width, label=col)
                ax.set_yticks(y)
                ax.set_yticklabels(top["ETF"].astype(str) + "/" + top["Underlying"].astype(str))
                ax.set_xlabel("pair weight")
                ax.set_title(f"B4 sizing waterfall — {pd.Timestamp(d).date()}")
                ax.legend(loc="best", fontsize=8)
                ax.grid(True, axis="x", alpha=0.3)
                fig.tight_layout()
                figs.append(fig)
    return figs
