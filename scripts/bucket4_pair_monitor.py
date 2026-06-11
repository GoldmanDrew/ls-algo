"""Bucket 4 pair monitor: realized edge vs expectation, borrow-vs-decay (REPORT-ONLY).

Phase 1 of the B4 optimization rollout. Reads the accounting history that the
daily pipeline already produces and answers, per pair:

  * trailing 20/60bd realized pair PnL (from cumulative ``pnl_bucket_4_by_pair.csv``)
  * annualized realized return on pair gross (from ``net_exposure_bucket_4_detail.csv``)
  * borrow paid over the window vs gross decay captured (borrow kill-switch signal)
  * edge capture ratio = realized annualized / screener expected ``bucket4_net_edge_annual``
  * demotion-ladder flags (half / freeze / exit / vol-floor) -- REPORT ONLY here;
    Phase 2 (``bucket4_pair_lifecycle.py``) turns these into sizing actions.

Outputs:
  data/runs/<latest>/b4_monitor/b4_pair_monitor.csv
  data/b4_observations.jsonl   (one summary line per run date, for the tuning loop)

Usage:
  python -m scripts.bucket4_pair_monitor [--runs-root data/runs] [--windows 20 60]
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

TRADING_DAYS = 252


def norm_sym(x: str) -> str:
    return str(x).strip().upper().replace(".", "-")


def _pair_key(etf: str, und: str) -> str:
    return f"{norm_sym(etf)}|{norm_sym(und)}"


# ---------------------------------------------------------------------------
# History loading
# ---------------------------------------------------------------------------
def list_run_dates(runs_root: Path) -> list[str]:
    """Run-date dirs (ascending) that contain per-pair B4 PnL."""
    out = []
    for d in sorted(runs_root.iterdir()):
        if d.is_dir() and (d / "accounting" / "pnl_bucket_4_by_pair.csv").is_file():
            out.append(d.name)
    return out


def load_pair_pnl_history(runs_root: Path, dates: list[str]) -> pd.DataFrame:
    """Long frame: one row per (date, pair) with cumulative pnl/borrow columns."""
    rows = []
    for ds in dates:
        p = runs_root / ds / "accounting" / "pnl_bucket_4_by_pair.csv"
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        if df.empty or "etf" not in df.columns:
            continue
        df["etf"] = df["etf"].map(norm_sym)
        df["underlying"] = df["underlying"].map(norm_sym)
        df["pair"] = df.apply(lambda r: _pair_key(r["etf"], r["underlying"]), axis=1)
        df["run_date"] = ds
        keep = ["run_date", "pair", "etf", "underlying", "total_pnl", "borrow_fees",
                "realized_pnl", "unrealized_pnl"]
        rows.append(df[[c for c in keep if c in df.columns]])
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    for c in ("total_pnl", "borrow_fees", "realized_pnl", "unrealized_pnl"):
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def load_pair_gross(runs_root: Path, run_date: str) -> dict[str, float]:
    """Current gross notional per pair from net_exposure_bucket_4_detail.csv.

    The underlying leg can be shared by multiple ETFs on the same underlying;
    allocate it pro-rata to each ETF leg's gross.
    """
    p = runs_root / run_date / "accounting" / "net_exposure_bucket_4_detail.csv"
    if not p.is_file():
        return {}
    df = pd.read_csv(p)
    if df.empty:
        return {}
    df["underlying"] = df["underlying"].map(norm_sym)
    df["symbol"] = df["symbol"].map(norm_sym)
    df["gross_notional_usd"] = pd.to_numeric(df["gross_notional_usd"], errors="coerce").fillna(0.0)

    etf_rows = df[df["leg_type"].astype(str).str.lower() == "etf"]
    und_rows = df[df["leg_type"].astype(str).str.lower() == "underlying"]
    und_gross = und_rows.groupby("underlying")["gross_notional_usd"].sum().to_dict()

    out: dict[str, float] = {}
    for und, grp in etf_rows.groupby("underlying"):
        etf_total = float(grp["gross_notional_usd"].sum())
        u_gross = float(und_gross.get(und, 0.0))
        for _, r in grp.iterrows():
            etf_g = float(r["gross_notional_usd"])
            share = etf_g / etf_total if etf_total > 1e-9 else 1.0 / max(len(grp), 1)
            out[_pair_key(r["symbol"], und)] = etf_g + share * u_gross
    return out


def load_expected_edges(screened_csv: Path) -> pd.DataFrame:
    cols = ["ETF", "Underlying", "bucket4_net_edge_annual", "net_edge_p50_annual",
            "vol_underlying_annual", "borrow_current", "gross_decay_annual"]
    df = pd.read_csv(screened_csv, usecols=lambda c: c in cols)
    df["ETF"] = df["ETF"].map(norm_sym)
    df["Underlying"] = df["Underlying"].map(norm_sym)
    df["pair"] = df.apply(lambda r: _pair_key(r["ETF"], r["Underlying"]), axis=1)
    return df.set_index("pair")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def trailing_delta(series: pd.Series, n: int) -> tuple[float, int]:
    """cum(t) - cum(t-n) plus actual observation span used (handles young pairs)."""
    s = series.dropna()
    if len(s) < 2:
        return np.nan, 0
    last = float(s.iloc[-1])
    if len(s) > n:
        base = float(s.iloc[-(n + 1)])
        span = n
    else:
        base = float(s.iloc[0])
        span = len(s) - 1
    return last - base, span


def build_monitor(
    hist: pd.DataFrame,
    gross_by_pair: dict[str, float],
    expected: pd.DataFrame,
    *,
    windows: list[int],
    thresholds: dict,
) -> pd.DataFrame:
    rows = []
    for pair, grp in hist.groupby("pair"):
        grp = grp.sort_values("run_date")
        cum_pnl = grp.set_index("run_date")["total_pnl"]
        cum_borrow = grp.set_index("run_date")["borrow_fees"]
        etf = grp["etf"].iloc[-1]
        und = grp["underlying"].iloc[-1]
        gross = float(gross_by_pair.get(pair, np.nan))

        row: dict = {
            "pair": pair, "etf": etf, "underlying": und,
            "gross_usd": gross, "n_obs_days": len(grp),
        }
        for n in windows:
            pnl, span = trailing_delta(cum_pnl, n)
            borrow, _ = trailing_delta(cum_borrow, n)  # borrow_fees are negative (cost)
            row[f"pnl_{n}d"] = pnl
            row[f"borrow_{n}d"] = borrow
            row[f"span_{n}d"] = span
            if np.isfinite(gross) and gross > 500 and span >= 5 and np.isfinite(pnl):
                row[f"ret_{n}d_annual"] = (pnl / gross) * (TRADING_DAYS / span)
            else:
                row[f"ret_{n}d_annual"] = np.nan

        exp = expected.loc[pair] if pair in expected.index else None
        exp_edge = float(exp.get("bucket4_net_edge_annual", np.nan)) if exp is not None else np.nan
        if not np.isfinite(exp_edge) and exp is not None:
            exp_edge = float(exp.get("net_edge_p50_annual", np.nan))
        und_vol = float(exp.get("vol_underlying_annual", np.nan)) if exp is not None else np.nan
        borrow_now = float(exp.get("borrow_current", np.nan)) if exp is not None else np.nan
        row["expected_edge_annual"] = exp_edge
        row["vol_underlying_annual"] = und_vol
        row["borrow_current"] = borrow_now

        n_main = max(windows)
        ret_main = row.get(f"ret_{n_main}d_annual", np.nan)
        row["edge_capture_ratio"] = (
            ret_main / exp_edge if np.isfinite(ret_main) and np.isfinite(exp_edge) and exp_edge > 1e-6
            else np.nan
        )

        # Borrow-vs-decay kill switch: borrow paid exceeded the gross decay captured
        # AND the borrow drag is material (>= exit_borrow_floor_annual on pair gross).
        # The materiality floor keeps a $10 borrow line from exiting a pair whose
        # losses are really a hedge/decay story (that is the ret thresholds' job).
        pnl_main = row.get(f"pnl_{n_main}d", np.nan)
        borrow_main = row.get(f"borrow_{n_main}d", np.nan)  # negative
        span_main = row.get(f"span_{n_main}d", 0)
        gross_capture = pnl_main - borrow_main if np.isfinite(pnl_main) and np.isfinite(borrow_main) else np.nan
        row["gross_capture_main"] = gross_capture
        borrow_ann_on_gross = (
            abs(borrow_main) / gross * (TRADING_DAYS / span_main)
            if np.isfinite(borrow_main) and np.isfinite(gross) and gross > 500 and span_main >= 5
            else np.nan
        )
        row["borrow_annual_on_gross"] = borrow_ann_on_gross
        row["borrow_exceeds_decay"] = bool(
            np.isfinite(gross_capture) and np.isfinite(borrow_main)
            and abs(borrow_main) > max(gross_capture, 0.0)
            and np.isfinite(borrow_ann_on_gross)
            and borrow_ann_on_gross >= thresholds["exit_borrow_floor_annual"]
            and span_main >= thresholds["min_obs_days"]
        )

        # ---- demotion ladder flags (report-only in Phase 1) ----
        min_obs = thresholds["min_obs_days"]
        enough = row.get(f"span_{n_main}d", 0) >= min_obs
        # Half-weight: under-capturing expectation AND not strongly positive outright
        # (a pair printing > half_ret_ok annualized is never demoted on capture alone).
        row["flag_half"] = bool(
            enough and np.isfinite(row["edge_capture_ratio"])
            and row["edge_capture_ratio"] < thresholds["half_capture_lt"]
            and np.isfinite(ret_main) and ret_main < thresholds["half_ret_ok"]
        )
        row["flag_freeze"] = bool(
            enough and np.isfinite(ret_main) and ret_main < thresholds["freeze_ret_lt"]
        )
        row["flag_exit"] = bool(
            (enough and np.isfinite(ret_main) and ret_main < thresholds["exit_ret_lt"])
            or row["borrow_exceeds_decay"]
        )
        row["flag_vol_floor"] = bool(
            np.isfinite(und_vol) and und_vol < thresholds["min_underlying_vol_keep"]
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values("pnl_%dd" % max(windows), na_position="last")


def default_thresholds(cfg: dict | None) -> dict:
    rules = (
        (cfg or {}).get("portfolio", {}).get("sleeves", {})
        .get("inverse_decay_bucket4", {}).get("rules", {})
    )
    lc = rules.get("pair_lifecycle") or {}
    return {
        "half_capture_lt": float(lc.get("half_capture_lt", 0.25)),
        "half_ret_ok": float(lc.get("half_ret_ok", 0.10)),
        "freeze_ret_lt": float(lc.get("freeze_ret_lt", -0.15)),
        "exit_ret_lt": float(lc.get("exit_ret_lt", -0.30)),
        "exit_borrow_floor_annual": float(lc.get("exit_borrow_floor_annual", 0.05)),
        "min_obs_days": int(lc.get("min_obs_days", 20)),
        "min_underlying_vol_keep": float(rules.get("min_underlying_vol_keep",
                                                   rules.get("min_underlying_vol", 0.4))),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Bucket 4 pair monitor (report-only).")
    ap.add_argument("--runs-root", type=Path, default=REPO / "data/runs")
    ap.add_argument("--screened", type=Path, default=REPO / "data/etf_screened_today.csv")
    ap.add_argument("--windows", type=int, nargs="*", default=[20, 60])
    ap.add_argument("--obs-jsonl", type=Path, default=REPO / "data/b4_observations.jsonl")
    ap.add_argument("--no-jsonl", action="store_true")
    ap.add_argument("--include-inactive", action="store_true",
                    help="keep registry pairs with no exposure and no PnL history")
    args = ap.parse_args(argv)

    dates = list_run_dates(args.runs_root)
    if not dates:
        print("[b4-monitor] no run dates with pnl_bucket_4_by_pair.csv found")
        return 1
    latest = dates[-1]
    hist = load_pair_pnl_history(args.runs_root, dates)
    if hist.empty:
        print("[b4-monitor] empty pair PnL history")
        return 1
    gross = load_pair_gross(args.runs_root, latest)
    expected = load_expected_edges(args.screened)

    try:
        from strategy_config import load_config
        cfg = load_config()
    except Exception:
        cfg = {}
    thresholds = default_thresholds(cfg)

    mon = build_monitor(hist, gross, expected, windows=args.windows, thresholds=thresholds)
    if not args.include_inactive:
        n_main_w = max(args.windows)
        active = (
            mon["gross_usd"].fillna(0.0).abs().gt(500.0)
            | mon[f"pnl_{n_main_w}d"].fillna(0.0).abs().gt(1.0)
        )
        n_drop = int((~active).sum())
        mon = mon[active].reset_index(drop=True)
        if n_drop:
            print(f"[b4-monitor] dropped {n_drop} inactive registry pairs (use --include-inactive to keep)")

    out_dir = args.runs_root / latest / "b4_monitor"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "b4_pair_monitor.csv"
    mon.to_csv(out_csv, index=False)

    n_main = max(args.windows)
    print(f"[b4-monitor] run_date={latest} pairs={len(mon)} history_days={len(dates)}")
    print(f"[b4-monitor] thresholds: {thresholds}")
    show = mon[[
        "pair", "gross_usd", f"pnl_{n_main}d", f"ret_{n_main}d_annual",
        f"borrow_{n_main}d", "expected_edge_annual", "edge_capture_ratio",
        "flag_half", "flag_freeze", "flag_exit", "flag_vol_floor",
    ]].copy()
    with pd.option_context("display.width", 200, "display.max_rows", 100):
        print(show.to_string(index=False, float_format=lambda v: f"{v:,.2f}"))
    print(f"\n[b4-monitor] wrote {out_csv}")
    n_flags = int(mon[["flag_half", "flag_freeze", "flag_exit", "flag_vol_floor"]].any(axis=1).sum())
    print(f"[b4-monitor] flags: half={int(mon['flag_half'].sum())} freeze={int(mon['flag_freeze'].sum())} "
          f"exit={int(mon['flag_exit'].sum())} vol_floor={int(mon['flag_vol_floor'].sum())} "
          f"(any={n_flags}/{len(mon)})")

    if not args.no_jsonl:
        hcp = (
            (cfg or {}).get("portfolio", {}).get("sleeves", {})
            .get("inverse_decay_bucket4", {}).get("rules", {})
            .get("bucket4_weekly_opt2", {}).get("hedge_cadence_policy", {})
        )
        obs = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "run_date": latest,
            "knobs": {
                k: hcp.get(k)
                for k in ("base_days", "k_tr", "m_vcr", "max_interval", "opt2_k", "source", "hedge_ratio_model")
            },
            "n_pairs": int(len(mon)),
            "ew_ret_main_annual": float(mon[f"ret_{n_main}d_annual"].mean(skipna=True)),
            "median_ret_main_annual": float(mon[f"ret_{n_main}d_annual"].median(skipna=True)),
            "total_pnl_main": float(mon[f"pnl_{n_main}d"].sum(skipna=True)),
            "total_borrow_main": float(mon[f"borrow_{n_main}d"].sum(skipna=True)),
            "n_flag_half": int(mon["flag_half"].sum()),
            "n_flag_freeze": int(mon["flag_freeze"].sum()),
            "n_flag_exit": int(mon["flag_exit"].sum()),
            "n_flag_vol_floor": int(mon["flag_vol_floor"].sum()),
        }
        args.obs_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with open(args.obs_jsonl, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(obs) + "\n")
        print(f"[b4-monitor] appended observation -> {args.obs_jsonl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
