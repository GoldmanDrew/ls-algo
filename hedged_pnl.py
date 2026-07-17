#!/usr/bin/env python3
"""Hedged vs unhedged PnL lens (additional view on top of bucket accounting).

Definitions
-----------
* Hedged PnL   = Bucket 1 + Bucket 2 + the *matched* slice of each Bucket 4 pair.
* Unhedged PnL = Bucket 3 + Bucket 5 + the *unmatched* slice of Bucket 4.
* Invariant:  hedged + unhedged == sum of bucket PnL (any float residual is
  dumped into unhedged so the totals always tie).

Bucket 4 pairs are short inverse ETF + short underlying. The two legs cancel
"up to the hedge ratio": for each ETF leg the beta-equivalent underlying short
that actually sits against it (realized book exposures, not the plan) is

    matched_usd = min(und_alloc_usd, |beta| * etf_gross_usd)

where ``und_alloc_usd`` is the underlying-leg gross allocated pro-rata across
the ETFs sharing that underlying (same convention as
``scripts.bucket4_eod_pair_charts.load_pair_gross_and_realized_h``). The
hedged fraction of the ETF leg is ``matched / (|beta| * etf_gross)`` and of the
underlying leg ``matched / und_alloc`` (handles over-hedged pairs).

Because hedge ratios drift, YTD hedged/unhedged is accumulated **daily**:
each run's per-leg PnL delta (diff of consecutive ``pnl_by_symbol.csv``,
bucket_4 rows) is split with *that day's* fractions and appended to
``data/ledger/hedged_pnl_history.csv``. Fallback order for a leg with PnL but
no exposure today: carry-forward of the prior run's fraction -> plan hedge
ratio from ``b4_hedge_cadence/b4_ratchet_targets.csv`` -> unhedged.

Per-run artifacts written next to the accounting outputs:
* ``hedged_pnl_split.json``      headline numbers + reconciliation status
* ``hedged_pnl_b4_by_pair.csv``  per-leg audit (fractions, matched USD, split)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent
RUNS_ROOT = REPO / "data" / "runs"
LEDGER_DIR = REPO / "data" / "ledger"
HEDGED_PNL_HISTORY_CSV = LEDGER_DIR / "hedged_pnl_history.csv"

SPLIT_JSON_NAME = "hedged_pnl_split.json"
B4_PAIR_CSV_NAME = "hedged_pnl_b4_by_pair.csv"

BUCKET_KEYS: tuple[str, ...] = ("bucket_1", "bucket_2", "bucket_3", "bucket_4", "bucket_5")
HEDGED_WHOLE_BUCKETS: tuple[str, ...] = ("bucket_1", "bucket_2")
UNHEDGED_WHOLE_BUCKETS: tuple[str, ...] = ("bucket_3", "bucket_5")

LEDGER_COLS: tuple[str, ...] = (
    "date",
    "total_pnl",
    "hedged_pnl",
    "unhedged_pnl",
    "hedged_daily",
    "unhedged_daily",
    "b12_daily",
    "b3_daily",
    "b5_daily",
    "b4_hedged_daily",
    "b4_unhedged_daily",
    "recon_residual",
)

# Ignore exposure rows below this gross when deriving fractions (splitter dust).
_MIN_GROSS_USD = 1.0
_RECON_TOL = 0.01


def _norm(x: object) -> str:
    return str(x).strip().upper().replace(".", "-")


def _pair_key(etf: object, underlying: object) -> str:
    return f"{_norm(etf)}|{_norm(underlying)}"


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _acct_dir(run_date: str, runs_root: Path) -> Path:
    return runs_root / run_date / "accounting"


def prior_accounting_run_date(run_date: str, runs_root: Path = RUNS_ROOT) -> str | None:
    """Latest run folder before ``run_date`` that has accounting totals."""
    try:
        target = pd.to_datetime(run_date).normalize()
    except (ValueError, TypeError):
        return None
    best: pd.Timestamp | None = None
    best_name: str | None = None
    if not runs_root.is_dir():
        return None
    for child in runs_root.iterdir():
        if not child.is_dir():
            continue
        try:
            dt = pd.to_datetime(child.name).normalize()
        except (ValueError, TypeError):
            continue
        if dt >= target:
            continue
        if not (child / "accounting" / "totals.json").exists():
            continue
        if best is None or dt > best:
            best = dt
            best_name = child.name
    return best_name


def _bucket_pnl_from_totals(totals: dict) -> dict[str, float]:
    bp = totals.get("bucket_pnl") or {}
    return {b: float(bp.get(b, 0.0) or 0.0) for b in BUCKET_KEYS}


def _load_totals(run_date: str, runs_root: Path) -> dict:
    path = _acct_dir(run_date, runs_root) / "totals.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _load_b4_symbol_pnl(run_date: str, runs_root: Path) -> pd.DataFrame:
    """Bucket-4 rows of pnl_by_symbol.csv: (symbol, underlying) -> YTD total_pnl."""
    df = _read_csv(_acct_dir(run_date, runs_root) / "pnl_by_symbol.csv")
    if df.empty or not {"symbol", "bucket", "total_pnl"}.issubset(df.columns):
        return pd.DataFrame(columns=["symbol", "underlying", "total_pnl"])
    b4 = df[df["bucket"].astype(str).str.strip().eq("bucket_4")].copy()
    b4["symbol"] = b4["symbol"].map(_norm)
    if "underlying" in b4.columns:
        b4["underlying"] = b4["underlying"].fillna(b4["symbol"]).map(_norm)
    else:
        b4["underlying"] = b4["symbol"]
    b4["total_pnl"] = pd.to_numeric(b4["total_pnl"], errors="coerce").fillna(0.0)
    return (
        b4.groupby(["symbol", "underlying"], as_index=False)["total_pnl"].sum()
    )


def _load_delta_by_etf(run_date: str, runs_root: Path) -> dict[str, float]:
    """|beta| per inverse ETF from bucket4_pairs.csv (fallback pnl_bucket_4_by_pair.csv)."""
    out: dict[str, float] = {}
    for name in ("bucket4_pairs.csv", "pnl_bucket_4_by_pair.csv"):
        df = _read_csv(_acct_dir(run_date, runs_root) / name)
        if df.empty or not {"etf", "delta"}.issubset(df.columns):
            continue
        for _, r in df.iterrows():
            etf = _norm(r["etf"])
            delta = pd.to_numeric(r.get("delta"), errors="coerce")
            if etf not in out and pd.notna(delta) and abs(float(delta)) > 0:
                out[etf] = abs(float(delta))
    return out


def load_plan_hedge_ratios(run_date: str, runs_root: Path = RUNS_ROOT) -> dict[str, float]:
    """Plan hedge ratio h per ``ETF|UNDERLYING`` from b4_ratchet_targets.csv (may be empty)."""
    df = _read_csv(runs_root / run_date / "b4_hedge_cadence" / "b4_ratchet_targets.csv")
    if df.empty or not {"ETF", "Underlying", "hedge_ratio"}.issubset(df.columns):
        return {}
    out: dict[str, float] = {}
    for _, r in df.iterrows():
        h = pd.to_numeric(r.get("hedge_ratio"), errors="coerce")
        if pd.notna(h) and float(h) >= 0:
            out[_pair_key(r["ETF"], r["Underlying"])] = float(h)
    return out


def _load_prior_fractions(
    prev_run_date: str | None, runs_root: Path
) -> tuple[dict[str, float], dict[str, float]]:
    """Prior run's leg fractions for carry-forward: (etf ``ETF|UND`` -> f, underlying -> f)."""
    if not prev_run_date:
        return {}, {}
    df = _read_csv(_acct_dir(prev_run_date, runs_root) / B4_PAIR_CSV_NAME)
    if df.empty or not {"leg_type", "symbol", "underlying", "f_hedged"}.issubset(df.columns):
        return {}, {}
    etf_f: dict[str, float] = {}
    und_f: dict[str, float] = {}
    for _, r in df.iterrows():
        f = pd.to_numeric(r.get("f_hedged"), errors="coerce")
        if pd.isna(f):
            continue
        if str(r["leg_type"]).strip().lower() == "etf":
            etf_f[_pair_key(r["symbol"], r["underlying"])] = float(f)
        else:
            und_f[_norm(r["underlying"])] = float(f)
    return etf_f, und_f


def compute_b4_leg_fractions(run_date: str, runs_root: Path = RUNS_ROOT) -> pd.DataFrame:
    """Per-leg hedged fractions from today's realized book exposures.

    Returns one row per B4 leg present in ``net_exposure_bucket_4_detail.csv``:
    ``leg_type`` (etf|underlying), ``symbol``, ``underlying``, ``gross_usd``,
    ``matched_usd``, ``f_hedged`` (NaN when the leg has no usable exposure).
    """
    det = _read_csv(_acct_dir(run_date, runs_root) / "net_exposure_bucket_4_detail.csv")
    needed = {"underlying", "symbol", "leg_type", "gross_notional_usd"}
    if det.empty or not needed.issubset(det.columns):
        return pd.DataFrame(
            columns=["leg_type", "symbol", "underlying", "gross_usd", "matched_usd", "f_hedged"]
        )
    det = det.copy()
    det["underlying"] = det["underlying"].map(_norm)
    det["symbol"] = det["symbol"].map(_norm)
    det["gross_notional_usd"] = pd.to_numeric(det["gross_notional_usd"], errors="coerce").fillna(0.0)
    det["leg_type"] = det["leg_type"].astype(str).str.strip().str.lower()

    delta_by_etf = _load_delta_by_etf(run_date, runs_root)
    etf_rows = det[det["leg_type"].eq("etf") & det["gross_notional_usd"].gt(_MIN_GROSS_USD)]
    und_gross = (
        det[det["leg_type"].eq("underlying")]
        .groupby("underlying")["gross_notional_usd"]
        .sum()
    )

    rows: list[dict] = []
    matched_by_und: dict[str, float] = {}
    for und, grp in etf_rows.groupby("underlying"):
        tot_etf_gross = float(grp["gross_notional_usd"].sum())
        ug = float(und_gross.get(und, 0.0))
        for _, r in grp.iterrows():
            eg = float(r["gross_notional_usd"])
            share = eg / tot_etf_gross if tot_etf_gross > 0 else 0.0
            und_alloc = ug * share
            beta = float(delta_by_etf.get(r["symbol"], np.nan))
            if np.isfinite(beta) and beta > 0:
                beta_equiv = beta * eg
                matched = min(und_alloc, beta_equiv)
                f_etf = matched / beta_equiv if beta_equiv > 0 else np.nan
            else:
                matched = np.nan
                f_etf = np.nan
            if np.isfinite(matched):
                matched_by_und[und] = matched_by_und.get(und, 0.0) + matched
            rows.append(
                {
                    "leg_type": "etf",
                    "symbol": r["symbol"],
                    "underlying": und,
                    "gross_usd": eg,
                    "matched_usd": matched,
                    "f_hedged": f_etf,
                }
            )

    for und, ug in und_gross.items():
        if ug <= _MIN_GROSS_USD:
            continue
        matched = matched_by_und.get(und)
        # No ETF legs against this underlying today -> the short is unmatched.
        f_und = (min(matched, ug) / ug) if matched is not None else 0.0
        rows.append(
            {
                "leg_type": "underlying",
                "symbol": und,
                "underlying": und,
                "gross_usd": float(ug),
                "matched_usd": float(matched) if matched is not None else 0.0,
                "f_hedged": float(np.clip(f_und, 0.0, 1.0)),
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["f_hedged"] = pd.to_numeric(out["f_hedged"], errors="coerce").clip(0.0, 1.0)
    return out


def _resolve_leg_fraction(
    *,
    leg_type: str,
    symbol: str,
    underlying: str,
    book: pd.DataFrame,
    prior_etf_f: dict[str, float],
    prior_und_f: dict[str, float],
    plan_h: dict[str, float],
) -> tuple[float, str]:
    """Hedged fraction for one leg: book exposure -> carry-forward -> plan h -> unhedged."""
    if not book.empty:
        if leg_type == "etf":
            hit = book[(book["leg_type"].eq("etf")) & (book["symbol"].eq(symbol))]
        else:
            hit = book[(book["leg_type"].eq("underlying")) & (book["underlying"].eq(underlying))]
        if not hit.empty:
            f = hit.iloc[0]["f_hedged"]
            if pd.notna(f):
                return float(f), "book"
    if leg_type == "etf":
        carried = prior_etf_f.get(_pair_key(symbol, underlying))
    else:
        carried = prior_und_f.get(underlying)
    if carried is not None and np.isfinite(carried):
        return float(np.clip(carried, 0.0, 1.0)), "carry"
    if leg_type == "etf":
        h_vals = [plan_h[_pair_key(symbol, underlying)]] if _pair_key(symbol, underlying) in plan_h else []
    else:
        h_vals = [h for key, h in plan_h.items() if key.endswith(f"|{underlying}")]
    if h_vals:
        h = float(np.mean(h_vals))
        if h > 0:
            f = min(1.0, h) if leg_type == "etf" else min(1.0, 1.0 / h)
            return float(f), "plan"
    return 0.0, "unhedged_default"


def _split_b4_daily(
    run_date: str,
    prev_run_date: str | None,
    runs_root: Path,
) -> tuple[float, float, pd.DataFrame]:
    """Split today's B4 per-leg PnL deltas into hedged/unhedged dollars.

    Returns ``(b4_hedged_daily, b4_unhedged_daily, per_leg_audit_df)``. When
    ``prev_run_date`` is None (seed run) the YTD values are treated as the
    day's delta.
    """
    cur = _load_b4_symbol_pnl(run_date, runs_root)
    if prev_run_date:
        prev = _load_b4_symbol_pnl(prev_run_date, runs_root)
    else:
        prev = pd.DataFrame(columns=["symbol", "underlying", "total_pnl"])
    merged = cur.merge(
        prev, on=["symbol", "underlying"], how="outer", suffixes=("_cur", "_prev")
    ).fillna({"total_pnl_cur": 0.0, "total_pnl_prev": 0.0})
    merged["daily_pnl"] = merged["total_pnl_cur"] - merged["total_pnl_prev"]

    book = compute_b4_leg_fractions(run_date, runs_root)
    prior_etf_f, prior_und_f = _load_prior_fractions(prev_run_date, runs_root)
    plan_h = load_plan_hedge_ratios(run_date, runs_root)
    if not plan_h and prev_run_date:
        plan_h = load_plan_hedge_ratios(prev_run_date, runs_root)

    book_keys: set[tuple[str, str, str]] = set()
    rows: list[dict] = []
    hedged = 0.0
    unhedged = 0.0
    for _, r in merged.iterrows():
        symbol = str(r["symbol"])
        underlying = str(r["underlying"])
        leg_type = "underlying" if symbol == underlying else "etf"
        daily = float(r["daily_pnl"])
        f, source = _resolve_leg_fraction(
            leg_type=leg_type,
            symbol=symbol,
            underlying=underlying,
            book=book,
            prior_etf_f=prior_etf_f,
            prior_und_f=prior_und_f,
            plan_h=plan_h,
        )
        leg_hedged = f * daily
        leg_unhedged = daily - leg_hedged
        hedged += leg_hedged
        unhedged += leg_unhedged
        book_keys.add((leg_type, symbol, underlying))
        gross = matched = np.nan
        if not book.empty:
            hit = book[
                book["leg_type"].eq(leg_type)
                & book["symbol"].eq(symbol)
                & book["underlying"].eq(underlying)
            ]
            if not hit.empty:
                gross = float(hit.iloc[0]["gross_usd"])
                matched = float(hit.iloc[0]["matched_usd"]) if pd.notna(hit.iloc[0]["matched_usd"]) else np.nan
        rows.append(
            {
                "leg_type": leg_type,
                "symbol": symbol,
                "underlying": underlying,
                "pair": _pair_key(symbol, underlying) if leg_type == "etf" else f"{underlying} (spot)",
                "gross_usd": gross,
                "matched_usd": matched,
                "f_hedged": f,
                "f_source": source,
                "daily_pnl": daily,
                "hedged_daily": leg_hedged,
                "unhedged_daily": leg_unhedged,
                "ytd_pnl": float(r["total_pnl_cur"]),
            }
        )

    # Legs held today with zero PnL delta still land in the audit CSV so the
    # next run can carry their fractions forward.
    if not book.empty:
        for _, b in book.iterrows():
            key = (str(b["leg_type"]), str(b["symbol"]), str(b["underlying"]))
            if key in book_keys or pd.isna(b["f_hedged"]):
                continue
            rows.append(
                {
                    "leg_type": key[0],
                    "symbol": key[1],
                    "underlying": key[2],
                    "pair": _pair_key(key[1], key[2]) if key[0] == "etf" else f"{key[2]} (spot)",
                    "gross_usd": float(b["gross_usd"]),
                    "matched_usd": float(b["matched_usd"]) if pd.notna(b["matched_usd"]) else np.nan,
                    "f_hedged": float(b["f_hedged"]),
                    "f_source": "book",
                    "daily_pnl": 0.0,
                    "hedged_daily": 0.0,
                    "unhedged_daily": 0.0,
                    "ytd_pnl": 0.0,
                }
            )

    audit = pd.DataFrame(rows)
    if not audit.empty:
        audit = audit.sort_values(
            ["underlying", "leg_type", "symbol"], kind="stable"
        ).reset_index(drop=True)
    return hedged, unhedged, audit


def _read_ledger(ledger_path: Path) -> pd.DataFrame:
    df = _read_csv(ledger_path)
    if df.empty:
        return pd.DataFrame(columns=list(LEDGER_COLS))
    df["date"] = df["date"].astype(str)
    for c in LEDGER_COLS[1:]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def compute_hedged_split(
    run_date: str,
    *,
    runs_root: Path = RUNS_ROOT,
    ledger_path: Path = HEDGED_PNL_HISTORY_CSV,
    prev_run_date: str | None | object = "auto",
    write: bool = True,
) -> dict:
    """Compute the hedged/unhedged PnL split for ``run_date`` and update the ledger.

    Returns a dict with YTD + daily hedged/unhedged numbers, per-component
    daily contributions and reconciliation status. When ``write`` is True the
    per-run artifacts and the ledger row for ``run_date`` are persisted
    (idempotent per date).
    """
    totals = _load_totals(run_date, runs_root)
    cur_buckets = _bucket_pnl_from_totals(totals)
    if prev_run_date == "auto":
        prev_run_date = prior_accounting_run_date(run_date, runs_root)
    if prev_run_date:
        prev_buckets = _bucket_pnl_from_totals(_load_totals(str(prev_run_date), runs_root))
    else:
        prev_buckets = {b: 0.0 for b in BUCKET_KEYS}
    daily_buckets = {b: cur_buckets[b] - prev_buckets[b] for b in BUCKET_KEYS}
    total_daily = sum(daily_buckets.values())

    b12_daily = sum(daily_buckets[b] for b in HEDGED_WHOLE_BUCKETS)
    b3_daily = daily_buckets["bucket_3"]
    b5_daily = daily_buckets["bucket_5"]
    b4_hedged_daily, b4_unhedged_daily, audit = _split_b4_daily(
        run_date, str(prev_run_date) if prev_run_date else None, runs_root
    )
    # Any gap between the per-symbol B4 diff and the bucket-level diff (e.g.
    # restated symbol files) goes to unhedged so the lens always ties out.
    b4_gap = daily_buckets["bucket_4"] - (b4_hedged_daily + b4_unhedged_daily)
    b4_unhedged_daily += b4_gap

    hedged_daily = b12_daily + b4_hedged_daily
    unhedged_daily = b3_daily + b5_daily + b4_unhedged_daily
    residual = total_daily - (hedged_daily + unhedged_daily)
    if abs(residual) > _RECON_TOL:
        print(
            f"[hedged-pnl] WARN daily reconciliation residual {residual:,.2f} "
            f"on {run_date}; assigning to unhedged"
        )
    unhedged_daily += residual

    ledger = _read_ledger(ledger_path)
    prior_rows = ledger[ledger["date"] < run_date]
    stale = ledger[ledger["date"] > run_date]
    if not stale.empty and write:
        print(
            f"[hedged-pnl] WARN ledger has {len(stale)} row(s) after {run_date}; "
            "their YTD values are now stale (rerun scripts/backfill_hedged_pnl.py)"
        )
    if not prior_rows.empty:
        base = prior_rows.iloc[-1]
        hedged_ytd = float(base["hedged_pnl"]) + hedged_daily
        unhedged_ytd = float(base["unhedged_pnl"]) + unhedged_daily
        seeded = False
    else:
        hedged_ytd = hedged_daily
        unhedged_ytd = unhedged_daily
        seeded = True

    row = {
        "date": run_date,
        "total_pnl": sum(cur_buckets.values()),
        "hedged_pnl": hedged_ytd,
        "unhedged_pnl": unhedged_ytd,
        "hedged_daily": hedged_daily,
        "unhedged_daily": unhedged_daily,
        "b12_daily": b12_daily,
        "b3_daily": b3_daily,
        "b5_daily": b5_daily,
        "b4_hedged_daily": b4_hedged_daily,
        "b4_unhedged_daily": b4_unhedged_daily,
        "recon_residual": residual,
    }

    result = {
        "run_date": run_date,
        "prev_run_date": str(prev_run_date) if prev_run_date else None,
        "seeded": seeded,
        "hedged_pnl_ytd": hedged_ytd,
        "unhedged_pnl_ytd": unhedged_ytd,
        "hedged_daily": hedged_daily,
        "unhedged_daily": unhedged_daily,
        "total_pnl_ytd": float(row["total_pnl"]),
        "total_daily": total_daily,
        "components": {
            "b12_daily": b12_daily,
            "b3_daily": b3_daily,
            "b5_daily": b5_daily,
            "b4_hedged_daily": b4_hedged_daily,
            "b4_unhedged_daily": b4_unhedged_daily,
        },
        "reconciliation": {
            "residual_daily": residual,
            "ytd_gap_vs_total": float(row["total_pnl"]) - (hedged_ytd + unhedged_ytd),
            "ok": abs(residual) <= _RECON_TOL,
        },
    }

    if write:
        ledger = pd.concat(
            [ledger[ledger["date"] != run_date], pd.DataFrame([row])],
            ignore_index=True,
        ).sort_values("date").reset_index(drop=True)
        ledger_path.parent.mkdir(parents=True, exist_ok=True)
        ledger[list(LEDGER_COLS)].to_csv(ledger_path, index=False)

        acct = _acct_dir(run_date, runs_root)
        acct.mkdir(parents=True, exist_ok=True)
        (acct / SPLIT_JSON_NAME).write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
        audit.to_csv(acct / B4_PAIR_CSV_NAME, index=False)

    return result


def load_hedged_split(run_date: str, runs_root: Path = RUNS_ROOT) -> dict | None:
    """Read a previously computed per-run split artifact (None when absent)."""
    path = _acct_dir(run_date, runs_root) / SPLIT_JSON_NAME
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Compute hedged vs unhedged PnL split for a run date")
    ap.add_argument("run_date", help="YYYY-MM-DD run date (accounting outputs must exist)")
    ap.add_argument("--no-write", action="store_true", help="Compute only; do not persist")
    args = ap.parse_args()
    res = compute_hedged_split(args.run_date, write=not args.no_write)
    print(json.dumps(res, indent=2))
