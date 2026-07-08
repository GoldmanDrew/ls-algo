"""Reconstruct Bucket-4 held leg notionals from committed accounting snapshots.

Morning ``generate_trade_plan`` uses **yesterday's** (or the latest available)
``net_exposure_bucket_4_detail.csv`` so the grow-only ratchet floors to
machine-independent held inventory rather than a local JSON file alone.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "data" / "runs"


def _norm_sym(x: str) -> str:
    return str(x).strip().upper().replace(".", "-")


def _detail_path(run_date: str, *, runs_root: Path | None = None) -> Path:
    root = runs_root or RUNS
    return root / str(run_date) / "accounting" / "net_exposure_bucket_4_detail.csv"


def _available_detail_dates(*, runs_root: Path | None = None, before: str | None = None) -> list[str]:
    root = runs_root or RUNS
    if not root.is_dir():
        return []
    out: list[str] = []
    for child in root.iterdir():
        if not child.is_dir():
            continue
        if before is not None and child.name > str(before):
            continue
        if _detail_path(child.name, runs_root=root).is_file():
            out.append(child.name)
    return sorted(out)


def resolve_held_detail_run_date(
    run_date: str,
    *,
    runs_root: Path | None = None,
    max_lookback: int = 10,
) -> tuple[str | None, Path | None]:
    """Pick the newest committed detail file on or before *run_date*."""
    candidates = [d for d in _available_detail_dates(runs_root=runs_root, before=run_date) if d <= str(run_date)]
    if not candidates:
        return None, None
    d = candidates[-1]
    p = _detail_path(d, runs_root=runs_root)
    return (d, p) if p.is_file() else (None, None)


def held_inverse_short_by_pair(
    run_date: str,
    *,
    runs_root: Path | None = None,
) -> dict[tuple[str, str], dict[str, float]]:
    """Positive USD gross shorts per (ETF, underlying) pair from accounting detail."""
    _, path = resolve_held_detail_run_date(run_date, runs_root=runs_root)
    if path is None:
        return {}
    df = pd.read_csv(path)
    if df.empty:
        return {}
    need = {"underlying", "symbol", "leg_type", "gross_notional_usd"}
    if not need.issubset(df.columns):
        return {}

    out: dict[tuple[str, str], dict[str, float]] = {}
    for und, grp in df.groupby("underlying"):
        und_n = _norm_sym(str(und))
        und_rows = grp[grp["leg_type"].astype(str).str.lower() == "underlying"]
        und_gross = float(pd.to_numeric(und_rows["gross_notional_usd"], errors="coerce").fillna(0.0).sum())
        etf_rows = grp[grp["leg_type"].astype(str).str.lower() == "etf"]
        for _, er in etf_rows.iterrows():
            etf_n = _norm_sym(str(er["symbol"]))
            inv_gross = abs(float(pd.to_numeric(er["gross_notional_usd"], errors="coerce") or 0.0))
            if inv_gross <= 0.0 and und_gross <= 0.0:
                continue
            out[(etf_n, und_n)] = {
                "inverse_etf_short_usd": inv_gross,
                "underlying_short_usd": und_gross,
            }
    return out


def write_held_legs_audit(
    run_date: str,
    *,
    runs_root: Path | None = None,
) -> Path | None:
    """Write human-readable reconstruction audit for the run folder."""
    src_date, _ = resolve_held_detail_run_date(run_date, runs_root=runs_root)
    legs = held_inverse_short_by_pair(run_date, runs_root=runs_root)
    if not legs:
        return None
    rows = [
        {
            "source_detail_date": src_date,
            "ETF": k[0],
            "Underlying": k[1],
            "inverse_etf_short_usd": v.get("inverse_etf_short_usd", 0.0),
            "underlying_short_usd": v.get("underlying_short_usd", 0.0),
        }
        for k, v in sorted(legs.items())
    ]
    out_dir = (runs_root or RUNS) / str(run_date) / "b4_hedge_cadence"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "b4_held_legs_reconstructed.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-date", required=True)
    ap.add_argument("--runs-root", default=str(RUNS))
    args = ap.parse_args(argv)
    runs_root = Path(args.runs_root).resolve()
    legs = held_inverse_short_by_pair(args.run_date, runs_root=runs_root)
    src_date, src_path = resolve_held_detail_run_date(args.run_date, runs_root=runs_root)
    audit = write_held_legs_audit(args.run_date, runs_root=runs_root)
    print(
        f"[b4_reconstruct_state] run={args.run_date} source={src_date or 'none'} "
        f"pairs={len(legs)} audit={audit}"
    )
    if src_path is None:
        print("[WARN] no net_exposure_bucket_4_detail.csv found on or before run date", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
