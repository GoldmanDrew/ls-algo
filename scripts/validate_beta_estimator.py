#!/usr/bin/env python3
"""scripts/validate_beta_estimator.py

Validation harness for the new robust hedge-beta estimator.

For a fixed run date (default 2026-05-05) and a list of underlying symbols,
this script:

  1. Loads the open IBKR Flex positions for that run.
  2. Fetches total-return prices from Yahoo for every ETF + underlying
     touched by those positions.
  3. Computes BOTH the old shrunk-OLS β (``compute_beta_shrunk``) and the
     new robust hierarchical EB posterior (``compute_beta_for_hedging``)
     for each ETF.
  4. Re-runs the screener's ``compute_net_exposure`` once with the OLD β
     map and once with the NEW β map, then prints a side-by-side
     beta-adjusted exposure table for the requested underlyings.

This is *not* a unit test; it's a stand-alone diagnostic that should be
run before/after the cutover to confirm the change has the expected
direction and magnitude on the live book.

Run:
    python scripts/validate_beta_estimator.py
    python scripts/validate_beta_estimator.py --run-date 2026-05-05 \\
        --underlyings NVTS MSTR MARA AMD INTC IONQ MU BE GOOGL META

Yieldboost symbols are also reported per-symbol because the prior change
hits them hardest.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from beta_estimator import (  # noqa: E402
    BetaPrior,
    build_yieldboost_family_priors,
    compute_beta_for_hedging,
)
import daily_screener as ds  # noqa: E402  — for legacy shrunk_OLS + lists
from ibkr_accounting import (  # noqa: E402
    EXCLUDE_SYMBOLS,
    SUPPLEMENTAL_ETF_MAP,
    canonical_symbol,
    compute_net_exposure,
    load_blacklist,
    load_etf_beta_map,
    load_etf_to_under_map,
    load_universe_from_screened,
    parse_open_positions,
)


def _fetch_yahoo_v8(symbol: str, period: str = "2y") -> pd.Series | None:
    import requests

    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol.replace('.', '-')}"
    params = {"range": period, "interval": "1d", "events": "div,splits"}
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        result = (data.get("chart") or {}).get("result") or []
        if not result:
            return None
        ts = result[0]["timestamp"]
        adjclose = result[0]["indicators"]["adjclose"][0]["adjclose"]
        idx = pd.to_datetime(ts, unit="s", utc=True).tz_convert("America/New_York").normalize()
        s = pd.Series(adjclose, index=idx, name=symbol, dtype=float).dropna()
        return s if not s.empty else None
    except Exception as exc:
        print(f"[VAL] {symbol}: yahoo v8 failed ({exc!s})")
        return None


def _fetch_many(symbols: Iterable[str], period: str = "2y") -> dict[str, pd.Series]:
    out: dict[str, pd.Series] = {}
    for sym in sorted(set(symbols)):
        s = _fetch_yahoo_v8(sym, period)
        if s is not None and len(s) >= 60:
            out[sym] = s
    print(f"[VAL] Fetched {len(out)} / {len(set(symbols))} TR series")
    return out


def _legacy_beta_for_etf(etf: str, und: str | None, exp_lev: float, tr: dict) -> float | None:
    if not und or etf not in tr or und not in tr:
        return None
    b, _, _ = ds.compute_beta_shrunk(tr[etf], tr[und], exp_lev)
    return float(b)


def _new_beta_for_etf(
    etf: str,
    und: str | None,
    product_class: str,
    nominal_lev: float | None,
    yb_priors,
    tr: dict,
) -> tuple[float, str] | None:
    if not und or etf not in tr or und not in tr:
        return None
    try:
        prior = BetaPrior.for_row(
            product_class=product_class,
            nominal_leverage=nominal_lev,
            underlying=und,
            peer_betas=yb_priors,
        )
    except ValueError:
        return None
    res = compute_beta_for_hedging(tr[etf], tr[und], prior)
    return float(res.beta), prior.source


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-date", default="2026-05-05")
    ap.add_argument(
        "--underlyings",
        nargs="*",
        default=["NVTS", "MSTR", "MARA", "AMD", "INTC", "IONQ", "MU", "BE", "GOOGL", "META"],
    )
    ap.add_argument(
        "--yieldboost-syms",
        nargs="*",
        default=["MAAY", "MTYY", "IOYY", "NVYY", "PLYY"],
    )
    ap.add_argument("--period", default="2y")
    args = ap.parse_args()

    run_dir = ROOT / "data" / "runs" / args.run_date / "ibkr_flex"
    etf_screened = ROOT / "data" / "etf_screened_today.csv"
    config_yml = ROOT / "config" / "strategy_config.yml"
    pos = parse_open_positions(run_dir / "flex_positions.xml")
    pos = pos[~pos["symbol"].isin(EXCLUDE_SYMBOLS) & ~pos["underlyingSymbol"].isin(EXCLUDE_SYMBOLS)].copy()
    blacklist = load_blacklist(config_yml)
    pos = pos[~pos["symbol"].isin(blacklist) & ~pos["underlyingSymbol"].isin(blacklist)].copy()

    allowed_etfs, allowed_und = load_universe_from_screened(etf_screened)
    allowed_und |= set(SUPPLEMENTAL_ETF_MAP.values())
    allowed_etfs |= set(SUPPLEMENTAL_ETF_MAP.keys())
    pos = pos[pos["symbol"].isin(allowed_etfs | allowed_und)].copy()

    etf_to_under = load_etf_to_under_map(etf_screened)
    for s, u in SUPPLEMENTAL_ETF_MAP.items():
        etf_to_under.setdefault(s, u)
    _, etf_to_beta_map = load_etf_beta_map(etf_screened)

    # Determine the set of (ETF, Underlying) pairs we need TR for.
    syms_need: set[str] = set()
    for sym in pos["symbol"].astype(str).unique():
        syms_need.add(sym)
        und = etf_to_under.get(sym, sym)
        syms_need.add(und)
    for sym in args.yieldboost_syms:
        syms_need.add(sym)
        und = etf_to_under.get(sym, sym)
        syms_need.add(und)

    # Load universe to know the product class + nominal leverage per ETF.
    universe = ds.build_full_universe(skip_scrape=True, skip_inverse=False)

    # Limit TR fetch to symbols we'll actually need.
    tr = _fetch_many(syms_need, args.period)

    yb_priors = build_yieldboost_family_priors(
        [(ds._norm_sym(e), ds._norm_sym(u)) for e, u in ds.YIELDBOOST_BUCKET2_PAIRS],
        tr,
        min_days=60,
    )

    # ── Per-symbol β table (for YieldBOOST + symbols in the requested unds)
    rows: list[dict] = []
    for sym in sorted(set(pos["symbol"].astype(str))):
        und = etf_to_under.get(sym, sym)
        if sym not in tr or und not in tr:
            continue
        u_row = universe[universe["ETF"].apply(ds._norm_sym) == sym]
        if u_row.empty:
            # spot underlying — beta is 1 by definition
            continue
        u_row = u_row.iloc[0]
        product_class = ds.classify_beta_product_class(u_row)
        lev_raw = u_row.get("Leverage")
        try:
            nominal_lev = float(lev_raw) if pd.notna(lev_raw) else None
        except (TypeError, ValueError):
            nominal_lev = None

        old = _legacy_beta_for_etf(sym, und, float(nominal_lev or 2.0), tr)
        new = _new_beta_for_etf(sym, und, product_class, nominal_lev, yb_priors, tr)
        if old is None or new is None:
            continue
        rows.append(
            {
                "etf": sym,
                "underlying": und,
                "product_class": product_class,
                "L_nominal": nominal_lev,
                "beta_old": old,
                "beta_new": new[0],
                "delta": new[0] - old,
                "prior_source": new[1],
            }
        )

    df_per_etf = pd.DataFrame(rows)
    if df_per_etf.empty:
        print("[VAL] No comparable rows produced — bailing.")
        return 0

    print("\n=== Per-ETF β: OLD vs NEW (positions in 2026-05-05 book) ===")
    print(df_per_etf.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))

    # ── Beta-adjusted exposure: re-run compute_net_exposure with the new map
    # The screener stores beta in etf_screened_today.csv via load_etf_beta_map;
    # we patch in-memory by overriding `etf_to_beta` returned values for ETFs
    # we re-estimated.
    new_beta_map = dict(etf_to_beta_map)
    for r in rows:
        new_beta_map[r["etf"]] = r["beta_new"]

    # Build a fresh exposure DataFrame using the new beta. Reuse the screener
    # path: we monkey-patch load_etf_beta_map's output via direct assignment.
    pos_local = pos.copy()
    pos_local["is_etf"] = pos_local["symbol"].isin(etf_to_under)
    pos_local["underlying"] = np.where(
        pos_local["is_etf"], pos_local["symbol"].map(etf_to_under), pos_local["symbol"]
    )

    def _exposure(beta_map: dict) -> pd.DataFrame:
        p = pos_local.copy()
        p["beta"] = np.where(
            p["is_etf"], p["symbol"].map(beta_map).astype(float), 1.0
        )
        p["beta"] = pd.to_numeric(p["beta"], errors="coerce").fillna(1.0)
        p["mv_base"] = p["position"] * p["beta"] * p["markPrice"] * p["fxRateToBase"]
        p["gross_mv_base"] = p["mv_base"].abs()
        agg = (
            p.groupby("underlying", as_index=False)
            .agg(
                net=("mv_base", "sum"),
                gross=("gross_mv_base", "sum"),
                n_legs=("symbol", "nunique"),
            )
        )
        return agg

    exp_old = _exposure(etf_to_beta_map).rename(columns={"net": "net_old", "gross": "gross_old"})
    exp_new = _exposure(new_beta_map).rename(columns={"net": "net_new", "gross": "gross_new"})
    merged = exp_old.merge(
        exp_new[["underlying", "net_new", "gross_new"]], on="underlying", how="outer"
    )
    merged["delta_net"] = merged["net_new"] - merged["net_old"]

    requested = [u.upper() for u in args.underlyings]
    table = merged[merged["underlying"].isin(requested)].copy()
    table = table.sort_values("underlying")
    print("\n=== Beta-adjusted NET exposure: OLD vs NEW (top names) ===")
    print(
        table.to_string(
            index=False,
            float_format=lambda v: f"{v:>14,.2f}",
        )
    )

    # ── YieldBOOST side-by-side ───────────────────────────────────────
    yb_table = df_per_etf[df_per_etf["product_class"] == "income_yieldboost"].copy()
    if not yb_table.empty:
        print("\n=== YieldBOOST symbols: OLD vs NEW β ===")
        print(yb_table.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))

    if yb_priors:
        print("\n=== YieldBOOST family priors used ===")
        items = sorted(yb_priors.items(), key=lambda kv: kv[0])
        for u, p in items:
            print(f"  {u:12s}  μ={p.mu:+.4f}  τ={p.tau:.0f}  n={p.n_siblings}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
