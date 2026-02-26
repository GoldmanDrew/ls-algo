#!/usr/bin/env python3
"""
fetch_ibkr_margin.py — Pull per-symbol maintenance margin from IBKR via TWS API.

Requires:
  - TWS or IB Gateway running (TWS port=7497, Gateway port=4002)
  - pip install ib_insync

Reads:  data/etf_screened_today.csv (or --csv path)
        config/strategy_config.yml  (for IBKR connection params)
Writes: data/ibkr_margin_requirements.csv
        data/ibkr_pair_margin.csv

Usage:
  python fetch_ibkr_margin.py                          # uses config for host/port
  python fetch_ibkr_margin.py --port 4002              # override for IB Gateway
  python fetch_ibkr_margin.py --symbols AMZN CRWV SOXL # ad-hoc test
  python fetch_ibkr_margin.py --max-symbols 20         # quick test
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

UNSET_DOUBLE = 1.7976931348623157e308
NOTIONAL_PER_TEST = 10_000
RATE_LIMIT_SLEEP = 0.4
MAX_RETRIES = 2


def _load_config() -> dict:
    """Load strategy_config.yml from common locations."""
    script_dir = Path(__file__).resolve().parent
    for candidate in [
        script_dir / "strategy_config.yml",
        script_dir / "config" / "strategy_config.yml",
        Path("strategy_config.yml"),
        Path("config/strategy_config.yml"),
    ]:
        if candidate.exists():
            with open(candidate) as f:
                cfg = yaml.safe_load(f) or {}
            print(f"[CONFIG] Loaded: {candidate}")
            return cfg
    print("[CONFIG] No strategy_config.yml found — using defaults")
    return {}


def _parse_margin_val(s: str) -> float:
    try:
        v = float(s)
        return np.nan if v >= UNSET_DOUBLE * 0.9 else v
    except (ValueError, TypeError):
        return np.nan


def fetch_margin_for_symbol(ib, symbol: str, notional: float = NOTIONAL_PER_TEST) -> dict:
    """Fetch init + maint margin for BUY and SELL via whatIfOrder."""
    from ib_insync import Stock, MarketOrder

    result = {"symbol": symbol}
    try:
        contract = Stock(symbol, "SMART", "USD")
        qualified = ib.qualifyContracts(contract)
        if not qualified:
            return {**result, "error": "qualify_failed"}

        ib.reqMarketDataType(4)  # frozen/delayed OK
        [ticker] = ib.reqTickers(contract)
        ib.sleep(0.3)

        price = ticker.marketPrice()
        if not np.isfinite(price) or price <= 0:
            price = ticker.close
        if not np.isfinite(price) or price <= 0:
            return {**result, "error": "no_price"}

        shares = max(1, int(notional / price))
        actual_notional = shares * price
        result.update({"price": round(price, 4), "shares_tested": shares,
                       "notional": round(actual_notional, 2)})

        # BUY (long)
        for attempt in range(MAX_RETRIES + 1):
            try:
                buy_state = ib.whatIfOrder(contract, MarketOrder("BUY", shares))
                init_long = _parse_margin_val(buy_state.initMarginChange)
                maint_long = _parse_margin_val(buy_state.maintMarginChange)
                result["init_margin_long"] = init_long
                result["maint_margin_long"] = maint_long
                result["init_pct_long"] = round(init_long / actual_notional, 6) if np.isfinite(init_long) else np.nan
                result["maint_pct_long"] = round(maint_long / actual_notional, 6) if np.isfinite(maint_long) else np.nan
                result["warning_long"] = buy_state.warningText or ""
                break
            except Exception as e:
                if attempt < MAX_RETRIES:
                    ib.sleep(1)
                else:
                    result["error_long"] = str(e)

        ib.sleep(RATE_LIMIT_SLEEP)

        # SELL (short)
        for attempt in range(MAX_RETRIES + 1):
            try:
                sell_state = ib.whatIfOrder(contract, MarketOrder("SELL", shares))
                init_short = abs(_parse_margin_val(sell_state.initMarginChange))
                maint_short = abs(_parse_margin_val(sell_state.maintMarginChange))
                result["init_margin_short"] = init_short
                result["maint_margin_short"] = maint_short
                result["init_pct_short"] = round(init_short / actual_notional, 6) if np.isfinite(init_short) else np.nan
                result["maint_pct_short"] = round(maint_short / actual_notional, 6) if np.isfinite(maint_short) else np.nan
                result["warning_short"] = sell_state.warningText or ""
                break
            except Exception as e:
                if attempt < MAX_RETRIES:
                    ib.sleep(1)
                else:
                    result["error_short"] = str(e)

        ib.sleep(RATE_LIMIT_SLEEP)

    except Exception as e:
        result["error"] = str(e)
    return result


def compute_pair_margin(margin_df: pd.DataFrame,
                        screened_df: pd.DataFrame) -> pd.DataFrame:
    """pair_maint = maint_short(ETF) + (1/|β|) × maint_long(Underlying)."""
    lookup = margin_df.set_index("symbol")
    rows = []
    for _, r in screened_df.iterrows():
        etf = str(r.get("ETF", "")).strip().upper()
        und = str(r.get("Underlying", "")).strip().upper()
        beta = float(r["Beta"]) if pd.notna(r.get("Beta")) else np.nan
        if not etf or not und or und == "NAN" or not np.isfinite(beta):
            continue
        hr = 1.0 / max(abs(beta), 0.5)
        em = lookup.loc[etf, "maint_pct_short"] if etf in lookup.index else np.nan
        um = lookup.loc[und, "maint_pct_long"] if und in lookup.index else np.nan
        pm = (em + hr * um) if np.isfinite(em) and np.isfinite(um) else np.nan
        rows.append({"ETF": etf, "Underlying": und, "Beta": beta,
                     "maint_pct_short_etf": em, "maint_pct_long_und": um,
                     "pair_maint_pct": round(pm, 6) if np.isfinite(pm) else np.nan})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description="Fetch IBKR margin via TWS API")
    ap.add_argument("--csv", default="data/etf_screened_today.csv")
    ap.add_argument("--output", default="data/ibkr_margin_requirements.csv")
    ap.add_argument("--pair-output", default="data/ibkr_pair_margin.csv")
    ap.add_argument("--host", default=None, help="TWS host (default: from config or 127.0.0.1)")
    ap.add_argument("--port", type=int, default=None, help="TWS=7497, Gateway=4002 (default: from config)")
    ap.add_argument("--client-id", type=int, default=None, help="API client ID (default: from config)")
    ap.add_argument("--symbols", nargs="*", help="Ad-hoc symbol list (overrides CSV)")
    ap.add_argument("--notional", type=float, default=NOTIONAL_PER_TEST)
    ap.add_argument("--max-symbols", type=int, default=None)
    args = ap.parse_args()

    # ── Read connection params from config, CLI overrides ──
    cfg = _load_config()
    ibkr_cfg = cfg.get("ibkr", {}) or {}

    host = args.host or str(ibkr_cfg.get("host", "127.0.0.1"))
    port = args.port or int(ibkr_cfg.get("port", 7497))
    client_id = args.client_id or int(ibkr_cfg.get("client_id", 99))

    # ── Build symbol list ──
    screened_df = None
    if args.symbols:
        symbols = [s.upper().replace(".", "-") for s in args.symbols]
        print(f"Ad-hoc mode: {len(symbols)} symbols")
    else:
        csv_path = Path(args.csv)
        if not csv_path.exists():
            print(f"ERROR: CSV not found: {csv_path}")
            return 1
        screened_df = pd.read_csv(csv_path)
        etfs = screened_df["ETF"].dropna().astype(str).str.strip().str.upper().tolist()
        unds = [u for u in screened_df["Underlying"].dropna().astype(str).str.strip().str.upper().tolist()
                if u and u != "NAN"]
        symbols = sorted(set(etfs + unds))
        print(f"Loaded {len(screened_df)} pairs → {len(symbols)} unique symbols")

    if args.max_symbols:
        symbols = symbols[:args.max_symbols]
        print(f"Capped to {len(symbols)} symbols")

    # ── Connect ──
    try:
        from ib_insync import IB
    except ImportError:
        print("ERROR: ib_insync not installed. Run: pip install ib_insync")
        return 1

    ib = IB()
    print(f"\nConnecting to {host}:{port} (clientId={client_id}) ...")
    try:
        ib.connect(host, port, clientId=client_id, timeout=10)
    except Exception as e:
        print(f"ERROR: Could not connect: {e}")
        print("Make sure TWS or IB Gateway is running with API enabled.")
        return 1
    print(f"Connected. Accounts: {ib.managedAccounts()}")

    # ── Fetch ──
    results = []
    t0 = time.monotonic()
    n = len(symbols)
    for i, sym in enumerate(symbols, 1):
        if i % 25 == 0 or i == 1 or i == n:
            elapsed = time.monotonic() - t0
            eta = (elapsed / i * (n - i)) if i > 1 else 0
            print(f"  [{i}/{n}] {sym}  ({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)")
        row = fetch_margin_for_symbol(ib, sym, notional=args.notional)
        row["timestamp"] = datetime.utcnow().isoformat()
        results.append(row)

    ib.disconnect()
    elapsed = time.monotonic() - t0
    print(f"\nDone: {len(results)} symbols in {elapsed:.0f}s")

    # ── Save ──
    margin_df = pd.DataFrame(results)
    ok = margin_df["maint_pct_long"].notna().sum() if "maint_pct_long" in margin_df.columns else 0
    err = margin_df.get("error", pd.Series(dtype=str)).notna().sum()
    print(f"Success: {ok} | Errors: {err}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    margin_df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")

    # ── Pair margin ──
    if screened_df is not None and not margin_df.empty:
        pair_df = compute_pair_margin(margin_df, screened_df)
        if not pair_df.empty:
            pair_path = Path(args.pair_output)
            pair_path.parent.mkdir(parents=True, exist_ok=True)
            pair_df.to_csv(pair_path, index=False)
            print(f"Saved: {pair_path}")
            valid = pair_df["pair_maint_pct"].dropna()
            if len(valid):
                print(f"\n  Pair margin: {valid.min()*100:.1f}%–{valid.max()*100:.1f}% "
                      f"(median {valid.median()*100:.1f}%)")
                print(f"\n  Most capital-intensive:")
                for _, r in pair_df.nlargest(10, "pair_maint_pct").iterrows():
                    print(f"    {r['ETF']:8s}/{r['Underlying']:6s}  "
                          f"pair={r['pair_maint_pct']*100:5.1f}%  "
                          f"(short={r['maint_pct_short_etf']*100:5.1f}%, "
                          f"long={r['maint_pct_long_und']*100:5.1f}%)")
                print(f"\n  Most capital-efficient:")
                for _, r in pair_df.nsmallest(10, "pair_maint_pct").iterrows():
                    print(f"    {r['ETF']:8s}/{r['Underlying']:6s}  "
                          f"pair={r['pair_maint_pct']*100:5.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
