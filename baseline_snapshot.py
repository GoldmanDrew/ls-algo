#!/usr/bin/env python3
"""
baseline_snapshot.py

Creates a baseline snapshot of IBKR positions to "segment" strategy holdings
inside a single IBKR account.

Strategy holdings are computed later as:
    strategy_qty(symbol) = current_ib_qty(symbol) - baseline_qty(symbol)

This script never places orders.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd
from ib_insync import IB


# ---------------------------
# Symbol normalization helpers
# ---------------------------

IB_SYMBOL_MAP: Dict[str, Tuple[str, Optional[str]]] = {
    "BRK-B": ("BRK B", "NYSE"),
    "BRK-A": ("BRK A", "NYSE"),
}

REVERSE_IB_SYMBOL_MAP: Dict[str, str] = {ib_sym: uni for uni, (ib_sym, _) in IB_SYMBOL_MAP.items()}


def universal_symbol_from_ib(symbol: str) -> str:
    s = str(symbol).strip().upper()
    return REVERSE_IB_SYMBOL_MAP.get(s, s)


# ---------------------------
# IBKR connection
# ---------------------------

def connect_ib(host: str, port: int, client_id: int) -> IB:
    ib = IB()
    ib.connect(host, port, clientId=client_id)
    if not ib.isConnected():
        raise RuntimeError("Failed to connect to IBKR.")
    return ib


def fetch_positions_df(ib: IB) -> pd.DataFrame:
    rows = []
    for p in ib.positions():
        c = p.contract
        rows.append(
            {
                "account": p.account,
                "symbol_ib": str(c.symbol),
                "localSymbol": str(getattr(c, "localSymbol", "")),
                "conId": int(getattr(c, "conId", 0)),
                "secType": str(getattr(c, "secType", "")),
                "currency": str(getattr(c, "currency", "")),
                "exchange": str(getattr(c, "exchange", "")),
                "primaryExchange": str(getattr(c, "primaryExchange", "")),
                "qty": float(p.position),
                "avgCost": float(getattr(p, "avgCost", 0.0)),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame(
            columns=[
                "account",
                "symbol_ib",
                "localSymbol",
                "conId",
                "secType",
                "currency",
                "exchange",
                "primaryExchange",
                "qty",
                "avgCost",
            ]
        )
    df["symbol"] = df["symbol_ib"].map(universal_symbol_from_ib)
    df["captured_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return df[
        [
            "captured_at",
            "account",
            "symbol",
            "symbol_ib",
            "localSymbol",
            "conId",
            "secType",
            "currency",
            "exchange",
            "primaryExchange",
            "qty",
            "avgCost",
        ]
    ]

from strategy_config import load_config

def main() -> None:
    cfg = load_config()  # reads config/strategy_config.yaml

    host = cfg["ibkr"]["host"]
    port = int(cfg["ibkr"]["port"])
    client_id = int(cfg["ibkr"]["client_id"])

    baseline_path = Path(cfg["paths"]["baseline_csv"])
    baseline_path.parent.mkdir(parents=True, exist_ok=True)

    ib = connect_ib(host, port, client_id)
    try:
        df = fetch_positions_df(ib)
        df.to_csv(baseline_path, index=False)
        print(f"[BASELINE] Wrote {len(df)} rows to: {baseline_path}")
    finally:
        ib.disconnect()


if __name__ == "__main__":
    main()
