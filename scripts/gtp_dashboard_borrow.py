"""Load average IBKR borrow from sibling ``etf-dashboard`` ``dashboard_data.json``."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _norm_sym(x: str) -> str:
    return str(x).strip().upper().replace(".", "-")


def dashboard_data_candidates() -> list[Path]:
    return [
        Path("../etf-dashboard/data/dashboard_data.json"),
        Path("../../etf-dashboard/data/dashboard_data.json"),
        Path.cwd().parent / "etf-dashboard" / "data" / "dashboard_data.json",
        Path.cwd() / "etf-dashboard" / "data" / "dashboard_data.json",
        Path(r"C:/Users/werdn/Documents/Investing/etf-dashboard/data/dashboard_data.json"),
    ]


def load_dashboard_borrow_avg_annual() -> tuple[dict[str, float], Path | None]:
    """
    Return (ticker -> annual borrow as **decimal**, e.g. 0.05 for 5%), resolved path or None.

    Expects ``dashboard_data.json`` to be either a list of records or ``{"records": [...]}``.
    Each record should include ``ETF`` and ``borrow_avg_annual`` (numeric; percent or decimal).
    """
    src = next((p for p in dashboard_data_candidates() if p.exists()), None)
    if src is None:
        return {}, None
    raw = json.loads(src.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]]
    if isinstance(raw, list):
        rows = raw
    else:
        rows = list(raw.get("records") or raw.get("data") or [])
    out: dict[str, float] = {}
    for r in rows:
        if not isinstance(r, dict):
            continue
        sym = r.get("ETF") or r.get("etf") or r.get("ticker")
        if sym is None:
            continue
        k = _norm_sym(str(sym))
        v = r.get("borrow_avg_annual")
        if v is None:
            continue
        x = float(pd.to_numeric(v, errors="coerce"))
        if not np.isfinite(x):
            continue
        # Heuristic: values > 1 are percent points (e.g. 3.5 meaning 3.5%)
        if abs(x) > 1.0:
            x = x / 100.0
        out[k] = x
    return out, src


def merge_dashboard_borrow_into_map(
    borrow_map: dict[str, float],
    etf_syms: list[str],
) -> dict[str, float]:
    """Override *borrow_map* with dashboard ``borrow_avg_annual`` where present."""
    dash, src = load_dashboard_borrow_avg_annual()
    if not dash:
        print("[dashboard] borrow_avg: no dashboard_data.json found; keeping existing BORROW_MAP")
        return dict(borrow_map)
    out = dict(borrow_map)
    hit = 0
    for e in etf_syms:
        k = _norm_sym(e)
        if k in dash and np.isfinite(dash[k]):
            out[k] = float(dash[k])
            hit += 1
    print(f"[dashboard] borrow_avg overlay: {hit}/{len(etf_syms)} from {src}")
    return out
