"""Fetch and cache yfinance ``Ticker.info`` for sector_loader vendor/heuristic tiers."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Iterable

from .factor_map import OVERRIDE_SECTOR_MAP

CACHE_PATH_DEFAULT = Path("data/cache/sector_vendor.json")
CACHE_MAX_AGE_HOURS = 168.0  # 7 days
FETCH_SLEEP_SECONDS = 0.35

_INFO_KEYS = (
    "sector",
    "industry",
    "longName",
    "shortName",
    "longBusinessSummary",
)


def _load_cache(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"symbols": {}}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"symbols": {}}


def _save_cache(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _extract_info(raw: dict[str, Any] | None) -> dict[str, Any]:
    if not raw:
        return {}
    out: dict[str, Any] = {}
    for key in _INFO_KEYS:
        val = raw.get(key)
        if val is not None and str(val).strip():
            out[key] = val
    return out


def fetch_vendor_info(
    symbols: Iterable[str],
    *,
    cache_path: Path | None = None,
    refresh_max_age_hours: float = CACHE_MAX_AGE_HOURS,
    yf_module: Any | None = None,
) -> dict[str, dict[str, Any]]:
    """Return ``{SYMBOL: {sector, industry, longName, ...}}`` with disk cache."""
    cache_path = cache_path or CACHE_PATH_DEFAULT
    now = time.time()
    cache = _load_cache(cache_path)
    sym_cache: dict[str, dict[str, Any]] = dict(cache.get("symbols") or {})

    needed: list[str] = []
    out: dict[str, dict[str, Any]] = {}
    for raw in symbols:
        sym = str(raw or "").strip().upper()
        if not sym:
            continue
        entry = sym_cache.get(sym) or {}
        age_h = entry.get("fetched_at_epoch")
        info = _extract_info(entry.get("info"))
        if info and age_h is not None and (now - float(age_h)) / 3600.0 <= refresh_max_age_hours:
            out[sym] = info
            continue
        needed.append(sym)

    if needed:
        yf = yf_module
        if yf is None:
            import yfinance as yf  # noqa: PLC0415

        # Override map wins tier-1; skip network for those symbols.
        needed = [s for s in needed if s not in OVERRIDE_SECTOR_MAP]

        for sym in needed:
            info: dict[str, Any] = {}
            try:
                info = _extract_info(yf.Ticker(sym).info)
            except Exception:
                info = {}
            sym_cache[sym] = {"fetched_at_epoch": now, "info": info}
            if info:
                out[sym] = info
            if FETCH_SLEEP_SECONDS > 0:
                time.sleep(FETCH_SLEEP_SECONDS)

        cache["symbols"] = sym_cache
        cache["updated_at_epoch"] = now
        _save_cache(cache_path, cache)

    return out
