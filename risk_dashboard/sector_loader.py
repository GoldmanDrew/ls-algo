"""Tiered sector attribution for book underlyings.

Replaces the single-source ``lookup_underlying`` helper in
``risk_dashboard.factor_map`` with a 5-tier resolver:

    1. OVERRIDE_SECTOR_MAP     -- hand-curated thematic buckets
       (quantum, crypto-equity, evtol, drones, space, ...). GICS-style
       feeds cannot express these.
    2. Screener metadata       -- ``data/etf_screened_today.csv``
       per-underlying ``theme`` / ``sector`` columns when present.
    3. Vendor (yfinance)       -- ``Ticker(sym).info["sector"]`` and
       ``["industry"]`` mapped through ``VENDOR_SECTOR_MAP``.
    4. Heuristic               -- keyword regex on ``longName`` /
       ``industry`` for new thematics not yet in the override map.
    5. Default                 -- ``"other"``.

Output dict::

    {
        "sector": "semis",
        "sector_source": "vendor",          # tier name
        "sector_confidence": 0.85,           # 0..1, drops by tier
    }

The vendor / heuristic tiers are pure functions of the data passed in
-- this module does NOT call yfinance directly. ``risk_dashboard``
callers fetch vendor info once per build and pass it in.
"""

from __future__ import annotations

import re
from typing import Any, Mapping

from .factor_map import OVERRIDE_SECTOR_MAP


__all__ = [
    "TIER_CONFIDENCE",
    "VENDOR_SECTOR_MAP",
    "HEURISTIC_PATTERNS",
    "resolve_sector",
    "batch_resolve",
]


TIER_CONFIDENCE: dict[str, float] = {
    "override": 1.00,
    "screener": 0.90,
    "vendor": 0.75,
    "heuristic": 0.55,
    "default": 0.10,
}


# yfinance ``info["sector"]`` / ``["industry"]`` strings mapped into
# our taxonomy. Lowercased on lookup. Industry-level keys win over
# sector-level (more specific).
VENDOR_SECTOR_MAP: dict[str, str] = {
    # GICS sectors
    "technology": "software",
    "communication services": "mega-cap-tech",
    "consumer cyclical": "consumer",
    "consumer defensive": "consumer",
    "healthcare": "healthcare",
    "health care": "healthcare",
    "financial services": "fintech",
    "financial": "fintech",
    "industrials": "industrials",
    "energy": "energy",
    "utilities": "energy",
    "basic materials": "metals",
    "materials": "metals",
    "real estate": "other",
    # Industries (more specific; override sector hit)
    "semiconductors": "semis",
    "semiconductor equipment & materials": "semis",
    "software-infrastructure": "software",
    "software-application": "software",
    "information technology services": "software",
    "internet content & information": "mega-cap-tech",
    "consumer electronics": "mega-cap-tech",
    "credit services": "fintech",
    "capital markets": "fintech",
    "insurance-diversified": "insurtech",
    "insurance-life": "healthcare",
    "biotechnology": "healthcare",
    "drug manufacturers-general": "healthcare",
    "drug manufacturers-specialty & generic": "healthcare",
    "medical devices": "healthcare",
    "oil & gas e&p": "energy",
    "oil & gas integrated": "energy",
    "oil & gas midstream": "energy",
    "uranium": "nuclear",
    "utilities-renewable": "clean",
    "utilities-regulated electric": "energy",
    "gold": "metals",
    "silver": "metals",
    "copper": "metals",
    "other precious metals & mining": "metals",
    "other industrial metals & mining": "metals",
    "aluminum": "metals",
    "lithium": "metals",
    "rare earths & strategic minerals": "metals",
    "aerospace & defense": "industrials",
    "airlines": "industrials",
    "auto manufacturers": "consumer",
    "auto & truck dealerships": "consumer",
    "internet retail": "consumer",
    "restaurants": "consumer",
}


# Heuristic regex set keyed by sector. Iteration order matters: more
# specific multi-word patterns (crypto-equity miner) must precede the
# generic single-token patterns (crypto bitcoin) so the resolver picks
# the tighter label first.
HEURISTIC_PATTERNS: dict[str, list[str]] = {
    "crypto-equity": [
        r"\bcrypto miner\b",
        r"\bbitcoin miner\b",
        r"\bdigital asset (trust|treasury)\b",
    ],
    "quantum": [r"\bquantum\b", r"\bqpu\b"],
    "nuclear": [r"\bnuclear\b", r"\buranium\b", r"\bsmr\b", r"small modular reactor"],
    "crypto": [r"\bbitcoin\b", r"\bether(eum)?\b", r"\bcrypto\b", r"\bxrp\b", r"\bsolana\b"],
    "drones": [r"\bdrone\b", r"\buav\b", r"unmanned aerial"],
    "evtol": [r"\bevtol\b", r"electric (vertical|aircraft)", r"\bair taxi\b"],
    "space": [r"\bspace\b", r"\bsatellite\b", r"\blaunch vehicle\b", r"\borbital\b"],
    "metals": [r"\blithium\b", r"\brare earth\b", r"\bcopper\b", r"\bgold (miner|mining)\b"],
    "clean": [r"\bhydrogen\b", r"fuel cell", r"\bsolar\b", r"battery (storage|tech)"],
    "fintech": [r"\bfintech\b", r"\bneobank\b", r"buy[- ]now[- ]pay[- ]later"],
    "semis": [r"\bsemiconductor\b", r"\bfab(less)?\b", r"\bwafer\b"],
    "software": [r"\bsoftware\b", r"\bsaas\b", r"\bcloud\b", r"\bai (platform|infra)\b"],
    "healthcare": [r"\bbiotech\b", r"\bpharmaceutical\b", r"\bmedical device\b"],
    "insurtech": [r"\binsurtech\b", r"digital insur"],
    "china": [r"\bchina\b", r"\bchinese\b", r"hong kong"],
}


_COMPILED_HEURISTICS: dict[str, list[re.Pattern[str]]] | None = None


def _compile_heuristics() -> dict[str, list[re.Pattern[str]]]:
    global _COMPILED_HEURISTICS
    if _COMPILED_HEURISTICS is None:
        _COMPILED_HEURISTICS = {
            sec: [re.compile(p, re.IGNORECASE) for p in patterns]
            for sec, patterns in HEURISTIC_PATTERNS.items()
        }
    return _COMPILED_HEURISTICS


def _result(sector: str, source: str) -> dict[str, Any]:
    return {
        "sector": sector,
        "sector_source": source,
        "sector_confidence": TIER_CONFIDENCE.get(source, 0.1),
    }


def _try_override(symbol: str) -> dict[str, Any] | None:
    sec = OVERRIDE_SECTOR_MAP.get(symbol)
    if sec:
        return _result(sec, "override")
    return None


def _try_screener(screener_row: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not screener_row:
        return None
    for key in (
        "underlying_sector",
        "sector",
        "theme",
        "underlying_theme",
        "product_class",
        "Delta_product_class",
    ):
        val = screener_row.get(key)
        if val is None:
            continue
        text = str(val).strip().lower()
        if not text or text in {"nan", "none", "other", "unknown"}:
            continue
        return _result(text, "screener")
    return None


def _try_vendor(vendor_info: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not vendor_info:
        return None
    industry = (vendor_info.get("industry") or "").strip().lower()
    if industry and industry in VENDOR_SECTOR_MAP:
        return _result(VENDOR_SECTOR_MAP[industry], "vendor")
    sector = (vendor_info.get("sector") or "").strip().lower()
    if sector and sector in VENDOR_SECTOR_MAP:
        return _result(VENDOR_SECTOR_MAP[sector], "vendor")
    return None


def _try_heuristic(vendor_info: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not vendor_info:
        return None
    text_fields = []
    for key in ("longName", "long_name", "shortName", "short_name", "industry", "longBusinessSummary"):
        val = vendor_info.get(key)
        if val:
            text_fields.append(str(val))
    if not text_fields:
        return None
    blob = " ".join(text_fields).lower()
    for sector, patterns in _compile_heuristics().items():
        for pat in patterns:
            if pat.search(blob):
                return _result(sector, "heuristic")
    return None


def resolve_sector(
    symbol: str,
    *,
    screener_row: Mapping[str, Any] | None = None,
    vendor_info: Mapping[str, Any] | None = None,
    use_override: bool = False,
) -> dict[str, Any]:
    """Resolve a single underlying through the 5-tier chain.

    Parameters
    ----------
    symbol:
        Underlying ticker (case-insensitive). Returns ``{"sector": "other",
        "sector_source": "default"}`` if blank.
    screener_row:
        Optional row from ``data/etf_screened_today.csv`` aggregated to
        the underlying. We check ``underlying_sector`` / ``sector`` /
        ``theme`` columns in order.
    vendor_info:
        Optional yfinance ``Ticker.info`` dict. We read ``industry``,
        ``sector``, ``longName`` for vendor + heuristic tiers.

    The function is pure -- no network calls. Callers fetch and cache
    vendor info upstream.
    """
    sym = (symbol or "").strip().upper()
    if not sym:
        return _result("other", "default")

    tiers: list[tuple[Any, Any]] = []
    if use_override:
        tiers.append((_try_override, sym))
    tiers.extend(
        [
            (_try_screener, screener_row),
            (_try_vendor, vendor_info),
            (_try_heuristic, vendor_info),
        ]
    )
    for tier_fn, arg in tiers:
        hit = tier_fn(arg)  # type: ignore[arg-type]
        if hit is not None:
            return hit

    return _result("other", "default")


def batch_resolve(
    symbols: list[str],
    *,
    screener_rows: Mapping[str, Mapping[str, Any]] | None = None,
    vendor_info_by_symbol: Mapping[str, Mapping[str, Any]] | None = None,
    use_override: bool = False,
) -> dict[str, dict[str, Any]]:
    """Convenience wrapper. Resolves a list of underlyings in one call."""
    out: dict[str, dict[str, Any]] = {}
    s_rows = screener_rows or {}
    v_info = vendor_info_by_symbol or {}
    for raw in symbols:
        sym = (raw or "").strip().upper()
        if not sym:
            continue
        out[sym] = resolve_sector(
            sym,
            screener_row=s_rows.get(sym),
            vendor_info=v_info.get(sym),
            use_override=use_override,
        )
    return out
