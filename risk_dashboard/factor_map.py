"""Underlying -> sector and underlying -> SPY-beta map.

Hand-curated for the names that actually appear in the book. Defaults are
explicit so the dashboard can show "beta coverage" honestly instead of
silently assuming 1.0 for everything.

If a name is not in `SECTOR_MAP`, we tag it `other`. If it is not in
`BETA_TO_SPY`, we use `DEFAULT_SINGLE_NAME_BETA` and mark `beta_source =
"default"` so the SPA can dim those rows.
"""

from __future__ import annotations

from typing import Any

DEFAULT_SINGLE_NAME_BETA: float = 1.20
DEFAULT_BROAD_INDEX_BETA: float = 1.00

# Thematic overrides for sectors that GICS / vendor sector tags can't
# express (quantum, crypto-equity, evtol, drones, space, insurtech).
# This is tier 1 in :mod:`risk_dashboard.sector_loader`.
OVERRIDE_SECTOR_MAP: dict[str, str] = {
    # Broad indices / sector ETFs
    "SPY": "broad", "QQQ": "broad", "IWM": "broad", "DIA": "broad",
    "ARKK": "broad", "SOXX": "semis", "SOEZ": "broad",
    "EWW": "intl", "EWY": "intl", "FXI": "china", "KWEB": "china",
    "XOP": "energy", "COPX": "metals", "GDX": "metals", "GDXJ": "metals",
    "URA": "nuclear",
    # Semis / AI infra
    "NVDA": "semis", "AMD": "semis", "INTC": "semis", "AVGO": "semis",
    "MU": "semis", "MRVL": "semis", "ASML": "semis", "SMCI": "semis",
    "ALAB": "semis", "ANET": "semis", "TSM": "semis", "QCOM": "semis",
    "ARM": "semis", "KLAC": "semis", "LRCX": "semis", "TER": "semis",
    "CRDO": "semis", "COHR": "semis", "WDC": "semis", "SNDK": "semis",
    "GLW": "semis", "LITE": "semis", "APH": "semis", "CLS": "semis",
    "NVTS": "semis",
    # Crypto equity
    "COIN": "crypto-equity", "MSTR": "crypto-equity", "MARA": "crypto-equity",
    "RIOT": "crypto-equity", "CLSK": "crypto-equity", "HUT": "crypto-equity",
    "IREN": "crypto-equity", "WULF": "crypto-equity", "CIFR": "crypto-equity",
    "CORZ": "crypto-equity", "BMNR": "crypto-equity", "BLSH": "crypto-equity",
    "BULL": "crypto-equity", "GLXY": "crypto-equity", "GEMI": "crypto-equity",
    "SBET": "crypto-equity",
    # Crypto / digital asset exposure
    "IBIT": "crypto", "ETHA": "crypto", "XRPZ": "crypto",
    # Mega-cap tech
    "AAPL": "mega-cap-tech", "MSFT": "mega-cap-tech", "META": "mega-cap-tech",
    "GOOGL": "mega-cap-tech", "AMZN": "mega-cap-tech", "TSLA": "mega-cap-tech",
    "NFLX": "mega-cap-tech", "ORCL": "mega-cap-tech", "ADBE": "mega-cap-tech",
    "CRM": "mega-cap-tech",
    # Software / cloud / cybersecurity
    "PLTR": "software", "SNOW": "software", "CRWD": "software", "PANW": "software",
    "NET": "software", "OKTA": "software", "NOW": "software", "ZETA": "software",
    "DUOL": "software", "APP": "software", "SHOP": "software", "U": "software",
    "RBLX": "software", "TTD": "software", "DKNG": "software", "SPOT": "software",
    "CRWV": "software", "NBIS": "software", "CRCL": "software", "DJT": "software",
    "BBAI": "software", "AI": "software", "TEM": "software", "FIG": "software",
    "FIGR": "software", "TDOG": "software",
    # Fintech
    "AFRM": "fintech", "HOOD": "fintech", "SOFI": "fintech", "UPST": "fintech",
    "PYPL": "fintech", "NU": "fintech", "XYZ": "fintech", "ETOR": "fintech",
    "RKT": "fintech",
    # Consumer / health / other liquid
    "LLY": "healthcare", "UNH": "healthcare", "NVO": "healthcare", "MRNA": "healthcare",
    "SRPT": "healthcare", "HIMS": "healthcare", "OSCR": "healthcare",
    "LMND": "insurtech", "CNC": "healthcare", "CMG": "consumer",
    "UBER": "consumer", "GRAB": "consumer", "UPS": "industrials",
    "BABA": "china", "PONY": "china", "XPEV": "china", "NIO": "china",
    "CVNA": "consumer", "OPEN": "consumer", "BA": "industrials",
    "GME": "consumer", "SNAP": "software", "RDDT": "software",
    # Energy / nuclear / clean
    "CEG": "nuclear", "VST": "nuclear", "OKLO": "nuclear", "SMR": "nuclear",
    "LEU": "nuclear", "UEC": "nuclear", "UUUU": "nuclear", "DNN": "nuclear",
    "NNE": "nuclear", "GEV": "industrials",
    "PLUG": "clean", "QS": "clean", "LAC": "metals", "ALB": "metals",
    "MP": "metals", "NEM": "metals", "PAAS": "metals", "FCX": "metals",
    # Space / drones / quantum
    "RKLB": "space", "ASTS": "space", "LUNR": "space", "AVAV": "drones",
    "RCAT": "drones", "ONDS": "drones", "KTOS": "drones", "PL": "space",
    "RDW": "space", "JOBY": "evtol", "ACHR": "evtol",
    "QBTS": "quantum", "IONQ": "quantum", "RGTI": "quantum", "QUBT": "quantum",
    "SATS": "satellite", "FLY": "evtol", "EOSE": "clean",
    # Misc industrials / mid-cap
    "DELL": "hardware", "VRT": "hardware", "USAR": "industrials",
    "KEEL": "industrials", "BE": "clean", "B": "industrials",
    "CRML": "metals", "TSUI": "industrials", "NOK": "industrials",
    "AAL": "industrials",
    "LCID": "consumer",
    "SOUN": "software",
}

# SPY-beta map for the highest-conviction names. Everything else falls
# back to DEFAULT_SINGLE_NAME_BETA. Numbers are approximate but
# directional. (See README "factor map provenance".)
BETA_TO_SPY: dict[str, float] = {
    "SPY": 1.00, "QQQ": 1.15, "IWM": 1.15, "DIA": 0.90,
    "ARKK": 1.55, "SOXX": 1.45, "XOP": 1.10, "GDX": 0.90, "GDXJ": 1.10,
    "EWW": 0.95, "EWY": 1.05, "FXI": 0.75, "KWEB": 0.85,
    # Mega-cap
    "AAPL": 1.20, "MSFT": 1.05, "META": 1.30, "GOOGL": 1.15, "AMZN": 1.30,
    "TSLA": 1.95, "NFLX": 1.30, "ORCL": 1.05, "ADBE": 1.20, "CRM": 1.30,
    # Semis
    "NVDA": 1.70, "AMD": 1.80, "AVGO": 1.40, "ASML": 1.35, "TSM": 1.20,
    "MU": 1.60, "MRVL": 1.65, "SMCI": 2.10, "INTC": 1.10, "QCOM": 1.30,
    "ARM": 1.65, "ALAB": 1.90, "KLAC": 1.50, "LRCX": 1.55, "ANET": 1.45,
    # Crypto-equity / crypto
    "COIN": 2.40, "MSTR": 2.80, "MARA": 2.40, "RIOT": 2.30, "CLSK": 2.40,
    "HUT": 2.30, "IREN": 2.30, "WULF": 2.30, "CIFR": 2.30, "CORZ": 2.30,
    "BMNR": 2.50, "BULL": 2.40, "GLXY": 2.40, "GEMI": 2.30, "SBET": 2.30,
    "BLSH": 2.30,
    "IBIT": 1.30, "ETHA": 1.30, "XRPZ": 1.30,
    # Software / cloud
    "PLTR": 1.80, "SNOW": 1.45, "CRWD": 1.40, "PANW": 1.20, "NET": 1.50,
    "NOW": 1.20, "OKTA": 1.30, "APP": 1.80, "SHOP": 1.60, "U": 1.70,
    "RBLX": 1.60, "TTD": 1.50, "DKNG": 1.45, "SPOT": 1.30, "CRWV": 2.30,
    "NBIS": 2.10, "CRCL": 2.30, "DJT": 1.90, "BBAI": 2.00, "AI": 2.10,
    "TEM": 2.00, "PLUG": 2.00, "FIG": 2.30, "FIGR": 2.30, "TDOG": 1.80,
    # Fintech
    "AFRM": 2.10, "HOOD": 1.90, "SOFI": 1.90, "UPST": 2.50, "PYPL": 1.30,
    "NU": 1.30, "XYZ": 2.00, "ETOR": 1.80, "RKT": 1.80,
    # Healthcare / consumer
    "LLY": 0.50, "UNH": 0.65, "NVO": 0.55, "MRNA": 1.20, "SRPT": 1.30,
    "HIMS": 1.70, "OSCR": 1.70, "LMND": 1.90, "CNC": 0.70, "CMG": 1.15,
    "UBER": 1.30, "GRAB": 1.30, "UPS": 0.95, "BA": 1.50,
    "BABA": 0.85, "PONY": 1.20, "XPEV": 1.50, "NIO": 1.60,
    "CVNA": 2.50, "OPEN": 2.30, "GME": 1.80, "SNAP": 1.70, "RDDT": 1.80,
    # Energy / nuclear / metals
    "CEG": 1.05, "VST": 1.10, "OKLO": 1.90, "SMR": 1.80, "LEU": 1.70,
    "UEC": 1.80, "UUUU": 1.80, "DNN": 1.70, "NNE": 1.90, "GEV": 1.20,
    "LAC": 1.50, "ALB": 1.50, "MP": 1.60, "NEM": 0.70, "PAAS": 0.80,
    "FCX": 1.30,
    # Space / drones / quantum
    "RKLB": 2.10, "ASTS": 2.30, "LUNR": 2.10, "AVAV": 1.30, "RCAT": 2.30,
    "ONDS": 2.20, "KTOS": 1.30, "PL": 1.80, "RDW": 1.70,
    "JOBY": 2.00, "ACHR": 2.30, "QBTS": 2.50, "IONQ": 2.30, "RGTI": 2.30,
    "QUBT": 2.30, "SATS": 1.50, "FLY": 1.70, "EOSE": 2.10,
    # Hardware / industrials / other
    "DELL": 1.30, "VRT": 1.40, "COHR": 1.50, "CRDO": 2.10, "WDC": 1.40,
    "SNDK": 1.30, "GLW": 1.00, "LITE": 1.50, "APH": 1.20, "CLS": 1.45,
    "NVTS": 1.80, "USAR": 1.50, "KEEL": 1.40, "BE": 2.00, "B": 1.20,
    "CRML": 1.80, "TSUI": 1.30, "NOK": 0.80, "AAL": 1.30,
}


# Backwards-compatible alias. New code should import OVERRIDE_SECTOR_MAP.
SECTOR_MAP: dict[str, str] = OVERRIDE_SECTOR_MAP


def lookup_underlying(symbol: str) -> dict[str, Any]:
    """Return sector + beta-to-SPY for a given underlying symbol.

    Legacy helper kept for backwards compatibility. New code should
    call :func:`risk_dashboard.sector_loader.resolve_sector` and
    :func:`risk_dashboard.beta_loader.compute_betas`. This function
    only reads the override map and curated beta dict; it does NOT
    walk the vendor/heuristic tiers.
    """
    key = (symbol or "").strip().upper()
    sector = OVERRIDE_SECTOR_MAP.get(key)
    sector_source = "curated" if sector else "default"
    if not sector:
        sector = "other"
    beta = BETA_TO_SPY.get(key)
    beta_source = "curated"
    if beta is None:
        beta = DEFAULT_BROAD_INDEX_BETA if sector == "broad" else DEFAULT_SINGLE_NAME_BETA
        beta_source = "default"
    return {
        "sector": sector,
        "beta_to_spy": float(beta),
        "sector_source": sector_source,
        "beta_source": beta_source,
    }
