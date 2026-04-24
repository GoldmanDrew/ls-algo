#!/usr/bin/env python3
"""
daily_screener.py — Single-file daily pipeline for leveraged ETF pair strategy.

Consolidates:  universe building → beta calculation → borrow fetch → screening → decay/vol → CSV

Replaces the multi-notebook workflow:
  1. etf_screener.py notebook cells (pair definitions, YieldMax/Roundhill scraping)
  2. etf_analytics.py notebook cells (beta via OLS)
  3. etf_screener.py CLI (FTP borrow, screening logic)
  4. etf_analytics.py (decay + volatility enrichment)

Changes from prior version:
  - FTP retry with exponential backoff (3 attempts) + disk cache fallback
  - Expected decay column: 0.5 × |β| × |β−1| × σ² × 252
  - Blended decay scoring for generate_trade_plan.py sizing
  - Graceful core package integration (optional)

Run daily:
  python daily_screener.py                           # full pipeline, output → data/etf_screened_today.csv
  python daily_screener.py --skip-scrape             # skip YieldMax/Roundhill scraping (use cached)
  python daily_screener.py --skip-inverse            # skip inverse ETF universe (bucket C)
  python daily_screener.py --lookback 1y             # shorter history for faster runs
  python daily_screener.py --output my_output.csv    # custom output path

Screener schema v2 (uncertainty + product_class) is applied after decay enrichment
(``screener_v2_fields.enrich_screener_v2_fields``); all new columns are add-only for downstream CSV/JSON.

"""
from __future__ import annotations

import argparse
import builtins
import ftplib
import io
import random
import os
import re
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Iterable, Set

import numpy as np
import pandas as pd
import requests
import yaml
import yfinance as yf

from expense_ratios import fetch_expense_ratios
from screener_v2_fields import enrich_screener_v2_fields

warnings.filterwarnings("ignore", category=FutureWarning)

TRADING_DAYS = 252
RISK_FREE_FALLBACK = 0.045  # ~13-week T-bill fallback when ^IRX fetch fails


def _fetch_risk_free_rate(default: float = RISK_FREE_FALLBACK) -> float:
    """Live 13-week T-bill yield (annualised decimal) via yfinance ^IRX.

    ^IRX quotes yield as a percentage (e.g. 4.52 → 4.52 %). Returns *default*
    on any failure so the pipeline never blocks on a data-source hiccup.
    """
    try:
        hist = yf.Ticker("^IRX").history(period="10d", auto_adjust=False)
        if hist is not None and "Close" in hist.columns:
            close = pd.to_numeric(hist["Close"], errors="coerce").dropna()
            if len(close):
                v = float(close.iloc[-1])
                if 0.0 < v < 20.0:
                    return round(v / 100.0, 6)
    except Exception as e:
        print(f"[RF] Failed to fetch ^IRX ({e}); falling back to {default:.3%}")
    return default

# ── Optional core package integration ──
# If core/ is available, use shared norm_sym and expected_gross_decay.
# Otherwise fall back to local implementations (this file is fully self-contained).
try:
    from core.symbols import norm_sym as _core_norm_sym
    from core.portfolio import expected_gross_decay as _expected_gross_decay
    _HAS_CORE = True
except ImportError:
    _HAS_CORE = False


# ══════════════════════════════════════════════════════════════════
# SECTION 1 — PAIR UNIVERSE DEFINITIONS
# ══════════════════════════════════════════════════════════════════

leverage_pairs = [
    # --- Single Stock 2X / 1X ---
    ("AAPU", "AAPL"), ("AMUU", "AMD"),  ("AMZU", "AMZN"), ("ASMU", "ASML"),
    ("AVL",  "AVGO"), ("BOEU", "BA"),   ("BABU", "BABA"), ("BRKU", "BRK-B"),
    ("CONX", "COIN"), ("CSCL", "CSCO"), ("FRDU", "F"),    ("GGLL", "GOOGL"),
    ("HODU", "HOOD"), ("LINT", "INTC"), ("ELIL", "LLY"),  ("LMTL", "LMT"),
    ("METU", "META"), ("MRVU", "MRVL"), ("MSFU", "MSFT"), ("MUU",  "MU"),
    ("NFXL", "NFLX"), ("NVDU", "NVDA"), ("ORCU", "ORCL"), ("PALU", "PANW"),
    ("PLTU", "PLTR"), ("QCMU", "QCOM"), ("SHPU", "SHOP"), ("SOFA", "SOFI"),
    ("TSLL", "TSLA"), ("TSMX", "TSM"),  ("XOMX", "XOM"),
    # --- 3X Equity ---
    ("YINN", "FXI"),  ("MEXX", "EWW"),  ("KORU", "EWY"),  ("HIBL", "SPHB"),
    ("LABU", "XBI"),  ("SOXL", "SOXX"),
    # --- 2X Thematic / Equity --
    ("CHAU", "ASHR"), ("CWEB", "KWEB"), ("ERX",  "XLE"),  ("NUGT", "GDX"),
    ("JNUG", "GDXJ"), ("GUSH", "XOP"), ("URAA", "URA"), ("TQQQ", "QQQ"), ("SPXL", "SPY"), ("URTY", "IWM"),
]

leverage_pairs_leverageshares = [
    ("AALG", "AAL"),  ("ABNG", "ABNB"), ("ADBG", "ADBE"), ("AMDG", "AMD"),
    ("ARMG", "ARM"),  ("ASMG", "ASML"), ("AVGG", "AVGO"), ("BAIG", "BBAI"),
    ("BEG",  "BE"),   ("BIDG", "BIDU"), ("BLSG", "BLSH"), ("BMNG", "BMNR"),
    ("BOEG", "BA"),   ("BULG", "BULL"), ("CIFG", "CIFR"), ("CMGG", "CMG"),
    ("CNCG", "CNC"),  ("COIG", "COIN"), ("COTG", "COST"), ("CRCG", "CRCL"),
    ("CRMG", "CRM"),  ("CRWG", "CRWV"), ("DUOG", "DUOL"), ("FIGG", "FIG"),
    ("FUTG", "FUTU"), ("GEMG", "GEMI"), ("GEVG", "GEV"),  ("GLGG", "GLXY"),
    ("GRAG", "GRAB"), ("HOOG", "HOOD"), ("IREG", "IREN"), ("KLAG", "KLAC"),
    ("LACG", "LAC"),  ("LULG", "LULU"), ("MPG",  "MP"),   ("NBIG", "NBIS"),
    ("NEMG", "NEM"),  ("NETG", "NET"),  ("NIOG", "NIO"),  ("NUG",  "NU"),
    ("NVDG", "NVDA"), ("OKTG", "OKTA"), ("OPEG", "OPEN"), ("OSCG", "OSCR"),
    ("PANG", "PANW"), ("PBRG", "PBR"),  ("PLTG", "PLTR"), ("PYPG", "PYPL"),
    ("RTXG", "RTX"),  ("SATG", "SATS"), ("SBU",  "SBUX"), ("SNAG", "SNAP"),
    ("SPOG", "SPOT"), ("TERG", "TER"),  ("TSLG", "TSLA"), ("TSMG", "TSM"),
    ("UNHG", "UNH"),  ("UPSG", "UPS"),  ("VALG", "VALE"), ("XYZG", "XYZ"),
]

new_pairs = [
    ("MSTU", "MSTR"), ("NVDX", "NVDA"), ("TSLT", "TSLA"), ("BTCL", "IBIT"),
    ("ETU",  "ETHA"), ("CCUP", "CRCL"), ("CRWU", "CRWV"), ("AAPX", "AAPL"),
    ("GOOX", "GOOGL"),("MSFX", "MSFT"), ("NFLU", "NFLX"), ("ROBN", "HOOD"),
    ("ARMU", "ARM"),  ("DJTU", "DJT"),  ("RBLU", "RBLX"), ("SNOU", "SNOW"),
    ("SMUP", "SMR"),  ("DKUP", "DKNG"), ("BULU", "BULL"), ("GLXU", "GLXY"),
    ("AFRU", "AFRM"), ("AXUP", "AXON"), ("KTUP", "KTOS"), ("TTDU", "TTD"),
    ("BKNU", "BKNG"), ("PXIU", "UPXI"), ("BMNU", "BMNR"), ("SBTU", "SBET"),
    ("CIFU", "CIFR"), ("SOLX", "GSOL"), ("XRPK", "XRPZ"), ("XRPT", "XRPZ"),
    ("CHNU", "CHNL"), ("CRDX", "CRDD"), ("STLU", "STLR"),
    ("GMEU", "GME"),  ("APLX", "APLD"), ("APPX", "APP"),  ("ARCX", "ACHR"),
    ("ASTX", "ASTS"), ("AURU", "AUR"),  ("BEX",  "BE"),   ("BLSX", "BLSH"),
    ("CEGX", "CEG"),  ("CELT", "CELH"), ("CLSX", "CLSK"), ("COZX", "CORZ"),
    ("CRDU", "CRDO"), ("CSEX", "CLS"),  ("CWVX", "CRWV"), ("DASX", "DASH"),
    ("DOGD", "DDOG"), ("ENPX", "ENPH"), ("FLYT", "FLY"),  ("GEVX", "GEV"),
    ("IREX", "IREN"), ("JOBX", "JOBY"), ("LABX", "ALAB"), ("LRCU", "LRCX"),
    ("LYFX", "LYFT"), ("MDBX", "MDB"),  ("NEBX", "NBIS"), ("NETX", "NET"),
    ("NNEX", "NNE"),  ("NVTX", "NVTS"), ("NWMX", "NEM"),  ("OKTX", "OKTA"),
    ("OPEX", "OPEN"), ("PONX", "PONY"), ("QBTX", "QBTS"), ("QSX",  "QS"),
    ("QUBX", "QUBT"), ("RGTU", "RGTI"), ("SMU",  "SMR"),  ("SNPX", "SNPS"),
    ("SRPU", "SRPT"), ("TARK", "ARKK"), ("TEMT", "TEM"),  ("UNX",  "U"),
    ("UPSX", "UPST"), ("VOYX", "VOYG"), ("WULX", "WULF"), ("CRMU", "CRML"),
    ("UECG", "UEC"),  ("DNNG", "DNN"),  ("RCAX", "RCAT"), ("RDWU", "RDW"),
    ("OKLL", "OKLO"), ("GDXU", "GDX"),  ("MRNX", "MRNA"), ("ZETX", "ZETA"),
    ("LITX", "LITE"), ("SNXX", "SNDK"), ("WDCX", "WDC"),  ("LNOK", "NOK"),
    ("USGG", "USAR"), ("ONDG", "ONDS"), ("PLUL", "PLUG"), ("ALBG", "ALB"),
    ("UUUG", "UUUU"), ("HUTG", "HUT"),  ("XPEG", "XPEV"), ("ORLG", "ORLY"),
    ("LUNL", "LUNR"), ("RKTL", "RKT"),  ("EOSU", "EOSE"), ("BTFL", "KEEL"),
    ("FGRU", "FIGR"), ("APHU", "APH"),  ("COPZ", "COPX"), ("LEUX", "LEU"),
    ("COHX", "COHR"), ("AXPG", "AXP"),  ("FCXG", "FCX"), ("GLWG", "GLW"),
    ("SNDU", "SNDK"), ("PAAU", "PAAS"),
    ("BITX", "IBIT"), ("ETHU", "ETHA"), ("XXRP", "XRPZ"),
    # Tradr 2X long
    ("ONDL", "ONDS"), ("MSTX", "MSTR"), ("SMCX", "SMCI"), ("ORCX", "ORCL"),
    ("IONX", "IONQ"), ("HIMZ", "HIMS"), ("IRE", "IREN"), ("AVGX", "AVGO"),
    ("RKLX", "RKLB"), ("RGTX", "RGTI"), ("SOFX", "SOFI"), ("NVOX", "NVO"),
    ("LLYX", "LLY"), ("SOUX", "SOUN"), ("QPUX", "IONQ"), ("HOOX", "HOOD"),
    ("RIOX", "RIOT"), ("QSU", "QS"), ("MPL", "MP"), ("OSCX", "OSCR"),
    ("DKNX", "DKNG"), ("ANEL", "ANET"), ("VSTL", "VST"), ("CVNX", "CVNA"),
    ("LMNX", "LMND"), ("AVXX", "AVAV"), ("BU", "B"), ("PLU", "PL"),
    # 2026-03 launches (Direxion + Tradr)
    ("ADBU", "ADBE"), ("PYPU", "PYPL"), ("TXNU", "TXN"),  ("UNHU", "UNH"),
    ("AAOX", "AAOI"), ("HLXX", "HL"),   ("IBX",  "IBM"),
    # 2026-04-24 Tradr 2X long (AXT, Coupang, Monolithic Power, Seagate)
    ("AXTX", "AXT"), ("CPNX", "CPNG"), ("MPWX", "MPWR"), ("STXX", "STX"),
]

proshares_pairs_levered = [
    ("CRCA", "CRCL"), ("NVDB", "NVDA"), ("PLTA", "PLTR"), ("TSLI", "TSLA"),
    ("COIA", "COIN"), ("BITU", "IBIT"), ("ETHT", "ETHA"), ("UXRP", "XRPZ"),
    ("SOLT", "SOEZ"),
]

graniteshares_pairs_leveraged = [
    ("TSLR", "TSLA"), ("AAPB", "AAPL"), ("AMDL", "AMD"),  ("AMZZ", "AMZN"),
    ("AVGU", "AVGO"), ("BABX", "BABA"), ("BULX", "BULL"), ("CONL", "COIN"),
    ("CRWL", "CRWD"), ("DLLL", "DELL"), ("ETRL", "ETOR"), ("GOU",  "GOOGL"),
    ("INTW", "INTC"), ("IONL", "IONQ"), ("ISUL", "ISRG"), ("LCDL", "LCID"),
    ("MRAL", "MARA"), ("FBL",  "META"), ("MVLL", "MRVL"), ("MSFL", "MSFT"),
    ("MSTP", "MSTR"), ("MULL", "MU"),   ("NBIL", "NBIS"), ("NOWL", "NOW"),
    ("NVDL", "NVDA"), ("PDDL", "PDD"),  ("PTIR", "PLTR"), ("QCML", "QCOM"),
    ("RDTL", "RDDT"), ("RVNL", "RIVN"), ("SMCL", "SMCI"), ("TSMU", "TSM"),
    ("UBRL", "UBER"), ("VRTL", "VRT"),  ("AMYY", "AMD"),  ("AZYY", "AMZN"),
    ("BBYY", "BABA"), ("XBTY", "IBIT"), ("COYY", "COIN"), ("NUGY", "GDX"),
    ("HMYY", "HIMS"), ("HOYY", "HOOD"), ("IOYY", "IONQ"), ("MAAY", "MARA"),
    ("FBYY", "META"), ("MTYY", "MSTR"), ("NVYY", "NVDA"), ("PLYY", "PLTR"),
    ("QBY",  "QBTS"), ("TQQY", "QQQ"),  ("RGYY", "RGTI"), ("RTYY", "RIOT"),
    ("SEMY", "SOXX"), ("SMYY", "SMCI"), ("YSPY", "SPY"),  ("TSYY", "TSLA"),
]

leverage_pairs_capped_accel = [
    ("COIO", "COIN"), ("MSOO", "MSTR"), ("NVDO", "NVDA"),
    ("PLOO", "PLTR"), ("TSLO", "TSLA"),
]

# GraniteShares YieldBOOST (1x overlay vs underlying) — realized β lands in bucket_2 in screening.
YIELDBOOST_BUCKET2_PAIRS = [
    ("MUYY", "MU"),
    ("TMYY", "TSM"),
    ("CWY", "CRWV"),
]

covered_call_pairs = [
    ("QYLD", "QQQ"),  ("QYLG", "QQQ"),  ("QQQX", "QQQ"),  ("JEPQ", "QQQ"),
    ("XYLD", "SPY"),  ("XYLG", "SPY"),  ("JEPI", "SPY"),  ("SPYI", "SPY"),
    ("RYLD", "IWM"),
]


# ── Inverse ETF Universe (Bucket C) ──
BENCHMARK_MAP = {
    "SPX": "SPY",  "NDX": "QQQ",   "DJIA": "DIA",  "RUT": "IWM",
    "SOX": "SOXX", "FIN": "XLF",   "BIOTECH": "XBI","TECH": "XLK",
    "WTI": "USO",  "COIN": "COIN", "TSLA": "TSLA", "MSTR": "MSTR", "NVDA": "NVDA",
    "AMZN": "AMZN",
    "LITE": "LITE", "SNDK": "SNDK",
    "BTC": "IBIT", "ETH": "ETHA",  "CRCL": "CRCL", "CRWV": "CRWV",
    "GDX": "GDX",  "SLV": "SLV",   "XLE": "XLE",   "XOP": "XOP",
    "TLT": "TLT",  "MSCIJP": "EWJ","APLD": "APLD", "CLSK": "CLSK",
    "IREN": "IREN", "BE": "BE", "NBIS": "NBIS", "SMR": "SMR", "SPHB": "SPHB",
    "AMD": "AMD", "ASTS": "ASTS", "BMNR": "BMNR", "HOOD": "HOOD", "IONQ": "IONQ",
    "OKLO": "OKLO", "PLTR": "PLTR", "QBTS": "QBTS", "RGTI": "RGTI", "RKLB": "RKLB",
    "SMCI": "SMCI", "TSM": "TSM",
    # Volatility pair: UVIX (−2× daily vs SVIX) hedged to SVIX; both legs short in book.
    "SVIX": "SVIX",
}

INVERSE_ETF_UNIVERSE = [
    ("SDS",  -2, "SPX"),  ("QID",  -2, "NDX"),  ("DXD",  -2, "DJIA"), ("TWM",  -2, "RUT"),
    ("SCO",  -2, "WTI"),  ("MSTZ", -2, "MSTR"), ("NVDQ", -2, "NVDA"), ("BTCZ", -2, "BTC"),
    ("CONI", -2, "COIN"), ("MSDD", -2, "MSTR"), ("NVD",  -2, "NVDA"), ("TSDD", -2, "TSLA"),
    ("ETHD", -2, "ETH"),  ("CRCD", -2, "CRCL"), ("CORD", -2, "CRWV"), ("TSLQ", -2, "TSLA"),
    ("ZSL",  -2, "SLV"),  ("SQQQ", -3, "NDX"),  ("SPXS", -3, "SPX"),  ("TZA",  -3, "RUT"),
    ("SOXS", -3, "SOX"),  ("FAZ",  -3, "FIN"),   ("LABD", -3, "BIOTECH"),
    ("TECS", -3, "TECH"), ("SDOW", -3, "DJIA"), ("DUST", -3, "GDX"),
    ("WEBS", -3, "TECH"), ("FNGD", -3, "TECH"),
    ("REW",  -2, "TECH"), ("SKF",  -2, "FIN"),   ("SPXU", -3, "SPX"),
    ("DUG",  -2, "XLE"),  ("DRIP", -2, "XOP"),  ("TMV",  -3, "TLT"),
    ("TBT",  -2, "TLT"),  ("TBX",  -2, "TLT"),  ("NVDS", -1.5, "NVDA"),
    ("EWV",  -2, "MSCIJP"), ("APLZ", -2, "APLD"), ("AMZO", -2, "AMZN"),
    ("BEZ", -2, "BE"), ("NBIZ", -2, "NBIS"), ("SMZ", -2, "SMR"),
    ("QBTZ", -2, "QBTS"), ("IONZ", -2, "IONQ"),
    ("HIBS", -3, "SPHB"),  ("CLSZ", -2, "CLSK"), ("IREZ", -2, "IREN"),
    ("RGTZ", -2, "RGTI"), ("PLTZ", -2, "PLTR"), ("SMST", -2, "MSTR"), ("SMCZ", -2, "SMCI"),
    ("BMNZ", -2, "BMNR"), ("HOOZ", -2, "HOOD"), ("DAMD", -2, "AMD"), ("RKLZ", -2, "RKLB"),
    ("STSM", -2, "TSM"), ("OKLS", -2, "OKLO"),     ("ASTN", -2, "ASTS"),
    # 2026-04-23 Tradr 2X inverse (Lumentum, SanDisk — underlying SNDK)
    ("LITZ", -2, "LITE"), ("SNDQ", -2, "SNDK"),
    ("UVIX", -2, "SVIX"),
]


# ══════════════════════════════════════════════════════════════════
# SECTION 2 — UNIVERSE BUILDING
# ══════════════════════════════════════════════════════════════════

def _norm_sym(x) -> str:
    if _HAS_CORE:
        return _core_norm_sym(x)
    return builtins.str(x).strip().upper().replace(".", "-")


def _dedupe_by_etf(pairs: list[tuple]) -> list[tuple]:
    """Keep first occurrence of each ETF ticker."""
    seen = set()
    out = []
    for etf, und in pairs:
        if etf in seen:
            continue
        seen.add(etf)
        out.append((etf, und))
    return out


_YIELDMAX_CACHE = Path("data/scrape_cache_yieldmax.csv")
_ROUNDHILL_CACHE = Path("data/scrape_cache_roundhill.csv")


def _save_scrape_cache(df: pd.DataFrame, path: Path) -> None:
    """Persist a successful scrape result so future failures can fall back."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
    except Exception as e:
        print(f"[WARN] Could not write scrape cache {path}: {e}")


def _load_scrape_cache(path: Path, label: str) -> pd.DataFrame:
    """Load a cached scrape result on failure.  Returns empty DF if no cache."""
    if path.exists():
        try:
            df = pd.read_csv(path)
            age_h = (datetime.now()
                     - datetime.fromtimestamp(path.stat().st_mtime)).total_seconds() / 3600
            print(f"[WARN] {label}: falling back to cached scrape "
                  f"({len(df)} rows, {age_h:.0f}h old)")
            return df
        except Exception as e:
            print(f"[WARN] {label}: cache read failed: {e}")
    return pd.DataFrame(columns=["ETF", "Underlying"])


def _scrape_yieldmax() -> pd.DataFrame:
    """Scrape YieldMax single-stock option income ETFs → DataFrame(ETF, Underlying).

    On success the result is cached to disk.  On failure (timeout, HTTP
    error, etc.) the most recent cached result is returned instead of an
    empty DataFrame, so the universe doesn't silently shrink.
    """
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 Chrome/120.0.0.0"})
    url = "https://yieldmaxetfs.com/nav-group/yieldmax-single-stock-option-income-etfs/"
    try:
        resp = session.get(url, timeout=15)
        resp.raise_for_status()
    except Exception as e:
        print(f"[WARN] YieldMax scrape failed: {e}")
        return _load_scrape_cache(_YIELDMAX_CACHE, "YieldMax")

    from bs4 import BeautifulSoup
    soup = BeautifulSoup(resp.text, "html.parser")
    rows = []
    for a in soup.find_all("a", href=True):
        ticker = a.get_text(strip=True)
        if not re.fullmatch(r"[A-Z]{3,5}", ticker):
            continue
        name_node = a.find_next(string=re.compile(r"^YieldMax "))
        if not name_node:
            continue
        m = re.search(r"YieldMax\s+([A-Z\.]{2,6})\s+Option Income", name_node.strip())
        if m:
            rows.append((ticker, m.group(1)))
    df = pd.DataFrame(rows, columns=["ETF", "Underlying"]).drop_duplicates()
    if not df.empty:
        _save_scrape_cache(df, _YIELDMAX_CACHE)
    return df


def _scrape_roundhill() -> pd.DataFrame:
    """Scrape Roundhill WeeklyPay ETFs → DataFrame(ETF, Underlying).

    On success the result is cached to disk.  On failure the most
    recent cached result is returned.
    """
    BAD_TOKENS = {"NEW","ETF","ETFS","WEEKLYPAY","WEEKLY","PAY",
                  "ISSUE","TREASURY","INCOME","STRATEGY","FUND","TRUST","INC"}
    OVERRIDES = {"TSYW": "TLT", "BRKW": "BRK-B"}
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 Chrome/120.0.0.0"})
    try:
        resp = session.get("https://www.roundhillinvestments.com/weeklypay-etfs", timeout=15)
        resp.raise_for_status()
    except Exception as e:
        print(f"[WARN] Roundhill scrape failed: {e}")
        return _load_scrape_cache(_ROUNDHILL_CACHE, "Roundhill")

    from bs4 import BeautifulSoup
    soup = BeautifulSoup(resp.text, "html.parser")
    rows = []
    ticker_re = re.compile(r"^[A-Z][A-Z0-9\.\-]{1,5}$")
    for a in soup.find_all("a", href=True):
        txt = " ".join(a.get_text(" ", strip=True).split())
        if "WeeklyPay" not in txt:
            continue
        parts = txt.split()
        if len(parts) < 2:
            continue
        etf = parts[0]
        if not re.fullmatch(r"[A-Z]{3,5}W", etf):
            continue
        underlying = None
        for tok in parts[1:8]:
            t = tok.strip("®™:,;()[]")
            if ticker_re.fullmatch(t) and t not in BAD_TOKENS and t != etf:
                underlying = t
                break
        if underlying is None and etf.endswith("W"):
            underlying = etf[:-1]
        if underlying:
            rows.append((etf, OVERRIDES.get(etf, underlying)))
    df = pd.DataFrame(rows, columns=["ETF", "Underlying"]).drop_duplicates()
    if not df.empty:
        _save_scrape_cache(df, _ROUNDHILL_CACHE)
    return df


def build_full_universe(skip_scrape: bool = False, skip_inverse: bool = False) -> pd.DataFrame:
    """
    Build the complete ETF universe from all sources.

    Returns DataFrame with columns: ETF, Underlying, Leverage
    """
    # 1. Combine all leveraged pairs (dedupe by ETF)
    existing_etfs = {etf for etf, _ in leverage_pairs}
    combined = list(leverage_pairs)
    combined += [(e, u) for e, u in new_pairs if e not in existing_etfs]

    all_levered = _dedupe_by_etf(
        combined
        + leverage_pairs_leverageshares
        + leverage_pairs_capped_accel
        + proshares_pairs_levered
        + graniteshares_pairs_leveraged
    )
    dx_df = pd.DataFrame(all_levered, columns=["ETF", "Underlying"])
    dx_df["Leverage"] = 2.0
    print(f"[UNIVERSE] Leveraged pairs: {len(dx_df)}")

    # 2. Covered-call / income sleeve (1x): QYLD family + YieldBOOST bucket-2 names
    cc_df = pd.DataFrame(
        covered_call_pairs + YIELDBOOST_BUCKET2_PAIRS,
        columns=["ETF", "Underlying"],
    )
    cc_df["Leverage"] = 1.0

    # 3. Scraped income products
    income_parts = [cc_df]
    if not skip_scrape:
        ym_df = _scrape_yieldmax()
        ym_df["Leverage"] = 1.0
        print(f"[UNIVERSE] YieldMax scraped: {len(ym_df)}")
        income_parts.append(ym_df)

        rh_df = _scrape_roundhill()
        rh_df["Leverage"] = 1.0
        print(f"[UNIVERSE] Roundhill scraped: {len(rh_df)}")
        income_parts.append(rh_df)

    income_df = pd.concat(income_parts, ignore_index=True)

    # 4. Combine all
    all_df = pd.concat([dx_df, income_df], ignore_index=True)
    all_df = all_df.dropna(subset=["Underlying"])
    all_df = all_df[all_df["Underlying"].astype(str).str.strip() != ""]
    all_df = all_df.drop_duplicates(subset=["ETF"]).reset_index(drop=True)
    all_df["ETF"] = all_df["ETF"].apply(_norm_sym)
    all_df["Underlying"] = all_df["Underlying"].apply(_norm_sym)

    # Filter out known bad tickers (scraped erroneously, delisted, etc.)
    # Delisted Tradr ETFs are hardcoded here so the screener produces a
    # clean CSV on GitHub Actions (no TWS) as well as locally.
    # When running locally with TWS, validate_ibkr_contracts catches
    # future delistings automatically.
    _BAD_TICKERS = {
        "JP", "JPO",
        # Tradr ETFs — liquidated Jan–Mar 2026
        "AURU",   # AUR  — closed 2026-03-03
        "AXUP",   # AXON — closed 2026-03-16
        "BKNU",   # BKNG — closed 2026-03-16
        "BLSX",   # BLSH — closed 2026-02-19
        "CELT",   # CELH — closed 2026-01-23
        "DASX",   # DASH — closed 2026-02-19
        "DKUP",   # DKNG — closed 2026-03-16
        "ARMU",   # ARM  — stale/no-trade as of 2026-03-23
        "BULU",   # BULL — stale/no-trade as of 2026-03-23
        "LYFX",   # LYFT — closed 2026-03-03
        "NETX",   # NET  — closed 2026-03-13
        "NWMX",   # NEM  — closed 2026-01-30
        "NNEX",   # NNE  — delisted
        "OKTX",   # OKTA — closed 2026-02-04
        "PXIU",   # UPXI — closed 2026-03-16
        "QSX",    # QS   — closed 2026-01-30
        "FRDU",   # FRDU — closed 2026-03-16
    }
    bad_mask = all_df["ETF"].isin(_BAD_TICKERS) | all_df["Underlying"].isin(_BAD_TICKERS)
    if bad_mask.any():
        dropped_etfs = sorted(set(all_df.loc[bad_mask, "ETF"].astype(str)))
        print(f"[UNIVERSE] Dropped {bad_mask.sum()} rows with bad ETF tickers: {dropped_etfs}")
        all_df = all_df[~bad_mask].reset_index(drop=True)

    # 5. Inverse ETFs (bucket C)
    if not skip_inverse:
        inv_rows = []
        seen = set(all_df["ETF"].values)
        for etf, lev, group in INVERSE_ETF_UNIVERSE:
            etf_n = _norm_sym(etf)
            if etf_n in seen:
                continue
            seen.add(etf_n)
            benchmark = BENCHMARK_MAP.get(group)
            if benchmark:
                inv_rows.append({"ETF": etf_n, "Underlying": _norm_sym(benchmark),
                                 "Leverage": lev})
        if inv_rows:
            inv_df = pd.DataFrame(inv_rows)
            all_df = pd.concat([all_df, inv_df], ignore_index=True)
            print(f"[UNIVERSE] Inverse ETFs added: {len(inv_rows)}")

    all_df = all_df.drop_duplicates(subset=["ETF"]).reset_index(drop=True)
    print(f"[UNIVERSE] Total universe: {len(all_df)} ETFs")
    return all_df


# ══════════════════════════════════════════════════════════════════
# SECTION 3 — TOTAL RETURN SERIES + BETA CALCULATION
# ══════════════════════════════════════════════════════════════════

# Common split/reverse-split ratios.
#
# IMPORTANT:
# Restrict to integer split factors and their reciprocals only.
# Broad fractional grids (e.g. 3/2, 5/3, 5/8) can misclassify normal
# large ETF moves as "splits", which then distorts beta estimates.
_INTEGER_SPLIT_FACTORS = (2, 3, 4, 5, 10, 15, 20, 25, 50)
_SPLIT_RATIOS = sorted(
    set(list(_INTEGER_SPLIT_FACTORS) + [1.0 / f for f in _INTEGER_SPLIT_FACTORS]),
    reverse=True,
)

_SPLIT_TOL = 0.005         # ±0.5 % tolerance around each ratio
_JUMP_FLOOR = 0.40         # ignore daily moves < 40 %
_CONTEXT_WINDOW = 20       # days of context for local vol estimate
_ZSCORE_THRESHOLD = 4.0    # jump must be > 4σ vs local vol to be a split

# Manual split overrides aligned with the v9 backtest data treatment.
# For a 1-for-10 reverse split, pre-split history must be back-adjusted by 0.1
# to the post-split basis.
_MANUAL_SPLIT_OVERRIDES: dict[str, dict[str, float]] = {
    "SMUP": {
        "2026-01-26": 0.1,
    },
    "EOSU": {
        "2026-04-15": 25,
    },
}

# Known Yahoo symbol aliases for recently renamed products.
# Used only when the primary symbol returns no data.
_YF_SYMBOL_FALLBACKS: dict[str, str] = {
    "SMUP": "SMU",
}


def _clean_split_artifacts(prices: pd.Series) -> pd.Series:
    """Detect and correct unadjusted splits/reverse-splits in a price series.

    Approach:  walk through daily price ratios (p[t] / p[t-1]).
    If a ratio matches a known split factor (within tolerance) AND
    the jump is an extreme outlier relative to local volatility
    (z-score > 4), correct it by dividing all subsequent prices by
    the split factor.

    The z-score approach (instead of a fixed neighbor threshold) handles
    volatile penny/crypto stocks correctly: a 20:1 reverse split is
    still 10+ sigma even on a stock with 200% annualized vol.

    Returns a corrected copy of the series.
    """
    if len(prices) < 3:
        return prices.copy()

    prices = prices.copy()
    vals = prices.values.astype(float)

    # Pre-compute daily log returns for z-score calculation
    log_ratios = np.full(len(vals), np.nan)
    for i in range(1, len(vals)):
        if vals[i - 1] > 0 and np.isfinite(vals[i - 1]) and vals[i] > 0:
            log_ratios[i] = np.log(vals[i] / vals[i - 1])

    for i in range(1, len(vals) - 1):
        if vals[i - 1] == 0 or not np.isfinite(vals[i - 1]):
            continue
        ratio = vals[i] / vals[i - 1]
        daily_ret = ratio - 1.0

        # Skip small moves — not a split
        if abs(daily_ret) < _JUMP_FLOOR:
            continue

        # Check if this ratio matches a known split factor
        matched_factor = None
        for sf in _SPLIT_RATIOS:
            if abs(ratio - sf) / sf < _SPLIT_TOL:
                matched_factor = sf
                break
            # Also check inverse (reverse-split looks like 1/sf)
            inv_sf = 1.0 / sf
            if abs(ratio - inv_sf) / inv_sf < _SPLIT_TOL:
                matched_factor = inv_sf
                break

        if matched_factor is None:
            continue

        # Z-score test: is this jump an extreme outlier vs local vol?
        # Use a window of returns EXCLUDING the candidate split day.
        start = max(1, i - _CONTEXT_WINDOW)
        end = min(len(log_ratios), i + _CONTEXT_WINDOW + 1)
        context = [log_ratios[j] for j in range(start, end)
                   if j != i and np.isfinite(log_ratios[j])]

        if len(context) >= 5:
            local_std = float(np.std(context))
            if local_std > 0:
                log_jump = abs(np.log(ratio))
                zscore = log_jump / local_std
                if zscore < _ZSCORE_THRESHOLD:
                    # Jump is within normal vol range → real price move
                    continue

        # If we don't have enough context, fall back to accepting the
        # correction (better to fix a split than leave 839% vol).

        # Correct: divide everything from day i onward by the factor
        vals[i:] /= matched_factor

        # Recompute log_ratios for the corrected region so subsequent
        # iterations see clean data
        for j in range(max(1, i), min(len(vals), i + 2)):
            if vals[j - 1] > 0 and vals[j] > 0:
                log_ratios[j] = np.log(vals[j] / vals[j - 1])

    prices.iloc[:] = vals
    return prices


def _apply_manual_split_overrides(series: pd.Series, ticker: str) -> pd.Series:
    """
    Apply explicit split overrides to a total-return price series.

    Behavior mirrors the v9 backtest:
      - `factor` is a pre-split back-adjust multiplier.
      - all history strictly before split date is multiplied by factor.
      - if split date is absent, apply on nearest next trading day
        within 3 calendar days.

    Self-healing: before applying, checks the price ratio across the
    split boundary. If the pre/post prices are already close (ratio
    within 3×), Yahoo has retroactively adjusted and the override is
    skipped to avoid double-adjustment.
    """
    if series.empty:
        return series

    tkr = _norm_sym(ticker)
    overrides = _MANUAL_SPLIT_OVERRIDES.get(tkr, {})
    if not overrides:
        return series

    out = series.sort_index().copy()
    applied = []
    for ds, factor in overrides.items():
        ts = pd.Timestamp(ds)
        # Align timezone with the price index before comparisons.
        idx_tz = out.index.tz
        if idx_tz is not None and ts.tzinfo is None:
            ts = ts.tz_localize(idx_tz)
        elif idx_tz is None and ts.tzinfo is not None:
            ts = ts.tz_convert("UTC").tz_localize(None)
        apply_ts = None
        if ts in out.index:
            apply_ts = ts
        else:
            nxt = out.index[out.index >= ts]
            if len(nxt) > 0 and (nxt[0] - ts).days <= 3:
                apply_ts = nxt[0]

        if apply_ts is None:
            print(f"[TR][split-override] {tkr} {ts.date()} not applied (date missing)")
            continue

        f = float(factor)

        # Self-healing: check if Yahoo already adjusted the split.
        # Compare the last pre-split price to the first post-split price.
        # If they're already within 3× of each other, the data is clean.
        pre = out.loc[out.index < apply_ts]
        post = out.loc[out.index >= apply_ts]
        if len(pre) > 0 and len(post) > 0:
            px_before = float(pre.iloc[-1])
            px_after = float(post.iloc[0])
            if px_before > 0 and px_after > 0:
                raw_ratio = px_after / px_before
                if 1/3 < raw_ratio < 3.0:
                    print(
                        f"[TR][split-override] {tkr} {ts.date()} SKIPPED — "
                        f"Yahoo already adjusted (pre=${px_before:.2f} post=${px_after:.2f} "
                        f"ratio={raw_ratio:.2f})"
                    )
                    continue

        out.loc[out.index < apply_ts] = out.loc[out.index < apply_ts] * f
        applied.append((ts, apply_ts, f))

    for req_ts, apply_ts, f in applied:
        print(
            f"[TR][split-override] {tkr} requested {req_ts.date()} "
            f"applied {apply_ts.date()} back-adjust x{f:g}"
        )
    return out


def _get_total_return_series(ticker: str, period: str = "2y") -> pd.Series:
    """Fetch adjusted-close total-return series from Yahoo Finance v8 API.

    Uses Yahoo's adjclose (split + dividend adjusted) directly instead of
    yfinance, which returns incorrect adjusted prices in v0.2.40.
    """
    def _fetch_one(sym: str) -> pd.Series:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}"
        params = {"range": period, "interval": "1d", "events": "div,splits"}
        headers = {"User-Agent": "Mozilla/5.0"}
        last_exc: Exception | None = None

        # Retry transient Yahoo/API failures (e.g., 429, empty JSON payload)
        # before declaring the symbol unpriceable.
        for attempt in range(1, 4):
            try:
                resp = requests.get(url, params=params, headers=headers, timeout=15)
                if resp.status_code == 429:
                    raise requests.HTTPError("429 Too Many Requests")
                resp.raise_for_status()
                data = resp.json()
                result = (data.get("chart") or {}).get("result") or []
                if not result:
                    raise ValueError("empty chart result")
                timestamps = result[0]["timestamp"]
                adjclose = result[0]["indicators"]["adjclose"][0]["adjclose"]
                idx = pd.to_datetime(timestamps, unit="s", utc=True).tz_convert(
                    "America/New_York"
                ).normalize()
                s = pd.Series(adjclose, index=idx, name=ticker, dtype=float).dropna()
                if s.empty:
                    raise ValueError("series empty after dropna")
                return s
            except Exception as e:
                last_exc = e
                if attempt < 3:
                    sleep_s = (0.5 * (2 ** (attempt - 1))) + random.uniform(0.0, 0.25)
                    time.sleep(sleep_s)

        raise RuntimeError(f"yahoo v8 failed after retries: {last_exc}")

    def _fetch_one_yf(sym: str) -> pd.Series:
        # Secondary fallback path if Yahoo v8 keeps failing.
        # We prefer Adj Close when available; fall back to Close.
        h = yf.download(
            sym,
            period=period,
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,
        )
        if h is None or h.empty:
            raise ValueError("yfinance history empty")

        if isinstance(h.columns, pd.MultiIndex):
            lvl0 = set(h.columns.get_level_values(0))
            col0 = "Adj Close" if "Adj Close" in lvl0 else ("Close" if "Close" in lvl0 else None)
            if col0 is None:
                raise ValueError("yfinance missing close columns (multiindex)")
            sub = h[col0]
            # Single ticker should collapse to one series; if multiple cols,
            # pick the first non-empty column deterministically.
            if isinstance(sub, pd.DataFrame):
                if sub.shape[1] == 0:
                    raise ValueError("yfinance close dataframe empty")
                s_raw = sub.iloc[:, 0]
            else:
                s_raw = sub
        else:
            col = "Adj Close" if "Adj Close" in h.columns else ("Close" if "Close" in h.columns else None)
            if col is None:
                raise ValueError("yfinance missing close columns")
            s_raw = h[col]

        s = pd.to_numeric(s_raw, errors="coerce").dropna()
        if s.empty:
            raise ValueError("yfinance close series empty")
        idx = pd.to_datetime(s.index, utc=True).tz_convert("America/New_York").normalize()
        return pd.Series(s.values, index=idx, name=ticker, dtype=float)

    primary = _norm_sym(ticker)
    tried = [primary]
    fallback = _YF_SYMBOL_FALLBACKS.get(primary)
    if fallback and fallback != primary:
        tried.append(fallback)

    last_err: Exception | None = None
    for i, sym in enumerate(tried):
        try:
            s = _fetch_one(sym)
            if i > 0:
                print(f"[TR][fallback] {primary} -> {sym} ({len(s)} rows)")
            try:
                s = _apply_manual_split_overrides(s, ticker)
                s = _clean_split_artifacts(s)
            except Exception as e_clean:
                # Never drop a symbol solely because cleanup failed.
                print(
                    f"[TR][warn] {primary} cleanup failed after {sym} fetch "
                    f"({type(e_clean).__name__}: {e_clean}); using raw series"
                )
            return s
        except Exception as e:
            last_err = e

    # Last resort: yfinance history endpoint for symbols Yahoo v8 rejected.
    for i, sym in enumerate(tried):
        try:
            s = _fetch_one_yf(sym)
            src = f"{primary}->{sym}" if i > 0 else primary
            print(f"[TR][yf-fallback] {src} ({len(s)} rows)")
            try:
                s = _apply_manual_split_overrides(s, ticker)
                s = _clean_split_artifacts(s)
            except Exception as e_clean:
                print(
                    f"[TR][warn] {primary} cleanup failed on yf fallback "
                    f"({type(e_clean).__name__}: {e_clean}); using raw series"
                )
            return s
        except Exception as e:
            last_err = e

    if last_err is not None:
        print(f"[TR][err] {primary} fetch failed ({type(last_err).__name__}: {last_err})")
    return pd.Series(dtype=float, name=ticker)


def download_all_tr_series(tickers: list[str], period: str = "2y",
                           max_workers: int = 8) -> dict[str, pd.Series]:
    """Download total-return series for all tickers in parallel."""
    print(f"[TR] Downloading {len(tickers)} total-return series (period={period}) ...")
    t0 = time.monotonic()
    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_get_total_return_series, t, period): t for t in tickers}
        done = 0
        for future in as_completed(futures):
            ticker = futures[future]
            try:
                s = future.result()
                if not s.empty:
                    results[ticker.upper().replace(".", "-")] = s
            except Exception:
                pass
            done += 1
            if done % 50 == 0:
                print(f"  ... {done}/{len(tickers)}")
    elapsed = time.monotonic() - t0
    print(f"[TR] Got {len(results)}/{len(tickers)} tickers [{elapsed:.1f}s]")
    return results


def drop_stale_etfs(
    universe: pd.DataFrame,
    tr_map: dict[str, pd.Series],
    *,
    max_stale_days: int = 5,
    protected_etfs: Set[str] | None = None,
) -> pd.DataFrame:
    """
    Remove ETFs whose last traded date is more than *max_stale_days*
    calendar days ago, or that have no price data at all.

    This catches delisted / liquidated ETFs automatically — once an ETF
    stops trading, its price series goes stale and gets dropped on the
    next screener run.  No manual watchlist or SEC scraping needed.

    Protected ETFs (whitelist + flow) are still dropped if stale — a
    delisted ETF can't be traded regardless of protection status.

    Returns the filtered universe DataFrame.
    """
    if protected_etfs is None:
        protected_etfs = set()

    today = pd.Timestamp.now(tz="America/New_York").normalize()
    stale_cutoff = today - pd.Timedelta(days=max_stale_days)

    stale_etfs: list[tuple[str, str]] = []   # (etf, reason)

    for _, row in universe.iterrows():
        etf = _norm_sym(row["ETF"])
        ts = tr_map.get(etf)
        if ts is None or ts.empty:
            stale_etfs.append((etf, "no_price_data"))
            continue
        last_date = ts.index[-1]
        # Ensure tz-aware comparison
        if last_date.tzinfo is None:
            last_date = last_date.tz_localize("America/New_York")
        if last_date < stale_cutoff:
            stale_etfs.append((etf, f"last_trade={last_date.strftime('%Y-%m-%d')}"))

    if not stale_etfs:
        return universe

    stale_set = {etf for etf, _ in stale_etfs}
    mask = universe["ETF"].apply(_norm_sym).isin(stale_set)
    n_drop = mask.sum()

    # Print summary
    print(f"[STALE] Dropping {n_drop} ETFs with no trading activity "
          f"in the last {max_stale_days} days:")
    for etf, reason in sorted(stale_etfs)[:20]:
        prot_tag = " (protected)" if etf in protected_etfs else ""
        und = universe.loc[universe["ETF"].apply(_norm_sym) == etf, "Underlying"]
        und_str = _norm_sym(und.iloc[0]) if len(und) > 0 else "?"
        print(f"  {etf:10s} -> {und_str:10s}  {reason}{prot_tag}")
    if len(stale_etfs) > 20:
        print(f"  ... and {len(stale_etfs) - 20} more")

    return universe[~mask].reset_index(drop=True)


def validate_ibkr_contracts(
    universe: pd.DataFrame,
    *,
    host: str = "127.0.0.1",
    port: int = 7496,
    client_id: int = 90,
    timeout_per_batch: float = 10.0,
    batch_size: int = 40,
) -> pd.DataFrame:
    """
    Connect to IBKR and qualify each ETF contract.  Drop any ETF where
    IBKR returns 'No security definition' — these are untradeable
    (delisted, ticker changed, etc.) regardless of what Yahoo says.

    Falls back gracefully if IBKR is not reachable: logs a warning and
    returns the universe unchanged.
    """
    try:
        from ib_insync import IB, Stock
    except ImportError:
        print("[IBKR-CHECK] ib_insync not installed; skipping contract validation.")
        return universe

    etf_syms = sorted(universe["ETF"].apply(_norm_sym).unique())
    print(f"[IBKR-CHECK] Validating {len(etf_syms)} ETF contracts on {host}:{port} ...")

    ib = IB()
    try:
        ib.connect(host, port, clientId=client_id, timeout=15, readonly=True)
    except Exception as e:
        print(f"[IBKR-CHECK] WARNING: Could not connect to IBKR ({e}). "
              f"Skipping contract validation — stale-price filter is the only defense.")
        return universe

    try:
        invalid: list[str] = []
        for i in range(0, len(etf_syms), batch_size):
            batch = etf_syms[i : i + batch_size]
            contracts = [Stock(s.replace("-", " "), "SMART", "USD") for s in batch]
            try:
                ib.qualifyContracts(*contracts)
            except Exception:
                pass  # individual failures handled below

            for sym, c in zip(batch, contracts):
                if c.conId == 0:  # qualification failed
                    invalid.append(sym)

        if not invalid:
            print(f"[IBKR-CHECK] All {len(etf_syms)} contracts valid.")
            return universe

        invalid_set = set(invalid)
        mask = universe["ETF"].apply(_norm_sym).isin(invalid_set)
        n_drop = mask.sum()

        print(f"[IBKR-CHECK] Dropping {n_drop} ETFs with no IBKR security definition:")
        for sym in sorted(invalid)[:20]:
            und = universe.loc[universe["ETF"].apply(_norm_sym) == sym, "Underlying"]
            und_str = _norm_sym(und.iloc[0]) if len(und) > 0 else "?"
            print(f"  {sym:10s} -> {und_str:10s}  (no security definition)")
        if len(invalid) > 20:
            print(f"  ... and {len(invalid) - 20} more")

        return universe[~mask].reset_index(drop=True)

    finally:
        try:
            ib.disconnect()
        except Exception:
            pass


def compute_beta_ols(etf_tr: pd.Series, und_tr: pd.Series,
                     min_days: int = 60) -> tuple[float | None, int]:
    """OLS: r_etf = alpha + beta × r_und. Returns (beta, n_obs)."""
    etf_tr = etf_tr[~etf_tr.index.duplicated(keep='last')]
    und_tr = und_tr[~und_tr.index.duplicated(keep='last')]
    df = pd.concat(
        [etf_tr.rename("etf"), und_tr.rename("und")], axis=1, sort=True
    ).dropna()
    if len(df) < min_days + 1:
        return None, 0
    r_etf = df["etf"].pct_change().dropna()
    r_und = df["und"].pct_change().dropna()
    valid = r_etf.index.intersection(r_und.index)
    r_etf, r_und = r_etf.loc[valid], r_und.loc[valid]
    if len(r_etf) < min_days:
        return None, 0
    cov = np.cov(r_etf.values, r_und.values)
    var_und = cov[1, 1]
    if var_und < 1e-12:
        return None, 0
    return round(float(cov[0, 1] / var_und), 6), len(r_etf)


def add_betas(universe_df: pd.DataFrame, tr_map: dict[str, pd.Series],
              min_days: int = 60) -> pd.DataFrame:
    """Add Beta, Beta_n_obs, Beta_source columns from OLS regression."""
    df = universe_df.copy()
    betas, nobs, sources = [], [], []
    for _, row in df.iterrows():
        etf = _norm_sym(row["ETF"])
        und = _norm_sym(row["Underlying"]) if pd.notna(row.get("Underlying")) else None
        exp_lev = float(row.get("Leverage", 2.0))

        if not und or etf not in tr_map or und not in tr_map:
            betas.append(exp_lev)
            nobs.append(0)
            sources.append("imputed_missing_prices")
            continue

        b, n = compute_beta_ols(tr_map[etf], tr_map[und], min_days)
        if b is None or n < 2:
            betas.append(exp_lev)
            nobs.append(n)
            sources.append("imputed_no_overlap")
        elif exp_lev != 0 and abs(b) > 0.3 and (b > 0) != (exp_lev > 0):
            # OLS beta sign disagrees with expected leverage direction
            print(f"  [BETA] WARNING: {etf} OLS β={b:.4f} sign disagrees with "
                  f"expected leverage={exp_lev:.1f}; using expected. Likely bad data.")
            betas.append(exp_lev)
            nobs.append(n)
            sources.append("imputed_sign_mismatch")
        else:
            betas.append(b)
            nobs.append(n)
            sources.append("ols" if n >= 30 else "ols_short_history")

    df["Beta"] = betas
    df["Beta_n_obs"] = nobs
    df["Beta_source"] = sources

    ols_count = (df["Beta_source"] == "ols").sum()
    imp_count = df["Beta_source"].str.startswith("imputed").sum()
    print(f"[BETA] OLS: {ols_count} | Imputed: {imp_count} | Total: {len(df)}")
    return df


# ══════════════════════════════════════════════════════════════════
# SECTION 4 — IBKR FTP BORROW DATA (with retry + cache)
# ══════════════════════════════════════════════════════════════════

FTP_HOST = "ftp2.interactivebrokers.com"
FTP_USER = "shortstock"
FTP_PASS = ""
FTP_FILE = "usa.txt"

BORROW_CACHE_PATH = Path("data/borrow_cache.csv")


def _parse_ftp_text(text: str) -> pd.DataFrame:
    """Parse raw IBKR FTP short-stock text into a DataFrame."""
    lines = [ln for ln in text.splitlines() if ln.strip()]
    header_idx = next((i for i, ln in enumerate(lines) if ln.startswith("#SYM|")), None)
    if header_idx is None:
        raise ValueError("Could not find header line '#SYM|'")
    header_cols = [c.strip().lstrip("#").lower() for c in lines[header_idx].split("|")]
    data_lines = lines[header_idx + 1:]
    df = pd.read_csv(io.StringIO("\n".join(data_lines)), sep="|", header=None, engine="python")
    df = df.iloc[:, :min(len(header_cols), df.shape[1])]
    df.columns = header_cols[:df.shape[1]]
    df = df.drop(columns=[c for c in df.columns if not c or str(c).startswith("unnamed")], errors="ignore")
    return df


def fetch_ibkr_shortstock_file(
    filename: str = FTP_FILE,
    max_retries: int = 3,
    cache_path: Path = BORROW_CACHE_PATH,
) -> pd.DataFrame:
    """Download and parse IBKR short stock availability FTP file.

    Retries up to *max_retries* times with exponential backoff.
    On success, caches the parsed result to *cache_path*.
    On failure, falls back to the cached file (with a warning).
    """
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            print(f"[FTP] Connecting to {FTP_HOST} ... (attempt {attempt}/{max_retries})")
            ftp = ftplib.FTP(timeout=30)
            try:
                ftp.connect(FTP_HOST, 21)
                ftp.login(user=FTP_USER, passwd=FTP_PASS)
                ftp.set_pasv(True)
                buf = io.BytesIO()
                ftp.retrbinary(f"RETR {filename}", buf.write)
            finally:
                try: ftp.quit()
                except Exception:
                    try: ftp.close()
                    except Exception: pass

            text = buf.getvalue().decode("utf-8", errors="ignore")
            df = _parse_ftp_text(text)
            print(f"[FTP] Parsed {len(df)} rows")

            # Cache successful fetch
            try:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(cache_path, index=False)
                print(f"[FTP] Cached to {cache_path}")
            except Exception as e:
                print(f"[FTP] Warning: could not write cache: {e}")

            return df

        except Exception as e:
            last_err = e
            if attempt < max_retries:
                wait = 2 ** attempt  # 2s, 4s
                print(f"[FTP] Attempt {attempt} failed: {e}")
                print(f"[FTP] Retrying in {wait}s ...")
                time.sleep(wait)

    # All retries exhausted — try cache fallback
    print(f"[FTP] All {max_retries} attempts failed: {last_err}")
    if cache_path.exists():
        age_hours = (datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)).total_seconds() / 3600
        print(f"[FTP] ⚠ Falling back to cached borrow data ({age_hours:.1f}h old)")
        df = pd.read_csv(cache_path)
        print(f"[FTP] Loaded {len(df)} rows from cache")
        return df
    else:
        raise ConnectionError(
            f"IBKR FTP unreachable after {max_retries} attempts and no cache at {cache_path}. "
            f"Last error: {last_err}"
        )


def _parse_rate_to_decimal(x) -> float:
    if x is None: return np.nan
    s = str(x).strip()
    if s == "" or s.upper() in {"N/A","NA","NONE","NULL"}: return np.nan
    s = s.replace("%", "").strip()
    try: return float(s) / 100.0
    except Exception: return np.nan


def get_ibkr_borrow_snapshot(etf_list: Iterable[str]) -> pd.DataFrame:
    """Fetch borrow rates from IBKR FTP for a list of ETF symbols."""
    etf_list = list(dict.fromkeys([_norm_sym(x) for x in etf_list if str(x).strip()]))
    short_df = fetch_ibkr_shortstock_file(FTP_FILE)
    for req in ("sym", "feerate"):
        if req not in short_df.columns:
            raise ValueError(f"Expected '{req}' column; got: {list(short_df.columns)}")
    df = short_df.copy()
    df["sym"] = df["sym"].astype(str).str.upper().str.strip()
    df["borrow_fee_annual"] = df["feerate"].map(_parse_rate_to_decimal)
    df["available_int"] = pd.to_numeric(df.get("available", 0), errors="coerce").fillna(0)

    agg = df.groupby("sym", as_index=False).agg(
        borrow_fee_annual=("borrow_fee_annual", "max"),
        shares_available=("available_int", "max"),
    ).rename(columns={"sym": "ETF"})
    agg["borrow_current"] = agg["borrow_fee_annual"]
    agg["borrow_spiking"] = False
    agg["borrow_missing_from_ftp"] = False

    present = set(agg["ETF"].values)
    missing = [s for s in etf_list if s not in present]
    if missing:
        missing_df = pd.DataFrame({
            "ETF": missing, "borrow_fee_annual": np.nan,
            "shares_available": 0, "borrow_current": np.nan,
            "borrow_spiking": False, "borrow_missing_from_ftp": True,
        })
        agg = pd.concat([agg, missing_df], ignore_index=True)
    return agg.drop_duplicates(subset=["ETF"], keep="first").reset_index(drop=True)


def apply_sub2_borrow_floor(
    df: pd.DataFrame,
    tr_map: dict[str, pd.Series],
    floor_price_usd: float = 2.0,
) -> pd.DataFrame:
    """Scale borrow for sub-$2 names to reflect minimum charge base.

    For short borrow charged on max(price, floor), the effective annualized
    borrow rate on market value is:
        effective_rate = raw_rate * max(price, floor) / price
    so names below the floor get multiplied by (floor / price).
    """
    out = df.copy()
    out["ETF"] = out["ETF"].astype(str).map(_norm_sym)

    # Use latest available TR close for each ETF as price reference.
    etfs = out["ETF"].dropna().unique().tolist()
    last_px_map: dict[str, float] = {}
    for etf in etfs:
        s = tr_map.get(etf)
        if s is None or len(s) == 0:
            last_px_map[etf] = np.nan
            continue
        s_num = pd.to_numeric(s, errors="coerce").dropna()
        last_px_map[etf] = float(s_num.iloc[-1]) if len(s_num) else np.nan

    out["borrow_price_ref"] = out["ETF"].map(last_px_map)
    raw_borrow = pd.to_numeric(out.get("borrow_current"), errors="coerce")
    out["borrow_current_raw_annual"] = raw_borrow

    factor = pd.Series(1.0, index=out.index, dtype=float)
    px = pd.to_numeric(out["borrow_price_ref"], errors="coerce")
    m = raw_borrow.notna() & px.notna() & (px > 0) & (px < float(floor_price_usd))
    factor.loc[m] = float(floor_price_usd) / px.loc[m]
    out["borrow_floor_factor"] = factor

    out["borrow_current"] = raw_borrow * factor
    n_adj = int(m.sum())
    if n_adj > 0:
        med_mult = float(factor.loc[m].median())
        print(
            f"[BORROW] Applied sub-${floor_price_usd:.0f} borrow floor adjustment to "
            f"{n_adj} ETF(s) (median multiplier {med_mult:.2f}x)"
        )
    return out


# ══════════════════════════════════════════════════════════════════
# SECTION 5 — SCREENING LOGIC
# ══════════════════════════════════════════════════════════════════

# Fallbacks only used when strategy_config.yml is missing entirely.
# In normal operation these are always overridden by screener: section in config.
_FALLBACK = {
    "borrow_low": 0.08,
    "purgatory_margin": 0.04,
    "min_shares_available": 1000,
    "whitelist_hard_borrow_cap": 0.25,
    "min_beta_days": 20,
    "borrow_floor_price_usd": 2.0,
}


@dataclass
class ScreeningParams:
    borrow_low: float = _FALLBACK["borrow_low"]
    purgatory_margin: float = _FALLBACK["purgatory_margin"]
    min_shares_available: int = _FALLBACK["min_shares_available"]
    exclude_negative_cagr: bool = False
    protected_etfs: Set[str] | None = None
    hard_borrow_cap: float = _FALLBACK["whitelist_hard_borrow_cap"]


def screen_universe(df: pd.DataFrame, params: ScreeningParams) -> pd.DataFrame:
    """Apply borrow-based screening: purgatory, protected, diagnostics.

    Philosophy: include by default, exclude only with positive evidence.
    FTP-missing ETFs are treated as "unknown borrow" and included — the
    plan generator applies per-sleeve borrow caps that handle NaN correctly.
    """
    df = df.copy()
    df["ETF"] = df["ETF"].astype(str).map(_norm_sym)
    df["borrow_current"] = pd.to_numeric(df.get("borrow_current"), errors="coerce")
    df["shares_available"] = pd.to_numeric(df.get("shares_available"), errors="coerce").fillna(0).astype(int)
    if "borrow_spiking" not in df.columns: df["borrow_spiking"] = False
    if "borrow_missing_from_ftp" not in df.columns: df["borrow_missing_from_ftp"] = False

    protected = params.protected_etfs or set()
    df["protected"] = df["ETF"].isin(protected)

    if "cagr_port_hist" in df.columns:
        df["cagr_port_hist"] = pd.to_numeric(df["cagr_port_hist"], errors="coerce")
        df["cagr_positive"] = df["cagr_port_hist"] > 0
    else:
        df["cagr_positive"] = pd.NA

    # Borrow classification (informational — used by plan generator)
    borrow_known = df["borrow_current"].notna() & (~df["borrow_missing_from_ftp"])
    df["borrow_leq_low"] = borrow_known & (df["borrow_current"] <= params.borrow_low)
    df["borrow_gt_low"] = borrow_known & (df["borrow_current"] > params.borrow_low)

    hard_cap = float(params.hard_borrow_cap)
    df["protected_ok"] = df["protected"] & borrow_known & (df["borrow_current"] <= hard_cap)
    df["protected_bad"] = df["protected"] & borrow_known & (df["borrow_current"] > hard_cap)

    # Purgatory: borrow confirmed between soft cap and hard cap, not protected.
    # These get 0-target rows in proposed_trades (keep-open, don't size).
    normal_purg = (borrow_known
                   & (df["borrow_current"] > params.borrow_low)
                   & (df["borrow_current"] <= (params.borrow_low + params.purgatory_margin))
                   & (~df["protected"]))
    df["purgatory"] = normal_purg | df["protected_bad"]

    # Confirmed-expensive: borrow known > hard_cap, not protected.
    # Only these are excluded — everything else passes through to the
    # plan generator which applies per-sleeve borrow caps.
    confirmed_exclude = (borrow_known
                         & (df["borrow_current"] > hard_cap)
                         & (~df["protected"]))

    # We no longer emit include_for_algo. Downstream sizing relies on sleeve
    # membership rules in generate_trade_plan, with purgatory explicitly excluded.
    # Keep confirmed_exclude as a diagnostic-only signal.
    if params.exclude_negative_cagr and "cagr_port_hist" in df.columns:
        confirmed_exclude = confirmed_exclude | (df["cagr_positive"] != True)

    # Diagnostic columns (kept for backward compatibility)
    df["exclude_borrow_gt_low"] = df["borrow_gt_low"] & (~df["protected"])
    df["exclude_no_shares"] = df["shares_available"] < params.min_shares_available
    df["exclude_borrow_spike"] = df["borrow_spiking"].fillna(False)
    return df


# ══════════════════════════════════════════════════════════════════
# SECTION 6 — DECAY + VOLATILITY ENRICHMENT (with expected + blended)
# ══════════════════════════════════════════════════════════════════

_VOL_CAP_ANNUAL = 5.0   # 500 % — loose backstop for truly broken data
                        # Split artifacts are now cleaned at the source
                        # (_clean_split_artifacts), so this should rarely
                        # bind.  Kept as a last-resort safety net only.


def _annualized_vol(tr_series: pd.Series, min_days: int = 60,
                    cap: float = _VOL_CAP_ANNUAL,
                    max_days: int = TRADING_DAYS) -> float | None:
    """Annualized realized vol from the last *max_days* of a TR series.

    Capping the lookback to 1 year (252 trading days) prevents stale
    high-vol regimes from inflating expected-decay estimates for
    underlyings whose recent vol is much lower.

    NOTE: This is the *legacy* σ estimator:
        σ = std(pct_change(TR)) · √252
    i.e. centered (subtracts sample mean) and built from simple returns.
    For feeding ``expected_gross_decay`` — which is an Itô identity in
    log space — use :func:`_annualized_second_moment_log` instead, which
    returns ``√( mean(ln(TR_t/TR_{t-1})²) · 252 )``. The two disagree by
    the drift term μ² and by a log/simple wedge that is small at daily
    frequency but grows with vol. See docs/decay_methodology.md.
    """
    tr = tr_series.dropna()
    if len(tr) < min_days + 1: return None
    tr = tr.iloc[-max_days - 1:]  # keep max_days of returns (+1 for pct_change)
    ret = tr.pct_change().dropna()
    if len(ret) < min_days: return None
    vol = float(ret.std() * np.sqrt(TRADING_DAYS))
    if cap and vol > cap:
        vol = cap
    return round(vol, 6)


def _annualized_second_moment_log(
    tr_series: pd.Series,
    min_days: int = 60,
    cap: float = _VOL_CAP_ANNUAL,
    max_days: int = TRADING_DAYS,
) -> float | None:
    """Annualized σ aligned with the Itô decay identity.

    Returns ``σ = √( mean(r_t²) · 252 )`` where r_t is the DAILY LOG
    return of *tr_series*. This is the σ that makes

        0.5 · |β| · |β−1| · σ²

    equal — in expectation — to the realized daily hedge drag

        mean( β · ln(1+r_und_t) − ln(1+r_etf_t) ) · 252

    under a noise-free β× daily-rebalance tracker. The key properties
    that distinguish it from :func:`_annualized_vol`:

    1. **Log returns**, not simple returns.  ln(1+r) is what appears on
       both legs of the realized decay identity.
    2. **Uncentered second moment** ``mean(r²)`` instead of ``std(r)²``.
       The Itô correction term is quadratic in the realization r, so
       μ² must stay in — subtracting it (as .std does) systematically
       understates expected decay for high-drift names (MSTR, NVDA,
       COIN …).
    3. Same 252 annualization factor used elsewhere — keeps realized
       and expected on the same clock.

    Parameters mirror :func:`_annualized_vol` so the two are drop-in
    comparable.
    """
    tr = tr_series.dropna()
    if len(tr) < min_days + 1:
        return None
    tr = tr.iloc[-max_days - 1:]
    r = np.log(tr / tr.shift(1)).dropna()
    r = r[np.isfinite(r)]
    if len(r) < min_days:
        return None
    m2 = float((r ** 2).mean())
    if m2 <= 0:
        return None
    sigma = float(np.sqrt(m2 * TRADING_DAYS))
    if cap and sigma > cap:
        sigma = cap
    return round(sigma, 6)


def expected_gross_decay(
    beta: float,
    sigma_annual: float,
    *,
    expense_ratio: float = 0.0,
    risk_free_rate: float = 0.0,
    mgr_borrow_on_underlying: float = 0.0,
) -> float:
    """Theoretical annualized gross decay per $1 ETF short (total carry).

    Closed-form from Avellaneda–Zhang (2009) for a β× daily-reset LETF. The
    realized estimator in ``_compute_gross_decay`` captures *all four* terms
    below; prior versions of this function returned only the first, which made
    ``blended_gross_decay`` an apples-to-oranges mix. We now return the full
    expression so realized and expected live in the same units:

        expected = 0.5·|β|·|β−1|·σ²            (volatility drag)
                 + f                           (LETF expense ratio)
                 + (β − 1)·r                   (financing term)
                 + |β|·λ̄_mgr    if β<0        (manager's borrow on underlying)

    Sign convention: **positive = accrues to the short seller** (same as
    realized decay). A β=2 bull LETF at σ=30 %, r=4.5 %, f=1 % returns
    0.09 + 0.01 + 0.045 = 14.5 %/yr. A β=−2 inverse at σ=30 %, r=4.5 %, f=1 %,
    λ̄_mgr=0 returns 0.27 + 0.01 − 0.135 = 14.5 %/yr.

    Parameters
    ----------
    beta : float
        LETF leverage (signed). e.g. +2, +3, −1, −2.
    sigma_annual : float
        Annualised vol of the **underlying** (decimal, e.g. 0.30 for 30 %).
    expense_ratio : float, default 0.0
        LETF management fee (decimal). Pass 0.0 to fall back to pre-fix behaviour.
    risk_free_rate : float, default 0.0
        Annualised risk-free rate (decimal, e.g. 0.045). Typically sourced
        from ^IRX (13-week T-bill) at runtime.
    mgr_borrow_on_underlying : float, default 0.0
        For inverse LETFs (β<0): the borrow fee the fund's manager pays to
        short the underlying, as an annualised decimal. 0.0 when unknown —
        caller should set ``expected_gross_decay_reliable`` accordingly.
    """
    if _HAS_CORE:
        try:
            return _expected_gross_decay(
                beta,
                sigma_annual,
                expense_ratio=expense_ratio,
                risk_free_rate=risk_free_rate,
                mgr_borrow_on_underlying=mgr_borrow_on_underlying,
            )
        except TypeError:
            # core/ hasn't been upgraded to the extended signature — recompute
            # locally instead of silently dropping the extra terms.
            pass
    abs_b = abs(beta)
    abs_bm1 = abs(beta - 1.0)
    drag = 0.5 * abs_b * abs_bm1 * sigma_annual ** 2
    fin = (beta - 1.0) * risk_free_rate
    mgr = abs_b * mgr_borrow_on_underlying if beta < 0 else 0.0
    return round(drag + expense_ratio + fin + mgr, 6)


_WEEKS_PER_YEAR = 52
_MIN_WEEKS = 4    # ~20 trading days


def _split_suspect_week_ends(prices: pd.Series) -> set[pd.Timestamp]:
    """Identify week-ending dates impacted by split-like daily jumps."""
    s = pd.to_numeric(prices, errors="coerce").dropna()
    if len(s) < 3:
        return set()
    s = s[~s.index.duplicated(keep="last")].sort_index()

    vals = s.values.astype(float)
    idx = s.index
    suspect_weeks: set[pd.Timestamp] = set()
    for i in range(1, len(vals)):
        p0, p1 = vals[i - 1], vals[i]
        if not (np.isfinite(p0) and np.isfinite(p1) and p0 > 0 and p1 > 0):
            continue
        ratio = p1 / p0
        if ratio <= 0:
            continue
        if abs(np.log(ratio)) < _JUMP_FLOOR:
            continue
        if any(abs(ratio - sf) / sf <= _SPLIT_TOL for sf in _SPLIT_RATIOS):
            wk_end = idx[i].to_period("W-FRI").end_time.tz_localize(None).normalize()
            suspect_weeks.add(wk_end)
    return suspect_weeks


def _compute_gross_decay_weekly(
    etf_tr: pd.Series,
    und_tr: pd.Series,
    beta: float,
    min_weeks: int = _MIN_WEEKS,
    label: str | None = None,
    drop_split_suspect_weeks: bool = False,
) -> float | None:
    """
    LEGACY: gross annualized decay per $1 ETF short, WEEKLY frequency × 52.

    Kept for A/B comparison against :func:`_compute_gross_decay_daily`.
    Introduces a +3.2 % annualization wedge vs the daily × 252 estimator
    (52·5 = 260 ≠ 252) which systematically inflates realized decay
    relative to the Itô identity. Do not use for new production code —
    call :func:`_compute_gross_decay` instead (which aliases the daily
    version).

    LOG RETURNS are used for BOTH bull and inverse ETFs:

        weekly_pnl = β × ln(1+r_und_w) − ln(1+r_etf_w)
        gross_decay_annual = mean(weekly_pnl) × 52
    """
    etf_tr = etf_tr[~etf_tr.index.duplicated(keep='last')]
    und_tr = und_tr[~und_tr.index.duplicated(keep='last')]
    combined = pd.concat(
        [etf_tr.rename("etf"), und_tr.rename("und")], axis=1, sort=True
    ).dropna()
    if len(combined) < min_weeks * 5:  # need enough daily points to form weeks
        return None
    if abs(float(beta)) < 0.1:
        return None

    # Resample to weekly (last trading day of each week)
    weekly = combined.resample("W-FRI").last().dropna()
    if len(weekly) < min_weeks + 1:
        return None

    # Log returns for both bull and inverse
    r_etf = np.log(weekly["etf"] / weekly["etf"].shift(1))
    r_und = np.log(weekly["und"] / weekly["und"].shift(1))
    valid = r_etf.notna() & r_und.notna() & np.isfinite(r_etf) & np.isfinite(r_und)
    r_etf, r_und = r_etf[valid], r_und[valid]
    if len(r_etf) < min_weeks:
        return None

    weekly_pnl = float(beta) * r_und - r_etf
    if drop_split_suspect_weeks:
        suspect_weeks = _split_suspect_week_ends(etf_tr)
        if suspect_weeks:
            keep = ~weekly_pnl.index.normalize().isin(suspect_weeks)
            dropped = int((~keep).sum())
            weekly_pnl = weekly_pnl[keep]
            if dropped > 0 and label:
                print(f"[DECAY][split-guard] {label}: dropped {dropped} split-suspect week(s)")
            if len(weekly_pnl) < min_weeks:
                return None

    return round(float(weekly_pnl.mean()) * _WEEKS_PER_YEAR, 6)


_MIN_DAYS_DECAY = 40  # min aligned daily returns for realized gross decay (daily × 252)


def _compute_gross_decay_daily(
    etf_tr: pd.Series,
    und_tr: pd.Series,
    beta: float,
    min_days: int = _MIN_DAYS_DECAY,
    label: str | None = None,
    drop_split_suspect_days: bool = False,
) -> float | None:
    """Realized gross annualized decay per $1 ETF short — DAILY log form.

    Algebraic identity used:

        daily_drag_t = β · ln(1+r_und_t) − ln(1+r_etf_t)
        gross_decay_annual = mean(daily_drag_t) · 252

    For a noise-free β× daily-rebalance tracker this identity equals
    ``0.5·β·(β−1) · mean(r_und_t²) · 252`` (discrete Itô), which is
    exactly the quantity returned by
    ``expected_gross_decay(beta, _annualized_second_moment_log(und_tr))``.

    Using daily frequency × 252 — rather than weekly × 52 — eliminates
    the 260 vs 252 annualization wedge that biased the legacy weekly
    estimator upward by ~3.2 %. Using log returns on both legs (not just
    the underlying) keeps the estimator algebraically matched to the
    expected-decay formula.

    Notes
    -----
    * We intentionally do NOT resample or winsorize by default. Daily
      microstructure noise (bid-ask bounce) is mean-zero and washes out
      over the averaging window; the old weekly resample suppressed it
      at the cost of an annualization bias.
    * Split-suspect day dropping is available as an opt-in via
      ``drop_split_suspect_days`` for the targeted split-repair path.
    """
    etf_tr = etf_tr[~etf_tr.index.duplicated(keep='last')]
    und_tr = und_tr[~und_tr.index.duplicated(keep='last')]
    combined = pd.concat(
        [etf_tr.rename("etf"), und_tr.rename("und")], axis=1, sort=True
    ).dropna()
    if len(combined) < min_days + 1:
        return None
    if abs(float(beta)) < 0.1:
        return None

    r_etf = np.log(combined["etf"] / combined["etf"].shift(1))
    r_und = np.log(combined["und"] / combined["und"].shift(1))
    valid = r_etf.notna() & r_und.notna() & np.isfinite(r_etf) & np.isfinite(r_und)
    r_etf, r_und = r_etf[valid], r_und[valid]
    if len(r_etf) < min_days:
        return None

    daily_drag = float(beta) * r_und - r_etf

    if drop_split_suspect_days:
        # |ln ratio| ≥ _JUMP_FLOOR is the same gate _split_suspect_week_ends
        # uses; drop days where the UNDERLYING jumped like a split, since
        # those days dominate the squared-return term and usually indicate
        # unadjusted corporate actions rather than tradable volatility.
        keep = np.abs(r_und) < _JUMP_FLOOR
        if keep.sum() >= min_days:
            dropped = int((~keep).sum())
            daily_drag = daily_drag[keep]
            if dropped > 0 and label:
                print(
                    f"[DECAY][split-guard] {label}: dropped {dropped} "
                    f"split-suspect day(s)"
                )

    return round(float(daily_drag.mean()) * TRADING_DAYS, 6)


def _compute_gross_decay(
    etf_tr: pd.Series,
    und_tr: pd.Series,
    beta: float,
    min_weeks: int | None = None,
    label: str | None = None,
    drop_split_suspect_weeks: bool = False,
    drop_split_suspect_days: bool = False,
    min_days: int = _MIN_DAYS_DECAY,
) -> float | None:
    """Realized gross decay — daily log-drag × 252 only (no weekly fallback).

    When aligned history has fewer than ``min_days`` observations,
    returns ``None``. :func:`enrich_with_decay_and_vol` then uses
    **100 % expected** decay in ``blended_gross_decay`` (see module
    constant ``_MIN_DAYS_DECAY`` and ``Beta_n_obs`` gate there).

    Existing callers that pass ``drop_split_suspect_weeks=True`` get
    translated into ``drop_split_suspect_days=True`` on the daily path.
    """
    drop_days = drop_split_suspect_days or drop_split_suspect_weeks
    return _compute_gross_decay_daily(
        etf_tr,
        und_tr,
        beta,
        min_days=min_days,
        label=label,
        drop_split_suspect_days=drop_days,
    )


def enrich_with_decay_and_vol(
    df: pd.DataFrame,
    tr_map: dict[str, pd.Series],
    min_days: int = 60,
    realized_trust_days: int = 252,
    expense_ratios: dict[str, tuple[float | None, str]] | None = None,
    risk_free_rate: float = 0.0,
    underlying_borrow_map: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Add vol + decay columns using pre-downloaded TR series.

    Columns added / overwritten
    ---------------------------
      - expense_ratio_annual         : LETF management fee (decimal)
      - expense_ratio_source         : provenance tag from expense_ratios.py
      - risk_free_rate_used          : RF rate used for this row's expected-decay calc
      - underlying_borrow_annual     : manager's borrow on underlying (inverse ETFs only)
      - expected_gross_decay_annual  : full 4-term Avellaneda–Zhang decay
      - expected_gross_decay_reliable: False if β<0 and we had to zero out λ̄_mgr
      - blended_gross_decay          : weighted mix of realized + expected
                                       (β≤0 or β>1.5); realized-only for 0<β≤1.5.
                                       If ``Beta_n_obs`` < ``_MIN_DAYS_DECAY``
                                       (40), blend uses **100 % expected** only.
      - net_decay_annual             : realized − ETF borrow
      - decay_score                  : blended − ETF borrow
    """
    expense_ratios = expense_ratios or {}
    underlying_borrow_map = underlying_borrow_map or {}
    print("[DECAY] Computing decay + volatility ...")
    vols_etf_raw = []        # NEW Itô-aligned σ (log, uncentered)
    vols_etf_legacy = []     # legacy σ (simple, centered) kept for diagnostics
    decays = []              # daily-log realized decay (None if < _MIN_DAYS_DECAY obs)
    decays_legacy = []       # legacy weekly × 52 realized decay
    betas_out, beta_nobs_out = [], []
    ok_decay = ok_vol = ok_expected = betas_computed = 0

    # ── PASS 1: betas, raw vols, realized decay ──
    for _, row in df.iterrows():
        etf = _norm_sym(row["ETF"])
        und = _norm_sym(row["Underlying"]) if pd.notna(row.get("Underlying")) else None

        beta_f = float(row["Beta"]) if pd.notna(row.get("Beta")) else None
        n_obs_i = int(row["Beta_n_obs"]) if pd.notna(row.get("Beta_n_obs")) else 0

        # Compute beta from OLS if missing
        if beta_f is None and und and etf in tr_map and und in tr_map:
            beta_f, n_obs_i = compute_beta_ols(tr_map[etf], tr_map[und], min_days)
            if beta_f is not None:
                betas_computed += 1
        betas_out.append(beta_f)
        beta_nobs_out.append(n_obs_i)

        # Itô-aligned σ: √(mean(log-return²) · 252). Matches the measure
        # used by the realized-decay estimator (daily log returns), so the
        # Itô identity 0.5·|β|·|β−1|·σ² ≡ mean(daily_drag)·252 holds.
        vol_etf = _annualized_second_moment_log(tr_map[etf], min_days) if etf in tr_map else None
        vols_etf_raw.append(vol_etf)

        # Keep the legacy centered-simple σ in parallel for diagnostics.
        vol_etf_legacy = _annualized_vol(tr_map[etf], min_days) if etf in tr_map else None
        vols_etf_legacy.append(vol_etf_legacy)

        # Realized gross decay — daily log-return form (aliased via
        # _compute_gross_decay → _compute_gross_decay_daily). Legacy
        # weekly × 52 retained in a diagnostic column below.
        decay = None
        if beta_f and abs(beta_f) >= 0.1 and und and etf in tr_map and und in tr_map:
            decay = _compute_gross_decay(tr_map[etf], tr_map[und], beta_f, label=etf)
        decays.append(decay)
        if decay is not None: ok_decay += 1

        decay_legacy = None
        if beta_f and abs(beta_f) >= 0.1 and und and etf in tr_map and und in tr_map:
            decay_legacy = _compute_gross_decay_weekly(
                tr_map[etf], tr_map[und], beta_f, label=etf
            )
        decays_legacy.append(decay_legacy)

    df["Beta"] = betas_out
    df["Beta_n_obs"] = beta_nobs_out
    df["vol_etf_annual"] = vols_etf_raw
    df["vol_etf_annual_legacy"] = vols_etf_legacy
    df["gross_decay_annual"] = decays
    df["gross_decay_annual_legacy_weekly"] = decays_legacy

    # ── PASS 2: resolve underlying vol per ticker ──
    # For each underlying, compute raw vol from its price series, then
    # cross-check against implied vols from its ETFs (vol_etf / |β|).
    # If raw vol is corrupted (e.g. unadjusted splits), use the best
    # ETF-implied vol instead — picking the ETF with the most history
    # and highest |β| (tightest leverage relationship).
    # All ETFs sharing the same underlying get the SAME vol_und.
    _VOL_RATIO_MAX = 2.0  # allow up to 2× before overriding

    # Group rows by underlying
    und_syms = df["Underlying"].dropna().apply(_norm_sym).unique()
    resolved_vol_und = {}         # underlying → final (Itô-aligned) σ
    resolved_vol_und_legacy = {}  # underlying → legacy centered-simple σ

    for und in und_syms:
        # Raw σ of underlying, Itô-aligned: √(mean(log-return²) · 252).
        # Using the new estimator keeps expected = 0.5·|β|·|β−1|·σ² in the
        # same measure as the realized daily-drag estimator in Pass 1.
        raw_vol = _annualized_second_moment_log(tr_map[und], min_days) if und in tr_map else None

        # Collect implied vols from all ETFs on this underlying
        mask = df["Underlying"].apply(
            lambda x: _norm_sym(x) == und if pd.notna(x) else False)
        implied_candidates = []  # (implied_vol, weight)
        for idx in df.index[mask]:
            b = betas_out[idx]
            ve = vols_etf_raw[idx]
            nobs = beta_nobs_out[idx]
            if b and abs(b) >= 0.5 and ve and ve > 0 and nobs:
                implied = ve / abs(b)
                # Weight: prefer more history and higher leverage
                weight = nobs * abs(b)
                implied_candidates.append((implied, weight))

        if not implied_candidates:
            resolved_vol_und[und] = raw_vol
            continue

        # Best implied vol = weighted average, weighted by n_obs × |β|
        total_w = sum(w for _, w in implied_candidates)
        best_implied = sum(v * w for v, w in implied_candidates) / total_w

        if raw_vol is None or raw_vol <= 0:
            # No raw vol — use implied
            resolved_vol_und[und] = round(best_implied, 6)
        elif best_implied > 0 and raw_vol / best_implied > _VOL_RATIO_MAX:
            # Raw vol is suspiciously high — override with implied
            resolved_vol_und[und] = round(best_implied, 6)
            print(f"  [VOL-FIX] {und}: raw={raw_vol*100:.1f}% → implied={best_implied*100:.1f}% "
                  f"(from {len(implied_candidates)} ETF(s))")
        else:
            resolved_vol_und[und] = raw_vol

    # ── Parallel legacy-σ resolution (diagnostics only) ──
    # Mirrors the resolution logic above but fed from the centered,
    # simple-return σ estimator so downstream tools can reproduce the
    # pre-fix behaviour exactly. Runs as its own loop so that
    # underlyings for which the new loop hits `continue` (no implied
    # candidates) still get a legacy σ emitted.
    for und in und_syms:
        raw_vol_legacy = _annualized_vol(tr_map[und], min_days) if und in tr_map else None
        mask = df["Underlying"].apply(
            lambda x, u=und: _norm_sym(x) == u if pd.notna(x) else False)
        implied_candidates_legacy = []
        for idx in df.index[mask]:
            b = betas_out[idx]
            ve = vols_etf_legacy[idx]
            nobs = beta_nobs_out[idx]
            if b and abs(b) >= 0.5 and ve and ve > 0 and nobs:
                implied_candidates_legacy.append((ve / abs(b), nobs * abs(b)))
        if implied_candidates_legacy:
            tw = sum(w for _, w in implied_candidates_legacy)
            bi = sum(v * w for v, w in implied_candidates_legacy) / tw
            if raw_vol_legacy is None or raw_vol_legacy <= 0:
                resolved_vol_und_legacy[und] = round(bi, 6)
            elif bi > 0 and raw_vol_legacy / bi > _VOL_RATIO_MAX:
                resolved_vol_und_legacy[und] = round(bi, 6)
            else:
                resolved_vol_und_legacy[und] = raw_vol_legacy
        else:
            resolved_vol_und_legacy[und] = raw_vol_legacy

    # Assign resolved vol_und to each row + compute expected decay (4 terms).
    # expense_ratios / underlying_borrow_map were passed in from main(); missing
    # entries fall back to 0.0 with a reliability flag so downstream consumers
    # can choose to de-weight those rows.
    vols_und, vols_und_legacy, expected_decays = [], [], []
    expected_decays_legacy = []
    er_vals, er_sources = [], []
    mgr_borrows, reliables, rf_vals = [], [], []
    for i, row in df.iterrows():
        etf = _norm_sym(row["ETF"]) if pd.notna(row.get("ETF")) else None
        und = _norm_sym(row["Underlying"]) if pd.notna(row.get("Underlying")) else None
        vol_und = resolved_vol_und.get(und) if und else None
        vol_und_legacy = resolved_vol_und_legacy.get(und) if und else None
        vols_und.append(vol_und)
        vols_und_legacy.append(vol_und_legacy)
        if vol_und is not None: ok_vol += 1

        er_val, er_src = (None, "missing")
        if etf and etf in expense_ratios:
            er_val, er_src = expense_ratios[etf]
        er_vals.append(er_val)
        er_sources.append(er_src)

        beta_f = betas_out[i]
        mgr_borrow = 0.0
        reliable = True
        if beta_f is not None and beta_f < 0:
            # Inverse LETF — manager pays borrow on the underlying. If we have
            # the underlying's borrow from IBKR FTP, use it; otherwise flag
            # the row as not-fully-reliable and fall back to 0.0.
            if und and und in underlying_borrow_map:
                raw = underlying_borrow_map[und]
                if raw is not None and np.isfinite(raw):
                    mgr_borrow = float(raw)
                else:
                    reliable = False
            else:
                reliable = False
        mgr_borrows.append(mgr_borrow)
        reliables.append(reliable)
        rf_vals.append(risk_free_rate)

        exp_decay = None
        if beta_f and abs(beta_f) >= 0.1 and vol_und is not None and vol_und > 0:
            f_used = float(er_val) if er_val is not None else 0.0
            exp_decay = expected_gross_decay(
                beta_f,
                vol_und,
                expense_ratio=f_used,
                risk_free_rate=risk_free_rate,
                mgr_borrow_on_underlying=mgr_borrow,
            )
            ok_expected += 1
        expected_decays.append(exp_decay)

        # Legacy (simple-σ) expected decay — diagnostic only, so downstream
        # regressions can reproduce the old behaviour exactly.
        exp_decay_legacy = None
        if beta_f and abs(beta_f) >= 0.1 and vol_und_legacy is not None and vol_und_legacy > 0:
            f_used = float(er_val) if er_val is not None else 0.0
            exp_decay_legacy = expected_gross_decay(
                beta_f,
                vol_und_legacy,
                expense_ratio=f_used,
                risk_free_rate=risk_free_rate,
                mgr_borrow_on_underlying=mgr_borrow,
            )
        expected_decays_legacy.append(exp_decay_legacy)

    df["vol_underlying_annual"] = vols_und
    df["vol_underlying_annual_legacy"] = vols_und_legacy
    df["expense_ratio_annual"] = er_vals
    df["expense_ratio_source"] = er_sources
    df["underlying_borrow_annual"] = mgr_borrows
    df["risk_free_rate_used"] = rf_vals
    df["expected_gross_decay_annual"] = expected_decays
    df["expected_gross_decay_annual_legacy"] = expected_decays_legacy
    df["expected_gross_decay_reliable"] = reliables

    # Targeted split repair: only when realized vs expected is abnormally far apart.
    repaired = 0
    for i, row in df.iterrows():
        realized = row.get("gross_decay_annual")
        expected = row.get("expected_gross_decay_annual")
        beta_f = betas_out[i]
        etf = _norm_sym(row["ETF"]) if pd.notna(row.get("ETF")) else None
        und = _norm_sym(row["Underlying"]) if pd.notna(row.get("Underlying")) else None
        if (
            etf is None or und is None or etf not in tr_map or und not in tr_map
            or beta_f is None or abs(beta_f) < 0.1
            or pd.isna(realized) or pd.isna(expected)
        ):
            continue

        diff = abs(float(realized) - float(expected))
        ratio = (abs(float(realized)) + 1e-9) / (abs(float(expected)) + 1e-9)
        # "Vastly different": at least 100% annual gap and strong ratio mismatch.
        if not (diff >= 1.0 and (ratio > 2.5 or ratio < 0.4)):
            continue

        repaired_decay = _compute_gross_decay(
            tr_map[etf],
            tr_map[und],
            beta_f,
            label=etf,
            drop_split_suspect_weeks=True,
        )
        if repaired_decay is None:
            continue
        if abs(float(repaired_decay) - float(expected)) < diff:
            df.at[i, "gross_decay_annual"] = repaired_decay
            repaired += 1
            print(
                f"[DECAY][split-repair] {etf}: realized {float(realized)*100:.2f}% -> "
                f"{float(repaired_decay)*100:.2f}% (expected {float(expected)*100:.2f}%)"
            )

    # NOTE: no hard caps on decay values — the vol cap (_VOL_CAP_ANNUAL)
    # is the single control point.  Expected decay flows from σ² and is
    # legitimately >100% for inverse ETFs on volatile underlyings (e.g.
    # MSTZ/MSTR: expected ≈ 262%, realized ≈ 256%).
    if repaired:
        print(f"[DECAY] Applied targeted split repairs to {repaired} ticker(s)")


    # Blended gross decay: trust realized more as n_obs grows.
    # For 0 < β ≤ 1.5 (beta-hedgeable single-stock sleeves), use realized only —
    # do not mix in theoretical expected decay.
    # If Beta_n_obs < _MIN_DAYS_DECAY (40): insufficient history for the daily
    # realized estimator — use **100 % expected** in the blend (not weekly).
    # w_realized = min(1.0, n_obs / realized_trust_days) when n_obs >= _MIN_DAYS_DECAY.
    blended = []
    for _, row in df.iterrows():
        realized = row["gross_decay_annual"]
        expected = row["expected_gross_decay_annual"]
        n_obs = int(row["Beta_n_obs"]) if pd.notna(row.get("Beta_n_obs")) else 0
        beta_f = row.get("Beta")
        beta_f = float(beta_f) if pd.notna(beta_f) else np.nan
        realized_only = pd.notna(beta_f) and (beta_f > 0) and (beta_f <= 1.5)

        if realized_only:
            blended.append(realized if pd.notna(realized) else np.nan)
            continue

        if n_obs < _MIN_DAYS_DECAY:
            if pd.notna(expected):
                blended.append(round(float(expected), 6))
            elif pd.notna(realized):
                blended.append(realized)
            else:
                blended.append(np.nan)
            continue

        if pd.notna(realized) and pd.notna(expected):
            w_realized = min(1.0, n_obs / realized_trust_days)
            blended.append(round(w_realized * realized + (1.0 - w_realized) * expected, 6))
        elif pd.notna(realized):
            blended.append(realized)
        elif pd.notna(expected):
            blended.append(expected)
        else:
            blended.append(np.nan)
    df["blended_gross_decay"] = blended

    # Net decay (realized only, backward compat)
    borrow_net = pd.to_numeric(df.get("borrow_current"), errors="coerce").fillna(0.0)
    df["net_decay_annual"] = np.where(
        df["gross_decay_annual"].notna(),
        df["gross_decay_annual"] - borrow_net, np.nan)

    # Decay score: blended gross decay minus borrow (for generate_trade_plan sizing)
    # Missing borrow treated as 0 — same as _decay_score_weights in generate_trade_plan.
    borrow_current = pd.to_numeric(df.get("borrow_current"), errors="coerce").fillna(0.0)
    df["decay_score"] = np.where(
        df["blended_gross_decay"].notna(),
        df["blended_gross_decay"] - borrow_current, np.nan)

    # ── Vol-ratio diagnostic ──
    # For levered ETFs, vol_etf should be ≈ |β| × vol_und.  Flag
    # rows where the ratio deviates by more than 3× (data-quality issue).
    _vol_ratio_warn = []
    for idx, row in df.iterrows():
        vu = row.get("vol_underlying_annual")
        ve = row.get("vol_etf_annual")
        b  = row.get("Beta")
        if pd.notna(vu) and pd.notna(ve) and pd.notna(b) and vu > 0 and abs(b) > 0.1:
            expected_ve = abs(b) * vu
            ratio = ve / expected_ve if expected_ve > 0 else np.nan
            if pd.notna(ratio) and (ratio < 0.20 or ratio > 3.0):
                _vol_ratio_warn.append(
                    f"  {row['ETF']:8s} vol_und={vu*100:6.1f}%  vol_etf={ve*100:6.1f}%  "
                    f"β={b:+.2f}  ratio={ratio:.2f}")
    if _vol_ratio_warn:
        print(f"\n[DECAY] ⚠ Vol-ratio outliers ({len(_vol_ratio_warn)} rows — "
              f"vol_etf / (|β| × vol_und) far from 1.0):")
        for line in _vol_ratio_warn[:10]:
            print(line)
        if len(_vol_ratio_warn) > 10:
            print(f"  ... and {len(_vol_ratio_warn) - 10} more")

    print(f"\n[DECAY] Beta OLS fill-ins: {betas_computed} | Vol: {ok_vol}/{len(df)} "
          f"| Realized decay: {ok_decay}/{len(df)} | Expected decay: {ok_expected}/{len(df)}")
    return df


# ══════════════════════════════════════════════════════════════════
# SECTION 7 — MARGIN 
# ══════════════════════════════════════════════════════════════════
def estimate_margin_requirements(df: pd.DataFrame) -> pd.DataFrame:
    """Estimate IBKR margin requirements from leverage factor.

    IBKR scales margin proportionally to the ETF's leverage multiple:
      maint_long  = min(1.0, |leverage| × 0.25)
      maint_short = min(1.0, |leverage| × 0.30)
      init_long   = max(0.50, maint_long)
      init_short  = max(0.50, maint_short)

    Examples:
      1x long:  25% maint, 50% init   |  1x short:  30% maint, 50% init
      2x long:  50% maint, 50% init   |  2x short:  60% maint, 60% init
      3x long:  75% maint, 75% init   |  3x short:  90% maint, 90% init

    Hard-to-borrow or borrow-spiking symbols get 100% short margin.
    """
    lev = df["Beta"].abs().fillna(1.0)

    df["maint_pct_long"]  = (lev * 0.25).clip(upper=1.0)
    df["maint_pct_short"] = (lev * 0.30).clip(upper=1.0)
    df["init_pct_long"]   = df["maint_pct_long"].clip(lower=0.50)
    df["init_pct_short"]  = df["maint_pct_short"].clip(lower=0.50)

    # Override: hard-to-borrow / spiking → 100% short margin
    if "borrow_spiking" in df.columns:
        spike = df["borrow_spiking"].fillna(False)
        df.loc[spike, "maint_pct_short"] = 1.0
        df.loc[spike, "init_pct_short"]  = 1.0

    if "borrow_missing_from_ftp" in df.columns:
        missing = df["borrow_missing_from_ftp"].fillna(False)
        df.loc[missing, "maint_pct_short"] = 1.0
        df.loc[missing, "init_pct_short"]  = 1.0

    return df

# ══════════════════════════════════════════════════════════════════
# SECTION 8 — MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════

def main() -> int:
    ap = argparse.ArgumentParser(description="Daily ETF screener pipeline")
    ap.add_argument("--run-date", default=date.today().isoformat())
    ap.add_argument("--output", default=None, help="Output CSV path (default: data/etf_screened_today.csv)")
    ap.add_argument("--lookback", default="2y", help="Price history lookback (default: 2y)")
    ap.add_argument("--threads", type=int, default=8, help="Download threads")
    ap.add_argument("--skip-scrape", action="store_true", help="Skip YieldMax/Roundhill scraping")
    ap.add_argument("--skip-inverse", action="store_true", help="Skip inverse ETF universe")
    ap.add_argument("--skip-ftp", action="store_true", help="Skip IBKR FTP borrow fetch")
    ap.add_argument("--skip-ibkr-check", action="store_true",
                    help="Skip IBKR contract validation (requires TWS/Gateway running)")
    ap.add_argument("--min-beta-days", type=int, default=None,
                    help="Min overlapping days for OLS beta (default: from config or 20)")
    ap.add_argument("--config", default=None, help="Path to strategy_config.yml")
    args = ap.parse_args()

    run_date = args.run_date
    script_dir = Path(__file__).resolve().parent
    data_dir = script_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    output_path = Path(args.output) if args.output else data_dir / "etf_screened_today.csv"
    dated_dir = data_dir / "runs" / run_date
    dated_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"  DAILY SCREENER PIPELINE — {run_date}")
    print("=" * 70)

    # ── Load config ──
    cfg = {}
    config_path = Path(args.config) if args.config else None
    if config_path is None:
        # Search common locations
        for candidate in [
            script_dir / "strategy_config.yml",
            script_dir / "config" / "strategy_config.yml",
            Path("strategy_config.yml"),
            Path("config/strategy_config.yml"),
        ]:
            if candidate.exists():
                config_path = candidate
                break
    if config_path and config_path.exists():
        with open(config_path) as f:
            cfg = yaml.safe_load(f) or {}
        print(f"[CONFIG] Loaded: {config_path}")

    screener_cfg = cfg.get("screener", {}) or {}
    sleeves_cfg = cfg.get("portfolio", {}).get("sleeves", {}) or {}

    # Protected ETFs from config
    wl_list = sleeves_cfg.get("whitelist_stock", {}).get("universe", {}).get("etfs", []) or []
    flow_list = sleeves_cfg.get("flow_program", {}).get("universe", {}).get("shorts", []) or []
    protected = {_norm_sym(x) for x in (list(wl_list) + list(flow_list)) if str(x).strip()}

    params = ScreeningParams(
        borrow_low=float(screener_cfg.get("borrow_low", _FALLBACK["borrow_low"])),
        purgatory_margin=float(screener_cfg.get("purgatory_margin", _FALLBACK["purgatory_margin"])),
        min_shares_available=int(screener_cfg.get("min_shares_available", _FALLBACK["min_shares_available"])),
        exclude_negative_cagr=bool(screener_cfg.get("exclude_negative_cagr", False)),
        protected_etfs=protected,
        hard_borrow_cap=float(screener_cfg.get("hard_borrow_cap",
                              screener_cfg.get("whitelist_hard_borrow_cap", _FALLBACK["whitelist_hard_borrow_cap"]))),
    )
    borrow_floor_price_usd = float(
        screener_cfg.get("borrow_floor_price_usd", _FALLBACK["borrow_floor_price_usd"])
    )
    print(f"[CONFIG] borrow_low={params.borrow_low:.2%}  hard_cap={params.hard_borrow_cap:.2%}  "
          f"protected={len(protected)}")
    print(f"[CONFIG] borrow_floor_price_usd=${borrow_floor_price_usd:.2f}")

    # Resolve min_beta_days: CLI > config > fallback
    min_beta_days = (args.min_beta_days
                     or int(screener_cfg.get("min_beta_days", _FALLBACK["min_beta_days"])))
    print(f"[CONFIG] min_beta_days={min_beta_days}")

    # ── Step 1: Build universe ──
    print("\n" + "─" * 70)
    print("STEP 1 — Build Universe")
    print("─" * 70)
    universe = build_full_universe(skip_scrape=args.skip_scrape, skip_inverse=args.skip_inverse)

    # ── Step 2: Download prices + compute betas ──
    print("\n" + "─" * 70)
    print("STEP 2 — Download Prices + Compute Betas")
    print("─" * 70)
    etf_syms = universe["ETF"].apply(_norm_sym).tolist()
    und_syms = universe["Underlying"].dropna().apply(_norm_sym).tolist()
    all_tickers = sorted(set(etf_syms + und_syms))

    tr_map = download_all_tr_series(all_tickers, period=args.lookback, max_workers=args.threads)

    # Drop ETFs that have stopped trading (delisted, liquidated, etc.)
    # before computing betas — a dead ETF has no valid price series
    # and would otherwise get an imputed beta and stay in the CSV.
    stale_days = int(screener_cfg.get("max_stale_days", 5))
    pre_count = len(universe)
    universe = drop_stale_etfs(
        universe, tr_map,
        max_stale_days=stale_days,
        protected_etfs=protected,
    )
    if len(universe) < pre_count:
        # Re-derive ticker lists after dropping stale ETFs
        etf_syms = universe["ETF"].apply(_norm_sym).tolist()
        und_syms = universe["Underlying"].dropna().apply(_norm_sym).tolist()
        all_tickers = sorted(set(etf_syms + und_syms))

    # IBKR contract validation — catch delisted ETFs that Yahoo still
    # serves stale prices for.  Requires TWS/Gateway running.
    if not args.skip_ibkr_check:
        ibkr_cfg = cfg.get("ibkr", {})
        pre_ibkr = len(universe)
        universe = validate_ibkr_contracts(
            universe,
            host=str(ibkr_cfg.get("host", "127.0.0.1")),
            port=int(ibkr_cfg.get("port", 7496)),
            client_id=90,  # dedicated screener client, won't conflict with rebalancer
        )
        if len(universe) < pre_ibkr:
            etf_syms = universe["ETF"].apply(_norm_sym).tolist()
            und_syms = universe["Underlying"].dropna().apply(_norm_sym).tolist()
            all_tickers = sorted(set(etf_syms + und_syms))
    else:
        print("[IBKR-CHECK] Skipped (--skip-ibkr-check)")

    universe = add_betas(universe, tr_map, min_days=min_beta_days)

    # Save intermediate beta file
    beta_csv = dated_dir / "all_pairs_with_betas.csv"
    universe.to_csv(beta_csv, index=False)
    print(f"[BETA] Saved: {beta_csv}")

    # ── Step 3: Fetch borrow data ──
    # Pull borrow for BOTH ETFs and their underlyings in a single FTP call —
    # the underlying borrow feeds the |β|·λ̄_mgr term in expected_gross_decay
    # for inverse LETFs (see Avellaneda–Zhang 2009).
    print("\n" + "─" * 70)
    print("STEP 3 — Fetch IBKR Borrow Data")
    print("─" * 70)
    underlying_syms_all = sorted({
        _norm_sym(u) for u in universe["Underlying"].dropna().unique()
        if str(u).strip()
    })
    if args.skip_ftp:
        print("[FTP] Skipped (--skip-ftp)")
        borrow_df = pd.DataFrame({"ETF": universe["ETF"].unique()})
        borrow_df["borrow_fee_annual"] = np.nan
        borrow_df["borrow_current"] = np.nan
        borrow_df["shares_available"] = 0
        borrow_df["borrow_missing_from_ftp"] = True
        borrow_df["borrow_spiking"] = False
        underlying_borrow_map: dict[str, float] = {}
    else:
        # Ensure protected ETFs are in universe
        missing_prot = sorted([t for t in protected if t not in set(universe["ETF"].values)])
        if missing_prot:
            add_rows = pd.DataFrame({"ETF": missing_prot, "Underlying": None, "Leverage": np.nan})
            universe = pd.concat([universe, add_rows], ignore_index=True)
            universe = universe.drop_duplicates(subset=["ETF"]).reset_index(drop=True)
            print(f"[FTP] Added {len(missing_prot)} protected ETFs to universe")

        try:
            combined_syms = sorted(
                set(universe["ETF"].unique().tolist()) | set(underlying_syms_all)
            )
            combined_borrow = get_ibkr_borrow_snapshot(combined_syms)
            etf_set = set(universe["ETF"].unique().tolist())
            borrow_df = combined_borrow[combined_borrow["ETF"].isin(etf_set)].reset_index(drop=True)
            # Build underlying borrow map from the same FTP pull.
            under_borrow = combined_borrow[~combined_borrow["ETF"].isin(etf_set)]
            underlying_borrow_map = {
                r["ETF"]: float(r["borrow_current"])
                for _, r in under_borrow.iterrows()
                if pd.notna(r.get("borrow_current"))
            }
            hit = sum(1 for v in underlying_borrow_map.values() if v is not None)
            print(f"[FTP] Underlying borrow rates collected for "
                  f"{hit}/{len(underlying_syms_all)} underlyings")
        except (ConnectionError, OSError) as e:
            print(f"[FTP] ⚠ Borrow fetch failed: {e}")
            print("[FTP] Continuing with empty borrow data (all positions will show borrow_current=NaN)")
            borrow_df = pd.DataFrame({"ETF": universe["ETF"].unique()})
            borrow_df["borrow_fee_annual"] = np.nan
            borrow_df["borrow_current"] = np.nan
            borrow_df["shares_available"] = 0
            borrow_df["borrow_missing_from_ftp"] = True
            borrow_df["borrow_spiking"] = False
            underlying_borrow_map = {}

    metrics = universe.merge(borrow_df, on="ETF", how="left")
    metrics = apply_sub2_borrow_floor(
        metrics,
        tr_map=tr_map,
        floor_price_usd=borrow_floor_price_usd,
    )

    # ── Step 4: Screen ──
    print("\n" + "─" * 70)
    print("STEP 4 — Apply Screening Logic")
    print("─" * 70)
    screened = screen_universe(metrics, params)
    purg = int(screened["purgatory"].sum())
    print(f"[SCREEN] Purgatory: {purg} | Total: {len(screened)}")

    # ── Step 5: Decay + vol (UPDATED: now also computes expected + blended) ──
    print("\n" + "─" * 70)
    print("STEP 5 — Compute Decay + Volatility")
    print("─" * 70)
    # Read realized_trust_days from config if available
    weighting_cfg = (sleeves_cfg.get("core_leveraged", {})
                     .get("weighting", {}))
    realized_trust_days = int(weighting_cfg.get("realized_trust_days", 252))

    # Expense ratios: issuer scrape → yfinance → manual override.
    # Risk-free: live 13-week T-bill yield (^IRX) with fallback.
    # Both are needed so expected_gross_decay matches what realized measures.
    etf_tickers_for_er = sorted({
        _norm_sym(t) for t in screened["ETF"].dropna().unique() if str(t).strip()
    })
    expense_map = fetch_expense_ratios(etf_tickers_for_er)
    risk_free_rate = _fetch_risk_free_rate()
    print(f"[RF]  Risk-free rate (^IRX 13w) = {risk_free_rate:.3%}")

    screened = enrich_with_decay_and_vol(
        screened,
        tr_map,
        min_days=min_beta_days,
        realized_trust_days=realized_trust_days,
        expense_ratios=expense_map,
        risk_free_rate=risk_free_rate,
        underlying_borrow_map=underlying_borrow_map,
    )

    # Step 5b — Schema v2 (uncertainty bands, product_class; add-only columns)
    screened = enrich_screener_v2_fields(
        screened, tr_map, min_days=min_beta_days
    )

    # ------------------------------------------------------------------
    # STEP 6 — Estimate Margin Requirements
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 6 — Estimate Margin Requirements")
    print("=" * 70)
    screened = estimate_margin_requirements(screened)
    n_margin = len(screened)
    med_ml = screened["maint_pct_long"].median()
    med_ms = screened["maint_pct_short"].median()
    print(f"[MARGIN] Estimated margin for {n_margin} rows from leverage factor")
    print(f"[MARGIN] Maint long:  median={med_ml:.0%}  |  Maint short: median={med_ms:.0%}")

    # Bucket labels and Bucket 4 eligibility diagnostics.
    screened["bucket"] = np.select(
        [
            pd.to_numeric(screened.get("Beta"), errors="coerce").lt(0.0),
            pd.to_numeric(screened.get("Beta"), errors="coerce").gt(1.5),
            pd.to_numeric(screened.get("Beta"), errors="coerce").gt(0.0),
        ],
        ["bucket_4", "bucket_1", "bucket_2"],
        default="",
    )
    screened["bucket4_net_edge_annual"] = np.where(
        screened["bucket"].eq("bucket_4"),
        pd.to_numeric(screened.get("net_decay_annual"), errors="coerce"),
        np.nan,
    )
    shares_avail = pd.to_numeric(screened.get("shares_available"), errors="coerce")
    borrow_cur = pd.to_numeric(screened.get("borrow_current"), errors="coerce")
    borrow_missing = screened.get(
        "borrow_missing_from_ftp",
        pd.Series(False, index=screened.index, dtype=bool),
    ).fillna(False).astype(bool)
    screened["inverse_shortable"] = (
        screened["bucket"].eq("bucket_4")
        & ((shares_avail > 0) | shares_avail.isna() | borrow_missing)
        & (borrow_cur.isna() | (borrow_cur >= 0.0))
    )

    # ── Drop helper columns, save ──
    drop_cols = ["Leverage", "ExpectedLeverage"]
    screened = screened.drop(columns=[c for c in drop_cols if c in screened.columns])

    screened.to_csv(dated_dir / "etf_screened_today.csv", index=False)
    screened.to_csv(output_path, index=False)
    print(f"\n{'=' * 70}")
    print(f"[DONE] Saved: {output_path}")
    print(f"[DONE] Dated: {dated_dir / 'etf_screened_today.csv'}")
    active = int((screened["purgatory"] != True).sum())  # noqa: E712
    print(f"[DONE] {len(screened)} pairs | {active} active (non-purgatory) | {purg} purgatory")

    # Summary stats
    valid_net = screened["net_decay_annual"].dropna()
    if len(valid_net) > 0:
        print(f"[DONE] Net decay: median={valid_net.median()*100:.2f}%  "
              f"range=[{valid_net.min()*100:.2f}%, {valid_net.max()*100:.2f}%]")

    valid_score = screened["decay_score"].dropna()
    if len(valid_score) > 0:
        print(f"[DONE] Decay score: median={valid_score.median()*100:.2f}%  "
              f"range=[{valid_score.min()*100:.2f}%, {valid_score.max()*100:.2f}%]")

    top5 = screened[screened["decay_score"].notna()].nlargest(5, "decay_score")
    if len(top5) > 0:
        print(f"\n  Top 5 by decay score (blended gross − borrow):")
        for _, r in top5.iterrows():
            bn = r.get("borrow_current")
            exp = r.get("expected_gross_decay_annual")
            print(f"    {r['ETF']:8s}  score={r['decay_score']*100:6.2f}%  "
                  f"realized={r['gross_decay_annual']*100:6.2f}%  "
                  f"expected={exp*100 if pd.notna(exp) else 0:6.2f}%  "
                  f"borrow={bn*100 if pd.notna(bn) else 0:5.2f}%  β={r['Beta']:.2f}")

    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())