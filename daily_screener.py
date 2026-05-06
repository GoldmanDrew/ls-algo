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
  # borrow_history.json is auto-discovered when present (see discover_default_borrow_history_json).

Screener schema v2 (uncertainty + product_class) is applied after decay enrichment
(``screener_v2_fields.enrich_screener_v2_fields``); all new columns are add-only for downstream CSV/JSON.

"""
from __future__ import annotations

import argparse
import builtins
import ftplib
import json
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

from beta_estimator import (
    BetaPrior,
    BetaResult,
    build_yieldboost_family_priors,
    compute_beta_for_hedging,
)
from decay_distribution import enrich_with_decay_distribution
from expense_ratios import fetch_expense_ratios
from screener_v2_fields import enrich_screener_v2_fields, load_borrow_history_json
from splits import (
    SplitEvent,
    SYMBOL_ALIASES,
    apply_split_events,
    clean_split_artifacts,
    detect_heuristic_splits,
    load_flex_splits_csv,
    load_legacy_manual_overrides,
    load_splits_overrides_csv,
    merge_split_events,
    parse_yahoo_split_events,
)
from splits import (  # legacy-name re-exports for in-file usage
    _INTEGER_SPLIT_FACTORS,
    _SPLIT_RATIOS,
    _SPLIT_TOL,
    _JUMP_FLOOR,
    _CONTEXT_WINDOW,
    _ZSCORE_THRESHOLD,
)

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
    ("LUNL", "LUNR"), ("RKTL", "RKT"),  ("EOSU", "EOSE"), ("KEEX", "KEEL"),
    ("FGRU", "FIGR"), ("APHU", "APH"),  ("COPZ", "COPX"), ("LEUX", "LEU"),
    ("COHX", "COHR"), ("AXPG", "AXP"),  ("FCXG", "FCX"), ("GLWG", "GLW"),
    ("SNDU", "SNDK"), ("PAAU", "PAAS"),
    ("BITX", "IBIT"), ("ETHU", "ETHA"), ("XXRP", "XRPZ"),
    ("TXXD", "TDOG"), ("TXXS", "TSUI"),  # 21shares 2X vs spot crypto ETFs (TDOG, TSUI)
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
    # 2026-04-24 Tradr 2X long (AXTI, Coupang, Monolithic Power, Seagate)
    ("AXTX", "AXTI"), ("CPNX", "CPNG"), ("MPWX", "MPWR"), ("STXX", "STX"),
    ("STXL", "STX"),  # Defiance 2X Long STX (vs Tradr STXX)
    ("AMA", "AMAT"),  # Defiance 2X Long AMAT
    # 2026-05-05 Defiance + T-REX 2X long (AMKR, AXTI)
    ("AMKL", "AMKR"),  # Defiance Daily Target 2X Long AMKR
    ("AXTU", "AXTI"),  # T-REX 2X Long AXTI (REX / Tuttle; distinct from Tradr AXTX)
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

# GraniteShares YieldBOOST (1x put-spread income overlay vs underlying) —
# realized β lands in bucket_2 in screening, but these are not 2x LETFs.
YIELDBOOST_BUCKET2_PAIRS = [
    ("AMYY", "AMD"), ("AZYY", "AMZN"), ("BBYY", "BABA"), ("COYY", "COIN"),
    ("CRY", "CRCL"), ("CWY", "CRWV"), ("HMYY", "HIMS"), ("HOYY", "HOOD"), ("IOYY", "IONQ"),
    ("MAAY", "MARA"), ("FBYY", "META"), ("MTYY", "MSTR"), ("MUYY", "MU"),
    ("NUGY", "GDX"), ("NVYY", "NVDA"), ("PLYY", "PLTR"), ("QBY", "QBTS"),
    ("RGYY", "RGTI"), ("RTYY", "RIOT"), ("SEMY", "SOXX"), ("SMYY", "SMCI"),
    ("TMYY", "TSM"), ("TQQY", "QQQ"), ("TSYY", "TSLA"), ("XBTY", "IBIT"), ("XEY", "ETHA"),
    ("YSPY", "SPY"),
]

# VIX futures ETPs are not clean LETFs on a primitive spot underlying.  The
# simple Itô LETF expected-decay identity misses roll yield, vol risk premium,
# jumps, and mean reversion, so keep a separate model class for them.
VOLATILITY_ETP_SYMBOLS = {
    "UVIX", "SVIX", "UVXY", "SVXY", "VXX", "VIXY", "VIXM",
    "VIX", "VIX1D", "VIX3M",
}

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
    ("TECS", -3, "TECH"), ("SDOW", -3, "DJIA"),
    # DUST: -2× daily inverse gold miners (NYSE Arca Gold Miners Index); GDX hedge proxy.
    ("DUST", -2, "GDX"),
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


def load_strategy_blacklist(cfg: dict, *, base_dir: Path | None = None) -> set[str]:
    """Load strategy blacklist entries from YAML and optional paths.blacklist_txt."""
    out: set[str] = set()
    for sym in (cfg.get("strategy", {}) or {}).get("blacklist", []) or []:
        s = _norm_sym(sym)
        if s:
            out.add(s)
    rel_txt = str((cfg.get("paths", {}) or {}).get("blacklist_txt", "") or "").strip()
    if rel_txt and base_dir is not None:
        p = Path(base_dir) / rel_txt
        if p.is_file():
            for line in p.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                s = _norm_sym(line.split()[0])
                if s:
                    out.add(s)
    return out


def apply_strategy_blacklist_to_universe(universe: pd.DataFrame, blacklist: set[str]) -> pd.DataFrame:
    """Drop rows whose ETF or underlying is on the strategy blacklist.

    Used by trading helpers and tests. The screener CSV keeps all pairs and sets
    ``strategy_blacklisted`` so dashboards can still display them.
    """
    if not blacklist or universe.empty:
        return universe
    out = universe.copy()
    etf_n = out["ETF"].astype(str).map(_norm_sym)
    und_n = out["Underlying"].map(lambda x: _norm_sym(x) if pd.notna(x) and str(x).strip() else "")
    blocked = etf_n.isin(blacklist) | und_n.isin(blacklist)
    return out.loc[~blocked].copy()


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
    yieldboost_etfs = {_norm_sym(etf) for etf, _ in YIELDBOOST_BUCKET2_PAIRS}
    all_levered = [
        (etf, und) for etf, und in all_levered
        if _norm_sym(etf) not in yieldboost_etfs
    ]
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
    yieldboost_pairs = {
        (_norm_sym(etf), _norm_sym(und)) for etf, und in YIELDBOOST_BUCKET2_PAIRS
    }
    is_yieldboost = all_df.apply(
        lambda r: (r["ETF"], r["Underlying"]) in yieldboost_pairs,
        axis=1,
    )
    all_df["is_yieldboost"] = is_yieldboost
    all_df["scenario_style"] = np.where(is_yieldboost, "income_style", "")

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
            # Inverse ETFs are by definition not YieldBOOST income strategies;
            # default the columns set above for the leveraged + income paths
            # so the downstream merged frame never carries NaN booleans (which
            # silently coerce to True via ``bool(np.nan)`` and mis-route rows
            # to the YB put-spread Monte Carlo).
            inv_df["is_yieldboost"] = False
            inv_df["scenario_style"] = ""
            all_df = pd.concat([all_df, inv_df], ignore_index=True)
            print(f"[UNIVERSE] Inverse ETFs added: {len(inv_rows)}")

    all_df = all_df.drop_duplicates(subset=["ETF"]).reset_index(drop=True)
    if "is_yieldboost" in all_df.columns:
        all_df["is_yieldboost"] = all_df["is_yieldboost"].fillna(False).astype(bool)
    print(f"[UNIVERSE] Total universe: {len(all_df)} ETFs")
    return all_df


# ══════════════════════════════════════════════════════════════════
# SECTION 3 — TOTAL RETURN SERIES + BETA CALCULATION
# ══════════════════════════════════════════════════════════════════

# NOTE: split/reverse-split detection is now centralised in ``splits.py``
# (multi-source: flex / yahoo_events / overrides_csv / manual / heuristic).
# The legacy ``_clean_split_artifacts`` and ``_apply_manual_split_overrides``
# names are preserved as thin back-compat shims so external imports keep
# working. New code should call ``splits.clean_split_artifacts`` directly.

# Yahoo symbol aliases for recently renamed products. Mirrors splits.SYMBOL_ALIASES.
_YF_SYMBOL_FALLBACKS: dict[str, str] = dict(SYMBOL_ALIASES)

# Module-level state populated by ``main()``: the resolved on-disk paths for
# operator-managed split overrides and Flex-derived splits. Kept on the module
# so the per-ticker fetch helpers don't need extra plumbing through the
# ThreadPoolExecutor. Both default to ``None`` (off → behaviour unchanged).
_SPLITS_OVERRIDES_CSV_PATH: Path | None = None
_FLEX_SPLITS_CSV_PATH: Path | None = None

# Per-fetch audit log of applied SplitEvent objects, populated by
# ``_get_total_return_series``. Used by the ``--audit-splits`` CLI to render
# what fired without re-running detection.
_SPLITS_APPLIED_LOG: dict[str, list[SplitEvent]] = {}


def _clean_split_artifacts(prices: pd.Series, ticker: str | None = None) -> pd.Series:
    """Back-compat shim — delegates to ``splits.clean_split_artifacts``.

    The new multi-source pipeline includes Yahoo ``events.splits`` (when
    available via ``_get_total_return_series``), the operator-managed CSV,
    legacy in-code overrides, and the heuristic detector — with the
    off-by-one bug fixed so the most-recent bar is included.
    """
    return clean_split_artifacts(
        prices,
        ticker=ticker,
        splits_overrides_csv=_SPLITS_OVERRIDES_CSV_PATH,
        flex_splits_csv=_FLEX_SPLITS_CSV_PATH,
    )


def _apply_manual_split_overrides(series: pd.Series, ticker: str) -> pd.Series:
    """Back-compat shim — manual overrides are merged into the unified
    pipeline via ``splits.load_legacy_manual_overrides`` and the operator
    CSV. Calling this is idempotent now (self-healing skips re-application).
    """
    return clean_split_artifacts(
        series,
        ticker=ticker,
        splits_overrides_csv=_SPLITS_OVERRIDES_CSV_PATH,
        flex_splits_csv=_FLEX_SPLITS_CSV_PATH,
        use_heuristic=False,
    )


def _get_total_return_series(ticker: str, period: str = "2y") -> pd.Series:
    """Fetch adjusted-close total-return series from Yahoo Finance v8 API.

    Uses Yahoo's adjclose (split + dividend adjusted) directly instead of
    yfinance, which returns incorrect adjusted prices in v0.2.40. Also
    captures Yahoo's ``events.splits`` payload — when present, those split
    timestamps feed the multi-source split pipeline in ``splits.py`` so
    same-day reverse splits (where Yahoo lags on retro-adjusting history)
    are still corrected.
    """
    def _fetch_one(sym: str) -> tuple[pd.Series, dict]:
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
                events = ((result[0].get("events") or {}).get("splits") or {})
                return s, dict(events)
            except Exception as e:
                last_exc = e
                if attempt < 3:
                    sleep_s = (0.5 * (2 ** (attempt - 1))) + random.uniform(0.0, 0.25)
                    time.sleep(sleep_s)

        raise RuntimeError(f"yahoo v8 failed after retries: {last_exc}")

    def _fetch_one_yf(sym: str) -> tuple[pd.Series, dict]:
        # Secondary fallback path if Yahoo v8 keeps failing.
        # We prefer Adj Close when available; fall back to Close.
        # ``yf.download`` does not surface raw split-event timestamps the way
        # the v8 chart API does, so the events dict comes back empty here —
        # the heuristic detector in ``splits.py`` then carries the load.
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
        return pd.Series(s.values, index=idx, name=ticker, dtype=float), {}

    primary = _norm_sym(ticker)
    tried = [primary]
    fallback = _YF_SYMBOL_FALLBACKS.get(primary)
    if fallback and fallback != primary:
        tried.append(fallback)

    def _post_process(s: pd.Series, yahoo_events: dict) -> pd.Series:
        """Apply the unified multi-source split-cleanup pipeline."""
        try:
            cleaned, applied = clean_split_artifacts(
                s,
                ticker=ticker,
                yahoo_events=yahoo_events or None,
                splits_overrides_csv=_SPLITS_OVERRIDES_CSV_PATH,
                flex_splits_csv=_FLEX_SPLITS_CSV_PATH,
                return_log=True,
            )
            if applied:
                _SPLITS_APPLIED_LOG[primary] = list(applied)
            return cleaned
        except Exception as e_clean:
            # Never drop a symbol solely because cleanup failed.
            print(
                f"[TR][warn] {primary} cleanup failed "
                f"({type(e_clean).__name__}: {e_clean}); using raw series"
            )
            return s

    last_err: Exception | None = None
    for i, sym in enumerate(tried):
        try:
            s, yahoo_events = _fetch_one(sym)
            if i > 0:
                print(f"[TR][fallback] {primary} -> {sym} ({len(s)} rows)")
            return _post_process(s, yahoo_events)
        except Exception as e:
            last_err = e

    # Last resort: yfinance history endpoint for symbols Yahoo v8 rejected.
    for i, sym in enumerate(tried):
        try:
            s, yahoo_events = _fetch_one_yf(sym)
            src = f"{primary}->{sym}" if i > 0 else primary
            print(f"[TR][yf-fallback] {src} ({len(s)} rows)")
            return _post_process(s, yahoo_events)
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


# ── Beta estimator ─────────────────────────────────────────────────
# Single hedge-ratio estimator: OLS slope on aligned simple returns,
# Bayesian-shrunk to listed leverage L with autocorrelation-adjusted
# effective sample size. Replaces all prior magnitude / "implausible"
# / "compression" guards (those were binary snaps; this is the smooth
# version of the same idea).
#
#   β̂_OLS  = Cov(r_etf, r_und) / Var(r_und)        (simple returns)
#   ρ_AR1  = lag-1 autocorrelation of r_und
#   n_eff  = n × (1 − ρ_AR1) / (1 + ρ_AR1)         (clipped to [1, n])
#   k      = BETA_SHRINK_K_BASE × max(1, L²)        (stronger prior on
#                                                    higher-leverage
#                                                    designs)
#   w      = n_eff / (n_eff + k)
#   β̂      = w · β̂_OLS + (1 − w) · L
#
# Sign-disagreement is still treated as data corruption (snap to L).
# That is the only retained hard guard.

BETA_SHRINK_K_BASE = 60     # ≈ one quarter of effective trading days
BETA_MIN_DAYS = 60          # min aligned daily returns before shrinkage kicks in
BETA_AR1_FLOOR_RATIO = 0.10 # n_eff cannot fall below 10 % of n


def compute_beta_ols(etf_tr: pd.Series, und_tr: pd.Series,
                     min_days: int = BETA_MIN_DAYS
                     ) -> tuple[float | None, int]:
    """Raw OLS slope on aligned daily simple returns. (β̂, n_obs).

    Kept as a stand-alone diagnostic; production code should call
    :func:`compute_beta_shrunk` which composes this with shrinkage.
    """
    etf_tr = etf_tr[~etf_tr.index.duplicated(keep="last")]
    und_tr = und_tr[~und_tr.index.duplicated(keep="last")]
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


def _ar1_n_eff(returns: np.ndarray) -> tuple[float, int]:
    """Lag-1 autocorrelation and AR(1)-adjusted effective sample size.

    n_eff = n · (1 − ρ) / (1 + ρ). Returns (ρ, n_eff) with n_eff
    clipped to [BETA_AR1_FLOOR_RATIO · n, n].
    """
    n = len(returns)
    if n < 5:
        return 0.0, n
    r = np.asarray(returns, dtype=float)
    r = r - r.mean()
    denom = float((r * r).sum())
    if denom <= 0:
        return 0.0, n
    rho = float((r[1:] * r[:-1]).sum() / denom)
    rho = max(-0.95, min(0.95, rho))
    factor = (1.0 - rho) / (1.0 + rho)
    n_eff = int(round(max(BETA_AR1_FLOOR_RATIO * n, min(n, n * factor))))
    return rho, max(1, n_eff)


def compute_beta_shrunk(
    etf_tr: pd.Series,
    und_tr: pd.Series,
    exp_leverage: float,
    *,
    min_days: int = BETA_MIN_DAYS,
    k_base: float = BETA_SHRINK_K_BASE,
    leverage_scaled_prior: bool = True,
) -> tuple[float, int, str]:
    """Shrunk hedge-ratio estimator. Returns (β, n_obs, source).

    source ∈ {
        "shrunk_to_L",            # standard path (data + listing prior)
        "imputed_no_overlap",      # < min_days aligned returns
        "imputed_var_zero",        # Var(r_und) ≈ 0
        "imputed_sign_mismatch",   # OLS sign contradicts listing
    }
    """
    L = float(exp_leverage) if np.isfinite(exp_leverage) else 0.0

    raw = compute_beta_ols(etf_tr, und_tr, min_days=min_days)
    b_ols, n = raw
    if b_ols is None or n < 2:
        return L, n, "imputed_no_overlap"

    # Sign-disagreement is data corruption, not an estimator artifact.
    if L != 0 and abs(b_ols) > 0.3 and (b_ols > 0) != (L > 0):
        print(
            f"  [BETA] sign mismatch: OLS β={b_ols:.4f} vs expected L={L:.2f}; "
            "using L."
        )
        return L, n, "imputed_sign_mismatch"

    # Recompute aligned returns once for ρ_AR1 (cheap; could be returned
    # from compute_beta_ols but kept separate for readability).
    df = pd.concat(
        [etf_tr.rename("etf"), und_tr.rename("und")], axis=1, sort=True
    ).dropna()
    r_und = df["und"].pct_change().dropna().values
    if r_und.size < 2:
        return L, n, "imputed_no_overlap"

    _, n_eff = _ar1_n_eff(r_und)

    # Stronger prior for higher-leverage products: contractually defined.
    k = float(k_base) * (max(1.0, L * L) if leverage_scaled_prior else 1.0)
    w = float(n_eff) / float(n_eff + k)
    beta = w * float(b_ols) + (1.0 - w) * L
    return round(beta, 6), n, "shrunk_to_L"


_COVERED_CALL_ETFS = {_norm_sym(etf) for etf, _ in covered_call_pairs}
_YIELDBOOST_PAIRS_NORMED = [
    (_norm_sym(etf), _norm_sym(und)) for etf, und in YIELDBOOST_BUCKET2_PAIRS
]


def classify_beta_product_class(row: pd.Series) -> str:
    """Classify a universe row for the beta-prior router.

    The classification is *independent* of any realized β (we need the
    prior class to build the prior).  The order matters: vol-ETPs are
    flagged first so any LETF-style row that happens to also reference
    a vol-ETP underlying is still routed to the no-shrink branch.

    Returns one of:
        ``letf_long``       — positive nominal leverage (the only class that
                              uses ``mu_beta = L`` as the prior mean).
        ``letf_inverse``    — negative nominal leverage from
                              ``INVERSE_ETF_UNIVERSE`` (also uses ``mu_beta = L``).
        ``income_yieldboost`` — ``is_yieldboost == True``.
        ``covered_call_1x`` — ETF is in ``covered_call_pairs``.
        ``volatility_etp``  — ETF or underlying is in ``VOLATILITY_ETP_SYMBOLS``.
        ``scraped_income``  — 1× income row that did not match the lists above
                              (YieldMax / Roundhill scrape, or supplemental
                              1× income product).
        ``unknown``         — fallthrough.
    """
    etf = _norm_sym(row.get("ETF", ""))
    und = _norm_sym(row.get("Underlying", "")) if pd.notna(row.get("Underlying")) else ""

    if etf in VOLATILITY_ETP_SYMBOLS or und in VOLATILITY_ETP_SYMBOLS:
        return "volatility_etp"

    lev = row.get("Leverage")
    try:
        lev_f = float(lev) if pd.notna(lev) else None
    except (TypeError, ValueError):
        lev_f = None

    if lev_f is not None and lev_f < 0:
        return "letf_inverse"

    is_yb = row.get("is_yieldboost")
    if isinstance(is_yb, (bool, np.bool_)) and bool(is_yb):
        return "income_yieldboost"
    if is_yb is True:
        return "income_yieldboost"

    if etf in _COVERED_CALL_ETFS:
        return "covered_call_1x"

    if lev_f is not None and abs(lev_f - 1.0) < 0.01:
        return "scraped_income"

    if lev_f is not None and lev_f > 1.0:
        return "letf_long"

    return "unknown"


def add_betas(
    universe_df: pd.DataFrame,
    tr_map: dict[str, pd.Series],
    min_days: int = BETA_MIN_DAYS,
    *,
    cc_default_mu: float = 0.55,
    cc_default_tau: float = 30.0,
) -> pd.DataFrame:
    """Populate Beta + posterior diagnostics via :mod:`beta_estimator`.

    Outputs (CSV-stable):
        Beta, Beta_n_obs, Beta_source        — back-compat (Beta_source
            becomes ``posterior.<prior_source>`` for the new path,
            ``imputed_*`` for fallthrough).
        Beta_se, Beta_resid_sigma_annual,
        Beta_horizon_chosen, Beta_quality,
        Beta_prior_mu, Beta_prior_tau,
        Beta_prior_source, Beta_product_class.

    The two LETF classes are the only ones that read row[``Leverage``]
    for the prior mean; income / vol-ETP / unknown rows use class-level
    defaults or the YieldBOOST hierarchical family prior.
    """

    df = universe_df.copy()

    df["Beta_product_class"] = df.apply(classify_beta_product_class, axis=1)
    pclass_counts = df["Beta_product_class"].value_counts().to_dict()
    print(
        "[BETA] Product class counts: "
        + " | ".join(f"{k}={v}" for k, v in pclass_counts.items())
    )

    yb_priors = build_yieldboost_family_priors(
        _YIELDBOOST_PAIRS_NORMED,
        tr_map,
        min_days=min_days,
        tau_per_sibling=30.0,
        tau_cap=120.0,
    )
    if yb_priors:
        per_und = {
            k: v for k, v in yb_priors.items() if k != "__global__"
        }
        glb = yb_priors.get("__global__")
        glb_str = (
            f" | __global__: μ={glb.mu:+.3f} τ={glb.tau:.0f} n={glb.n_siblings}"
            if glb is not None
            else ""
        )
        per_und_str = ", ".join(
            f"{u}: μ={p.mu:+.3f} τ={p.tau:.0f} n={p.n_siblings}"
            for u, p in sorted(per_und.items())
        )
        print(f"[BETA] YieldBOOST family priors → {per_und_str}{glb_str}")
    else:
        print("[BETA] YieldBOOST family priors: (no siblings with sufficient history)")

    betas: list[float] = []
    nobs: list[int] = []
    sources: list[str] = []
    ses: list[float] = []
    resid_sigmas: list[float] = []
    horizons: list[int] = []
    qualities: list[str] = []
    prior_mus: list[float] = []
    prior_taus: list[float] = []
    prior_sources: list[str] = []

    for _, row in df.iterrows():
        etf = _norm_sym(row["ETF"])
        und = _norm_sym(row["Underlying"]) if pd.notna(row.get("Underlying")) else None
        product_class = str(row["Beta_product_class"])
        lev_raw = row.get("Leverage")
        try:
            lev_f = float(lev_raw) if pd.notna(lev_raw) else None
        except (TypeError, ValueError):
            lev_f = None

        # Build prior. ONLY letf_long / letf_inverse may use nominal L.
        try:
            prior = BetaPrior.for_row(
                product_class=product_class,
                nominal_leverage=lev_f,
                underlying=und,
                peer_betas=yb_priors,
                cc_default_mu=cc_default_mu,
                cc_default_tau=cc_default_tau,
            )
        except ValueError:
            # LETF row missing nominal leverage → impute via fallback path
            prior = BetaPrior(
                mu=0.0,
                tau=15.0,
                source="empirical_bayes_global",
                product_class="unknown",
            )

        if not und or etf not in tr_map or und not in tr_map:
            betas.append(float(prior.mu))
            nobs.append(0)
            sources.append("imputed_missing_prices")
            ses.append(float("inf"))
            resid_sigmas.append(float("nan"))
            horizons.append(1)
            qualities.append("imputed_missing_prices")
            prior_mus.append(float(prior.mu))
            prior_taus.append(float(prior.tau))
            prior_sources.append(prior.source)
            continue

        result: BetaResult = compute_beta_for_hedging(
            tr_map[etf], tr_map[und], prior, min_days=min_days
        )
        betas.append(float(result.beta))
        nobs.append(int(result.n_obs))
        sources.append(str(result.source))
        ses.append(float(result.beta_se))
        resid_sigmas.append(float(result.resid_sigma_annual))
        horizons.append(int(result.horizon))
        qualities.append(str(result.quality))
        prior_mus.append(float(result.prior_mu))
        prior_taus.append(float(result.prior_tau))
        prior_sources.append(str(result.prior_source))

    df["Beta"] = betas
    df["Beta_n_obs"] = nobs
    df["Beta_source"] = sources
    df["Beta_se"] = ses
    df["Beta_resid_sigma_annual"] = resid_sigmas
    df["Beta_horizon_chosen"] = horizons
    df["Beta_quality"] = qualities
    df["Beta_prior_mu"] = prior_mus
    df["Beta_prior_tau"] = prior_taus
    df["Beta_prior_source"] = prior_sources

    src_counts = df["Beta_source"].value_counts().to_dict()
    summary = " | ".join(f"{k}={v}" for k, v in src_counts.items())
    print(f"[BETA] {summary} | Total: {len(df)}")
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
    "borrow_low": 0.55,
    "purgatory_margin": 0.25,
    "min_shares_available": 1000,
    "whitelist_hard_borrow_cap": 0.55,
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


def recompute_purgatory_by_bucket(
    screened: pd.DataFrame,
    *,
    screener_cfg: dict | None,
    sleeves_cfg: dict | None,
) -> pd.DataFrame:
    """After ``bucket`` is assigned, set ``purgatory`` (borrow band OR soft net edge).

    **Not purgatory** requires both: (1) borrow not in the per-bucket elevated band below,
    and (2) ``net_edge_p50_annual`` > ``purgatory_net_edge_max_annual`` (default 5%),
    when that column exists. Otherwise the row is purgatory (0 new size downstream).

    **Hard exclusion** of negative median net edge is done in ``generate_trade_plan``
    (``net_edge_p50_annual < 0``), not via this flag.

    Borrow convention matches ``screen_universe``: ``borrow_current`` is annual
    **cost** to the short (higher = worse), same as ls-algo FTP feed.

    Protected (whitelist + flow): flow uses ``flow_hard_borrow_cap``; whitelist-only
    uses ``whitelist_hard_borrow_cap``. Above the applicable cap => ``protected_bad``
    => purgatory (keep-open semantics downstream).
    """
    out = screened.copy()
    sc = screener_cfg or {}
    sl = sleeves_cfg or {}
    pb = sc.get("per_bucket") or {}

    defaults: dict[str, dict[str, float]] = {
        "bucket_1": {"borrow_low": 0.55, "purgatory_margin": 0.25, "hard_borrow_cap": 0.75},
        "bucket_2": {"borrow_low": 0.20, "purgatory_margin": 0.15, "hard_borrow_cap": 0.35},
        "bucket_4": {"borrow_low": 1.00, "purgatory_margin": 0.20, "hard_borrow_cap": 1.20},
    }

    def _triple(tag: str) -> tuple[float, float, float]:
        raw = pb.get(tag) or {}
        base = defaults.get(tag, {"borrow_low": 1.0, "purgatory_margin": 0.0, "hard_borrow_cap": 1.0})
        low = float(raw.get("borrow_low", base["borrow_low"]))
        margin = float(raw.get("purgatory_margin", base["purgatory_margin"]))
        hard = float(raw.get("hard_borrow_cap", base["hard_borrow_cap"]))
        return low, margin, hard

    low1, m1, h1 = _triple("bucket_1")
    low2, m2, h2 = _triple("bucket_2")
    low4, m4, h4 = _triple("bucket_4")

    wl = sl.get("whitelist_stock", {}).get("universe", {}).get("etfs", []) or []
    fl = sl.get("flow_program", {}).get("universe", {}).get("shorts", []) or []
    wl_set = {_norm_sym(x) for x in wl if str(x).strip()}
    fl_set = {_norm_sym(x) for x in fl if str(x).strip()}
    protected = wl_set | fl_set
    wl_hard = float(sc.get("whitelist_hard_borrow_cap", sc.get("hard_borrow_cap", 0.55)))
    flow_hard = float(sc.get("flow_hard_borrow_cap", 0.40))

    borrow = pd.to_numeric(out.get("borrow_current"), errors="coerce")
    ftp_miss = out.get("borrow_missing_from_ftp", pd.Series(False, index=out.index))
    ftp_miss = ftp_miss.fillna(False).astype(bool)
    borrow_known = borrow.notna() & (~ftp_miss)

    etf_n = out["ETF"].astype(str).map(_norm_sym)
    in_flow = etf_n.isin(fl_set)
    in_wl_only = etf_n.isin(wl_set) & ~in_flow
    prot = etf_n.isin(protected)
    protected_ok = prot & borrow_known & (
        (in_flow & (borrow <= flow_hard)) | (in_wl_only & (borrow <= wl_hard))
    )
    protected_bad = prot & borrow_known & (~protected_ok)

    bkt = out.get("bucket", pd.Series("", index=out.index)).astype(str)
    low_sel = np.select(
        [bkt.eq("bucket_1"), bkt.eq("bucket_2"), bkt.eq("bucket_4")],
        [low1, low2, low4],
        default=np.nan,
    )
    margin_sel = np.select(
        [bkt.eq("bucket_1"), bkt.eq("bucket_2"), bkt.eq("bucket_4")],
        [m1, m2, m4],
        default=np.nan,
    )
    hard_sel = np.select(
        [bkt.eq("bucket_1"), bkt.eq("bucket_2"), bkt.eq("bucket_4")],
        [h1, h2, h4],
        default=np.nan,
    )
    upper = np.minimum(low_sel + margin_sel, hard_sel)
    borrow_purg = (
        borrow_known
        & (~prot)
        & np.isfinite(low_sel)
        & (borrow > low_sel)
        & (borrow <= upper)
    )
    borrow_purg = borrow_purg | protected_bad

    ne_hi = float(sc.get("purgatory_net_edge_max_annual", 0.05))
    if "net_edge_p50_annual" in out.columns:
        ne = pd.to_numeric(out["net_edge_p50_annual"], errors="coerce")
        net_purg = ne.notna() & (ne >= 0.0) & (ne <= ne_hi)
    else:
        net_purg = pd.Series(False, index=out.index)

    # Vol-ratio gate: when σ_etf vs |β|·σ_und is outside the per-leverage band
    # (default ±50 % for a 2x ETF), the row is treated as data-quality
    # purgatory until the underlying split / corporate action is resolved.
    # See ``recompute_vol_ratio_gate``.
    gate_cfg = _vol_ratio_gate_cfg(sc)
    if (
        gate_cfg.get("enabled", True)
        and gate_cfg.get("purgatory_on_outlier", True)
        and "vol_ratio_outlier" in out.columns
    ):
        vol_ratio_purg = (
            out["vol_ratio_outlier"].fillna(False).astype(bool)
        )
    else:
        vol_ratio_purg = pd.Series(False, index=out.index)

    out["purgatory"] = (borrow_purg | net_purg | vol_ratio_purg).fillna(False)
    return out


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
    4. Squared log returns are winsorized at the 1st/99th percentile (or
       wider tails for thin histories) so a single uncaught corporate
       action — typically a split that the upstream cleaner missed —
       cannot single-handedly dominate σ. ``_compute_gross_decay_daily``
       uses the same tier table; keeping the two in sync preserves the
       realized-vs-expected algebraic identity for clean symbols.

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
    sq = (r ** 2).to_numpy(dtype=float)
    n_sq = int(sq.size)
    if n_sq == 0:
        return None
    if n_sq >= 100:
        lo_p, hi_p = 1.0, 99.0
    elif n_sq >= 60:
        lo_p, hi_p = 2.0, 98.0
    else:
        lo_p, hi_p = 5.0, 95.0
    lo, hi = np.percentile(sq, [lo_p, hi_p])
    sq_w = np.clip(sq, lo, hi)
    m2 = float(np.mean(sq_w))
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
    sigma_cap: float | None = None,
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
    sigma_cap : float, optional
        When set, the **vol-drag term only** is computed at
        ``min(sigma_annual, sigma_cap)``. Financing, expense, and manager-borrow
        terms remain linear in their own inputs. Used by
        :func:`enrich_with_decay_and_vol` to clip σ at the realized-decay
        calibration ceiling so the Itô identity (γ ∝ σ²) cannot blow up on
        underlyings whose Yahoo TR σ disagrees with the LETF panel consensus
        (e.g. BMNR vs BMNG/BMNU/BMNZ).
    """
    sigma_used = float(sigma_annual)
    if sigma_cap is not None and np.isfinite(float(sigma_cap)) and float(sigma_cap) > 0.0:
        sigma_used = min(sigma_used, float(sigma_cap))

    if _HAS_CORE:
        try:
            return _expected_gross_decay(
                beta,
                sigma_used,
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
    drag = 0.5 * abs_b * abs_bm1 * sigma_used ** 2
    fin = (beta - 1.0) * risk_free_rate
    mgr = abs_b * mgr_borrow_on_underlying if beta < 0 else 0.0
    return round(drag + expense_ratio + fin + mgr, 6)


def _is_volatility_etp_symbol(symbol: object, underlying: object = "") -> bool:
    sym = _norm_sym(symbol) if symbol is not None and str(symbol).strip() else ""
    und = _norm_sym(underlying) if underlying is not None and str(underlying).strip() else ""
    return sym in VOLATILITY_ETP_SYMBOLS or und in VOLATILITY_ETP_SYMBOLS


def apply_volatility_etp_expected_decay_adjustment(df: pd.DataFrame) -> pd.DataFrame:
    """Use a model-aware expected-decay value for VIX futures ETPs.

    ``expected_gross_decay_annual`` originally contains the simple daily-LETF
    Itô identity.  For products like UVIX/SVIX that identity is incomplete: the
    dominant expected behavior also includes futures roll, vol risk premium,
    jumps, and mean reversion.  Schema v2 already computes the empirical gap as
    ``realized_tracking_component_annual = realized - simple_ito``.  Preserve the
    simple value in a separate column, then make the displayed expected value
    simple Itô + empirical adjustment for volatility ETPs.
    """
    out = df.copy()
    if "expected_gross_decay_annual" not in out.columns:
        return out

    simple = pd.to_numeric(out["expected_gross_decay_annual"], errors="coerce")
    out["expected_gross_decay_simple_ito_annual"] = simple
    if "expected_gross_decay_adjusted_annual" not in out.columns:
        out["expected_gross_decay_adjusted_annual"] = np.nan
    if "expected_decay_adjustment_annual" not in out.columns:
        out["expected_decay_adjustment_annual"] = np.nan
    if "expected_decay_model" not in out.columns:
        out["expected_decay_model"] = "simple_ito"

    adjustment = pd.to_numeric(
        out.get("realized_tracking_component_annual", pd.Series(np.nan, index=out.index)),
        errors="coerce",
    )
    realized = pd.to_numeric(
        out.get("gross_decay_annual", pd.Series(np.nan, index=out.index)),
        errors="coerce",
    )
    is_vol = out.apply(
        lambda r: _is_volatility_etp_symbol(r.get("ETF"), r.get("Underlying")),
        axis=1,
    )
    adjusted = simple + adjustment
    adjusted = adjusted.where(adjusted.notna(), realized)

    out.loc[is_vol, "expected_gross_decay_adjusted_annual"] = adjusted[is_vol].round(6)
    out.loc[is_vol, "expected_decay_adjustment_annual"] = adjustment[is_vol].round(6)
    out.loc[is_vol, "expected_decay_model"] = "volatility_etp_empirical_roll_adjusted"
    out.loc[is_vol & adjusted.notna(), "expected_gross_decay_annual"] = adjusted[is_vol & adjusted.notna()].round(6)
    if "expected_gross_decay_reliable" in out.columns:
        out.loc[is_vol, "expected_gross_decay_reliable"] = False
    if "product_class" in out.columns:
        out.loc[is_vol, "product_class"] = "volatility_etp"
    return out


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

    # Winsorize daily drag at the 1st / 99th percentile so a single bad
    # print (stale Yahoo adjclose vs fresh listing, corporate-action glitch,
    # or one-off gap) cannot dominate mean(drag)×252.  Thin histories use
    # slightly wider tails so we still keep ≥ ~1 day per side.
    drag_vals = daily_drag.to_numpy(dtype=float)
    n_drag = int(drag_vals.size)
    if n_drag >= 100:
        lo_p, hi_p = 1.0, 99.0
    elif n_drag >= 60:
        lo_p, hi_p = 2.0, 98.0
    else:
        lo_p, hi_p = 5.0, 95.0
    lo, hi = np.percentile(drag_vals, [lo_p, hi_p])
    drag_w = np.clip(drag_vals, lo, hi)
    raw_mean = float(np.mean(drag_vals))
    win_mean = float(np.mean(drag_w))
    if label and abs(win_mean - raw_mean) > 1e-6 * max(1.0, abs(raw_mean)):
        n_clip = int(np.sum((drag_vals < lo) | (drag_vals > hi)))
        if n_clip:
            print(
                f"[DECAY][winsor] {label}: clipped {n_clip} day(s) at "
                f"{lo_p:g}/{hi_p:g}% drag; mean {raw_mean*252:.2%}→{win_mean*252:.2%} annualized"
            )

    return round(win_mean * TRADING_DAYS, 6)


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
      - vol_underlying_source        : JSON probe summary per underlying (PASS 2)
      - sigma_realized_implied_annual: median √(G/(½|β||β−1|)) across LETFs on U
      - sigma_cap_annual             : 1.25× that median; caps vol-drag in expected
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
        exp_lev = float(row["Leverage"]) if pd.notna(row.get("Leverage")) else 2.0

        # Fill missing β via the same shrunk estimator used in add_betas.
        if beta_f is None and und and etf in tr_map and und in tr_map:
            beta_f, n_obs_i, _src = compute_beta_shrunk(
                tr_map[etf], tr_map[und], exp_lev, min_days=min_days
            )
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
    # For each underlying, build a *panel* of σ probes — raw Yahoo TR σ plus
    # one ETF-implied σ per LETF row on the same underlying — then pool them
    # robustly. The pool is bounded by a realized-decay calibration ceiling so
    # that γ ∝ σ² in the Itô identity cannot blow up when raw TR disagrees
    # with the LETF panel consensus (e.g. BMNR vs BMNG/BMNU/BMNZ).
    #
    # Algorithm summary (mirrors Diamond-Creek-Quant PR #8):
    #   1. probes = {σ_raw} ∪ {σ_etf_r / |β_r| | r on U}
    #   2. σ_pool = weighted_median(probes)
    #         weights: n_obs_r · |β_r| for implied; max(implied weight) · 0.5
    #         for raw (present but never dominant in a multi-probe panel).
    #   3. MAD outlier trim: drop probes with |σ - median| > 3 · MAD; recompute.
    #   4. realized-implied σ_real_med = median(√(G_r / (½|β_r||β_r-1|))).
    #         ceiling = SIGMA_CAP_SAFETY · σ_real_med.
    #   5. cap σ_pool at the ceiling.
    #   6. final fallbacks: if pool < min(implied)/2 use raw; if everything is
    #         missing use raw.
    SIGMA_CAP_SAFETY = 1.25
    MAD_TRIM_K = 3.0

    def _weighted_median(values: list[float], weights: list[float]) -> float | None:
        pairs = [
            (float(v), float(w))
            for v, w in zip(values, weights)
            if np.isfinite(v) and np.isfinite(w) and w > 0
        ]
        if not pairs:
            return None
        pairs.sort(key=lambda t: t[0])
        total = sum(w for _, w in pairs)
        if total <= 0:
            return None
        cum = 0.0
        half = total / 2.0
        for v, w in pairs:
            cum += w
            if cum >= half:
                return float(v)
        return float(pairs[-1][0])

    # Group rows by underlying
    und_syms = df["Underlying"].dropna().apply(_norm_sym).unique()
    resolved_vol_und: dict[str, float | None] = {}
    resolved_vol_und_legacy: dict[str, float | None] = {}
    sigma_und_source: dict[str, dict] = {}
    sigma_real_med_map: dict[str, float | None] = {}

    for und in und_syms:
        raw_vol = _annualized_second_moment_log(tr_map[und], min_days) if und in tr_map else None

        mask = df["Underlying"].apply(
            lambda x: _norm_sym(x) == und if pd.notna(x) else False)

        implied_candidates: list[tuple[float, float, str]] = []
        realized_implied: list[float] = []
        for idx in df.index[mask]:
            b = betas_out[idx]
            ve = vols_etf_raw[idx]
            nobs = beta_nobs_out[idx]
            if b and abs(b) >= 0.5 and ve and ve > 0 and nobs:
                implied = float(ve) / abs(float(b))
                weight = float(nobs) * abs(float(b))
                etf_name = _norm_sym(df.at[idx, "ETF"]) if pd.notna(df.at[idx, "ETF"]) else "?"
                implied_candidates.append((implied, weight, etf_name))

            g_real = decays[idx]
            if (
                b is not None
                and np.isfinite(b)
                and abs(b) >= 0.1
                and abs(b - 1.0) >= 0.05
                and g_real is not None
                and np.isfinite(g_real)
                and g_real > 0
            ):
                gamma_coef = 0.5 * abs(b) * abs(b - 1.0)
                if gamma_coef > 1e-9:
                    sigma_r = float(np.sqrt(max(float(g_real), 0.0) / gamma_coef))
                    if np.isfinite(sigma_r) and sigma_r > 0:
                        realized_implied.append(sigma_r)

        sigma_real_med = float(np.median(realized_implied)) if realized_implied else None
        sigma_real_med_map[und] = sigma_real_med
        ceiling = (
            SIGMA_CAP_SAFETY * sigma_real_med
            if sigma_real_med is not None and np.isfinite(sigma_real_med) and sigma_real_med > 0
            else None
        )

        if not implied_candidates and (raw_vol is None or raw_vol <= 0):
            resolved_vol_und[und] = raw_vol
            sigma_und_source[und] = {
                "method": "no_data",
                "n_probes": 0,
                "n_outliers_dropped": 0,
                "sigma_real_implied_median": sigma_real_med,
                "sigma_pool": raw_vol,
                "sigma_capped_to": None,
            }
            continue

        if not implied_candidates:
            resolved_vol_und[und] = raw_vol
            sigma_und_source[und] = {
                "method": "raw_only",
                "n_probes": 1,
                "n_outliers_dropped": 0,
                "sigma_real_implied_median": sigma_real_med,
                "sigma_pool": raw_vol,
                "sigma_capped_to": None,
            }
            continue

        max_implied_weight = max(w for _, w, _ in implied_candidates)
        probe_values: list[float] = []
        probe_weights: list[float] = []
        probe_labels: list[str] = []
        if raw_vol is not None and np.isfinite(raw_vol) and raw_vol > 0:
            probe_values.append(float(raw_vol))
            probe_weights.append(0.5 * max_implied_weight)
            probe_labels.append("raw_und")
        for v, w, etf_name in implied_candidates:
            probe_values.append(float(v))
            probe_weights.append(float(w))
            probe_labels.append(f"impl[{etf_name}]")

        pool0 = _weighted_median(probe_values, probe_weights)

        n_outliers_dropped = 0
        if pool0 is not None and len(probe_values) >= 3:
            med0 = float(np.median(probe_values))
            mad0 = float(np.median([abs(v - med0) for v in probe_values]))
            if mad0 > 0:
                kept = [
                    (v, w, lbl)
                    for v, w, lbl in zip(probe_values, probe_weights, probe_labels)
                    if abs(v - med0) <= MAD_TRIM_K * mad0
                ]
                if len(kept) >= 2 and len(kept) < len(probe_values):
                    n_outliers_dropped = len(probe_values) - len(kept)
                    pool0 = _weighted_median(
                        [v for v, _, _ in kept],
                        [w for _, w, _ in kept],
                    )

        sigma_pool = pool0 if pool0 is not None and pool0 > 0 else raw_vol

        capped_to: float | None = None
        method = "weighted_median"
        if sigma_pool is not None and ceiling is not None and sigma_pool > ceiling:
            capped_to = float(round(ceiling, 6))
            sigma_pool = float(ceiling)
            method = "weighted_median_capped"

        implied_only = [v for v, _, lbl in zip(probe_values, probe_weights, probe_labels) if lbl != "raw_und"]
        if (
            sigma_pool is not None
            and implied_only
            and sigma_pool > 0
            and sigma_pool < 0.5 * min(implied_only)
            and raw_vol is not None
            and raw_vol > 0
        ):
            sigma_pool = float(raw_vol)
            method = "raw_fallback"
            capped_to = None
            # Raw Yahoo σ can still exceed the realized-implied calibration
            # ceiling (e.g. BMNR vs a thin LETF panel). Never let raw_fallback
            # bypass the σ cap — that was blowing up expected_gross_decay.
            if ceiling is not None and sigma_pool > ceiling:
                capped_to = float(round(ceiling, 6))
                sigma_pool = float(ceiling)
                method = "raw_fallback_capped"

        resolved_vol_und[und] = round(float(sigma_pool), 6) if sigma_pool is not None else None
        sigma_und_source[und] = {
            "method": method,
            "n_probes": len(probe_values),
            "n_outliers_dropped": int(n_outliers_dropped),
            "sigma_real_implied_median": (
                float(round(sigma_real_med, 6)) if sigma_real_med is not None else None
            ),
            "sigma_pool": (
                float(round(sigma_pool, 6)) if sigma_pool is not None else None
            ),
            "sigma_capped_to": capped_to,
        }

        if len(probe_values) > 0:
            if sigma_real_med is not None:
                print(
                    f"  [VOL-PROBES] {und}: n_probes={len(probe_values)} "
                    f"n_outliers={n_outliers_dropped} "
                    f"σ_pool={(sigma_pool or 0)*100:.1f}% "
                    f"σ_realized={sigma_real_med * 100:.1f}%"
                )
            else:
                print(
                    f"  [VOL-PROBES] {und}: n_probes={len(probe_values)} "
                    f"n_outliers={n_outliers_dropped} "
                    f"σ_pool={(sigma_pool or 0)*100:.1f}% σ_realized=NA"
                )
            if capped_to is not None:
                print(
                    f"  [VOL-CAP] {und}: σ_pool capped to {capped_to*100:.1f}% "
                    f"= {SIGMA_CAP_SAFETY:g}× realized-implied σ"
                )

    # ── Parallel legacy-σ resolution (diagnostics only) ──
    # Centered, simple-return σ kept so downstream tools can reproduce the
    # pre-Itô behaviour. Uses the same robust weighted-median + realized-decay
    # ceiling as the production path, just on legacy probes.
    for und in und_syms:
        raw_vol_legacy = _annualized_vol(tr_map[und], min_days) if und in tr_map else None
        mask = df["Underlying"].apply(
            lambda x, u=und: _norm_sym(x) == u if pd.notna(x) else False)
        implied_legacy: list[tuple[float, float]] = []
        for idx in df.index[mask]:
            b = betas_out[idx]
            ve = vols_etf_legacy[idx]
            nobs = beta_nobs_out[idx]
            if b and abs(b) >= 0.5 and ve and ve > 0 and nobs:
                implied_legacy.append((float(ve) / abs(float(b)), float(nobs) * abs(float(b))))

        if not implied_legacy and (raw_vol_legacy is None or raw_vol_legacy <= 0):
            resolved_vol_und_legacy[und] = raw_vol_legacy
            continue
        if not implied_legacy:
            resolved_vol_und_legacy[und] = raw_vol_legacy
            continue

        max_w = max(w for _, w in implied_legacy)
        vals = [v for v, _ in implied_legacy]
        wts = [w for _, w in implied_legacy]
        if raw_vol_legacy is not None and np.isfinite(raw_vol_legacy) and raw_vol_legacy > 0:
            vals.insert(0, float(raw_vol_legacy))
            wts.insert(0, 0.5 * max_w)

        pool_legacy = _weighted_median(vals, wts)
        if pool_legacy is None:
            resolved_vol_und_legacy[und] = raw_vol_legacy
            continue

        sigma_real_med_u = sigma_real_med_map.get(und)
        if (
            sigma_real_med_u is not None
            and np.isfinite(sigma_real_med_u)
            and sigma_real_med_u > 0
        ):
            ceiling_legacy = SIGMA_CAP_SAFETY * sigma_real_med_u
            if pool_legacy > ceiling_legacy:
                pool_legacy = float(ceiling_legacy)

        resolved_vol_und_legacy[und] = round(float(pool_legacy), 6)

    # Assign resolved vol_und to each row + compute expected decay (4 terms).
    # expense_ratios / underlying_borrow_map were passed in from main(); missing
    # entries fall back to 0.0 with a reliability flag so downstream consumers
    # can choose to de-weight those rows.
    vols_und, vols_und_legacy, expected_decays = [], [], []
    expected_decays_legacy = []
    er_vals, er_sources = [], []
    mgr_borrows, reliables, rf_vals = [], [], []
    vol_und_source_payloads: list[str | None] = []
    sigma_caps_per_row: list[float | None] = []
    for i, row in df.iterrows():
        etf = _norm_sym(row["ETF"]) if pd.notna(row.get("ETF")) else None
        und = _norm_sym(row["Underlying"]) if pd.notna(row.get("Underlying")) else None
        vol_und = resolved_vol_und.get(und) if und else None
        vol_und_legacy = resolved_vol_und_legacy.get(und) if und else None
        vols_und.append(vol_und)
        vols_und_legacy.append(vol_und_legacy)
        if vol_und is not None: ok_vol += 1

        if und and und in sigma_und_source:
            try:
                vol_und_source_payloads.append(
                    json.dumps(sigma_und_source[und], separators=(",", ":"))
                )
            except (TypeError, ValueError):
                vol_und_source_payloads.append(None)
        else:
            vol_und_source_payloads.append(None)

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

        sigma_real_med_u = sigma_real_med_map.get(und)
        sigma_cap_row = (
            float(SIGMA_CAP_SAFETY * sigma_real_med_u)
            if sigma_real_med_u is not None and np.isfinite(sigma_real_med_u) and sigma_real_med_u > 0
            else None
        )
        sigma_caps_per_row.append(sigma_cap_row)

        exp_decay = None
        if beta_f and abs(beta_f) >= 0.1 and vol_und is not None and vol_und > 0:
            f_used = float(er_val) if er_val is not None else 0.0
            exp_decay = expected_gross_decay(
                beta_f,
                vol_und,
                expense_ratio=f_used,
                risk_free_rate=risk_free_rate,
                mgr_borrow_on_underlying=mgr_borrow,
                sigma_cap=sigma_cap_row,
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
                sigma_cap=sigma_cap_row,
            )
        expected_decays_legacy.append(exp_decay_legacy)

    df["vol_underlying_annual"] = vols_und
    df["vol_underlying_annual_legacy"] = vols_und_legacy
    df["vol_underlying_source"] = vol_und_source_payloads
    df["sigma_realized_implied_annual"] = [
        sigma_real_med_map.get(_norm_sym(u)) if pd.notna(u) else None
        for u in df["Underlying"]
    ]
    df["sigma_cap_annual"] = sigma_caps_per_row
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
    # For levered ETFs, vol_etf should track |β| × vol_und. We flag rows
    # whose ratio deviates from 1.0 in either direction. Default warn-band
    # is [0.20, 1.75]; the per-leverage gate inside the trade plan uses a
    # tighter [0.5, 1.5] band by default. Both can be overridden via
    # ``screener.vol_ratio_gate`` in YAML — see ``recompute_vol_ratio_gate``
    # for the structured columns this populates.
    df = recompute_vol_ratio_gate(df, screener_cfg=None)

    print(f"\n[DECAY] Beta OLS fill-ins: {betas_computed} | Vol: {ok_vol}/{len(df)} "
          f"| Realized decay: {ok_decay}/{len(df)} | Expected decay: {ok_expected}/{len(df)}")
    return df


def _vol_ratio_gate_cfg(screener_cfg: dict | None) -> dict:
    """Resolve the per-leverage vol-ratio gate from YAML.

    Schema::

        screener:
          vol_ratio_gate:
            enabled: true                 # toggle gate effects on purgatory
            diagnostic_min: 0.20          # warn-band lower bound
            diagnostic_max: 1.75          # warn-band upper bound
            purgatory_on_outlier: true    # whether outliers force purgatory
            by_abs_beta:                  # tighter gate per |β| bucket
              "1.0": {min: 0.5, max: 1.5}
              "2.0": {min: 0.5, max: 1.5}
              "3.0": {min: 0.5, max: 1.5}
            default: {min: 0.5, max: 1.5}

    Missing keys fall back to the safe defaults below.
    """
    sc = screener_cfg or {}
    gate = (sc.get("vol_ratio_gate") or {}) if isinstance(sc, dict) else {}
    return {
        "enabled": bool(gate.get("enabled", True)),
        "diagnostic_min": float(gate.get("diagnostic_min", 0.20)),
        "diagnostic_max": float(gate.get("diagnostic_max", 1.75)),
        "purgatory_on_outlier": bool(gate.get("purgatory_on_outlier", True)),
        "by_abs_beta": {
            str(k): {
                "min": float((v or {}).get("min", 0.5)),
                "max": float((v or {}).get("max", 1.5)),
            }
            for k, v in (gate.get("by_abs_beta") or {}).items()
        },
        "default": {
            "min": float((gate.get("default") or {}).get("min", 0.5)),
            "max": float((gate.get("default") or {}).get("max", 1.5)),
        },
    }


def _vol_ratio_bounds_for_beta(abs_beta: float, gate: dict) -> tuple[float, float]:
    key = f"{round(abs_beta, 1):.1f}"
    by_beta = gate.get("by_abs_beta", {}) or {}
    if key in by_beta:
        return by_beta[key]["min"], by_beta[key]["max"]
    default = gate.get("default", {"min": 0.5, "max": 1.5})
    return float(default["min"]), float(default["max"])


def recompute_vol_ratio_gate(
    df: pd.DataFrame,
    *,
    screener_cfg: dict | None,
) -> pd.DataFrame:
    """Add structured ``vol_ratio_value`` / ``vol_ratio_outlier`` columns.

    Replaces the legacy print-only diagnostic. Downstream consumers
    (generate_trade_plan, sizing, purgatory recompute) can refuse to size
    a sleeve when ``vol_ratio_outlier`` is True. The diagnostic warn-band
    stays loose (0.20–1.75) so we still get a console heads-up on borderline
    rows; the gate-band ([0.5, 1.5] by default) drives the boolean flag.
    """
    out = df.copy()
    gate = _vol_ratio_gate_cfg(screener_cfg)

    diag_lo = float(gate.get("diagnostic_min", 0.20))
    diag_hi = float(gate.get("diagnostic_max", 1.75))

    vu_arr = pd.to_numeric(out.get("vol_underlying_annual"), errors="coerce")
    ve_arr = pd.to_numeric(out.get("vol_etf_annual"), errors="coerce")
    b_arr = pd.to_numeric(out.get("Beta"), errors="coerce")

    ratio_vals: list[float] = []
    outlier_flags: list[bool] = []
    warn_lines: list[str] = []
    for i in out.index:
        vu = vu_arr.loc[i] if i in vu_arr.index else np.nan
        ve = ve_arr.loc[i] if i in ve_arr.index else np.nan
        b = b_arr.loc[i] if i in b_arr.index else np.nan
        if pd.notna(vu) and pd.notna(ve) and pd.notna(b) and vu > 0 and abs(b) >= 0.1:
            expected_ve = abs(b) * vu
            r = (ve / expected_ve) if expected_ve > 0 else np.nan
        else:
            r = np.nan

        if pd.notna(r) and pd.notna(b) and abs(b) >= 0.5:
            lo, hi = _vol_ratio_bounds_for_beta(abs(float(b)), gate)
            is_outlier = bool((r < lo) or (r > hi))
        else:
            is_outlier = False

        ratio_vals.append(float(r) if pd.notna(r) else np.nan)
        outlier_flags.append(is_outlier)

        if pd.notna(r) and (r < diag_lo or r > diag_hi):
            warn_lines.append(
                f"  {str(out.at[i, 'ETF']):8s} vol_und={(vu or 0)*100:6.1f}%  "
                f"vol_etf={(ve or 0)*100:6.1f}%  β={b:+.2f}  ratio={r:.2f}"
            )

    out["vol_ratio_value"] = ratio_vals
    out["vol_ratio_outlier"] = outlier_flags

    if warn_lines:
        print(
            f"\n[DECAY] ⚠ Vol-ratio outliers ({len(warn_lines)} rows — "
            f"vol_etf / (|β| × vol_und) outside [{diag_lo:.2f}, {diag_hi:.2f}]):"
        )
        for line in warn_lines[:10]:
            print(line)
        if len(warn_lines) > 10:
            print(f"  ... and {len(warn_lines) - 10} more")

    n_gate = int(sum(1 for f in outlier_flags if f))
    if n_gate:
        print(
            f"[DECAY] vol-ratio gate: {n_gate} row(s) flagged for purgatory "
            f"(per-|β| band)"
        )

    return out


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

def discover_default_borrow_history_json() -> Path | None:
    """Resolve ``borrow_history.json`` when no CLI/env path is set.

    Search order:
      1. ``ETF_DASHBOARD_ROOT``/data/borrow_history.json
      2. Sibling checkout: ``<ls-algo>/../etf-dashboard/data/borrow_history.json``
      3. In-repo copy: ``<ls-algo>/data/borrow_history.json`` (e.g. CI curl)
    """
    root = Path(__file__).resolve().parent
    candidates: list[Path] = []
    env_root = (os.environ.get("ETF_DASHBOARD_ROOT") or "").strip()
    if env_root:
        candidates.append(Path(env_root) / "data" / "borrow_history.json")
    candidates.append(root.parent / "etf-dashboard" / "data" / "borrow_history.json")
    candidates.append(root / "data" / "borrow_history.json")
    for p in candidates:
        try:
            if p.is_file():
                return p
        except OSError:
            continue
    return None


def _audit_universe_symbols(skip_inverse: bool) -> list[str]:
    """Build the canonical screener universe (ETFs + underlyings) for audits."""
    df = build_full_universe(skip_scrape=True, skip_inverse=skip_inverse)
    if df.empty:
        return []
    etf_syms = df["ETF"].apply(_norm_sym).tolist()
    und_syms = df["Underlying"].dropna().apply(_norm_sym).tolist()
    return sorted(set(t for t in etf_syms + und_syms if t))


def _run_audit_splits(args: argparse.Namespace) -> int:
    """Execute the read-only ``--audit-splits`` workflow.

    For each symbol in the resolved universe (or ``--symbols`` subset):

      1. Pulls the TR series via ``_get_total_return_series`` (which already
         routes through ``splits.clean_split_artifacts`` and records applied
         events into ``_SPLITS_APPLIED_LOG``).
      2. Re-fetches the RAW (uncorrected) series from Yahoo v8 directly to
         compute baseline σ for comparison. This avoids re-implementing the
         fetch path.
      3. Reports each applied event plus σ_raw vs σ_clean.

    With ``--write-overrides``, any heuristic-only event whose
    ``(symbol, ex_date)`` pair is not already present in the splits overrides
    CSV is appended.
    """
    print("=" * 70)
    print(f"  --audit-splits  run-date={args.run_date}  window={args.audit_window}d")
    print("=" * 70)

    if args.symbols:
        syms = [_norm_sym(s) for s in args.symbols.split(",") if str(s).strip()]
    else:
        print("[AUDIT] Building universe (skip-scrape, skip-inverse=False) ...")
        syms = _audit_universe_symbols(skip_inverse=False)
    if not syms:
        print("[AUDIT] No symbols resolved; aborting.")
        return 1
    print(f"[AUDIT] {len(syms)} symbol(s) to probe.")

    # Heuristic for "audit window" lookback: 30d covers same-day + recent.
    audit_period = "30d" if args.audit_window <= 30 else f"{args.audit_window + 5}d"

    rows: list[dict] = []
    new_overrides: list[SplitEvent] = []
    cutoff = pd.Timestamp(args.run_date) - pd.Timedelta(days=int(args.audit_window) + 3)

    # Snapshot of overrides CSV before this run so we only write NEW rows.
    existing_overrides = load_splits_overrides_csv(_SPLITS_OVERRIDES_CSV_PATH)
    existing_keys: set[tuple[str, pd.Timestamp]] = {
        (e.symbol, e.ex_date) for e in existing_overrides
    }

    for sym in syms:
        try:
            cleaned = _get_total_return_series(sym, period=audit_period)
        except Exception as e:
            print(f"[AUDIT] {sym} fetch failed: {e}")
            continue
        if cleaned.empty:
            continue

        applied = _SPLITS_APPLIED_LOG.get(_norm_sym(sym), [])
        # Filter to events within the requested window.
        applied = [e for e in applied if e.ex_date >= cutoff]
        if not applied:
            continue

        # Build a parallel raw series by re-running the upstream Yahoo fetch
        # without our cleanup (use yfinance via the bare API). Cheap because
        # we already paid for the network round-trip above.
        try:
            t = yf.Ticker(sym)
            raw_df = t.history(period=audit_period, auto_adjust=False)
            raw = raw_df["Close"].dropna() if "Close" in raw_df else pd.Series(dtype=float)
        except Exception:
            raw = pd.Series(dtype=float)

        sigma_raw = _annualized_second_moment_log(raw, min_days=10) if len(raw) >= 11 else None
        sigma_clean = _annualized_second_moment_log(cleaned, min_days=10) if len(cleaned) >= 11 else None

        for ev in applied:
            rows.append(
                {
                    "symbol": sym,
                    "ex_date": ev.ex_date.date().isoformat(),
                    "factor": float(ev.factor),
                    "source": ev.source,
                    "sigma_raw": (
                        round(float(sigma_raw), 4)
                        if sigma_raw is not None
                        else None
                    ),
                    "sigma_clean": (
                        round(float(sigma_clean), 4)
                        if sigma_clean is not None
                        else None
                    ),
                    "delta": (
                        round(float(sigma_raw - sigma_clean), 4)
                        if (sigma_raw is not None and sigma_clean is not None)
                        else None
                    ),
                    "note": ev.note[:80],
                }
            )
            if (
                ev.source == "heuristic"
                and (sym, ev.ex_date) not in existing_keys
            ):
                new_overrides.append(ev)

    if not rows:
        print("[AUDIT] No split events fired in the audit window.")
        return 0

    audit_df = pd.DataFrame(rows).sort_values(by=["ex_date", "symbol"]).reset_index(drop=True)
    print()
    print(audit_df.to_string(index=False))

    if args.write_overrides and new_overrides:
        # Convert SplitEvents to the CSV schema and append.
        out_path = _SPLITS_OVERRIDES_CSV_PATH
        if out_path is None:
            print("[AUDIT] No splits_overrides_csv path resolved; cannot write overrides.")
            return 0
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.is_file():
            try:
                existing_df = pd.read_csv(out_path)
            except Exception:
                existing_df = pd.DataFrame(
                    columns=["symbol", "ex_date", "numerator", "denominator", "source", "note"]
                )
        else:
            existing_df = pd.DataFrame(
                columns=["symbol", "ex_date", "numerator", "denominator", "source", "note"]
            )
        added_rows: list[dict] = []
        for ev in new_overrides:
            # SplitEvent.factor is the price-multiplier den/num (1-for-10
            # reverse → 10). The CSV stores num,den in share-multiplier form,
            # so invert: reverse split factor>1 → num=1, den=round(factor).
            if ev.factor >= 1.0:
                num = 1
                den = int(round(ev.factor))
            else:
                num = int(round(1.0 / ev.factor))
                den = 1
            added_rows.append(
                {
                    "symbol": ev.symbol,
                    "ex_date": ev.ex_date.date().isoformat(),
                    "numerator": num,
                    "denominator": den,
                    "source": ev.source,
                    "note": ev.note,
                }
            )
        merged = pd.concat([existing_df, pd.DataFrame(added_rows)], ignore_index=True)
        merged = merged.drop_duplicates(subset=["symbol", "ex_date"], keep="last")
        merged = merged.sort_values(by=["symbol", "ex_date"]).reset_index(drop=True)
        merged.to_csv(out_path, index=False)
        print(
            f"[AUDIT] Wrote {len(added_rows)} new override(s) to {out_path} "
            f"(file now has {len(merged)} rows)."
        )

    return 0


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
    ap.add_argument(
        "--borrow-history-path",
        default=None,
        help=(
            "Path to etf-dashboard borrow_history.json (or set BORROW_HISTORY_PATH). "
            "Enables weighted resampling of borrow for net_edge_* block bootstrap."
        ),
    )
    ap.add_argument(
        "--borrow-weight-halflife-days",
        type=float,
        default=90.0,
        help="Calendar-day half-life for 0.5**(age/H) borrow weights (default: 90)",
    )
    ap.add_argument(
        "--audit-splits",
        action="store_true",
        help=(
            "Run a read-only split audit instead of the full screener. Probes "
            "the universe (or --symbols) for splits applied by any source "
            "(yahoo_events / flex / overrides_csv / manual_override / "
            "heuristic) over the last --audit-window trading days, and prints "
            "raw vs corrected σ for each. Use --write-overrides to append "
            "newly detected events into the splits overrides CSV."
        ),
    )
    ap.add_argument(
        "--audit-window",
        type=int,
        default=5,
        help="--audit-splits lookback in trading days (default: 5)",
    )
    ap.add_argument(
        "--symbols",
        default=None,
        help=(
            "Comma-separated subset of tickers for --audit-splits. "
            "Default: full screener universe."
        ),
    )
    ap.add_argument(
        "--write-overrides",
        action="store_true",
        help=(
            "When set with --audit-splits, append newly-detected events "
            "(not already in splits_overrides.csv) into that file."
        ),
    )
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

    # ── Splits override paths (multi-source corporate-action handling) ──
    # Both paths are optional. Missing files → corresponding source is silent
    # and the heuristic + Yahoo events keep working as before.
    paths_cfg = cfg.get("paths", {}) or {}

    def _resolve_data_path(rel: str | None, default_rel: str) -> Path:
        """Resolve a config-driven data path relative to the script root."""
        s = (rel or default_rel).strip()
        p = Path(s)
        if not p.is_absolute():
            p = script_dir / p
        return p

    global _SPLITS_OVERRIDES_CSV_PATH, _FLEX_SPLITS_CSV_PATH
    _SPLITS_OVERRIDES_CSV_PATH = _resolve_data_path(
        paths_cfg.get("splits_overrides_csv"),
        "data/splits_overrides.csv",
    )
    _FLEX_SPLITS_CSV_PATH = _resolve_data_path(
        paths_cfg.get("flex_splits_csv"),
        "data/splits_from_flex.csv",
    )
    if _SPLITS_OVERRIDES_CSV_PATH.is_file():
        print(f"[SPLITS] overrides CSV: {_SPLITS_OVERRIDES_CSV_PATH}")
    else:
        print(f"[SPLITS] overrides CSV: {_SPLITS_OVERRIDES_CSV_PATH} (absent — opt-in)")
    if _FLEX_SPLITS_CSV_PATH.is_file():
        print(f"[SPLITS] flex CSV:      {_FLEX_SPLITS_CSV_PATH}")
    else:
        print(f"[SPLITS] flex CSV:      {_FLEX_SPLITS_CSV_PATH} (absent — opt-in)")

    # ── Audit splits (read-only mode) ──
    if args.audit_splits:
        return _run_audit_splits(args)

    # Protected ETFs from config
    wl_list = sleeves_cfg.get("whitelist_stock", {}).get("universe", {}).get("etfs", []) or []
    flow_list = sleeves_cfg.get("flow_program", {}).get("universe", {}).get("shorts", []) or []
    protected = {_norm_sym(x) for x in (list(wl_list) + list(flow_list)) if str(x).strip()}
    blacklist = load_strategy_blacklist(cfg, base_dir=script_dir)
    if blacklist:
        print(f"[CONFIG] blacklist={len(blacklist)} symbol(s)")

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

    borrow_hist_path = args.borrow_history_path or os.environ.get("BORROW_HISTORY_PATH")
    if not borrow_hist_path:
        auto_bh = discover_default_borrow_history_json()
        if auto_bh is not None:
            borrow_hist_path = str(auto_bh)
            print(f"[BORROW] Auto-discovered borrow history: {borrow_hist_path}")
    borrow_history_map = None
    if borrow_hist_path:
        bh = Path(borrow_hist_path).expanduser()
        if bh.is_file():
            try:
                borrow_history_map = load_borrow_history_json(bh)
                print(
                    f"[BORROW] Loaded weighted-resample history: {bh} "
                    f"({len(borrow_history_map)} symbols)"
                )
            except (OSError, ValueError, json.JSONDecodeError) as e:
                print(f"[BORROW] ⚠ Failed to load borrow history ({e}); using point-in-time borrow")
                borrow_history_map = None
        else:
            print(f"[BORROW] ⚠ borrow history path not found: {bh}")

    try:
        asof_for_v2 = date.fromisoformat(str(run_date)[:10])
    except ValueError:
        asof_for_v2 = date.today()

    # ── Step 1: Build universe ──
    print("\n" + "─" * 70)
    print("STEP 1 — Build Universe")
    print("─" * 70)
    universe = build_full_universe(skip_scrape=args.skip_scrape, skip_inverse=args.skip_inverse)
    if blacklist and not universe.empty:
        und_blk = universe["Underlying"].map(
            lambda x: _norm_sym(x) if pd.notna(x) and str(x).strip() else ""
        )
        n_blk = int((universe["ETF"].astype(str).map(_norm_sym).isin(blacklist) | und_blk.isin(blacklist)).sum())
        if n_blk:
            print(
                f"[UNIVERSE] {n_blk} row(s) touch strategy blacklist "
                f"(kept in CSV for dashboards; excluded from trade plan / rebalancer only)."
            )

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
    print(f"[SCREEN] Row count after borrow diagnostics: {len(screened)}")

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

    # Step 5b — Distributional forecast of gross decay. Runs *before* the
    # schema-v2 enrichment so the bootstrap below can anchor-shift its
    # realized-drag draws onto the model-based ``expected_gross_decay_p50_annual``.
    # LETF / Inverse rows go through the HARQ-Log empirical-lognormal mapping;
    # YieldBOOST rows dispatch into the put-spread Monte Carlo
    # (yieldboost_decay.yieldboost_decay_distribution). See
    # decay_distribution.py and yieldboost_decay.py for the models.
    screened = enrich_with_decay_distribution(
        screened,
        tr_map,
        horizon_days=TRADING_DAYS,
        norm_sym=_norm_sym,
    )

    # Step 5c — Schema v2 (product_class, gross_edge_definition, anchor-shifted
    # net-edge bootstrap). The bootstrap reads ``expected_gross_decay_p50_annual``
    # from Step 5b to forward-anchor its realized block-bootstrap draws; rows
    # without a meaningful expected forecast (passive_low_beta) skip the shift
    # via the ``expected_decay_available`` gate inside enrich_screener_v2_fields.
    screened = enrich_screener_v2_fields(
        screened,
        tr_map,
        min_days=min_beta_days,
        borrow_history_map=borrow_history_map,
        borrow_weight_halflife_days=float(args.borrow_weight_halflife_days),
        asof_date=asof_for_v2,
    )
    screened = apply_volatility_etp_expected_decay_adjustment(screened)

    # Step 5d — Apply the "expected decay = N/A" policy for passive low-β
    # rows. The simple Itô identity says (β² − β)/2·σ² ≈ 0 around β ≈ 1, so
    # any "expected decay" we report there is at best noise. The distributional
    # forecast inherits the same problem (the cb factor multiplies the
    # lognormal IV quantiles by ~0). For the dashboard to consistently render
    # "—" for passive_low_beta and fall back to the realized measure, we
    # null-out the entire family of expected/distributional decay columns
    # for those rows here. See screener_v2_fields._product_class for the
    # taxonomy and _expected_decay_available for the policy.
    if "product_class" in screened.columns:
        passive_mask = screened["product_class"].astype(str).eq("passive_low_beta")
    else:
        passive_mask = pd.Series(False, index=screened.index)
    if passive_mask.any():
        passive_null_cols = [
            "expected_gross_decay_annual",
            "expected_gross_decay_annual_legacy",
            "expected_gross_decay_adjusted_annual",
            "expected_gross_decay_simple_ito_annual",
            "expected_decay_adjustment_annual",
            "blended_gross_decay",
            "expected_gross_decay_p10_annual",
            "expected_gross_decay_p50_annual",
            "expected_gross_decay_p90_annual",
            "expected_gross_decay_mean_annual",
            "expected_logIV_mu_annual",
            "expected_logIV_sigma_annual",
            "expected_gross_decay_dist_n_obs",
            "expected_gross_decay_dist_horizon_days",
            "mechanical_decay_annual",
        ]
        for col in passive_null_cols:
            if col in screened.columns:
                screened.loc[passive_mask, col] = np.nan
        if "expected_gross_decay_dist_model" in screened.columns:
            screened.loc[passive_mask, "expected_gross_decay_dist_model"] = (
                "passive_low_beta_na"
            )
        if "expected_gross_decay_reliable" in screened.columns:
            screened.loc[passive_mask, "expected_gross_decay_reliable"] = False
        n_passive = int(passive_mask.sum())
        print(
            f"[DECAY-DIST] passive_low_beta policy applied: nulled expected "
            f"decay columns on {n_passive} row(s); dashboard falls back to realized."
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

    # Recompute the vol-ratio gate with the YAML-resolved per-leverage band.
    # ``enrich_with_decay_and_vol`` already populated the column with default
    # bounds; this pass refreshes ``vol_ratio_outlier`` using the operator's
    # configured limits before purgatory consumes it.
    screened = recompute_vol_ratio_gate(screened, screener_cfg=screener_cfg)

    screened = recompute_purgatory_by_bucket(
        screened, screener_cfg=screener_cfg, sleeves_cfg=sleeves_cfg
    )
    purg = int(screened["purgatory"].sum())
    print(f"[SCREEN] Purgatory (per-bucket borrow bands + vol-ratio gate): {purg} | Total: {len(screened)}")

    if blacklist:
        etf_n = screened["ETF"].astype(str).map(_norm_sym)
        und_n = screened["Underlying"].map(
            lambda x: _norm_sym(x) if pd.notna(x) and str(x).strip() else ""
        )
        screened = screened.copy()
        screened["strategy_blacklisted"] = (etf_n.isin(blacklist) | und_n.isin(blacklist)).astype(bool)
    else:
        screened["strategy_blacklisted"] = False

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