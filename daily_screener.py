#!/usr/bin/env python3
"""
daily_screener.py — Single-file daily pipeline for leveraged ETF pair strategy.

Consolidates:  universe building → beta calculation → borrow fetch → screening → decay/vol → CSV

Replaces the multi-notebook workflow:
  1. etf_screener.py notebook cells (pair definitions, YieldMax/Roundhill scraping)
  2. etf_analytics.py notebook cells (beta via OLS)
  3. etf_screener.py CLI (FTP borrow, screening logic)
  4. etf_analytics.py (decay + volatility enrichment)

Run daily:
  python daily_screener.py                           # full pipeline, output → data/etf_screened_today.csv
  python daily_screener.py --skip-scrape             # skip YieldMax/Roundhill scraping (use cached)
  python daily_screener.py --skip-inverse            # skip inverse ETF universe (bucket C)
  python daily_screener.py --lookback 1y             # shorter history for faster runs
  python daily_screener.py --output my_output.csv    # custom output path

If ibkr_margin_requirements.csv exists in data/, it will be merged in automatically.
"""
from __future__ import annotations

import argparse
import builtins
import ftplib
import io
import os
import re
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Iterable, Set

import numpy as np
import pandas as pd
import requests
import yaml
import yfinance as yf

warnings.filterwarnings("ignore", category=FutureWarning)

TRADING_DAYS = 252


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
    ("YINN", "FXI"),  ("MEXX", "EWW"),  ("KORU", "EWY"),  ("HIBL", "SPY"),
    ("LABU", "XBI"),  ("SOXL", "SOXX"),
    # --- 2X Thematic / Equity --
    ("CHAU", "ASHR"), ("CWEB", "KWEB"), ("ERX",  "XLE"),  ("NUGT", "GDX"),
    ("JNUG", "GDXJ"), ("GUSH", "XOP"), ("URAA", "URA"),
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
    ("LITX", "LITE"), ("SNXX", "SNDX"), ("WDCX", "WDC"),  ("LNOK", "NOK"),
    ("USGG", "USAR"), ("ONDG", "ONDS"), ("PLUL", "PLUG"), ("ALBG", "ALB"),
    ("UUUG", "UUUU"), ("HUTG", "HUT"),  ("XPEG", "XPEV"), ("ORLG", "ORLY"),
    ("LUNL", "LUNR"), ("RKTL", "RKT"),  ("EOSU", "EOSE"), ("BTFL", "BITF"),
    ("FGRU", "FIGR"), ("APHU", "APH"),  ("COPZ", "COPX"), ("LEUX", "LEU"),
    ("COHX", "COHR"), ("AXPG", "AXP"),  ("FCXG", "FCX"),
    ("BITX", "IBIT"), ("ETHU", "ETHA"), ("XXRP", "XRPZ"),
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

covered_call_pairs = [
    ("QYLD", "QQQ"),  ("QYLG", "QQQ"),  ("QQQX", "QQQ"),  ("JEPQ", "QQQ"),
    ("XYLD", "SPY"),  ("XYLG", "SPY"),  ("JEPI", "SPY"),  ("SPYI", "SPY"),
    ("RYLD", "IWM"),
]


# ── Inverse ETF Universe (Bucket C) ──
BENCHMARK_MAP = {
    "SPX": "SPY",  "NDX": "QQQ",   "DJIA": "DIA",  "RUT": "IWM",
    "SOX": "SOXX", "FIN": "XLF",   "BIOTECH": "XBI","TECH": "XLK",
    "WTI": "USO",  "TSLA": "TSLA", "MSTR": "MSTR", "NVDA": "NVDA",
    "BTC": "IBIT", "ETH": "ETHA",  "CRCL": "CRCL", "CRWV": "CRWV",
    "GDX": "GDX",  "SLV": "SLV",   "XLE": "XLE",   "XOP": "XOP",
    "TLT": "TLT",  "MSCIJP": "EWJ","APLD": "APLD", "CLSK": "CLSK",
}

INVERSE_ETF_UNIVERSE = [
    ("SDS",  -2, "SPX"),  ("QID",  -2, "NDX"),  ("DXD",  -2, "DJIA"), ("TWM",  -2, "RUT"),
    ("SCO",  -2, "WTI"),  ("MSTZ", -2, "MSTR"), ("NVDQ", -2, "NVDA"), ("BTCZ", -2, "BTC"),
    ("ETHD", -2, "ETH"),  ("CRCD", -2, "CRCL"), ("CORD", -2, "CRWV"), ("TSLQ", -2, "TSLA"),
    ("ZSL",  -2, "SLV"),  ("SQQQ", -3, "NDX"),  ("SPXS", -3, "SPX"),  ("TZA",  -3, "RUT"),
    ("SOXS", -3, "SOX"),  ("FAZ",  -3, "FIN"),   ("LABD", -3, "BIOTECH"),
    ("TECS", -3, "TECH"), ("SDOW", -3, "DJIA"), ("DUST", -3, "GDX"),
    ("TTXD", -2, "TECH"), ("TSXD", -2, "SOX"),  ("WEBS", -3, "TECH"), ("FNGD", -3, "TECH"),
    ("REW",  -2, "TECH"), ("SKF",  -2, "FIN"),   ("SPXU", -3, "SPX"),
    ("DUG",  -2, "XLE"),  ("DRIP", -2, "XOP"),  ("TMV",  -3, "TLT"),
    ("TBT",  -2, "TLT"),  ("TBX",  -2, "TLT"),  ("NVDS", -1.5, "NVDA"),
    ("EWV",  -2, "MSCIJP"), ("APLZ", -2, "APLD"),
    ("HIBS", -3, "SPX"),  ("CLSZ", -2, "CLSK"),
]


# ══════════════════════════════════════════════════════════════════
# SECTION 2 — UNIVERSE BUILDING
# ══════════════════════════════════════════════════════════════════

def _norm_sym(x) -> str:
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


def _scrape_yieldmax() -> pd.DataFrame:
    """Scrape YieldMax single-stock option income ETFs → DataFrame(ETF, Underlying)."""
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 Chrome/120.0.0.0"})
    url = "https://yieldmaxetfs.com/nav-group/yieldmax-single-stock-option-income-etfs/"
    try:
        resp = session.get(url, timeout=15)
        resp.raise_for_status()
    except Exception as e:
        print(f"[WARN] YieldMax scrape failed: {e}")
        return pd.DataFrame(columns=["ETF", "Underlying"])

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
    return pd.DataFrame(rows, columns=["ETF", "Underlying"]).drop_duplicates()


def _scrape_roundhill() -> pd.DataFrame:
    """Scrape Roundhill WeeklyPay ETFs → DataFrame(ETF, Underlying)."""
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
        return pd.DataFrame(columns=["ETF", "Underlying"])

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
    return pd.DataFrame(rows, columns=["ETF", "Underlying"]).drop_duplicates()


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

    # 2. Covered call pairs (1x)
    cc_df = pd.DataFrame(covered_call_pairs, columns=["ETF", "Underlying"])
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

def _get_total_return_series(ticker: str, period: str = "2y") -> pd.Series:
    """Build total-return price series: TR_t = TR_{t-1} × (Close_t + Div_t) / Close_{t-1}."""
    try:
        t = yf.Ticker(ticker)
        df = t.history(period=period, auto_adjust=False, actions=True)
        if df.empty or "Close" not in df.columns:
            return pd.Series(dtype=float, name=ticker)
        close = df["Close"]
        divs = df.get("Dividends", pd.Series(0.0, index=df.index)).reindex(close.index, fill_value=0.0)
        rel = (close + divs) / close.shift(1)
        rel.iloc[0] = 1.0
        tr = close.iloc[0] * rel.cumprod()
        tr.name = ticker
        return tr
    except Exception:
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


def compute_beta_ols(etf_tr: pd.Series, und_tr: pd.Series,
                     min_days: int = 60) -> tuple[float | None, int]:
    """OLS: r_etf = alpha + beta × r_und. Returns (beta, n_obs)."""
    df = pd.concat([etf_tr.rename("etf"), und_tr.rename("und")], axis=1).dropna()
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

    # Drop bad tickers
    bad = {"JP", "JPO"}
    mask = ~(df["ETF"].isin(bad) | df["Underlying"].isin(bad))
    dropped = len(df) - mask.sum()
    if dropped > 0:
        print(f"[BETA] Dropped {dropped} rows with bad tickers")
    df = df[mask].reset_index(drop=True)

    ols_count = (df["Beta_source"] == "ols").sum()
    imp_count = df["Beta_source"].str.startswith("imputed").sum()
    print(f"[BETA] OLS: {ols_count} | Imputed: {imp_count} | Total: {len(df)}")
    return df


# ══════════════════════════════════════════════════════════════════
# SECTION 4 — IBKR FTP BORROW DATA
# ══════════════════════════════════════════════════════════════════

FTP_HOST = os.getenv("IBKR_FTP_HOST") or "ftp2.interactivebrokers.com"
FTP_USER = os.getenv("IBKR_FTP_USER") or "shortstock"
FTP_PASS = os.getenv("IBKR_FTP_PASS") or ""
FTP_FILE = os.getenv("IBKR_FTP_FILE") or "usa.txt"


def fetch_ibkr_shortstock_file(filename: str = FTP_FILE) -> pd.DataFrame:
    """Download and parse IBKR short stock availability FTP file."""
    print(f"[FTP] Connecting to {FTP_HOST} ...")
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
    print(f"[FTP] Parsed {len(df)} rows")
    return df


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
    for req in ("sym", "rebaterate", "feerate"):
        if req not in short_df.columns:
            raise ValueError(f"Expected '{req}' column; got: {list(short_df.columns)}")
    df = short_df.copy()
    df["sym"] = df["sym"].astype(str).str.upper().str.strip()
    df["borrow_fee_annual"] = df["feerate"].map(_parse_rate_to_decimal)
    df["borrow_rebate_annual"] = df["rebaterate"].map(_parse_rate_to_decimal)
    df["available_int"] = pd.to_numeric(df.get("available", 0), errors="coerce").fillna(0)
    df["borrow_net_annual"] = df["borrow_fee_annual"] - df["borrow_rebate_annual"]
    m = df["borrow_net_annual"].notna()
    df.loc[m, "borrow_net_annual"] = df.loc[m, "borrow_net_annual"].clip(lower=0)

    agg = df.groupby("sym", as_index=False).agg(
        borrow_fee_annual=("borrow_fee_annual", "max"),
        borrow_rebate_annual=("borrow_rebate_annual", "max"),
        borrow_net_annual=("borrow_net_annual", "max"),
        shares_available=("available_int", "max"),
    ).rename(columns={"sym": "ETF"})
    agg["borrow_current"] = agg["borrow_net_annual"]
    agg["borrow_spiking"] = False
    agg["borrow_missing_from_ftp"] = False

    present = set(agg["ETF"].values)
    missing = [s for s in etf_list if s not in present]
    if missing:
        missing_df = pd.DataFrame({
            "ETF": missing, "borrow_fee_annual": np.nan, "borrow_rebate_annual": np.nan,
            "borrow_net_annual": np.nan, "shares_available": 0, "borrow_current": np.nan,
            "borrow_spiking": False, "borrow_missing_from_ftp": True,
        })
        agg = pd.concat([agg, missing_df], ignore_index=True)
    return agg.drop_duplicates(subset=["ETF"], keep="first").reset_index(drop=True)


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
    """Apply borrow-based screening: include_for_algo, purgatory, protected logic."""
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

    df["borrow_leq_low"] = df["borrow_current"].notna() & (df["borrow_current"] <= params.borrow_low)
    df["borrow_gt_low"] = ~df["borrow_leq_low"]

    hard_cap = float(params.hard_borrow_cap)
    df["protected_ok"] = (df["protected"] & df["borrow_current"].notna()
                          & (~df["borrow_missing_from_ftp"]) & (df["borrow_current"] <= hard_cap))
    df["protected_bad"] = df["protected"] & (~df["protected_ok"])

    include_base = df["borrow_leq_low"] | df["protected_ok"]
    if params.exclude_negative_cagr and "cagr_port_hist" in df.columns:
        include_base = include_base & (df["cagr_positive"] == True)
    df["include_for_algo"] = include_base

    normal_purg = (df["borrow_current"].notna() & (~df["borrow_missing_from_ftp"])
                   & (df["borrow_current"] > params.borrow_low)
                   & (df["borrow_current"] <= (params.borrow_low + params.purgatory_margin))
                   & (~df["protected"]))
    df["purgatory"] = normal_purg | df["protected_bad"]

    df["exclude_borrow_gt_low"] = df["borrow_gt_low"] & (~df["protected_ok"])
    df["exclude_no_shares"] = df["shares_available"] < params.min_shares_available
    df["exclude_borrow_spike"] = df["borrow_spiking"].fillna(False)
    return df


# ══════════════════════════════════════════════════════════════════
# SECTION 6 — DECAY + VOLATILITY ENRICHMENT
# ══════════════════════════════════════════════════════════════════

def _annualized_vol(tr_series: pd.Series, min_days: int = 60) -> float | None:
    tr = tr_series.dropna()
    if len(tr) < min_days + 1: return None
    ret = tr.pct_change().dropna()
    if len(ret) < min_days: return None
    return round(float(ret.std() * np.sqrt(TRADING_DAYS)), 6)


def _compute_gross_decay(etf_tr: pd.Series, und_tr: pd.Series,
                         beta: float, min_days: int = 60) -> float | None:
    """
    Gross annualized decay per $1 ETF short.
    Bull (β>0): simple returns |β|×r_und − r_etf
    Inverse (β<0): log returns β×ln(1+r_und) − ln(1+r_etf)
    """
    df = pd.concat([etf_tr.rename("etf"), und_tr.rename("und")], axis=1).dropna()
    if len(df) < min_days + 1: return None
    abs_beta = abs(float(beta))
    if abs_beta < 0.1: return None

    if beta > 0:
        r_etf = df["etf"].pct_change()
        r_und = df["und"].pct_change()
        valid = r_etf.notna() & r_und.notna()
        r_etf, r_und = r_etf[valid], r_und[valid]
        if len(r_etf) < min_days: return None
        daily_pnl = abs_beta * r_und - r_etf
    else:
        r_etf = np.log(df["etf"] / df["etf"].shift(1))
        r_und = np.log(df["und"] / df["und"].shift(1))
        valid = r_etf.notna() & r_und.notna() & np.isfinite(r_etf) & np.isfinite(r_und)
        r_etf, r_und = r_etf[valid], r_und[valid]
        if len(r_etf) < min_days: return None
        daily_pnl = float(beta) * r_und - r_etf

    return round(float(daily_pnl.mean()) * TRADING_DAYS, 6)


def enrich_with_decay_and_vol(df: pd.DataFrame, tr_map: dict[str, pd.Series],
                              min_days: int = 60) -> pd.DataFrame:
    """Add vol + decay columns using pre-downloaded TR series."""
    print("[DECAY] Computing decay + volatility ...")
    vols_und, vols_etf, decays = [], [], []
    betas_out, beta_nobs_out = [], []
    ok_decay = ok_vol = betas_computed = 0

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

        vol_und = _annualized_vol(tr_map[und], min_days) if und and und in tr_map else None
        vol_etf = _annualized_vol(tr_map[etf], min_days) if etf in tr_map else None
        vols_und.append(vol_und)
        vols_etf.append(vol_etf)
        if vol_und is not None: ok_vol += 1

        decay = None
        if beta_f and abs(beta_f) >= 0.1 and und and etf in tr_map and und in tr_map:
            decay = _compute_gross_decay(tr_map[etf], tr_map[und], beta_f, min_days)
        decays.append(decay)
        if decay is not None: ok_decay += 1

    df["Beta"] = betas_out
    df["Beta_n_obs"] = beta_nobs_out
    df["vol_underlying_annual"] = vols_und
    df["vol_etf_annual"] = vols_etf
    df["gross_decay_annual"] = decays

    borrow_net = pd.to_numeric(df.get("borrow_net_annual"), errors="coerce")
    df["net_decay_annual"] = np.where(
        df["gross_decay_annual"].notna() & borrow_net.notna(),
        df["gross_decay_annual"] - borrow_net, np.nan)

    print(f"[DECAY] Beta OLS fill-ins: {betas_computed} | Vol: {ok_vol}/{len(df)} | Decay: {ok_decay}/{len(df)}")
    return df


# ══════════════════════════════════════════════════════════════════
# SECTION 7 — MARGIN ENRICHMENT (optional, from fetch_ibkr_margin.py output)
# ══════════════════════════════════════════════════════════════════

def merge_margin_data(df: pd.DataFrame, margin_csv: Path) -> pd.DataFrame:
    """
    If ibkr_margin_requirements.csv exists, merge maint_pct columns.
    Also merges pair-level margin if ibkr_pair_margin.csv exists.
    """
    if not margin_csv.exists():
        print(f"[MARGIN] No margin file found at {margin_csv} — skipping")
        return df

    margin_df = pd.read_csv(margin_csv)
    print(f"[MARGIN] Loaded {len(margin_df)} symbols from {margin_csv}")

    # Build symbol → maint_pct lookup
    margin_lookup = margin_df.set_index("symbol")[
        ["maint_pct_long", "maint_pct_short", "init_pct_long", "init_pct_short"]
    ].to_dict("index")

    # For each pair: short ETF margin + long underlying margin
    pair_maint = []
    for _, row in df.iterrows():
        etf = _norm_sym(row["ETF"])
        und = _norm_sym(row["Underlying"]) if pd.notna(row.get("Underlying")) else None
        beta = float(row["Beta"]) if pd.notna(row.get("Beta")) else np.nan

        etf_short = margin_lookup.get(etf, {}).get("maint_pct_short", np.nan)
        und_long = margin_lookup.get(und, {}).get("maint_pct_long", np.nan) if und else np.nan

        if np.isfinite(etf_short) and np.isfinite(und_long) and np.isfinite(beta):
            hr = 1.0 / max(abs(beta), 0.5)
            pair_maint.append(round(etf_short + hr * und_long, 6))
        else:
            pair_maint.append(np.nan)

    df["maint_pct_short_etf"] = [margin_lookup.get(_norm_sym(r["ETF"]), {}).get("maint_pct_short", np.nan)
                                  for _, r in df.iterrows()]
    df["maint_pct_long_und"] = [margin_lookup.get(_norm_sym(r["Underlying"]), {}).get("maint_pct_long", np.nan)
                                 if pd.notna(r.get("Underlying")) else np.nan for _, r in df.iterrows()]
    df["pair_maint_pct"] = pair_maint

    valid = df["pair_maint_pct"].dropna()
    if len(valid) > 0:
        print(f"[MARGIN] Pair maint range: {valid.min()*100:.1f}% – {valid.max()*100:.1f}%  "
              f"(median {valid.median()*100:.1f}%)")
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
    ap.add_argument("--min-beta-days", type=int, default=None,
                    help="Min overlapping days for OLS beta (default: from config or 20)")
    ap.add_argument("--config", default=None, help="Path to strategy_config.yml")
    ap.add_argument("--margin-csv", default="data/ibkr_margin_requirements.csv",
                    help="Path to margin requirements CSV (from fetch_ibkr_margin.py)")
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
    print(f"[CONFIG] borrow_low={params.borrow_low:.2%}  hard_cap={params.hard_borrow_cap:.2%}  "
          f"protected={len(protected)}")

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
    universe = add_betas(universe, tr_map, min_days=min_beta_days)

    # Save intermediate beta file
    beta_csv = dated_dir / "all_pairs_with_betas.csv"
    universe.to_csv(beta_csv, index=False)
    print(f"[BETA] Saved: {beta_csv}")

    # ── Step 3: Fetch borrow data ──
    print("\n" + "─" * 70)
    print("STEP 3 — Fetch IBKR Borrow Data")
    print("─" * 70)
    if args.skip_ftp:
        print("[FTP] Skipped (--skip-ftp)")
        borrow_df = pd.DataFrame({"ETF": universe["ETF"].unique()})
        borrow_df["borrow_net_annual"] = np.nan
        borrow_df["borrow_current"] = np.nan
        borrow_df["shares_available"] = 0
        borrow_df["borrow_missing_from_ftp"] = True
        borrow_df["borrow_spiking"] = False
    else:
        # Ensure protected ETFs are in universe
        missing_prot = sorted([t for t in protected if t not in set(universe["ETF"].values)])
        if missing_prot:
            add_rows = pd.DataFrame({"ETF": missing_prot, "Underlying": None, "Leverage": np.nan})
            universe = pd.concat([universe, add_rows], ignore_index=True)
            universe = universe.drop_duplicates(subset=["ETF"]).reset_index(drop=True)
            print(f"[FTP] Added {len(missing_prot)} protected ETFs to universe")

        borrow_df = get_ibkr_borrow_snapshot(universe["ETF"].unique())

    metrics = universe.merge(borrow_df, on="ETF", how="left")

    # ── Step 4: Screen ──
    print("\n" + "─" * 70)
    print("STEP 4 — Apply Screening Logic")
    print("─" * 70)
    screened = screen_universe(metrics, params)
    included = int(screened["include_for_algo"].sum())
    purg = int(screened["purgatory"].sum())
    print(f"[SCREEN] Included: {included} | Purgatory: {purg} | Total: {len(screened)}")

    # ── Step 5: Decay + vol ──
    print("\n" + "─" * 70)
    print("STEP 5 — Compute Decay + Volatility")
    print("─" * 70)
    screened = enrich_with_decay_and_vol(screened, tr_map, min_days=min_beta_days)

    # ── Step 6: Margin enrichment (optional) ──
    margin_path = Path(args.margin_csv)
    if margin_path.exists():
        print("\n" + "─" * 70)
        print("STEP 6 — Merge Margin Requirements")
        print("─" * 70)
        screened = merge_margin_data(screened, margin_path)

    # ── Drop helper columns, save ──
    drop_cols = ["Leverage", "ExpectedLeverage"]
    screened = screened.drop(columns=[c for c in drop_cols if c in screened.columns])

    screened.to_csv(dated_dir / "etf_screened_today.csv", index=False)
    screened.to_csv(output_path, index=False)
    print(f"\n{'=' * 70}")
    print(f"[DONE] Saved: {output_path}")
    print(f"[DONE] Dated: {dated_dir / 'etf_screened_today.csv'}")
    print(f"[DONE] {len(screened)} pairs | {included} included | {purg} purgatory")

    # Summary stats
    valid_net = screened["net_decay_annual"].dropna()
    if len(valid_net) > 0:
        print(f"[DONE] Net decay: median={valid_net.median()*100:.2f}%  "
              f"range=[{valid_net.min()*100:.2f}%, {valid_net.max()*100:.2f}%]")

    top5 = screened[screened["net_decay_annual"].notna()].nlargest(5, "net_decay_annual")
    if len(top5) > 0:
        print(f"\n  Top 5 by net decay:")
        for _, r in top5.iterrows():
            bn = r.get("borrow_net_annual")
            print(f"    {r['ETF']:8s}  net={r['net_decay_annual']*100:6.2f}%  "
                  f"gross={r['gross_decay_annual']*100:6.2f}%  "
                  f"borrow={bn*100 if pd.notna(bn) else 0:5.2f}%  β={r['Beta']:.2f}")

    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
