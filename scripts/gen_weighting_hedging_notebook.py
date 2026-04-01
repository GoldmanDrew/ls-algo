#!/usr/bin/env python3
"""
Generate the weighting & hedging strategy backtest notebook.
Run: python scripts/gen_weighting_hedging_notebook.py
Output: notebooks/weighting_hedging_backtest.ipynb
"""

import json
from pathlib import Path


def md_cell(source: str) -> dict:
    lines = source.strip().split("\n")
    formatted = [l + "\n" for l in lines[:-1]] + [lines[-1]] if lines else []
    return {"cell_type": "markdown", "metadata": {}, "source": formatted}


def code_cell(source: str) -> dict:
    lines = source.strip().split("\n")
    formatted = [l + "\n" for l in lines[:-1]] + [lines[-1]] if lines else []
    return {
        "cell_type": "code",
        "metadata": {},
        "source": formatted,
        "outputs": [],
        "execution_count": None,
    }


def build_notebook() -> dict:
    cells = []

    # ==================================================================
    # CELL: Title
    # ==================================================================
    cells.append(md_cell("""# Weighting & Hedging Strategy Backtest
## Bucket 1: Beta > 1.5 (Positive) Leveraged ETF Pairs

This notebook tests:
1. **Weighting strategies**: Equal weight vs decay-score vs inverse-borrow weighting
2. **Borrow threshold sensitivity**: How inclusion thresholds affect returns
3. **Hedging / rebalance strategies**: D/W/M/Q/Never and drift-based rebalancing
4. **Hypothesis tests**:
   - H₀: Equal weighting ≥ decay-score weighting (Sharpe)
   - H₀: Weekly rebalance to 0 net exposure is optimal

Backtest period: 2023-01-01 to present"""))

    # ==================================================================
    # CELL: Imports
    # ==================================================================
    cells.append(code_cell("""import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

plt.style.use("default")
np.random.seed(42)

TRADING_DAYS = 252
DEFAULT_BORROW_ANNUAL = 0.05
MARGIN_ANNUAL = 0.04428
MARGIN_DAYCOUNT = 360
BACKTEST_START = "2023-01-01"

repo_root = Path.cwd().parent
data_dir = repo_root / "data"
"""))

    # ==================================================================
    # CELL: Section 1 header
    # ==================================================================
    cells.append(md_cell("## 1. Data Loading & Filtering"))

    # ==================================================================
    # CELL: Load and filter
    # ==================================================================
    cells.append(code_cell("""# Load screened universe
screen_df = pd.read_csv(data_dir / "etf_screened_today.csv")

# Normalize tickers
for col in ["ETF", "Underlying"]:
    screen_df[col] = (
        screen_df[col].astype(str).str.strip()
        .str.replace(".", "-", regex=False).str.upper()
    )

screen_df["Beta"] = pd.to_numeric(screen_df["Beta"], errors="coerce")
for c in ["borrow_current", "borrow_fee_annual", "blended_gross_decay", "decay_score"]:
    if c in screen_df.columns:
        screen_df[c] = pd.to_numeric(screen_df[c], errors="coerce")

if "borrow_current" not in screen_df.columns and "borrow_fee_annual" in screen_df.columns:
    screen_df["borrow_current"] = screen_df["borrow_fee_annual"]

# Bucket 1: Beta > 1.5, positive, include_for_algo
bucket1 = screen_df[
    (screen_df["include_for_algo"] == True) &
    (screen_df["Beta"] > 1.5)
].copy().reset_index(drop=True)

print(f"Full universe: {len(screen_df)} rows")
print(f"Bucket 1 (Beta > 1.5, positive, included): {len(bucket1)} pairs")
print(f"Beta range: {bucket1['Beta'].min():.2f} - {bucket1['Beta'].max():.2f}")
print(f"Borrow range: {bucket1['borrow_current'].min():.3f} - {bucket1['borrow_current'].max():.3f}")
print()
bucket1[["ETF", "Underlying", "Beta", "borrow_current",
         "blended_gross_decay", "decay_score"]].sort_values("Beta", ascending=False)
"""))

    # ==================================================================
    # CELL: Download prices
    # ==================================================================
    cells.append(code_cell("""import yfinance as yf

def _norm_sym(x: str) -> str:
    return str(x).upper().replace(".", "-").strip()

def build_prices_tr_from_yf(tickers, start="1900-01-01"):
    tickers = [_norm_sym(t) for t in tickers]
    df = yf.download(
        tickers, start=start, auto_adjust=False, progress=True,
        group_by="column", actions=False, threads=True,
    )
    if isinstance(df.columns, pd.MultiIndex):
        px = df.get("Adj Close", df.get("Close")).copy()
    else:
        px = df.get("Adj Close", df.get("Close")).to_frame()
    px.index = pd.to_datetime(px.index).tz_localize(None).normalize()
    px.columns = [f"{_norm_sym(c)}_TR" for c in px.columns]
    return px.sort_index()

# Collect all tickers
all_tickers = sorted(set(bucket1["ETF"].tolist() + bucket1["Underlying"].tolist()))
print(f"Downloading {len(all_tickers)} tickers...")

prices = build_prices_tr_from_yf(all_tickers, start="2022-01-01")
print(f"Price data: {prices.index.min().date()} -> {prices.index.max().date()}, {len(prices)} rows")
"""))

    # ==================================================================
    # CELL: Build borrow maps
    # ==================================================================
    cells.append(code_cell("""def make_borrow_daily_map(screen_pass, etf_col="ETF",
                         borrow_col="borrow_current",
                         default_annual=DEFAULT_BORROW_ANNUAL):
    out = {}
    for _, row in screen_pass.iterrows():
        etf = _norm_sym(row[etf_col])
        ann = row.get(borrow_col, np.nan)
        if not np.isfinite(ann):
            ann = default_annual
        out[etf] = float(ann) / TRADING_DAYS
    return out

borrow_daily_map = make_borrow_daily_map(bucket1)
print(f"Borrow map: {len(borrow_daily_map)} ETFs")
for etf, bd in sorted(borrow_daily_map.items(), key=lambda x: x[1], reverse=True)[:10]:
    print(f"  {etf}: {bd * 252:.1%} annual")
"""))

    # ==================================================================
    # CELL: Section 2 header
    # ==================================================================
    cells.append(md_cell("## 2. Backtest Infrastructure"))

    # ==================================================================
    # CELL: Utility functions
    # ==================================================================
    cells.append(code_cell("""def _to_naive_utc_index(idx):
    idx = pd.to_datetime(idx)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    return idx

def get_rebalance_days(idx, freq):
    f = str(freq).upper().strip()
    idx = _to_naive_utc_index(pd.DatetimeIndex(idx))
    if len(idx) == 0:
        return set()
    days = {pd.Timestamp(idx[0])}
    if f in ("NEVER",):
        return days
    if f in ("D", "DAILY"):
        return set(pd.Timestamp(x) for x in idx)
    if f not in ("W", "M", "Q"):
        raise ValueError(f"freq must be D/W/M/Q/NEVER, got {freq}")
    s = pd.Series(1, index=idx)
    grouper = {"W": "W-FRI", "M": "ME", "Q": "QE"}[f]
    for _, grp in s.groupby(pd.Grouper(freq=grouper)):
        if len(grp.index) > 0:
            days.add(grp.index[-1])
    return days

def build_pair_start_dates(px, pairs):
    out = {}
    for und, etf in pairs:
        und_col = f"{_norm_sym(und)}_TR"
        etf_col = f"{_norm_sym(etf)}_TR"
        if und_col not in px.columns or etf_col not in px.columns:
            out[(_norm_sym(und), _norm_sym(etf))] = None
            continue
        df = px[[und_col, etf_col]].dropna(how="any")
        out[(_norm_sym(und), _norm_sym(etf))] = df.index[0] if not df.empty else None
    return out

def build_pairs_and_beta_map(screen_pass, min_abs_beta=1.50):
    pairs, beta_map = [], {}
    for _, row in screen_pass.iterrows():
        etf = _norm_sym(row["ETF"])
        und = _norm_sym(row["Underlying"])
        beta = row.get("Beta", np.nan)
        if not np.isfinite(beta) or abs(beta) < min_abs_beta:
            continue
        key = (und, etf)
        if key not in beta_map:
            pairs.append(key)
            beta_map[key] = float(beta)
    return pairs, beta_map

# Extended perf stats
def max_drawdown_from_curve(curve):
    curve = curve.dropna()
    if curve.empty:
        return np.nan
    return float((curve / curve.cummax() - 1.0).min())

def sharpe_ratio(ret, rf_annual=0.0):
    ret = ret.dropna()
    if ret.empty:
        return np.nan
    rf_daily = (1.0 + rf_annual) ** (1.0 / TRADING_DAYS) - 1.0
    ex = ret - rf_daily
    vol = ex.std(ddof=0)
    if not np.isfinite(vol) or vol <= 0:
        return np.nan
    return float(ex.mean() / vol * np.sqrt(TRADING_DAYS))

def sortino_ratio(ret, mar_annual=0.0):
    ret = ret.dropna()
    if ret.empty:
        return np.nan
    mar_daily = (1.0 + mar_annual) ** (1.0 / TRADING_DAYS) - 1.0
    downside = ret[ret < mar_daily] - mar_daily
    dd = downside.std(ddof=0)
    if not np.isfinite(dd) or dd <= 0:
        return np.nan
    return float((ret.mean() - mar_daily) / dd * np.sqrt(TRADING_DAYS))

def perf_stats_extended(ret):
    ret = ret.dropna()
    n = len(ret)
    if n == 0:
        return {k: np.nan for k in
                ["TradingDays","TotalReturn","CAGR","AnnVol","MaxDD","Sharpe","Sortino","Calmar"]}

    eq = (1.0 + ret).cumprod()
    mdd = max_drawdown_from_curve(eq)
    total_return = float(eq.iloc[-1] - 1.0)
    days = (eq.index[-1] - eq.index[0]).days
    years = days / 365.25 if days > 0 else np.nan
    cagr = float(eq.iloc[-1] ** (1.0 / years) - 1.0) if (np.isfinite(years) and years > 0) else np.nan
    ann_vol = float(ret.std(ddof=0) * np.sqrt(TRADING_DAYS))
    shp = sharpe_ratio(ret)
    srt = sortino_ratio(ret)
    calmar = float(cagr / abs(mdd)) if (np.isfinite(cagr) and np.isfinite(mdd) and mdd < 0) else np.nan

    return {
        "TradingDays": n, "TotalReturn": total_return, "CAGR": cagr,
        "AnnVol": ann_vol, "MaxDD": mdd, "Sharpe": shp,
        "Sortino": srt, "Calmar": calmar,
    }

print("Infrastructure loaded.")
"""))

    # ==================================================================
    # CELL: Weighting functions
    # ==================================================================
    cells.append(code_cell("""# ============================================================
# Weighting strategies
# ============================================================

def compute_equal_weights(pairs):
    n = len(pairs)
    return {p: 1.0 / n for p in pairs} if n > 0 else {}


def compute_decay_score_weights(
    pairs, screen_pass,
    borrow_aversion=3.0,
    margin_efficiency_power=0.5,
    eq_blend=0.50,
    max_name_weight=0.08,
):
    \"\"\"
    Decay-score weighting as in generate_trade_plan.py:
    raw = blended_gross_decay - borrow_aversion * borrow_current
    margin_adj = (1/|Beta|)^margin_power
    final = eq_blend * equal + (1-eq_blend) * normalized(raw * margin_adj)
    \"\"\"
    n = len(pairs)
    if n == 0:
        return {}

    lookup = {}
    for _, row in screen_pass.iterrows():
        key = (_norm_sym(row["Underlying"]), _norm_sym(row["ETF"]))
        lookup[key] = row

    scores, betas = [], []
    for p in pairs:
        row = lookup.get(p)
        if row is None:
            scores.append(0.0); betas.append(2.0)
            continue
        decay = float(row.get("blended_gross_decay", 0) or 0)
        borrow = float(row.get("borrow_current", 0) or 0)
        beta = abs(float(row.get("Beta", 2.0) or 2.0))
        scores.append(decay - borrow_aversion * borrow)
        betas.append(max(beta, 0.1))

    scores = np.array(scores)
    betas = np.array(betas)

    margin_adj = np.power(1.0 / betas, margin_efficiency_power)
    adjusted = np.clip(scores * margin_adj, 0, None)

    sig_total = adjusted.sum()
    signal_w = adjusted / sig_total if sig_total > 0 else np.zeros(n)

    eq_w = np.ones(n) / n
    final_w = eq_blend * eq_w + (1.0 - eq_blend) * signal_w

    for _ in range(10):
        excess = np.maximum(final_w - max_name_weight, 0.0)
        if excess.sum() < 1e-12:
            break
        final_w = np.minimum(final_w, max_name_weight)
        uncapped = final_w < max_name_weight - 1e-12
        if uncapped.any():
            uc_total = final_w[uncapped].sum()
            if uc_total > 0:
                final_w[uncapped] += excess.sum() * (final_w[uncapped] / uc_total)

    s = final_w.sum()
    final_w = final_w / s if s > 0 else eq_w
    return {p: float(w) for p, w in zip(pairs, final_w)}


def compute_inverse_borrow_weights(pairs, screen_pass, max_name_weight=0.08):
    \"\"\"Weight inversely proportional to borrow cost.\"\"\"
    n = len(pairs)
    if n == 0:
        return {}

    lookup = {}
    for _, row in screen_pass.iterrows():
        key = (_norm_sym(row["Underlying"]), _norm_sym(row["ETF"]))
        lookup[key] = row

    raw_w = []
    for p in pairs:
        row = lookup.get(p)
        borrow = float(row.get("borrow_current", 0.05) or 0.05) if row is not None else 0.05
        raw_w.append(1.0 / max(borrow, 0.001))

    raw_w = np.array(raw_w)
    w = raw_w / raw_w.sum()

    for _ in range(10):
        excess = np.maximum(w - max_name_weight, 0.0)
        if excess.sum() < 1e-12:
            break
        w = np.minimum(w, max_name_weight)
        uncapped = w < max_name_weight - 1e-12
        if uncapped.any():
            uc_total = w[uncapped].sum()
            if uc_total > 0:
                w[uncapped] += excess.sum() * (w[uncapped] / uc_total)

    s = w.sum()
    w = w / s if s > 0 else np.ones(n) / n
    return {p: float(wi) for p, wi in zip(pairs, w)}

print("Weighting strategies loaded.")
"""))

    # ==================================================================
    # CELL: Core simulation engine
    # ==================================================================
    cells.append(code_cell("""def simulate_portfolio(
    tr_prices,
    pairs,
    beta_map,
    borrow_daily_map,
    pair_weights=None,
    *,
    freq="W",
    drift_threshold=None,
    target_gross_mult=4.0,
    initial_equity=1.0,
    default_borrow_daily=DEFAULT_BORROW_ANNUAL / TRADING_DAYS,
    margin_annual=MARGIN_ANNUAL,
    margin_daycount=MARGIN_DAYCOUNT,
    clamp_beta_min_abs=0.50,
    backtest_start=BACKTEST_START,
):
    \"\"\"
    Portfolio simulation with custom pair weights and flexible rebalancing.

    pair_weights : dict[(und, etf) -> float] summing to ~1, or None for equal weight.
    freq : D/W/M/Q/NEVER
    drift_threshold : if set, only rebalance when |net_exposure/equity| > threshold
    \"\"\"
    px = tr_prices.copy()
    px.index = _to_naive_utc_index(px.index)
    px = px.loc[px.index >= pd.Timestamp(backtest_start)]
    if px.empty:
        return pd.Series(dtype=float), pd.Series(dtype=int)

    idx = px.index

    if drift_threshold is None:
        rebal_days = get_rebalance_days(idx, freq)
    else:
        rebal_days = {pd.Timestamp(idx[0])}

    pair_start = build_pair_start_dates(px, pairs)

    if pair_weights is None:
        pair_weights = {p: 1.0 / len(pairs) for p in pairs}

    equity = float(initial_equity)
    shares_under = {}
    shares_etf = {}
    port_ret = pd.Series(index=idx, dtype=float)
    pair_count = pd.Series(index=idx, dtype=int)
    margin_daily = float(margin_annual) / float(margin_daycount)

    first_active_day = None
    prev_t = None

    def eligible_pairs_on(t):
        return [(u, e) for (u, e), d0 in pair_start.items()
                if d0 is not None and d0 <= t]

    def do_rebalance(t, active_pairs):
        nonlocal shares_under, shares_etf
        n = len(active_pairs)
        if n == 0:
            shares_under, shares_etf = {}, {}
            return

        gross_target = target_gross_mult * equity
        active_w = {p: pair_weights.get(p, 0.0) for p in active_pairs}
        w_sum = sum(active_w.values())
        if w_sum <= 0:
            active_w = {p: 1.0 / n for p in active_pairs}
            w_sum = 1.0
        active_w = {p: v / w_sum for p, v in active_w.items()}

        su, se = {}, {}
        for und, etf in active_pairs:
            und_n, etf_n = _norm_sym(und), _norm_sym(etf)
            und_col, etf_col = f"{und_n}_TR", f"{etf_n}_TR"
            if und_col not in px.columns or etf_col not in px.columns:
                continue
            pu = float(px.at[t, und_col])
            pe = float(px.at[t, etf_col])
            if not (np.isfinite(pu) and pu > 0 and np.isfinite(pe) and pe > 0):
                continue

            beta_abs = abs(beta_map.get((und, etf), 2.0))
            beta_abs = max(beta_abs, clamp_beta_min_abs)
            short_ratio = 1.0 / beta_abs

            w = active_w.get((und, etf), 0.0)
            gross_this = gross_target * w
            a = gross_this / (1.0 + short_ratio)

            su[und_n] = su.get(und_n, 0.0) + (a / pu)
            se[etf_n] = se.get(etf_n, 0.0) - (a * short_ratio / pe)

        shares_under, shares_etf = su, se

    def compute_net_exposure(t):
        long_not = sum(
            sh * float(px.at[t, f"{_norm_sym(u)}_TR"])
            for u, sh in shares_under.items()
            if f"{_norm_sym(u)}_TR" in px.columns
            and np.isfinite(float(px.at[t, f"{_norm_sym(u)}_TR"]))
        )
        short_not = sum(
            abs(sh) * float(px.at[t, f"{_norm_sym(e)}_TR"])
            for e, sh in shares_etf.items()
            if f"{_norm_sym(e)}_TR" in px.columns
            and np.isfinite(float(px.at[t, f"{_norm_sym(e)}_TR"]))
        )
        return long_not - short_not

    for t in idx:
        should_rebal = False
        if prev_t is None:
            should_rebal = True
        elif drift_threshold is not None:
            net_exp = compute_net_exposure(t)
            if equity > 0 and abs(net_exp / equity) > drift_threshold:
                should_rebal = True
        elif t in rebal_days:
            should_rebal = True

        if should_rebal:
            active_pairs = eligible_pairs_on(t)
            pair_count.at[t] = len(active_pairs)
            if first_active_day is None and len(active_pairs) > 0:
                first_active_day = t
            do_rebalance(t, active_pairs)
        else:
            pair_count.at[t] = pair_count.at[prev_t] if prev_t is not None else 0

        if not shares_under and not shares_etf:
            port_ret.at[t] = 0.0
            prev_t = t
            continue
        if prev_t is None:
            port_ret.at[t] = 0.0
            prev_t = t
            continue

        pnl = 0.0
        for und, sh in shares_under.items():
            col = f"{_norm_sym(und)}_TR"
            if col not in px.columns:
                continue
            p0, p1 = float(px.at[prev_t, col]), float(px.at[t, col])
            if np.isfinite(p0) and np.isfinite(p1):
                pnl += sh * (p1 - p0)

        borrow_cost = 0.0
        for etf, sh in shares_etf.items():
            col = f"{_norm_sym(etf)}_TR"
            if col not in px.columns:
                continue
            p0, p1 = float(px.at[prev_t, col]), float(px.at[t, col])
            if np.isfinite(p0) and np.isfinite(p1):
                pnl += sh * (p1 - p0)
            b = float(borrow_daily_map.get(_norm_sym(etf), default_borrow_daily))
            if np.isfinite(p0):
                borrow_cost += abs(sh) * p0 * b

        long_not = sum(
            sh * float(px.at[prev_t, f"{_norm_sym(u)}_TR"])
            for u, sh in shares_under.items()
            if f"{_norm_sym(u)}_TR" in px.columns
            and np.isfinite(float(px.at[prev_t, f"{_norm_sym(u)}_TR"]))
        )
        short_not = sum(
            abs(sh) * float(px.at[prev_t, f"{_norm_sym(e)}_TR"])
            for e, sh in shares_etf.items()
            if f"{_norm_sym(e)}_TR" in px.columns
            and np.isfinite(float(px.at[prev_t, f"{_norm_sym(e)}_TR"]))
        )
        debit = max(0.0, (long_not - short_not) - equity)
        margin_interest = debit * margin_daily

        pnl -= (borrow_cost + margin_interest)
        r = pnl / equity if equity != 0 else 0.0
        equity *= (1.0 + r)
        port_ret.at[t] = r
        prev_t = t

    if first_active_day is not None:
        port_ret = port_ret.loc[first_active_day:]
        pair_count = pair_count.loc[first_active_day:]

    return port_ret.dropna(), pair_count

print("Simulation engine loaded.")
"""))

    # ==================================================================
    # CELL: Section 3 header
    # ==================================================================
    cells.append(md_cell("""## 3. Weighting Strategy Comparison

Compare three weighting strategies across borrow threshold filters:
- **Equal weight**: uniform allocation across all qualifying pairs
- **Decay-score**: signal = gross_decay - borrow_aversion * borrow (from `generate_trade_plan.py`)
- **Inverse-borrow**: weight inversely proportional to borrow cost"""))

    # ==================================================================
    # CELL: Run weighting backtests
    # ==================================================================
    cells.append(code_cell("""# Build base pairs/beta
all_pairs, beta_map = build_pairs_and_beta_map(bucket1, min_abs_beta=1.50)
print(f"Total pairs (Beta > 1.5): {len(all_pairs)}")

# Borrow thresholds to test
BORROW_THRESHOLDS = [0.05, 0.10, 0.20, 0.30, 1.0]  # 1.0 = no filter
WEIGHTING_METHODS = {
    "Equal": compute_equal_weights,
    "Decay-Score": lambda pairs: compute_decay_score_weights(pairs, bucket1),
    "Inverse-Borrow": lambda pairs: compute_inverse_borrow_weights(pairs, bucket1),
}

# Store results
weighting_results = {}
weighting_returns = {}

for borrow_cap in BORROW_THRESHOLDS:
    # Filter pairs by borrow threshold
    filtered_pairs = []
    for p in all_pairs:
        und, etf = p
        etf_borrow = borrow_daily_map.get(etf, DEFAULT_BORROW_ANNUAL / TRADING_DAYS) * TRADING_DAYS
        if etf_borrow <= borrow_cap:
            filtered_pairs.append(p)

    if len(filtered_pairs) == 0:
        print(f"  Borrow cap {borrow_cap:.0%}: 0 pairs, skipping")
        continue

    print(f"\\nBorrow cap {borrow_cap:.0%}: {len(filtered_pairs)} pairs")

    for method_name, weight_fn in WEIGHTING_METHODS.items():
        weights = weight_fn(filtered_pairs)
        label = f"{method_name} | Borrow<={borrow_cap:.0%}"

        ret, pc = simulate_portfolio(
            prices, filtered_pairs, beta_map, borrow_daily_map,
            pair_weights=weights,
            freq="W",
            target_gross_mult=4.0,
            backtest_start=BACKTEST_START,
        )

        s = perf_stats_extended(ret)
        s["Method"] = method_name
        s["BorrowCap"] = borrow_cap
        s["NumPairs"] = len(filtered_pairs)
        weighting_results[label] = s
        weighting_returns[label] = ret
        print(f"  {label}: CAGR={s['CAGR']:.1%} Sharpe={s['Sharpe']:.2f} MaxDD={s['MaxDD']:.1%}")

weighting_df = pd.DataFrame(weighting_results).T
print("\\n" + "="*80)
print(weighting_df[["Method","BorrowCap","NumPairs","CAGR","AnnVol","MaxDD","Sharpe","Sortino","Calmar"]].to_string())
"""))

    # ==================================================================
    # CELL: Weighting visualization
    # ==================================================================
    cells.append(code_cell("""fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# 1. Equity curves by weighting method (at the broadest borrow cap that has all 3 methods)
ax = axes[0, 0]
best_cap = max(BORROW_THRESHOLDS)
for label, ret in weighting_returns.items():
    if f"Borrow<={best_cap:.0%}" in label:
        eq = (1 + ret).cumprod()
        ax.plot(eq.index, eq.values, label=label.split(" | ")[0])
ax.set_title(f"Equity Curves by Weighting (Borrow cap={best_cap:.0%})")
ax.legend()
ax.grid(True)
ax.set_ylabel("Growth of $1")

# 2. CAGR by method and borrow cap
ax = axes[0, 1]
for method in WEIGHTING_METHODS.keys():
    caps = []
    cagrs = []
    for label, s in weighting_results.items():
        if s["Method"] == method:
            caps.append(s["BorrowCap"])
            cagrs.append(s["CAGR"])
    ax.plot(caps, cagrs, "o-", label=method)
ax.set_title("CAGR by Weighting Method & Borrow Cap")
ax.set_xlabel("Borrow Cap")
ax.set_ylabel("CAGR")
ax.legend()
ax.grid(True)

# 3. Sharpe by method and borrow cap
ax = axes[1, 0]
for method in WEIGHTING_METHODS.keys():
    caps = []
    sharpes = []
    for label, s in weighting_results.items():
        if s["Method"] == method:
            caps.append(s["BorrowCap"])
            sharpes.append(s["Sharpe"])
    ax.plot(caps, sharpes, "o-", label=method)
ax.set_title("Sharpe by Weighting Method & Borrow Cap")
ax.set_xlabel("Borrow Cap")
ax.set_ylabel("Sharpe Ratio")
ax.legend()
ax.grid(True)

# 4. MaxDD by method and borrow cap
ax = axes[1, 1]
for method in WEIGHTING_METHODS.keys():
    caps = []
    dds = []
    for label, s in weighting_results.items():
        if s["Method"] == method:
            caps.append(s["BorrowCap"])
            dds.append(s["MaxDD"])
    ax.plot(caps, dds, "o-", label=method)
ax.set_title("Max Drawdown by Weighting Method & Borrow Cap")
ax.set_xlabel("Borrow Cap")
ax.set_ylabel("Max DD")
ax.legend()
ax.grid(True)

plt.tight_layout()
plt.show()
"""))

    # ==================================================================
    # CELL: Section 4 header
    # ==================================================================
    cells.append(md_cell("""## 4. Hedging / Rebalance Strategy Comparison

Test different rebalance frequencies and drift-based rebalancing.
Use the best weighting method from Section 3 (or default to decay-score).

Strategies:
- **Daily (D)**: Rebalance every day
- **Weekly (W)**: Rebalance last trading day of each week
- **Monthly (M)**: Rebalance last trading day of each month
- **Quarterly (Q)**: Rebalance last trading day of each quarter
- **Never**: Set once and never rebalance
- **Drift 10%/20%/50%**: Only rebalance when net exposure exceeds threshold"""))

    # ==================================================================
    # CELL: Run hedging backtests
    # ==================================================================
    cells.append(code_cell("""# Use all pairs (broadest borrow cap) with decay-score weighting
hedge_pairs = all_pairs
hedge_weights = compute_decay_score_weights(hedge_pairs, bucket1)

# Frequency-based strategies
FREQ_STRATEGIES = ["D", "W", "M", "Q", "NEVER"]

# Drift-based strategies
DRIFT_THRESHOLDS = [0.10, 0.20, 0.50]

hedge_results = {}
hedge_returns = {}

print("Running frequency-based rebalance strategies...")
for freq in FREQ_STRATEGIES:
    label = f"Rebal={freq}"
    ret, pc = simulate_portfolio(
        prices, hedge_pairs, beta_map, borrow_daily_map,
        pair_weights=hedge_weights,
        freq=freq,
        target_gross_mult=4.0,
        backtest_start=BACKTEST_START,
    )
    s = perf_stats_extended(ret)
    s["Strategy"] = label
    s["Type"] = "frequency"
    hedge_results[label] = s
    hedge_returns[label] = ret
    print(f"  {label}: CAGR={s['CAGR']:.1%} Sharpe={s['Sharpe']:.2f} MaxDD={s['MaxDD']:.1%}")

print("\\nRunning drift-based rebalance strategies...")
for drift in DRIFT_THRESHOLDS:
    label = f"Drift={drift:.0%}"
    ret, pc = simulate_portfolio(
        prices, hedge_pairs, beta_map, borrow_daily_map,
        pair_weights=hedge_weights,
        drift_threshold=drift,
        target_gross_mult=4.0,
        backtest_start=BACKTEST_START,
    )
    s = perf_stats_extended(ret)
    s["Strategy"] = label
    s["Type"] = "drift"
    hedge_results[label] = s
    hedge_returns[label] = ret
    print(f"  {label}: CAGR={s['CAGR']:.1%} Sharpe={s['Sharpe']:.2f} MaxDD={s['MaxDD']:.1%}")

hedge_df = pd.DataFrame(hedge_results).T
print("\\n" + "="*80)
print(hedge_df[["Strategy","CAGR","AnnVol","MaxDD","Sharpe","Sortino","Calmar"]].to_string())
"""))

    # ==================================================================
    # CELL: Hedging visualization
    # ==================================================================
    cells.append(code_cell("""fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 1. Equity curves
ax = axes[0]
for label, ret in hedge_returns.items():
    eq = (1 + ret).cumprod()
    ax.plot(eq.index, eq.values, label=label, alpha=0.8)
ax.set_title("Equity Curves by Rebalance Strategy")
ax.legend(fontsize=8)
ax.grid(True)
ax.set_ylabel("Growth of $1")

# 2. Sharpe bar chart
ax = axes[1]
labels = list(hedge_results.keys())
sharpes = [hedge_results[l]["Sharpe"] for l in labels]
colors = ["steelblue" if hedge_results[l]["Type"] == "frequency" else "coral" for l in labels]
ax.barh(labels, sharpes, color=colors)
ax.set_title("Sharpe Ratio by Strategy")
ax.set_xlabel("Sharpe")
ax.grid(True, axis="x")

# 3. Risk-return scatter
ax = axes[2]
for label, s in hedge_results.items():
    marker = "o" if s["Type"] == "frequency" else "s"
    ax.scatter(s["AnnVol"], s["CAGR"], marker=marker, s=80)
    ax.annotate(label, (s["AnnVol"], s["CAGR"]), fontsize=7, ha="left")
ax.set_title("Risk-Return by Strategy")
ax.set_xlabel("Annualized Vol")
ax.set_ylabel("CAGR")
ax.grid(True)

plt.tight_layout()
plt.show()
"""))

    # ==================================================================
    # CELL: Section 5 header
    # ==================================================================
    cells.append(md_cell("""## 5. Hypothesis Testing

### Test 1: H₀ — Equal weighting is at least as good as decay-score weighting
- One-sided paired test: does decay-score produce higher mean daily returns?
- Bootstrap CI on Sharpe ratio difference
- Newey-West HAC standard errors to handle autocorrelation

### Test 2: H₀ — Weekly rebalance to 0 net exposure is optimal
- Compare weekly vs each alternative using paired tests
- Bonferroni correction for multiple comparisons
- Bootstrap Sharpe difference CIs"""))

    # ==================================================================
    # CELL: Hypothesis test functions
    # ==================================================================
    cells.append(code_cell("""def newey_west_se(x, max_lag=None):
    \"\"\"Newey-West HAC standard error for mean of x.\"\"\"
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 2:
        return np.nan

    if max_lag is None:
        max_lag = int(np.floor(4 * (n / 100) ** (2/9)))

    xbar = x.mean()
    e = x - xbar

    gamma0 = np.dot(e, e) / n
    nw_var = gamma0

    for j in range(1, max_lag + 1):
        w = 1.0 - j / (max_lag + 1)  # Bartlett kernel
        gamma_j = np.dot(e[j:], e[:-j]) / n
        nw_var += 2 * w * gamma_j

    return np.sqrt(max(nw_var, 0) / n)


def paired_ttest_hac(r1, r2, alternative="greater"):
    \"\"\"
    HAC-robust paired t-test.
    H0: mean(r1) <= mean(r2) [if alternative='greater', tests if r1 > r2]
    Returns t-stat, p-value, mean_diff, se.
    \"\"\"
    r1 = np.asarray(r1, dtype=float)
    r2 = np.asarray(r2, dtype=float)

    # Align lengths
    n = min(len(r1), len(r2))
    d = r1[:n] - r2[:n]
    d = d[np.isfinite(d)]

    if len(d) < 10:
        return np.nan, np.nan, np.nan, np.nan

    mean_d = d.mean()
    se = newey_west_se(d)

    if se <= 0 or not np.isfinite(se):
        return np.nan, np.nan, mean_d, se

    t_stat = mean_d / se

    if alternative == "greater":
        p_val = 1.0 - stats.t.cdf(t_stat, df=len(d) - 1)
    elif alternative == "less":
        p_val = stats.t.cdf(t_stat, df=len(d) - 1)
    else:
        p_val = 2 * (1.0 - stats.t.cdf(abs(t_stat), df=len(d) - 1))

    return t_stat, p_val, mean_d, se


def bootstrap_sharpe_diff(r1, r2, n_boot=10000, ci=0.95):
    \"\"\"
    Bootstrap confidence interval for Sharpe(r1) - Sharpe(r2).
    \"\"\"
    r1 = np.asarray(r1, dtype=float)
    r2 = np.asarray(r2, dtype=float)
    n = min(len(r1), len(r2))
    r1, r2 = r1[:n], r2[:n]

    mask = np.isfinite(r1) & np.isfinite(r2)
    r1, r2 = r1[mask], r2[mask]
    n = len(r1)

    if n < 20:
        return np.nan, np.nan, np.nan

    def sharpe(r):
        if r.std() == 0:
            return 0.0
        return r.mean() / r.std() * np.sqrt(TRADING_DAYS)

    obs_diff = sharpe(r1) - sharpe(r2)

    boot_diffs = np.empty(n_boot)
    rng = np.random.default_rng(42)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_diffs[i] = sharpe(r1[idx]) - sharpe(r2[idx])

    alpha = 1 - ci
    lo = np.percentile(boot_diffs, 100 * alpha / 2)
    hi = np.percentile(boot_diffs, 100 * (1 - alpha / 2))

    return obs_diff, lo, hi


def permutation_test_mean_diff(r1, r2, n_perm=10000):
    \"\"\"
    Permutation test for mean(r1) > mean(r2).
    Returns observed diff and p-value.
    \"\"\"
    r1 = np.asarray(r1, dtype=float)
    r2 = np.asarray(r2, dtype=float)
    n = min(len(r1), len(r2))
    r1, r2 = r1[:n], r2[:n]

    mask = np.isfinite(r1) & np.isfinite(r2)
    r1, r2 = r1[mask], r2[mask]
    n = len(r1)

    obs_diff = r1.mean() - r2.mean()
    combined = np.concatenate([r1, r2])
    rng = np.random.default_rng(42)

    count = 0
    for _ in range(n_perm):
        rng.shuffle(combined)
        perm_diff = combined[:n].mean() - combined[n:].mean()
        if perm_diff >= obs_diff:
            count += 1

    p_val = count / n_perm
    return obs_diff, p_val

print("Hypothesis test functions loaded.")
"""))

    # ==================================================================
    # CELL: Test 1 - Equal vs Decay Score
    # ==================================================================
    cells.append(code_cell("""print("=" * 80)
print("TEST 1: H0 - Equal weighting >= Decay-score weighting")
print("=" * 80)
print("Alternative: Decay-score weighting produces higher returns / Sharpe\\n")

# Get returns for both strategies at broadest borrow cap, weekly rebalance
best_cap = max(BORROW_THRESHOLDS)
eq_label = f"Equal | Borrow<={best_cap:.0%}"
ds_label = f"Decay-Score | Borrow<={best_cap:.0%}"

if eq_label in weighting_returns and ds_label in weighting_returns:
    r_eq = weighting_returns[eq_label].values
    r_ds = weighting_returns[ds_label].values

    # 1. HAC paired t-test: is decay-score mean return > equal?
    t_stat, p_val, mean_diff, se = paired_ttest_hac(r_ds, r_eq, alternative="greater")
    print(f"--- HAC Paired t-test (decay > equal) ---")
    print(f"  Mean daily return diff (DS - EQ): {mean_diff:.6f}")
    print(f"  HAC SE: {se:.6f}")
    print(f"  t-stat: {t_stat:.3f}")
    print(f"  p-value (one-sided): {p_val:.4f}")
    if p_val < 0.05:
        print(f"  ** REJECT H0 at 5%: Decay-score significantly outperforms equal weight **")
    else:
        print(f"  Cannot reject H0 at 5%: No significant difference")

    # 2. Bootstrap Sharpe difference
    obs_diff, lo, hi = bootstrap_sharpe_diff(r_ds, r_eq, n_boot=10000)
    print(f"\\n--- Bootstrap Sharpe Difference (DS - EQ) ---")
    print(f"  Observed: {obs_diff:.3f}")
    print(f"  95% CI: [{lo:.3f}, {hi:.3f}]")
    if lo > 0:
        print(f"  ** CI excludes 0: Decay-score has significantly higher Sharpe **")
    else:
        print(f"  CI includes 0: Difference not significant at 95%")

    # 3. Permutation test
    obs_diff_perm, p_perm = permutation_test_mean_diff(r_ds, r_eq, n_perm=10000)
    print(f"\\n--- Permutation Test (DS > EQ) ---")
    print(f"  Observed mean diff: {obs_diff_perm:.6f}")
    print(f"  p-value: {p_perm:.4f}")
    if p_perm < 0.05:
        print(f"  ** REJECT H0 at 5% via permutation test **")
    else:
        print(f"  Cannot reject H0 at 5%")

    # Also test inverse-borrow vs equal
    ib_label = f"Inverse-Borrow | Borrow<={best_cap:.0%}"
    if ib_label in weighting_returns:
        print(f"\\n{'='*80}")
        print("SUPPLEMENTARY: Inverse-Borrow vs Equal")
        r_ib = weighting_returns[ib_label].values
        t2, p2, md2, se2 = paired_ttest_hac(r_ib, r_eq, alternative="greater")
        print(f"  HAC t-test (IB > EQ): t={t2:.3f}, p={p2:.4f}")
        obs2, lo2, hi2 = bootstrap_sharpe_diff(r_ib, r_eq, n_boot=10000)
        print(f"  Bootstrap Sharpe diff: {obs2:.3f} [{lo2:.3f}, {hi2:.3f}]")
else:
    print("Missing return series. Check borrow thresholds / labels.")
"""))

    # ==================================================================
    # CELL: Test 2 - Weekly rebalance optimality
    # ==================================================================
    cells.append(code_cell("""print("=" * 80)
print("TEST 2: H0 - Weekly rebalance to 0 net exposure is optimal")
print("=" * 80)
print("Alternative: Some other rebalance strategy produces higher returns / Sharpe\\n")

weekly_label = "Rebal=W"
if weekly_label not in hedge_returns:
    print("No weekly returns found!")
else:
    r_weekly = hedge_returns[weekly_label].values

    # Compare against all other strategies
    alternatives = [l for l in hedge_returns.keys() if l != weekly_label]
    n_comparisons = len(alternatives)
    bonferroni_alpha = 0.05 / n_comparisons

    print(f"Comparing weekly vs {n_comparisons} alternatives")
    print(f"Bonferroni-corrected alpha: {bonferroni_alpha:.4f}\\n")

    test2_rows = []
    for alt_label in alternatives:
        r_alt = hedge_returns[alt_label].values

        # HAC t-test: is alternative better than weekly?
        t_stat, p_val, mean_diff, se = paired_ttest_hac(r_alt, r_weekly, alternative="greater")

        # Bootstrap Sharpe diff
        obs_diff, lo, hi = bootstrap_sharpe_diff(r_alt, r_weekly, n_boot=10000)

        significant_raw = p_val < 0.05
        significant_bonf = p_val < bonferroni_alpha
        ci_excludes_zero = lo > 0

        test2_rows.append({
            "Alternative": alt_label,
            "MeanDiff_daily": mean_diff,
            "HAC_tstat": t_stat,
            "HAC_pval": p_val,
            "Reject_5pct": significant_raw,
            "Reject_Bonferroni": significant_bonf,
            "Sharpe_diff": obs_diff,
            "Sharpe_CI_lo": lo,
            "Sharpe_CI_hi": hi,
            "CI_excl_zero": ci_excludes_zero,
        })

        sig_marker = "***" if significant_bonf else ("**" if significant_raw else "")
        print(f"  {alt_label}:")
        print(f"    mean_diff={mean_diff:.6f}  t={t_stat:.3f}  p={p_val:.4f} {sig_marker}")
        print(f"    Sharpe diff={obs_diff:.3f} [{lo:.3f}, {hi:.3f}]")

    test2_df = pd.DataFrame(test2_rows)
    print("\\n--- Summary ---")
    any_reject = test2_df["Reject_5pct"].any()
    any_reject_bonf = test2_df["Reject_Bonferroni"].any()

    if any_reject_bonf:
        winners = test2_df[test2_df["Reject_Bonferroni"]]["Alternative"].tolist()
        print(f"REJECT H0 (Bonferroni): {winners} significantly outperform weekly rebalance")
    elif any_reject:
        winners = test2_df[test2_df["Reject_5pct"]]["Alternative"].tolist()
        print(f"REJECT H0 (raw 5%): {winners} outperform weekly (not significant after Bonferroni)")
    else:
        print("FAIL TO REJECT H0: No strategy significantly outperforms weekly rebalance")
"""))

    # ==================================================================
    # CELL: Summary dashboard
    # ==================================================================
    cells.append(md_cell("## 6. Summary Dashboard"))

    cells.append(code_cell("""# Consolidated summary
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# 1. Best weighting by Sharpe
ax = axes[0, 0]
wdf = weighting_df.copy()
wdf["label"] = wdf["Method"] + " (borrow<=" + wdf["BorrowCap"].apply(lambda x: f"{x:.0%}") + ")"
top = wdf.nlargest(10, "Sharpe")
ax.barh(top["label"], top["Sharpe"], color="steelblue")
ax.set_title("Top 10 Weighting Configs by Sharpe")
ax.set_xlabel("Sharpe")
ax.grid(True, axis="x")

# 2. Best hedging by Sharpe
ax = axes[0, 1]
hdf = hedge_df.copy()
ax.barh(hdf["Strategy"], hdf["Sharpe"], color="coral")
ax.set_title("Rebalance Strategies by Sharpe")
ax.set_xlabel("Sharpe")
ax.grid(True, axis="x")

# 3. Best overall equity curve
ax = axes[1, 0]
# Pick best weighting + best hedging equity curves
best_w_label = weighting_df.loc[weighting_df["Sharpe"].idxmax()].name if not weighting_df.empty else None
best_h_label = hedge_df.loc[hedge_df["Sharpe"].idxmax(), "Strategy"] if not hedge_df.empty else None

if best_w_label and best_w_label in weighting_returns:
    eq = (1 + weighting_returns[best_w_label]).cumprod()
    ax.plot(eq.index, eq.values, label=f"Best Weight: {best_w_label}", color="steelblue")
if best_h_label and best_h_label in hedge_returns:
    eq = (1 + hedge_returns[best_h_label]).cumprod()
    ax.plot(eq.index, eq.values, label=f"Best Hedge: {best_h_label}", color="coral")
# Also plot equal/weekly as baseline
eq_w_base = f"Equal | Borrow<={max(BORROW_THRESHOLDS):.0%}"
if eq_w_base in weighting_returns:
    eq = (1 + weighting_returns[eq_w_base]).cumprod()
    ax.plot(eq.index, eq.values, label="Baseline: Equal/Weekly", color="gray", ls="--")
ax.set_title("Best vs Baseline Equity Curves")
ax.legend(fontsize=8)
ax.grid(True)
ax.set_ylabel("Growth of $1")

# 4. Hypothesis test summary
ax = axes[1, 1]
ax.axis("off")
summary_text = "HYPOTHESIS TEST RESULTS\\n" + "="*40 + "\\n\\n"

# Test 1
if eq_label in weighting_returns and ds_label in weighting_returns:
    _, p1, _, _ = paired_ttest_hac(
        weighting_returns[ds_label].values,
        weighting_returns[eq_label].values,
        alternative="greater"
    )
    verdict1 = "REJECT" if p1 < 0.05 else "FAIL TO REJECT"
    summary_text += f"Test 1: Equal >= Decay-Score\\n"
    summary_text += f"  p={p1:.4f} -> {verdict1} H0\\n\\n"

# Test 2
if weekly_label in hedge_returns:
    best_alt_p = 1.0
    best_alt_name = ""
    for alt in [l for l in hedge_returns if l != weekly_label]:
        _, p_a, _, _ = paired_ttest_hac(
            hedge_returns[alt].values,
            hedge_returns[weekly_label].values,
            alternative="greater"
        )
        if p_a < best_alt_p:
            best_alt_p = p_a
            best_alt_name = alt
    verdict2 = "REJECT" if best_alt_p < 0.05 else "FAIL TO REJECT"
    summary_text += f"Test 2: Weekly rebal is optimal\\n"
    summary_text += f"  Best alt: {best_alt_name}\\n"
    summary_text += f"  p={best_alt_p:.4f} -> {verdict2} H0\\n"

ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
        fontsize=11, verticalalignment="top", fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

plt.suptitle("Strategy Comparison Dashboard", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()
"""))

    # ==================================================================
    # CELL: Save results
    # ==================================================================
    cells.append(code_cell("""# Save detailed results to CSV
output_dir = repo_root / "data"
output_dir.mkdir(parents=True, exist_ok=True)

weighting_df.to_csv(output_dir / "backtest_weighting_comparison.csv")
hedge_df.to_csv(output_dir / "backtest_hedge_comparison.csv")

print(f"Saved weighting results: {output_dir / 'backtest_weighting_comparison.csv'}")
print(f"Saved hedging results: {output_dir / 'backtest_hedge_comparison.csv'}")

# Print final summary table
print("\\n" + "="*80)
print("FINAL RESULTS SUMMARY")
print("="*80)
print("\\n--- Weighting Strategies ---")
cols = ["Method", "BorrowCap", "NumPairs", "CAGR", "AnnVol", "MaxDD", "Sharpe", "Calmar"]
cols = [c for c in cols if c in weighting_df.columns]
print(weighting_df[cols].sort_values("Sharpe", ascending=False).head(10).to_string())

print("\\n--- Hedging Strategies ---")
cols = ["Strategy", "CAGR", "AnnVol", "MaxDD", "Sharpe", "Calmar"]
cols = [c for c in cols if c in hedge_df.columns]
print(hedge_df[cols].sort_values("Sharpe", ascending=False).to_string())
"""))

    # Build notebook
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.11.0",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    return nb


if __name__ == "__main__":
    nb = build_notebook()
    out_path = Path("notebooks/weighting_hedging_backtest.ipynb")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"Wrote {out_path} ({len(nb['cells'])} cells)")
