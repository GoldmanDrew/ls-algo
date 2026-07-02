"""
Bucket 5 experiment harness.

Two designed experiments, both driven off the live UVIX/SVIX panel and the
VIX/VIX3M "simple ratio" (contango < 1, backwardation > 1):

  EXPERIMENT A -- Deep-OTM put sweep
  ----------------------------------
  Grid over SPX put moneyness {20, 25, 30, 35}% OTM x premium size
  {0.25, 0.5, 1.0, 2.0}% equity/roll, holding a mid carry ratio (rho=1.0).
  Reports calm-market drag, self-funding, and crash payoff (real Aug-2024 +
  synthetic -20/-30/-40% SPX crash w/ VIX spike). Deep puts are priced with a
  skew model so they are NOT treated as free (see bucket5_put_overlay).

  EXPERIMENT B -- Dynamic ratio sizing policies
  ---------------------------------------------
  A family of policies mapping the VIX/VIX3M ratio -> (rho, gross). rho is the
  SVIX/UVIX short-notional ratio; gross scales the whole book. Policies:
    * static            : constant rho, constant gross (baseline)
    * linear_rho        : rho ramps with ratio; gross flat
    * step_gate         : 2012-article 0.917 threshold (rho_low/full gross below,
                          rho_high + gross cut above)
    * linear_rho_gross  : rho ramps AND gross de-risks as ratio rises
    * hysteresis_gate   : enter/exit bands around the gate (less churn)
    * sigmoid           : smooth logistic transition centered on a threshold
  Each is scored on carry-only and carry+put metrics, including the synthetic
  crash, so we can compare carry vs tail trade-offs.

Run::

    python scripts/bucket5_experiments.py            # both experiments
    python scripts/bucket5_experiments.py A          # just the put sweep
    python scripts/bucket5_experiments.py B          # just the ratio policies
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scripts.bucket5_carry_bt import (
        BETA_SVIX,
        BETA_UVIX,
        perf_stats,
        run_carry_backtest,
    )
    from scripts.bucket5_data import INCEPTION, load_vol_panel, rebalance_dates
    from scripts.bucket5_put_overlay import (
        BackspreadConfig,
        PutOverlayConfig,
        bs_put,
        effective_iv,
        hedge_crash_value,
        load_spx_spot,
        run_backspread_overlay,
        run_put_overlay,
    )
except ImportError:
    from bucket5_carry_bt import (  # type: ignore
        BETA_SVIX,
        BETA_UVIX,
        perf_stats,
        run_carry_backtest,
    )
    from bucket5_data import INCEPTION, load_vol_panel, rebalance_dates  # type: ignore
    from bucket5_put_overlay import (  # type: ignore
        BackspreadConfig,
        PutOverlayConfig,
        bs_put,
        effective_iv,
        hedge_crash_value,
        load_spx_spot,
        run_backspread_overlay,
        run_put_overlay,
    )


# ===========================================================================
# Dynamic ratio / gross policy library
# ===========================================================================
@dataclass
class Policy:
    name: str
    rho: pd.Series
    gross: pd.Series


def _clip(s: pd.Series, lo: float, hi: float) -> pd.Series:
    return s.clip(lo, hi)


def policy_static(ratio: pd.Series, *, rho: float = 1.0, gross: float = 1.0) -> Policy:
    return Policy(
        f"static(rho={rho})",
        pd.Series(rho, index=ratio.index),
        pd.Series(gross, index=ratio.index),
    )


def policy_linear_rho(
    ratio: pd.Series, *, rho_min=0.5, rho_max=2.0, r_lo=0.85, r_hi=0.95, gross=1.0
) -> Policy:
    frac = ((ratio - r_lo) / (r_hi - r_lo)).clip(0, 1)
    rho = rho_min + (rho_max - rho_min) * frac
    return Policy(
        f"linear_rho[{r_lo},{r_hi}]",
        rho.rename("rho"),
        pd.Series(gross, index=ratio.index),
    )


def policy_step_gate(
    ratio: pd.Series,
    *,
    thr=0.917,
    rho_low=0.8,
    rho_high=1.8,
    gross_full=1.0,
    gross_cut=0.4,
) -> Policy:
    """2012-article hard gate: calm (ratio<thr) vs danger (ratio>=thr)."""
    danger = ratio >= thr
    rho = pd.Series(np.where(danger, rho_high, rho_low), index=ratio.index)
    gross = pd.Series(np.where(danger, gross_cut, gross_full), index=ratio.index)
    return Policy(f"step_gate(thr={thr})", rho, gross)


def policy_linear_rho_gross(
    ratio: pd.Series,
    *,
    rho_min=0.6,
    rho_max=1.8,
    gross_max=1.2,
    gross_min=0.4,
    r_lo=0.85,
    r_hi=0.97,
) -> Policy:
    frac = ((ratio - r_lo) / (r_hi - r_lo)).clip(0, 1)
    rho = rho_min + (rho_max - rho_min) * frac
    gross = gross_max - (gross_max - gross_min) * frac
    return Policy(f"linear_rho_gross[{r_lo},{r_hi}]", rho.rename("rho"), gross.rename("gross"))


def policy_hysteresis_gate(
    ratio: pd.Series,
    *,
    enter=0.95,
    exit=0.88,
    rho_low=0.8,
    rho_high=1.8,
    gross_full=1.0,
    gross_cut=0.4,
) -> Policy:
    """Enter danger when ratio>=enter; stay until ratio<exit (reduces churn)."""
    danger = False
    rhos, grosses = [], []
    for r in ratio:
        if not danger and r >= enter:
            danger = True
        elif danger and r < exit:
            danger = False
        rhos.append(rho_high if danger else rho_low)
        grosses.append(gross_cut if danger else gross_full)
    return Policy(
        f"hysteresis[{exit},{enter}]",
        pd.Series(rhos, index=ratio.index),
        pd.Series(grosses, index=ratio.index),
    )


def policy_sigmoid(
    ratio: pd.Series,
    *,
    center=0.92,
    steepness=40.0,
    rho_min=0.6,
    rho_max=1.8,
    gross_max=1.1,
    gross_min=0.4,
) -> Policy:
    z = 1.0 / (1.0 + np.exp(-steepness * (ratio - center)))
    rho = rho_min + (rho_max - rho_min) * z
    gross = gross_max - (gross_max - gross_min) * z
    return Policy(f"sigmoid(c={center})", rho.rename("rho"), gross.rename("gross"))


def build_policies(ratio: pd.Series) -> list[Policy]:
    return [
        policy_static(ratio, rho=1.0),
        policy_static(ratio, rho=1.5),
        policy_linear_rho(ratio),
        policy_step_gate(ratio),
        policy_step_gate(ratio, thr=0.95, gross_cut=0.3),
        policy_linear_rho_gross(ratio),
        policy_hysteresis_gate(ratio),
        policy_sigmoid(ratio),
    ]


# ===========================================================================
# Synthetic crash stress (live sample has no 2018/2020-scale event)
# ===========================================================================
@dataclass
class CrashScenario:
    name: str
    spx_drop: float        # terminal SPX move (e.g. -0.30)
    vix_mult: float        # VIX level multiplier at trough
    days: int              # trading days over which it unfolds
    vfut_move: float       # cumulative short-VIX-future move driving UVIX/SVIX


CRASH_SCENARIOS = (
    CrashScenario("mild_-20%", -0.20, 2.0, 15, 1.2),
    CrashScenario("severe_-30%", -0.30, 3.0, 20, 2.0),
    CrashScenario("volmageddon_-40%", -0.40, 4.5, 10, 3.5),
)


def crash_combined_pnl(
    rho: float,
    gross: float,
    put_cfg: PutOverlayConfig,
    *,
    spot: float,
    atm_iv: float,
    put_dte: int,
    put_notional_frac: float,
    scn: CrashScenario,
) -> dict:
    """Approximate combined P&L (fraction of equity) for ``scn``.

    Carry leg: both legs short. UVIX return = clip(2*vfut, floor -1), SVIX
    return = clip(-vfut, floor -1). Short P&L = -(wU*rU + wS*rS) * gross.
    Put leg: long puts repriced at crashed spot + spiked, skewed IV.
    ``put_notional_frac`` is put premium book value as a fraction of equity
    (so the two legs are on the same scale).
    """
    g = scn.vfut_move
    r_u = max(2.0 * g, -1.0)
    r_s = max(-g, -1.0)
    w_u = 1.0 / (1.0 + rho)
    w_s = rho / (1.0 + rho)
    carry_pnl = -gross * (w_u * r_u + w_s * r_s)

    # Put leg: value before vs after.
    strike = spot * (1.0 - put_cfg.otm_pct)
    t0 = max(put_dte / 252.0, 1 / 365)
    iv0 = effective_iv(atm_iv, put_cfg.otm_pct, put_cfg)
    v0 = bs_put(spot, strike, t0, iv0, put_cfg.risk_free)

    spot1 = spot * (1.0 + scn.spx_drop)
    t1 = max((put_dte - scn.days) / 252.0, 1 / 365)
    atm1 = atm_iv * scn.vix_mult
    money1 = max(0.0, 1.0 - strike / spot1)
    iv1 = effective_iv(atm1, money1, put_cfg)
    v1 = bs_put(spot1, strike, t1, iv1, put_cfg.risk_free)

    put_pnl = put_notional_frac * (v1 / v0 - 1.0) if v0 > 0 else 0.0
    return {
        "scenario": scn.name,
        "carry_pnl_frac": carry_pnl,
        "put_pnl_frac": put_pnl,
        "combined_pnl_frac": carry_pnl + put_pnl,
    }


# ===========================================================================
# Shared run helpers
# ===========================================================================
def _window_return(equity: pd.Series, start: str, end: str) -> float:
    a, b = pd.Timestamp(start), pd.Timestamp(end)
    w = equity.loc[(equity.index >= a) & (equity.index <= b)]
    if len(w) < 2:
        return float("nan")
    return float(w.iloc[-1] / w.iloc[0] - 1.0)


HIST_STRESS = (
    ("feb18", "2018-02-01", "2018-02-15"),
    ("mar20", "2020-02-20", "2020-03-23"),
    ("aug24", "2024-08-01", "2024-08-31"),
)


def _aug2024(put_bt: pd.DataFrame) -> tuple[float, float]:
    a, b = pd.Timestamp("2024-08-01"), pd.Timestamp("2024-08-31")
    w = put_bt.loc[(put_bt.index >= a) & (put_bt.index <= b)]
    if len(w) < 2:
        return np.nan, np.nan
    carry = w["equity_carry"].iloc[-1] / w["equity_carry"].iloc[0] - 1
    comb = w["combined_equity"].iloc[-1] / w["combined_equity"].iloc[0] - 1
    return carry, comb


def _run_one(panel, rebal, spx, iv, rho_s, gross_s, put_cfg):
    carry = run_carry_backtest(panel, rho_s, rebal, gross_daily=gross_s)
    puts = run_put_overlay(
        panel.index, carry["equity"], spx, iv, put_cfg, carry_pnl=carry["financing_pnl"]
    )
    return carry, puts


def _put_notional_frac(puts: pd.DataFrame) -> float:
    """Average put MTM as a fraction of carry equity (for crash scaling)."""
    m = puts["put_mtm"] / puts["equity_carry"].replace(0, np.nan)
    return float(m.replace([np.inf, -np.inf], np.nan).dropna().tail(252).mean() or 0.0)


# ===========================================================================
# EXPERIMENT A -- deep OTM put sweep
# ===========================================================================
def experiment_a(panel, rebal, spx, iv, *, rho: float = 1.0) -> pd.DataFrame:
    otm_grid = [0.20, 0.25, 0.30, 0.35]
    size_grid = [0.0025, 0.005, 0.010, 0.020]  # premium frac equity / roll
    rho_s = pd.Series(rho, index=panel.index)
    gross_s = pd.Series(1.0, index=panel.index)
    spot0 = float(spx.reindex(panel.index).ffill().iloc[-1])
    atm0 = float((panel["vix"] / 100.0).iloc[-1])

    rows = []
    for otm in otm_grid:
        for size in size_grid:
            put_cfg = PutOverlayConfig(otm_pct=otm, premium_frac_equity=size, carry_budget_frac=10.0)
            carry, puts = _run_one(panel, rebal, spx, iv, rho_s, gross_s, put_cfg)
            cstats = perf_stats(carry)
            n = len(puts)
            comb_total = puts["combined_equity"].iloc[-1] / puts["combined_equity"].iloc[0] - 1
            comb_cagr = (1 + comb_total) ** (252 / (n - 1)) - 1
            comb_dd = puts["drawdown_combined"].min()
            premium = puts["premium_spent"].sum()
            carry_gain = carry["equity"].iloc[-1] - carry["equity"].iloc[0]
            aug_carry, aug_comb = _aug2024(puts)
            pnf = _put_notional_frac(puts)

            crash = {
                scn.name: crash_combined_pnl(
                    rho, 1.0, put_cfg, spot=spot0, atm_iv=atm0,
                    put_dte=put_cfg.buy_dte, put_notional_frac=pnf, scn=scn,
                )["combined_pnl_frac"]
                for scn in CRASH_SCENARIOS
            }
            rows.append(
                {
                    "otm_pct": otm,
                    "size_frac": size,
                    "carry_CAGR": cstats["CAGR"],
                    "combined_CAGR": comb_cagr,
                    "drag_bps": (cstats["CAGR"] - comb_cagr) * 1e4,
                    "combined_MaxDD": comb_dd,
                    "premium_paid": premium,
                    "carry_gain": carry_gain,
                    "self_funded": carry_gain > premium,
                    "rolls": puts.attrs.get("roll_count", 0),
                    "aug24_combined": aug_comb,
                    **{f"crash_{k}": v for k, v in crash.items()},
                }
            )
    return pd.DataFrame(rows)


# ===========================================================================
# PUT GRID -- rho=1.5 x OTM(20/25/30%) x equity size (1.5-4%)
# ===========================================================================
def experiment_put_grid(
    panel,
    rebal,
    spx,
    iv,
    *,
    rho: float = 1.5,
    otm_grid: tuple[float, ...] = (0.20, 0.25, 0.30),
    size_grid: tuple[float, ...] = (0.015, 0.02, 0.025, 0.03, 0.035, 0.04),
) -> pd.DataFrame:
    """Deep-put sweep at fixed rho (default 1.5, weekly rebalance assumed upstream)."""
    rho_s = pd.Series(rho, index=panel.index)
    gross_s = pd.Series(1.0, index=panel.index)
    spot0 = float(spx.reindex(panel.index).ffill().iloc[-1])
    atm0 = float((panel["vix"] / 100.0).iloc[-1])

    rows = []
    for otm in otm_grid:
        for size in size_grid:
            put_cfg = PutOverlayConfig(
                otm_pct=otm,
                premium_frac_equity=size,
                carry_budget_frac=10.0,
            )
            carry, puts = _run_one(panel, rebal, spx, iv, rho_s, gross_s, put_cfg)
            cstats = perf_stats(carry)
            n = len(puts)
            comb_total = puts["combined_equity"].iloc[-1] / puts["combined_equity"].iloc[0] - 1
            comb_cagr = (1 + comb_total) ** (252 / (n - 1)) - 1 if comb_total > -1 else np.nan
            comb_dd = float(puts["drawdown_combined"].min())
            comb_ret = puts["combined_ret"]
            comb_vol = float(comb_ret.std() * np.sqrt(252)) if len(comb_ret) > 2 else np.nan
            comb_sharpe = (
                float(comb_ret.mean() / comb_ret.std() * np.sqrt(252))
                if comb_ret.std() > 0
                else np.nan
            )
            comb_calmar = comb_cagr / abs(comb_dd) if comb_dd < 0 and np.isfinite(comb_cagr) else np.nan
            premium = float(puts["premium_spent"].sum())
            carry_gain = float(carry["equity"].iloc[-1] - carry["equity"].iloc[0])
            aug_carry, aug_comb = _aug2024(puts)
            pnf = _put_notional_frac(puts)
            crash = {
                scn.name: crash_combined_pnl(
                    rho, 1.0, put_cfg, spot=spot0, atm_iv=atm0,
                    put_dte=put_cfg.buy_dte, put_notional_frac=pnf, scn=scn,
                )["combined_pnl_frac"]
                for scn in CRASH_SCENARIOS
            }
            rows.append(
                {
                    "rho": rho,
                    "otm_pct": otm,
                    "size_pct": size * 100,
                    "carry_CAGR": cstats["CAGR"],
                    "combined_CAGR": comb_cagr,
                    "combined_MaxDD": comb_dd,
                    "combined_Sharpe": comb_sharpe,
                    "combined_Calmar": comb_calmar,
                    "combined_Vol": comb_vol,
                    "drag_bps": (cstats["CAGR"] - comb_cagr) * 1e4 if np.isfinite(comb_cagr) else np.nan,
                    "premium_paid": premium,
                    "self_funded": carry_gain > premium,
                    "rolls": puts.attrs.get("roll_count", 0),
                    "aug24_carry": aug_carry,
                    "aug24_combined": aug_comb,
                    **{f"crash_{k}": v for k, v in crash.items()},
                }
            )
    return pd.DataFrame(rows)


# ===========================================================================
# EXPERIMENT B -- dynamic ratio sizing policies
# ===========================================================================
def experiment_b(panel, rebal, spx, iv, *, put_cfg: PutOverlayConfig | None = None) -> pd.DataFrame:
    put_cfg = put_cfg or PutOverlayConfig(otm_pct=0.25, premium_frac_equity=0.005, carry_budget_frac=0.5)
    policies = build_policies(panel["ratio"])
    spot0 = float(spx.reindex(panel.index).ffill().iloc[-1])
    atm0 = float((panel["vix"] / 100.0).iloc[-1])

    rows = []
    for pol in policies:
        carry, puts = _run_one(panel, rebal, spx, iv, pol.rho, pol.gross, put_cfg)
        cstats = perf_stats(carry)
        n = len(puts)
        comb_total = puts["combined_equity"].iloc[-1] / puts["combined_equity"].iloc[0] - 1
        comb_cagr = (1 + comb_total) ** (252 / (n - 1)) - 1
        comb_dd = puts["drawdown_combined"].min()
        aug_carry, aug_comb = _aug2024(puts)
        pnf = _put_notional_frac(puts)
        avg_rho = float(pol.rho.mean())
        avg_gross = float(pol.gross.mean())
        rho_turn = float(pol.rho.diff().abs().sum())

        crash = {
            scn.name: crash_combined_pnl(
                avg_rho, avg_gross, put_cfg, spot=spot0, atm_iv=atm0,
                put_dte=put_cfg.buy_dte, put_notional_frac=pnf, scn=scn,
            )["combined_pnl_frac"]
            for scn in CRASH_SCENARIOS
        }
        rows.append(
            {
                "policy": pol.name,
                "avg_rho": avg_rho,
                "avg_gross": avg_gross,
                "rho_turnover": rho_turn,
                "carry_CAGR": cstats["CAGR"],
                "carry_MaxDD": cstats["Max Drawdown"],
                "carry_Sharpe": cstats["Sharpe"],
                "carry_Calmar": cstats["Calmar"],
                "combined_CAGR": comb_cagr,
                "combined_MaxDD": comb_dd,
                "aug24_carry": aug_carry,
                "aug24_combined": aug_comb,
                **{f"crash_{k}": v for k, v in crash.items()},
            }
        )
    return pd.DataFrame(rows)


# ===========================================================================
# EXPERIMENT C -- rho (1..2) x static/dynamic x rebalance frequency
# ===========================================================================
# Rebalance-frequency labels -> pandas anchor passed to rebalance_dates.
FREQS = {
    "daily": "B",
    "weekly": "W-FRI",
    "biweekly": "2W-FRI",
    "monthly": "BME",
}


def dynamic_rho_centered(
    ratio: pd.Series, center: float, *, half: float = 0.5, r_lo: float = 0.85, r_hi: float = 0.97
) -> pd.Series:
    """rho that ramps with the VIX/VIX3M ratio, averaging ~``center``.

    Contango (low ratio) -> center-half (lean into carry, more short UVIX);
    backwardation (high ratio) -> center+half (more short SVIX hedge). Clipped
    to [1.0, 2.0] so the sweep stays in the requested band.
    """
    frac = ((ratio - r_lo) / (r_hi - r_lo)).clip(0, 1)
    rho = (center - half) + (2 * half) * frac
    return rho.clip(1.0, 2.0).rename("rho")


def experiment_c(panel, spx, iv, *, put_cfg: PutOverlayConfig | None = None) -> pd.DataFrame:
    put_cfg = put_cfg or PutOverlayConfig(otm_pct=0.25, premium_frac_equity=0.005, carry_budget_frac=0.5)
    rho_grid = [1.0, 1.25, 1.5, 1.75, 2.0]
    spot0 = float(spx.reindex(panel.index).ffill().iloc[-1])
    atm0 = float((panel["vix"] / 100.0).iloc[-1])

    rows = []
    for freq_name, freq in FREQS.items():
        rebal = rebalance_dates(panel.index, freq)
        for rho in rho_grid:
            for mode in ("static", "dynamic"):
                if mode == "static":
                    rho_s = pd.Series(rho, index=panel.index)
                else:
                    rho_s = dynamic_rho_centered(panel["ratio"], rho)
                gross_s = pd.Series(1.0, index=panel.index)  # gross fixed -> isolate rho
                carry, puts = _run_one(panel, rebal, spx, iv, rho_s, gross_s, put_cfg)
                cs = perf_stats(carry)
                n = len(puts)
                comb_total = puts["combined_equity"].iloc[-1] / puts["combined_equity"].iloc[0] - 1
                comb_cagr = (1 + comb_total) ** (252 / (n - 1)) - 1
                aug_carry, aug_comb = _aug2024(puts)
                pnf = _put_notional_frac(puts)
                vman = crash_combined_pnl(
                    float(rho_s.mean()), 1.0, put_cfg, spot=spot0, atm_iv=atm0,
                    put_dte=put_cfg.buy_dte, put_notional_frac=pnf,
                    scn=CRASH_SCENARIOS[1],  # severe -30%
                )["combined_pnl_frac"]
                rows.append(
                    {
                        "freq": freq_name,
                        "mode": mode,
                        "rho_target": rho,
                        "avg_rho": float(rho_s.mean()),
                        "carry_CAGR": cs["CAGR"],
                        "carry_MaxDD": cs["Max Drawdown"],
                        "carry_Sharpe": cs["Sharpe"],
                        "carry_Calmar": cs["Calmar"],
                        "ann_friction_$": cs["Total Friction $"] / ((n - 1) / 252.0),
                        "combined_CAGR": comb_cagr,
                        "aug24_combined": aug_comb,
                        "crash_severe30": vman,
                        "rebalances": cs["Rebalances"],
                        **{
                            f"{name}_carry": _window_return(puts["equity_carry"], s, e)
                            for name, s, e in HIST_STRESS
                        },
                    }
                )
    return pd.DataFrame(rows)


# ===========================================================================
# EXPERIMENT D -- what makes the crash payoff positive? (incl. 1x3 backspread)
# ===========================================================================
def experiment_d(panel, rebal, spx, iv) -> dict:
    """Solve for configs with POSITIVE crash payoff and price the 1x3 backspread.

    Part 1: grid over (rho, gross, hedge kind, hedge size) -> combined crash P&L
            fraction for each scenario; flag positive.
    Part 2: in-sample calm-market drag of the 1x3 backspread vs a plain deep put,
            confirming the backspread is ~self-financing.
    """
    spot0 = float(spx.reindex(panel.index).ffill().iloc[-1])
    atm0 = float((panel["vix"] / 100.0).iloc[-1])

    # ---- Part 1: crash-payoff solver -------------------------------------
    rho_grid = [1.0, 1.5, 2.0]
    gross_grid = [1.0, 0.5, 0.25]
    hedge_specs = [
        ("none", None),
        ("deep_put_30_1.5%", ("deep_put", 0.30, 0.015)),
        ("deep_put_30_3%", ("deep_put", 0.30, 0.030)),
        ("backspread_12x30_1.5%", ("backspread", 0.30, 0.015)),
        ("backspread_12x30_3%", ("backspread", 0.30, 0.030)),
        ("backspread_15x35_3%", ("backspread", 0.35, 0.030)),
    ]
    rows = []
    for rho in rho_grid:
        w_u = 1.0 / (1.0 + rho)
        w_s = rho / (1.0 + rho)
        for gross in gross_grid:
            for hname, hspec in hedge_specs:
                rec = {"rho": rho, "gross": gross, "hedge": hname}
                for scn in CRASH_SCENARIOS:
                    g = scn.vfut_move
                    r_u = max(2.0 * g, -1.0)
                    r_s = max(-g, -1.0)
                    carry_pnl = -gross * (w_u * r_u + w_s * r_s)
                    hedge_pnl = 0.0
                    if hspec is not None:
                        kind, otm_far, size = hspec
                        otm_near = 0.12 if otm_far <= 0.30 else 0.15
                        hedge_pnl = hedge_crash_value(
                            spot=spot0, atm_iv=atm0, dte=126,
                            spx_drop=scn.spx_drop, vix_mult=scn.vix_mult, days=scn.days,
                            kind=kind, otm_far=otm_far, otm_near=otm_near,
                            long_premium_frac_equity=size,
                        )
                    rec[f"{scn.name}"] = carry_pnl + hedge_pnl
                rows.append(rec)
    crash_df = pd.DataFrame(rows)

    # ---- Part 2: calm-market drag of backspread vs deep put --------------
    carry = run_carry_backtest(panel, pd.Series(1.5, index=panel.index), rebal)
    eq = carry["equity"]
    deep = run_put_overlay(
        panel.index, eq, spx, iv,
        PutOverlayConfig(otm_pct=0.30, premium_frac_equity=0.015, carry_budget_frac=10.0),
    )
    bspread = run_backspread_overlay(
        panel.index, eq, spx, iv,
        BackspreadConfig(otm_near=0.12, otm_far=0.30, far_ratio=3, long_premium_frac_equity=0.015),
    )
    n = len(eq)
    ann = 252 / (n - 1)
    drag = pd.DataFrame(
        {
            "structure": ["carry_only(rho=1.5)", "+deep_put_30", "+backspread_12x30"],
            "CAGR": [
                (eq.iloc[-1] / eq.iloc[0]) ** ann - 1,
                (deep["combined_equity"].iloc[-1] / deep["combined_equity"].iloc[0]) ** ann - 1,
                (bspread["combined_equity"].iloc[-1] / bspread["combined_equity"].iloc[0]) ** ann - 1,
            ],
            "MaxDD": [
                carry["drawdown"].min(),
                deep["drawdown_combined"].min(),
                bspread["drawdown_combined"].min(),
            ],
            "rolls": [np.nan, deep.attrs.get("roll_count", 0), bspread.attrs.get("roll_count", 0)],
        }
    )
    return {"crash": crash_df, "drag": drag}


ASSUMPTIONS = """\
ASSUMPTIONS (all experiments)
  Data           : LIVE era default (UVIX/SVIX from 2022-03-30, yfinance).
                   Pass EXTENDED on CLI for synthetic pre-2022 (^SHORTVOL) history.
  Carry legs     : SHORT UVIX (+2x vol-fut beta) & SHORT SVIX (-1x). Daily-reset.
  rho            : SVIX-short / UVIX-short notional. Gross book = gross_mult*equity.
  Borrow         : UVIX 2.84%/yr, SVIX 3.47%/yr (borrow_cache.csv). No short-proceeds
                   credit; borrow does not widen in stress in this backtest.
  Costs          : 1 bp commission + 20 bps slippage per rebalanced notional.
  Initial capital: $1,000,000.
  SPX puts       : ThetaData EOD at roll entry + MTM when cached (THETADATA_API_KEY);
                   else Black-Scholes + skew (+3 vol pts / 10% OTM).
  Crash scenarios: synthetic instantaneous shocks (see experiment D).
"""


# ===========================================================================
# Runner
# ===========================================================================
def main(which: str = "AB", *, live_only: bool = True) -> None:
    if live_only:
        panel = load_vol_panel(start=INCEPTION, use_synthetic=False)
    else:
        panel = load_vol_panel()
    rebal = rebalance_dates(panel.index)
    spx = load_spx_spot(panel.index.min().strftime("%Y-%m-%d"))
    iv = (panel["vix"] / 100.0).rename("iv")

    dest = Path("data/runs") / pd.Timestamp.today().strftime("%Y-%m-%d") / "bucket5"
    dest = dest / ("live" if live_only else "extended")
    dest.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 240)
    pd.set_option("display.max_columns", 40)

    print(ASSUMPTIONS)
    era = "LIVE ONLY (2022-03-30+)" if live_only else "EXTENDED (synthetic pre-2022)"
    print(f"era: {era}")
    print(f"sample: {panel.index.min().date()} -> {panel.index.max().date()}  ({len(panel)} rows)")
    try:
        from scripts.bucket5_theta import prefetch_roll_puts, theta_available
    except ImportError:
        from bucket5_theta import prefetch_roll_puts, theta_available  # type: ignore
    if theta_available():
        pf = prefetch_roll_puts(panel.index, spx.reindex(panel.index).ffill())
        print(f"theta prefetch: {pf}")
    else:
        print("theta: THETADATA_API_KEY not set -> BS skew put pricing")

    if "A" in which:
        a = experiment_a(panel, rebal, spx, iv)
        a.to_csv(dest / "experiment_A_deep_otm.csv", index=False)
        print("\n=== EXPERIMENT A: deep-OTM put sweep (rho=1.0) ===")
        print("(crash_* = combined P&L as fraction of equity in each synthetic crash)")
        print(a.round(4).to_string(index=False))

    if "B" in which:
        b = experiment_b(panel, rebal, spx, iv)
        b.to_csv(dest / "experiment_B_ratio_policies.csv", index=False)
        print("\n=== EXPERIMENT B: dynamic ratio sizing policies (25% OTM puts) ===")
        print(b.round(4).to_string(index=False))

    if "C" in which:
        c = experiment_c(panel, spx, iv)
        c.to_csv(dest / "experiment_C_rho_freq.csv", index=False)
        print("\n=== EXPERIMENT C: rho(1..2) x static/dynamic x rebalance frequency ===")
        print("(gross fixed at 1.0 to isolate rho; crash_severe30 = combined P&L frac in -30% scenario)")
        print(c.round(4).to_string(index=False))

    if "G" in which:
        g = experiment_put_grid(panel, rebal, spx, iv)
        g.to_csv(dest / "experiment_put_grid_rho15.csv", index=False)
        print("\n=== PUT GRID: rho=1.5 weekly x OTM(20/25/30%) x size(1.5-4% equity) ===")
        print(g.round(4).to_string(index=False))

    if "D" in which:
        d = experiment_d(panel, rebal, spx, iv)
        d["crash"].to_csv(dest / "experiment_D_crash_solver.csv", index=False)
        d["drag"].to_csv(dest / "experiment_D_backspread_drag.csv", index=False)
        print("\n=== EXPERIMENT D.1: crash payoff solver (combined P&L as frac of equity) ===")
        print("(positive numbers = the product MAKES money in that crash)")
        print(d["crash"].round(3).to_string(index=False))
        print("\n=== EXPERIMENT D.2: calm-market drag, 1x3 backspread vs deep put (rho=1.5) ===")
        print(d["drag"].round(4).to_string(index=False))

    print(f"\nsaved -> {dest}")


if __name__ == "__main__":
    args = [a.upper() for a in sys.argv[1:]]
    live_only = "EXTENDED" not in args
    which = "".join(a for a in args if a not in ("LIVE", "EXTENDED", "PUTGRID"))
    if "PUTGRID" in args:
        which = which + "G"
    if not which:
        which = "ABCD"
    main(which, live_only=live_only)
