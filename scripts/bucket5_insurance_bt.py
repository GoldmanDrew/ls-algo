"""
Bucket 5 "insurance product" backtest.

This is a structural variant of the short-UVIX / short-SVIX carry sleeve aimed at
turning the strategy into a *positive-carry tail-hedge* whose crash outcome is an
insurance PAYOUT rather than a drawdown. Three ideas, all live-era testable:

1. ADAPTIVE ~14-DAY CADENCE (ported from Bucket 4's tr_vcr cadence engine)
   Bucket 4 rebalances on a state-dependent clock:
       interval = clip(round(base_days / denom), min, max)
   We reuse the same *shape* but drive ``denom`` with the Bucket-5 regime signal
   (the VIX/VIX3M "simple ratio"): calm contango -> stretch toward ``base_days``
   (~14, biweekly); flattening / backwardation -> rebalance much faster so the
   book is re-hedged into a spike instead of drifting.

2. CONTANGO / BACKWARDATION POSITION POLICY  (``regime_rho_gross``)
   rho = SVIX-short / UVIX-short notional.  Short UVIX is the decay/carry engine
   (it blows up in a spike); short SVIX is the convex hedge (it pays in a spike).
   So we move the legs with the term structure:
       deep contango (ratio low)  -> LOWER rho  (more short UVIX) -> harvest decay
       flat / backwardation (high)-> RAISE rho toward 2 (more short SVIX) AND cut
                                     the sleeve gross (de-risk the whole book).

3. CAPITAL-EFFICIENT COLLATERAL + PUT LADDER
   Run the short/short at a *small* gross fraction (``sleeve_frac``) so the sleeve
   net carry is only ~4-8%/yr, park the rest of the equity in T-bills / box
   spreads (``tbill_rate``), and spend part of the (carry + bill) income on a
   LADDER of long SPX puts (several strikes, 6M->3M roll). Sized so a crash is a
   net positive payout.

Run::

    python scripts/bucket5_insurance_bt.py            # base config, live era
    python scripts/bucket5_insurance_bt.py sweep      # sleeve_frac x ladder sweep
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scripts.bucket5_carry_bt import perf_stats, run_carry_backtest
    from scripts.bucket5_data import DEFAULT_BORROW_SVIX, DEFAULT_BORROW_UVIX, INCEPTION, load_vol_panel
    from scripts.bucket5_put_overlay import (
        BackspreadConfig,
        PutOverlayConfig,
        _pick_expiry,
        _trading_dte,
        bs_put,
        effective_iv,
        hedge_crash_value,
        load_spx_spot,
        run_backspread_overlay,
        run_put_overlay,
    )
    from scripts.bucket5_experiments import CRASH_SCENARIOS, CrashScenario
except ImportError:
    from bucket5_carry_bt import perf_stats, run_carry_backtest  # type: ignore
    from bucket5_data import DEFAULT_BORROW_SVIX, DEFAULT_BORROW_UVIX, INCEPTION, load_vol_panel  # type: ignore
    from bucket5_put_overlay import (  # type: ignore
        BackspreadConfig,
        PutOverlayConfig,
        _pick_expiry,
        _trading_dte,
        bs_put,
        effective_iv,
        hedge_crash_value,
        load_spx_spot,
        run_backspread_overlay,
        run_put_overlay,
    )
    from bucket5_experiments import CRASH_SCENARIOS, CrashScenario  # type: ignore


# ===========================================================================
# 1. Adaptive cadence (contango-driven, Bucket-4 style ~14-day base)
# ===========================================================================
def adaptive_rebal_dates(
    ratio: pd.Series,
    *,
    base_days: float = 14.0,
    k_stress: float = 6.0,
    r_lo: float = 0.88,
    r_hi: float = 1.00,
    min_interval: int = 2,
    max_interval: int = 21,
) -> pd.DatetimeIndex:
    """State-dependent rebalance schedule driven by the VIX/VIX3M ratio.

    ``denom = 1 + k_stress * clip((ratio - r_lo)/(r_hi - r_lo), 0, 1)`` so:
      * deep contango (ratio <= r_lo)  -> denom 1     -> interval = base_days (~14)
      * backwardation (ratio >= r_hi)  -> denom 1+k   -> interval ~ base/(1+k)

    Mirrors ``bucket4_hedge_cadence.build_rebal_dates`` (step the calendar by the
    current interval) but uses the term-structure signal instead of TR/VCR.
    """
    cal = pd.DatetimeIndex(ratio.index).sort_values().unique()
    dates: list[pd.Timestamp] = []
    i, n = 0, len(cal)
    while i < n:
        d = pd.Timestamp(cal[i])
        dates.append(d)
        r = float(ratio.loc[d]) if d in ratio.index else np.nan
        stress = 0.0 if not np.isfinite(r) else float(np.clip((r - r_lo) / (r_hi - r_lo), 0.0, 1.0))
        denom = 1.0 + k_stress * stress
        interval = int(np.clip(round(base_days / denom), min_interval, max_interval))
        i += max(min_interval, interval)
    return pd.DatetimeIndex(dates)


# ===========================================================================
# 2. Contango / backwardation position policy
# ===========================================================================
@dataclass
class RegimePolicy:
    """Maps VIX/VIX3M ratio -> (rho, gross multiplier of the sleeve)."""

    rho_contango: float = 1.0     # deep contango -> lean into carry (more short UVIX)
    rho_backwardation: float = 2.0  # backwardation -> vol-neutral (more short SVIX)
    gross_contango: float = 1.0    # full sleeve when calm
    gross_backwardation: float = 0.35  # cut the book in stress
    r_lo: float = 0.88
    r_hi: float = 1.00

    def series(self, ratio: pd.Series) -> tuple[pd.Series, pd.Series]:
        frac = ((ratio - self.r_lo) / (self.r_hi - self.r_lo)).clip(0.0, 1.0)
        rho = (self.rho_contango + (self.rho_backwardation - self.rho_contango) * frac).rename("rho")
        gross = (self.gross_contango + (self.gross_backwardation - self.gross_contango) * frac).rename("gross")
        return rho, gross


# ===========================================================================
# 3. Put ladder overlay (multi-strike, reuses tested single-strike pricing)
# ===========================================================================
@dataclass
class LadderRung:
    otm_pct: float
    per_roll_frac: float   # premium budget for this rung, fraction of equity per roll
    quantity_multiplier: int = 1  # production can scale the pre-existing integer quantity exactly


def build_ladder(strikes_weights: list[tuple[float, float]], total_per_roll: float) -> list[LadderRung]:
    """Build rungs from (otm_pct, weight) pairs; weights need not sum to 1."""
    tw = sum(w for _, w in strikes_weights) or 1.0
    return [LadderRung(otm, total_per_roll * w / tw) for otm, w in strikes_weights]


@dataclass
class HedgeBudgetPolicy:
    """Regime- and vol-conditional scaling of total hedge premium per roll.

    Idea: in deep contango / low VIX, tail puts are *cheap* -> spend MORE (``contango_mult``,
    ``vix_calm_boost``). When the term structure is already flat/inverted, throttle new
    premium slightly (``stress_mult``) because existing positions are already live.
    """

    enabled: bool = True
    contango_mult: float = 1.20
    stress_mult: float = 0.85
    vix_lo: float = 14.0
    vix_hi: float = 28.0
    vix_calm_boost: float = 1.10
    r_lo: float = 0.88
    r_hi: float = 1.00

    def multiplier(self, ratio_val: float, vix_val: float) -> float:
        if not self.enabled:
            return 1.0
        if np.isfinite(ratio_val):
            stress = float(np.clip((ratio_val - self.r_lo) / (self.r_hi - self.r_lo), 0.0, 1.0))
            m = self.contango_mult + (self.stress_mult - self.contango_mult) * stress
        else:
            m = 1.0
        if np.isfinite(vix_val) and self.vix_hi > self.vix_lo:
            calm = float(np.clip((self.vix_hi - vix_val) / (self.vix_hi - self.vix_lo), 0.0, 1.0))
            m *= 1.0 + (self.vix_calm_boost - 1.0) * calm
        return max(0.25, m)


@dataclass
class BackspreadHedge:
    otm_near: float = 0.12
    otm_far: float = 0.30
    far_ratio: int = 3
    premium_frac: float = 0.024


def _map_backspread(out: pd.DataFrame) -> pd.DataFrame:
    """Normalize backspread overlay columns to hedge-layer schema."""
    mapped = pd.DataFrame({
        "put_mtm": out["bs_mtm"],
        "put_cash_flow": out["bs_cash_flow"],
        "realized": pd.Series(0.0, index=out.index),
    }, index=out.index)
    mapped.attrs["roll_count"] = out.attrs.get("roll_count", 0)
    mapped.attrs["realized_total"] = 0.0
    mapped.attrs["per_rung_notional_frac"] = {}
    mapped.attrs["hedge_kind"] = "backspread"
    return mapped


def run_put_ladder(
    dates: pd.DatetimeIndex,
    equity: pd.Series,
    spot: pd.Series,
    iv: pd.Series,
    rungs: list[LadderRung],
    *,
    buy_dte: int = 126,
    roll_dte: int = 63,
    risk_free: float = 0.04,
) -> pd.DataFrame:
    """Sum independent long-put rungs (each 6M->3M) into one ladder MTM series.

    Each rung is a standalone ``run_put_overlay`` on the *same* equity path with
    its own strike and budget; total ladder MTM / cash flow is the sum. The Theta
    pricer (when ``THETADATA_API_KEY`` is set) is used per rung automatically.
    """
    idx = dates.intersection(equity.index).intersection(spot.index).intersection(iv.index)
    put_mtm = pd.Series(0.0, index=idx)
    put_cash = pd.Series(0.0, index=idx)
    premium_total = 0.0
    rolls = 0
    per_rung = {}
    for rg in rungs:
        cfg = PutOverlayConfig(
            otm_pct=rg.otm_pct,
            buy_dte=buy_dte,
            roll_dte=roll_dte,
            premium_frac_equity=rg.per_roll_frac,
            carry_budget_frac=10.0,   # do not gate on trailing carry here
            risk_free=risk_free,
        )
        sub = run_put_overlay(idx, equity, spot, iv, cfg)
        put_mtm = put_mtm.add(sub["put_mtm"], fill_value=0.0)
        put_cash = put_cash.add(sub["put_cash_flow"], fill_value=0.0)
        premium_total += float(sub["premium_spent"].sum())
        rolls = max(rolls, sub.attrs.get("roll_count", 0))
        # average MTM notional as a fraction of equity (for crash scaling)
        eqv = sub["equity_carry"].replace(0, np.nan)
        per_rung[rg.otm_pct] = float((sub["put_mtm"] / eqv).replace([np.inf, -np.inf], np.nan).dropna().tail(252).mean() or 0.0)

    out = pd.DataFrame({"put_mtm": put_mtm, "put_cash_flow": put_cash})
    out.attrs["premium_total"] = premium_total
    out.attrs["roll_count"] = rolls
    out.attrs["per_rung_notional_frac"] = per_rung
    return out


# ===========================================================================
# 3b. Profit-taking / monetization engine
# ===========================================================================
@dataclass
class MonetizeConfig:
    """Rules for harvesting put convexity when the hedge surges.

    profit_tiers : (multiple, frac_of_remaining_to_sell). Scale out as the put's
        value/cost-basis crosses each multiple (each tier fires once).
    vix_tiers    : (vix_level, frac_of_remaining). Override that monetizes on
        outright vol spikes between profit tiers.
    giveback_frac/min_mult : once up >= min_mult x, sell down (to the runner floor)
        if MTM falls giveback_frac from its peak (stops a 6x round-tripping to 1x).
    bank_frac    : on a FULL exit, fraction of proceeds banked/redeployed; the
        rest is re-armed into a fresh 6M OTM put at the lower spot (barbell).
    rearm        : enable redeploy-into-fresh-puts on full exit.

    runner_frac  : PATIENCE knob. Tiers / VIX / give-back can never sell the rung
        below this fraction of its entry size — a small "runner" stays long so a
        SECOND leg down still pays. The runner is only released (full exit) when
        ``mult >= runner_mult`` or at the scheduled 6M->3M roll / expiry.
    runner_mult  : multiple at which the runner itself is finally harvested.
    """

    profit_tiers: tuple[tuple[float, float], ...] = ((3.0, 0.34), (5.0, 0.5), (8.0, 1.0))
    vix_tiers: tuple[tuple[float, float], ...] = ((45.0, 0.5), (65.0, 1.0))
    giveback_frac: float = 0.35
    giveback_min_mult: float = 2.0
    bank_frac: float = 0.6
    rearm: bool = True
    runner_frac: float = 0.15
    runner_mult: float = 12.0


@dataclass
class RedeployPolicy:
    """Where harvested (banked) put proceeds go, conditioned on the regime at the
    time of harvest. The weight is the SLEEVE share; the remainder goes to T-bills.

    The economic case: a put harvest almost always coincides with a vol spike
    (flat / backwardated term structure). That is precisely when forward short-vol
    carry is richest (vol mean-reverts, contango re-steepens), so we lean the
    proceeds into the sleeve in backwardation and park them in bills when calm.
    The runner puts (see ``MonetizeConfig.runner_frac``) cover the second-leg risk
    that this sleeve redeploy takes on.
    """

    sleeve_w_contango: float = 0.20      # calm: mostly dry powder in bills
    sleeve_w_backwardation: float = 0.65  # post-spike: lean into short-vol reversion
    r_lo: float = 0.88
    r_hi: float = 1.00

    def sleeve_weight(self, ratio_val: float) -> float:
        if not np.isfinite(ratio_val):
            return self.sleeve_w_contango
        frac = float(np.clip((ratio_val - self.r_lo) / (self.r_hi - self.r_lo), 0.0, 1.0))
        return self.sleeve_w_contango + (self.sleeve_w_backwardation - self.sleeve_w_contango) * frac


def _theta_px(dt: pd.Timestamp, strike: float, theta_exp, s: float, atm: float,
              t_rem: float, pcfg: PutOverlayConfig) -> float:
    """Per-share put price: ThetaData cache if available, else Black-Scholes."""
    if theta_exp is not None:
        try:
            from scripts.bucket5_theta import put_mtm_on_date
        except ImportError:
            from bucket5_theta import put_mtm_on_date  # type: ignore
        px = put_mtm_on_date(pd.Timestamp(dt), strike, theta_exp)
        if px is not None:
            return float(px)
    iv = effective_iv(atm, max(0.0, 1.0 - strike / s), pcfg)
    return bs_put(s, strike, t_rem, iv, pcfg.risk_free)


def run_monetizing_put(
    idx: pd.DatetimeIndex,
    equity: pd.Series,
    spot: pd.Series,
    iv: pd.Series,
    vix: pd.Series,
    rung: LadderRung,
    mon: MonetizeConfig,
    *,
    ratio: pd.Series | None = None,
    hedge_budget: HedgeBudgetPolicy | None = None,
    buy_dte: int = 126,
    roll_dte: int = 63,
    risk_free: float = 0.04,
    min_gap: int = 55,
) -> pd.DataFrame:
    """Single long-put rung with profit-taking, give-back stop, and re-arm.

    Realized monetization proceeds flow through ``put_cash_flow``; unrealized
    value is ``put_mtm``. So ``combined = base + put_mtm + cumsum(put_cash_flow)``
    captures the harvested gains even after the put decays back.
    """
    pcfg = PutOverlayConfig(otm_pct=rung.otm_pct, buy_dte=buy_dte, roll_dte=roll_dte,
                            risk_free=risk_free)
    eq = equity.reindex(idx).ffill()
    spx = spot.reindex(idx).ffill()
    vol = iv.reindex(idx).ffill().clip(lower=0.08, upper=1.5)
    vx = vix.reindex(idx).ffill()
    ratio_s = ratio.reindex(idx).ffill() if ratio is not None else None

    contracts = 0
    entry_contracts = 0       # size at entry, for the runner floor
    strike = np.nan
    expiry: pd.Timestamp | None = None
    theta_exp = None
    cost_px = np.nan          # per-share entry premium (cost basis)
    peak_px = 0.0             # peak per-share MTM since entry
    tiers_fired: set[int] = set()
    vix_fired: set[int] = set()
    cash = 0.0               # cumulative premium + realized proceeds (signed)
    rows: list[dict] = []
    rolls = 0
    realized_total = 0.0
    days_since_roll = 10_000
    monetize_events: list[dict] = []

    def _budget_at(dt: pd.Timestamp, eq_i: float) -> float:
        raw = rung.per_roll_frac * eq_i
        if hedge_budget is None or not hedge_budget.enabled:
            return raw
        r = float(ratio_s.loc[dt]) if ratio_s is not None and dt in ratio_s.index else np.nan
        v = float(vx.loc[dt]) if dt in vx.index else np.nan
        return raw * hedge_budget.multiplier(r, v)

    def _open(dt, s, atm, budget) -> None:
        nonlocal contracts, entry_contracts, strike, expiry, theta_exp, cost_px, peak_px, tiers_fired, vix_fired, cash, rolls, days_since_roll
        new_expiry = _pick_expiry(idx, dt, pcfg.buy_dte, pcfg.roll_dte + 10)
        if new_expiry is None or budget <= 100:
            return
        k = s * (1.0 - pcfg.otm_pct)
        try:
            from scripts.bucket5_theta import put_mid_on_date
        except ImportError:
            from bucket5_theta import put_mid_on_date  # type: ignore
        px_buy = None
        theta_hit = put_mid_on_date(pd.Timestamp(dt), s, pcfg.otm_pct, _trading_dte(new_expiry, dt))
        te = None
        if theta_hit is not None:
            px_buy, meta = theta_hit
            k = float(meta.get("strike", k))
            te = meta.get("exp")
        if px_buy is None:
            t_buy = max(_trading_dte(new_expiry, dt) / 252.0, 1 / 252)
            px_buy = bs_put(s, k, t_buy, effective_iv(atm, pcfg.otm_pct, pcfg), pcfg.risk_free)
        if px_buy <= 1e-6:
            return
        base_n = max(1, int(budget / (px_buy * pcfg.contract_multiplier)))
        n = base_n * max(1, int(rung.quantity_multiplier))
        contracts = n
        entry_contracts = n
        strike = k
        expiry = new_expiry
        theta_exp = te
        cost_px = px_buy
        peak_px = px_buy
        tiers_fired = set()
        vix_fired = set()
        cash -= px_buy * pcfg.contract_multiplier * n
        rolls += 1
        days_since_roll = 0

    for i, dt in enumerate(idx):
        s = float(spx.loc[dt])
        atm = float(vol.loc[dt])
        eq_i = float(eq.loc[dt])
        vix_now = float(vx.loc[dt]) if np.isfinite(vx.loc[dt]) else np.nan
        realized_today = 0.0
        dte = _trading_dte(expiry, dt) if expiry is not None else 0

        # initial entry
        if i == 0 and contracts == 0 and s > 0 and eq_i > 0:
            _open(dt, s, atm, _budget_at(dt, eq_i))
            dte = _trading_dte(expiry, dt) if expiry is not None else 0

        # current per-share value
        px = 0.0
        if contracts > 0 and expiry is not None and np.isfinite(strike):
            t_rem = max(dte / 252.0, 1 / 365)
            px = _theta_px(dt, strike, theta_exp, s, atm, t_rem, pcfg)
            peak_px = max(peak_px, px)

        # ---- monetization (only with a live position) ----
        if contracts > 0 and np.isfinite(cost_px) and cost_px > 0 and px > 0:
            mult = px / cost_px
            sell_frac = 0.0
            giveback = False
            profit_hit: str | None = None
            vix_hit: str | None = None
            for j, (lvl, frac) in enumerate(mon.profit_tiers):
                if j not in tiers_fired and mult >= lvl:
                    tiers_fired.add(j)
                    sell_frac = max(sell_frac, frac)
                    profit_hit = f"profit_{lvl:g}x"
            for j, (lvl, frac) in enumerate(mon.vix_tiers):
                if j not in vix_fired and np.isfinite(vix_now) and vix_now >= lvl:
                    vix_fired.add(j)
                    sell_frac = max(sell_frac, frac)
                    vix_hit = f"vix_{lvl:g}"
            if (mult >= mon.giveback_min_mult and peak_px > 0
                    and px <= peak_px * (1.0 - mon.giveback_frac)):
                sell_frac = max(sell_frac, 1.0)
                giveback = True

            if sell_frac > 0:
                # Runner floor: never sell below runner_frac of entry size unless the
                # runner itself is being released (mult >= runner_mult). Patience: a
                # small stub stays long to capture a second leg down.
                runner_release = mult >= mon.runner_mult
                if runner_release or mon.runner_frac <= 0.0:
                    floor = 0
                else:
                    floor = min(contracts, int(np.ceil(mon.runner_frac * entry_contracts)))
                if sell_frac >= 0.999 or giveback:
                    target_remaining = floor
                else:
                    target_remaining = max(floor, contracts - int(round(contracts * sell_frac)))
                n_sell = contracts - target_remaining
                # a fired tier must trim >=1 contract whenever we are above the floor
                # (rungs hold few contracts, so fractional scale-outs round to zero).
                if n_sell <= 0 and contracts > floor:
                    n_sell = 1
                if n_sell > 0:
                    proceeds = px * pcfg.contract_multiplier * n_sell
                    contracts -= n_sell
                    if giveback:
                        kind = "giveback"
                    elif vix_hit:
                        kind = vix_hit
                    elif profit_hit:
                        kind = profit_hit
                    elif sell_frac >= 0.999:
                        kind = "full_exit"
                    else:
                        kind = "partial_trim"
                    monetize_events.append({
                        "date": pd.Timestamp(dt).strftime("%Y-%m-%d"),
                        "kind": kind,
                        "usd": round(float(proceeds), 2),
                        "otm_pct": round(float(rung.otm_pct), 4),
                        "mult": round(float(mult), 3),
                        "vix": round(float(vix_now), 2) if np.isfinite(vix_now) else None,
                        "contracts_sold": int(n_sell),
                    })
                    if contracts == 0:
                        # full exit -> bank a share, re-arm the rest into fresh puts
                        bank = proceeds * mon.bank_frac
                        cash += proceeds            # receive full sale
                        realized_today += bank      # banked portion is externally redeployable
                        rearm_budget = proceeds - bank
                        expiry = None
                        theta_exp = None
                        strike = np.nan
                        if mon.rearm and rearm_budget > 100 and s > 0:
                            _open(dt, s, atm, rearm_budget)  # spends rearm_budget from cash
                            monetize_events.append({
                                "date": pd.Timestamp(dt).strftime("%Y-%m-%d"),
                                "kind": "rearm",
                                "usd": round(float(rearm_budget), 2),
                                "otm_pct": round(float(rung.otm_pct), 4),
                                "mult": round(float(mult), 3),
                                "vix": round(float(vix_now), 2) if np.isfinite(vix_now) else None,
                                "contracts_sold": 0,
                            })
                            dte = _trading_dte(expiry, dt) if expiry is not None else 0
                            if contracts > 0 and np.isfinite(strike):
                                t_rem = max(dte / 252.0, 1 / 365)
                                px = _theta_px(dt, strike, theta_exp, s, atm, t_rem, pcfg)
                    else:
                        # partial scale-out: runner stays long; all proceeds redeployable
                        cash += proceeds
                        realized_today += proceeds

        # ---- scheduled 6M->3M roll on what remains ----
        need_roll = contracts > 0 and 0 < dte <= pcfg.roll_dte and days_since_roll >= min_gap
        if need_roll and s > 0 and eq_i > 0:
            t_rem = max(dte / 252.0, 1 / 365)
            px_sell = _theta_px(dt, strike, theta_exp, s, atm, t_rem, pcfg)
            roll_proceeds = px_sell * pcfg.contract_multiplier * contracts
            monetize_events.append({
                "date": pd.Timestamp(dt).strftime("%Y-%m-%d"),
                "kind": "scheduled_roll",
                "usd": round(float(roll_proceeds), 2),
                "otm_pct": round(float(rung.otm_pct), 4),
                "mult": round(float(px_sell / cost_px), 3) if np.isfinite(cost_px) and cost_px > 0 else None,
                "vix": round(float(vix_now), 2) if np.isfinite(vix_now) else None,
                "contracts_sold": int(contracts),
            })
            cash += roll_proceeds
            contracts = 0
            expiry = None
            theta_exp = None
            strike = np.nan
            _open(dt, s, atm, _budget_at(dt, eq_i))
            dte = _trading_dte(expiry, dt) if expiry is not None else 0
            px = 0.0
            if contracts > 0 and np.isfinite(strike):
                t_rem = max(dte / 252.0, 1 / 365)
                px = _theta_px(dt, strike, theta_exp, s, atm, t_rem, pcfg)
        days_since_roll += 1

        put_mtm = px * pcfg.contract_multiplier * contracts if contracts > 0 else 0.0
        realized_total += realized_today
        rows.append({
            "date": dt,
            "put_mtm": put_mtm,
            "put_cash_flow": cash - (rows[-1]["_cum_cash"] if rows else 0.0),
            "_cum_cash": cash,
            "realized": realized_today,
            "contracts": contracts,
            "mult": (px / cost_px) if (np.isfinite(cost_px) and cost_px > 0) else np.nan,
        })

    out = pd.DataFrame(rows).set_index("date")
    out.attrs["roll_count"] = rolls
    out.attrs["realized_total"] = realized_total
    out.attrs["monetize_events"] = monetize_events
    return out


def run_monetizing_ladder(
    dates: pd.DatetimeIndex,
    equity: pd.Series,
    spot: pd.Series,
    iv: pd.Series,
    vix: pd.Series,
    rungs: list[LadderRung],
    mon: MonetizeConfig,
    *,
    ratio: pd.Series | None = None,
    hedge_budget: HedgeBudgetPolicy | None = None,
    buy_dte: int = 126,
    roll_dte: int = 63,
    risk_free: float = 0.04,
) -> pd.DataFrame:
    """Sum monetizing rungs into one ladder MTM + realized-cash series."""
    idx = dates.intersection(equity.index).intersection(spot.index).intersection(iv.index)
    put_mtm = pd.Series(0.0, index=idx)
    put_cash = pd.Series(0.0, index=idx)
    realized = pd.Series(0.0, index=idx)
    realized_total = 0.0
    rolls = 0
    per_rung = {}
    monetize_events: list[dict] = []
    for rg in rungs:
        sub = run_monetizing_put(
            idx, equity, spot, iv, vix, rg, mon,
            ratio=ratio, hedge_budget=hedge_budget,
            buy_dte=buy_dte, roll_dte=roll_dte, risk_free=risk_free,
        )
        put_mtm = put_mtm.add(sub["put_mtm"], fill_value=0.0)
        put_cash = put_cash.add(sub["put_cash_flow"], fill_value=0.0)
        realized = realized.add(sub["realized"], fill_value=0.0)
        realized_total += sub.attrs.get("realized_total", 0.0)
        rolls = max(rolls, sub.attrs.get("roll_count", 0))
        monetize_events.extend(sub.attrs.get("monetize_events", []))
        eqv = equity.reindex(idx).ffill().replace(0, np.nan)
        per_rung[rg.otm_pct] = float((sub["put_mtm"] / eqv).replace([np.inf, -np.inf], np.nan).dropna().tail(252).mean() or 0.0)

    out = pd.DataFrame({"put_mtm": put_mtm, "put_cash_flow": put_cash, "realized": realized})
    out.attrs["roll_count"] = rolls
    out.attrs["realized_total"] = realized_total
    out.attrs["per_rung_notional_frac"] = per_rung
    out.attrs["hedge_kind"] = "ladder"
    out.attrs["monetize_events"] = sorted(monetize_events, key=lambda e: (e["date"], e.get("kind", "")))
    return out


def run_hedge_layer(
    dates: pd.DatetimeIndex,
    equity: pd.Series,
    spot: pd.Series,
    iv: pd.Series,
    vix: pd.Series,
    ratio: pd.Series,
    cfg: "InsuranceConfig",
) -> pd.DataFrame:
    """Dispatch to ladder (monetized), backspread, or hybrid hedge overlay."""
    kind = getattr(cfg, "hedge_kind", "ladder") or "ladder"
    mon = cfg.monetize

    if kind == "backspread":
        bs = cfg.backspread or BackspreadHedge()
        bcfg = BackspreadConfig(
            otm_near=bs.otm_near, otm_far=bs.otm_far, far_ratio=bs.far_ratio,
            long_premium_frac_equity=bs.premium_frac,
        )
        raw = run_backspread_overlay(dates, equity, spot, iv, bcfg)
        return _map_backspread(raw)

    if kind == "hybrid":
        split = float(getattr(cfg, "hybrid_ladder_frac", 0.70))
        bs = cfg.backspread or BackspreadHedge()
        lad_rungs = [LadderRung(r.otm_pct, r.per_roll_frac * split) for r in cfg.rungs]
        bs_frac = bs.premium_frac * (1.0 - split)
        parts: list[pd.DataFrame] = []
        if mon is not None:
            parts.append(run_monetizing_ladder(
                dates, equity, spot, iv, vix, lad_rungs, mon,
                ratio=ratio, hedge_budget=cfg.hedge_budget,
            ))
        else:
            parts.append(run_put_ladder(dates, equity, spot, iv, lad_rungs))
        bcfg = BackspreadConfig(
            otm_near=bs.otm_near, otm_far=bs.otm_far, far_ratio=bs.far_ratio,
            long_premium_frac_equity=bs_frac,
        )
        parts.append(_map_backspread(run_backspread_overlay(dates, equity, spot, iv, bcfg)))
        idx = parts[0].index
        out = pd.DataFrame({
            "put_mtm": sum(p["put_mtm"].reindex(idx).fillna(0.0) for p in parts),
            "put_cash_flow": sum(p["put_cash_flow"].reindex(idx).fillna(0.0) for p in parts),
            "realized": sum(p["realized"].reindex(idx).fillna(0.0) for p in parts),
        }, index=idx)
        out.attrs["hedge_kind"] = "hybrid"
        out.attrs["realized_total"] = float(sum(p.attrs.get("realized_total", 0.0) for p in parts))
        events: list[dict] = []
        for p in parts:
            events.extend(p.attrs.get("monetize_events", []))
        out.attrs["monetize_events"] = sorted(events, key=lambda e: (e["date"], e.get("kind", "")))
        return out

    if mon is not None:
        return run_monetizing_ladder(
            dates, equity, spot, iv, vix, cfg.rungs, mon,
            ratio=ratio, hedge_budget=cfg.hedge_budget,
        )
    return run_put_ladder(dates, equity, spot, iv, cfg.rungs)


# ===========================================================================
# 4. The insurance-product backtest
# ===========================================================================
@dataclass
class InsuranceConfig:
    sleeve_frac: float = 0.20          # production: 20% gross in short/short sleeve
    tbill_rate: float = 0.043          # collateral yield (T-bills / box spreads)
    regime: RegimePolicy = field(default_factory=RegimePolicy)
    base_days: float = 14.0            # adaptive cadence base (Bucket-4 14d)
    cadence_k: float = 6.0
    rungs: list[LadderRung] = field(default_factory=lambda: [
        LadderRung(0.10, 0.008),
        LadderRung(0.20, 0.008),
        LadderRung(0.30, 0.008),
    ])
    initial_capital: float = 1_000_000.0
    uvix_slip_bps: float = 5.0         # per user: 5 bps ETP slippage
    fee_bps: float = 1.0
    monetize: MonetizeConfig | None = field(default_factory=lambda: MonetizeConfig(runner_frac=0.0))
    redeploy: RedeployPolicy | None = field(default_factory=RedeployPolicy)
    hedge_kind: str = "ladder"               # ladder | backspread | hybrid
    hedge_budget: HedgeBudgetPolicy | None = None
    backspread: BackspreadHedge | None = None
    hybrid_ladder_frac: float = 0.70         # hybrid: fraction of hedge budget -> ladder
    borrow_uvix_annual: float | None = None  # None -> DEFAULT_BORROW_UVIX
    borrow_svix_annual: float | None = None
    short_proceeds_annual: float = 0.0


def run_insurance(panel: pd.DataFrame, spx: pd.Series, iv: pd.Series, cfg: InsuranceConfig) -> dict:
    ratio = panel["ratio"]
    vix = panel["vix"]
    rebal = adaptive_rebal_dates(ratio, base_days=cfg.base_days, k_stress=cfg.cadence_k)
    rho_s, gross_regime = cfg.regime.series(ratio)
    gross_s = (gross_regime * cfg.sleeve_frac).rename("gross")

    # --- carry sleeve (small gross) ---
    carry = run_carry_backtest(
        panel, rho_s, rebal,
        gross_daily=gross_s,
        initial_capital=cfg.initial_capital,
        fee_bps=cfg.fee_bps,
        slippage_bps=cfg.uvix_slip_bps,
        borrow_uvix_annual=cfg.borrow_uvix_annual if cfg.borrow_uvix_annual is not None else DEFAULT_BORROW_UVIX,
        borrow_svix_annual=cfg.borrow_svix_annual if cfg.borrow_svix_annual is not None else DEFAULT_BORROW_SVIX,
        short_proceeds_annual=cfg.short_proceeds_annual,
    )
    sleeve_ret = carry["ret"]

    # --- T-bill collateral: yield on idle equity (1 - gross actually deployed) ---
    deployed = (carry["gross"] / carry["equity"].replace(0, np.nan)).clip(0, 1).fillna(0.0)
    tbill_daily = (cfg.tbill_rate / 252.0) * (1.0 - deployed)
    base_ret = sleeve_ret + tbill_daily
    base_equity = (1.0 + base_ret).cumprod() * cfg.initial_capital
    base_equity.iloc[0] = cfg.initial_capital

    # --- put ladder funded from carry + bill income ---
    ladder = run_hedge_layer(panel.index, base_equity, spx, iv, vix, ratio, cfg)
    lad = ladder.reindex(base_equity.index).ffill().fillna(0.0)
    cum_cash = lad["put_cash_flow"].cumsum()
    realized = lad["realized"] if "realized" in lad else pd.Series(0.0, index=base_equity.index)
    realized_cum = realized.cumsum()

    # --- redeployment of harvested cash (opportunity-cost allocator) ----------
    # Baseline keeps banked proceeds idle inside cum_cash. ``redeploy_extra`` adds
    # ONLY the incremental growth from routing that cash into the sleeve (earns the
    # realized short-vol return) and/or T-bills, split by the regime at harvest.
    redeploy_extra = pd.Series(0.0, index=base_equity.index)
    if cfg.redeploy is not None and float(realized.abs().sum()) > 0:
        ratio_at = ratio.reindex(base_equity.index).ffill()
        sleeve_w = ratio_at.map(cfg.redeploy.sleeve_weight).clip(0.0, 1.0)
        inj_sleeve = (realized * sleeve_w).fillna(0.0)
        inj_tbill = (realized * (1.0 - sleeve_w)).fillna(0.0)
        sret = sleeve_ret.reindex(base_equity.index).fillna(0.0)
        tret = (cfg.tbill_rate / 252.0)
        sb = tb = 0.0
        extra_vals = []
        cum_p = 0.0
        for dt in base_equity.index:
            sb = sb * (1.0 + float(sret.loc[dt])) + float(inj_sleeve.loc[dt])
            tb = tb * (1.0 + tret) + float(inj_tbill.loc[dt])
            cum_p += float(realized.loc[dt])
            extra_vals.append(sb + tb - cum_p)   # growth above idle principal
        redeploy_extra = pd.Series(extra_vals, index=base_equity.index)

    combined_equity = base_equity + lad["put_mtm"] + cum_cash + redeploy_extra

    out = pd.DataFrame({
        "ratio": ratio.reindex(base_equity.index),
        "rho": rho_s.reindex(base_equity.index),
        "gross_frac": gross_s.reindex(base_equity.index),
        "sleeve_equity": carry["equity"],
        "base_equity": base_equity,
        "put_mtm": lad["put_mtm"],
        "put_cash_cum": cum_cash,
        "realized_cum": realized_cum,
        "redeploy_extra": redeploy_extra,
        "combined_equity": combined_equity,
    })
    out["combined_ret"] = out["combined_equity"].pct_change().fillna(0.0)
    out["drawdown"] = out["combined_equity"].div(out["combined_equity"].cummax()).sub(1.0)
    out.attrs["rebalances"] = len(rebal)
    out.attrs["ladder"] = ladder.attrs
    out.attrs["carry"] = carry
    return {"bt": out, "carry": carry, "ladder": ladder, "rebal": rebal, "cfg": cfg}


# ===========================================================================
# Stats + crash payoff
# ===========================================================================
def _ann_stats(bt: pd.DataFrame) -> dict:
    n = len(bt)
    ret = bt["combined_ret"]
    total = bt["combined_equity"].iloc[-1] / bt["combined_equity"].iloc[0] - 1.0
    ann = 252 / (n - 1)
    cagr = (1 + total) ** ann - 1 if total > -1 else np.nan
    vol = ret.std() * np.sqrt(252)
    dd = bt["drawdown"].min()
    return {
        "CAGR": cagr,
        "Vol": vol,
        "Sharpe": (ret.mean() * 252 / vol) if vol > 0 else np.nan,
        "MaxDD": dd,
        "Calmar": (cagr / abs(dd)) if dd < 0 else np.nan,
    }


def crash_payoff(res: dict, spx: pd.Series, panel: pd.DataFrame) -> dict:
    """Combined crash P&L (fraction of equity) for each scenario."""
    cfg: InsuranceConfig = res["cfg"]
    bt = res["bt"]
    spot0 = float(spx.reindex(panel.index).ffill().iloc[-1])
    atm0 = float((panel["vix"] / 100.0).iloc[-1])
    rho_eff = float(bt["rho"].tail(63).mean())
    gross_eff = float(bt["gross_frac"].tail(63).mean())
    w_u = 1.0 / (1.0 + rho_eff)
    w_s = rho_eff / (1.0 + rho_eff)
    kind = getattr(cfg, "hedge_kind", "ladder") or "ladder"
    bs = cfg.backspread or BackspreadHedge()

    payoff = {}
    for scn in CRASH_SCENARIOS:
        g = scn.vfut_move
        r_u = max(2.0 * g, -1.0)
        r_s = max(-g, -1.0)
        carry_frac = -gross_eff * (w_u * r_u + w_s * r_s)
        put_frac = 0.0
        if kind == "backspread":
            put_frac = hedge_crash_value(
                spot=spot0, atm_iv=atm0, dte=126, spx_drop=scn.spx_drop,
                vix_mult=scn.vix_mult, days=scn.days, kind="backspread",
                otm_far=bs.otm_far, otm_near=bs.otm_near, far_ratio=bs.far_ratio,
                long_premium_frac_equity=bs.premium_frac,
            )
        elif kind == "hybrid":
            split = float(cfg.hybrid_ladder_frac)
            for rg in cfg.rungs:
                put_frac += hedge_crash_value(
                    spot=spot0, atm_iv=atm0, dte=126, spx_drop=scn.spx_drop,
                    vix_mult=scn.vix_mult, days=scn.days, kind="deep_put",
                    otm_far=rg.otm_pct, long_premium_frac_equity=rg.per_roll_frac * split,
                )
            put_frac += hedge_crash_value(
                spot=spot0, atm_iv=atm0, dte=126, spx_drop=scn.spx_drop,
                vix_mult=scn.vix_mult, days=scn.days, kind="backspread",
                otm_far=bs.otm_far, otm_near=bs.otm_near, far_ratio=bs.far_ratio,
                long_premium_frac_equity=bs.premium_frac * (1.0 - split),
            )
        else:
            for rg in cfg.rungs:
                put_frac += hedge_crash_value(
                    spot=spot0, atm_iv=atm0, dte=126, spx_drop=scn.spx_drop,
                    vix_mult=scn.vix_mult, days=scn.days, kind="deep_put",
                    otm_far=rg.otm_pct, long_premium_frac_equity=rg.per_roll_frac,
                )
        payoff[scn.name] = carry_frac + put_frac
    return payoff


def _window_return(eq: pd.Series, start: str, end: str) -> float:
    w = eq.loc[(eq.index >= pd.Timestamp(start)) & (eq.index <= pd.Timestamp(end))]
    return float(w.iloc[-1] / w.iloc[0] - 1.0) if len(w) >= 2 else float("nan")


def summarize(res: dict, spx: pd.Series, panel: pd.DataFrame) -> dict:
    bt = res["bt"]
    carry = res["carry"]
    cfg: InsuranceConfig = res["cfg"]
    n = len(bt)
    years = (n - 1) / 252.0
    sleeve_financing = float(carry["financing_pnl"].sum() / years / carry["equity"].mean())
    sleeve_total_ret = carry["equity"].iloc[-1] / carry["equity"].iloc[0] - 1.0
    sleeve_cagr = (1 + sleeve_total_ret) ** (252 / (n - 1)) - 1
    stats = _ann_stats(bt)
    crash = crash_payoff(res, spx, panel)
    return {
        "sleeve_frac": cfg.sleeve_frac,
        "tbill_rate": cfg.tbill_rate,
        "ladder_per_roll_%": round(sum(r.per_roll_frac for r in cfg.rungs) * 100, 3),
        "rebalances": res["bt"].attrs["rebalances"],
        "sleeve_financing_%/yr": sleeve_financing,
        "sleeve_carry_%/yr": sleeve_cagr,
        "combined_CAGR": stats["CAGR"],
        "combined_Vol": stats["Vol"],
        "combined_Sharpe": stats["Sharpe"],
        "combined_MaxDD": stats["MaxDD"],
        "combined_Calmar": stats["Calmar"],
        "ladder_premium_$": res["ladder"].attrs.get("premium_total", 0.0),
        "aug24": _window_return(bt["combined_equity"], "2024-08-01", "2024-08-31"),
        "crash_mild_-20%": crash["mild_-20%"],
        "crash_severe_-30%": crash["severe_-30%"],
        "crash_volmageddon_-40%": crash["volmageddon_-40%"],
    }


# ===========================================================================
# Runners
# ===========================================================================
def _load_live():
    panel = load_vol_panel(start=INCEPTION, use_synthetic=False)
    spx = load_spx_spot(panel.index.min().strftime("%Y-%m-%d"))
    iv = (panel["vix"] / 100.0).rename("iv")
    return panel, spx, iv


LADDER_2P4 = [
    LadderRung(0.10, 0.008),
    LadderRung(0.20, 0.008),
    LadderRung(0.30, 0.008),
]

LADDER_2X = [
    LadderRung(0.10, 0.008, quantity_multiplier=2),
    LadderRung(0.20, 0.008, quantity_multiplier=2),
    LadderRung(0.30, 0.008, quantity_multiplier=2),
]

EXTENDED_START = "2008-01-01"


def production_config(**overrides) -> InsuranceConfig:
    """Production B insurance product: exactly 2x each prior integer put quantity."""
    defaults = dict(hedge_budget=HedgeBudgetPolicy(), rungs=LADDER_2X)
    defaults.update(overrides)
    return replace(InsuranceConfig(), **defaults)


def reverse_solve_put_contracts(
    *,
    equity_usd: float,
    spx_spot: float,
    atm_iv: float,
    rungs: list[LadderRung],
    hedge_budget: HedgeBudgetPolicy | None = None,
    ratio: float = float("nan"),
    vix: float = float("nan"),
    buy_dte: int = 126,
    risk_free: float = 0.04,
    contract_multiplier: float = 100.0,
) -> dict:
    """Reverse-solve integer SPX contracts from the production premium budget.

    Prices use the backtest's Black-Scholes + skew fallback. Live execution must
    substitute executable asks and recompute the displayed floor formula.
    """
    equity_usd = max(0.0, float(equity_usd))
    spx_spot = max(0.0, float(spx_spot))
    atm_iv = float(np.clip(atm_iv, 0.08, 1.5))
    mult = 1.0
    if hedge_budget is not None and hedge_budget.enabled:
        mult = float(hedge_budget.multiplier(float(ratio), float(vix)))
    t = max(float(buy_dte) / 252.0, 1.0 / 252.0)
    rows: list[dict] = []
    for rung in rungs:
        strike = spx_spot * (1.0 - float(rung.otm_pct))
        pcfg = PutOverlayConfig(otm_pct=float(rung.otm_pct), risk_free=risk_free)
        px = bs_put(
            spx_spot,
            strike,
            t,
            effective_iv(atm_iv, float(rung.otm_pct), pcfg),
            risk_free,
        )
        unit = max(0.0, float(px) * float(contract_multiplier))
        baseline_budget = equity_usd * float(rung.per_roll_frac) * mult
        baseline_contracts = max(1, int(baseline_budget // unit)) if unit > 0 else 0
        contracts = baseline_contracts * max(1, int(rung.quantity_multiplier))
        budget = baseline_budget * max(1, int(rung.quantity_multiplier))
        rows.append({
            "otm_pct": float(rung.otm_pct),
            "strike": float(strike),
            "modeled_put_price": float(px),
            "contract_multiplier": float(contract_multiplier),
            "dynamic_budget_multiplier": mult,
            "baseline_budget_usd": baseline_budget,
            "target_budget_usd": budget,
            "baseline_contracts": baseline_contracts,
            "target_contracts": contracts,
            "premium_used_usd": contracts * unit,
            "unspent_budget_usd": budget - contracts * unit,
        })
    return {
        "method": "premium_budget_reverse_solve_modeled_quotes",
        "equity_usd": equity_usd,
        "spx_spot": spx_spot,
        "atm_iv": atm_iv,
        "buy_dte": int(buy_dte),
        "dynamic_budget_multiplier": mult,
        "baseline_total_contracts": sum(r["baseline_contracts"] for r in rows),
        "target_total_contracts": sum(r["target_contracts"] for r in rows),
        "target_total_budget_usd": sum(r["target_budget_usd"] for r in rows),
        "premium_used_usd": sum(r["premium_used_usd"] for r in rows),
        "rungs": rows,
        "execution_formula": "target contracts = 2 * max(1, floor(baseline rung budget / (executable ask * 100)))",
    }


def save_insurance_plots(res: dict, dest: Path, *, tag: str = "insurance") -> list[Path]:
    """Save equity, drawdown, regime, and crash-payoff charts."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bt = res["bt"]
    cfg: InsuranceConfig = res["cfg"]
    saved: list[Path] = []
    dest.mkdir(parents=True, exist_ok=True)

    # 1) Equity stack: sleeve + T-bills vs combined (+ puts)
    fig, ax = plt.subplots(figsize=(11, 5.5), constrained_layout=True)
    ax.plot(bt.index, bt["base_equity"] / 1e6, label="Sleeve + T-bills (no puts)", lw=1.4, color="#2563eb")
    ax.plot(bt.index, bt["combined_equity"] / 1e6, label="Combined (+ put ladder)", lw=1.6, color="#0f766e")
    ax.set_ylabel("Equity ($M)")
    ax.set_title(
        f"Bucket 5 insurance product — sleeve_frac={cfg.sleeve_frac:.0%}, "
        f"ladder={sum(r.per_roll_frac for r in cfg.rungs)*100:.1f}%/roll"
    )
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    p1 = dest / f"{tag}_equity.png"
    fig.savefig(p1, dpi=130)
    plt.close(fig)
    saved.append(p1)

    # 2) Drawdown
    fig, ax = plt.subplots(figsize=(11, 3.8), constrained_layout=True)
    ax.fill_between(bt.index, bt["drawdown"] * 100, 0, alpha=0.35, color="#b91c1c")
    ax.plot(bt.index, bt["drawdown"] * 100, lw=1.0, color="#991b1b")
    ax.set_ylabel("Drawdown (%)")
    ax.set_title("Combined drawdown")
    ax.grid(alpha=0.3)
    p2 = dest / f"{tag}_drawdown.png"
    fig.savefig(p2, dpi=130)
    plt.close(fig)
    saved.append(p2)

    # 3) Regime: VIX/VIX3M ratio, rho, gross
    fig, axes = plt.subplots(3, 1, figsize=(11, 7.5), sharex=True, constrained_layout=True)
    axes[0].plot(bt.index, bt["ratio"], color="#6366f1", lw=1.0)
    axes[0].axhline(cfg.regime.r_lo, ls=":", color="gray", lw=0.8)
    axes[0].axhline(cfg.regime.r_hi, ls=":", color="gray", lw=0.8)
    axes[0].set_ylabel("VIX/VIX3M")
    axes[0].set_title("Regime signal (contango → backwardation)")
    axes[1].plot(bt.index, bt["rho"], color="#ea580c", lw=1.0)
    axes[1].set_ylabel("rho (SVIX/UVIX)")
    axes[2].plot(bt.index, bt["gross_frac"] * 100, color="#059669", lw=1.0)
    axes[2].set_ylabel("Gross (% eq)")
    axes[2].set_xlabel("Date")
    for ax in axes:
        ax.grid(alpha=0.3)
    p3 = dest / f"{tag}_regime.png"
    fig.savefig(p3, dpi=130)
    plt.close(fig)
    saved.append(p3)

    # 4) Put ladder MTM + harvested (realized) cash
    fig, ax = plt.subplots(figsize=(11, 3.8), constrained_layout=True)
    ax.plot(bt.index, bt["put_mtm"] / 1e3, color="#7c3aed", lw=1.2, label="Put MTM (unrealized)")
    if "realized_cum" in bt and float(bt["realized_cum"].abs().max()) > 0:
        ax.plot(bt.index, bt["realized_cum"] / 1e3, color="#16a34a", lw=1.4,
                label="Harvested cash (cumulative, realized)")
        ax.legend(loc="upper left", fontsize=8)
    ax.set_ylabel("$k")
    ax.set_title("SPX put ladder — MTM vs harvested profit")
    ax.grid(alpha=0.3)
    p4 = dest / f"{tag}_put_mtm.png"
    fig.savefig(p4, dpi=130)
    plt.close(fig)
    saved.append(p4)

    return saved


def main(mode: str = "base") -> None:
    panel, spx, iv = _load_live()
    pd.set_option("display.width", 240)
    pd.set_option("display.max_columns", 40)
    print(f"live sample: {panel.index.min().date()} -> {panel.index.max().date()} ({len(panel)} rows)")
    print("adaptive cadence base=14d (contango-driven), regime rho/gross policy, T-bill collateral, put ladder\n")

    dest = Path("data/runs") / pd.Timestamp.today().strftime("%Y-%m-%d") / "bucket5" / "live"
    dest.mkdir(parents=True, exist_ok=True)

    if mode == "sweep":
        rows = []
        sleeve_grid = [0.15, 0.20, 0.25, 0.30]
        ladder_grid = {
            "ladder_1.2%": [LadderRung(0.10, 0.004), LadderRung(0.20, 0.004), LadderRung(0.30, 0.004)],
            "ladder_2.4%": [LadderRung(0.10, 0.008), LadderRung(0.20, 0.008), LadderRung(0.30, 0.008)],
            "ladder_3.0%_deep": [LadderRung(0.15, 0.010), LadderRung(0.25, 0.010), LadderRung(0.35, 0.010)],
        }
        for sf in sleeve_grid:
            for lname, rungs in ladder_grid.items():
                cfg = InsuranceConfig(sleeve_frac=sf, rungs=rungs)
                res = run_insurance(panel, spx, iv, cfg)
                s = summarize(res, spx, panel)
                s["ladder"] = lname
                rows.append(s)
        df = pd.DataFrame(rows)
        cols = ["sleeve_frac", "ladder", "sleeve_carry_%/yr", "combined_CAGR", "combined_MaxDD",  # noqa
                "combined_Sharpe", "combined_Calmar", "aug24",
                "crash_mild_-20%", "crash_severe_-30%", "crash_volmageddon_-40%"]
        df = df[cols + [c for c in df.columns if c not in cols]]
        out = dest / "insurance_sweep.csv"
        df.to_csv(out, index=False)
        print("=== INSURANCE PRODUCT SWEEP (live era) ===")
        print(df[cols].round(4).to_string(index=False))
        print(f"\nsaved -> {out}")
    elif mode == "plot":
        cfg = InsuranceConfig(sleeve_frac=0.20, rungs=LADDER_2P4)
        res = run_insurance(panel, spx, iv, cfg)
        s = summarize(res, spx, panel)
        tag = "insurance_sf20_ladder24"
        res["bt"].to_csv(dest / f"{tag}_path.csv")
        plots = save_insurance_plots(res, dest, tag=tag)
        print("=== INSURANCE PRODUCT (sleeve_frac=0.20, ladder=2.4%/roll) ===")
        for k, v in s.items():
            if isinstance(v, float):
                print(f"  {k:28s} {v:.4f}")
            else:
                print(f"  {k:28s} {v}")
        print(f"\nsaved path -> {dest / f'{tag}_path.csv'}")
        for p in plots:
            print(f"saved plot -> {p}")
    elif mode == "monetize":
        rows = []
        variants = {
            "buy_and_roll (no harvest)": None,
            "monetize (tiers+VIX+giveback+rearm)": MonetizeConfig(),
        }
        results = {}
        for name, mon in variants.items():
            cfg = InsuranceConfig(sleeve_frac=0.20, rungs=LADDER_2P4, monetize=mon)
            res = run_insurance(panel, spx, iv, cfg)
            results[name] = res
            s = summarize(res, spx, panel)
            s["variant"] = name
            s["realized_$"] = float(res["ladder"].attrs.get("realized_total", 0.0))
            rows.append(s)
        df = pd.DataFrame(rows)
        cols = ["variant", "sleeve_carry_%/yr", "combined_CAGR", "combined_MaxDD",
                "combined_Sharpe", "combined_Calmar", "realized_$", "aug24",
                "crash_mild_-20%", "crash_severe_-30%", "crash_volmageddon_-40%"]
        df = df[cols]
        out = dest / "insurance_monetize_compare.csv"
        df.to_csv(out, index=False)
        print("=== PROFIT-TAKING COMPARISON (sleeve_frac=0.20, ladder=2.4%/roll) ===")
        print(df.round(4).to_string(index=False))
        print(f"\nsaved -> {out}")
        mres = results["monetize (tiers+VIX+giveback+rearm)"]
        mres["bt"].to_csv(dest / "insurance_monetize_path.csv")
        plots = save_insurance_plots(mres, dest, tag="insurance_monetize")
        for p in plots:
            print(f"saved plot -> {p}")
    elif mode == "redeploy":
        # 2x2: {liquidate vs keep-runner} x {idle cash vs regime redeploy}.
        liquidate = MonetizeConfig(runner_frac=0.0)                     # sell all at tiers, re-arm
        runner = MonetizeConfig(runner_frac=0.15, runner_mult=12.0)     # keep a 15% runner
        variants = {
            "A_liquidate_idle": dict(monetize=liquidate, redeploy=None),
            "B_liquidate_redeploy": dict(monetize=liquidate, redeploy=RedeployPolicy()),
            "C_runner_idle": dict(monetize=runner, redeploy=None),
            "D_runner_redeploy": dict(monetize=runner, redeploy=RedeployPolicy()),
        }
        rows = []
        results = {}
        for name, kw in variants.items():
            cfg = InsuranceConfig(sleeve_frac=0.20, rungs=LADDER_2P4, **kw)
            res = run_insurance(panel, spx, iv, cfg)
            results[name] = res
            s = summarize(res, spx, panel)
            s["variant"] = name
            s["realized_$"] = float(res["ladder"].attrs.get("realized_total", 0.0))
            s["redeploy_extra_$"] = float(res["bt"]["redeploy_extra"].iloc[-1])
            rows.append(s)
        df = pd.DataFrame(rows)
        cols = ["variant", "sleeve_carry_%/yr", "combined_CAGR", "combined_MaxDD",
                "combined_Sharpe", "combined_Calmar", "realized_$", "redeploy_extra_$",
                "crash_mild_-20%", "crash_severe_-30%", "crash_volmageddon_-40%"]
        df = df[cols]
        out = dest / "insurance_redeploy_compare.csv"
        df.to_csv(out, index=False)
        print("=== REDEPLOYMENT / PATIENCE COMPARISON (sleeve 20%, ladder 2.4%/roll) ===")
        print(df.round(4).to_string(index=False))
        print(f"\nsaved -> {out}")
        best = results["D_runner_redeploy"]
        best["bt"].to_csv(dest / "insurance_redeploy_path.csv")
        plots = save_insurance_plots(best, dest, tag="insurance_redeploy")
        for p in plots:
            print(f"saved plot -> {p}")
    else:
        cfg = InsuranceConfig()
        res = run_insurance(panel, spx, iv, cfg)
        s = summarize(res, spx, panel)
        print("=== INSURANCE PRODUCT (base config) ===")
        for k, v in s.items():
            print(f"  {k:24s} {v}")
        res["bt"].to_csv(dest / "insurance_base_path.csv")
        print(f"\nsaved -> {dest / 'insurance_base_path.csv'}")


if __name__ == "__main__":
    mode = sys.argv[1].lower() if len(sys.argv) > 1 else "base"
    main(mode)
