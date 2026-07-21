"""Bucket 5 Production B — pure, side-effect-free policy functions.

This module is the single live-side source of the Production B rules described
in ``docs/bucket5_production_b_integration_plan_2026-07-18.md``. Parity with
the research implementation in ``scripts/bucket5_insurance_bt.py``
(``RegimePolicy``, ``HedgeBudgetPolicy``, ``MonetizeConfig``,
``reverse_solve_put_contracts``) is enforced by ``tests/test_bucket5_policy.py``.

Everything here is deterministic and free of I/O so that research replays and
the live GTP/rebalancer path consume the exact same arithmetic.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import yaml

CONFIG_PATH_DEFAULT = Path("config") / "bucket5_production.yml"

ALLOWED_MODES = ("placeholder", "shadow", "production")
ALLOWED_KILL_MODES = (
    "normal", "no_new_risk", "reduce_only", "flatten_carry", "exit_options", "halt_all",
)


# =============================================================================
# Config loading
# =============================================================================

def load_b5_config(path: str | Path | None = None) -> dict:
    """Load and validate ``config/bucket5_production.yml``.

    Fails closed: an unreadable file, unknown mode, or unknown kill mode raises
    rather than defaulting to a live-capable state.
    """
    p = Path(path) if path is not None else CONFIG_PATH_DEFAULT
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    cfg = raw.get("bucket5_production") or {}
    mode = str(cfg.get("mode", "placeholder")).strip().lower()
    if mode not in ALLOWED_MODES:
        raise ValueError(f"bucket5_production.mode={mode!r} not in {ALLOWED_MODES}")
    kill = str(cfg.get("kill_mode", "normal")).strip().lower()
    if kill not in ALLOWED_KILL_MODES:
        raise ValueError(f"bucket5_production.kill_mode={kill!r} not in {ALLOWED_KILL_MODES}")
    if str(cfg.get("account", "live")).strip().lower() != "live":
        raise ValueError("bucket5_production.account must be 'live' (paper path is not implemented)")
    cfg["mode"] = mode
    cfg["kill_mode"] = kill
    return cfg


def config_hash(cfg: dict) -> str:
    """Deterministic short hash of the frozen economic parameters."""
    import json

    frozen = {k: cfg.get(k) for k in ("strategy_version", "capital", "policy", "options")}
    blob = json.dumps(frozen, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:12]


# =============================================================================
# Regime / carry sizing (parity: RegimePolicy in bucket5_insurance_bt)
# =============================================================================

@dataclass(frozen=True)
class RegimeParams:
    r_lo: float = 0.88
    r_hi: float = 1.00
    rho_contango: float = 1.0
    rho_backwardation: float = 2.0
    gross_contango: float = 1.0
    gross_backwardation: float = 0.35


def _clip01(x: float) -> float:
    return min(1.0, max(0.0, x))


def regime_state(ratio: float, p: RegimeParams = RegimeParams()) -> tuple[float, float]:
    """Map a VIX/VIX3M ratio to (rho, gross_multiplier).

    Mirrors ``RegimePolicy.series`` exactly for a scalar input. A non-finite
    ratio is NOT defaulted here — callers must fail closed on missing signals.
    """
    if not math.isfinite(ratio):
        raise ValueError("regime_state requires a finite VIX/VIX3M ratio (fail closed on missing signal)")
    frac = _clip01((ratio - p.r_lo) / (p.r_hi - p.r_lo))
    rho = p.rho_contango + (p.rho_backwardation - p.rho_contango) * frac
    gross = p.gross_contango + (p.gross_backwardation - p.gross_contango) * frac
    return float(rho), float(gross)


def carry_targets(
    *,
    effective_b5_nav: float,
    ratio: float,
    sleeve_frac: float = 0.20,
    regime: RegimeParams = RegimeParams(),
    max_carry_gross_usd: float | None = None,
) -> dict:
    """Solve the paired UVIX/SVIX short targets from Production B policy.

    carry_gross = sleeve_frac * effective_b5_nav * regime_gross_multiplier
    uvix_short  = carry_gross / (1 + rho)
    svix_short  = carry_gross - uvix_short
    """
    rho, gross_mult = regime_state(ratio, regime)
    carry_gross = max(0.0, float(sleeve_frac) * max(0.0, float(effective_b5_nav)) * gross_mult)
    if max_carry_gross_usd is not None:
        carry_gross = min(carry_gross, max(0.0, float(max_carry_gross_usd)))
    uvix_short = carry_gross / (1.0 + rho)
    svix_short = carry_gross - uvix_short
    return {
        "ratio": float(ratio),
        "rho": rho,
        "gross_multiplier": gross_mult,
        "carry_gross_usd": carry_gross,
        "uvix_short_usd": uvix_short,
        "svix_short_usd": svix_short,
    }


# =============================================================================
# Cadence (parity: adaptive_rebal_dates step rule)
# =============================================================================

def cadence_interval_days(
    ratio: float,
    *,
    base_days: float = 14.0,
    k_stress: float = 6.0,
    r_lo: float = 0.88,
    r_hi: float = 1.00,
    min_interval: int = 2,
    max_interval: int = 21,
) -> int:
    """Trading-day interval until the next scheduled carry rebalance."""
    stress = 0.0 if not math.isfinite(ratio) else _clip01((ratio - r_lo) / (r_hi - r_lo))
    denom = 1.0 + k_stress * stress
    return int(min(max_interval, max(min_interval, round(base_days / denom))))


# =============================================================================
# Hedge budget multiplier (parity: HedgeBudgetPolicy.multiplier)
# =============================================================================

@dataclass(frozen=True)
class HedgeBudgetParams:
    enabled: bool = True
    contango_mult: float = 1.20
    stress_mult: float = 0.85
    vix_lo: float = 14.0
    vix_hi: float = 28.0
    vix_calm_boost: float = 1.10
    r_lo: float = 0.88
    r_hi: float = 1.00


def hedge_budget_multiplier(ratio: float, vix: float, p: HedgeBudgetParams = HedgeBudgetParams()) -> float:
    if not p.enabled:
        return 1.0
    if math.isfinite(ratio):
        stress = _clip01((ratio - p.r_lo) / (p.r_hi - p.r_lo))
        m = p.contango_mult + (p.stress_mult - p.contango_mult) * stress
    else:
        m = 1.0
    if math.isfinite(vix) and p.vix_hi > p.vix_lo:
        calm = _clip01((p.vix_hi - vix) / (p.vix_hi - p.vix_lo))
        m *= 1.0 + (p.vix_calm_boost - 1.0) * calm
    return max(0.25, m)


# =============================================================================
# Put ladder sizing from executable asks (plan sections 4.2 / amendment)
# =============================================================================

@dataclass(frozen=True)
class RungSpec:
    otm_pct: float
    per_roll_frac: float
    quantity_multiplier: int = 1

    @property
    def rung_id(self) -> str:
        return f"otm{int(round(self.otm_pct * 100)):02d}"


def rungs_from_config(cfg: dict) -> list[RungSpec]:
    ladder = ((cfg.get("policy") or {}).get("ladder") or {})
    out: list[RungSpec] = []
    for r in ladder.get("rungs") or []:
        out.append(RungSpec(
            otm_pct=float(r["otm_pct"]),
            per_roll_frac=float(r["per_roll_frac"]),
            quantity_multiplier=int(r.get("quantity_multiplier", 1)),
        ))
    return out


def solve_rung_contracts(
    *,
    rung: RungSpec,
    effective_b5_nav: float,
    budget_multiplier: float,
    executable_ask: float,
    contract_multiplier: float = 100.0,
    allow_min_one: bool = False,
) -> dict:
    """Integer contracts for one rung from its premium budget and executable ask.

    Production rule (plan amendment):
        baseline = floor(rung budget * multiplier / (ask * mult))
        target   = quantity_multiplier * baseline
    ``allow_min_one=False`` (live default) never rounds a 0 up to 1: if a single
    contract breaches the rung budget the rung is left in cash and flagged
    under-covered instead of overspending (plan: "never round up beyond the
    premium budget"). Research parity mode (allow_min_one=True) reproduces the
    backtest's ``max(1, floor(...))``.
    """
    unit = max(0.0, float(executable_ask)) * float(contract_multiplier)
    baseline_budget = max(0.0, float(effective_b5_nav)) * rung.per_roll_frac * float(budget_multiplier)
    if unit <= 0.0:
        baseline = 0
    else:
        baseline = int(baseline_budget // unit)
        if allow_min_one:
            baseline = max(1, baseline)
    qmult = max(1, int(rung.quantity_multiplier))
    contracts = baseline * qmult
    budget = baseline_budget * qmult
    premium = contracts * unit
    return {
        "rung_id": rung.rung_id,
        "otm_pct": rung.otm_pct,
        "baseline_budget_usd": baseline_budget,
        "target_budget_usd": budget,
        "baseline_contracts": baseline,
        "target_contracts": contracts,
        "unit_premium_usd": unit,
        "premium_used_usd": premium,
        "unspent_budget_usd": budget - premium,
        "under_covered": bool(contracts == 0 and baseline_budget > 0.0),
    }


# =============================================================================
# Monetization state machine (parity: MonetizeConfig semantics)
# =============================================================================

@dataclass(frozen=True)
class MonetizeParams:
    profit_tiers: tuple[tuple[float, float], ...] = ((3.0, 0.34), (5.0, 0.5), (8.0, 1.0))
    vix_tiers: tuple[tuple[float, float], ...] = ((45.0, 0.5), (65.0, 1.0))
    giveback_frac: float = 0.35
    giveback_min_mult: float = 2.0
    bank_frac: float = 0.6
    rearm: bool = True
    runner_frac: float = 0.15
    runner_mult: float = 12.0


def monetize_params_from_config(cfg: dict) -> MonetizeParams:
    m = ((cfg.get("policy") or {}).get("monetize") or {})
    def _tiers(key: str, default):
        v = m.get(key)
        if v is None:
            return default
        return tuple((float(a), float(b)) for a, b in v)
    return MonetizeParams(
        profit_tiers=_tiers("profit_tiers", MonetizeParams.profit_tiers),
        vix_tiers=_tiers("vix_tiers", MonetizeParams.vix_tiers),
        giveback_frac=float(m.get("giveback_frac", 0.35)),
        giveback_min_mult=float(m.get("giveback_min_mult", 2.0)),
        bank_frac=float(m.get("bank_frac", 0.6)),
        rearm=bool(m.get("rearm", True)),
        runner_frac=float(m.get("runner_frac", 0.15)),
        runner_mult=float(m.get("runner_mult", 12.0)),
    )


@dataclass
class LotState:
    """Minimal monetization-relevant state of one option lot (from the ledger)."""

    entry_contracts: int
    remaining_contracts: int
    cost_basis_usd: float            # total premium paid for the entry
    peak_mult: float = 1.0           # peak (executable value / cost basis) since entry
    profit_tiers_fired: tuple[float, ...] = ()
    vix_tiers_fired: tuple[float, ...] = ()


def monetization_decision(
    lot: LotState,
    *,
    executable_bid_value_usd: float,
    vix: float,
    p: MonetizeParams = MonetizeParams(),
) -> dict:
    """Decide how many contracts to sell now, and why.

    Pure function of (lot state, executable liquidation value, VIX). Tiers fire
    once (idempotent via ``*_tiers_fired``); the runner floor caps every partial
    sale; the giveback rule and runner release produce full exits.

    Returns a dict with ``sell_contracts`` (int), ``reason`` (str or None),
    ``fired_profit_tiers`` / ``fired_vix_tiers`` (tuples to persist), and
    ``updated_peak_mult``.
    """
    out = {
        "sell_contracts": 0,
        "reason": None,
        "fired_profit_tiers": tuple(lot.profit_tiers_fired),
        "fired_vix_tiers": tuple(lot.vix_tiers_fired),
        "updated_peak_mult": lot.peak_mult,
        "full_exit": False,
    }
    if lot.remaining_contracts <= 0 or lot.cost_basis_usd <= 0:
        return out
    # Multiple of cost, using executable liquidation value scaled to the whole
    # entry so partially-sold lots compare like-for-like with the backtest.
    per_contract_value = executable_bid_value_usd / max(1, lot.remaining_contracts)
    mult = (per_contract_value * lot.entry_contracts) / lot.cost_basis_usd
    peak = max(lot.peak_mult, mult)
    out["updated_peak_mult"] = peak

    runner_floor = int(math.floor(p.runner_frac * lot.entry_contracts))
    sellable = max(0, lot.remaining_contracts - runner_floor)

    def _partial(frac_of_remaining: float, reason: str, fired_key: str, fired_val: float) -> None:
        n = int(math.floor(frac_of_remaining * lot.remaining_contracts))
        n = min(n, sellable)
        if n <= 0:
            return
        out["sell_contracts"] = n
        out["reason"] = reason
        out[fired_key] = tuple(sorted(set(out[fired_key]) | {fired_val}))

    def _full(reason: str) -> None:
        out["sell_contracts"] = lot.remaining_contracts
        out["reason"] = reason
        out["full_exit"] = True

    # Runner release: final harvest of everything (ignores runner floor).
    if mult >= p.runner_mult:
        _full(f"runner_release_mult_{p.runner_mult:g}x")
        return out

    # Ordered state transitions: profit tiers first, then VIX tiers, then giveback.
    for tier_mult, frac in p.profit_tiers:
        if mult >= tier_mult and tier_mult not in lot.profit_tiers_fired:
            if frac >= 1.0 and sellable >= lot.remaining_contracts:
                _full(f"profit_tier_{tier_mult:g}x_full")
                out["fired_profit_tiers"] = tuple(sorted(set(out["fired_profit_tiers"]) | {tier_mult}))
            else:
                _partial(frac, f"profit_tier_{tier_mult:g}x", "fired_profit_tiers", tier_mult)
            return out

    for vix_level, frac in p.vix_tiers:
        if math.isfinite(vix) and vix >= vix_level and vix_level not in lot.vix_tiers_fired:
            if frac >= 1.0 and sellable >= lot.remaining_contracts:
                _full(f"vix_tier_{vix_level:g}_full")
                out["fired_vix_tiers"] = tuple(sorted(set(out["fired_vix_tiers"]) | {vix_level}))
            else:
                _partial(frac, f"vix_tier_{vix_level:g}", "fired_vix_tiers", vix_level)
            return out

    if peak >= p.giveback_min_mult and mult <= peak * (1.0 - p.giveback_frac):
        n = sellable
        if n > 0:
            out["sell_contracts"] = n
            out["reason"] = f"giveback_{p.giveback_frac:.0%}_from_peak_{peak:.2f}x"
        return out

    return out


# =============================================================================
# Redeployment (parity: RedeployPolicy.sleeve_weight)
# =============================================================================

def redeploy_sleeve_weight(
    ratio: float,
    *,
    sleeve_w_contango: float = 0.20,
    sleeve_w_backwardation: float = 0.65,
    r_lo: float = 0.88,
    r_hi: float = 1.00,
) -> float:
    if not math.isfinite(ratio):
        return sleeve_w_contango
    frac = _clip01((ratio - r_lo) / (r_hi - r_lo))
    return sleeve_w_contango + (sleeve_w_backwardation - sleeve_w_contango) * frac


# =============================================================================
# Deterministic intent identity (plan section 3.6)
# =============================================================================

def intent_id(
    *,
    strategy_version: str,
    asof: str,
    action_type: str,
    instrument_key: str,
    parent_lot: str = "",
    target_stage: str = "",
) -> str:
    """Deterministic id: strategy_version + asof + action + contract + lot + stage."""
    raw = "|".join([
        str(strategy_version), str(asof), str(action_type).upper(),
        str(instrument_key), str(parent_lot), str(target_stage),
    ])
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:10]
    return f"B5-{action_type.upper()[:6]}-{asof.replace('-', '')}-{digest}"


def order_ref_for_intent(intent: str, prefix: str = "B5P") -> str:
    """IB orderRef for a B5 option intent — always inside the B5P| namespace."""
    return f"{prefix}|{intent}"
