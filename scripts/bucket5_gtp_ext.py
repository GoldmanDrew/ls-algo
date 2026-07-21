"""Bucket 5 Production B — generate_trade_plan extension.

Called from ``generate_trade_plan.main()`` after ``proposed_trades.csv`` is
written. Behavior by ``bucket5_production.mode``:

placeholder : no-op (legacy sleeve sizing owns UVIX/SVIX; no option intents).
shadow      : computes Production B carry + put-ladder targets from live cached
              signals and writes them to ``data/runs/<date>/bucket5_production/``
              (decision manifest, carry targets, option intents) WITHOUT
              touching ``proposed_trades.csv``. Archives placeholder-vs-policy
              deltas for the shadow exit gates.
production  : additionally overrides the UVIX/SVIX plan rows in
              ``proposed_trades.csv`` with policy targets and stamps
              ``b5_owner=production`` so the rebalancer can verify single
              ownership (fail closed if the marker is missing).

All targets derive from ``effective_b5_nav = b5_allocated_nav * ramp_factor``.
Option contract counts computed here use MODELED prices and are planning
estimates only — the rebalancer recomputes from executable asks and never
exceeds the rung budget (plan sections 3.5 / 4.2).
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scripts.bucket5_policy import (
        HedgeBudgetParams, RegimeParams, cadence_interval_days, carry_targets,
        config_hash, hedge_budget_multiplier, intent_id, load_b5_config,
        rungs_from_config,
    )
    from scripts.bucket5_ledger import Bucket5Ledger, intent_event_id
except ImportError:  # pragma: no cover - direct script execution
    from bucket5_policy import (  # type: ignore
        HedgeBudgetParams, RegimeParams, cadence_interval_days, carry_targets,
        config_hash, hedge_budget_multiplier, intent_id, load_b5_config,
        rungs_from_config,
    )
    from bucket5_ledger import Bucket5Ledger, intent_event_id  # type: ignore

B5_SLEEVE = "volatility_etp_bucket5"
CARRY_SYMBOLS = ("UVIX", "SVIX")


# ---------------------------------------------------------------------------
# Signals
# ---------------------------------------------------------------------------

def _load_close_series(csv_path: str | Path) -> pd.Series:
    df = pd.read_csv(csv_path)
    dt = pd.to_datetime(df["date"]).dt.normalize()
    return pd.Series(pd.to_numeric(df["close"], errors="coerce").values, index=dt).dropna()


def load_signals(cfg: dict, run_date: str) -> dict:
    """Load VIX / VIX3M / SPX from the research caches; fail closed on staleness."""
    sig_cfg = cfg.get("signals") or {}
    max_age = int(sig_cfg.get("max_signal_age_days", 5))
    vix = _load_close_series(sig_cfg.get("vix_csv", "data/cache/bucket5/^VIX.csv"))
    vix3m = _load_close_series(sig_cfg.get("vix3m_csv", "data/cache/bucket5/^VIX3M.csv"))
    spx = _load_close_series(sig_cfg.get("spx_csv", "data/cache/bucket5/^GSPC.csv"))
    common = vix.index.intersection(vix3m.index)
    if len(common) == 0:
        raise ValueError("No common VIX/VIX3M dates in signal caches")
    asof = common.max()
    rd = pd.Timestamp(run_date).normalize()
    age_days = int(np.busday_count(asof.date(), rd.date())) if rd > asof else 0
    stale = age_days > max_age
    vix_v = float(vix.loc[asof])
    ratio = vix_v / float(vix3m.loc[asof])
    spx_asof = spx.index[spx.index <= rd].max() if (spx.index <= rd).any() else spx.index.max()
    return {
        "asof": asof.strftime("%Y-%m-%d"),
        "age_days": age_days,
        "stale": stale,
        "vix": vix_v,
        "vix3m": float(vix3m.loc[asof]),
        "ratio": ratio,
        "spx_spot": float(spx.loc[spx_asof]),
        "spx_asof": pd.Timestamp(spx_asof).strftime("%Y-%m-%d"),
    }


# ---------------------------------------------------------------------------
# Modeled option price (planning estimate only — never an executable quote)
# ---------------------------------------------------------------------------

def _modeled_put_price(spot: float, otm_pct: float, atm_iv: float, dte_days: int, risk_free: float = 0.04) -> float:
    try:
        from scripts.bucket5_put_overlay import PutOverlayConfig, bs_put, effective_iv
    except ImportError:  # pragma: no cover
        from bucket5_put_overlay import PutOverlayConfig, bs_put, effective_iv  # type: ignore
    strike = spot * (1.0 - otm_pct)
    t = max(dte_days / 252.0, 1.0 / 252.0)
    pcfg = PutOverlayConfig(otm_pct=otm_pct, risk_free=risk_free)
    return float(bs_put(spot, strike, t, effective_iv(atm_iv, otm_pct, pcfg), risk_free))


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_b5_gtp_extension(
    *,
    run_date: str,
    proposed: pd.DataFrame,
    proposed_paths: list[Path],
    config_path: str | Path | None = None,
) -> pd.DataFrame | None:
    """Emit B5 Production B artifacts; returns the (possibly modified) plan.

    ``proposed_paths`` are the CSVs already written by GTP (dated + latest);
    in production mode they are rewritten with the policy-sized carry rows.
    """
    try:
        cfg = load_b5_config(config_path)
    except FileNotFoundError:
        return None
    mode = cfg["mode"]
    if mode == "placeholder":
        return None

    out_dir = Path("data") / "runs" / run_date / str((cfg.get("paths") or {}).get("run_subdir", "bucket5_production"))
    out_dir.mkdir(parents=True, exist_ok=True)

    strategy_version = str(cfg.get("strategy_version", "B5PROD-1"))
    kill_mode = cfg["kill_mode"]
    cap = cfg.get("capital") or {}
    b5_nav = float(cap.get("b5_allocated_nav", 0.0))
    ramp = float(cap.get("ramp_factor", 0.0))
    effective_nav = b5_nav * ramp
    max_carry = float(cap.get("max_carry_gross_usd", 0.0)) or None

    manifest: dict = {
        "run_date": run_date,
        "mode": mode,
        "kill_mode": kill_mode,
        "strategy_version": strategy_version,
        "config_hash": config_hash(cfg),
        "b5_allocated_nav": b5_nav,
        "ramp_factor": ramp,
        "effective_b5_nav": effective_nav,
        "health": "green",
        "notes": [],
    }

    # ------------------------------------------------------------- signals
    try:
        sig = load_signals(cfg, run_date)
    except Exception as ex:
        manifest["health"] = "no_new_risk"
        manifest["notes"].append(f"signal load failed: {ex}")
        (out_dir / "decision_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"[B5] FAIL-CLOSED: signal load failed ({ex}); wrote no_new_risk manifest.")
        return None
    manifest["signals"] = sig
    if sig["stale"]:
        manifest["health"] = "no_new_risk"
        manifest["notes"].append(
            f"signals stale: asof={sig['asof']} age={sig['age_days']}d > max_signal_age_days"
        )
        (out_dir / "decision_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"[B5] FAIL-CLOSED: stale signals (asof={sig['asof']}); no B5 targets emitted.")
        return None

    pol = cfg.get("policy") or {}
    reg_cfg = pol.get("regime") or {}
    regime = RegimeParams(
        r_lo=float(reg_cfg.get("r_lo", 0.88)), r_hi=float(reg_cfg.get("r_hi", 1.00)),
        rho_contango=float(reg_cfg.get("rho_contango", 1.0)),
        rho_backwardation=float(reg_cfg.get("rho_backwardation", 2.0)),
        gross_contango=float(reg_cfg.get("gross_contango", 1.0)),
        gross_backwardation=float(reg_cfg.get("gross_backwardation", 0.35)),
    )
    hb_cfg = pol.get("hedge_budget") or {}
    hb = HedgeBudgetParams(
        enabled=bool(hb_cfg.get("enabled", True)),
        contango_mult=float(hb_cfg.get("contango_mult", 1.20)),
        stress_mult=float(hb_cfg.get("stress_mult", 0.85)),
        vix_lo=float(hb_cfg.get("vix_lo", 14.0)), vix_hi=float(hb_cfg.get("vix_hi", 28.0)),
        vix_calm_boost=float(hb_cfg.get("vix_calm_boost", 1.10)),
        r_lo=regime.r_lo, r_hi=regime.r_hi,
    )

    # ------------------------------------------------------------- carry
    carry = carry_targets(
        effective_b5_nav=effective_nav,
        ratio=sig["ratio"],
        sleeve_frac=float(pol.get("sleeve_frac", 0.20)),
        regime=regime,
        max_carry_gross_usd=max_carry,
    )
    cad = pol.get("cadence") or {}
    carry["cadence_interval_days"] = cadence_interval_days(
        sig["ratio"],
        base_days=float(cad.get("base_days", 14.0)), k_stress=float(cad.get("k_stress", 6.0)),
        min_interval=int(cad.get("min_interval", 2)), max_interval=int(cad.get("max_interval", 21)),
    )
    manifest["carry"] = carry

    # Placeholder-vs-policy delta (shadow exit-gate evidence).
    placeholder_rows = pd.DataFrame()
    if isinstance(proposed, pd.DataFrame) and not proposed.empty and "sleeve" in proposed.columns:
        placeholder_rows = proposed.loc[
            proposed["sleeve"].astype(str).eq(B5_SLEEVE)
            | proposed.get("ETF", pd.Series(dtype=str)).astype(str).str.upper().isin(CARRY_SYMBOLS)
        ].copy()
    if not placeholder_rows.empty:
        ph_short = pd.to_numeric(placeholder_rows.get("short_usd", 0.0), errors="coerce").fillna(0.0).abs().sum()
        ph_long = pd.to_numeric(placeholder_rows.get("long_usd", 0.0), errors="coerce").fillna(0.0).abs().sum()
        manifest["placeholder_vs_policy"] = {
            "placeholder_gross_usd": float(ph_short + ph_long),
            "policy_gross_usd": float(carry["carry_gross_usd"]),
            "delta_usd": float(carry["carry_gross_usd"] - (ph_short + ph_long)),
        }

    carry_df = pd.DataFrame([
        {
            "run_date": run_date, "symbol": "UVIX", "action": "SHORT",
            "target_short_usd": carry["uvix_short_usd"],
            "rho": carry["rho"], "gross_multiplier": carry["gross_multiplier"],
            "intent_id": intent_id(
                strategy_version=strategy_version, asof=run_date,
                action_type="CARRY", instrument_key="UVIX",
            ),
        },
        {
            "run_date": run_date, "symbol": "SVIX", "action": "SHORT",
            "target_short_usd": carry["svix_short_usd"],
            "rho": carry["rho"], "gross_multiplier": carry["gross_multiplier"],
            "intent_id": intent_id(
                strategy_version=strategy_version, asof=run_date,
                action_type="CARRY", instrument_key="SVIX",
            ),
        },
    ])
    carry_df.to_csv(out_dir / "carry_targets.csv", index=False)

    # ------------------------------------------------------------- options
    opt_cfg = cfg.get("options") or {}
    instrument = str(opt_cfg.get("instrument", "XSP")).upper()
    contract_mult = float(opt_cfg.get("contract_multiplier", 100.0))
    scale = 0.1 if instrument == "XSP" else 1.0   # XSP = SPX / 10
    ladder = pol.get("ladder") or {}
    buy_dte = int(ladder.get("buy_dte", 126))
    roll_dte = int(ladder.get("roll_dte", 63))
    rungs = rungs_from_config(cfg)
    mult = hedge_budget_multiplier(sig["ratio"], sig["vix"], hb)
    manifest["hedge_budget_multiplier"] = mult

    # Target expiry: buy_dte trading days out (approximate calendar mapping;
    # the rebalancer snaps to the nearest listed expiry).
    target_expiry = (pd.Timestamp(run_date) + pd.tseries.offsets.BDay(buy_dte)).strftime("%Y-%m-%d")

    ledger = Bucket5Ledger((cfg.get("paths") or {}).get("ledger_events", "data/bucket5_ledger/events.jsonl"))
    open_lots = ledger.open_lots()

    # Map open lots to rungs via strike distance from the CURRENT ladder.
    def _nearest_rung_id(strike_underlying: float) -> str | None:
        best, best_d = None, float("inf")
        for rg in rungs:
            tgt = sig["spx_spot"] * (1.0 - rg.otm_pct)
            d = abs(strike_underlying - tgt) / max(1.0, tgt)
            if d < best_d:
                best, best_d = rg.rung_id, d
        return best if best_d < 0.08 else None

    lots_by_rung: dict[str, list[dict]] = {}
    for lot in open_lots.values():
        try:
            k_under = float(lot.get("strike") or 0.0) / scale
        except (TypeError, ValueError):
            continue
        rid = _nearest_rung_id(k_under)
        if rid:
            lots_by_rung.setdefault(rid, []).append(lot)

    intent_rows: list[dict] = []
    risk_increasing_blocked = kill_mode in ("no_new_risk", "reduce_only", "flatten_carry", "halt_all")
    for rg in rungs:
        strike_under = sig["spx_spot"] * (1.0 - rg.otm_pct)
        strike_instr = strike_under * scale
        modeled_px_under = _modeled_put_price(sig["spx_spot"], rg.otm_pct, sig["vix"] / 100.0, buy_dte)
        modeled_px = modeled_px_under * scale
        unit = modeled_px * contract_mult
        baseline_budget = effective_nav * rg.per_roll_frac * mult
        qmult = max(1, int(rg.quantity_multiplier))
        baseline_contracts = int(baseline_budget // unit) if unit > 0 else 0
        target_contracts = baseline_contracts * qmult

        existing = lots_by_rung.get(rg.rung_id, [])
        held = sum(int(l.get("remaining_contracts", 0)) for l in existing)
        # Roll check: any lot inside roll_dte of expiry needs a replacement.
        needs_roll = []
        for l in existing:
            try:
                exp = pd.Timestamp(str(l.get("expiry", "")))
                dte = int(np.busday_count(pd.Timestamp(run_date).date(), exp.date()))
                if dte <= roll_dte:
                    needs_roll.append((l, dte))
            except Exception:
                continue

        if held <= 0:
            action, reason = "BUY", "entry"
            qty = target_contracts
        elif needs_roll:
            action, reason = "BUY", "roll_replacement"
            qty = target_contracts
        else:
            continue  # rung covered, nothing to do

        if qty <= 0:
            manifest["notes"].append(
                f"rung {rg.rung_id}: under-covered (modeled unit ${unit:,.0f} > budget ${baseline_budget:,.0f})"
            )
        iid = intent_id(
            strategy_version=strategy_version, asof=run_date, action_type="PUTBUY",
            instrument_key=f"{instrument}|{target_expiry}|{strike_instr:.1f}|P",
            target_stage=rg.rung_id,
        )
        intent_rows.append({
            "intent_id": iid,
            "run_date": run_date,
            "action": action,
            "reason": reason,
            "blocked_by_kill_mode": bool(risk_increasing_blocked),
            "rung_id": rg.rung_id,
            "otm_pct": rg.otm_pct,
            "instrument": instrument,
            "sec_type": "OPT",
            "right": "P",
            "exchange": str(opt_cfg.get("exchange", "SMART")),
            "currency": str(opt_cfg.get("currency", "USD")),
            "multiplier": contract_mult,
            "target_expiry": target_expiry,
            "target_dte_days": buy_dte,
            "target_strike_underlying": round(strike_under, 2),
            "target_strike_instrument": round(strike_instr, 2),
            "modeled_put_price": round(modeled_px, 4),
            "modeled_contracts": target_contracts,
            "contracts_source": "modeled_planning_estimate",
            "baseline_budget_usd": round(baseline_budget, 2),
            "target_budget_usd": round(baseline_budget * qmult, 2),
            "quantity_multiplier": qmult,
            "conId": "",           # resolved by the rebalancer against the live chain
            "order_ref_prefix": "B5P",
            "held_contracts_rung": held,
            "roll_sell_conids": ";".join(str(l.get("conId")) for l, _ in needs_roll),
            "strategy_version": strategy_version,
            "config_hash": manifest["config_hash"],
        })
        ledger.append("INTENT_EMITTED", intent_event_id(iid), {
            "intent_id": iid, "run_date": run_date, "action": action, "reason": reason,
            "rung_id": rg.rung_id, "instrument": instrument,
            "target_expiry": target_expiry, "target_strike_instrument": round(strike_instr, 2),
            "modeled_contracts": target_contracts, "target_budget_usd": round(baseline_budget * qmult, 2),
            "strategy_version": strategy_version, "config_hash": manifest["config_hash"],
        })

    intents_df = pd.DataFrame(intent_rows)
    intents_path = out_dir / "option_intents.csv"
    intents_df.to_csv(intents_path, index=False)
    # Sibling artifact next to proposed_trades.csv (plan: first-class plan rows).
    sibling = Path("data") / "runs" / run_date / "proposed_trades_b5_options.csv"
    intents_df.to_csv(sibling, index=False)
    manifest["option_intents"] = len(intent_rows)

    # ------------------------------------------------- production carry override
    # B5 pair-row schema (see data/runs/<date>/proposed_trades.csv):
    #   ETF=UVIX, Underlying=SVIX; short_usd / etf_target_usd = UVIX leg;
    #   long_usd / underlying_target_usd = SVIX leg; negative values = short.
    modified = None
    if mode == "production" and isinstance(proposed, pd.DataFrame) and not proposed.empty:
        modified = proposed.copy()
        if "b5_owner" not in modified.columns:
            modified["b5_owner"] = ""
        etf_u = modified.get("ETF", pd.Series(dtype=str)).astype(str).str.upper()
        b5_mask = modified["sleeve"].astype(str).eq(B5_SLEEVE) if "sleeve" in modified.columns else etf_u.isin(CARRY_SYMBOLS)
        b5_mask = b5_mask | etf_u.isin(CARRY_SYMBOLS)
        n_owner_rows = int(b5_mask.sum())
        if n_owner_rows != 1:
            # Fail closed: exactly one B5 pair row expected; anything else is a
            # dual-ownership or schema problem (plan section 3.1).
            manifest["health"] = "no_new_risk"
            manifest["notes"].append(
                f"production mode expected exactly 1 UVIX/SVIX pair row, found {n_owner_rows}; fail closed"
            )
            modified = None
        else:
            idx = modified.index[b5_mask][0]
            etf = str(modified.at[idx, "ETF"]).upper()
            und = str(modified.at[idx, "Underlying"]).upper() if "Underlying" in modified.columns else ""
            uvix_leg = -abs(carry["uvix_short_usd"])
            svix_leg = -abs(carry["svix_short_usd"])
            etf_leg = uvix_leg if etf == "UVIX" else svix_leg
            und_leg = svix_leg if und == "SVIX" else uvix_leg
            field_map = {
                "gross_target_usd": carry["carry_gross_usd"],
                "optimal_gross_target_usd": carry["carry_gross_usd"],
                "short_usd": etf_leg,
                "etf_target_usd": etf_leg,
                "optimal_short_usd": etf_leg,
                "optimal_etf_target_usd": etf_leg,
                "long_usd": und_leg,
                "underlying_target_usd": und_leg,
                "optimal_long_usd": und_leg,
                "optimal_underlying_target_usd": und_leg,
            }
            for col, val in field_map.items():
                if col in modified.columns:
                    modified.at[idx, col] = val
            modified.at[idx, "b5_owner"] = "production"
            for p in proposed_paths:
                modified.to_csv(p, index=False)
            print(
                "[B5] production mode: pair row re-sized by Production B policy "
                f"(gross ${carry['carry_gross_usd']:,.0f}; UVIX ${abs(uvix_leg):,.0f} / SVIX ${abs(svix_leg):,.0f}) "
                "and stamped b5_owner=production."
            )

    (out_dir / "decision_manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    print(
        f"[B5] mode={mode} health={manifest['health']} ratio={sig['ratio']:.3f} "
        f"carry_gross=${carry['carry_gross_usd']:,.0f} (UVIX ${carry['uvix_short_usd']:,.0f} / "
        f"SVIX ${carry['svix_short_usd']:,.0f}) option_intents={len(intent_rows)} -> {out_dir}"
    )
    return modified
