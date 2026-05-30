#!/usr/bin/env python3
"""
phase2b_resize.py

Phase 2b — Resize Existing Pairs
================================
Bidirectional leg-by-leg resize toward plan USD targets, gated by a
hysteresis band to prevent over-trading. Inserted between Phase 2
(Establish) and Phase 3 (Hedge) in ``rebalance_strategy.main()``.

Decision rule per leg
---------------------
    abs_target  = |target_usd|
    abs_current = |current_qty * price|
    abs_drift   = abs_current - abs_target            # signed
    floor       = max(min_trim_usd, min_grow_usd)
    enter_thr   = max(enter_band_pct * abs_target, floor)
    exit_thr    = max(exit_band_pct  * abs_target, floor)

    if |abs_drift| <= enter_thr:    skip   (deadband / hysteresis)
    else:                           trade  |abs_drift| - exit_thr

Direction
---------
    long_under leg:  drift > 0 -> SELL (trim);  drift < 0 -> BUY  (grow)
    short_etf  leg:  drift > 0 -> BUY  (cover); drift < 0 -> SELL (grow short)

Stage 1 (this module): no tax routing, no substitution. The
``tax_router`` argument on :func:`build_resize_trades` is a forward hook
for Stage 2 and is a no-op when ``None``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict, field, fields
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple

import pandas as pd

from execute_trade_plan import (
    tprint, stop_requested,
    execute_leg, ExecResult,
    norm_sym, append_csv_row,
    CoordinatorCancelService,
    is_short_unavailable_now,
    classify_plan_leg_bucket, build_long_spot_bucket_token,
)


# --------------------------------------------------------------------------
# Config + decision dataclasses
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class ResizeBandConfig:
    """Hysteresis band parameters for Phase 2b."""
    enabled: bool = True
    enter_band_pct: float = 0.15
    exit_band_pct: float = 0.05
    min_trim_usd: float = 250.0
    min_grow_usd: float = 250.0

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "ResizeBandConfig":
        d = dict(d or {})
        return cls(
            enabled=bool(d.get("enabled", True)),
            enter_band_pct=float(d.get("enter_band_pct", 0.15)),
            exit_band_pct=float(d.get("exit_band_pct", 0.05)),
            min_trim_usd=float(d.get("min_trim_usd", 250.0)),
            min_grow_usd=float(d.get("min_grow_usd", 250.0)),
        )


@dataclass
class ResizeDecision:
    """One row of resize_decisions.csv telemetry."""
    underlying: str = ""
    etf: str = ""
    leg_side: str = ""              # "long_under" | "short_etf"
    target_usd: float = 0.0
    current_usd: float = 0.0
    delta_usd: float = 0.0          # current - target  (signed)
    enter_threshold_usd: float = 0.0
    exit_threshold_usd: float = 0.0
    decision: str = "skip"          # "trim"|"grow"|"skip"|"block"|"harvest_sub_sell"|"harvest_sub_buy"|"lt_only_trim"
    reason: str = ""
    action: Optional[str] = None    # "BUY" | "SELL" | None
    qty: int = 0
    ref_price: float = 0.0
    trade_usd: float = 0.0
    # ---- Stage 2 (tax-aware) fields; default 0/empty when tax routing is off ----
    est_realized_pnl_usd: float = 0.0
    st_qty_consumed: float = 0.0
    lt_qty_consumed: float = 0.0
    lots_consumed_count: int = 0
    substitute_of: str = ""         # if this row is the BUY-substitute leg, the original symbol
    swap_with: str = ""             # if this row participates in a swap, the partner symbol
    harvested_loss_usd: float = 0.0


# --------------------------------------------------------------------------
# Pure decision logic (band check + direction)
# --------------------------------------------------------------------------

def _band_decide(
    *,
    side: str,
    target_usd: float,
    current_usd: float,
    cfg: ResizeBandConfig,
) -> ResizeDecision:
    """
    Evaluate hysteresis band for one leg.

    Returns a partially-filled ResizeDecision. If decision is ``trim`` or
    ``grow``, ``action`` and ``trade_usd`` are populated; ``qty`` and
    ``ref_price`` are filled in by the caller (which knows the price).
    """
    abs_target  = abs(float(target_usd))
    abs_current = abs(float(current_usd))
    delta_usd   = float(current_usd) - float(target_usd)
    abs_drift   = abs_current - abs_target

    floor = max(cfg.min_trim_usd, cfg.min_grow_usd)
    enter_thr = max(cfg.enter_band_pct * abs_target, floor)
    exit_thr  = max(cfg.exit_band_pct  * abs_target, floor)

    out = ResizeDecision(
        leg_side=side,
        target_usd=float(target_usd),
        current_usd=float(current_usd),
        delta_usd=delta_usd,
        enter_threshold_usd=float(enter_thr),
        exit_threshold_usd=float(exit_thr),
    )

    if target_usd > 0 and current_usd < 0:
        out.reason = "sign_mismatch_long_target_short_current"
        return out
    if target_usd < 0 and current_usd > 0:
        out.reason = "sign_mismatch_short_target_long_current"
        return out

    if abs(abs_drift) <= enter_thr:
        out.reason = "within_enter_band"
        return out

    trade_usd_abs = max(0.0, abs(abs_drift) - exit_thr)
    over = abs_drift > 0   # over-exposed in absolute terms

    if side not in ("long_under", "short_etf"):
        out.reason = f"unknown_leg_side:{side}"
        return out

    # Action depends purely on which side of the target we're on:
    #   delta_usd = current - target
    #   delta > 0 -> SELL (current is "above" target on the number line)
    #   delta < 0 -> BUY  (current is "below" target on the number line)
    # This is sign-agnostic and works for:
    #   * long_under with target > 0 (B1/YB long underlying)
    #   * long_under with target < 0 (B4 short underlying)
    #   * long_under with mixed B1+B4 net target either sign
    #   * short_etf  with target < 0 (B1 LETF, YB LETF, B4 inverse ETF)
    action   = "SELL" if delta_usd > 0 else "BUY"
    # Trim = move toward zero |notional|; grow = move away from zero.
    decision = "trim" if over else "grow"

    needed_floor = cfg.min_trim_usd if decision == "trim" else cfg.min_grow_usd
    if trade_usd_abs < needed_floor:
        out.reason = f"trade_usd<{needed_floor:.0f}_floor"
        return out

    out.decision = decision
    out.reason   = f"abs_drift={abs_drift:+.0f}|enter={enter_thr:.0f}|exit={exit_thr:.0f}"
    out.action   = action
    out.trade_usd = float(trade_usd_abs)
    return out


# --------------------------------------------------------------------------
# Build phase: turn plan rows into resize candidates with bands applied
# --------------------------------------------------------------------------

def _resize_target_columns(
    plan: pd.DataFrame, target_basis: str
) -> Tuple[str, str, str, str]:
    """Resolve the (band_long_col, band_short_col, exec_long_col, exec_short_col) used by
    :func:`build_resize_trades` based on ``target_basis``.

    - ``executable`` (legacy): bands and orders both use ``long_usd`` / ``short_usd``.
    - ``optimal``: bands and orders both use ``optimal_long_usd`` / ``optimal_short_usd``
      (no clipping by today's executable — pair with FTP cap in the executor).
    - ``hybrid`` (default): bands use ``optimal_*`` (so we don't trim winners that today's
      liquidity squeezed), orders are **clipped** by ``long_usd`` / ``short_usd`` so we never
      ask for more shares than today's pipeline said are available.
    """
    basis = str(target_basis or "hybrid").strip().lower()
    has_opt = "optimal_long_usd" in plan.columns and "optimal_short_usd" in plan.columns
    if basis == "executable" or not has_opt:
        return ("long_usd", "short_usd", "long_usd", "short_usd")
    if basis == "optimal":
        return ("optimal_long_usd", "optimal_short_usd",
                "optimal_long_usd", "optimal_short_usd")
    return ("optimal_long_usd", "optimal_short_usd", "long_usd", "short_usd")


def build_resize_trades(
    *,
    hedgeable_plan: pd.DataFrame,
    strat_pos: Dict[str, float],
    prices: Dict[str, float],
    purgatory_etfs: Set[str],
    flow_etfs: Set[str],
    blacklist: Optional[Set[str]] = None,
    cfg: ResizeBandConfig,
    skip_underlyings: Optional[Set[str]] = None,
    tax_router: Optional[Callable[[List[Dict], List[ResizeDecision]],
                                  Tuple[List[Dict], List[ResizeDecision]]]] = None,
    target_basis: str = "hybrid",
) -> Tuple[List[Dict], List[ResizeDecision]]:
    """
    Iterate the hedgeable plan and produce resize trade candidates whose
    legs fall outside the enter band.

    Returns
    -------
    trades : list[dict]
        Trades to submit. Each dict::

            {
                "underlying":       str,
                "etf":              str,                # "" for under leg
                "leg_side":         "long_under" | "short_etf",
                "symbol":           str,                # what to trade
                "action":           "BUY" | "SELL",
                "qty":              int,
                "ref_price":        float,
                "trade_usd_target": float,
                "decision":         "trim" | "grow",
                "reason":           str,
                "target_usd":       float,
                "current_usd":      float,
            }

    decisions : list[ResizeDecision]
        One row per leg evaluated, including skips. Used for telemetry.
    """
    blacklist        = set(blacklist or set())
    skip_underlyings = set(skip_underlyings or set())

    trades: List[Dict] = []
    decisions: List[ResizeDecision] = []

    if hedgeable_plan is None or hedgeable_plan.empty:
        return trades, decisions

    band_long_col, band_short_col, exec_long_col, exec_short_col = _resize_target_columns(
        hedgeable_plan, target_basis
    )

    grouped = hedgeable_plan.groupby("Underlying", sort=True)

    for under, grp in grouped:
        under = norm_sym(str(under))
        if under in skip_underlyings or under in blacklist:
            continue

        px_under = float(prices.get(under) or 0.0)

        if px_under <= 0.0:
            for _, row in grp.iterrows():
                etf = norm_sym(str(row["ETF"]))
                decisions.append(ResizeDecision(
                    underlying=under, etf=etf, leg_side="long_under",
                    target_usd=float(row.get(band_long_col, row.get("long_usd", 0.0)) or 0.0),
                    decision="skip", reason="no_price_for_underlying",
                ))
            continue

        cur_under_sh  = float(strat_pos.get(under, 0.0))
        cur_under_usd = cur_under_sh * px_under

        total_band_long = float(pd.to_numeric(
            grp.get(band_long_col, grp.get("long_usd", grp.get("underlying_target_usd", 0.0))),
            errors="coerce",
        ).fillna(0.0).sum())
        total_exec_long = float(pd.to_numeric(
            grp.get(exec_long_col, grp.get("long_usd", grp.get("underlying_target_usd", 0.0))),
            errors="coerce",
        ).fillna(0.0).sum())

        u_dec = _band_decide(
            side="long_under",
            target_usd=total_band_long,
            current_usd=cur_under_usd,
            cfg=cfg,
        )
        u_dec.underlying = under
        u_dec.etf = ",".join(sorted({norm_sym(str(e)) for e in grp["ETF"]}))

        if u_dec.decision in ("trim", "grow") and u_dec.action and u_dec.trade_usd > 0:
            # Hybrid mode: clip a "grow" step so we don't push past today's executable target.
            # Trim steps remain unchanged — selling out of an over-sized position never violates
            # liquidity (we already hold the shares) and we want to honor the structural target.
            if (
                str(target_basis or "hybrid").strip().lower() == "hybrid"
                and u_dec.decision == "grow"
                and band_long_col != exec_long_col
            ):
                # max growth this run = |total_exec_long - cur_under_usd|, sign-aware.
                if total_band_long >= 0:
                    max_step_usd = max(0.0, total_exec_long - cur_under_usd)
                else:
                    max_step_usd = max(0.0, cur_under_usd - total_exec_long)
                if u_dec.trade_usd > max_step_usd + 1e-9:
                    u_dec.trade_usd = float(max_step_usd)
                    u_dec.reason = (u_dec.reason or "") + "|clipped_to_executable"
                    if u_dec.trade_usd <= 0:
                        u_dec.decision = "skip"
                        u_dec.reason = u_dec.reason + "|qty=0_after_exec_clip"

        if u_dec.decision in ("trim", "grow") and u_dec.action and u_dec.trade_usd > 0:
            qty = int(math.floor(u_dec.trade_usd / px_under))
            # Cap qty only when we're consuming existing inventory:
            #   * SELL: cap by current long shares (can't sell more long than we hold).
            #     If currently flat or short, this SELL is opening/growing a short
            #     (B4 path) — broker handles locate; no inventory cap here.
            #   * BUY: cap by current short shares (can't buy back more than we are short).
            #     If currently flat or long, this BUY is opening/growing a long;
            #     no inventory cap.
            # The sign-mismatch guard in _band_decide already blocks position flips,
            # so we never need to sell through zero here.
            if u_dec.action == "SELL" and cur_under_sh > 0:
                qty = min(qty, int(round(cur_under_sh)))
            elif u_dec.action == "BUY" and cur_under_sh < 0:
                qty = min(qty, int(round(-cur_under_sh)))
            if qty <= 0:
                u_dec.decision = "skip"
                u_dec.reason   = u_dec.reason + "|qty=0_after_cap"
            else:
                # B1/B2 long-spot split across the group's sleeves, so the
                # netted underlying order can carry the true division tag.
                _long_b1 = _long_b2 = 0.0
                for _, _lr in grp.iterrows():
                    _lu = float(_lr.get(band_long_col, _lr.get("long_usd", 0.0)) or 0.0)
                    _bkt = classify_plan_leg_bucket(
                        sleeve=_lr.get("sleeve"), delta=_lr.get("Delta"), long_usd=_lu
                    )
                    if _bkt == "b1":
                        _long_b1 += _lu
                    elif _bkt == "b2":
                        _long_b2 += _lu
                u_dec.qty = int(qty)
                u_dec.ref_price = float(px_under)
                trades.append({
                    "underlying":       under,
                    "etf":              "",
                    "leg_side":         "long_under",
                    "symbol":           under,
                    "action":           u_dec.action,
                    "qty":              int(qty),
                    "ref_price":        float(px_under),
                    "trade_usd_target": float(u_dec.trade_usd),
                    "decision":         u_dec.decision,
                    "reason":           u_dec.reason,
                    "target_usd":       float(total_band_long),
                    "executable_target_usd": float(total_exec_long),
                    "current_usd":      float(cur_under_usd),
                    "long_usd_b1":      float(_long_b1),
                    "long_usd_b2":      float(_long_b2),
                })

        decisions.append(u_dec)

        for _, row in grp.iterrows():
            etf = norm_sym(str(row["ETF"]))
            band_target_usd = float(row.get(band_short_col, row.get("short_usd", row.get("etf_target_usd", 0.0))) or 0.0)
            exec_target_usd = float(row.get(exec_short_col, row.get("short_usd", row.get("etf_target_usd", 0.0))) or 0.0)
            target_usd = band_target_usd
            cur_etf_sh  = float(strat_pos.get(etf, 0.0))

            if etf in purgatory_etfs or etf in flow_etfs:
                decisions.append(ResizeDecision(
                    underlying=under, etf=etf, leg_side="short_etf",
                    target_usd=target_usd,
                    decision="skip",
                    reason="purgatory" if etf in purgatory_etfs else "flow_etf",
                ))
                continue

            px_etf = float(prices.get(etf) or 0.0)
            if px_etf <= 0.0:
                decisions.append(ResizeDecision(
                    underlying=under, etf=etf, leg_side="short_etf",
                    target_usd=target_usd,
                    decision="skip", reason="no_price_for_etf",
                ))
                continue

            cur_etf_usd = cur_etf_sh * px_etf

            e_dec = _band_decide(
                side="short_etf",
                target_usd=target_usd,
                current_usd=cur_etf_usd,
                cfg=cfg,
            )
            e_dec.underlying = under
            e_dec.etf = etf

            if e_dec.decision in ("trim", "grow") and e_dec.action and e_dec.trade_usd > 0:
                if (
                    str(target_basis or "hybrid").strip().lower() == "hybrid"
                    and e_dec.decision == "grow"
                    and band_short_col != exec_short_col
                ):
                    # Short targets are negative; growing means making position more negative.
                    if band_target_usd <= 0:
                        max_step_usd = max(0.0, cur_etf_usd - exec_target_usd)
                    else:
                        max_step_usd = max(0.0, exec_target_usd - cur_etf_usd)
                    if e_dec.trade_usd > max_step_usd + 1e-9:
                        e_dec.trade_usd = float(max_step_usd)
                        e_dec.reason = (e_dec.reason or "") + "|clipped_to_executable"
                        if e_dec.trade_usd <= 0:
                            e_dec.decision = "skip"
                            e_dec.reason = e_dec.reason + "|qty=0_after_exec_clip"

            if e_dec.decision in ("trim", "grow") and e_dec.action and e_dec.trade_usd > 0:
                qty = int(math.floor(e_dec.trade_usd / px_etf))
                if e_dec.action == "BUY":
                    qty = min(qty, max(0, int(abs(round(cur_etf_sh)))))
                if qty <= 0:
                    e_dec.decision = "skip"
                    e_dec.reason   = e_dec.reason + "|qty=0_after_cap"
                else:
                    e_dec.qty = int(qty)
                    e_dec.ref_price = float(px_etf)
                    trades.append({
                        "underlying":       under,
                        "etf":              etf,
                        "leg_side":         "short_etf",
                        "symbol":           etf,
                        "action":           e_dec.action,
                        "qty":              int(qty),
                        "ref_price":        float(px_etf),
                        "trade_usd_target": float(e_dec.trade_usd),
                        "decision":         e_dec.decision,
                        "reason":           e_dec.reason,
                        "target_usd":       float(target_usd),
                        "executable_target_usd": float(exec_target_usd),
                        "current_usd":      float(cur_etf_usd),
                    })

            decisions.append(e_dec)

    if tax_router is not None:
        trades, decisions = tax_router(trades, decisions)

    return trades, decisions


# --------------------------------------------------------------------------
# Telemetry: write resize_decisions.csv (one row per leg evaluated)
# --------------------------------------------------------------------------

_DECISION_COLS: List[str] = [
    "ts", "run_date", "strategy_tag",
    *[f.name for f in fields(ResizeDecision)],
]


def write_resize_decisions(
    decisions: List[ResizeDecision],
    out_csv: Path,
    *,
    run_date: str,
    strategy_tag: str,
) -> int:
    """Append decisions to ``out_csv``. Returns rows written."""
    if not decisions:
        return 0
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    n = 0
    for d in decisions:
        row = {
            "ts": now,
            "run_date": run_date,
            "strategy_tag": strategy_tag,
            **asdict(d),
        }
        ordered = {k: row.get(k) for k in _DECISION_COLS}
        append_csv_row(out_csv, ordered)
        n += 1
    return n


# --------------------------------------------------------------------------
# Print helper for operator review
# --------------------------------------------------------------------------

def print_resize_summary(
    trades: List[Dict],
    decisions: List[ResizeDecision],
) -> None:
    """Print a compact summary of what Phase 2b will do."""
    n_trim_under = sum(1 for t in trades if t["leg_side"] == "long_under"  and t["decision"] == "trim")
    n_grow_under = sum(1 for t in trades if t["leg_side"] == "long_under"  and t["decision"] == "grow")
    n_trim_etf   = sum(1 for t in trades if t["leg_side"] == "short_etf"   and t["decision"] == "trim")
    n_grow_etf   = sum(1 for t in trades if t["leg_side"] == "short_etf"   and t["decision"] == "grow")
    n_skip       = sum(1 for d in decisions if d.decision == "skip")

    total_usd = sum(abs(float(t.get("trade_usd_target", 0.0))) for t in trades)

    tprint("[PHASE2B] Summary:")
    tprint(f"   long_under  trim={n_trim_under}  grow={n_grow_under}")
    tprint(f"   short_etf   trim={n_trim_etf}    grow={n_grow_etf}")
    tprint(f"   skipped (within band / no price / purgatory / sign-mismatch): {n_skip}")
    tprint(f"   total resize notional: ~${total_usd:,.0f}")

    if not trades:
        return

    tprint("[PHASE2B] Planned trades:")
    for t in sorted(trades, key=lambda x: (x["underlying"], x["leg_side"], x["symbol"])):
        tprint(
            f"   {t['underlying']:<10} {t['leg_side']:<11} "
            f"{t['action']:<4} {t['qty']:>6d} {t['symbol']:<8} "
            f"@ {t['ref_price']:>8.4f}  ~${t['trade_usd_target']:,.0f}  "
            f"({t['decision']}: {t['reason']})"
        )


# --------------------------------------------------------------------------
# Execute phase: serial execution under coordinator IB
# --------------------------------------------------------------------------

def execute_resize_serial(
    *,
    ib,
    trades: List[Dict],
    short_map: Dict[str, dict],
    blocked_short_etfs: Set[str],
    exec_cfg: dict,
    strategy_tag: str,
    run_date: str,
    limit_bps: float,
    timeout: float,
    max_retries: int,
    dry_run: bool,
    cancel_service: CoordinatorCancelService,
    log_exposure_event: Callable,
    short_first: bool = True,
    screener_avail_map: Optional[Dict[str, int]] = None,
) -> List[Dict]:
    """
    Execute resize trades serially through the coordinator IB connection.

    Sort order (when ``short_first=True``):
      1. SELLs first (these include both 'grow short' and 'trim long')
      2. then BUYs ('cover short' and 'grow long')
    so any short-side capacity issues are surfaced before we add length.

    Returns
    -------
    fills : list[dict]
        Fill records using the existing rebalance/execution schema.
    """
    fills: List[Dict] = []
    if not trades:
        return fills

    def _sort_key(t: Dict):
        primary = 0 if (short_first and t["action"] == "SELL") else 1
        return (primary, t["underlying"], t.get("etf") or "", t["symbol"])

    trades_sorted = sorted(trades, key=_sort_key)

    for t in trades_sorted:
        if stop_requested():
            tprint("[PHASE2B] Shutdown requested; stopping resize loop.")
            break

        symbol   = t["symbol"]
        action   = t["action"]
        qty      = int(t["qty"])
        px       = float(t["ref_price"])
        under    = t["underlying"]
        etf      = t.get("etf") or ""
        leg_side = t["leg_side"]
        decision = t["decision"]

        capped_qty = qty

        if leg_side == "short_etf" and action == "SELL":
            if symbol in blocked_short_etfs:
                tprint(f"[PHASE2B][{under}/{symbol}] SKIP: previously blocked this run.")
                fills.append({
                    "filled_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "run_date": run_date, "strategy_tag": strategy_tag,
                    "pair_id": f"{under}__RESIZE",
                    "underlying": under, "etf": etf,
                    "px_under": None, "px_etf": float(px),
                    "target_sh_under": None, "target_sh_etf": None,
                    "delta_sh_under": 0, "delta_sh_etf": -qty,
                    "filled_sh_under": 0, "filled_sh_etf": 0,
                    "notes": "P2B_SKIP_ALREADY_BLOCKED",
                })
                continue

            blocked, why = is_short_unavailable_now(
                symbol, short_map=short_map, screener_avail_map=screener_avail_map,
            )
            if blocked:
                src = "FTP available=0" if why == "ftp_avail0" else "screener shares_available<=0"
                tprint(f"[PHASE2B][{under}/{symbol}] SKIP: {src} (wanted {qty}).")
                fills.append({
                    "filled_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "run_date": run_date, "strategy_tag": strategy_tag,
                    "pair_id": f"{under}__RESIZE",
                    "underlying": under, "etf": etf,
                    "px_under": None, "px_etf": float(px),
                    "target_sh_under": None, "target_sh_etf": None,
                    "delta_sh_under": 0, "delta_sh_etf": -qty,
                    "filled_sh_under": 0, "filled_sh_etf": 0,
                    "notes": f"P2B_SKIP_NO_LOCATE_{why.upper()} wants_short={qty}",
                })
                continue

            sm = short_map.get(symbol, {}) or {}
            avail = sm.get("available")
            if avail is not None:
                avail = int(avail)
                if avail < qty:
                    capped_qty = avail
                    tprint(
                        f"[PHASE2B][{under}/{symbol}] CAP: wants {qty} new short, "
                        f"FTP available={avail}; capping to {avail}."
                    )

        order_ref = (
            f"{strategy_tag}|{under}__RESIZE|{symbol}|"
            f"{leg_side.upper()}_{decision.upper()}"
        )
        # Tag the B1/B2 long-spot split on the netted underlying leg so
        # accounting records the true division (Phase 3 forward attribution).
        if leg_side == "long_under":
            _lsb_tok = build_long_spot_bucket_token(
                t.get("long_usd_b1", 0.0), t.get("long_usd_b2", 0.0)
            )
            if _lsb_tok:
                order_ref = f"{order_ref}|{_lsb_tok}"
        tprint(
            f"[PHASE2B][{under}] {decision.upper()} {leg_side} -> "
            f"{action} {capped_qty} {symbol} @ {px:.4f}"
        )

        res: ExecResult = execute_leg(
            ib=ib, symbol=symbol, action=action, qty=capped_qty,
            ref_price=px, bps=limit_bps, order_ref=order_ref,
            exec_cfg=exec_cfg, timeout=timeout, max_retries=max_retries,
            dry_run=dry_run, context=f"{under}|RESIZE",
            cancel_service=cancel_service,
        )

        filled_signed = int(res.filled) if action == "BUY" else -int(res.filled)
        signed_qty    = capped_qty       if action == "BUY" else -capped_qty

        if res.status == "SHORT_BLOCKED" and leg_side == "short_etf":
            blocked_short_etfs.add(symbol)

        log_exposure_event(
            stage="POST_RESIZE",
            pair_id=f"{under}__RESIZE",
            underlying=under, etf=etf, symbol=symbol,
            delta_sh=int(signed_qty),
            filled_sh=int(filled_signed),
            trade=res.trade,
        )

        fills.append({
            "filled_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "run_date": run_date, "strategy_tag": strategy_tag,
            "pair_id": f"{under}__RESIZE",
            "underlying": under, "etf": etf,
            "px_under": float(px) if leg_side == "long_under" else None,
            "px_etf":   float(px) if leg_side == "short_etf" else None,
            "target_sh_under": None, "target_sh_etf": None,
            "delta_sh_under": signed_qty if leg_side == "long_under" else 0,
            "delta_sh_etf":   signed_qty if leg_side == "short_etf" else 0,
            "filled_sh_under": filled_signed if leg_side == "long_under" else 0,
            "filled_sh_etf":   filled_signed if leg_side == "short_etf" else 0,
            "notes": f"P2B_{decision.upper()}_{leg_side.upper()} status={res.status}",
        })

    return fills
