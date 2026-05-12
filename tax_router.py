#!/usr/bin/env python3
"""
tax_router.py

Stage 2 tax routing for Phase 2b resize trades.

Plugs into ``phase2b_resize.build_resize_trades`` via its
``tax_router`` parameter. Given a list of resize candidate trades
plus per-leg decisions, the router:

  * passes through pure GROW trades (no realized event)
  * for TRIM trades:
      - estimates realized P&L using the per-symbol :class:`LotView`
        and the assumed account default lot method (HIFO etc.)
      - if a GAIN and ``prefer_long_term_lots`` is set, limits the
        trim to LT inventory only (or annotates LT-first ordering)
      - if a LOSS above the substitution floor, looks up a configured
        substitute and emits a paired SELL-original / BUY-substitute
      - if a LOSS below the floor, emits a pure trim with the loss
        booked
      - if a LOSS with no substitute available, skips the trim and
        annotates ``deferred_reason``

Every routing decision is back-annotated onto the matching
:class:`ResizeDecision` so the resize_decisions.csv telemetry captures
the tax-routing trail (est P&L, lots consumed, ST/LT split,
substitute used, harvested loss, etc.).

The router never persists any state. Wiring code in
``rebalance_strategy.main()`` is responsible for calling
:func:`record_completed_swaps_from_fills` after execution to update
the substitution engine.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

from phase2b_resize import ResizeDecision
from substitution_engine import SubstitutionConfig, SubstitutionEngine
from tax_lot_view import EstPnL, LotView


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class TaxConfig:
    enabled: bool = True
    lot_method_assumed: str = "HIFO"     # must match TWS Account Configuration default
    prefer_long_term_lots: bool = True
    st_lt_holding_days: int = 365

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "TaxConfig":
        d = dict(d or {})
        return cls(
            enabled=bool(d.get("enabled", True)),
            lot_method_assumed=str(d.get("lot_method_assumed", "HIFO")).upper(),
            prefer_long_term_lots=bool(d.get("prefer_long_term_lots", True)),
            st_lt_holding_days=int(d.get("st_lt_holding_days", 365)),
        )


# --------------------------------------------------------------------------
# Per-trim classification
# --------------------------------------------------------------------------

@dataclass
class TrimClassification:
    """Result of classifying a single trim candidate."""
    action: str                   # "pure_trim" | "lt_only_trim" | "harvest_via_sub" | "skip"
    qty: int = 0
    est_pnl_usd: float = 0.0
    st_qty: float = 0.0
    lt_qty: float = 0.0
    lots_consumed_count: int = 0
    substitute: str = ""
    note: str = ""                # free-text annotation


def classify_trim(
    *,
    symbol: str,
    qty: int,
    current_px: float,
    leg_side: str,
    lot_view: Optional[LotView],
    tax_cfg: TaxConfig,
    sub_engine: Optional[SubstitutionEngine],
) -> TrimClassification:
    """Classify a single trim candidate.

    Returns
    -------
    TrimClassification
        ``action`` is one of:
          * ``pure_trim`` — execute the trim as-is (loss below sub floor,
            sub disabled, gain with no LT preference, or no lot data)
          * ``lt_only_trim`` — limit qty to LT inventory only, prefer LT
            lots in matching
          * ``harvest_via_sub`` — emit SELL-original + BUY-substitute swap
          * ``skip`` — defer trim (e.g. loss but no substitute available)
    """
    # No lot data -> can't reason about realized P&L; pass through.
    if lot_view is None or not lot_view.lots:
        return TrimClassification(
            action="pure_trim", qty=int(qty),
            est_pnl_usd=0.0, note="no_lot_data",
        )

    method = tax_cfg.lot_method_assumed
    days = tax_cfg.st_lt_holding_days

    est: EstPnL = lot_view.estimate_realized_pnl(
        qty=float(qty), current_px=float(current_px),
        method=method, st_lt_holding_days=days,
    )

    # Cap qty at available inventory (broker would do same but be explicit)
    if est.qty_consumed < qty:
        qty = int(est.qty_consumed)
        if qty <= 0:
            return TrimClassification(
                action="skip", note="no_inventory_to_close",
            )

    # ---- GAIN branch: prefer LT when configured -----------------------------
    if est.realized_pnl_usd >= 0:
        if tax_cfg.prefer_long_term_lots:
            st_qty, lt_qty = lot_view.st_lt_split(days)
            if lt_qty >= qty:
                # Plenty of LT — re-estimate with LT-first ordering for telemetry
                est_lt = lot_view.estimate_realized_pnl(
                    qty=float(qty), current_px=float(current_px),
                    method=method, st_lt_holding_days=days, prefer_lt=True,
                )
                return TrimClassification(
                    action="lt_only_trim", qty=int(qty),
                    est_pnl_usd=est_lt.realized_pnl_usd,
                    st_qty=est_lt.st_qty, lt_qty=est_lt.lt_qty,
                    lots_consumed_count=len(est_lt.lots_consumed),
                    note="lt_inventory_sufficient",
                )
            if lt_qty > 0:
                limited = int(lt_qty)
                est_lt = lot_view.estimate_realized_pnl(
                    qty=float(limited), current_px=float(current_px),
                    method=method, st_lt_holding_days=days, prefer_lt=True,
                )
                return TrimClassification(
                    action="lt_only_trim", qty=limited,
                    est_pnl_usd=est_lt.realized_pnl_usd,
                    st_qty=est_lt.st_qty, lt_qty=est_lt.lt_qty,
                    lots_consumed_count=len(est_lt.lots_consumed),
                    note=f"limited_to_lt_qty={limited}",
                )
            # No LT inventory — fall through to pure trim with the gain.

        return TrimClassification(
            action="pure_trim", qty=int(qty),
            est_pnl_usd=est.realized_pnl_usd,
            st_qty=est.st_qty, lt_qty=est.lt_qty,
            lots_consumed_count=len(est.lots_consumed),
            note="gain_no_lt_preference" if not tax_cfg.prefer_long_term_lots else "gain_no_lt_inventory",
        )

    # ---- LOSS branch: route to substitution if eligible ---------------------
    loss_usd = abs(est.realized_pnl_usd)

    if sub_engine is None or not sub_engine.config.enabled:
        return TrimClassification(
            action="pure_trim", qty=int(qty),
            est_pnl_usd=est.realized_pnl_usd,
            st_qty=est.st_qty, lt_qty=est.lt_qty,
            lots_consumed_count=len(est.lots_consumed),
            note="substitution_disabled",
        )

    if loss_usd < sub_engine.config.min_loss_usd_to_substitute:
        return TrimClassification(
            action="pure_trim", qty=int(qty),
            est_pnl_usd=est.realized_pnl_usd,
            st_qty=est.st_qty, lt_qty=est.lt_qty,
            lots_consumed_count=len(est.lots_consumed),
            note=(f"loss_below_min({loss_usd:.0f}<"
                  f"{sub_engine.config.min_loss_usd_to_substitute:.0f})"),
        )

    # leg_side is "long_under" or "short_etf"; map to "long"/"short" for engine
    sub_leg = "long" if leg_side.startswith("long") else "short"
    substitute = sub_engine.find_substitute(symbol, leg=sub_leg)
    if not substitute:
        return TrimClassification(
            action="skip", qty=int(qty),
            est_pnl_usd=est.realized_pnl_usd,
            note="no_substitute_for_loss_trim",
        )

    return TrimClassification(
        action="harvest_via_sub", qty=int(qty),
        est_pnl_usd=est.realized_pnl_usd,
        st_qty=est.st_qty, lt_qty=est.lt_qty,
        lots_consumed_count=len(est.lots_consumed),
        substitute=substitute,
        note=f"harvest_loss={loss_usd:.0f}_via={substitute}",
    )


# --------------------------------------------------------------------------
# tax_router factory: matches phase2b_resize.build_resize_trades hook
# --------------------------------------------------------------------------

TaxRouterFn = Callable[
    [List[Dict], List[ResizeDecision]],
    Tuple[List[Dict], List[ResizeDecision]],
]


def make_tax_router(
    *,
    lot_views: Dict[str, LotView],
    prices: Dict[str, float],
    tax_cfg: TaxConfig,
    sub_engine: Optional[SubstitutionEngine],
) -> TaxRouterFn:
    """Build a router function that transforms (trades, decisions).

    The returned callable matches the ``tax_router`` hook signature in
    :func:`phase2b_resize.build_resize_trades`. It is pure (no state
    mutation) — wiring code in ``rebalance_strategy`` is responsible
    for calling :func:`record_completed_swaps_from_fills` after the
    execution phase to update the substitution engine.
    """

    def _annotate(d: Optional[ResizeDecision], cls: TrimClassification) -> None:
        if d is None:
            return
        d.est_realized_pnl_usd = float(cls.est_pnl_usd)
        d.st_qty_consumed = float(cls.st_qty)
        d.lt_qty_consumed = float(cls.lt_qty)
        d.lots_consumed_count = int(cls.lots_consumed_count)
        if cls.substitute:
            d.swap_with = cls.substitute
        if cls.note:
            d.reason = (d.reason + "|" if d.reason else "") + f"tax:{cls.note}"

    def router(
        trades: List[Dict],
        decisions: List[ResizeDecision],
    ) -> Tuple[List[Dict], List[ResizeDecision]]:
        if not tax_cfg.enabled or not trades:
            return trades, decisions

        # Two indexes because build_resize_trades uses different etf
        # conventions for long_under (empty in trade, comma-joined ETFs in
        # decision) vs short_etf (specific ETF symbol in both):
        #   * exact     - matches by (underlying, etf, leg_side)
        #   * by_leg    - matches by (underlying, leg_side); fallback for
        #                 long_under where the trade has etf=""
        idx_exact: Dict[Tuple[str, str, str], ResizeDecision] = {
            (d.underlying, d.etf, d.leg_side): d for d in decisions
        }
        idx_by_leg: Dict[Tuple[str, str], ResizeDecision] = {
            (d.underlying, d.leg_side): d for d in decisions
            if d.leg_side == "long_under"
        }

        def _lookup(t: Dict) -> Optional[ResizeDecision]:
            key = (t["underlying"], t.get("etf") or "", t["leg_side"])
            d = idx_exact.get(key)
            if d is not None:
                return d
            if t["leg_side"] == "long_under":
                return idx_by_leg.get((t["underlying"], "long_under"))
            return None

        out_trades: List[Dict] = []
        for t in trades:
            decision = t.get("decision")
            if decision != "trim":
                out_trades.append(t)
                continue

            symbol   = t["symbol"]
            qty      = int(t["qty"])
            leg_side = t["leg_side"]
            current_px = float(prices.get(symbol, t.get("ref_price", 0.0)))

            lv = lot_views.get(symbol)
            cls = classify_trim(
                symbol=symbol, qty=qty, current_px=current_px,
                leg_side=leg_side, lot_view=lv,
                tax_cfg=tax_cfg, sub_engine=sub_engine,
            )

            d = _lookup(t)

            if cls.action == "skip":
                _annotate(d, cls)
                if d is not None:
                    d.decision = "skip"
                    d.qty = 0
                continue

            if cls.action == "lt_only_trim":
                t = {**t, "qty": int(cls.qty)}
                _annotate(d, cls)
                if d is not None:
                    d.decision = "lt_only_trim"
                    d.qty = int(cls.qty)
                out_trades.append(t)
                continue

            if cls.action == "harvest_via_sub":
                # We need a price for the substitute; without it the swap
                # cannot proceed and we skip the trim with a clear reason.
                sub_sym = cls.substitute
                sub_px = float(prices.get(sub_sym, 0.0))
                if sub_px <= 0:
                    cls = TrimClassification(
                        action="skip", qty=qty,
                        est_pnl_usd=cls.est_pnl_usd,
                        note=f"sub_no_price={sub_sym}",
                    )
                    _annotate(d, cls)
                    if d is not None:
                        d.decision = "skip"
                        d.qty = 0
                    continue

                # Original SELL-original leg: tag with swap metadata so the
                # post-execution wiring can persist the swap state.
                t_sell = {
                    **t,
                    "decision":      "harvest_sub_sell",
                    "swap_with":     sub_sym,
                    "substitute_of": "",
                    "est_pnl":       float(cls.est_pnl_usd),
                }
                _annotate(d, cls)
                if d is not None:
                    d.decision = "harvest_sub_sell"
                    d.harvested_loss_usd = abs(float(cls.est_pnl_usd))
                out_trades.append(t_sell)

                # Substitute leg: BUY equivalent dollar exposure (long swap)
                # or SELL equivalent (short swap, Stage 3 LETF case).
                sub_action = "BUY" if leg_side == "long_under" else "SELL"
                notional = float(qty) * float(current_px)
                sub_qty = int(notional / sub_px)
                if sub_qty <= 0:
                    cls_sub = TrimClassification(
                        action="skip", qty=0,
                        note=f"sub_qty=0_for={sub_sym}",
                    )
                    _annotate(d, cls_sub)
                    if d is not None:
                        d.decision = "skip"
                        d.qty = 0
                    # Drop the SELL too — never break the long without the BUY
                    out_trades.pop()
                    continue

                t_buy = {
                    "underlying":       t["underlying"],
                    "etf":              "" if leg_side == "long_under" else sub_sym,
                    "leg_side":         leg_side,
                    "symbol":           sub_sym,
                    "action":           sub_action,
                    "qty":              int(sub_qty),
                    "ref_price":        float(sub_px),
                    "trade_usd_target": float(notional),
                    "decision":         "harvest_sub_buy",
                    "reason":           f"substitute_for={symbol}",
                    "target_usd":       float(t.get("target_usd", 0.0)),
                    "current_usd":      0.0,
                    "swap_with":        symbol,
                    "substitute_of":    symbol,
                    "est_pnl":          0.0,
                }
                # Append a synthetic decision row for the substitute leg so
                # telemetry includes the harvest BUY.
                sub_dec = ResizeDecision(
                    underlying=t["underlying"],
                    etf="" if leg_side == "long_under" else sub_sym,
                    leg_side=leg_side,
                    target_usd=float(t.get("target_usd", 0.0)),
                    current_usd=0.0,
                    delta_usd=-float(notional),
                    enter_threshold_usd=0.0, exit_threshold_usd=0.0,
                    decision="harvest_sub_buy",
                    reason=f"substitute_for={symbol}",
                    action=sub_action, qty=int(sub_qty),
                    ref_price=float(sub_px), trade_usd=float(notional),
                    substitute_of=symbol, swap_with=symbol,
                )
                decisions.append(sub_dec)
                out_trades.append(t_buy)
                continue

            # pure_trim: pass through with annotation
            _annotate(d, cls)
            out_trades.append(t)

        return out_trades, decisions

    return router


# --------------------------------------------------------------------------
# Post-execution: persist successful swaps
# --------------------------------------------------------------------------

def record_completed_swaps_from_fills(
    trades: List[Dict],
    fills: List[Dict],
    sub_engine: Optional[SubstitutionEngine],
) -> int:
    """Inspect post-execution fills and persist any successful swap pairs.

    A swap is "successful" if BOTH the SELL-original and BUY-substitute
    legs filled (any non-zero filled qty). Partial fills count — better
    to track an in-progress swap than miss a completed one.

    Returns
    -------
    int
        Number of swaps persisted.
    """
    if sub_engine is None or not trades:
        return 0

    # Build a fill lookup: { (symbol, action) -> total_filled_abs }
    fill_qty: Dict[Tuple[str, str], int] = {}
    for f in fills:
        sym = f.get("etf") or ""
        if not sym:
            sym = f.get("underlying", "") or ""
            filled = int(f.get("filled_sh_under", 0) or 0)
        else:
            filled = int(f.get("filled_sh_etf", 0) or 0)
        if filled == 0:
            continue
        action = "BUY" if filled > 0 else "SELL"
        fill_qty[(sym, action)] = fill_qty.get((sym, action), 0) + abs(filled)

    n_persisted = 0
    seen_swaps: set = set()

    # Find SELL-original legs that have a swap_with paired BUY-substitute
    for t in trades:
        if not t.get("swap_with"):
            continue
        # Identify direction: SELL original leg has substitute_of == ""
        if t.get("substitute_of"):
            continue  # this is the substitute BUY leg; handled via the sell

        original_sym = t["symbol"]
        substitute_sym = t["swap_with"]
        if (original_sym, substitute_sym) in seen_swaps:
            continue

        sell_filled = fill_qty.get((original_sym, "SELL"), 0)
        # Substitute leg action is BUY for long swap, SELL for short swap
        leg_side = t.get("leg_side", "long_under")
        sub_action = "BUY" if leg_side == "long_under" else "SELL"
        sub_filled = fill_qty.get((substitute_sym, sub_action), 0)

        if sell_filled > 0 and sub_filled > 0:
            sub_engine.record_swap(
                original=original_sym,
                substitute=substitute_sym,
                qty=int(sub_filled),
                leg="long" if leg_side == "long_under" else "short",
            )
            seen_swaps.add((original_sym, substitute_sym))
            n_persisted += 1

    return n_persisted
