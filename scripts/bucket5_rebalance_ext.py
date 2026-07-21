"""Bucket 5 Production B — rebalance_strategy option-execution phase.

Called from ``rebalance_strategy.main()`` after the stock phases, with the
already-connected coordinator IB session. Behavior by mode
(``config/bucket5_production.yml``):

placeholder : refuse to touch option intents (logged skip).
shadow      : FORCED dry-run — resolve contracts, snapshot quotes, size from
              executable asks, archive hypothetical orders; zero submissions.
production  : submit limit orders under orderRef ``B5P|<intent_id>`` — but only
              for intents listed in
              ``data/runs/<date>/bucket5_production/approved_intents.csv``
              (manual release, plan amendment 2026-07-20), and only while
              ``kill_mode`` permits the action.

Fail-closed rules implemented here (plan sections 3.5 / 3.6 / 6):
- exact qualified contract with a real ``conId`` or skip;
- two-sided current quote, bounded spread, or skip — never a model fallback;
- contracts recomputed from the executable ask; never exceed the rung budget;
- pilot contract caps enforced in production mode;
- idempotent: intents already SUBMITTED/FILLED in the ledger are not resent,
  and open orders in the ``B5P|`` namespace are checked before submit;
- cancel hygiene: this phase only ever cancels orders whose orderRef starts
  with ``B5P|``.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

try:
    from scripts.bucket5_policy import load_b5_config, order_ref_for_intent
    from scripts.bucket5_ledger import (
        Bucket5Ledger, fill_event_id, submit_event_id,
    )
except ImportError:  # pragma: no cover
    from bucket5_policy import load_b5_config, order_ref_for_intent  # type: ignore
    from bucket5_ledger import Bucket5Ledger, fill_event_id, submit_event_id  # type: ignore


def _b5_dir(run_date: str, cfg: dict) -> Path:
    sub = str((cfg.get("paths") or {}).get("run_subdir", "bucket5_production"))
    return Path("data") / "runs" / run_date / sub


def _load_approved_intents(run_date: str, cfg: dict) -> set[str]:
    p = _b5_dir(run_date, cfg) / "approved_intents.csv"
    if not p.exists():
        return set()
    try:
        df = pd.read_csv(p)
    except Exception:
        return set()
    col = "intent_id" if "intent_id" in df.columns else df.columns[0]
    return set(df[col].astype(str).str.strip())


def _append_row(path: Path, row: dict) -> None:
    df = pd.DataFrame([row])
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, mode="a", header=not path.exists(), index=False)


def _snapshot_option_quote(ib, contract, timeout_s: float = 6.0) -> dict:
    """Snapshot bid/ask for a qualified option contract."""
    t = ib.reqMktData(contract, "", snapshot=False, regulatorySnapshot=False)
    waited = 0.0
    bid = ask = None
    while waited < timeout_s:
        ib.sleep(0.5)
        waited += 0.5
        b = getattr(t, "bid", None)
        a = getattr(t, "ask", None)
        if b is not None and a is not None and b == b and a == a and b > 0 and a > 0:
            bid, ask = float(b), float(a)
            break
    try:
        ib.cancelMktData(contract)
    except Exception:
        pass
    return {
        "bid": bid,
        "ask": ask,
        "quote_ts_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
    }


def _resolve_contract(ib, intent: dict) -> tuple[object | None, str]:
    """Resolve the intent to one exact qualified contract (nearest listed
    expiry >= target, nearest listed strike). Returns (contract, note)."""
    from ib_insync import Option

    instrument = str(intent.get("instrument", "XSP")).upper()
    right = str(intent.get("right", "P")).upper()
    target_expiry = str(intent.get("target_expiry", "")).replace("-", "")
    target_strike = float(intent.get("target_strike_instrument", 0.0))
    exchange = str(intent.get("exchange", "SMART"))

    # If GTP already pinned a conId, verify it directly.
    con_id = str(intent.get("conId", "") or "").strip()
    if con_id and con_id.lower() not in ("nan", "none", ""):
        try:
            from ib_insync import Contract

            c = Contract(conId=int(float(con_id)), exchange=exchange)
            (q,) = ib.qualifyContracts(c)
            return q, "pinned_conId"
        except Exception as ex:
            return None, f"pinned conId {con_id} failed to qualify: {ex}"

    try:
        probe = Option(instrument, target_expiry, target_strike, right, exchange)
        cds = ib.reqContractDetails(probe)
        if not cds:
            # Widen: let IB list strikes/expiries for the class and pick nearest.
            probe = Option(instrument, "", 0.0, right, exchange)
            cds = ib.reqContractDetails(probe)
        if not cds:
            return None, "no contract details returned"
        tgt_dt = pd.Timestamp(str(intent.get("target_expiry", "")))
        best, best_score = None, float("inf")
        for cd in cds:
            c = cd.contract
            try:
                exp = pd.Timestamp(str(c.lastTradeDateOrContractMonth))
            except Exception:
                continue
            days_off = abs((exp - tgt_dt).days)
            # Prefer expiries at/after target (roll target is a minimum DTE).
            if exp < tgt_dt:
                days_off += 15
            strike_off = abs(float(c.strike) - target_strike) / max(1.0, target_strike)
            score = days_off + 1000.0 * strike_off
            if score < best_score:
                best, best_score = c, score
        if best is None:
            return None, "no listed expiry/strike matched"
        (q,) = ib.qualifyContracts(best)
        if not getattr(q, "conId", 0):
            return None, "qualified contract missing conId"
        return q, f"resolved expiry={q.lastTradeDateOrContractMonth} strike={q.strike}"
    except Exception as ex:
        return None, f"contract resolution error: {ex}"


def run_b5_options_phase(
    *,
    ib,
    run_date: str,
    dry_run: bool,
    tprint=print,
    config_path: str | Path | None = None,
) -> None:
    """Execute (or dry-run) the day's B5 option intents. Never raises."""
    try:
        _run(ib=ib, run_date=run_date, dry_run=dry_run, tprint=tprint, config_path=config_path)
    except Exception as ex:
        tprint(f"[B5-OPT] WARNING: phase aborted fail-closed: {ex}")


def _run(*, ib, run_date: str, dry_run: bool, tprint, config_path) -> None:
    try:
        cfg = load_b5_config(config_path)
    except FileNotFoundError:
        return
    mode = cfg["mode"]
    if mode == "placeholder":
        tprint("[B5-OPT] mode=placeholder — option intents refused by design.")
        return

    b5_dir = _b5_dir(run_date, cfg)
    intents_path = b5_dir / "option_intents.csv"
    if not intents_path.exists():
        tprint(f"[B5-OPT] No option intents for {run_date} ({intents_path} missing); nothing to do.")
        return
    intents = pd.read_csv(intents_path)
    if intents.empty:
        tprint("[B5-OPT] Option intents file is empty; nothing to do.")
        return

    exec_cfg = cfg.get("execution") or {}
    opt_cfg = cfg.get("options") or {}
    kill_mode = cfg["kill_mode"]
    strategy_version = str(cfg.get("strategy_version", "B5PROD-1"))

    effective_dry_run = bool(dry_run) or mode != "production"
    if mode == "shadow":
        tprint("[B5-OPT] mode=shadow — FORCED dry-run; no orders will be submitted.")
    if kill_mode == "halt_all":
        tprint("[B5-OPT] kill_mode=halt_all — cancelling open B5P| orders and stopping.")
        _cancel_b5_open_orders(ib, tprint)
        return

    approved = _load_approved_intents(run_date, cfg)
    require_approval = bool(exec_cfg.get("require_manual_approval", True))
    ledger = Bucket5Ledger((cfg.get("paths") or {}).get("ledger_events", "data/bucket5_ledger/events.jsonl"))
    already_submitted = ledger.submitted_intent_ids()
    already_filled = ledger.filled_intent_ids()

    max_spread_frac = float(exec_cfg.get("max_spread_frac_of_mid", 0.25))
    cross_frac = float(exec_cfg.get("limit_cross_frac", 0.25))
    pilot_cap = int(opt_cfg.get("pilot_max_contracts_per_intent", 1))
    pilot_open_cap = int(opt_cfg.get("pilot_max_total_open_contracts", 6))
    contract_mult = float(opt_cfg.get("contract_multiplier", 100.0))

    open_contracts = sum(
        int(l.get("remaining_contracts", 0)) for l in ledger.open_lots().values()
    )

    # Open B5P| orders on the broker (idempotency source #2).
    open_refs: set[str] = set()
    try:
        for o in ib.reqAllOpenOrders():
            ref = str(getattr(o.order, "orderRef", "") or getattr(o, "orderRef", "") or "")
            if ref.startswith("B5P|"):
                open_refs.add(ref)
    except Exception:
        try:
            for o in ib.openTrades():
                ref = str(getattr(o.order, "orderRef", "") or "")
                if ref.startswith("B5P|"):
                    open_refs.add(ref)
        except Exception:
            pass

    audit_path = b5_dir / "option_execution_audit.csv"
    fills_path = b5_dir / "fills.csv"
    n_submitted = 0

    for _, intent in intents.iterrows():
        iid = str(intent.get("intent_id", "")).strip()
        action = str(intent.get("action", "BUY")).upper()
        order_ref = order_ref_for_intent(iid, str(exec_cfg.get("order_ref_prefix", "B5P")))
        base = {
            "run_date": run_date, "intent_id": iid, "action": action,
            "mode": mode, "kill_mode": kill_mode, "dry_run": effective_dry_run,
            "order_ref": order_ref, "strategy_version": strategy_version,
        }

        def _skip(reason: str, **extra) -> None:
            tprint(f"[B5-OPT] SKIP {iid}: {reason}")
            _append_row(audit_path, {**base, "outcome": "skipped", "reason": reason, **extra})

        if not iid:
            _skip("missing intent_id")
            continue
        if bool(intent.get("blocked_by_kill_mode", False)):
            _skip(f"blocked by kill_mode at plan time ({kill_mode})")
            continue
        risk_increasing = action == "BUY"
        if risk_increasing and kill_mode in ("no_new_risk", "reduce_only", "flatten_carry"):
            _skip(f"kill_mode={kill_mode} blocks risk-increasing intents")
            continue
        if iid in already_filled:
            _skip("intent already has a FILL in the ledger (idempotency)")
            continue
        if order_ref in open_refs:
            _skip("open broker order already exists for this orderRef (idempotency)")
            continue
        if iid in already_submitted and not effective_dry_run:
            _skip("intent already SUBMITTED in the ledger; resume/cancel manually, never resubmit")
            continue

        # ---------------------------------------------------------- contract
        contract, note = _resolve_contract(ib, intent)
        if contract is None:
            _skip(f"contract resolution failed: {note}")
            continue
        con_id = int(getattr(contract, "conId", 0) or 0)

        # ---------------------------------------------------------- quote
        q = _snapshot_option_quote(ib, contract)
        bid, ask = q.get("bid"), q.get("ask")
        if bid is None or ask is None or bid <= 0 or ask < bid:
            _skip("no valid two-sided quote (fail closed; no model fallback)",
                  conId=con_id, bid=bid, ask=ask)
            continue
        mid = 0.5 * (bid + ask)
        spread = ask - bid
        if mid <= 0 or spread / mid > max_spread_frac:
            _skip(f"spread {spread / mid:.1%} of mid exceeds max {max_spread_frac:.0%}",
                  conId=con_id, bid=bid, ask=ask)
            continue

        # ---------------------------------------------------------- sizing
        budget = float(intent.get("target_budget_usd", 0.0) or 0.0)
        qmult = max(1, int(intent.get("quantity_multiplier", 1) or 1))
        unit = ask * contract_mult
        if action == "BUY":
            baseline = int((budget / qmult) // unit) if unit > 0 else 0
            qty = baseline * qmult
            if qty <= 0:
                _skip(f"executable ask ${ask:.2f} -> unit ${unit:,.0f} breaches rung budget "
                      f"${budget:,.0f}; rung stays in cash (under-coverage alert)",
                      conId=con_id, bid=bid, ask=ask)
                continue
            if mode == "production":
                qty = min(qty, pilot_cap)
                if open_contracts + qty > pilot_open_cap:
                    _skip(f"pilot open-contract cap: open={open_contracts} + qty={qty} > {pilot_open_cap}",
                          conId=con_id)
                    continue
            limit_px = round(min(ask, mid + cross_frac * spread), 2)
        else:  # SELL (monetization / roll close) — sized by intent qty
            qty = int(float(intent.get("modeled_contracts", 0) or 0))
            if qty <= 0:
                _skip("SELL intent without positive quantity")
                continue
            limit_px = round(max(bid, mid - cross_frac * spread), 2)

        premium = qty * limit_px * contract_mult
        detail = {
            "conId": con_id,
            "local_symbol": getattr(contract, "localSymbol", ""),
            "expiry": getattr(contract, "lastTradeDateOrContractMonth", ""),
            "strike": getattr(contract, "strike", None),
            "right": getattr(contract, "right", "P"),
            "trading_class": getattr(contract, "tradingClass", ""),
            "bid": bid, "ask": ask, "mid": round(mid, 4),
            "quote_ts_utc": q["quote_ts_utc"],
            "qty": qty, "limit_px": limit_px,
            "premium_at_limit_usd": round(premium, 2),
            "resolution_note": note,
        }

        if effective_dry_run:
            tprint(
                f"[B5-OPT][DRY-RUN] {action} {qty}x {detail['local_symbol'] or contract.symbol} "
                f"conId={con_id} @ limit {limit_px} (bid {bid} / ask {ask}) ref={order_ref}"
            )
            _append_row(audit_path, {**base, "outcome": "dry_run_ok", "reason": "", **detail})
            continue

        # ---------------------------------------------------------- approval
        if require_approval and iid not in approved:
            _skip("manual approval required: intent_id not in approved_intents.csv",
                  **detail)
            continue

        # ---------------------------------------------------------- submit
        from ib_insync import LimitOrder

        order = LimitOrder(action, qty, limit_px)
        order.orderRef = order_ref
        order.tif = "DAY"
        order.outsideRth = False
        ledger.append("ORDER_SUBMITTED", submit_event_id(iid), {
            "intent_id": iid, "run_date": run_date, "action": action,
            "conId": con_id, "qty": qty, "limit_px": limit_px,
            "order_ref": order_ref, **{k: detail[k] for k in
                ("local_symbol", "expiry", "strike", "right", "trading_class")},
        })
        trade = ib.placeOrder(contract, order)
        tprint(f"[B5-OPT] SUBMITTED {action} {qty}x conId={con_id} @ {limit_px} ref={order_ref}")
        timeout = float(exec_cfg.get("timeout_sec", 120))
        waited = 0.0
        while waited < timeout and not trade.isDone():
            ib.sleep(1.0)
            waited += 1.0
        status = str(trade.orderStatus.status)
        filled = int(trade.orderStatus.filled or 0)
        avg_px = float(trade.orderStatus.avgFillPrice or 0.0)
        if filled > 0:
            fills = trade.fills or []
            exec_id = str(fills[-1].execution.execId) if fills else f"noexec-{run_date}"
            fees = 0.0
            for f in fills:
                cr = getattr(f, "commissionReport", None)
                if cr is not None and getattr(cr, "commission", None) is not None:
                    fees += float(cr.commission)
            signed_qty = filled if action == "BUY" else -filled
            ledger.append("FILL", fill_event_id(iid, exec_id), {
                "intent_id": iid, "run_date": run_date, "conId": con_id,
                "qty": signed_qty, "price": avg_px, "fees": fees,
                "multiplier": contract_mult, "order_ref": order_ref,
                **{k: detail[k] for k in
                   ("local_symbol", "expiry", "strike", "right", "trading_class")},
                "underlying": str(intent.get("instrument", "")),
                "sec_type": "OPT",
            })
            _append_row(fills_path, {**base, **detail, "outcome": status,
                                     "filled": filled, "avg_fill_px": avg_px, "fees": fees})
            open_contracts += filled if action == "BUY" else -filled
            n_submitted += 1
        if status not in ("Filled",) and not trade.isDone():
            # Bounded cancel: only our own B5P| order.
            try:
                ib.cancelOrder(order)
                ledger.append("ORDER_CANCELLED", f"CANCEL|{iid}|a1", {
                    "intent_id": iid, "run_date": run_date, "conId": con_id,
                    "order_ref": order_ref, "status_at_cancel": status, "filled": filled,
                })
                tprint(f"[B5-OPT] Cancelled unfilled remainder of {order_ref} (status={status}).")
            except Exception as ex:
                tprint(f"[B5-OPT] WARNING: cancel of {order_ref} failed: {ex}")
        _append_row(audit_path, {**base, "outcome": status, "reason": "",
                                 **detail, "filled": filled, "avg_fill_px": avg_px})

    tprint(
        f"[B5-OPT] Phase complete: {len(intents)} intents, {n_submitted} live orders "
        f"({'dry-run' if effective_dry_run else 'production'}); audit -> {audit_path}"
    )


def _cancel_b5_open_orders(ib, tprint) -> None:
    """Cancel ONLY orders in the B5P| namespace (cancel hygiene)."""
    try:
        for tr in ib.openTrades():
            ref = str(getattr(tr.order, "orderRef", "") or "")
            if ref.startswith("B5P|"):
                try:
                    ib.cancelOrder(tr.order)
                    tprint(f"[B5-OPT] halt_all: cancelled {ref}")
                except Exception as ex:
                    tprint(f"[B5-OPT] halt_all: cancel {ref} failed: {ex}")
    except Exception as ex:
        tprint(f"[B5-OPT] halt_all: open-order scan failed: {ex}")
