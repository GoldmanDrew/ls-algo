"""Bucket 5 Production B — event-sourced option/carry ledger.

Append-only JSONL event log plus a deterministic replay that materializes the
authoritative B5 option lot book keyed by IBKR ``conId`` (plan section 3.3/3.6).
Plan rows are intents; THIS ledger is the inventory of record.

Event kinds
-----------
INTENT_EMITTED     GTP produced a target (option or carry) with a deterministic id
ORDER_SUBMITTED    rebalancer sent an order (orderRef = B5P|<intent_id>)
ORDER_CANCELLED    order cancelled / expired without (full) fill
FILL               (partial) fill; creates/extends/reduces a conId lot
TIER_FIRED         a monetization tier fired (idempotency: fires once per lot)
PEAK_MARK          new peak executable mark for a lot
ROLL_LINK          links a re-arm/roll child lot to its parent intent
RECONCILE          daily reconciliation result snapshot
KILL_MODE          operator/kill-mode transition
NOTE               free-form operator annotation

Idempotency: every event carries an ``event_id``; appending an event whose id
is already present is a no-op, so a rerun/restart can never double-book a fill
or fire a tier twice.
"""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

DEFAULT_LEDGER_PATH = Path("data") / "bucket5_ledger" / "events.jsonl"

EVENT_KINDS = {
    "INTENT_EMITTED", "ORDER_SUBMITTED", "ORDER_CANCELLED", "FILL",
    "TIER_FIRED", "PEAK_MARK", "ROLL_LINK", "RECONCILE", "KILL_MODE", "NOTE",
}


def _utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


class Bucket5Ledger:
    """Append-only JSONL ledger with idempotent appends and lot replay."""

    def __init__(self, path: str | Path = DEFAULT_LEDGER_PATH):
        self.path = Path(path)
        self._seen_ids: set[str] | None = None

    # ------------------------------------------------------------------ I/O
    def load_events(self) -> list[dict]:
        if not self.path.exists():
            return []
        events: list[dict] = []
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError as ex:
                    raise ValueError(
                        f"Corrupt B5 ledger line in {self.path}: {ex}. "
                        "Fail closed: fix or archive the ledger before trading."
                    ) from ex
        return events

    def _ensure_seen(self) -> set[str]:
        if self._seen_ids is None:
            self._seen_ids = {str(e.get("event_id", "")) for e in self.load_events()}
        return self._seen_ids

    def has_event(self, event_id: str) -> bool:
        return str(event_id) in self._ensure_seen()

    def append(self, kind: str, event_id: str, payload: dict | None = None) -> bool:
        """Append one event. Returns False (no-op) if event_id already exists."""
        if kind not in EVENT_KINDS:
            raise ValueError(f"Unknown B5 ledger event kind: {kind!r}")
        event_id = str(event_id)
        if not event_id:
            raise ValueError("event_id is required")
        seen = self._ensure_seen()
        if event_id in seen:
            return False
        row = {"event_id": event_id, "kind": kind, "ts_utc": _utcnow()}
        row.update(payload or {})
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, sort_keys=True, default=str) + "\n")
        seen.add(event_id)
        return True

    # ----------------------------------------------------------------- Replay
    def build_lots(self) -> dict[str, dict]:
        """Replay all events into the option lot book keyed by str(conId).

        Lot fields: conId, local_symbol, underlying, sec_type, expiry, strike,
        right, trading_class, multiplier, entry_contracts, remaining_contracts,
        cost_basis_usd, fees_usd, realized_cash_usd, peak_mult,
        profit_tiers_fired, vix_tiers_fired, entry_intent_id, parent_intent_id,
        first_fill_ts, last_event_ts.
        """
        lots: dict[str, dict] = {}
        for e in self.load_events():
            kind = e.get("kind")
            ts = e.get("ts_utc", "")
            if kind == "FILL":
                con_id = str(e.get("conId", "") or "")
                if not con_id:
                    continue
                lot = lots.setdefault(con_id, {
                    "conId": con_id,
                    "local_symbol": e.get("local_symbol", ""),
                    "underlying": e.get("underlying", ""),
                    "sec_type": e.get("sec_type", "OPT"),
                    "expiry": e.get("expiry", ""),
                    "strike": e.get("strike"),
                    "right": e.get("right", "P"),
                    "trading_class": e.get("trading_class", ""),
                    "multiplier": e.get("multiplier", 100.0),
                    "entry_contracts": 0,
                    "remaining_contracts": 0,
                    "cost_basis_usd": 0.0,
                    "fees_usd": 0.0,
                    "realized_cash_usd": 0.0,
                    "peak_mult": 1.0,
                    "profit_tiers_fired": [],
                    "vix_tiers_fired": [],
                    "entry_intent_id": e.get("intent_id", ""),
                    "parent_intent_id": e.get("parent_intent_id", ""),
                    "first_fill_ts": ts,
                    "last_event_ts": ts,
                })
                qty = int(e.get("qty", 0) or 0)          # signed: + buy, - sell
                px = float(e.get("price", 0.0) or 0.0)   # per-share premium
                mult = float(e.get("multiplier", lot.get("multiplier") or 100.0) or 100.0)
                fees = float(e.get("fees", 0.0) or 0.0)
                lot["fees_usd"] += fees
                lot["last_event_ts"] = ts
                if qty > 0:
                    lot["entry_contracts"] += qty
                    lot["remaining_contracts"] += qty
                    lot["cost_basis_usd"] += qty * px * mult
                elif qty < 0:
                    sell_n = min(-qty, int(lot["remaining_contracts"]))
                    lot["remaining_contracts"] -= sell_n
                    lot["realized_cash_usd"] += sell_n * px * mult
            elif kind == "TIER_FIRED":
                con_id = str(e.get("conId", "") or "")
                lot = lots.get(con_id)
                if lot is None:
                    continue
                tier_kind = str(e.get("tier_kind", "profit"))
                level = float(e.get("level", 0.0) or 0.0)
                key = "vix_tiers_fired" if tier_kind == "vix" else "profit_tiers_fired"
                if level not in lot[key]:
                    lot[key] = sorted(set(lot[key]) | {level})
                lot["last_event_ts"] = ts
            elif kind == "PEAK_MARK":
                con_id = str(e.get("conId", "") or "")
                lot = lots.get(con_id)
                if lot is None:
                    continue
                lot["peak_mult"] = max(float(lot.get("peak_mult", 1.0)), float(e.get("peak_mult", 1.0) or 1.0))
                lot["last_event_ts"] = ts
            elif kind == "ROLL_LINK":
                con_id = str(e.get("conId", "") or "")
                lot = lots.get(con_id)
                if lot is not None:
                    lot["parent_intent_id"] = e.get("parent_intent_id", lot.get("parent_intent_id", ""))
                    lot["last_event_ts"] = ts
        return lots

    def open_lots(self) -> dict[str, dict]:
        return {k: v for k, v in self.build_lots().items() if int(v.get("remaining_contracts", 0)) > 0}

    def submitted_intent_ids(self) -> set[str]:
        return {
            str(e.get("intent_id", ""))
            for e in self.load_events()
            if e.get("kind") == "ORDER_SUBMITTED" and e.get("intent_id")
        }

    def filled_intent_ids(self) -> set[str]:
        return {
            str(e.get("intent_id", ""))
            for e in self.load_events()
            if e.get("kind") == "FILL" and e.get("intent_id")
        }


# ---------------------------------------------------------------------------
# Convenience event constructors (keep event_id derivations in one place)
# ---------------------------------------------------------------------------

def intent_event_id(intent_id: str) -> str:
    return f"INTENT|{intent_id}"


def submit_event_id(intent_id: str, attempt: int = 1) -> str:
    return f"SUBMIT|{intent_id}|a{int(attempt)}"


def fill_event_id(intent_id: str, exec_id: str) -> str:
    return f"FILL|{intent_id}|{exec_id}"


def tier_event_id(con_id: str, tier_kind: str, level: float) -> str:
    return f"TIER|{con_id}|{tier_kind}|{level:g}"
