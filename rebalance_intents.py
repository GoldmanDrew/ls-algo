"""Pure helpers for cross-phase rebalance intent coordination."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping


def signed_qty(trade: Mapping[str, Any], *, qty: float | None = None) -> float:
    amount = abs(float(trade.get("qty", 0.0) if qty is None else qty))
    action = str(trade.get("action", "")).strip().upper()
    if action == "BUY":
        return amount
    if action == "SELL":
        return -amount
    return 0.0


def project_positions(
    positions: Mapping[str, float],
    trades: Iterable[Mapping[str, Any]],
) -> dict[str, float]:
    out = {str(symbol): float(shares) for symbol, shares in positions.items()}
    for trade in trades:
        symbol = str(trade.get("symbol", "")).strip().upper()
        if not symbol:
            continue
        out[symbol] = out.get(symbol, 0.0) + signed_qty(trade)
    return out


def coalesce_intents(trades: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Net signed quantities into at most one intent per symbol."""
    grouped: dict[str, dict[str, Any]] = {}
    for raw in trades:
        trade = dict(raw)
        symbol = str(trade.get("symbol", "")).strip().upper()
        if not symbol:
            continue
        signed = signed_qty(trade)
        if symbol not in grouped:
            grouped[symbol] = {
                **trade,
                "symbol": symbol,
                "_signed_qty": 0.0,
                "source_phases": [],
                "source_reasons": [],
                "gross_intent_qty": 0.0,
            }
        row = grouped[symbol]
        row["_signed_qty"] += signed
        row["gross_intent_qty"] += abs(signed)
        phase = str(trade.get("source_phase", trade.get("phase", "")) or "")
        reason = str(trade.get("reason", "") or "")
        if phase and phase not in row["source_phases"]:
            row["source_phases"].append(phase)
        if reason and reason not in row["source_reasons"]:
            row["source_reasons"].append(reason)

    out: list[dict[str, Any]] = []
    for row in grouped.values():
        signed = float(row.pop("_signed_qty"))
        gross = float(row.get("gross_intent_qty", 0.0) or 0.0)
        if abs(signed) < 1e-9:
            continue
        row["action"] = "BUY" if signed > 0 else "SELL"
        row["qty"] = int(abs(signed))
        row["netted_qty"] = max(0.0, gross - abs(signed))
        row["source_phases"] = "|".join(row["source_phases"])
        row["source_reasons"] = "|".join(row["source_reasons"])
        if row["qty"] > 0:
            out.append(row)
    return out


def clip_against_opposing_intents(
    primary: Iterable[Mapping[str, Any]],
    opposing: Iterable[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Clip earlier intents by predictable later opposite-side quantities."""
    opposing_signed: dict[str, float] = {}
    for trade in opposing:
        symbol = str(trade.get("symbol", "")).strip().upper()
        if symbol:
            opposing_signed[symbol] = opposing_signed.get(symbol, 0.0) + signed_qty(trade)

    adjusted: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    for raw in primary:
        trade = dict(raw)
        symbol = str(trade.get("symbol", "")).strip().upper()
        current = signed_qty(trade)
        later = opposing_signed.get(symbol, 0.0)
        clipped = 0.0
        if current * later < 0:
            clipped = min(abs(current), abs(later))
            current += clipped if current < 0 else -clipped
            opposing_signed[symbol] = later + clipped if later < 0 else later - clipped
        if clipped > 0:
            audit.append(
                {
                    "event": "PREDICTED_CHURN_NETTED",
                    "symbol": symbol,
                    "source_phase": str(trade.get("source_phase", "phase2b")),
                    "original_action": str(trade.get("action", "")),
                    "original_qty": abs(signed_qty(trade)),
                    "netted_qty": clipped,
                    "remaining_qty": abs(current),
                }
            )
        if abs(current) >= 1.0:
            trade["action"] = "BUY" if current > 0 else "SELL"
            trade["qty"] = int(abs(current))
            if clipped > 0:
                trade["reason"] = (
                    str(trade.get("reason", "") or "") + "|projected_phase3_net"
                ).strip("|")
                trade["netted_qty"] = float(trade.get("netted_qty", 0.0) or 0.0) + clipped
            adjusted.append(trade)
    return adjusted, audit


@dataclass
class SameRunIntentLedger:
    enabled: bool = True
    projected_positions: dict[str, float] = field(default_factory=dict)
    records: list[dict[str, Any]] = field(default_factory=list)
    _directions: dict[str, set[str]] = field(default_factory=dict)

    def set_projected_positions(self, positions: Mapping[str, float]) -> None:
        self.projected_positions = {
            str(symbol).strip().upper(): float(shares)
            for symbol, shares in positions.items()
        }

    def record(
        self,
        trade: Mapping[str, Any],
        *,
        phase: str,
        status: str,
        qty: float | None = None,
        reason: str = "",
    ) -> None:
        symbol = str(trade.get("symbol", "")).strip().upper()
        action = str(trade.get("action", "")).strip().upper()
        if not symbol or action not in {"BUY", "SELL"}:
            return
        amount = abs(float(trade.get("qty", 0.0) if qty is None else qty))
        self.records.append(
            {
                "event": status,
                "phase": phase,
                "symbol": symbol,
                "action": action,
                "qty": amount,
                "reason": reason or str(trade.get("reason", "") or ""),
                "intent_id": str(trade.get("intent_id", "") or ""),
            }
        )
        if status in {"SUBMITTED", "FILLED", "PARTIAL"} and amount > 0:
            self._directions.setdefault(symbol, set()).add(action)

    def guard(
        self,
        trade: Mapping[str, Any],
        *,
        phase: str,
        allow_risk_override: bool,
        override_reason: str = "",
        evidence: Mapping[str, Any] | None = None,
    ) -> tuple[bool, dict[str, Any], dict[str, Any] | None]:
        candidate = dict(trade)
        symbol = str(candidate.get("symbol", "")).strip().upper()
        action = str(candidate.get("action", "")).strip().upper()
        prior = self._directions.get(symbol, set())
        opposite = "SELL" if action == "BUY" else "BUY"
        if not self.enabled or opposite not in prior:
            return True, candidate, None

        evidence_dict = dict(evidence or {})
        if allow_risk_override:
            candidate["churn_guard"] = "RISK_OVERRIDE"
            candidate["risk_override_reason"] = override_reason or "projection_drift"
            row = {
                "event": "RISK_OVERRIDE",
                "phase": phase,
                "symbol": symbol,
                "action": action,
                "qty": abs(signed_qty(candidate)),
                "reason": candidate["risk_override_reason"],
                **evidence_dict,
            }
            self.records.append(row)
            return True, candidate, row

        row = {
            "event": "SAME_RUN_CHURN",
            "phase": phase,
            "symbol": symbol,
            "action": action,
            "qty": abs(signed_qty(candidate)),
            "reason": "opposes_prior_same_run_intent",
            **evidence_dict,
        }
        self.records.append(row)
        return False, candidate, row


__all__ = [
    "SameRunIntentLedger",
    "clip_against_opposing_intents",
    "coalesce_intents",
    "project_positions",
    "signed_qty",
]
