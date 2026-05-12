#!/usr/bin/env python3
"""
substitution_engine.py

ETF substitution engine for tax-loss harvesting via Phase 2b.

When a Phase 2b trim would realize a loss on a long underlying, the
substitution engine looks up a configured pool of substitute ETFs
(e.g. ``IBIT -> [FBTC, BITB, ARKB]`` for spot BTC exposure) and
returns a candidate substitute. The Phase 2b tax router then emits a
paired swap: SELL the original at a loss, BUY the substitute for
equivalent dollar exposure.

State is persisted to ``data/active_substitutions.json``::

    {
      "IBIT": {
        "substitute":   "FBTC",
        "swap_in_date": "2026-05-11",
        "qty":          100,
        "leg":          "long",
        "swap_back_due": null
      }
    }

The "substantially identical" determination is the user's call; the
engine treats the configured pool as authoritative. Wash-sale rule
mechanics are sidestepped by virtue of substitution: we never hold
cash from a loss-trim, so no 30-day "buy back the same security"
clock starts on the original.

Stage 2 default behavior keeps the substitute indefinitely
(``maybe_swap_back`` returns ``[]``). Optional swap-back is reserved
for future iterations.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class SubstitutionConfig:
    enabled: bool = True
    min_loss_usd_to_substitute: float = 500.0
    hold_substitute_days: int = 31
    underlyings: Dict[str, List[str]] = field(default_factory=dict)
    # Optional: ETF (short-leg) pools — Stage 3 feature; Stage 2 ignores.
    letfs: Dict[str, List[str]] = field(default_factory=dict)
    # If a substitute's borrow cost exceeds this, skip and try the next.
    max_substitute_borrow_annual: float = 1.0

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "SubstitutionConfig":
        d = dict(d or {})
        return cls(
            enabled=bool(d.get("enabled", True)),
            min_loss_usd_to_substitute=float(d.get("min_loss_usd_to_substitute", 500.0)),
            hold_substitute_days=int(d.get("hold_substitute_days", 31)),
            underlyings={
                str(k).upper(): [str(s).upper() for s in (v or [])]
                for k, v in (d.get("underlyings") or {}).items()
            },
            letfs={
                str(k).upper(): [str(s).upper() for s in (v or [])]
                for k, v in (d.get("letfs") or {}).items()
            },
            max_substitute_borrow_annual=float(d.get("max_substitute_borrow_annual", 1.0)),
        )


# --------------------------------------------------------------------------
# Swap state
# --------------------------------------------------------------------------

@dataclass
class Swap:
    original: str
    substitute: str
    swap_in_date: date
    qty: int
    leg: str = "long"                   # "long" or "short"
    swap_back_due: Optional[date] = None


# --------------------------------------------------------------------------
# Engine
# --------------------------------------------------------------------------

class SubstitutionEngine:
    """Looks up substitutes, persists state, and (optionally) schedules swap-backs."""

    def __init__(
        self,
        config: SubstitutionConfig,
        state_path: Path,
        *,
        screener_borrow: Optional[Dict[str, float]] = None,
        screener_universe: Optional[Set[str]] = None,
    ):
        self.config = config
        self.state_path = Path(state_path)
        self.screener_borrow: Dict[str, float] = dict(screener_borrow or {})
        self.screener_universe: Set[str] = set(screener_universe or set())
        self._state: Dict[str, dict] = self._load_state()

    # ------------------------------------------------------------------
    # State persistence
    # ------------------------------------------------------------------

    def _load_state(self) -> Dict[str, dict]:
        if not self.state_path.exists():
            return {}
        try:
            data = json.loads(self.state_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
        except (json.JSONDecodeError, OSError):
            pass
        return {}

    def _save_state(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(
            json.dumps(self._state, indent=2, default=str),
            encoding="utf-8",
        )

    # ------------------------------------------------------------------
    # Lookups
    # ------------------------------------------------------------------

    def in_use_substitutes(self) -> Set[str]:
        """All substitute symbols currently in use across active swaps."""
        return {
            v.get("substitute", "").upper()
            for v in self._state.values()
            if v.get("substitute")
        }

    def find_substitute(self, original: str, leg: str = "long") -> Optional[str]:
        """Return the first eligible substitute for ``original``.

        Eligibility, in order:
          1. Configured in ``underlyings`` (long) or ``letfs`` (short, Stage 3) pool.
          2. Not already in-use as a substitute for another original.
          3. Present in today's screener universe (if one was provided).
          4. Borrow cost <= ``max_substitute_borrow_annual`` (if known and short leg).
        """
        if not self.config.enabled:
            return None
        original = original.upper()

        if leg == "long":
            pool = self.config.underlyings.get(original, [])
        else:
            pool = self.config.letfs.get(original, [])
        if not pool:
            return None

        in_use = self.in_use_substitutes()

        for cand in pool:
            cand = cand.upper()
            if cand == original:
                continue
            if cand in in_use:
                continue
            if self.screener_universe and cand not in self.screener_universe:
                continue
            if leg == "short":
                br = self.screener_borrow.get(cand)
                if br is not None and br > self.config.max_substitute_borrow_annual:
                    continue
            return cand

        return None

    # ------------------------------------------------------------------
    # State mutation
    # ------------------------------------------------------------------

    def record_swap(
        self,
        *,
        original: str,
        substitute: str,
        qty: int,
        leg: str = "long",
        swap_in_date: Optional[date] = None,
    ) -> Swap:
        """Persist a swap to active_substitutions.json. Idempotent on (original)."""
        if swap_in_date is None:
            swap_in_date = date.today()
        swap_back_due = (
            swap_in_date + timedelta(days=self.config.hold_substitute_days)
            if self.config.hold_substitute_days > 0
            else None
        )
        swap = Swap(
            original=original.upper(),
            substitute=substitute.upper(),
            swap_in_date=swap_in_date,
            qty=int(qty),
            leg=leg,
            swap_back_due=swap_back_due,
        )
        self._state[original.upper()] = {
            "substitute":     substitute.upper(),
            "swap_in_date":   swap_in_date.isoformat(),
            "qty":            int(qty),
            "leg":            leg,
            "swap_back_due":  swap_back_due.isoformat() if swap_back_due else None,
        }
        self._save_state()
        return swap

    def clear_swap(self, original: str) -> None:
        """Remove a swap entry (e.g. after unwind back to original)."""
        original = original.upper()
        if original in self._state:
            del self._state[original]
            self._save_state()

    def active_substitutions(self) -> Dict[str, Swap]:
        """Materialize current state into Swap objects (skips malformed entries)."""
        out: Dict[str, Swap] = {}
        for orig, v in self._state.items():
            try:
                swap_in = date.fromisoformat(v["swap_in_date"])
                swap_back_raw = v.get("swap_back_due")
                swap_back = (
                    date.fromisoformat(swap_back_raw)
                    if swap_back_raw else None
                )
                out[orig] = Swap(
                    original=orig,
                    substitute=str(v["substitute"]).upper(),
                    swap_in_date=swap_in,
                    qty=int(v.get("qty", 0)),
                    leg=str(v.get("leg", "long")),
                    swap_back_due=swap_back,
                )
            except (KeyError, ValueError, TypeError):
                continue
        return out

    # ------------------------------------------------------------------
    # Optional swap-back (default no-op for Stage 2)
    # ------------------------------------------------------------------

    def maybe_swap_back(self, asof: Optional[date] = None) -> List[Swap]:
        """Return swaps whose hold period has elapsed and should be unwound.

        Stage 2 default: keeps substitute indefinitely (returns ``[]``).
        Future iterations may enable opportunistic swap-back based on
        per-substitute config (e.g. ``swap_back_if_better_borrow``).
        """
        return []
