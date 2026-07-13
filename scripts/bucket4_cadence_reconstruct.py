"""Reconstruct Bucket-4 per-pair rebalance cadence timestamps from the broker.

Cadence timing (``data/b4_cadence_state.json``) records the last date each
``(ETF, underlying)`` pair actually resized, so ``bucket4_cadence_gate`` can
defer pairs that are not yet due. That JSON is a *local mutable* cache: it is
not committed, and on a fresh machine it does not exist, which makes every pair
look "due" and can trigger a synchronized rebalance burst.

This module makes the timing **machine-independent without committing cadence
data**, mirroring ``scripts/b4_reconstruct_state.py`` (which reconstructs the
grow-only ratchet floor from committed accounting). Here the source of truth is
the broker's own execution record: the IBKR Flex ``flex_trades.xml`` report that
the daily pipeline already saves under ``data/runs/<D>/ibkr_flex/``.

Per-pair signal: the **inverse-ETF leg**. Each inverse ETF (e.g. MSTX, MSTU,
MTYY, MSTZ) maps to exactly one B4 pair, so its most recent trade date is an
unambiguous "this pair last rebalanced" marker. The underlying leg (e.g. MSTR)
is netted/shared across pairs and B1/B2, so it cannot disambiguate per-pair
timing and is deliberately ignored (worst case a pair looks due slightly early,
which is the safe direction).

Resolution per pair (most-recent evidence wins, never regress):
    last_rebalance = max(local_cache_date, broker_trade_date)
When a pair has neither, seed a *staggered* date::
    last_rebalance = run_date - offset_business_days
where ``offset`` is a deterministic hash of the pair key in ``[0, interval]`` so
a fresh machine does not fire every pair on the same day.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "data" / "runs"

# Flex reports are wide (full history); reject nothing on size, but only treat
# equity/ETF legs as cadence-bearing trades.
_TRADE_TAGS = ("Trade", "TradeConfirm", "Order")
_EQUITY_ASSET_CATEGORIES = frozenset({"STK", "ETF", ""})


def _norm_sym(x: object) -> str:
    """Match ``execute_trade_plan.norm_sym`` so keys line up with the gate."""
    return str(x or "").upper().strip().replace(".", "-")


def _pair_key(etf: str, underlying: str) -> str:
    return f"{_norm_sym(etf)}|{_norm_sym(underlying)}"


def _yyyymmdd_to_iso(raw: str | None) -> str | None:
    """``20260711`` (or ``20260711;145545``) -> ``2026-07-11``."""
    if not raw:
        return None
    s = str(raw).strip()
    if ";" in s:
        s = s.split(";", 1)[0].strip()
    s = s.replace("-", "")
    if len(s) < 8 or not s[:8].isdigit():
        return None
    return f"{s[0:4]}-{s[4:6]}-{s[6:8]}"


# ---------------------------------------------------------------------------
# Flex trade file discovery + parsing
# ---------------------------------------------------------------------------
def _flex_trades_path(run_date: str, *, runs_root: Path | None = None) -> Path:
    root = runs_root or RUNS
    return root / str(run_date) / "ibkr_flex" / "flex_trades.xml"


def latest_flex_trades_file(
    run_date: str,
    *,
    runs_root: Path | None = None,
) -> tuple[str | None, Path | None]:
    """Newest committed ``flex_trades.xml`` on or before ``run_date``.

    The trades report carries full account history, so the freshest file on or
    before the run date is sufficient; no need to stitch multiple days.
    """
    root = runs_root or RUNS
    if not root.is_dir():
        return None, None
    candidates: list[str] = []
    for child in root.iterdir():
        if not child.is_dir() or child.name > str(run_date):
            continue
        if _flex_trades_path(child.name, runs_root=root).is_file():
            candidates.append(child.name)
    if not candidates:
        return None, None
    d = sorted(candidates)[-1]
    return d, _flex_trades_path(d, runs_root=root)


def parse_flex_trade_last_dates(
    path: Path,
    *,
    on_or_before: str | None = None,
) -> dict[str, str]:
    """Map normalized symbol -> most recent ISO trade date from a Flex XML.

    Only equity/ETF legs are considered. ``on_or_before`` (YYYY-MM-DD) drops
    trades stamped after that calendar date so historical replays stay
    point-in-time.
    """
    if path is None or not Path(path).is_file():
        return {}
    cutoff = (on_or_before or "").replace("-", "")[:8] or None
    out: dict[str, str] = {}
    try:
        for _, elem in ET.iterparse(str(path), events=("end",)):
            if elem.tag not in _TRADE_TAGS:
                continue
            a = elem.attrib
            asset = str(a.get("assetCategory", "")).strip().upper()
            if asset not in _EQUITY_ASSET_CATEGORIES:
                elem.clear()
                continue
            sym = _norm_sym(a.get("symbol", ""))
            raw_date = str(a.get("tradeDate", "") or a.get("reportDate", "") or "").strip()
            elem.clear()
            if not sym or not raw_date:
                continue
            ymd = raw_date.split(";", 1)[0].replace("-", "")[:8]
            if not ymd.isdigit() or len(ymd) < 8:
                continue
            if cutoff is not None and ymd > cutoff:
                continue
            iso = _yyyymmdd_to_iso(ymd)
            if iso is None:
                continue
            prev = out.get(sym)
            if prev is None or iso > prev:
                out[sym] = iso
    except ET.ParseError:
        return out
    return out


def broker_last_rebalance_by_pair(
    pairs: Sequence[tuple[str, str]],
    *,
    run_date: str,
    runs_root: Path | None = None,
    flex_path: Path | None = None,
) -> dict[tuple[str, str], str]:
    """Last inverse-ETF trade date per ``(ETF, underlying)`` pair, from Flex.

    A pair's timing is keyed off its **inverse ETF** symbol, which is unique to
    that pair. Pairs whose ETF never traded (within the report) are omitted.
    """
    if not pairs:
        return {}
    path = flex_path
    if path is None:
        _, path = latest_flex_trades_file(run_date, runs_root=runs_root)
    if path is None:
        return {}
    sym_dates = parse_flex_trade_last_dates(path, on_or_before=run_date)
    if not sym_dates:
        return {}
    out: dict[tuple[str, str], str] = {}
    for etf, und in pairs:
        d = sym_dates.get(_norm_sym(etf))
        if d:
            out[(_norm_sym(etf), _norm_sym(und))] = d
    return out


# ---------------------------------------------------------------------------
# Staggered seeding (deterministic, no synchronized first-run burst)
# ---------------------------------------------------------------------------
def stagger_seed_offset(pair_key: str, interval_days: int) -> int:
    """Deterministic offset in ``[0, interval_days]`` from the pair key.

    Deterministic (not RNG) so a dry-run and the following real run on the same
    day agree, and so two machines seed the same spread. Uses a stable hash
    (Python's builtin ``hash`` is salted per process).
    """
    span = max(0, int(interval_days))
    if span == 0:
        return 0
    h = hashlib.sha1(pair_key.encode("utf-8")).hexdigest()
    return int(h, 16) % (span + 1)


def stagger_seed_date(run_date: str, interval_days: int, pair_key: str) -> str:
    """``run_date`` minus a hashed number of business days in ``[0, interval]``."""
    offset = stagger_seed_offset(pair_key, interval_days)
    d = np.datetime64(str(run_date), "D")
    if offset > 0:
        d = np.busday_offset(d, -offset, roll="backward")
    return str(np.datetime_as_string(d, unit="D"))


# ---------------------------------------------------------------------------
# Reconciliation
# ---------------------------------------------------------------------------
def reconcile_cadence_state(
    state: Mapping[str, str],
    *,
    pair_intervals: Mapping[tuple[str, str], int],
    run_date: str,
    runs_root: Path | None = None,
    flex_path: Path | None = None,
    enable_broker: bool = True,
) -> tuple[dict[str, str], dict[str, str]]:
    """Backfill machine-independent ``last_rebalance`` for every active pair.

    Returns ``(new_state, provenance)`` where provenance maps pair_key ->
    one of ``{"cache", "broker", "both", "stagger"}``. Existing unrelated keys
    in ``state`` are preserved. Never regresses a known date (takes the max of
    cache vs broker evidence).
    """
    new_state: dict[str, str] = {str(k): str(v) for k, v in (state or {}).items()}
    provenance: dict[str, str] = {}

    pairs = list(pair_intervals.keys())
    broker: dict[tuple[str, str], str] = {}
    if enable_broker and pairs:
        broker = broker_last_rebalance_by_pair(
            pairs, run_date=run_date, runs_root=runs_root, flex_path=flex_path
        )

    for (etf, und), interval in pair_intervals.items():
        pk = _pair_key(etf, und)
        cache_date = new_state.get(pk)
        broker_date = broker.get((_norm_sym(etf), _norm_sym(und)))
        candidates = [d for d in (cache_date, broker_date) if d]
        if candidates:
            new_state[pk] = max(candidates)
            if cache_date and broker_date:
                provenance[pk] = "both"
            elif broker_date:
                provenance[pk] = "broker"
            else:
                provenance[pk] = "cache"
        else:
            new_state[pk] = stagger_seed_date(run_date, int(interval), pk)
            provenance[pk] = "stagger"
    return new_state, provenance


# ---------------------------------------------------------------------------
# CLI: inspect / seed cadence state on a machine without a full rebalance run
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    from scripts.bucket4_cadence_gate import (
        load_cadence_state,
        save_cadence_state,
    )

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-date", required=True)
    ap.add_argument("--runs-root", default=str(RUNS))
    ap.add_argument(
        "--pairs",
        default="",
        help="Comma list of ETF:UND pairs (e.g. MSTX:MSTR,MSTU:MSTR). "
        "If omitted, reconstructs only pairs already present in the state file.",
    )
    ap.add_argument("--interval", type=int, default=14, help="fallback interval for stagger seeding")
    ap.add_argument("--state-json", default="data/b4_cadence_state.json")
    ap.add_argument("--persist", action="store_true", help="write the reconciled state back to --state-json")
    ap.add_argument("--no-broker", action="store_true", help="skip Flex reconstruction (stagger-seed only)")
    args = ap.parse_args(argv)

    runs_root = Path(args.runs_root).resolve()
    state_path = Path(args.state_json)
    state = load_cadence_state(state_path)

    pair_intervals: dict[tuple[str, str], int] = {}
    if args.pairs.strip():
        for tok in args.pairs.split(","):
            tok = tok.strip()
            if not tok or ":" not in tok:
                continue
            etf, und = tok.split(":", 1)
            pair_intervals[(_norm_sym(etf), _norm_sym(und))] = int(args.interval)
    else:
        for pk in state:
            if "|" in pk:
                etf, und = pk.split("|", 1)
                pair_intervals[(_norm_sym(etf), _norm_sym(und))] = int(args.interval)

    if not pair_intervals:
        print("[b4_cadence_reconstruct] no pairs to reconstruct (pass --pairs or seed state first)", file=sys.stderr)
        return 1

    src_date, src_path = latest_flex_trades_file(args.run_date, runs_root=runs_root)
    new_state, prov = reconcile_cadence_state(
        state,
        pair_intervals=pair_intervals,
        run_date=args.run_date,
        runs_root=runs_root,
        enable_broker=not args.no_broker,
    )
    print(
        f"[b4_cadence_reconstruct] run={args.run_date} "
        f"flex_source={src_date or 'none'} pairs={len(pair_intervals)}"
    )
    for (etf, und) in sorted(pair_intervals):
        pk = _pair_key(etf, und)
        print(f"  {etf}/{und}: last_rebalance={new_state[pk]} ({prov.get(pk)})")

    if args.persist:
        save_cadence_state(state_path, new_state)
        print(f"[b4_cadence_reconstruct] wrote {state_path}")
    else:
        print("[b4_cadence_reconstruct] dry (pass --persist to write)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
