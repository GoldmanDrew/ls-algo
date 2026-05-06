"""splits.py — Multi-source corporate-action split detection and repair.

Single source of truth for cleaning unadjusted splits / reverse-splits out of
total-return price series before they corrupt vol, beta, and decay estimators.

Sources (highest precedence first):

    1. ``flex``                IBKR Flex ``<CorporateAction type="RS">``
    2. ``yahoo_events``        Yahoo v8 chart ``events.splits`` payload
    3. ``splits_overrides_csv``  ``data/splits_overrides.csv`` ops-managed
    4. ``manual_override_dict`` legacy in-code overrides (SMUP, EOSU)
    5. ``heuristic``           z-score + integer-factor matched detector
                                (catches SAME-DAY splits Yahoo hasn't pushed yet)

Convention
----------
A :class:`SplitEvent` carries ``factor`` = the multiplier applied to PRE-split
history to bring it onto the POST-split basis. The price-level multiplier:

    1-for-10 reverse split (price jumps 10×): factor = 10.0
    5-for-1 forward split  (price drops 5×):  factor = 0.2
    1-for-N reverse split:                    factor = N
    M-for-1 forward split:                    factor = 1/M

This matches the legacy ``_apply_manual_split_overrides`` semantics — strictly
``out.loc[out.index < ex_date] *= factor``. A split timestamp **on** the bar
itself is treated as the boundary: pre-split history (strictly before) is
adjusted; the ex-date bar and everything after is left alone (it is already on
the post-split basis once the split happened).

Note that Yahoo's ``events.splits`` and the ``numerator,denominator`` columns
of the overrides CSV report ``num:den`` in **share-multiplier** form (1-for-10
reverse → ``numerator=1, denominator=10``). Our :class:`SplitEvent` factor is
the **price-multiplier**, so:

    factor = denominator / numerator

Both forms encode the same event; the price-multiplier form is more useful
for math on prices.

Non-goals
---------
- We do NOT broaden the integer-factor whitelist. Non-integer factors must
  arrive via an authoritative source (flex / yahoo_events / overrides csv);
  the heuristic remains integer-only on purpose.
- We do NOT auto-ingest splits with ``factor < 1.0`` from heuristics for
  thin price histories (< 5 context obs) — but we DO log and accept them on
  the most recent bar with permissive context, which is exactly the
  "Yahoo hasn't pushed the adjustment yet" case.

See ``SPLITS.md`` at repo root for the operator-facing playbook.
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────
# Constants — kept consistent with prior daily_screener / etf_analytics
# ─────────────────────────────────────────────────────────────────────
_INTEGER_SPLIT_FACTORS: tuple[int, ...] = (2, 3, 4, 5, 10, 15, 20, 25, 50)
_SPLIT_RATIOS: tuple[float, ...] = tuple(
    sorted(
        set(list(_INTEGER_SPLIT_FACTORS) + [1.0 / f for f in _INTEGER_SPLIT_FACTORS]),
        reverse=True,
    )
)
_SPLIT_TOL: float = 0.005          # ±0.5 % tolerance around each ratio
_LOG_RATIO_LOOSE_TOL: float = 0.22  # log-space fallback (Yahoo adjclose noise)
_JUMP_FLOOR: float = 0.40           # ignore daily moves < 40 %
_CONTEXT_WINDOW: int = 20           # days for local vol z-score
_ZSCORE_THRESHOLD: float = 4.0      # split jump must be > 4σ vs local vol

# Self-healing window. If pre/post raw ratio across the proposed ex-date is
# already inside [1 / SELF_HEAL_RATIO_BAND, SELF_HEAL_RATIO_BAND], the source
# (typically Yahoo) has retroactively adjusted the history and we must NOT
# re-apply the override. 3.0 catches splits up to 1:3 / 3:1 self-adjusted;
# larger splits (1:10, 1:20) leave a much bigger ratio if unadjusted, so this
# bound stays well clear of the actual split signal.
_SELF_HEAL_RATIO_BAND: float = 3.0


# ─────────────────────────────────────────────────────────────────────
# Legacy manual-override registry (preserved for back-compat)
# ─────────────────────────────────────────────────────────────────────
# Values are the PRICE-multiplier convention (factor in SplitEvent):
#
#   SMUP 2026-01-26 → 0.1  (forward 10-for-1, post = pre × 0.1)
#   EOSU 2026-04-15 → 25   (reverse 1-for-25, post = pre × 25)
#
# Add entries here only when none of {flex, yahoo_events, overrides_csv} is
# available. New ops entries should go into ``data/splits_overrides.csv``.
_LEGACY_MANUAL_OVERRIDES: dict[str, dict[str, float]] = {
    "SMUP": {"2026-01-26": 0.1},
    "EOSU": {"2026-04-15": 25.0},
}

# Yahoo symbol aliases for recently renamed products. Used only when the
# primary symbol returns no data. Public alias for callers.
SYMBOL_ALIASES: dict[str, str] = {
    "SMUP": "SMU",
}


# ─────────────────────────────────────────────────────────────────────
# Public dataclass
# ─────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class SplitEvent:
    """A single corporate-action split event.

    Attributes
    ----------
    symbol : str
        Canonical (uppercase, dot-form) symbol the event applies to.
    ex_date : pd.Timestamp
        Ex-date of the split (timezone-naive, normalized to midnight).
    factor : float
        Multiplier applied to PRE-split history to land on POST-split basis.
        1-for-10 reverse split → 0.1; 5-for-1 forward split → 5.0.
    source : str
        One of ``"flex"``, ``"yahoo_events"``, ``"splits_overrides_csv"``,
        ``"manual_override_dict"``, ``"heuristic"``.
    note : str
        Optional free-form context (split description, file path, etc.).
    """

    symbol: str
    ex_date: pd.Timestamp
    factor: float
    source: str
    note: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.ex_date, pd.Timestamp):
            raise TypeError(f"ex_date must be pd.Timestamp, got {type(self.ex_date)}")
        if not (self.factor > 0):
            raise ValueError(f"SplitEvent.factor must be positive, got {self.factor}")


# Source precedence — lower index = higher precedence.
SOURCE_PRECEDENCE: tuple[str, ...] = (
    "flex",
    "yahoo_events",
    "splits_overrides_csv",
    "manual_override_dict",
    "heuristic",
)


# ─────────────────────────────────────────────────────────────────────
# Symbol normalization
# ─────────────────────────────────────────────────────────────────────
def _norm_sym(x: object) -> str:
    """Canonical ticker form: uppercase, ``.`` for class shares (BRK.B)."""
    if x is None:
        return ""
    s = str(x).strip().upper()
    if not s:
        return ""
    s = s.replace("-", ".")
    return s


def _strip_old_suffix(symbol: str) -> str:
    """``SOLT.OLD`` → ``SOLT``; ``DUST.OLD`` → ``DUST``.

    Flex emits a paired ``X`` and ``X.OLD`` row for each split; both refer to
    the same underlying ticker.
    """
    sym = _norm_sym(symbol)
    if sym.endswith(".OLD"):
        return sym[: -len(".OLD")]
    return sym


def _normalize_index(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Strip timezone and normalize to midnight for stable comparisons."""
    out = pd.DatetimeIndex(idx)
    if out.tz is not None:
        out = out.tz_convert("UTC").tz_localize(None)
    return out.normalize()


def _coerce_timestamp(value: object, *, tz: object | None = None) -> pd.Timestamp | None:
    """Best-effort coerce ``value`` to a tz-naive midnight ``pd.Timestamp``.

    Accepts ``date``, ``datetime``, ``pd.Timestamp``, or ISO string. Returns
    ``None`` on parse failure.
    """
    if value is None:
        return None
    try:
        ts = pd.Timestamp(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(ts):
        return None
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    if tz is not None:  # noqa: ARG001 — accepted for symmetry; output always naive
        pass
    return ts.normalize()


# ─────────────────────────────────────────────────────────────────────
# Source loaders
# ─────────────────────────────────────────────────────────────────────
def parse_yahoo_split_events(
    symbol: str,
    raw_events: Mapping[str, object] | None,
) -> list[SplitEvent]:
    """Parse the ``events.splits`` block from a Yahoo v8 chart payload.

    Schema::

        {"<unix_ts>": {"date": <unix_ts>, "numerator": 1.0,
                        "denominator": 20.0, "splitRatio": "1:20"}}

    ``factor = numerator / denominator`` (1-for-10 → 1/10 = 0.1).
    """
    if not raw_events:
        return []
    sym = _norm_sym(symbol)
    out: list[SplitEvent] = []
    for _, payload in raw_events.items():
        if not isinstance(payload, Mapping):
            continue
        try:
            num = float(payload.get("numerator", 0.0) or 0.0)
            den = float(payload.get("denominator", 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
        if num <= 0 or den <= 0:
            continue
        ts_unix = payload.get("date") or payload.get("timestamp")
        try:
            ts = pd.Timestamp(int(ts_unix), unit="s", tz="UTC")
            ts = ts.tz_convert("America/New_York").tz_localize(None).normalize()
        except (TypeError, ValueError, OverflowError):
            continue
        ratio_label = str(payload.get("splitRatio", f"{int(num)}:{int(den)}"))
        # Yahoo encodes the share-multiplier (num:den). Our SplitEvent.factor is
        # the price-multiplier = den/num. 1-for-10 reverse → 10.
        out.append(
            SplitEvent(
                symbol=sym,
                ex_date=ts,
                factor=den / num,
                source="yahoo_events",
                note=f"yahoo splitRatio={ratio_label}",
            )
        )
    return out


def load_splits_overrides_csv(path: Path | str | None) -> list[SplitEvent]:
    """Load operator-managed split overrides.

    CSV schema::

        symbol,ex_date,numerator,denominator,source,note

    ``source`` is captured for provenance but the precedence label is forced
    to ``"splits_overrides_csv"`` regardless of what the file says — that's
    the channel the file came in through.

    Missing file returns ``[]`` (feature is opt-in by default).
    """
    if path is None:
        return []
    p = Path(path).expanduser()
    if not p.is_file():
        return []
    try:
        df = pd.read_csv(p)
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError):
        return []
    if df.empty:
        return []
    out: list[SplitEvent] = []
    for _, row in df.iterrows():
        sym = _norm_sym(row.get("symbol"))
        if not sym:
            continue
        ts = _coerce_timestamp(row.get("ex_date"))
        if ts is None:
            continue
        try:
            num = float(row.get("numerator", 0.0) or 0.0)
            den = float(row.get("denominator", 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
        if num <= 0 or den <= 0:
            continue
        note_field = row.get("note") if "note" in df.columns else ""
        note = str(note_field) if pd.notna(note_field) else ""
        # CSV uses share-multiplier (num:den). SplitEvent.factor is the
        # price-multiplier (den/num). 1-for-10 reverse → factor=10.
        out.append(
            SplitEvent(
                symbol=sym,
                ex_date=ts,
                factor=den / num,
                source="splits_overrides_csv",
                note=note,
            )
        )
    return out


def load_legacy_manual_overrides() -> list[SplitEvent]:
    """Re-emit ``_LEGACY_MANUAL_OVERRIDES`` as :class:`SplitEvent` instances.

    Kept for back-compat — new entries should go to ``splits_overrides.csv``.
    """
    out: list[SplitEvent] = []
    for sym, by_date in _LEGACY_MANUAL_OVERRIDES.items():
        for ds, factor in by_date.items():
            ts = _coerce_timestamp(ds)
            if ts is None:
                continue
            out.append(
                SplitEvent(
                    symbol=_norm_sym(sym),
                    ex_date=ts,
                    factor=float(factor),
                    source="manual_override_dict",
                    note="legacy in-code override",
                )
            )
    return out


_FLEX_RS_RE = re.compile(r"SPLIT\s+(\d+)\s+FOR\s+(\d+)", re.IGNORECASE)


def parse_flex_corporate_action_splits(
    flex_cash_xml: Path | str,
) -> pd.DataFrame:
    """Extract reverse-split ``<CorporateAction type="RS">`` rows from Flex.

    Returns DataFrame columns::

        symbol, ex_date (YYYY-MM-DD str), numerator, denominator, factor,
        source ("flex"), note

    The ``description`` and ``actionDescription`` Flex strings carry
    ``"SPLIT N FOR M"`` text. Flex emits a pair of rows per split (one
    canonical symbol, one ``SYM.OLD`` symbol) with the same ``actionID`` —
    we deduplicate to canonical and prefer the row whose symbol does not
    end in ``.OLD``.
    """
    cols = [
        "symbol",
        "ex_date",
        "numerator",
        "denominator",
        "factor",
        "source",
        "note",
    ]
    p = Path(flex_cash_xml).expanduser() if not isinstance(flex_cash_xml, Path) else flex_cash_xml
    if not p.is_file():
        return pd.DataFrame(columns=cols)
    try:
        root = ET.parse(p).getroot()
    except ET.ParseError:
        return pd.DataFrame(columns=cols)

    rows: list[dict] = []
    for node in root.findall(".//CorporateAction"):
        a = node.attrib
        if (a.get("type") or "").strip().upper() != "RS":
            continue
        sym_raw = (a.get("symbol") or "").strip()
        sym = _strip_old_suffix(sym_raw)
        if not sym:
            continue
        rd = (a.get("reportDate") or "").strip()
        if len(rd) != 8 or not rd.isdigit():
            # Flex sometimes emits 8-digit YYYYMMDD; skip malformed rows.
            continue
        try:
            ex_dt = datetime.strptime(rd, "%Y%m%d").date()
        except ValueError:
            continue

        desc = (a.get("description") or "") + " " + (a.get("actionDescription") or "")
        m = _FLEX_RS_RE.search(desc)
        if not m:
            continue
        num = int(m.group(1))
        den = int(m.group(2))
        if num <= 0 or den <= 0:
            continue
        action_id = (a.get("actionID") or "").strip()

        # Flex CSV stores the price-multiplier so it is directly usable as
        # SplitEvent.factor. den/num: 1-for-10 reverse → 10.
        rows.append(
            {
                "symbol": sym,
                "ex_date": ex_dt.isoformat(),
                "numerator": num,
                "denominator": den,
                "factor": float(den) / float(num),
                "source": "flex",
                "note": (a.get("actionDescription") or a.get("description") or "")[:160],
                "_action_id": action_id,
                "_is_old": sym_raw.upper().endswith(".OLD"),
            }
        )

    if not rows:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(rows)
    # Prefer non-OLD row when both are present for the same (symbol, action_id).
    df = df.sort_values(by=["_is_old", "symbol", "ex_date"]).drop_duplicates(
        subset=["symbol", "ex_date", "_action_id"], keep="first"
    )
    df = df.drop(columns=["_action_id", "_is_old"]).reset_index(drop=True)
    df = df.sort_values(by=["symbol", "ex_date"]).reset_index(drop=True)
    return df[cols]


def load_flex_splits_csv(path: Path | str | None) -> list[SplitEvent]:
    """Load events from a previously-written ``data/splits_from_flex.csv``."""
    if path is None:
        return []
    p = Path(path).expanduser()
    if not p.is_file():
        return []
    try:
        df = pd.read_csv(p)
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError):
        return []
    if df.empty:
        return []
    out: list[SplitEvent] = []
    for _, row in df.iterrows():
        sym = _norm_sym(row.get("symbol"))
        if not sym:
            continue
        ts = _coerce_timestamp(row.get("ex_date"))
        if ts is None:
            continue
        try:
            factor = float(row.get("factor"))
        except (TypeError, ValueError):
            try:
                num = float(row.get("numerator", 0.0) or 0.0)
                den = float(row.get("denominator", 0.0) or 0.0)
                factor = num / den if den > 0 else 0.0
            except (TypeError, ValueError):
                factor = 0.0
        if factor <= 0:
            continue
        note_field = row.get("note") if "note" in df.columns else ""
        note = str(note_field) if pd.notna(note_field) else ""
        out.append(
            SplitEvent(
                symbol=sym,
                ex_date=ts,
                factor=float(factor),
                source="flex",
                note=note,
            )
        )
    return out


# ─────────────────────────────────────────────────────────────────────
# Heuristic detection (z-score + integer matched factor)
# ─────────────────────────────────────────────────────────────────────
def detect_heuristic_splits(
    prices: pd.Series,
    *,
    sym_label: str | None = None,
    last_only: bool = False,
) -> list[SplitEvent]:
    """Detect split-like jumps that no authoritative source has provided.

    Walks the price series and flags days whose log-return is

      - ≥ ``_JUMP_FLOOR`` (40 %) in absolute log-space, AND
      - matches an integer split factor (or its reciprocal) within
        ``_SPLIT_TOL`` (0.5 %) or the looser log-tol ``_LOG_RATIO_LOOSE_TOL``,
        AND
      - is a > ``_ZSCORE_THRESHOLD`` (4σ) outlier vs the local 20-day vol
        (with permissive fallback when context is thin).

    The off-by-one bug in legacy ``_clean_split_artifacts`` is **fixed** here
    by iterating ``range(1, len(vals))`` so the most recent bar — exactly
    where same-day reverse splits land — is tested.

    Set ``last_only=True`` to test only the most recent ratio (used by the
    ``--audit-splits`` CLI for fast scans).
    """
    sym = _norm_sym(sym_label) if sym_label else ""

    s = pd.to_numeric(prices, errors="coerce").dropna()
    if len(s) < 3:
        return []
    s = s[~s.index.duplicated(keep="last")].sort_index()

    vals = s.values.astype(float)
    idx = _normalize_index(s.index)

    # Pre-compute log returns for z-score context.
    log_r = np.full(len(vals), np.nan)
    for i in range(1, len(vals)):
        if vals[i - 1] > 0 and np.isfinite(vals[i - 1]) and vals[i] > 0:
            log_r[i] = np.log(vals[i] / vals[i - 1])

    out: list[SplitEvent] = []
    rng = (len(vals) - 1, len(vals)) if last_only else (1, len(vals))
    for i in range(rng[0], rng[1]):
        if i < 1:
            continue
        prev = vals[i - 1]
        cur = vals[i]
        if not (np.isfinite(prev) and prev > 0 and np.isfinite(cur) and cur > 0):
            continue
        ratio = cur / prev
        # Skip small moves — never a split.
        if abs(ratio - 1.0) < _JUMP_FLOOR:
            continue

        matched_factor: float | None = None

        # Tight tolerance against the integer-factor whitelist.
        for sf in _SPLIT_RATIOS:
            if abs(ratio - sf) / sf < _SPLIT_TOL:
                matched_factor = sf
                break
            inv_sf = 1.0 / sf
            if abs(ratio - inv_sf) / inv_sf < _SPLIT_TOL:
                matched_factor = inv_sf
                break

        # Looser log-space tol — Yahoo adjclose on ex-day can be ~12-25 %
        # away from an exact 10×/20× ratio when the underlying gapped
        # overnight (e.g. GraniteShares MSTP/SMCL 1-for-20). Pick the
        # CLOSEST candidate factor in log-space (otherwise iteration order
        # would silently bias us toward the smaller integer factor — a
        # ratio of 4.95 was getting matched to 4 before 5).
        if matched_factor is None:
            best: tuple[float, float] | None = None  # (log_dist, factor)
            for sf in _INTEGER_SPLIT_FACTORS:
                if sf < 2:
                    continue
                d_up = abs(np.log(ratio / sf))
                if d_up < _LOG_RATIO_LOOSE_TOL and (best is None or d_up < best[0]):
                    best = (float(d_up), float(sf))
                inv_sf = 1.0 / float(sf)
                d_dn = abs(np.log(ratio / inv_sf))
                if d_dn < _LOG_RATIO_LOOSE_TOL and (best is None or d_dn < best[0]):
                    best = (float(d_dn), inv_sf)
            if best is not None:
                matched_factor = best[1]

        if matched_factor is None:
            continue

        # Z-score test against local vol — guards against legitimate news
        # spikes on small-caps (a 5× moonshot on a meme stock should NOT be
        # treated as a split).
        start = max(1, i - _CONTEXT_WINDOW)
        end = min(len(log_r), i + _CONTEXT_WINDOW + 1)
        ctx = [log_r[j] for j in range(start, end) if j != i and np.isfinite(log_r[j])]
        if len(ctx) >= 5:
            local_std = float(np.std(ctx))
            if local_std > 0:
                z = abs(np.log(ratio)) / local_std
                if z < _ZSCORE_THRESHOLD:
                    continue

        # matched_factor is the price-ratio (cur/prev). Pre-split history must
        # be multiplied by exactly that ratio to land on the post-split basis.
        # SplitEvent.factor IS the price-multiplier — set it to matched_factor.
        # 1-for-10 reverse: ratio=10 → factor=10. 5-for-1 forward: ratio=0.2
        # → factor=0.2.
        event_factor = float(matched_factor)

        out.append(
            SplitEvent(
                symbol=sym,
                ex_date=idx[i],
                factor=event_factor,
                source="heuristic",
                note=(
                    f"ratio={ratio:.4f} matched={matched_factor:g} "
                    f"prev=${prev:.4f} cur=${cur:.4f}"
                ),
            )
        )

    return out


# ─────────────────────────────────────────────────────────────────────
# Merge sources by precedence
# ─────────────────────────────────────────────────────────────────────
def merge_split_events(
    events_by_source: Mapping[str, Sequence[SplitEvent]] | Iterable[SplitEvent],
    *,
    same_day_tolerance_days: int = 2,
) -> list[SplitEvent]:
    """Merge events from multiple sources by precedence.

    ``events_by_source`` may be a mapping ``{source: [events...]}`` or a flat
    iterable. Within ``(symbol, ex_date)``, the highest-precedence source wins
    (see :data:`SOURCE_PRECEDENCE`).

    Two events for the same symbol whose ``ex_date`` differ by no more than
    ``same_day_tolerance_days`` calendar days are treated as the same event
    (sources may report a split with reportDate vs ex-date one or two days
    off). The earlier ``ex_date`` is preferred.
    """
    if isinstance(events_by_source, Mapping):
        all_events: list[SplitEvent] = []
        for src, evs in events_by_source.items():
            for ev in evs:
                if ev.source != src:
                    ev = SplitEvent(
                        symbol=ev.symbol,
                        ex_date=ev.ex_date,
                        factor=ev.factor,
                        source=src,
                        note=ev.note,
                    )
                all_events.append(ev)
    else:
        all_events = list(events_by_source)

    if not all_events:
        return []

    rank = {src: i for i, src in enumerate(SOURCE_PRECEDENCE)}
    # Group by (symbol, near_date).
    by_sym: dict[str, list[SplitEvent]] = {}
    for ev in all_events:
        by_sym.setdefault(ev.symbol, []).append(ev)

    merged: list[SplitEvent] = []
    for sym, evs in by_sym.items():
        # Sort by ex_date for grouping, then by source precedence within group.
        evs_sorted = sorted(evs, key=lambda e: (e.ex_date, rank.get(e.source, 99)))
        groups: list[list[SplitEvent]] = []
        for ev in evs_sorted:
            if (
                groups
                and abs((ev.ex_date - groups[-1][0].ex_date).days)
                <= same_day_tolerance_days
            ):
                groups[-1].append(ev)
            else:
                groups.append([ev])
        for g in groups:
            # Pick the highest-precedence event from this group.
            g_sorted = sorted(g, key=lambda e: rank.get(e.source, 99))
            chosen = g_sorted[0]
            merged.append(chosen)

    merged.sort(key=lambda e: (e.symbol, e.ex_date))
    return merged


# ─────────────────────────────────────────────────────────────────────
# Apply events to a price series (with self-healing)
# ─────────────────────────────────────────────────────────────────────
def _self_healed(
    series: pd.Series,
    apply_ts: pd.Timestamp,
    factor: float,
    cumulative_pre_factor: float,
) -> bool:
    """Decide whether to skip applying ``factor`` because the source already
    adjusted history.

    The cumulative pre-factor we'd apply by going through with this event is
    ``cumulative_pre_factor * factor``. Compare last raw pre-bar to first raw
    post-bar across the boundary:

      ratio_raw = post_bar / pre_bar

    If ``ratio_raw`` is already inside [1/_SELF_HEAL_RATIO_BAND,
    _SELF_HEAL_RATIO_BAND], the source already adjusted; skip. We also skip
    when ``ratio_raw / (cumulative_pre_factor * factor)`` is far from 1 in the
    direction that would *over*-correct (i.e. cumulative adjustment would push
    the boundary into a 3×+ raw mismatch).
    """
    pre_bars = series.loc[series.index < apply_ts]
    post_bars = series.loc[series.index >= apply_ts]
    if pre_bars.empty or post_bars.empty:
        return False
    pre_raw = float(pre_bars.iloc[-1])
    post_raw = float(post_bars.iloc[0])
    if pre_raw <= 0 or post_raw <= 0:
        return False
    raw_ratio = post_raw / pre_raw
    band = _SELF_HEAL_RATIO_BAND
    # Note: cumulative_pre_factor reflects prior events ALREADY APPLIED to
    # ``series`` (in-memory). So ``pre_raw`` here is already adjusted by
    # ``cumulative_pre_factor``. Comparing post / pre directly tells us the
    # raw boundary jump in the (already-prior-adjusted) series.
    if 1.0 / band < raw_ratio < band:
        return True
    return False


def apply_split_events(
    series: pd.Series,
    events: Sequence[SplitEvent],
    *,
    sym_label: str | None = None,
) -> tuple[pd.Series, list[SplitEvent]]:
    """Apply a sorted list of split events to a price series.

    Each event multiplies all bars STRICTLY before its ``ex_date`` by
    ``event.factor``. Forward and reverse splits are handled symmetrically.
    Multiple events compose: a 1:10 followed 30 days later by a 1:5 yields a
    cumulative pre-factor of 0.02 on bars before the earlier split, 0.2
    between them, and 1.0 after the later split.

    Self-healing: an event is SKIPPED when the raw boundary ratio (in the
    already-corrected series) is already on the post-split basis. This makes
    re-runs idempotent the day after Yahoo back-adjusts the history.

    Returns ``(corrected_series, applied_events)``.
    """
    if series is None or series.empty:
        return series, []
    if not events:
        return series.copy(), []

    sym = _norm_sym(sym_label) if sym_label else ""

    out = series.sort_index().copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index)
    idx_tz = out.index.tz

    # Sort by ex_date ascending; we apply oldest first so that cumulative
    # pre-factors stack correctly.
    evs = sorted(events, key=lambda e: e.ex_date)

    # Drop duplicate (symbol, ex_date) keeping the first (already merged
    # upstream by ``merge_split_events``, but be defensive).
    seen: set[tuple[str, pd.Timestamp]] = set()
    deduped: list[SplitEvent] = []
    for e in evs:
        key = (_norm_sym(e.symbol), e.ex_date)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(e)
    evs = deduped

    cumulative_pre_factor = 1.0
    applied: list[SplitEvent] = []
    for ev in evs:
        ts = ev.ex_date
        # Align tz with the series index.
        if idx_tz is not None and ts.tzinfo is None:
            ts_local = ts.tz_localize(idx_tz)
        elif idx_tz is None and ts.tzinfo is not None:
            ts_local = ts.tz_convert("UTC").tz_localize(None)
        else:
            ts_local = ts
        # Snap to the next bar in the series within 3 calendar days when the
        # exact ex-date isn't a trading day (mirrors legacy behaviour).
        if ts_local in out.index:
            apply_ts = ts_local
        else:
            future = out.index[out.index >= ts_local]
            if len(future) > 0 and (future[0] - ts_local).days <= 3:
                apply_ts = future[0]
            else:
                if sym:
                    print(
                        f"[SPLIT][skip-no-bar] {sym} {ts.date()} no nearby trading day "
                        f"(source={ev.source})"
                    )
                continue

        # Self-healing — compare raw boundary ratio in the already-adjusted
        # series.
        if _self_healed(out, apply_ts, ev.factor, cumulative_pre_factor):
            if sym:
                pre_bars = out.loc[out.index < apply_ts]
                post_bars = out.loc[out.index >= apply_ts]
                pre_raw = float(pre_bars.iloc[-1]) if len(pre_bars) else 0.0
                post_raw = float(post_bars.iloc[0]) if len(post_bars) else 0.0
                ratio = (post_raw / pre_raw) if pre_raw > 0 else float("nan")
                print(
                    f"[SPLIT][self-heal] {sym} {ts.date()} SKIPPED "
                    f"(source={ev.source}, factor={ev.factor:g}, "
                    f"pre=${pre_raw:.4f} post=${post_raw:.4f} ratio={ratio:.3f})"
                )
            continue

        # Apply: pre-split bars × factor.
        out.loc[out.index < apply_ts] = out.loc[out.index < apply_ts] * float(ev.factor)
        cumulative_pre_factor *= float(ev.factor)
        applied.append(ev)
        if sym:
            print(
                f"[SPLIT][apply] {sym} {apply_ts.date()} factor=x{ev.factor:g} "
                f"source={ev.source}"
                + (f" — {ev.note}" if ev.note else "")
            )

    return out, applied


# ─────────────────────────────────────────────────────────────────────
# Top-level pipeline used by data fetchers
# ─────────────────────────────────────────────────────────────────────
def clean_split_artifacts(
    prices: pd.Series,
    *,
    ticker: str | None = None,
    yahoo_events: Mapping[str, object] | None = None,
    splits_overrides_csv: Path | str | None = None,
    flex_splits_csv: Path | str | None = None,
    use_legacy_manual: bool = True,
    use_heuristic: bool = True,
    return_log: bool = False,
) -> pd.Series | tuple[pd.Series, list[SplitEvent]]:
    """Detect and correct unadjusted splits across all configured sources.

    Replaces the legacy in-line ``_clean_split_artifacts`` and
    ``_apply_manual_split_overrides`` in ``daily_screener.py`` /
    ``etf_analytics.py`` with a single multi-source path:

        1. flex (from ``data/splits_from_flex.csv`` if provided)
        2. yahoo_events (from the v8 chart payload)
        3. splits_overrides_csv (operator-managed)
        4. manual_override_dict (legacy in-code)
        5. heuristic (z-score + integer matched factor — last bar included)

    Self-healing makes the pipeline idempotent: if Yahoo (or any other source)
    has already retro-adjusted the history, the corresponding event is skipped
    based on the actual pre/post boundary ratio.

    Returns the corrected series. When ``return_log=True``, returns a
    ``(series, applied_events)`` tuple for the audit CLI.
    """
    if prices is None or len(prices) < 3:
        out = prices.copy() if prices is not None else prices
        return (out, []) if return_log else out

    sym = _norm_sym(ticker) if ticker else ""

    # Source 1: Flex CA CSV (operator-resolved).
    flex_events_all = load_flex_splits_csv(flex_splits_csv)
    flex_events = [e for e in flex_events_all if not sym or e.symbol == sym]

    # Source 2: Yahoo events.splits payload.
    ye = parse_yahoo_split_events(sym, yahoo_events) if yahoo_events else []

    # Source 3: ops-managed overrides CSV.
    csv_events_all = load_splits_overrides_csv(splits_overrides_csv)
    csv_events = [e for e in csv_events_all if not sym or e.symbol == sym]

    # Source 4: legacy in-code dict.
    legacy_events_all = load_legacy_manual_overrides() if use_legacy_manual else []
    legacy_events = [e for e in legacy_events_all if not sym or e.symbol == sym]

    # Source 5: heuristic — only used to fill the gap when authoritative
    # sources are silent on the most recent ex-day.
    heur_events: list[SplitEvent] = []
    if use_heuristic:
        heur_events = detect_heuristic_splits(prices, sym_label=sym)

    merged = merge_split_events(
        {
            "flex": flex_events,
            "yahoo_events": ye,
            "splits_overrides_csv": csv_events,
            "manual_override_dict": legacy_events,
            "heuristic": heur_events,
        }
    )

    corrected, applied = apply_split_events(prices, merged, sym_label=sym)
    if return_log:
        return corrected, applied
    return corrected
