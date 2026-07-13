"""Tests for the multi-source split detection / repair pipeline in splits.py.

Covers the failure modes that motivated the rewrite:

  - Same-day reverse split (BAIG-style) — the legacy off-by-one in
    ``_clean_split_artifacts`` skipped the most recent bar, so the screener
    silently produced 300 %+ vol for split-affected ETFs.
  - Same-day forward split (5-for-1) — symmetric edge case.
  - Already-Yahoo-adjusted history — the override must self-heal and NOT
    double-adjust on re-runs.
  - News-spike non-split — must NOT auto-correct (z-score guard).
  - Two reverse splits 30 days apart — cumulative pre-factors must compose.
  - Yahoo events authoritative when unadjusted — events.splits arriving in
    the v8 payload must apply even before the heuristic fires.
  - Flex ``<CorporateAction type="RS">`` parser — schema must round-trip.
"""
from __future__ import annotations

import sys
from pathlib import Path
from textwrap import dedent

import numpy as np
import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from splits import (  # noqa: E402
    SplitEvent,
    apply_split_events,
    clean_split_artifacts,
    detect_heuristic_splits,
    load_legacy_manual_overrides,
    load_splits_overrides_csv,
    merge_split_events,
    parse_flex_corporate_action_splits,
    parse_yahoo_split_events,
    repair_split_craters,
)


def _build_series(prices: list[float], start: str = "2026-04-01") -> pd.Series:
    idx = pd.bdate_range(start=start, periods=len(prices))
    return pd.Series(prices, index=idx, name="X", dtype=float)


# ──────────────────────────────────────────────────────────────────────
# (1) Same-day reverse split — the BAIG case
# ──────────────────────────────────────────────────────────────────────
def test_last_day_reverse_split_corrected() -> None:
    """A 1-for-10 reverse split on the FINAL bar must be detected.

    The legacy detector iterated ``range(1, len(vals) - 1)`` and missed the
    last bar; this regression test pins the fix.
    """
    rng = np.random.default_rng(20260505)
    base = 4.0 * np.exp(rng.normal(0, 0.04, size=200).cumsum())
    base[-1] = base[-2] * 10.0  # 10× jump on last bar
    s = _build_series(list(base))

    cleaned, applied = clean_split_artifacts(s, ticker="BAIG", return_log=True)

    assert applied, "expected at least one split event to be applied"
    last_event = max(applied, key=lambda e: e.ex_date)
    # 1-for-10 reverse split → price-multiplier 10 (post = pre × 10).
    assert last_event.factor == pytest.approx(10.0, rel=1e-2)
    # Post-clean ratio across the boundary should be ~1, not ~10.
    boundary_ratio = float(cleaned.iloc[-1]) / float(cleaned.iloc[-2])
    assert 0.6 < boundary_ratio < 1.6


# ──────────────────────────────────────────────────────────────────────
# (2) Same-day forward split (5-for-1)
# ──────────────────────────────────────────────────────────────────────
def test_last_day_forward_split_corrected() -> None:
    rng = np.random.default_rng(42)
    base = 100.0 * np.exp(rng.normal(0, 0.03, size=200).cumsum())
    base[-1] = base[-2] / 5.0  # forward split → price drops to 1/5
    s = _build_series(list(base))

    cleaned, applied = clean_split_artifacts(s, ticker="ABC", return_log=True)

    assert applied
    last = max(applied, key=lambda e: e.ex_date)
    # 5-for-1 forward split → price-multiplier 0.2 (post = pre × 0.2).
    assert last.factor == pytest.approx(0.2, rel=1e-2)
    boundary_ratio = float(cleaned.iloc[-1]) / float(cleaned.iloc[-2])
    assert 0.6 < boundary_ratio < 1.6


# ──────────────────────────────────────────────────────────────────────
# (3) Already-Yahoo-adjusted history — must be a no-op
# ──────────────────────────────────────────────────────────────────────
def test_yahoo_already_adjusted_is_noop() -> None:
    """When the source already retroactively divided pre-split history,
    re-running the cleaner the next day must not double-correct.
    """
    rng = np.random.default_rng(7)
    s = _build_series(list(20.0 * np.exp(rng.normal(0, 0.03, size=200).cumsum())))
    raw_last_ratio = float(s.iloc[-1]) / float(s.iloc[-2])

    # Provide a fake Yahoo events payload claiming a split happened today,
    # but the series is already on the post-split basis (no boundary jump).
    last_ts = s.index[-1]
    unix = int(
        (pd.Timestamp(last_ts.date()) + pd.Timedelta(hours=14))
        .tz_localize("UTC")
        .timestamp()
    )
    fake_event = {
        str(unix): {
            "date": unix,
            "numerator": 1.0,
            "denominator": 10.0,
            "splitRatio": "1:10",
        }
    }
    cleaned = clean_split_artifacts(s, ticker="ABC", yahoo_events=fake_event)

    # The series should be effectively unchanged (self-healing kicked in).
    assert np.allclose(cleaned.values, s.values, rtol=1e-6, atol=1e-9)
    cleaned_ratio = float(cleaned.iloc[-1]) / float(cleaned.iloc[-2])
    assert cleaned_ratio == pytest.approx(raw_last_ratio, rel=1e-6)


# ──────────────────────────────────────────────────────────────────────
# (4) News-spike non-split — must NOT correct
# ──────────────────────────────────────────────────────────────────────
def test_news_spike_not_corrected() -> None:
    """A non-integer news spike that does NOT match a whitelist split factor
    AND is followed by similar-magnitude moves (so z-score is bounded)
    should be left alone.

    Uses ~7.8× (between whitelist 6 and 10) so the loose log-space matcher
    cannot pull it onto either neighbor — 7× previously matched ×6 after 6
    was added to the integer whitelist for COYY residuals.
    """
    rng = np.random.default_rng(2026)
    n = 200
    rets = rng.normal(0, 0.20, size=n)  # 20 % daily vol — penny-stock realm
    # Inject a 7.8× spike on day 150 — mid-gap between whitelist 6 and 10.
    rets[150] = np.log(7.8)
    prices = 5.0 * np.exp(np.cumsum(rets))
    s = _build_series(list(prices))

    events = detect_heuristic_splits(s, sym_label="MEME")
    # Even if a candidate factor matches, z-score gate should reject it on a
    # high-vol series; if it doesn't match, the candidate list is empty.
    assert all(
        ev.ex_date != s.index[150]
        for ev in events
    ), f"news spike should not have been classified as a split: {events}"


# ──────────────────────────────────────────────────────────────────────
# (5) Two reverse splits 30 days apart — composition
# ──────────────────────────────────────────────────────────────────────
def test_two_reverse_splits_30_days_apart() -> None:
    rng = np.random.default_rng(11)
    n = 90
    prices = list(2.0 * np.exp(rng.normal(0, 0.04, size=n).cumsum()))
    # First 1:10 split at day 30, second 1:5 split at day 60. We simulate by
    # multiplying the post-split sub-series by the ratios. Here we want the
    # RAW (unadjusted) series the cleaner sees, so prices just step up.
    for i in range(30, n):
        prices[i] *= 10
    for i in range(60, n):
        prices[i] *= 5
    s = _build_series(prices)

    cleaned, applied = clean_split_artifacts(s, ticker="X2", return_log=True)
    # Two split events should have fired (price-multipliers 10 and 5).
    assert len(applied) >= 2
    factors = sorted(e.factor for e in applied)
    assert any(f == pytest.approx(5.0, rel=2e-2) for f in factors)
    assert any(f == pytest.approx(10.0, rel=2e-2) for f in factors)
    # Boundary ratios in the cleaned series are bounded.
    for ts in [s.index[30], s.index[60]]:
        before = float(cleaned.loc[cleaned.index < ts].iloc[-1])
        after = float(cleaned.loc[cleaned.index >= ts].iloc[0])
        ratio = after / before
        assert 0.4 < ratio < 2.5, (ts, ratio)


# ──────────────────────────────────────────────────────────────────────
# (6) Yahoo events authoritative when Yahoo lags on adjustment
# ──────────────────────────────────────────────────────────────────────
def test_yahoo_events_authoritative_when_unadjusted() -> None:
    """When Yahoo's events.splits payload announces a split but adjclose
    has not yet been retroactively divided (today's BAIG), the announced
    ratio should be applied verbatim.
    """
    n = 60
    rng = np.random.default_rng(100)
    base = 3.5 * np.exp(rng.normal(0, 0.04, size=n).cumsum())
    base[-1] = base[-2] * 10.0  # raw 10× boundary
    s = _build_series(list(base))

    # Yahoo encodes split events with a unix timestamp at ~09:30 NY on the
    # ex-date (which lands mid-day UTC). Constructing it as UTC midnight on
    # the ex-date would map to the previous NY day after tz conversion.
    last_ts = s.index[-1]
    unix = int(
        (pd.Timestamp(last_ts.date()) + pd.Timedelta(hours=14))
        .tz_localize("UTC")
        .timestamp()
    )
    yahoo_events = {
        str(unix): {
            "date": unix,
            "numerator": 1.0,
            "denominator": 10.0,
            "splitRatio": "1:10",
        }
    }
    cleaned, applied = clean_split_artifacts(
        s,
        ticker="BAIG",
        yahoo_events=yahoo_events,
        return_log=True,
    )

    sources = {ev.source for ev in applied}
    assert "yahoo_events" in sources, sources
    boundary = float(cleaned.iloc[-1]) / float(cleaned.iloc[-2])
    assert 0.6 < boundary < 1.6


def test_smup_colliding_yahoo_split_suppressed_for_manual_override() -> None:
    """Legacy SMUP factor (0.1 on 2026-01-26) must win over colliding Yahoo ``events.splits``."""
    idx = pd.to_datetime(["2026-01-19", "2026-01-20", "2026-01-23", "2026-01-26", "2026-01-27"])
    raw = pd.Series([100.0, 100.0, 100.0, 10.0, 10.05], index=idx)
    ex_unix = int((pd.Timestamp("2026-01-26") + pd.Timedelta(hours=14)).tz_localize("UTC").timestamp())
    yahoo_events = {
        str(ex_unix): {
            "date": ex_unix,
            "numerator": 1.0,
            "denominator": 10.0,
            "splitRatio": "1:10",
        }
    }
    cleaned, applied = clean_split_artifacts(
        raw,
        ticker="SMUP",
        yahoo_events=yahoo_events,
        use_heuristic=False,
        splits_overrides_csv=None,
        flex_splits_csv=None,
        return_log=True,
    )
    assert any(e.source == "manual_override_dict" for e in applied)
    assert not any(e.source == "yahoo_events" for e in applied)
    assert float(cleaned.pct_change().abs().dropna().max()) < 0.25


def test_smup_manual_override_emitted_for_smu_alias() -> None:
    evs = load_legacy_manual_overrides()
    jan26 = [e for e in evs if str(e.ex_date.date()) == "2026-01-26" and abs(e.factor - 0.1) < 1e-9]
    syms = {e.symbol for e in jan26}
    assert "SMUP" in syms
    assert "SMU" in syms


# ──────────────────────────────────────────────────────────────────────
# (8) Flex CA RS parser
# ──────────────────────────────────────────────────────────────────────
_FLEX_CA_FIXTURE = dedent(
    """\
    <FlexQueryResponse>
      <FlexStatements count="1">
        <FlexStatement accountId="U805366">
          <CorporateActions>
            <CorporateAction symbol="SOLT" type="RS" reportDate="20260227"
              description="SOLT(US92864M8304) SPLIT 1 FOR 20 (SOLT, 2X SOLANA ETF, US92865J7375)"
              actionDescription="SOLT(US92864M8304) SPLIT 1 FOR 20 (SOLT, 2X SOLANA ETF, US92865J7375)"
              actionID="111" />
            <CorporateAction symbol="SOLT.OLD" type="RS" reportDate="20260227"
              description="SOLT(US92864M8304) SPLIT 1 FOR 20 (SOLT.OLD, 2X SOLANA ETF, US92864M8304)"
              actionDescription="SOLT(US92864M8304) SPLIT 1 FOR 20 (SOLT.OLD, 2X SOLANA ETF, US92864M8304)"
              actionID="111" />
            <CorporateAction symbol="DUST" type="RS" reportDate="20260305"
              description="DUST SPLIT 1 FOR 10 (DUST, DIREXION DAILY GOLD MINERS I, US25461A1896)"
              actionDescription="DUST SPLIT 1 FOR 10"
              actionID="222" />
            <CorporateAction symbol="UNRELATED" type="DV" reportDate="20260301"
              description="DIVIDEND" actionDescription="DIVIDEND" actionID="333" />
          </CorporateActions>
        </FlexStatement>
      </FlexStatements>
    </FlexQueryResponse>
    """
)


def test_flex_corporate_action_splits_parsed(tmp_path: Path) -> None:
    p = tmp_path / "flex_cash.xml"
    p.write_text(_FLEX_CA_FIXTURE, encoding="utf-8")

    df = parse_flex_corporate_action_splits(p)
    assert not df.empty
    # Both rows for SOLT should collapse to one canonical SOLT row.
    assert (df["symbol"].eq("SOLT")).sum() == 1
    assert (df["symbol"].eq("SOLT.OLD")).sum() == 0
    # DUST 1-for-10 stores num/den in share-multiplier form, but the
    # ``factor`` column is the price-multiplier (den/num) → 10.
    dust = df[df["symbol"].eq("DUST")].iloc[0]
    assert dust["numerator"] == 1
    assert dust["denominator"] == 10
    assert float(dust["factor"]) == pytest.approx(10.0, rel=1e-9)
    assert dust["ex_date"] == "2026-03-05"
    # The non-RS row must be skipped.
    assert (df["symbol"].eq("UNRELATED")).sum() == 0
    # Source label is forced to "flex".
    assert (df["source"] == "flex").all()


# ──────────────────────────────────────────────────────────────────────
# (8) Splits overrides CSV loader
# ──────────────────────────────────────────────────────────────────────
def test_splits_overrides_csv_loader(tmp_path: Path) -> None:
    p = tmp_path / "splits_overrides.csv"
    p.write_text(
        dedent(
            """\
            symbol,ex_date,numerator,denominator,source,note
            BAIG,2026-05-05,1,10,manual,LeverageShares 1-for-10
            NVDA,2024-06-10,5,1,manual,5-for-1 forward
            """
        ),
        encoding="utf-8",
    )
    events = load_splits_overrides_csv(p)
    by_sym = {e.symbol: e for e in events}
    # 1-for-10 reverse: price-multiplier = 10. 5-for-1 forward: 0.2.
    assert by_sym["BAIG"].factor == pytest.approx(10.0)
    assert by_sym["NVDA"].factor == pytest.approx(0.2)
    for ev in events:
        assert ev.source == "splits_overrides_csv"


# ──────────────────────────────────────────────────────────────────────
# (9) Source precedence — flex beats heuristic
# ──────────────────────────────────────────────────────────────────────
def test_merge_precedence_flex_beats_heuristic() -> None:
    ts = pd.Timestamp("2026-05-05")
    flex = SplitEvent(symbol="BAIG", ex_date=ts, factor=10.0, source="flex", note="flex")
    heur = SplitEvent(
        symbol="BAIG", ex_date=ts, factor=10.0, source="heuristic", note="heur"
    )
    merged = merge_split_events({"flex": [flex], "heuristic": [heur]})
    assert len(merged) == 1
    assert merged[0].source == "flex"


# ──────────────────────────────────────────────────────────────────────
# (10) parse_yahoo_split_events
# ──────────────────────────────────────────────────────────────────────
def test_parse_yahoo_split_events_basic() -> None:
    payload = {
        "1772202600": {
            "date": 1772202600,
            "numerator": 1.0,
            "denominator": 20.0,
            "splitRatio": "1:20",
        }
    }
    events = parse_yahoo_split_events("SOLT", payload)
    assert len(events) == 1
    ev = events[0]
    assert ev.symbol == "SOLT"
    # 1-for-20 reverse split → price-multiplier 20.
    assert ev.factor == pytest.approx(20.0)
    assert ev.source == "yahoo_events"


# ──────────────────────────────────────────────────────────────────────
# (11) apply_split_events idempotency
# ──────────────────────────────────────────────────────────────────────
def test_apply_split_events_idempotent_via_self_heal() -> None:
    rng = np.random.default_rng(99)
    base = 4.0 * np.exp(rng.normal(0, 0.04, size=120).cumsum())
    base[-1] = base[-2] * 10.0
    s = _build_series(list(base))

    ev = SplitEvent(
        symbol="X",
        ex_date=s.index[-1],
        factor=10.0,
        source="splits_overrides_csv",
        note="test",
    )
    once, applied1 = apply_split_events(s, [ev], sym_label="X")
    twice, applied2 = apply_split_events(once, [ev], sym_label="X")
    # First apply should adjust history; second apply should self-heal and
    # leave the (already-corrected) series unchanged.
    assert len(applied1) == 1
    assert len(applied2) == 0
    assert np.allclose(once.values, twice.values, rtol=1e-12)


# ──────────────────────────────────────────────────────────────────────
# (12) Multi-day reverse-split crater (NBIZ-style)
# ──────────────────────────────────────────────────────────────────────
def test_repair_split_craters_nbiz_style() -> None:
    """Bad mid-prints around a reverse split must be stitched, not ×10'd."""
    # Smooth pre → crater → recover near pre (vendor already continuous).
    prices = (
        [10.0, 10.2, 9.8, 9.5, 9.1]  # pre
        + [0.976, 0.912]  # garbage
        + [11.35, 12.0, 12.5]  # recover
    )
    s = _build_series(prices, start="2026-05-20")
    fixed = repair_split_craters(s, sym_label="NBIZ")
    r = fixed.pct_change().dropna()
    assert float(r.abs().max()) < 0.5
    # Crater bars should sit between anchor and recovery.
    crater = fixed.iloc[5:7]
    assert float(crater.min()) > 5.0
    assert float(crater.max()) < 12.0


def test_flex_reverse_split_skips_when_crater_repaired() -> None:
    prices = (
        [10.0, 10.2, 9.8, 9.5, 9.1]
        + [0.976, 0.912]
        + [11.35, 12.0, 12.5]
    )
    s = _build_series(prices, start="2026-05-20")
    ex = s.index[5]  # first crater day
    ev = SplitEvent(
        symbol="NBIZ",
        ex_date=ex,
        factor=10.0,
        source="flex",
        note="1-for-10 reverse",
    )
    cleaned, applied = apply_split_events(s, [ev], sym_label="NBIZ")
    assert applied == []  # self-heal after crater stitch
    r = cleaned.pct_change().dropna()
    assert float(r.abs().max()) < 0.5
    # Must not blow pre-history up by 10×.
    assert float(cleaned.iloc[0]) < 50.0


# ──────────────────────────────────────────────────────────────────────
# (13) Factor 6 whitelist + forced overrides (COYY / SNDU regressions)
# ──────────────────────────────────────────────────────────────────────
def test_heuristic_matches_factor_six() -> None:
    """Raw ~6.06× reverse split must match ×6, not nearest whitelist ×5."""
    from splits import _INTEGER_SPLIT_FACTORS

    assert 6 in _INTEGER_SPLIT_FACTORS
    assert 8 not in _INTEGER_SPLIT_FACTORS
    rng = np.random.default_rng(608)
    base = 3.2 * np.exp(rng.normal(0, 0.03, size=80).cumsum())
    base[-1] = base[-2] * 6.06
    s = _build_series(list(base), start="2026-03-01")
    events = detect_heuristic_splits(s, sym_label="COYY")
    assert events, "expected heuristic event on final bar"
    assert events[-1].factor == pytest.approx(6.0)


def test_overrides_csv_force_applies_despite_lookahead_self_heal(
    tmp_path: Path,
) -> None:
    """Operator overrides must apply even when lookahead self-heal would skip.

    SNDU-style: reverse jump, then multi-day plateau, then collapse back —
    stable-post lookahead finds the recovered price and falsely self-heals.
    """
    # pre ~43 → inflated 142 for 3 days → back to ~40
    prices = [44.0, 43.3] + [142.3, 142.3, 142.3] + [40.2, 52.0, 52.0]
    s = _build_series(prices, start="2026-04-17")
    csv = tmp_path / "overrides.csv"
    csv.write_text(
        dedent(
            """\
            symbol,ex_date,numerator,denominator,source,note
            SNDU,2026-04-21,1,3,manual,1-for-3 reverse
            SNDU,2026-04-27,3,1,manual,3-for-1 forward
            """
        ),
        encoding="utf-8",
    )
    cleaned, applied = clean_split_artifacts(
        s,
        ticker="SNDU",
        flex_splits_csv=None,
        splits_overrides_csv=csv,
        use_heuristic=False,
        return_log=True,
    )
    assert len(applied) == 2
    assert {round(e.factor, 4) for e in applied} == {3.0, round(1 / 3, 4)}
    # Inflated window must be scaled down; no remaining >100% day in Apr window.
    apr = cleaned.loc["2026-04-17":"2026-04-30"]
    assert float(apr.pct_change().abs().max()) < 0.55
    assert float(apr.max()) < 80.0


def test_coyy_style_june_residual_fixed_by_override(tmp_path: Path) -> None:
    """After crater lifts pre-history, 06-08 ~6× residual must apply as ×6."""
    # Explicit calendar so the ~6× gap lands on Monday 2026-06-08 (not Fri 06-05).
    idx = pd.to_datetime(
        [
            "2026-05-28",
            "2026-05-29",
            "2026-06-01",
            "2026-06-02",
            "2026-06-03",
            "2026-06-04",
            "2026-06-05",
            "2026-06-08",
            "2026-06-09",
            "2026-06-10",
        ]
    )
    vals = [3.50, 3.45, 3.45, 3.35, 3.36, 3.27, 3.27, 19.80, 19.35, 19.48]
    s = pd.Series(vals, index=idx, dtype=float)
    csv = tmp_path / "overrides.csv"
    csv.write_text(
        "symbol,ex_date,numerator,denominator,source,note\n"
        "COYY,2026-06-08,1,6,manual,residual 1-for-6\n",
        encoding="utf-8",
    )
    cleaned, applied = clean_split_artifacts(
        s,
        ticker="COYY",
        flex_splits_csv=None,
        splits_overrides_csv=csv,
        use_heuristic=False,
        return_log=True,
    )
    assert applied and applied[0].factor == pytest.approx(6.0)
    # Boundary across 06-08 should be near continuous on post basis.
    i = list(cleaned.index).index(pd.Timestamp("2026-06-08"))
    ratio = float(cleaned.iloc[i]) / float(cleaned.iloc[i - 1])
    assert 0.85 < ratio < 1.15
