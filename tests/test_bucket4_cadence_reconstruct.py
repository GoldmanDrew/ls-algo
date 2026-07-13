"""Tests for broker-based Bucket 4 cadence reconstruction (machine-independent
timing without committing data/b4_cadence_state.json)."""
from __future__ import annotations

from pathlib import Path

from scripts.bucket4_cadence_reconstruct import (
    broker_last_rebalance_by_pair,
    latest_flex_trades_file,
    parse_flex_trade_last_dates,
    reconcile_cadence_state,
    stagger_seed_date,
    stagger_seed_offset,
)

_FLEX_XML = """<?xml version="1.0"?>
<FlexQueryResponse>
 <FlexStatements>
  <FlexStatement>
   <Trades>
    <Trade assetCategory="STK" symbol="MSTX" underlyingSymbol="MSTR" buySell="SELL" quantity="-100" tradeDate="20260701" dateTime="20260701;145545"/>
    <Trade assetCategory="STK" symbol="MSTX" underlyingSymbol="MSTR" buySell="SELL" quantity="-50" tradeDate="20260709" dateTime="20260709;150000"/>
    <Trade assetCategory="STK" symbol="MSTU" underlyingSymbol="MSTR" buySell="SELL" quantity="-40" tradeDate="20260703" dateTime="20260703;150000"/>
    <Trade assetCategory="STK" symbol="MSTR" underlyingSymbol="MSTR" buySell="BUY" quantity="200" tradeDate="20260710" dateTime="20260710;150000"/>
    <Trade assetCategory="OPT" symbol="MSTZ" underlyingSymbol="MSTR" buySell="SELL" quantity="-5" tradeDate="20260711" dateTime="20260711;150000"/>
   </Trades>
  </FlexStatement>
 </FlexStatements>
</FlexQueryResponse>
"""


def _write_flex(tmp_path: Path, run_date: str) -> Path:
    d = tmp_path / "runs" / run_date / "ibkr_flex"
    d.mkdir(parents=True, exist_ok=True)
    p = d / "flex_trades.xml"
    p.write_text(_FLEX_XML, encoding="utf-8")
    return p


def test_parse_last_dates_takes_latest_and_skips_non_equity(tmp_path: Path):
    p = _write_flex(tmp_path, "2026-07-11")
    dates = parse_flex_trade_last_dates(p)
    assert dates["MSTX"] == "2026-07-09"  # latest of 07-01 / 07-09
    assert dates["MSTU"] == "2026-07-03"
    assert dates["MSTR"] == "2026-07-10"
    assert "MSTZ" not in dates  # OPT asset category excluded


def test_parse_respects_on_or_before(tmp_path: Path):
    p = _write_flex(tmp_path, "2026-07-11")
    dates = parse_flex_trade_last_dates(p, on_or_before="2026-07-05")
    assert dates["MSTX"] == "2026-07-01"  # 07-09 dropped by cutoff
    assert "MSTR" not in dates  # only traded 07-10


def test_broker_last_rebalance_keys_off_inverse_etf(tmp_path: Path):
    _write_flex(tmp_path, "2026-07-11")
    pairs = [("MSTX", "MSTR"), ("MSTU", "MSTR"), ("MSTZ", "MSTR")]
    out = broker_last_rebalance_by_pair(
        pairs, run_date="2026-07-11", runs_root=tmp_path / "runs"
    )
    assert out[("MSTX", "MSTR")] == "2026-07-09"
    assert out[("MSTU", "MSTR")] == "2026-07-03"
    assert ("MSTZ", "MSTR") not in out  # never traded as equity -> no timing


def test_latest_flex_file_picks_newest_on_or_before(tmp_path: Path):
    _write_flex(tmp_path, "2026-07-08")
    _write_flex(tmp_path, "2026-07-11")
    src, path = latest_flex_trades_file("2026-07-11", runs_root=tmp_path / "runs")
    assert src == "2026-07-11"
    src2, _ = latest_flex_trades_file("2026-07-10", runs_root=tmp_path / "runs")
    assert src2 == "2026-07-08"


def test_stagger_offset_deterministic_and_bounded():
    for interval in (0, 1, 5, 14, 21):
        for key in ("MSTX|MSTR", "MSTU|MSTR", "CLSZ|CLSK"):
            off = stagger_seed_offset(key, interval)
            assert 0 <= off <= interval
            assert off == stagger_seed_offset(key, interval)  # stable


def test_stagger_seed_date_is_business_day_before_run():
    d = stagger_seed_date("2026-07-13", 14, "MSTX|MSTR")  # Monday
    assert d <= "2026-07-13"


def test_reconcile_prefers_most_recent_and_staggers_unknown(tmp_path: Path):
    _write_flex(tmp_path, "2026-07-11")
    # Cache says MSTX rebalanced 07-06; broker says 07-09 -> take the newer 07-09.
    # MSTU cache is newer (07-08) than broker (07-03) -> keep 07-08.
    # NEWX has no broker trade + no cache -> stagger seed.
    state = {"MSTX|MSTR": "2026-07-06", "MSTU|MSTR": "2026-07-08"}
    pair_intervals = {
        ("MSTX", "MSTR"): 14,
        ("MSTU", "MSTR"): 14,
        ("NEWX", "NEWU"): 14,
    }
    new_state, prov = reconcile_cadence_state(
        state,
        pair_intervals=pair_intervals,
        run_date="2026-07-13",
        runs_root=tmp_path / "runs",
    )
    assert new_state["MSTX|MSTR"] == "2026-07-09"
    assert prov["MSTX|MSTR"] == "both"
    assert new_state["MSTU|MSTR"] == "2026-07-08"
    assert prov["MSTU|MSTR"] == "both"
    assert prov["NEWX|NEWU"] == "stagger"
    assert new_state["NEWX|NEWU"] <= "2026-07-13"


def test_reconcile_no_broker_file_falls_back_to_cache_and_stagger(tmp_path: Path):
    state = {"MSTX|MSTR": "2026-07-06"}
    pair_intervals = {("MSTX", "MSTR"): 14, ("MSTU", "MSTR"): 14}
    new_state, prov = reconcile_cadence_state(
        state,
        pair_intervals=pair_intervals,
        run_date="2026-07-13",
        runs_root=tmp_path / "runs",  # empty -> no flex file
    )
    assert new_state["MSTX|MSTR"] == "2026-07-06"
    assert prov["MSTX|MSTR"] == "cache"
    assert prov["MSTU|MSTR"] == "stagger"
