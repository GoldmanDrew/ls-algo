"""Tests for the conditional-crash sizing overlay (research).

Production crash-budget wiring: ``tests/test_b4_crash_budget.py``.
The old opt2 unconditional tail penalty was removed from ``v6_b4_pf_weights``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.b4_ta_research_lib import (  # noqa: E402
    CrashSpec,
    PairCtx,
    RatchetSpec,
    SimSpec,
    TaSpec,
    apply_crash_overlay,
    build_crash_panel,
    runup_vs_anchor,
    simulate_sleeve,
    trailing_tail_risk,
)
from scripts.v6_b4_pf_weights import V6PfParams  # noqa: E402


def _norm(s: str) -> str:
    return str(s).strip().upper().replace(".", "-")


# ---------------------------------------------------------------------------
# Signal primitives
# ---------------------------------------------------------------------------
class TestSignals:
    def test_runup_flags_rally_not_flat(self):
        idx = pd.bdate_range("2024-01-01", periods=400)
        flat = pd.Series(100.0, index=idx)
        rally = pd.Series(np.linspace(100, 300, len(idx)), index=idx)
        assert float(runup_vs_anchor(flat).iloc[-1]) == pytest.approx(0.0)
        # anchor = rolling 252d median ~= price ~126d back; linear 3x rally -> ~0.27
        assert float(runup_vs_anchor(rally).iloc[-1]) > 0.2

    def test_runup_clips_drawdown_to_zero(self):
        idx = pd.bdate_range("2024-01-01", periods=400)
        crash = pd.Series(np.linspace(300, 100, len(idx)), index=idx)
        assert float(runup_vs_anchor(crash).iloc[-1]) == pytest.approx(0.0)

    def test_trailing_tail_sees_recent_crash(self):
        idx = pd.bdate_range("2024-01-01", periods=300)
        px = np.full(len(idx), 100.0)
        px[250:270] = np.linspace(100, 55, 20)  # -45% crash
        px[270:] = 55.0
        tail = trailing_tail_risk(pd.Series(px, index=idx))
        assert float(tail.iloc[-1]) > 0.35

    def test_build_crash_panel_shape_and_shift(self):
        idx = pd.bdate_range("2024-01-01", periods=300)
        px = pd.DataFrame({"a_px": 50.0, "b_px": np.linspace(100, 250, len(idx))}, index=idx)
        panel = build_crash_panel([("ETFX", "UNDX")], {"ETFX": px})
        assert not panel.empty
        assert set(panel.columns) >= {"runup", "tail"}
        last = panel.xs("UNDX", level="underlying").iloc[-1]
        assert float(last["runup"]) > 0.0


# ---------------------------------------------------------------------------
# Overlay
# ---------------------------------------------------------------------------
def _snap_two_pairs(budget: float = 100_000.0) -> pd.DataFrame:
    rows = []
    for etf, und in (("RISKX", "AMD"), ("SAFEX", "KO")):
        gross = 0.5 * budget
        inv = gross / (1.0 + 0.45 * 2.0)  # h=0.45, beta=2
        rows.append({
            "snapshot_date": pd.Timestamp("2025-06-02"), "ETF": etf, "Underlying": und,
            "Delta": -2.0, "hedge_ratio": 0.45, "pair_weight": 0.5,
            "gross_target_usd": gross, "inverse_etf_short_usd": inv,
            "underlying_short_usd": gross - inv, "mode": "production",
        })
    return pd.DataFrame(rows)


def _crash_panel(day: str = "2025-06-02") -> pd.DataFrame:
    d = pd.Timestamp(day)
    rows = [
        {"date": d, "underlying": "AMD", "runup": 0.8, "tail": 0.30},
        {"date": d, "underlying": "KO", "runup": 0.0, "tail": 0.05},
    ]
    return pd.DataFrame(rows).set_index(["date", "underlying"]).sort_index()


class TestApplyCrashOverlay:
    def test_trim_only_and_no_renorm(self):
        snap = _snap_two_pairs()
        out = apply_crash_overlay(snap, _crash_panel(), pd.Timestamp("2025-06-10"),
                                  CrashSpec(enabled=True, inverse_es=True), budget=100_000.0)
        assert (out["crash_mult"] <= 1.0 + 1e-9).all()
        # freed weight NOT redeployed: total gross strictly below original budget share
        assert float(out["gross_target_usd"].sum()) < float(snap["gross_target_usd"].sum())

    def test_inverse_es_trims_risky_name_more(self):
        out = apply_crash_overlay(_snap_two_pairs(), _crash_panel(), pd.Timestamp("2025-06-10"),
                                  CrashSpec(enabled=True, inverse_es=True), budget=100_000.0)
        amd = out[out["Underlying"] == "AMD"].iloc[0]
        ko = out[out["Underlying"] == "KO"].iloc[0]
        assert float(amd["crash_mult"]) < float(ko["crash_mult"])
        # least-risky name keeps full size under pure inverse-ES
        assert float(ko["crash_mult"]) == pytest.approx(1.0)

    def test_retrace_floor_binds_when_tail_is_quiet(self):
        """AMD problem: huge run-up but quiet realized tail must still get flagged."""
        panel = _crash_panel()
        panel.loc[(pd.Timestamp("2025-06-02"), "AMD"), "tail"] = 0.02  # quiet history
        out = apply_crash_overlay(_snap_two_pairs(), panel, pd.Timestamp("2025-06-10"),
                                  CrashSpec(enabled=True, inverse_es=True, theta=0.5), budget=100_000.0)
        amd = out[out["Underlying"] == "AMD"].iloc[0]
        # retrace = 0.5 * 0.8/1.8 ~= 0.222 > tail 0.02
        assert float(amd["crash_C"]) == pytest.approx(0.5 * 0.8 / 1.8, rel=1e-6)
        assert float(amd["crash_mult"]) < 1.0

    def test_crash_cap_enforces_budget(self):
        budget = 100_000.0
        spec = CrashSpec(enabled=True, crash_cap=True, rho=0.0075)
        out = apply_crash_overlay(_snap_two_pairs(budget), _crash_panel(), pd.Timestamp("2025-06-10"),
                                  spec, budget=budget)
        viol = out["gross_target_usd"] * out["crash_L"].clip(lower=spec.l_floor) - spec.rho * budget
        assert (viol <= 1e-6).all()

    def test_h_tilt_raises_hedge_and_keeps_gross(self):
        budget = 100_000.0
        spec = CrashSpec(enabled=True, h_tilt=True, kappa=0.5, h_max=0.85)
        out = apply_crash_overlay(_snap_two_pairs(budget), _crash_panel(), pd.Timestamp("2025-06-10"),
                                  spec, budget=budget)
        amd = out[out["Underlying"] == "AMD"].iloc[0]
        ko = out[out["Underlying"] == "KO"].iloc[0]
        assert float(amd["crash_h"]) > 0.45
        assert float(amd["crash_h"]) <= 0.85 + 1e-9
        assert float(ko["crash_h"]) == pytest.approx(0.45)
        # size lever off -> gross unchanged, only the leg split moves
        assert float(out["gross_target_usd"].sum()) == pytest.approx(budget)
        base_inv = 0.5 * budget / (1.0 + 0.45 * 2.0)
        assert float(amd["inverse_etf_short_usd"]) < base_inv

    def test_missing_signal_is_neutral(self):
        out = apply_crash_overlay(_snap_two_pairs(), pd.DataFrame(), pd.Timestamp("2025-06-10"),
                                  CrashSpec(enabled=True, inverse_es=True, crash_cap=True), budget=100_000.0)
        assert (out["crash_mult"] == 1.0).all()

    def test_missing_policy_book_quantile_trims_short_history_names(self):
        budget = 100_000.0
        # KO has signal; AMD (short history) is missing from the panel
        panel = _crash_panel().drop(index=(pd.Timestamp("2025-06-02"), "AMD"))
        panel.loc[(pd.Timestamp("2025-06-02"), "KO"), ["runup", "tail"]] = [0.0, 0.60]
        spec = CrashSpec(enabled=True, crash_cap=True, rho=0.0075, missing_policy="book_quantile")
        out = apply_crash_overlay(_snap_two_pairs(budget), panel, pd.Timestamp("2025-06-10"),
                                  spec, budget=budget)
        amd = out[out["Underlying"] == "AMD"].iloc[0]
        # AMD inherits the book-quantile L (only KO has signal -> its L) and gets capped
        assert float(amd["crash_mult"]) < 1.0
        assert float(amd["crash_L"]) > 0.0
        # neutral policy on the same inputs would have left AMD untouched
        neutral = apply_crash_overlay(_snap_two_pairs(budget), panel, pd.Timestamp("2025-06-10"),
                                      CrashSpec(enabled=True, crash_cap=True, rho=0.0075), budget=budget)
        assert float(neutral[neutral["Underlying"] == "AMD"].iloc[0]["crash_mult"]) == pytest.approx(1.0)

    def test_stacks_on_h_tilt_reduces_L(self):
        budget = 100_000.0
        no_tilt = apply_crash_overlay(_snap_two_pairs(budget), _crash_panel(), pd.Timestamp("2025-06-10"),
                                      CrashSpec(enabled=True, inverse_es=True), budget=budget)
        tilt = apply_crash_overlay(_snap_two_pairs(budget), _crash_panel(), pd.Timestamp("2025-06-10"),
                                   CrashSpec(enabled=True, inverse_es=True, h_tilt=True), budget=budget)
        amd_no = no_tilt[no_tilt["Underlying"] == "AMD"].iloc[0]
        amd_ti = tilt[tilt["Underlying"] == "AMD"].iloc[0]
        assert float(amd_ti["crash_L"]) < float(amd_no["crash_L"])


# ---------------------------------------------------------------------------
# Simulator integration
# ---------------------------------------------------------------------------
def _mk_pair(etf: str, und: str, cal: pd.DatetimeIndex, *, rebal_at: list[int],
             r_und: np.ndarray | None = None) -> PairCtx:
    n = len(cal)
    rb = np.zeros(n, bool)
    for i in rebal_at:
        rb[i] = True
    return PairCtx(
        etf=etf, und=und, beta=2.0, borrow=0.10, edge=0.35, cal=cal,
        r_inv=np.zeros(n), r_und=r_und if r_und is not None else np.zeros(n),
        h=np.full(n, 0.45), is_rebal=rb,
    )


def _lookup_for(snap: pd.DataFrame) -> dict:
    d = pd.Timestamp(snap["snapshot_date"].iloc[0]).normalize()
    return {d: snap}


class TestSimulateWithCrashOverlay:
    def test_overlay_events_recorded_with_dispersion(self):
        cal = pd.bdate_range("2025-06-02", periods=12)
        pairs = [
            _mk_pair("RISKX", "AMD", cal, rebal_at=[5]),
            _mk_pair("SAFEX", "KO", cal, rebal_at=[5]),
        ]
        snap = _snap_two_pairs()
        panel = _crash_panel("2025-06-02")
        res = simulate_sleeve(
            pairs, _lookup_for(snap), budget=100_000.0, ta_panel=pd.DataFrame(),
            ta=TaSpec(False), ratchet=RatchetSpec(enabled=False, trim_enabled=False),
            sim=SimSpec(), crash_panel=panel,
            crash=CrashSpec(enabled=True, inverse_es=True),
        )
        ev = res.overlay_events
        assert ev is not None and not ev.empty
        mults = ev["crash_mult"]
        assert mults.min() < 0.999, "risky name must actually be trimmed"
        assert mults.max() > mults.min(), "multipliers must have cross-sectional dispersion"
        assert res.pair_pnl is not None
        assert list(res.pair_pnl.columns) == ["RISKX/AMD", "SAFEX/KO"]

    def test_disabled_crash_equals_legacy_path(self):
        cal = pd.bdate_range("2025-06-02", periods=12)
        pairs = [_mk_pair("RISKX", "AMD", cal, rebal_at=[5])]
        snap = _snap_two_pairs()
        kw = dict(budget=100_000.0, ta_panel=pd.DataFrame(), ta=TaSpec(False),
                  ratchet=RatchetSpec(enabled=False, trim_enabled=False), sim=SimSpec())
        a = simulate_sleeve(pairs, _lookup_for(snap), **kw)
        b = simulate_sleeve(pairs, _lookup_for(snap), crash_panel=_crash_panel(),
                            crash=CrashSpec(enabled=False), **kw)
        pd.testing.assert_series_equal(a.daily_return, b.daily_return)
        assert a.trade_cost == pytest.approx(b.trade_cost)

    def test_hysteresis_band_suppresses_small_resizes(self):
        cal = pd.bdate_range("2025-06-02", periods=12)
        # small drift: +0.5%/day underlying moves the held leg a few percent
        r_und = np.full(len(cal), 0.005)
        r_und[0] = 0.0
        pairs = [_mk_pair("RISKX", "AMD", cal, rebal_at=[5, 10], r_und=r_und)]
        snap = _snap_two_pairs()
        kw = dict(budget=100_000.0, ta_panel=pd.DataFrame(), ta=TaSpec(False),
                  ratchet=RatchetSpec(enabled=False, trim_enabled=False))
        loose = simulate_sleeve(pairs, _lookup_for(snap), sim=SimSpec(), **kw)
        banded = simulate_sleeve(
            pairs, _lookup_for(snap),
            sim=SimSpec(enter_band_pct=0.12, min_trade_usd=250.0), **kw,
        )
        # banded run must trade strictly less after the initial establish
        assert banded.trade_cost + banded.cover_cost < loose.trade_cost + loose.cover_cost


def test_v6_params_drop_retired_keys():
    """Stale YAML with retired knobs must not break the weight engine."""
    p = V6PfParams.from_opt2_dict({
        "dd_risk_lambda": 2.5,
        "risk_denom_coeff": 3.0,
        "tail_as_of": "latest",
        "exclude_if_borrow_annual_gt": 0.90,
        "borrow_linear_aversion": 1.5,
        "unknown_key": 1,
    })
    assert p.borrow_linear_aversion == pytest.approx(1.5)
    assert not hasattr(p, "dd_risk_lambda")
    assert not hasattr(p, "tail_as_of")
    assert not hasattr(p, "exclude_if_borrow_annual_gt")


def test_v6_params_borrow_ramp_knobs():
    """Continuous borrow ramp knobs load from YAML; sane defaults."""
    p = V6PfParams.from_opt2_dict({"borrow_ramp_lo": 0.7, "borrow_ramp_hi": 1.1})
    assert p.borrow_ramp_lo == pytest.approx(0.7)
    assert p.borrow_ramp_hi == pytest.approx(1.1)
    d = V6PfParams()
    assert d.borrow_ramp_lo == pytest.approx(0.80)
    assert d.borrow_ramp_hi == pytest.approx(1.20)


class TestWeightSmoothing:
    def _smooth(self):
        from scripts.bucket4_weekly_opt2 import smooth_pair_weights_trim_only
        return smooth_pair_weights_trim_only

    def test_cuts_immediate_increases_smoothed(self):
        f = self._smooth()
        prev = {("A", "X"): 0.5, ("B", "Y"): 0.5}
        solved = {("A", "X"): 0.2, ("B", "Y"): 0.8}   # A cut, B increase
        # Own-risk collapse on A => hard cut; B raises EMA.
        out = f(
            solved,
            prev,
            alpha=0.5,
            own_risk_weights={("A", "X"): 0.02, ("B", "Y"): 0.05},
            prev_own_risk_weights={("A", "X"): 0.05, ("B", "Y"): 0.05},
        )
        assert out[("A", "X")] == pytest.approx(0.2)
        assert out[("B", "Y")] == pytest.approx(0.65)

    def test_identical_solve_is_fixed_point(self):
        f = self._smooth()
        solved = {("A", "X"): 0.6, ("B", "Y"): 0.4}
        out1 = f(solved, {}, alpha=0.5)          # first run: state = solved
        out2 = f(solved, out1, alpha=0.5)        # same solve next run: no-op
        assert out2 == pytest.approx(out1)

    def test_first_run_is_noop(self):
        f = self._smooth()
        solved = {("A", "X"): 0.6, ("B", "Y"): 0.4}
        out = f(solved, {}, alpha=0.25)
        assert out == pytest.approx(solved)

    def test_dropped_pair_gets_no_weight(self):
        f = self._smooth()
        out = f({("A", "X"): 1.0}, {("A", "X"): 0.5, ("GONE", "Z"): 0.5}, alpha=0.5)
        assert ("GONE", "Z") not in out
        # survivor's increase is smoothed (0.5 -> 0.75), NOT snapped to 1.0:
        # the dropped pair's weight is freed to cash, not redeployed.
        assert out[("A", "X")] == pytest.approx(0.75)

    def test_new_entry_ramps_from_zero_when_history_exists(self):
        f = self._smooth()
        prev = {("A", "X"): 0.5}
        solved = {("A", "X"): 0.5, ("NEW", "N"): 0.24}
        out = f(solved, prev, alpha=0.5, ramp_new_entries=True, entry_alpha=0.25)
        # First week is entry_alpha * solved, not full size.
        assert out[("NEW", "N")] == pytest.approx(0.06)
        assert out[("A", "X")] == pytest.approx(0.5)

    def test_dilution_cut_is_smoothed_not_instant(self):
        f = self._smooth()
        # Incumbent A: own risk capacity unchanged, but post-scale target fell
        # because a new name claimed book share.
        prev = {("A", "X"): 0.40, ("B", "Y"): 0.40}
        target = {("A", "X"): 0.30, ("B", "Y"): 0.30, ("NEW", "N"): 0.40}
        own = {("A", "X"): 0.05, ("B", "Y"): 0.05, ("NEW", "N"): 0.05}
        own_prev = {("A", "X"): 0.05, ("B", "Y"): 0.05}
        out = f(
            target,
            prev,
            alpha=0.5,
            ramp_new_entries=True,
            entry_alpha=0.25,
            dilution_alpha=0.25,
            own_risk_weights=own,
            prev_own_risk_weights=own_prev,
        )
        # Dilution: A moves 25% of the way from 0.40 toward 0.30 = 0.375
        assert out[("A", "X")] == pytest.approx(0.375)
        assert out[("NEW", "N")] == pytest.approx(0.10)  # 0.25 * 0.40

    def test_hard_own_risk_cut_is_immediate(self):
        f = self._smooth()
        prev = {("A", "X"): 0.40}
        target = {("A", "X"): 0.20}
        # Own pre-scale capacity halved -> hard cut.
        out = f(
            target,
            prev,
            alpha=0.5,
            dilution_alpha=0.25,
            own_risk_weights={("A", "X"): 0.02},
            prev_own_risk_weights={("A", "X"): 0.05},
            hard_cut_rel=0.10,
        )
        assert out[("A", "X")] == pytest.approx(0.20)

    def test_soft_exit_fades_dropped_name(self):
        f = self._smooth()
        prev = {("A", "X"): 0.40, ("GONE", "Z"): 0.20}
        out = f(
            {("A", "X"): 0.40},
            prev,
            alpha=0.5,
            soft_exit_alpha=0.35,
            own_risk_weights={("A", "X"): 0.05},
            prev_own_risk_weights={("A", "X"): 0.05, ("GONE", "Z"): 0.03},
        )
        assert ("GONE", "Z") in out
        assert out[("GONE", "Z")] == pytest.approx(0.20 * 0.65)
        # Hard exit when soft_exit_alpha is None (legacy).
        hard = f({("A", "X"): 0.40}, prev, alpha=0.5, soft_exit_alpha=None)
        assert ("GONE", "Z") not in hard

    def test_ramp_skipped_on_first_ever_run(self):
        f = self._smooth()
        solved = {("A", "X"): 0.6, ("B", "Y"): 0.4}
        out = f(solved, {}, alpha=0.5, ramp_new_entries=True)
        # Empty state (day one) -> no-op, same as before the ramp existed.
        assert out == pytest.approx(solved)

    def test_no_trade_band_holds_small_moves(self):
        f = self._smooth()
        prev = {("A", "X"): 0.10, ("B", "Y"): 0.10}
        solved = {("A", "X"): 0.104, ("B", "Y"): 0.05}
        out = f(
            solved,
            prev,
            alpha=1.0,
            no_trade_band_rel=0.15,
            own_risk_weights={("A", "X"): 0.02, ("B", "Y"): 0.01},
            prev_own_risk_weights={("A", "X"): 0.02, ("B", "Y"): 0.02},
        )
        # A's +4% move is inside the 15% band -> held; B's own-risk cut passes.
        assert out[("A", "X")] == pytest.approx(0.10)
        assert out[("B", "Y")] == pytest.approx(0.05)

    def test_no_trade_band_abs_floor(self):
        f = self._smooth()
        prev = {("A", "X"): 0.002}
        solved = {("A", "X"): 0.0035}
        out = f(solved, prev, alpha=1.0, no_trade_band_abs=0.0025)
        # Move of 0.0015 < 25bp absolute band -> held.
        assert out[("A", "X")] == pytest.approx(0.002)
