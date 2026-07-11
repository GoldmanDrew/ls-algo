"""Tests for the production B4 conditional-crash budget (scripts/b4_crash_budget.py)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.b4_crash_budget import (
    CrashBudgetParams,
    book_default_cap_usd,
    cap_pair_weights,
    clamp_sized_to_crash_budget,
    compute_crash_caps,
    conditional_crash_stats,
    pair_loss,
)

P = CrashBudgetParams()


def _px(values, start="2023-01-02"):
    idx = pd.bdate_range(start, periods=len(values))
    return pd.Series(np.asarray(values, dtype=float), index=idx)


def _rally(n=800):
    # Flat, then 4x over the final year (AMD-style blow-off).
    # Linear ramp a->b has median (a+b)/2, so runup = (b-a)/(a+b) = 0.6.
    v = 100.0 * np.ones(n)
    v[-252:] = np.linspace(100.0, 400.0, 252)
    return _px(v)


def _flat(n=800):
    rng = np.random.default_rng(7)
    return _px(100.0 * np.exp(np.cumsum(rng.normal(0, 0.001, n))))


def _crashy(n=800):
    v = 100.0 * np.ones(n)
    v[400:420] *= np.linspace(1.0, 0.5, 20)  # 50% crash over 20d
    v[420:] = v[419]
    return _px(v)


# ---------------------------------------------------------------- signals
class TestConditionalCrashStats:
    def test_rally_has_positive_retrace(self):
        s = conditional_crash_stats(_rally(), P)
        assert s is not None
        assert s["runup"] > 0.3
        assert s["retrace"] == pytest.approx(P.theta * s["runup"] / (1 + s["runup"]))
        assert s["C"] >= s["retrace"]

    def test_flat_name_has_near_zero_retrace(self):
        s = conditional_crash_stats(_flat(), P)
        assert s is not None
        assert s["retrace"] < 0.02

    def test_crashy_name_tail_dominates(self):
        s = conditional_crash_stats(_crashy(), P)
        assert s is not None
        assert s["tail"] >= 0.5          # realized 50% 20d drop is in the window
        assert s["C"] == pytest.approx(max(s["tail"], s["retrace"]))

    def test_short_history_returns_none(self):
        assert conditional_crash_stats(_px([100, 101, 102, 103, 104]), P) is None


# ---------------------------------------------------------------- pair loss
class TestPairLoss:
    def test_unhedged_beta2_is_2c_times_convexity(self):
        c = 0.30
        assert pair_loss(c, h=0.0, beta=2.0, phi=0.5) == pytest.approx(2.0 * c * (1 + 0.5 * c))

    def test_decreasing_in_hedge_increasing_in_crash(self):
        assert pair_loss(0.3, 0.6, 2.0, 0.5) < pair_loss(0.3, 0.3, 2.0, 0.5)
        assert pair_loss(0.5, 0.45, 2.0, 0.5) > pair_loss(0.2, 0.45, 2.0, 0.5)

    def test_full_hedge_zero_loss(self):
        assert pair_loss(0.5, 1.0, 2.0, 0.5) == 0.0


# ---------------------------------------------------------------- caps + weights
def _norm(s: str) -> str:
    return s.strip().upper()


def _pair_cache():
    def entry(px):
        return {"prices": pd.DataFrame({"a_px": px.values, "b_px": px.values}, index=px.index),
                "kw": {"beta_a": -2.0, "beta_b": 1.0}}
    return {
        ("AMDS", "AMD"): entry(_rally()),       # stretched -> should be capped
        ("METD", "META"): entry(_flat()),       # calm -> mild/no cap
        ("NEWS", "NEW"): entry(_px([100, 101, 99, 100, 102])),  # short history
    }


def _hedges(cache, h=0.45):
    out = {}
    for (_, und), c in cache.items():
        idx = c["prices"].index
        out[und] = pd.Series(h, index=idx)
    return out


class TestComputeCrashCaps:
    def test_caps_table_and_missing_policy(self):
        cache = _pair_cache()
        caps = compute_crash_caps(
            pair_cache=cache, hedge_by_underlying=_hedges(cache), closes_broad=None,
            hedge_base=0.45, run_date="2026-07-08", budget_usd=100_000.0,
            params=P, norm_sym=_norm,
        )
        assert len(caps) == 3
        amd = caps.loc[caps["Underlying"] == "AMD"].iloc[0]
        new = caps.loc[caps["Underlying"] == "NEW"].iloc[0]
        assert bool(amd["signal_ok"]) and amd["crash_l_source"] == "signal"
        assert not bool(new["signal_ok"]) and new["crash_l_source"] == "book_quantile"
        # book_quantile assigns the q75 L of signal-ok names
        ok_l = caps.loc[caps["signal_ok"], "L"].clip(lower=P.l_floor)
        assert new["L"] == pytest.approx(float(ok_l.quantile(P.missing_l_quantile)))
        # cap = rho * budget / max(L, floor)
        assert amd["cap_usd"] == pytest.approx(
            P.rho * 100_000.0 / max(float(amd["L"]), P.l_floor)
        )

    def test_neutral_policy_leaves_missing_uncapped(self):
        cache = _pair_cache()
        params = CrashBudgetParams(missing_policy="neutral")
        caps = compute_crash_caps(
            pair_cache=cache, hedge_by_underlying=_hedges(cache), closes_broad=None,
            hedge_base=0.45, run_date="2026-07-08", budget_usd=100_000.0,
            params=params, norm_sym=_norm,
        )
        new = caps.loc[caps["Underlying"] == "NEW"].iloc[0]
        assert np.isinf(float(new["cap_usd"]))

    def test_l_ema_risk_up_immediate_risk_down_smoothed(self):
        cache = _pair_cache()
        params = CrashBudgetParams(l_ema_alpha=0.4)
        base = compute_crash_caps(
            pair_cache=cache, hedge_by_underlying=_hedges(cache), closes_broad=None,
            hedge_base=0.45, run_date="2026-07-08", budget_usd=100_000.0,
            params=params, norm_sym=_norm,
        )
        amd_l = float(base.loc[base["Underlying"] == "AMD", "L"].iloc[0])

        # prev L below the fresh estimate (risk UP): fresh L binds immediately.
        caps_up = compute_crash_caps(
            pair_cache=cache, hedge_by_underlying=_hedges(cache), closes_broad=None,
            hedge_base=0.45, run_date="2026-07-08", budget_usd=100_000.0,
            params=params, norm_sym=_norm,
            prev_l={("AMDS", "AMD"): amd_l * 0.5},
        )
        assert float(caps_up.loc[caps_up["Underlying"] == "AMD", "L"].iloc[0]) == pytest.approx(amd_l)

        # prev L above the fresh estimate (risk DOWN): only alpha of the drop
        # passes -> caps loosen gradually instead of stepping.
        prev = amd_l * 2.0
        caps_dn = compute_crash_caps(
            pair_cache=cache, hedge_by_underlying=_hedges(cache), closes_broad=None,
            hedge_base=0.45, run_date="2026-07-08", budget_usd=100_000.0,
            params=params, norm_sym=_norm,
            prev_l={("AMDS", "AMD"): prev},
        )
        row = caps_dn.loc[caps_dn["Underlying"] == "AMD"].iloc[0]
        assert float(row["L"]) == pytest.approx(0.6 * prev + 0.4 * amd_l)
        assert float(row["L_raw"]) == pytest.approx(amd_l)
        assert float(row["cap_usd"]) == pytest.approx(
            params.rho * 100_000.0 / max(float(row["L"]), params.l_floor)
        )

    def test_l_ema_disabled_by_default(self):
        cache = _pair_cache()
        base = compute_crash_caps(
            pair_cache=cache, hedge_by_underlying=_hedges(cache), closes_broad=None,
            hedge_base=0.45, run_date="2026-07-08", budget_usd=100_000.0,
            params=P, norm_sym=_norm,
        )
        amd_l = float(base.loc[base["Underlying"] == "AMD", "L"].iloc[0])
        caps = compute_crash_caps(
            pair_cache=cache, hedge_by_underlying=_hedges(cache), closes_broad=None,
            hedge_base=0.45, run_date="2026-07-08", budget_usd=100_000.0,
            params=P, norm_sym=_norm,
            prev_l={("AMDS", "AMD"): amd_l * 10.0},
        )
        # l_ema_alpha=1.0 (default): prev state is ignored entirely.
        assert float(caps.loc[caps["Underlying"] == "AMD", "L"].iloc[0]) == pytest.approx(amd_l)


class TestCapPairWeights:
    def test_trim_only_and_budget_shrinks(self):
        cache = _pair_cache()
        budget = 100_000.0
        caps = compute_crash_caps(
            pair_cache=cache, hedge_by_underlying=_hedges(cache), closes_broad=None,
            hedge_base=0.45, run_date="2026-07-08", budget_usd=budget,
            params=P, norm_sym=_norm,
        )
        pw = {("AMDS", "AMD"): 0.5, ("METD", "META"): 0.3, ("NEWS", "NEW"): 0.2}
        capped, budget_eff, tel = cap_pair_weights(pw, caps, budget, norm_sym=_norm)

        for k in pw:
            assert capped[k] <= pw[k] + 1e-12  # trim-only
        assert budget_eff <= budget
        assert budget_eff == pytest.approx(budget * sum(capped.values()))

        # Reproduce what compute_bucket4_targets does (renormalize x budget_eff)
        # and check every pair lands at min(solved, cap).
        cap_by_key = {
            (_norm(r["ETF"]), _norm(r["Underlying"])): float(r["cap_usd"])
            for _, r in caps.iterrows()
        }
        total = sum(capped.values())
        for k, w1 in capped.items():
            gross = (w1 / total) * budget_eff
            solved = pw[k] * budget
            assert gross == pytest.approx(min(solved, cap_by_key[k]), rel=1e-9)

    def test_no_caps_is_identity(self):
        pw = {("A", "B"): 0.6, ("C", "D"): 0.4}
        capped, budget_eff, _ = cap_pair_weights(pw, pd.DataFrame(), 50_000.0, norm_sym=_norm)
        assert capped == pytest.approx(pw)
        assert budget_eff == pytest.approx(50_000.0)

    def test_scale_to_budget_refills_and_keeps_proportions(self):
        cache = _pair_cache()
        budget = 100_000.0
        caps = compute_crash_caps(
            pair_cache=cache, hedge_by_underlying=_hedges(cache), closes_broad=None,
            hedge_base=0.45, run_date="2026-07-08", budget_usd=budget,
            params=P, norm_sym=_norm,
        )
        pw = {("AMDS", "AMD"): 0.5, ("METD", "META"): 0.3, ("NEWS", "NEW"): 0.2}
        trimmed, budget_cash, tel_cash = cap_pair_weights(
            pw, caps, budget, norm_sym=_norm, scale_to_budget=False
        )
        scaled, budget_full, tel_sc = cap_pair_weights(
            pw, caps, budget, norm_sym=_norm, scale_to_budget=True
        )
        assert budget_cash < budget
        assert budget_full == pytest.approx(budget)
        assert sum(scaled.values()) == pytest.approx(1.0)
        assert float(tel_sc["scale_mult"].iloc[0]) == pytest.approx(1.0 / sum(trimmed.values()))
        # Proportions of the trimmed book are preserved after scale.
        tsum = sum(trimmed.values())
        for k in pw:
            assert scaled[k] == pytest.approx(trimmed[k] / tsum)
        # Final dollars sum to the full budget.
        assert float(tel_sc["gross_final_usd"].sum()) == pytest.approx(budget)


# ---------------------------------------------------------------- final clamp
class TestClampSized:
    def _frame(self):
        return pd.DataFrame(
            {
                "ETF": ["AMDS", "METD", "TSLQ", "AAPL"],
                "Underlying": ["AMD", "META", "TSLA", "AAPL"],
                "sleeve": ["inverse_decay_bucket4"] * 3 + ["core_leveraged"],
                "gross_target_usd": [10_000.0, 5_000.0, 8_000.0, 20_000.0],
                "crash_budget_clamp_usd": [6_000.0, 9_000.0, np.nan, 1_000.0],
                "b4_opt2_inverse_etf_short_usd": [6_000.0, 3_000.0, 4_800.0, np.nan],
                "b4_opt2_underlying_short_usd": [4_000.0, 2_000.0, 3_200.0, np.nan],
            }
        )

    def test_clamps_only_violating_b4_rows(self):
        out = clamp_sized_to_crash_budget(self._frame())
        # AMDS above its 6k clamp -> trimmed, legs scaled proportionally (h preserved)
        assert out.loc[0, "gross_target_usd"] == pytest.approx(6_000.0)
        assert out.loc[0, "b4_opt2_inverse_etf_short_usd"] == pytest.approx(3_600.0)
        assert out.loc[0, "b4_opt2_underlying_short_usd"] == pytest.approx(2_400.0)
        # METD below clamp, TSLQ no clamp -> untouched
        assert out.loc[1, "gross_target_usd"] == pytest.approx(5_000.0)
        assert out.loc[2, "gross_target_usd"] == pytest.approx(8_000.0)
        # non-B4 sleeve never clamped even with a (bogus) clamp value
        assert out.loc[3, "gross_target_usd"] == pytest.approx(20_000.0)

    def test_noop_without_column(self):
        f = self._frame().drop(columns=["crash_budget_clamp_usd"])
        out = clamp_sized_to_crash_budget(f)
        assert out["gross_target_usd"].tolist() == f["gross_target_usd"].tolist()


# ---------------------------------------------------------------- default cap
class TestBookDefaultCap:
    def test_matches_book_quantile_formula(self):
        cache = _pair_cache()
        budget = 100_000.0
        caps = compute_crash_caps(
            pair_cache=cache, hedge_by_underlying=_hedges(cache), closes_broad=None,
            hedge_base=0.45, run_date="2026-07-08", budget_usd=budget,
            params=P, norm_sym=_norm,
        )
        cap = book_default_cap_usd(caps, budget, P)
        l_q = float(pd.to_numeric(caps["L"], errors="coerce").dropna()
                    .clip(lower=P.l_floor).quantile(P.missing_l_quantile))
        assert cap == pytest.approx(P.rho * budget / l_q)
        # Conservative: never looser than the loosest signal cap in the book
        assert cap <= float(pd.to_numeric(caps["cap_usd"], errors="coerce").max()) + 1e-6

    def test_empty_caps_returns_nan(self):
        assert np.isnan(book_default_cap_usd(pd.DataFrame(), 100_000.0, P))


# ---------------------------------------------------------------- params
def test_params_from_config_ignores_unknown_keys():
    p = CrashBudgetParams.from_config({"rho": 0.005, "enabled": True, "bogus": 1})
    assert p.rho == pytest.approx(0.005)
    assert p.theta == pytest.approx(0.5)
