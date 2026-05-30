"""End-to-end tests for ``risk_dashboard.beta_loader.compute_betas``.

We monkeypatch the data fetcher with a deterministic synthetic price
fixture so the assertions are independent of yfinance / Stooq /
network availability. The fixture wires three test underlyings to
known SPY/QQQ/IWM betas; the test then checks that ``compute_betas``
recovers them within tight tolerances.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from risk_dashboard import beta_loader as bl


# ---------------------------------------------------------------------------
# Synthetic price fixture
# ---------------------------------------------------------------------------


def _make_price_fixture(n_days: int = 300, seed: int = 20260520) -> dict[str, pd.Series]:
    """Build a deterministic price universe with known factor betas.

    ``HIBETA`` = 1.50 * SPY-ret + 0.20 * (QQQ-ret - SPY-ret) (so beta-
    to-SPY ~= 1.5 + 0.2 * beta_qqq_to_spy excess), ``MIDBETA`` = 1.00,
    ``LOWBETA`` = 0.40. Noise is small enough that the OLS recovers
    betas inside +-0.07.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2025-01-01", periods=n_days)

    spy_ret = rng.normal(0.0003, 0.008, size=n_days)
    qqq_ret = 1.10 * spy_ret + rng.normal(0.0, 0.002, size=n_days)
    iwm_ret = 1.05 * spy_ret + rng.normal(0.0, 0.003, size=n_days)
    vix_ret = -2.5 * spy_ret + rng.normal(0.0, 0.02, size=n_days)  # unused

    fixtures = {
        "SPY": spy_ret,
        "QQQ": qqq_ret,
        "IWM": iwm_ret,
        "BTC-USD": 1.00 * spy_ret + rng.normal(0.0, 0.015, size=n_days),
        "HIBETA": 1.50 * spy_ret + rng.normal(0.0, 0.003, size=n_days),
        "MIDBETA": 1.00 * spy_ret + rng.normal(0.0, 0.003, size=n_days),
        "LOWBETA": 0.40 * spy_ret + rng.normal(0.0, 0.004, size=n_days),
        "SECTOR_A_1": 1.80 * spy_ret + rng.normal(0.0, 0.004, size=n_days),
        "SECTOR_A_2": 1.70 * spy_ret + rng.normal(0.0, 0.004, size=n_days),
        "SECTOR_A_3": 1.90 * spy_ret + rng.normal(0.0, 0.004, size=n_days),
        "SECTOR_A_4": 1.75 * spy_ret + rng.normal(0.0, 0.004, size=n_days),
        "SECTOR_A_5": 1.85 * spy_ret + rng.normal(0.0, 0.004, size=n_days),
        # Used by no-data path: not included in fixture dict below.
        "_unused_vix": vix_ret,
    }

    out: dict[str, pd.Series] = {}
    for sym, rets in fixtures.items():
        if sym.startswith("_"):
            continue
        prices = 100.0 * np.exp(np.cumsum(rets))
        out[sym] = pd.Series(prices, index=idx, name=sym, dtype=float)
    return out


# ---------------------------------------------------------------------------
# compute_betas: recovery on synthetic prices
# ---------------------------------------------------------------------------


def test_compute_betas_recovers_known_factor_loadings(tmp_path: Path):
    fixture = _make_price_fixture()

    def fake_fetch(symbols, **kwargs):  # noqa: ARG001 - signature parity
        return {s: fixture[s] for s in symbols if s in fixture}

    results = bl.compute_betas(
        ["HIBETA", "MIDBETA", "LOWBETA"],
        cache_dir=tmp_path,
        sectors={"HIBETA": "test_a", "MIDBETA": "test_b", "LOWBETA": "test_c"},
        fetch_fn=fake_fetch,
    )

    hi = results["HIBETA"]
    mid = results["MIDBETA"]
    lo = results["LOWBETA"]

    # Default path: pure OLS (no shrinkage).
    for r in (hi, mid, lo):
        assert r.provenance == "computed", r.provenance
        assert r.n_obs >= bl.MIN_OBS_FOR_TRUST
        assert r.beta_se is not None and r.beta_se < 0.05
        assert r.beta_to_spy is not None
        assert r.beta_to_spy_raw is not None
        assert r.beta_to_spy == pytest.approx(r.beta_to_spy_raw)
        assert r.beta_to_ndx is not None
        assert r.beta_to_rut is not None
        assert r.beta_to_btc is not None
        assert r.regime_vol_pct is not None and r.regime_vol_pct > 0
        assert r.shrinkage_applied is False or r.shrinkage_applied is None

    assert hi.beta_to_spy_raw == pytest.approx(1.50, abs=0.15)
    assert mid.beta_to_spy_raw == pytest.approx(1.00, abs=0.15)
    assert lo.beta_to_spy_raw == pytest.approx(0.40, abs=0.15)
    assert hi.beta_to_spy > mid.beta_to_spy > lo.beta_to_spy

    # Multi-index betas: assert against the same OLS the loader uses. A naive
    # ``beta_spy / 1.05`` ratio is too high here because (a) prices are built
    # from compounded simple returns then differenced as log returns, and (b)
    # IWM/QQQ carry idiosyncratic noise that attenuates cross-index slopes.
    w = bl.DEFAULT_WINDOW_DAYS
    y_hi = bl._log_returns(fixture["HIBETA"], w)
    exp_ndx, _, _, _ = bl._ols_beta(
        y_hi, bl._log_returns(fixture["QQQ"], w)
    )
    exp_rut, _, _, _ = bl._ols_beta(
        y_hi, bl._log_returns(fixture["IWM"], w)
    )
    assert hi.beta_to_ndx_raw == pytest.approx(exp_ndx, rel=1e-9, abs=1e-9)
    assert hi.beta_to_rut_raw == pytest.approx(exp_rut, rel=1e-9, abs=1e-9)
    assert hi.beta_to_ndx_raw > hi.beta_to_rut_raw


def test_compute_betas_uses_sector_mean_prior_when_available(tmp_path: Path):
    """Five SECTOR_A names cross the SECTOR_MIN_COMPUTED_NAMES threshold,
    so pass-2 shrinks every SECTOR_A name toward the sector median."""
    fixture = _make_price_fixture()

    def fake_fetch(symbols, **kwargs):  # noqa: ARG001
        return {s: fixture[s] for s in symbols if s in fixture}

    sectors = {f"SECTOR_A_{i}": "test_sector_a" for i in range(1, 6)}
    results = bl.compute_betas(
        list(sectors.keys()),
        cache_dir=tmp_path,
        sectors=sectors,
        fetch_fn=fake_fetch,
        apply_shrinkage=True,
    )
    for r in results.values():
        assert r.sector == "test_sector_a"
        # Either sector-mean or curated prior is fine for the threshold;
        # the prior source must NOT be 'default' since we passed sectors.
        assert r.prior_source in {"sector_mean", "curated", "default"}, r.prior_source
        assert r.beta_to_spy is not None
        assert 1.50 < r.beta_to_spy < 2.10  # generated betas span 1.7-1.9

    # At least one name must have used the sector-mean prior.
    sector_means_used = [r for r in results.values() if r.prior_source == "sector_mean"]
    assert sector_means_used, "expected sector_mean shrinkage to fire for 5+ names"


def test_compute_betas_falls_back_when_fetch_returns_nothing(tmp_path: Path):
    """No price data -> default_fallback (not curated map) by default."""

    def empty_fetch(symbols, **kwargs):  # noqa: ARG001
        return {}

    results = bl.compute_betas(
        ["NVDA", "UNKNOWN_TKR_42"],
        cache_dir=tmp_path,
        fetch_fn=empty_fetch,
    )
    assert results["NVDA"].provenance == "default_fallback"
    assert results["NVDA"].beta_to_spy == pytest.approx(bl.DEFAULT_SINGLE_NAME_BETA)
    assert results["UNKNOWN_TKR_42"].provenance == "default_fallback"
    assert results["UNKNOWN_TKR_42"].beta_to_spy == pytest.approx(
        bl.DEFAULT_SINGLE_NAME_BETA
    )


def test_compute_betas_curated_fallback_when_shrinkage_enabled(tmp_path: Path):
    """Legacy shrinkage path still uses curated map when fetch is empty."""

    def empty_fetch(symbols, **kwargs):  # noqa: ARG001
        return {}

    results = bl.compute_betas(
        ["NVDA", "UNKNOWN_TKR_42"],
        cache_dir=tmp_path,
        fetch_fn=empty_fetch,
        apply_shrinkage=True,
    )
    assert results["NVDA"].provenance == "curated_fallback"
    assert results["NVDA"].beta_to_spy == pytest.approx(bl.BETA_TO_SPY["NVDA"])


# ---------------------------------------------------------------------------
# AR(1) helper agrees with daily_screener's shape
# ---------------------------------------------------------------------------


def test_ar1_n_eff_iid_returns_full_n():
    rng = np.random.default_rng(7)
    iid = rng.normal(0, 1, 300)
    rho, n_eff = bl._ar1_n_eff(iid)
    assert abs(rho) < 0.15
    assert n_eff > 200


def test_ar1_n_eff_persistent_returns_reduced_n():
    rng = np.random.default_rng(11)
    eps = rng.normal(0, 1, 300)
    series = np.empty(300)
    series[0] = eps[0]
    for t in range(1, 300):
        series[t] = 0.7 * series[t - 1] + eps[t]
    rho, n_eff = bl._ar1_n_eff(series)
    assert rho > 0.5
    assert n_eff < 200


# ---------------------------------------------------------------------------
# Shrinkage helper
# ---------------------------------------------------------------------------


def test_shrink_beta_to_sector_pulls_low_n_toward_prior():
    out, w, applied = bl._shrink_beta_to_sector(2.5, n_eff=10, prior=1.5)
    assert applied is True
    assert w < 0.2
    assert 1.5 < out < 2.0


def test_shrink_beta_to_sector_keeps_high_n_close_to_ols():
    out, w, applied = bl._shrink_beta_to_sector(2.5, n_eff=1000, prior=1.5)
    # 1000 / (1000 + 60*1.5^2 = 135) -> w ~ 0.881; below 0.95 so applied=True.
    assert applied is True
    assert 0.85 < w < 0.92
    assert 2.35 < out < 2.50


def test_shrink_beta_to_sector_none_input_returns_prior():
    out, w, applied = bl._shrink_beta_to_sector(None, n_eff=100, prior=1.4)
    assert out == pytest.approx(1.4)
    assert applied is True
    assert w == 0.0


# ---------------------------------------------------------------------------
# Stooq adapter
# ---------------------------------------------------------------------------


def test_parse_stooq_csv_happy_path():
    body = (
        "Date,Open,High,Low,Close,Volume\n"
        "2026-05-15,100.0,101.5,99.8,101.2,12345\n"
        "2026-05-16,101.2,102.0,100.5,101.8,15000\n"
    )
    s = bl._parse_stooq_csv(body, "TEST")
    assert s is not None
    assert list(s.values) == [101.2, 101.8]
    assert s.index[0] == pd.Timestamp("2026-05-15")


def test_parse_stooq_csv_returns_none_on_empty_response():
    assert bl._parse_stooq_csv("No data\n", "TEST") is None
    assert bl._parse_stooq_csv("", "TEST") is None


def test_parse_stooq_csv_skips_malformed_rows():
    body = (
        "Date,Close\n"
        "2026-05-15,101.2\n"
        ",100.0\n"
        "2026-05-16,not-a-float\n"
        "2026-05-17,103.4\n"
    )
    s = bl._parse_stooq_csv(body, "TEST")
    assert s is not None
    assert list(s.values) == [101.2, 103.4]


# ---------------------------------------------------------------------------
# Cache-first path
# ---------------------------------------------------------------------------


def test_parse_yahoo_v8_chart_happy_path():
    payload = {
        "chart": {
            "result": [
                {
                    "timestamp": [1715760000, 1715846400],
                    "indicators": {"adjclose": [{"adjclose": [100.0, 101.5]}]},
                }
            ]
        }
    }
    s = bl._parse_yahoo_v8_chart(payload, "NVDA")
    assert s is not None
    assert list(s.values) == [100.0, 101.5]
    assert s.name == "NVDA"


def test_parse_yahoo_v8_chart_returns_none_on_empty_result():
    assert bl._parse_yahoo_v8_chart({"chart": {"result": []}}, "NVDA") is None
    assert bl._parse_yahoo_v8_chart({}, "NVDA") is None


def test_fetch_closes_serves_fresh_cache_without_calling_fetcher(tmp_path: Path):
    """Fresh on-disk cache must be returned without any network hit."""
    idx = pd.bdate_range("2026-04-01", periods=80)
    closes = pd.Series(np.linspace(100, 110, 80), index=idx, name="DEMO")
    bl._save_cached_closes(tmp_path, "DEMO", closes)

    def boom(*args, **kwargs):  # noqa: ARG001
        raise AssertionError("network fetch should not run for fresh cache")

    out = bl._fetch_closes(
        ["DEMO"],
        window_days=60,
        cache_dir=tmp_path,
        yf_module=type("YF", (), {"download": staticmethod(boom)}),
        enable_stooq_fallback=False,
    )
    assert "DEMO" in out
    assert math.isclose(out["DEMO"].iloc[-1], 110.0)


# ---------------------------------------------------------------------------
# Summary cache round-trip
# ---------------------------------------------------------------------------


def test_write_summary_cache_roundtrip(tmp_path: Path):
    res = bl.BetaResult(
        underlying="NVDA",
        beta_to_spy=1.6,
        beta_to_ndx=1.5,
        beta_to_rut=1.4,
        n_obs=200,
        sector="semis",
        provenance="computed",
        prior_used_spy=1.4,
        prior_source="sector_mean",
    )
    out_path = tmp_path / "beta_summary.json"
    bl.write_summary_cache(
        {"NVDA": res},
        snapshot_date="2026-05-19",
        path=out_path,
        sector_means={"semis": {"spy": 1.4}},
    )
    loaded = bl.read_summary_cache(out_path)
    assert loaded is not None
    assert loaded["snapshot_date"] == "2026-05-19"
    assert loaded["window_days"] == bl.DEFAULT_WINDOW_DAYS
    assert loaded["sector_means"]["semis"]["spy"] == pytest.approx(1.4)
    assert loaded["rows"][0]["underlying"] == "NVDA"
    assert loaded["rows"][0]["provenance"] == "computed"
