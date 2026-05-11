import numpy as np
import pandas as pd

from generate_trade_plan import apply_covariance_balance


_CAPS = {
    "gross_sizing_caps": {
        "enabled": True,
        "max_pair_weight_cap": 0.99,
        "max_underlying_weight_cap": 0.99,
        "aum_use_pct": 0.0,
        "short_avail_use_pct": 0.0,
        "missing_shares_cap": 0.99,
        "shares_outstanding_use_frac": 0.0,
        "median_daily_volume_use_pct": 0.0,
    },
}


def _two_pair_df(g1: float = 500.0, g2: float = 500.0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ETF": ["AAA", "BBB"],
            "Underlying": ["U1", "U2"],
            "Beta": [2.0, 2.0],
            "beta_abs": [2.0, 2.0],
            "sleeve": ["core_leveraged", "core_leveraged"],
            "gross_target_usd": [g1, g2],
            "borrow_price_ref": [50.0, 50.0],
            "shares_available": [1e9, 1e9],
        }
    )


def _make_returns(n: int = 252, *, rho: float = 0.0, vol_ratio: float = 1.0, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    z1 = rng.standard_normal(n)
    z2 = rho * z1 + np.sqrt(max(1.0 - rho * rho, 0.0)) * rng.standard_normal(n)
    r1 = 0.01 * z1
    r2 = 0.01 * vol_ratio * z2
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    return pd.DataFrame({"U1": r1, "U2": r2}, index=idx)


def test_disabled_is_noop():
    df = _two_pair_df()
    out, diag = apply_covariance_balance(
        df,
        target_gross_usd=1_000_000.0,
        beta_floor=0.1,
        strategy={**_CAPS},
        returns_df=_make_returns(),
    )
    assert not diag.get("applied")
    np.testing.assert_allclose(out["gross_target_usd"].to_numpy(float), [500.0, 500.0])


def test_thin_data_skips():
    df = _two_pair_df()
    strategy = {**_CAPS, "covariance_balance": {"enabled": True, "min_obs": 100}}
    out, diag = apply_covariance_balance(
        df,
        target_gross_usd=1_000_000.0,
        beta_floor=0.1,
        strategy=strategy,
        returns_df=_make_returns(n=20),
    )
    assert not diag.get("applied")
    assert diag.get("reason") == "insufficient_returns"
    np.testing.assert_allclose(out["gross_target_usd"].to_numpy(float), [500.0, 500.0])


def test_perfect_corr_equal_exposure_yields_no_tilt():
    # Identical underlyings + identical exposure: penalty multipliers should be equal,
    # leaving final weights unchanged after renormalization to preserve gross.
    df = _two_pair_df()
    strategy = {**_CAPS, "covariance_balance": {"enabled": True, "shrink": 0.0, "penalty_strength": 0.85, "min_obs": 30}}
    R = _make_returns(rho=0.999)
    out, diag = apply_covariance_balance(
        df,
        target_gross_usd=1_000_000.0,
        beta_floor=0.1,
        strategy=strategy,
        returns_df=R,
    )
    assert diag.get("applied")
    g = out["gross_target_usd"].to_numpy(float)
    np.testing.assert_allclose(g.sum(), 1000.0, rtol=1e-5)
    # rho=0.999 returns near-identical multipliers; allow tiny finite-sample noise.
    np.testing.assert_allclose(g[0], g[1], rtol=2e-3)


def test_higher_vol_underlying_gets_attenuated():
    df = _two_pair_df()
    strategy = {**_CAPS, "covariance_balance": {"enabled": True, "shrink": 0.0, "penalty_strength": 1.5, "min_obs": 30}}
    R = _make_returns(rho=0.0, vol_ratio=4.0)
    out, diag = apply_covariance_balance(
        df,
        target_gross_usd=1_000_000.0,
        beta_floor=0.1,
        strategy=strategy,
        returns_df=R,
    )
    assert diag.get("applied")
    g = out["gross_target_usd"].to_numpy(float)
    np.testing.assert_allclose(g.sum(), 1000.0, rtol=1e-5)
    assert g[1] < g[0], f"high-vol underlying should be attenuated, got {g}"


def test_post_cov_recap_enforces_pair_cap():
    df = _two_pair_df(g1=900.0, g2=100.0)
    strategy = {
        "gross_sizing_caps": {**_CAPS["gross_sizing_caps"], "max_pair_weight_cap": 0.6, "missing_shares_cap": 0.99},
        "covariance_balance": {"enabled": True, "shrink": 0.0, "penalty_strength": 0.85, "min_obs": 30},
    }
    R = _make_returns(rho=0.0, vol_ratio=1.0)
    out, diag = apply_covariance_balance(
        df,
        target_gross_usd=1_000.0,
        beta_floor=0.1,
        strategy=strategy,
        returns_df=R,
    )
    assert diag.get("applied")
    g = out["gross_target_usd"].to_numpy(float)
    w = g / max(g.sum(), 1e-12)
    assert float(w.max()) <= 0.6 + 1e-6
