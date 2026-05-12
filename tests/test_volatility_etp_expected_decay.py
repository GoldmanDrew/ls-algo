import pandas as pd

from daily_screener import (
    apply_volatility_etp_expected_decay_adjustment,
    build_full_universe,
)


def test_volatility_etp_expected_decay_uses_empirical_adjustment():
    df = pd.DataFrame(
        [
            {
                "ETF": "UVIX",
                "Underlying": "SVIX",
                "product_class": "letf",
                "expected_gross_decay_annual": 1.059371,
                "gross_decay_annual": 2.162183,
                "realized_tracking_component_annual": 1.102812,
                "expected_gross_decay_reliable": True,
            }
        ]
    )

    out = apply_volatility_etp_expected_decay_adjustment(df)
    row = out.iloc[0]

    assert row["product_class"] == "volatility_etp"
    assert row["expected_decay_model"] == "volatility_etp_empirical_roll_adjusted"
    assert row["expected_gross_decay_simple_ito_annual"] == 1.059371
    assert row["expected_decay_adjustment_annual"] == 1.102812
    assert row["expected_gross_decay_adjusted_annual"] == 2.162183
    assert row["expected_gross_decay_annual"] == 2.162183
    assert not bool(row["expected_gross_decay_reliable"])


def test_yieldboost_pairs_are_income_sleeve_not_leveraged_sleeve():
    universe = build_full_universe(skip_scrape=True, skip_inverse=True)
    rows = universe[universe["ETF"].isin(["COYY", "TSYY", "MUYY", "CWY"])]

    assert set(rows["ETF"]) == {"COYY", "TSYY", "MUYY", "CWY"}
    assert set(rows["Leverage"]) == {1.0}
    assert rows["is_yieldboost"].all()
    assert set(rows["scenario_style"]) == {"income_style"}
